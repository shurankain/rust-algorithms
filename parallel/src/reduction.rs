// Parallel Reduction
//
// Reduces an array to a single value using an associative operation.
// Foundation for computing sums, products, min/max on GPUs.
//
// Tree-based reduction:
// - Work: O(n) - work-efficient
// - Depth: O(log n) - highly parallel
// - Each step halves the number of active elements
//
// Example (sum): [1, 2, 3, 4, 5, 6, 7, 8]
//   Step 1: [3, 7, 11, 15] (pairs: 1+2, 3+4, 5+6, 7+8)
//   Step 2: [10, 26]       (pairs: 3+7, 11+15)
//   Step 3: [36]           (final: 10+26)
//
// GPU considerations:
// - Minimize bank conflicts with proper stride patterns
// - Use shared memory for intermediate results
// - Handle non-power-of-2 sizes with padding or sequential tail
//
// Use cases:
// - Computing array sum/product
// - Finding min/max element
// - Counting elements matching predicate
// - Dot product computation

use std::cmp::Ordering;
use std::ops::{Add, Mul};

// Tree-based parallel reduction with custom operation
// Simulates GPU parallel execution pattern
pub fn parallel_reduce<T, F>(input: &[T], identity: T, op: F) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if input.is_empty() {
        return identity;
    }

    if input.len() == 1 {
        return input[0];
    }

    let mut current = input.to_vec();

    // Tree reduction: halve the size each step
    while current.len() > 1 {
        let half = current.len().div_ceil(2);
        let mut next = Vec::with_capacity(half);

        // Each iteration can be parallelized on GPU
        for i in 0..half {
            let left = current[i * 2];
            let right = if i * 2 + 1 < current.len() {
                current[i * 2 + 1]
            } else {
                identity // Handle odd-length arrays
            };
            next.push(op(left, right));
        }

        current = next;
    }

    current[0]
}

// Parallel sum reduction
pub fn parallel_sum<T>(input: &[T]) -> T
where
    T: Copy + Add<Output = T> + Default,
{
    parallel_reduce(input, T::default(), |a, b| a + b)
}

// Parallel product reduction
pub fn parallel_product<T>(input: &[T]) -> T
where
    T: Copy + Mul<Output = T> + From<u8>,
{
    parallel_reduce(input, T::from(1u8), |a, b| a * b)
}

// Parallel minimum reduction
pub fn parallel_min<T>(input: &[T]) -> Option<T>
where
    T: Copy + Ord,
{
    if input.is_empty() {
        return None;
    }

    Some(parallel_reduce(
        input,
        input[0],
        |a, b| if a < b { a } else { b },
    ))
}

// Parallel maximum reduction
pub fn parallel_max<T>(input: &[T]) -> Option<T>
where
    T: Copy + Ord,
{
    if input.is_empty() {
        return None;
    }

    Some(parallel_reduce(
        input,
        input[0],
        |a, b| if a > b { a } else { b },
    ))
}

// Parallel min with index (argmin)
pub fn parallel_argmin<T>(input: &[T]) -> Option<(usize, T)>
where
    T: Copy + Ord,
{
    if input.is_empty() {
        return None;
    }

    let indexed: Vec<(usize, T)> = input.iter().copied().enumerate().collect();

    let result = parallel_reduce(
        &indexed,
        (0, input[0]),
        |a, b| if a.1 <= b.1 { a } else { b },
    );

    Some(result)
}

// Parallel max with index (argmax)
pub fn parallel_argmax<T>(input: &[T]) -> Option<(usize, T)>
where
    T: Copy + Ord,
{
    if input.is_empty() {
        return None;
    }

    let indexed: Vec<(usize, T)> = input.iter().copied().enumerate().collect();

    let result = parallel_reduce(
        &indexed,
        (0, input[0]),
        |a, b| if a.1 >= b.1 { a } else { b },
    );

    Some(result)
}

// Parallel count matching predicate
pub fn parallel_count<T, P>(input: &[T], predicate: P) -> usize
where
    T: Copy,
    P: Fn(&T) -> bool,
{
    let flags: Vec<usize> = input
        .iter()
        .map(|x| if predicate(x) { 1 } else { 0 })
        .collect();
    parallel_reduce(&flags, 0, |a, b| a + b)
}

// Parallel all (logical AND reduction)
pub fn parallel_all<T, P>(input: &[T], predicate: P) -> bool
where
    T: Copy,
    P: Fn(&T) -> bool,
{
    if input.is_empty() {
        return true; // vacuous truth
    }

    let flags: Vec<bool> = input.iter().map(&predicate).collect();
    parallel_reduce(&flags, true, |a, b| a && b)
}

// Parallel any (logical OR reduction)
pub fn parallel_any<T, P>(input: &[T], predicate: P) -> bool
where
    T: Copy,
    P: Fn(&T) -> bool,
{
    if input.is_empty() {
        return false;
    }

    let flags: Vec<bool> = input.iter().map(&predicate).collect();
    parallel_reduce(&flags, false, |a, b| a || b)
}

// Parallel dot product
pub fn parallel_dot_product<T>(a: &[T], b: &[T]) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    if a.is_empty() {
        return T::default();
    }

    let products: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    parallel_sum(&products)
}

// Block-based reduction for very large arrays
// Reduces within blocks first, then reduces block results
pub struct BlockReductionResult<T> {
    pub result: T,
    pub block_results: Vec<T>, // Intermediate results (useful for debugging/visualization)
}

pub fn block_reduce<T, F>(
    input: &[T],
    block_size: usize,
    identity: T,
    op: F,
) -> BlockReductionResult<T>
where
    T: Copy,
    F: Fn(T, T) -> T + Copy,
{
    if input.is_empty() {
        return BlockReductionResult {
            result: identity,
            block_results: vec![],
        };
    }

    let num_blocks = input.len().div_ceil(block_size);
    let mut block_results = Vec::with_capacity(num_blocks);

    // Phase 1: Reduce within each block (can be parallel)
    for block in 0..num_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(input.len());
        let block_slice = &input[start..end];

        let block_result = parallel_reduce(block_slice, identity, op);
        block_results.push(block_result);
    }

    // Phase 2: Reduce block results
    let result = parallel_reduce(&block_results, identity, op);

    BlockReductionResult {
        result,
        block_results,
    }
}

// Parallel reduction for floating point with better numerical stability
// Uses Kahan summation within the reduction
pub fn parallel_sum_stable(input: &[f64]) -> f64 {
    if input.is_empty() {
        return 0.0;
    }

    if input.len() == 1 {
        return input[0];
    }

    // For small inputs, use Kahan summation directly
    if input.len() <= 16 {
        return kahan_sum(input);
    }

    // For larger inputs, divide and conquer with Kahan at leaves
    let mid = input.len() / 2;
    let left = parallel_sum_stable(&input[..mid]);
    let right = parallel_sum_stable(&input[mid..]);
    left + right
}

fn kahan_sum(input: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for &x in input {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

// Min/max for floating point (handles NaN)
pub fn parallel_min_f64(input: &[f64]) -> Option<f64> {
    if input.is_empty() {
        return None;
    }

    Some(parallel_reduce(input, f64::INFINITY, |a, b| {
        match a.partial_cmp(&b) {
            Some(Ordering::Less) | Some(Ordering::Equal) => a,
            Some(Ordering::Greater) => b,
            None => {
                // Handle NaN: propagate NaN if either is NaN
                if a.is_nan() { a } else { b }
            }
        }
    }))
}

pub fn parallel_max_f64(input: &[f64]) -> Option<f64> {
    if input.is_empty() {
        return None;
    }

    Some(parallel_reduce(input, f64::NEG_INFINITY, |a, b| {
        match a.partial_cmp(&b) {
            Some(Ordering::Greater) | Some(Ordering::Equal) => a,
            Some(Ordering::Less) => b,
            None => {
                if a.is_nan() {
                    a
                } else {
                    b
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sum() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(parallel_sum(&input), 36);
    }

    #[test]
    fn test_parallel_sum_odd_length() {
        let input = vec![1, 2, 3, 4, 5];
        assert_eq!(parallel_sum(&input), 15);
    }

    #[test]
    fn test_parallel_sum_empty() {
        let input: Vec<i32> = vec![];
        assert_eq!(parallel_sum(&input), 0);
    }

    #[test]
    fn test_parallel_sum_single() {
        let input = vec![42];
        assert_eq!(parallel_sum(&input), 42);
    }

    #[test]
    fn test_parallel_product() {
        let input = vec![1, 2, 3, 4];
        assert_eq!(parallel_product(&input), 24);
    }

    #[test]
    fn test_parallel_product_with_zero() {
        let input = vec![1, 2, 0, 4];
        assert_eq!(parallel_product(&input), 0);
    }

    #[test]
    fn test_parallel_min() {
        let input = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(parallel_min(&input), Some(1));
    }

    #[test]
    fn test_parallel_min_empty() {
        let input: Vec<i32> = vec![];
        assert_eq!(parallel_min(&input), None);
    }

    #[test]
    fn test_parallel_max() {
        let input = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(parallel_max(&input), Some(9));
    }

    #[test]
    fn test_parallel_max_empty() {
        let input: Vec<i32> = vec![];
        assert_eq!(parallel_max(&input), None);
    }

    #[test]
    fn test_parallel_argmin() {
        let input = vec![5, 2, 8, 1, 9, 3];
        let result = parallel_argmin(&input);
        assert_eq!(result, Some((3, 1))); // min=1 at index 3
    }

    #[test]
    fn test_parallel_argmax() {
        let input = vec![5, 2, 8, 1, 9, 3];
        let result = parallel_argmax(&input);
        assert_eq!(result, Some((4, 9))); // max=9 at index 4
    }

    #[test]
    fn test_parallel_count() {
        let input = vec![1, 5, 2, 8, 3, 9, 4];
        let count = parallel_count(&input, |&x| x > 4);
        assert_eq!(count, 3); // 5, 8, 9
    }

    #[test]
    fn test_parallel_count_none() {
        let input = vec![1, 2, 3];
        let count = parallel_count(&input, |&x| x > 10);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_parallel_count_all() {
        let input = vec![1, 2, 3];
        let count = parallel_count(&input, |&x| x > 0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_parallel_all() {
        let input = vec![2, 4, 6, 8];
        assert!(parallel_all(&input, |&x| x % 2 == 0));
        assert!(!parallel_all(&input, |&x| x > 5));
    }

    #[test]
    fn test_parallel_all_empty() {
        let input: Vec<i32> = vec![];
        assert!(parallel_all(&input, |&x| x > 0)); // vacuous truth
    }

    #[test]
    fn test_parallel_any() {
        let input = vec![1, 3, 5, 6, 7];
        assert!(parallel_any(&input, |&x| x % 2 == 0)); // 6 is even
        assert!(!parallel_any(&input, |&x| x > 10));
    }

    #[test]
    fn test_parallel_any_empty() {
        let input: Vec<i32> = vec![];
        assert!(!parallel_any(&input, |&x| x > 0));
    }

    #[test]
    fn test_parallel_dot_product() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(parallel_dot_product(&a, &b), 32);
    }

    #[test]
    fn test_parallel_dot_product_empty() {
        let a: Vec<i32> = vec![];
        let b: Vec<i32> = vec![];
        assert_eq!(parallel_dot_product(&a, &b), 0);
    }

    #[test]
    fn test_block_reduce_sum() {
        let input: Vec<i32> = (1..=12).collect();
        let result = block_reduce(&input, 4, 0, |a, b| a + b);

        assert_eq!(result.result, 78); // sum of 1..=12
        assert_eq!(result.block_results, vec![10, 26, 42]); // sums of each block of 4
    }

    #[test]
    fn test_block_reduce_uneven() {
        let input = vec![1, 2, 3, 4, 5, 6, 7]; // 7 elements, block size 3
        let result = block_reduce(&input, 3, 0, |a, b| a + b);

        assert_eq!(result.result, 28);
        assert_eq!(result.block_results.len(), 3); // ceil(7/3) = 3 blocks
    }

    #[test]
    fn test_parallel_sum_stable() {
        // Test with values that can cause floating point errors
        let input: Vec<f64> = vec![1e10, 1.0, -1e10, 2.0, 3.0];
        let result = parallel_sum_stable(&input);
        assert!((result - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_sum_stable_many_small() {
        let input: Vec<f64> = (0..1000).map(|_| 0.001).collect();
        let result = parallel_sum_stable(&input);
        assert!((result - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_min_f64() {
        let input = vec![3.0, 1.5, 2.0, 0.5];
        assert_eq!(parallel_min_f64(&input), Some(0.5));
    }

    #[test]
    fn test_parallel_max_f64() {
        let input = vec![3.0, 1.5, 2.0, 0.5];
        assert_eq!(parallel_max_f64(&input), Some(3.0));
    }

    #[test]
    fn test_parallel_min_f64_with_nan() {
        let input = vec![3.0, f64::NAN, 2.0];
        let result = parallel_min_f64(&input).unwrap();
        assert!(result.is_nan());
    }

    #[test]
    fn test_parallel_max_f64_with_nan() {
        let input = vec![3.0, f64::NAN, 2.0];
        let result = parallel_max_f64(&input).unwrap();
        assert!(result.is_nan());
    }

    #[test]
    fn test_power_of_two_sum() {
        // Test various power-of-2 sizes
        for &size in &[2, 4, 8, 16, 32, 64] {
            let input: Vec<i64> = (1..=size).collect();
            let expected = size * (size + 1) / 2;
            assert_eq!(parallel_sum(&input), expected, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_large_reduction() {
        let input: Vec<i64> = (1..=1000).collect();
        let result = parallel_sum(&input);
        assert_eq!(result, 500500); // n*(n+1)/2
    }

    #[test]
    fn test_custom_reduce() {
        // XOR reduction
        let input = vec![0b1010, 0b1100, 0b0011];
        let result = parallel_reduce(&input, 0, |a, b| a ^ b);
        assert_eq!(result, 0b1010 ^ 0b1100 ^ 0b0011);
    }

    #[test]
    fn test_argmin_with_duplicates() {
        let input = vec![3, 1, 4, 1, 5];
        let (idx, val) = parallel_argmin(&input).unwrap();
        assert_eq!(val, 1);
        assert!(idx == 1 || idx == 3); // Either index with value 1 is valid
    }

    #[test]
    fn test_argmax_with_duplicates() {
        let input = vec![3, 5, 4, 5, 2];
        let (idx, val) = parallel_argmax(&input).unwrap();
        assert_eq!(val, 5);
        assert!(idx == 1 || idx == 3);
    }
}
