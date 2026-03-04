// Parallel Prefix Sum (Scan)
//
// Computes prefix sums (running totals) of an array in parallel.
// Foundation for many GPU algorithms: sorting, stream compaction, histograms.
//
// Two types of scan:
// - Exclusive: output[i] = sum of input[0..i] (excludes input[i])
// - Inclusive: output[i] = sum of input[0..=i] (includes input[i])
//
// Example (inclusive): [1, 2, 3, 4] -> [1, 3, 6, 10]
// Example (exclusive): [1, 2, 3, 4] -> [0, 1, 3, 6]
//
// Parallel algorithm (Hillis-Steele):
// - Work: O(n log n) - not work-efficient but highly parallel
// - Depth: O(log n) - good for GPUs with many cores
// - Each step i: for all j >= 2^i: out[j] = in[j] + in[j - 2^i]
//
// Work-efficient algorithm (Blelloch) covered in separate module.
//
// Use cases:
// - Stream compaction (filtering arrays on GPU)
// - Radix sort building block
// - Histogram computation
// - Sparse matrix operations

use std::ops::Add;

// Sequential scan for reference and small inputs
pub fn sequential_inclusive_scan<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(input.len());
    let mut acc = T::default();

    for &x in input {
        acc = acc + x;
        result.push(acc);
    }

    result
}

pub fn sequential_exclusive_scan<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(input.len());
    let mut acc = T::default();

    for &x in input {
        result.push(acc);
        acc = acc + x;
    }

    result
}

// Hillis-Steele parallel scan (inclusive)
// Simulates parallel execution - each "step" can be done in parallel on GPU
// Returns (result, steps) where steps shows intermediate states for visualization
pub fn hillis_steele_inclusive<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    let n = input.len();
    let mut current = input.to_vec();
    let mut next = vec![T::default(); n];

    // log2(n) steps
    let mut stride = 1;
    while stride < n {
        // This loop can be parallelized on GPU
        for i in 0..n {
            if i >= stride {
                next[i] = current[i] + current[i - stride];
            } else {
                next[i] = current[i];
            }
        }
        std::mem::swap(&mut current, &mut next);
        stride *= 2;
    }

    current
}

// Hillis-Steele exclusive scan
pub fn hillis_steele_exclusive<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    // Compute inclusive scan then shift right
    let inclusive = hillis_steele_inclusive(input);
    let mut result = vec![T::default(); input.len()];

    result[1..input.len()].copy_from_slice(&inclusive[..(input.len() - 1)]);

    result
}

// Parallel scan with custom operation (generalized)
pub fn parallel_scan_with_op<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if input.is_empty() {
        return vec![];
    }

    let n = input.len();
    let mut current = input.to_vec();
    let mut next = vec![identity; n];

    let mut stride = 1;
    while stride < n {
        for i in 0..n {
            if i >= stride {
                next[i] = op(current[i], current[i - stride]);
            } else {
                next[i] = current[i];
            }
        }
        std::mem::swap(&mut current, &mut next);
        stride *= 2;
    }

    current
}

// Segmented scan - scan within segments defined by flags
// Useful for variable-length sequences in a single array
// flags[i] = true means start of new segment
pub fn segmented_inclusive_scan<T>(input: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    assert_eq!(input.len(), flags.len());

    let mut result = Vec::with_capacity(input.len());
    let mut acc = T::default();

    for i in 0..input.len() {
        if flags[i] {
            acc = input[i]; // Start new segment
        } else {
            acc = acc + input[i];
        }
        result.push(acc);
    }

    result
}

// Parallel block-based scan for large arrays
// Divides input into blocks, scans each block, then combines
pub struct BlockScanResult<T> {
    pub result: Vec<T>,
    pub block_sums: Vec<T>, // Sum of each block (useful for hierarchical scan)
}

pub fn block_scan<T>(input: &[T], block_size: usize) -> BlockScanResult<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return BlockScanResult {
            result: vec![],
            block_sums: vec![],
        };
    }

    let n = input.len();
    let num_blocks = n.div_ceil(block_size);

    // Phase 1: Scan within each block (can be parallel)
    let mut result = vec![T::default(); n];
    let mut block_sums = Vec::with_capacity(num_blocks);

    for block in 0..num_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(n);

        let mut acc = T::default();
        for i in start..end {
            acc = acc + input[i];
            result[i] = acc;
        }
        block_sums.push(acc);
    }

    // Phase 2: Scan block sums
    let scanned_block_sums = sequential_exclusive_scan(&block_sums);

    // Phase 3: Add block prefix to each element (can be parallel)
    for (block, &prefix) in scanned_block_sums.iter().enumerate().skip(1) {
        let start = block * block_size;
        let end = (start + block_size).min(n);

        for item in result.iter_mut().take(end).skip(start) {
            *item = *item + prefix;
        }
    }

    BlockScanResult { result, block_sums }
}

// Stream compaction using scan
// Returns indices of elements where predicate is true
pub fn stream_compact<T, P>(input: &[T], predicate: P) -> Vec<usize>
where
    P: Fn(&T) -> bool,
{
    // Create flags array
    let flags: Vec<i32> = input
        .iter()
        .map(|x| if predicate(x) { 1 } else { 0 })
        .collect();

    // Exclusive scan gives output positions (used in GPU version for scatter)
    let _positions = sequential_exclusive_scan(&flags);

    // Gather indices
    let mut result = Vec::new();
    for (i, &flag) in flags.iter().enumerate() {
        if flag == 1 {
            result.push(i);
        }
    }

    result
}

// Compact and gather values
pub fn stream_compact_values<T, P>(input: &[T], predicate: P) -> Vec<T>
where
    T: Copy,
    P: Fn(&T) -> bool,
{
    let indices = stream_compact(input, predicate);
    indices.iter().map(|&i| input[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_inclusive_scan() {
        let input = vec![1, 2, 3, 4, 5];
        let result = sequential_inclusive_scan(&input);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_sequential_exclusive_scan() {
        let input = vec![1, 2, 3, 4, 5];
        let result = sequential_exclusive_scan(&input);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_hillis_steele_inclusive() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = hillis_steele_inclusive(&input);
        assert_eq!(result, vec![1, 3, 6, 10, 15, 21, 28, 36]);
    }

    #[test]
    fn test_hillis_steele_exclusive() {
        let input = vec![1, 2, 3, 4, 5];
        let result = hillis_steele_exclusive(&input);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let input: Vec<i32> = (1..=16).collect();

        let seq = sequential_inclusive_scan(&input);
        let par = hillis_steele_inclusive(&input);

        assert_eq!(seq, par);
    }

    #[test]
    fn test_empty_input() {
        let input: Vec<i32> = vec![];

        assert!(sequential_inclusive_scan(&input).is_empty());
        assert!(sequential_exclusive_scan(&input).is_empty());
        assert!(hillis_steele_inclusive(&input).is_empty());
        assert!(hillis_steele_exclusive(&input).is_empty());
    }

    #[test]
    fn test_single_element() {
        let input = vec![42];

        assert_eq!(sequential_inclusive_scan(&input), vec![42]);
        assert_eq!(sequential_exclusive_scan(&input), vec![0]);
        assert_eq!(hillis_steele_inclusive(&input), vec![42]);
        assert_eq!(hillis_steele_exclusive(&input), vec![0]);
    }

    #[test]
    fn test_parallel_scan_with_op() {
        // Test with multiplication (product scan)
        let input = vec![1, 2, 3, 4];
        let result = parallel_scan_with_op(&input, 1, |a, b| a * b);
        assert_eq!(result, vec![1, 2, 6, 24]); // factorials
    }

    #[test]
    fn test_parallel_scan_max() {
        // Running maximum
        let input = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let result = parallel_scan_with_op(&input, i32::MIN, |a, b| a.max(b));
        assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9]);
    }

    #[test]
    fn test_segmented_scan() {
        let input = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = segmented_inclusive_scan(&input, &flags);
        assert_eq!(result, vec![1, 3, 6, 1, 3, 1, 3, 6, 10]);
    }

    #[test]
    fn test_block_scan() {
        let input: Vec<i32> = (1..=12).collect();
        let result = block_scan(&input, 4);

        // Should produce same result as regular scan
        let expected = sequential_inclusive_scan(&input);
        assert_eq!(result.result, expected);

        // Block sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        assert_eq!(result.block_sums, vec![10, 26, 42]);
    }

    #[test]
    fn test_block_scan_uneven() {
        let input = vec![1, 2, 3, 4, 5, 6, 7]; // 7 elements, block size 3
        let result = block_scan(&input, 3);

        let expected = sequential_inclusive_scan(&input);
        assert_eq!(result.result, expected);
    }

    #[test]
    fn test_stream_compact() {
        let input = vec![1, 5, 2, 8, 3, 9, 4];
        let indices = stream_compact(&input, |&x| x > 4);
        assert_eq!(indices, vec![1, 3, 5]); // indices of 5, 8, 9
    }

    #[test]
    fn test_stream_compact_values() {
        let input = vec![1, 5, 2, 8, 3, 9, 4];
        let values = stream_compact_values(&input, |&x| x > 4);
        assert_eq!(values, vec![5, 8, 9]);
    }

    #[test]
    fn test_stream_compact_none() {
        let input = vec![1, 2, 3];
        let indices = stream_compact(&input, |&x| x > 10);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_stream_compact_all() {
        let input = vec![1, 2, 3];
        let indices = stream_compact(&input, |&x| x > 0);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_floating_point_scan() {
        let input: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let result = sequential_inclusive_scan(&input);

        // Check approximate equality for floats
        assert!((result[0] - 0.1).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.6).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_input() {
        let input: Vec<i64> = (1..=1000).collect();

        let seq = sequential_inclusive_scan(&input);
        let par = hillis_steele_inclusive(&input);

        assert_eq!(seq, par);

        // Sum of 1..=1000 = 1000 * 1001 / 2 = 500500
        assert_eq!(*seq.last().unwrap(), 500500);
    }

    #[test]
    fn test_power_of_two_size() {
        // Hillis-Steele works best with power-of-2 sizes
        for &size in &[2, 4, 8, 16, 32, 64] {
            let input: Vec<i32> = (1..=size).collect();
            let seq = sequential_inclusive_scan(&input);
            let par = hillis_steele_inclusive(&input);
            assert_eq!(seq, par, "Failed for size {}", size);
        }
    }
}
