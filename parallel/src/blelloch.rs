// Work-Efficient Scan (Blelloch Algorithm)
//
// Unlike Hillis-Steele which does O(n log n) work, Blelloch achieves O(n) work
// while maintaining O(log n) depth - optimal for GPU with limited parallelism.
//
// Two-phase algorithm:
// 1. Up-sweep (reduce): Build partial sums tree bottom-up
// 2. Down-sweep: Propagate prefix sums top-down
//
// Example for [1, 2, 3, 4, 5, 6, 7, 8]:
//
// Up-sweep (reduce phase):
//   Level 0: [1, 2, 3, 4, 5, 6, 7, 8]
//   Level 1: [1, 3, 3, 7, 5, 11, 7, 15]  (pairs summed at positions 1,3,5,7)
//   Level 2: [1, 3, 3, 10, 5, 11, 7, 26] (sums at positions 3,7)
//   Level 3: [1, 3, 3, 10, 5, 11, 7, 36] (total sum at position 7)
//
// Down-sweep (distribute phase):
//   Set last element to identity (0)
//   Level 3: [1, 3, 3, 10, 5, 11, 7, 0]
//   Level 2: [1, 3, 3, 0, 5, 11, 7, 10]  (swap and add)
//   Level 1: [1, 0, 3, 3, 5, 10, 7, 21]
//   Level 0: [0, 1, 3, 6, 10, 15, 21, 28] (exclusive scan result)
//
// Complexity:
// - Work: O(n) - work-efficient!
// - Depth: O(log n)
// - Space: O(n) - in-place possible with careful indexing
//
// Trade-off vs Hillis-Steele:
// - Hillis-Steele: O(n log n) work, simpler, better for very wide GPUs
// - Blelloch: O(n) work, more complex, better when work matters

use std::ops::Add;

// Blelloch exclusive scan (work-efficient)
// Returns exclusive prefix sums: output[i] = sum of input[0..i]
pub fn blelloch_exclusive_scan<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    let n = input.len();

    // Pad to power of 2 for cleaner indexing
    let padded_len = n.next_power_of_two();
    let mut data = vec![T::default(); padded_len];
    data[..n].copy_from_slice(input);

    // Up-sweep (reduce) phase
    // Build partial sums tree
    let mut stride = 1;
    while stride < padded_len {
        // Process pairs at distance stride
        // Each iteration can be parallelized
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                data[right] = data[left] + data[right];
            }
        }
        stride *= 2;
    }

    // Set root to identity for exclusive scan
    data[padded_len - 1] = T::default();

    // Down-sweep phase
    // Distribute partial sums
    stride = padded_len / 2;
    while stride >= 1 {
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                let temp = data[left];
                data[left] = data[right];
                data[right] = temp + data[right];
            }
        }
        stride /= 2;
    }

    // Return only the original length
    data.truncate(n);
    data
}

// Blelloch inclusive scan
// Returns inclusive prefix sums: output[i] = sum of input[0..=i]
pub fn blelloch_inclusive_scan<T>(input: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    // Exclusive scan + shift left + add last element
    let exclusive = blelloch_exclusive_scan(input);

    exclusive
        .iter()
        .zip(input.iter())
        .map(|(&e, &i)| e + i)
        .collect()
}

// Generic Blelloch scan with custom associative operation
pub fn blelloch_scan_with_op<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if input.is_empty() {
        return vec![];
    }

    let n = input.len();
    let padded_len = n.next_power_of_two();
    let mut data = vec![identity; padded_len];
    data[..n].copy_from_slice(input);

    // Up-sweep
    let mut stride = 1;
    while stride < padded_len {
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                data[right] = op(data[left], data[right]);
            }
        }
        stride *= 2;
    }

    // Set root to identity
    data[padded_len - 1] = identity;

    // Down-sweep
    stride = padded_len / 2;
    while stride >= 1 {
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                let temp = data[left];
                data[left] = data[right];
                data[right] = op(temp, data[right]);
            }
        }
        stride /= 2;
    }

    data.truncate(n);
    data
}

// Visualization of Blelloch algorithm steps
#[derive(Debug, Clone)]
pub struct BlellochStep<T> {
    pub phase: String,
    pub level: usize,
    pub data: Vec<T>,
}

pub fn blelloch_scan_with_steps<T>(input: &[T]) -> (Vec<T>, Vec<BlellochStep<T>>)
where
    T: Copy + Add<Output = T> + Default,
{
    let mut steps = Vec::new();

    if input.is_empty() {
        return (vec![], steps);
    }

    let n = input.len();
    let padded_len = n.next_power_of_two();
    let mut data = vec![T::default(); padded_len];
    data[..n].copy_from_slice(input);

    steps.push(BlellochStep {
        phase: "initial".to_string(),
        level: 0,
        data: data[..n].to_vec(),
    });

    // Up-sweep
    let mut stride = 1;
    let mut level = 1;
    while stride < padded_len {
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                data[right] = data[left] + data[right];
            }
        }

        steps.push(BlellochStep {
            phase: "up-sweep".to_string(),
            level,
            data: data[..n].to_vec(),
        });

        stride *= 2;
        level += 1;
    }

    // Set root to identity
    data[padded_len - 1] = T::default();

    steps.push(BlellochStep {
        phase: "set-root".to_string(),
        level: 0,
        data: data[..n].to_vec(),
    });

    // Down-sweep
    stride = padded_len / 2;
    level = 1;
    while stride >= 1 {
        let step = stride * 2;
        for i in (0..padded_len).step_by(step) {
            let left = i + stride - 1;
            let right = i + step - 1;
            if right < padded_len {
                let temp = data[left];
                data[left] = data[right];
                data[right] = temp + data[right];
            }
        }

        steps.push(BlellochStep {
            phase: "down-sweep".to_string(),
            level,
            data: data[..n].to_vec(),
        });

        stride /= 2;
        level += 1;
    }

    (data[..n].to_vec(), steps)
}

// Block-based Blelloch scan for very large arrays
// More practical for real GPU implementations
pub struct BlockBlellochResult<T> {
    pub result: Vec<T>,
    pub block_sums: Vec<T>,
}

pub fn block_blelloch_scan<T>(input: &[T], block_size: usize) -> BlockBlellochResult<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return BlockBlellochResult {
            result: vec![],
            block_sums: vec![],
        };
    }

    let n = input.len();
    let num_blocks = n.div_ceil(block_size);

    // Phase 1: Scan within each block (parallel)
    let mut result = Vec::with_capacity(n);
    let mut block_sums = Vec::with_capacity(num_blocks);

    for block in 0..num_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(n);
        let block_data = &input[start..end];

        // Inclusive scan to get block sum
        let scanned = blelloch_inclusive_scan(block_data);

        // Store block sum (last element of inclusive scan)
        block_sums.push(*scanned.last().unwrap_or(&T::default()));

        result.extend(scanned);
    }

    // Phase 2: Scan block sums
    let scanned_block_sums = blelloch_exclusive_scan(&block_sums);

    // Phase 3: Add block prefix to each element (parallel)
    for (block, &prefix) in scanned_block_sums.iter().enumerate().skip(1) {
        let start = block * block_size;
        let end = (start + block_size).min(n);

        for item in result.iter_mut().take(end).skip(start) {
            *item = *item + prefix;
        }
    }

    BlockBlellochResult { result, block_sums }
}

// Compare work efficiency: Hillis-Steele vs Blelloch
#[derive(Debug)]
pub struct WorkComparison {
    pub n: usize,
    pub hillis_steele_work: usize, // O(n log n)
    pub blelloch_work: usize,      // O(n)
    pub hillis_steele_depth: usize,
    pub blelloch_depth: usize,
}

pub fn compare_work_efficiency(n: usize) -> WorkComparison {
    let log_n = (n as f64).log2().ceil() as usize;

    WorkComparison {
        n,
        hillis_steele_work: n * log_n,
        blelloch_work: 2 * (n - 1), // n-1 for up-sweep + n-1 for down-sweep
        hillis_steele_depth: log_n,
        blelloch_depth: 2 * log_n, // log n for each phase
    }
}

// Segmented Blelloch scan
// Performs independent scans within segments defined by flags
pub fn segmented_blelloch_scan<T>(input: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if input.is_empty() {
        return vec![];
    }

    assert_eq!(input.len(), flags.len());

    // Find segment boundaries
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;

    for (i, &flag) in flags.iter().enumerate() {
        if flag && i > start {
            segments.push((start, i));
            start = i;
        }
    }
    segments.push((start, input.len()));

    // Scan each segment independently
    let mut result = vec![T::default(); input.len()];

    for (seg_start, seg_end) in segments {
        let segment = &input[seg_start..seg_end];
        let scanned = blelloch_inclusive_scan(segment);

        result[seg_start..seg_end].copy_from_slice(&scanned);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blelloch_exclusive_scan() {
        let input = vec![1, 2, 3, 4, 5];
        let result = blelloch_exclusive_scan(&input);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_blelloch_inclusive_scan() {
        let input = vec![1, 2, 3, 4, 5];
        let result = blelloch_inclusive_scan(&input);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_blelloch_power_of_two() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = blelloch_exclusive_scan(&input);
        assert_eq!(result, vec![0, 1, 3, 6, 10, 15, 21, 28]);
    }

    #[test]
    fn test_blelloch_empty() {
        let input: Vec<i32> = vec![];
        assert!(blelloch_exclusive_scan(&input).is_empty());
        assert!(blelloch_inclusive_scan(&input).is_empty());
    }

    #[test]
    fn test_blelloch_single() {
        let input = vec![42];
        assert_eq!(blelloch_exclusive_scan(&input), vec![0]);
        assert_eq!(blelloch_inclusive_scan(&input), vec![42]);
    }

    #[test]
    fn test_blelloch_matches_sequential() {
        let input: Vec<i32> = (1..=16).collect();

        let exclusive = blelloch_exclusive_scan(&input);
        let inclusive = blelloch_inclusive_scan(&input);

        // Verify exclusive scan
        let mut expected_exclusive = vec![0];
        let mut acc = 0;
        for &x in input.iter().take(input.len() - 1) {
            acc += x;
            expected_exclusive.push(acc);
        }
        assert_eq!(exclusive, expected_exclusive);

        // Verify inclusive scan
        let mut expected_inclusive = Vec::new();
        acc = 0;
        for &x in &input {
            acc += x;
            expected_inclusive.push(acc);
        }
        assert_eq!(inclusive, expected_inclusive);
    }

    #[test]
    fn test_blelloch_with_op_multiplication() {
        let input = vec![1, 2, 3, 4];
        let result = blelloch_scan_with_op(&input, 1, |a, b| a * b);
        // Exclusive scan with multiplication: [1, 1, 2, 6]
        assert_eq!(result, vec![1, 1, 2, 6]);
    }

    #[test]
    fn test_blelloch_with_steps() {
        let input = vec![1, 2, 3, 4];
        let (result, steps) = blelloch_scan_with_steps(&input);

        assert_eq!(result, vec![0, 1, 3, 6]);
        assert!(!steps.is_empty());

        // Check phases exist
        assert!(steps.iter().any(|s| s.phase == "initial"));
        assert!(steps.iter().any(|s| s.phase == "up-sweep"));
        assert!(steps.iter().any(|s| s.phase == "set-root"));
        assert!(steps.iter().any(|s| s.phase == "down-sweep"));
    }

    #[test]
    fn test_block_blelloch_scan() {
        let input: Vec<i32> = (1..=12).collect();
        let result = block_blelloch_scan(&input, 4);

        // Should match regular inclusive scan
        let expected = blelloch_inclusive_scan(&input);
        assert_eq!(result.result, expected);

        // Check block sums
        assert_eq!(result.block_sums.len(), 3); // 12/4 = 3 blocks
    }

    #[test]
    fn test_block_blelloch_scan_uneven() {
        let input = vec![1, 2, 3, 4, 5, 6, 7]; // 7 elements, block size 3
        let result = block_blelloch_scan(&input, 3);

        let expected = blelloch_inclusive_scan(&input);
        assert_eq!(result.result, expected);
    }

    #[test]
    fn test_work_comparison() {
        let comp = compare_work_efficiency(1024);

        // Blelloch should have less work than Hillis-Steele
        assert!(comp.blelloch_work < comp.hillis_steele_work);

        // Both should have logarithmic depth
        assert!(comp.hillis_steele_depth <= 10); // log2(1024) = 10
        assert!(comp.blelloch_depth <= 20); // 2 * log2(1024)

        // Verify specific values
        assert_eq!(comp.hillis_steele_work, 1024 * 10);
        assert_eq!(comp.blelloch_work, 2 * 1023);
    }

    #[test]
    fn test_segmented_blelloch_scan() {
        let input = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = segmented_blelloch_scan(&input, &flags);

        // Segments: [1,2,3], [1,2], [1,2,3,4]
        // Results:  [1,3,6], [1,3], [1,3,6,10]
        assert_eq!(result, vec![1, 3, 6, 1, 3, 1, 3, 6, 10]);
    }

    #[test]
    fn test_blelloch_large_input() {
        let input: Vec<i64> = (1..=1000).collect();

        let exclusive = blelloch_exclusive_scan(&input);
        let inclusive = blelloch_inclusive_scan(&input);

        // Last element of exclusive should be sum of first 999 elements
        // Sum of 1..=999 = 999 * 1000 / 2 = 499500
        assert_eq!(exclusive[999], 499500);

        // Last element of inclusive should be sum of all 1000 elements
        // Sum of 1..=1000 = 1000 * 1001 / 2 = 500500
        assert_eq!(inclusive[999], 500500);
    }

    #[test]
    fn test_blelloch_non_power_of_two_sizes() {
        for size in [3, 5, 7, 9, 15, 17, 31, 33, 63, 65] {
            let input: Vec<i32> = (1..=size as i32).collect();
            let result = blelloch_inclusive_scan(&input);

            // Verify sum
            let expected_sum = (size * (size + 1) / 2) as i32;
            assert_eq!(
                *result.last().unwrap(),
                expected_sum,
                "Failed for size {}",
                size
            );
        }
    }

    #[test]
    fn test_blelloch_floating_point() {
        let input: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let result = blelloch_inclusive_scan(&input);

        assert!((result[0] - 0.1).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.6).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blelloch_vs_hillis_steele_result() {
        // Both algorithms should produce same result
        use crate::scan::{hillis_steele_exclusive, hillis_steele_inclusive};

        let input: Vec<i32> = (1..=32).collect();

        let blelloch_exc = blelloch_exclusive_scan(&input);
        let hillis_exc = hillis_steele_exclusive(&input);
        assert_eq!(blelloch_exc, hillis_exc);

        let blelloch_inc = blelloch_inclusive_scan(&input);
        let hillis_inc = hillis_steele_inclusive(&input);
        assert_eq!(blelloch_inc, hillis_inc);
    }

    #[test]
    fn test_blelloch_all_zeros() {
        let input = vec![0, 0, 0, 0, 0];
        let result = blelloch_exclusive_scan(&input);
        assert_eq!(result, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_blelloch_negative_numbers() {
        let input = vec![-1, -2, -3, -4];
        let result = blelloch_inclusive_scan(&input);
        assert_eq!(result, vec![-1, -3, -6, -10]);
    }
}
