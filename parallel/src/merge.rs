// Parallel Merge
//
// Merges two sorted arrays in parallel - core building block for GPU merge sort.
// Unlike sequential merge (O(n)), parallel merge achieves O(log n) depth.
//
// Key insight: Use binary search to find merge positions in parallel.
// For each element, find where it belongs in the output using rank computation.
//
// Rank of element x in merged output = rank_in_A(x) + rank_in_B(x)
// where rank is the number of elements less than x.
//
// Algorithm (Merge Path):
// 1. Divide output into chunks of size p (number of processors)
// 2. For each chunk, binary search to find "merge path" - the diagonal
//    crossing point that divides work evenly
// 3. Each processor merges its assigned portion sequentially
//
// Complexity:
// - Work: O(n) - same as sequential
// - Depth: O(log n) with O(n) processors, or O(n/p + log n) with p processors
//
// Use cases:
// - GPU merge sort (parallel sorting)
// - Merging sorted results from parallel workers
// - Database merge joins on GPU

use std::cmp::Ordering;

// Sequential merge for reference and small inputs
pub fn sequential_merge<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// Binary search for the first position where a[pos] >= value
// Returns the count of elements strictly less than value
fn lower_bound<T: Ord>(arr: &[T], value: &T) -> usize {
    let mut lo = 0;
    let mut hi = arr.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] < *value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// Binary search for the first position where a[pos] > value
// Returns the count of elements less than or equal to value
fn upper_bound<T: Ord>(arr: &[T], value: &T) -> usize {
    let mut lo = 0;
    let mut hi = arr.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] <= *value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// Compute rank (output position) of each element from A in merged output
// This can be done in parallel on GPU - each element computes independently
pub fn compute_ranks_a<T: Ord>(a: &[T], b: &[T]) -> Vec<usize> {
    a.iter()
        .enumerate()
        .map(|(i, x)| i + lower_bound(b, x))
        .collect()
}

// Compute rank (output position) of each element from B in merged output
pub fn compute_ranks_b<T: Ord>(a: &[T], b: &[T]) -> Vec<usize> {
    b.iter()
        .enumerate()
        .map(|(j, x)| j + upper_bound(a, x))
        .collect()
}

// Parallel merge using rank computation
// Simulates GPU execution: each element independently finds its position
pub fn parallel_merge<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let n = a.len() + b.len();
    let mut result = vec![a[0]; n]; // Initialize with dummy value

    // Phase 1: Compute output positions for all elements (parallel on GPU)
    let ranks_a = compute_ranks_a(a, b);
    let ranks_b = compute_ranks_b(a, b);

    // Phase 2: Scatter to output positions (parallel on GPU)
    for (i, &rank) in ranks_a.iter().enumerate() {
        result[rank] = a[i];
    }
    for (j, &rank) in ranks_b.iter().enumerate() {
        result[rank] = b[j];
    }

    result
}

// Merge path: find the diagonal intersection point for balanced partitioning
// Returns (i, j) where we should split: A[0..i] and B[0..j] contain exactly `diag` elements
fn merge_path<T: Ord>(a: &[T], b: &[T], diag: usize) -> (usize, usize) {
    let m = a.len();
    let n = b.len();

    // Binary search on the diagonal
    // We want to find (i, j) such that:
    // - i + j = diag
    // - a[i-1] <= b[j] (if valid indices)
    // - b[j-1] < a[i] (if valid indices)

    let mut lo = diag.saturating_sub(n);
    let mut hi = diag.min(m);

    while lo < hi {
        let i = lo + (hi - lo).div_ceil(2);
        let j = diag - i;

        if j < n && i > 0 && a[i - 1] > b[j] {
            hi = i - 1;
        } else {
            lo = i;
        }
    }

    (lo, diag - lo)
}

// Block-based parallel merge using merge path
// Divides work among `num_blocks` processors
pub fn block_parallel_merge<T: Ord + Copy>(a: &[T], b: &[T], num_blocks: usize) -> Vec<T> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let total = a.len() + b.len();
    let num_blocks = num_blocks.max(1).min(total);
    let block_size = total.div_ceil(num_blocks);

    let mut result = Vec::with_capacity(total);

    // Find merge path points for each block boundary
    let mut merge_points: Vec<(usize, usize)> = Vec::with_capacity(num_blocks + 1);
    merge_points.push((0, 0));

    for block in 1..num_blocks {
        let diag = (block * block_size).min(total);
        merge_points.push(merge_path(a, b, diag));
    }
    merge_points.push((a.len(), b.len()));

    // Process each block (can be parallel on GPU)
    for block in 0..num_blocks {
        let (a_start, b_start) = merge_points[block];
        let (a_end, b_end) = merge_points[block + 1];

        // Sequential merge within this block
        let a_slice = &a[a_start..a_end];
        let b_slice = &b[b_start..b_end];
        let merged = sequential_merge(a_slice, b_slice);
        result.extend(merged);
    }

    result
}

// Merge with custom comparator
pub fn parallel_merge_by<T, F>(a: &[T], b: &[T], compare: F) -> Vec<T>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let n = a.len() + b.len();
    let mut result = Vec::with_capacity(n);

    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match compare(&a[i], &b[j]) {
            Ordering::Less | Ordering::Equal => {
                result.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// Parallel merge sort using parallel merge as building block
pub fn parallel_merge_sort<T: Ord + Copy>(input: &[T]) -> Vec<T> {
    if input.len() <= 1 {
        return input.to_vec();
    }

    // Threshold for switching to insertion sort
    if input.len() <= 16 {
        let mut result = input.to_vec();
        result.sort();
        return result;
    }

    let mid = input.len() / 2;

    // Recursive sort (these can run in parallel)
    let left = parallel_merge_sort(&input[..mid]);
    let right = parallel_merge_sort(&input[mid..]);

    // Parallel merge
    parallel_merge(&left, &right)
}

// Merge K sorted arrays using parallel pairwise merging
pub fn parallel_merge_k<T: Ord + Copy>(arrays: &[Vec<T>]) -> Vec<T> {
    if arrays.is_empty() {
        return vec![];
    }
    if arrays.len() == 1 {
        return arrays[0].clone();
    }

    let mut current: Vec<Vec<T>> = arrays.to_vec();

    // Pairwise merge until single array remains
    // Each level can be parallelized
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len().div_ceil(2));

        for pair in current.chunks(2) {
            if pair.len() == 2 {
                next.push(parallel_merge(&pair[0], &pair[1]));
            } else {
                next.push(pair[0].clone());
            }
        }

        current = next;
    }

    current.into_iter().next().unwrap_or_default()
}

// Intersection of two sorted arrays (parallel-friendly)
pub fn parallel_intersection<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    result
}

// Union of two sorted arrays (parallel-friendly, removes duplicates)
pub fn parallel_union<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
            }
            Ordering::Greater => {
                if result.last() != Some(&b[j]) {
                    result.push(b[j]);
                }
                j += 1;
            }
            Ordering::Equal => {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
                j += 1;
            }
        }
    }

    while i < a.len() {
        if result.last() != Some(&a[i]) {
            result.push(a[i]);
        }
        i += 1;
    }

    while j < b.len() {
        if result.last() != Some(&b[j]) {
            result.push(b[j]);
        }
        j += 1;
    }

    result
}

// Difference of two sorted arrays: elements in A but not in B
pub fn parallel_difference<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() {
        return vec![];
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                j += 1;
            }
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_merge() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 4, 6, 8];
        assert_eq!(sequential_merge(&a, &b), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_sequential_merge_unequal() {
        let a = vec![1, 2];
        let b = vec![3, 4, 5, 6];
        assert_eq!(sequential_merge(&a, &b), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_sequential_merge_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(sequential_merge(&a, &b), vec![1, 2, 3]);
        assert_eq!(sequential_merge(&b, &a), vec![1, 2, 3]);
    }

    #[test]
    fn test_parallel_merge() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 4, 6, 8];
        assert_eq!(parallel_merge(&a, &b), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_parallel_merge_duplicates() {
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        assert_eq!(parallel_merge(&a, &b), vec![1, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn test_parallel_merge_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(parallel_merge(&a, &b), vec![1, 2, 3]);
        assert_eq!(parallel_merge(&b, &a), vec![1, 2, 3]);
    }

    #[test]
    fn test_parallel_merge_single() {
        let a = vec![1];
        let b = vec![2];
        assert_eq!(parallel_merge(&a, &b), vec![1, 2]);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let a = vec![1, 4, 7, 10, 13, 16, 19];
        let b = vec![2, 3, 5, 8, 11, 14, 17, 20];

        let seq = sequential_merge(&a, &b);
        let par = parallel_merge(&a, &b);

        assert_eq!(seq, par);
    }

    #[test]
    fn test_compute_ranks() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];

        let ranks_a = compute_ranks_a(&a, &b);
        let ranks_b = compute_ranks_b(&a, &b);

        // a[0]=1 goes to position 0 (0 from a, 0 from b less than 1)
        // a[1]=3 goes to position 2 (1 from a, 1 from b less than 3)
        // a[2]=5 goes to position 4 (2 from a, 2 from b less than 5)
        assert_eq!(ranks_a, vec![0, 2, 4]);

        // b[0]=2 goes to position 1 (0 from b, 1 from a <= 2)
        // b[1]=4 goes to position 3 (1 from b, 2 from a <= 4)
        // b[2]=6 goes to position 5 (2 from b, 3 from a <= 6)
        assert_eq!(ranks_b, vec![1, 3, 5]);
    }

    #[test]
    fn test_block_parallel_merge() {
        let a = vec![1, 3, 5, 7, 9, 11, 13, 15];
        let b = vec![2, 4, 6, 8, 10, 12, 14, 16];

        let result = block_parallel_merge(&a, &b, 4);
        let expected: Vec<i32> = (1..=16).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_block_parallel_merge_unequal() {
        let a = vec![1, 5, 9];
        let b = vec![2, 3, 4, 6, 7, 8, 10];

        let result = block_parallel_merge(&a, &b, 3);
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_merge_path() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 4, 6, 8];

        // At diagonal 4, we want 4 elements total
        let (i, j) = merge_path(&a, &b, 4);
        assert_eq!(i + j, 4);

        // Verify the merge path property
        if i > 0 && j < b.len() {
            assert!(a[i - 1] <= b[j]);
        }
    }

    #[test]
    fn test_parallel_merge_sort() {
        let input = vec![8, 4, 2, 9, 3, 1, 7, 5, 6];
        let result = parallel_merge_sort(&input);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_merge_sort_empty() {
        let input: Vec<i32> = vec![];
        assert!(parallel_merge_sort(&input).is_empty());
    }

    #[test]
    fn test_parallel_merge_sort_single() {
        let input = vec![42];
        assert_eq!(parallel_merge_sort(&input), vec![42]);
    }

    #[test]
    fn test_parallel_merge_sort_sorted() {
        let input: Vec<i32> = (1..=10).collect();
        assert_eq!(parallel_merge_sort(&input), input);
    }

    #[test]
    fn test_parallel_merge_sort_reverse() {
        let input: Vec<i32> = (1..=10).rev().collect();
        let expected: Vec<i32> = (1..=10).collect();
        assert_eq!(parallel_merge_sort(&input), expected);
    }

    #[test]
    fn test_parallel_merge_k() {
        let arrays = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
        let result = parallel_merge_k(&arrays);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_merge_k_empty() {
        let arrays: Vec<Vec<i32>> = vec![];
        assert!(parallel_merge_k(&arrays).is_empty());
    }

    #[test]
    fn test_parallel_merge_k_single() {
        let arrays = vec![vec![1, 2, 3]];
        assert_eq!(parallel_merge_k(&arrays), vec![1, 2, 3]);
    }

    #[test]
    fn test_parallel_merge_k_many() {
        let arrays: Vec<Vec<i32>> = (0..8)
            .map(|i| vec![i * 3 + 1, i * 3 + 2, i * 3 + 3])
            .collect();
        let result = parallel_merge_k(&arrays);
        let expected: Vec<i32> = (1..=24).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_intersection() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        assert_eq!(parallel_intersection(&a, &b), vec![3, 4, 5]);
    }

    #[test]
    fn test_parallel_intersection_empty() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert!(parallel_intersection(&a, &b).is_empty());
    }

    #[test]
    fn test_parallel_intersection_one_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert!(parallel_intersection(&a, &b).is_empty());
    }

    #[test]
    fn test_parallel_union() {
        let a = vec![1, 2, 3, 4];
        let b = vec![3, 4, 5, 6];
        assert_eq!(parallel_union(&a, &b), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_parallel_union_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(parallel_union(&a, &b), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_parallel_union_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(parallel_union(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn test_parallel_difference() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![2, 4];
        assert_eq!(parallel_difference(&a, &b), vec![1, 3, 5]);
    }

    #[test]
    fn test_parallel_difference_no_common() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        assert_eq!(parallel_difference(&a, &b), vec![1, 3, 5]);
    }

    #[test]
    fn test_parallel_difference_all_common() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        assert!(parallel_difference(&a, &b).is_empty());
    }

    #[test]
    fn test_parallel_difference_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert!(parallel_difference(&a, &b).is_empty());

        let a = vec![1, 2, 3];
        let b: Vec<i32> = vec![];
        assert_eq!(parallel_difference(&a, &b), vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_by_descending() {
        let a = vec![7, 5, 3, 1];
        let b = vec![8, 6, 4, 2];
        let result = parallel_merge_by(&a, &b, |x, y| y.cmp(x)); // reverse order
        assert_eq!(result, vec![8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_large_parallel_merge() {
        let a: Vec<i32> = (0..1000).step_by(2).collect(); // evens
        let b: Vec<i32> = (1..1000).step_by(2).collect(); // odds

        let result = parallel_merge(&a, &b);
        let expected: Vec<i32> = (0..1000).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_large_block_merge() {
        let a: Vec<i32> = (0..500).map(|x| x * 2).collect();
        let b: Vec<i32> = (0..500).map(|x| x * 2 + 1).collect();

        let result = block_parallel_merge(&a, &b, 8);
        let expected: Vec<i32> = (0..1000).collect();

        assert_eq!(result, expected);
    }
}
