// Odd-Even Merge Sort (Batcher's Algorithm)
//
// A parallel sorting network that combines odd-even merging with recursive sorting.
// Named after Kenneth Batcher who invented it in 1968.
//
// Key insight: Merge two sorted sequences by:
// 1. Recursively merge odd-indexed elements
// 2. Recursively merge even-indexed elements
// 3. Compare-swap adjacent pairs to fix any remaining inversions
//
// Why it works:
// After merging odds and evens separately, any inversion must be between
// adjacent elements - a beautiful mathematical property called the "0-1 principle".
//
// Complexity:
// - Comparisons: O(n log²n) - same as bitonic sort
// - Depth: O(log²n) parallel steps
// - Each step has O(n) independent comparisons
//
// Comparison with Bitonic Sort:
// - Both are sorting networks with same asymptotic complexity
// - Odd-even merge has simpler merge logic
// - Bitonic sort has more regular comparison patterns
// - Both are excellent for GPU implementation
//
// Sorting network property:
// - Fixed comparison sequence independent of data
// - No branches based on comparisons (just conditional swaps)
// - Perfect for SIMD and GPU execution

use std::cmp::Ordering;

// Core compare-and-swap for ascending order
fn compare_swap<T: Ord>(arr: &mut [T], i: usize, j: usize) {
    if i < arr.len() && j < arr.len() && arr[i] > arr[j] {
        arr.swap(i, j);
    }
}

// Odd-even merge: merge two sorted halves into one sorted sequence
// Assumes arr[lo..lo+n/2] and arr[lo+n/2..lo+n] are both sorted
fn odd_even_merge<T: Ord>(arr: &mut [T], lo: usize, n: usize, step: usize) {
    if n <= 1 {
        return;
    }

    if n == 2 {
        compare_swap(arr, lo, lo + step);
        return;
    }

    let half = n / 2;

    // Recursively merge odd-indexed elements
    odd_even_merge(arr, lo, half, step * 2);

    // Recursively merge even-indexed elements
    odd_even_merge(arr, lo + step, half, step * 2);

    // Compare-swap adjacent pairs to fix remaining inversions
    // This is the key step that makes odd-even merge work
    for i in 1..n - 1 {
        if i % 2 == 1 {
            compare_swap(arr, lo + i * step, lo + (i + 1) * step);
        }
    }
}

// Recursive odd-even merge sort
fn odd_even_merge_sort_recursive<T: Ord>(arr: &mut [T], lo: usize, n: usize) {
    if n <= 1 {
        return;
    }

    let half = n / 2;

    // Sort first half
    odd_even_merge_sort_recursive(arr, lo, half);

    // Sort second half
    odd_even_merge_sort_recursive(arr, lo + half, half);

    // Merge the two sorted halves
    odd_even_merge(arr, lo, n, 1);
}

// Main entry point for odd-even merge sort (in-place)
pub fn odd_even_merge_sort<T: Ord + Clone + Default>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    // Pad to power of 2 if needed
    let padded_len = n.next_power_of_two();

    if padded_len == n {
        odd_even_merge_sort_recursive(arr, 0, n);
    } else {
        // Pad and sort
        let mut padded: Vec<T> = arr.to_vec();
        padded.resize(padded_len, T::default());
        odd_even_merge_sort_recursive(&mut padded, 0, padded_len);
        arr.clone_from_slice(&padded[..n]);
    }
}

// Odd-even merge sort for power-of-2 arrays (no padding needed)
pub fn odd_even_merge_sort_power_of_two<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    assert!(
        n.is_power_of_two(),
        "Array length must be power of 2, got {}",
        n
    );

    odd_even_merge_sort_recursive(arr, 0, n);
}

// Iterative version - shows the parallel structure more clearly
pub fn odd_even_merge_sort_iterative<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    if !n.is_power_of_two() {
        panic!(
            "Iterative odd-even merge sort requires power of 2 length, got {}",
            n
        );
    }

    // Build sorted sequences of increasing size
    let mut size = 2;
    while size <= n {
        // Merge pairs of sorted sequences of size `size/2` into sorted sequences of size `size`
        let mut start = 0;
        while start < n {
            odd_even_merge_iterative_merge(arr, start, size);
            start += size;
        }
        size *= 2;
    }
}

// Iterative merge for a single block
fn odd_even_merge_iterative_merge<T: Ord>(arr: &mut [T], lo: usize, n: usize) {
    // Process in rounds, each round halves the "stride"
    let mut stride = n / 2;
    while stride > 0 {
        // Within each stride, compare-swap pairs
        for i in 0..n {
            let partner = i ^ stride;
            if partner > i {
                // Determine if this comparison should sort ascending or descending
                // based on position in the overall sorting network
                let block = i / stride;
                if block % 2 == 0 || stride == n / 2 {
                    // First merge step or even block: always compare-swap
                    if arr[lo + i] > arr[lo + partner] {
                        arr.swap(lo + i, lo + partner);
                    }
                }
            }
        }
        stride /= 2;
    }

    // Additional cleanup passes for odd-even merge
    // Compare adjacent pairs
    let mut step = 1;
    while step < n {
        for i in (step..n - step).step_by(step * 2) {
            compare_swap(arr, lo + i, lo + i + step);
        }
        step *= 2;
    }
}

// Odd-even merge sort with custom comparator
pub fn odd_even_merge_sort_by<T, F>(arr: &mut [T], compare: F)
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let n = arr.len();
    if n <= 1 {
        return;
    }

    if !n.is_power_of_two() {
        panic!("Odd-even merge sort requires power of 2 length, got {}", n);
    }

    odd_even_merge_sort_by_recursive(arr, 0, n, &compare);
}

fn odd_even_merge_sort_by_recursive<T, F>(arr: &mut [T], lo: usize, n: usize, compare: &F)
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if n <= 1 {
        return;
    }

    let half = n / 2;

    odd_even_merge_sort_by_recursive(arr, lo, half, compare);
    odd_even_merge_sort_by_recursive(arr, lo + half, half, compare);
    odd_even_merge_by(arr, lo, n, 1, compare);
}

fn odd_even_merge_by<T, F>(arr: &mut [T], lo: usize, n: usize, step: usize, compare: &F)
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if n <= 1 {
        return;
    }

    if n == 2 {
        if lo + step < arr.len() && compare(&arr[lo], &arr[lo + step]) == Ordering::Greater {
            arr.swap(lo, lo + step);
        }
        return;
    }

    let half = n / 2;

    odd_even_merge_by(arr, lo, half, step * 2, compare);
    odd_even_merge_by(arr, lo + step, half, step * 2, compare);

    for i in 1..n - 1 {
        if i % 2 == 1 {
            let idx1 = lo + i * step;
            let idx2 = lo + (i + 1) * step;
            if idx2 < arr.len() && compare(&arr[idx1], &arr[idx2]) == Ordering::Greater {
                arr.swap(idx1, idx2);
            }
        }
    }
}

// Non-mutating version
pub fn odd_even_merge_sorted<T: Ord + Clone + Default>(arr: &[T]) -> Vec<T> {
    let mut result = arr.to_vec();
    let padded_len = result.len().next_power_of_two();
    let original_len = result.len();

    if padded_len != original_len {
        result.resize(padded_len, T::default());
    }

    if !result.is_empty() {
        odd_even_merge_sort_power_of_two(&mut result);
    }

    result.truncate(original_len);
    result
}

// Visualization: get the comparison network
#[derive(Debug, Clone)]
pub struct CompareSwapOp {
    pub i: usize,
    pub j: usize,
    pub step: usize, // Which parallel step this belongs to
}

#[derive(Debug, Clone)]
pub struct SortingNetwork {
    pub size: usize,
    pub operations: Vec<CompareSwapOp>,
    pub depth: usize, // Number of parallel steps
}

pub fn generate_odd_even_merge_network(n: usize) -> SortingNetwork {
    if n == 0 || !n.is_power_of_two() {
        return SortingNetwork {
            size: n,
            operations: vec![],
            depth: 0,
        };
    }

    let mut operations = Vec::new();
    let mut step = 0;

    generate_network_recursive(n, 0, n, &mut operations, &mut step);

    let depth = if operations.is_empty() {
        0
    } else {
        operations.iter().map(|op| op.step).max().unwrap_or(0) + 1
    };

    SortingNetwork {
        size: n,
        operations,
        depth,
    }
}

fn generate_network_recursive(
    _n: usize,
    lo: usize,
    cnt: usize,
    ops: &mut Vec<CompareSwapOp>,
    step: &mut usize,
) {
    if cnt <= 1 {
        return;
    }

    let half = cnt / 2;

    // Sort first half
    generate_network_recursive(_n, lo, half, ops, step);

    // Sort second half
    generate_network_recursive(_n, lo + half, half, ops, step);

    // Generate merge network
    generate_merge_network(lo, cnt, 1, ops, step);
}

fn generate_merge_network(
    lo: usize,
    n: usize,
    step_size: usize,
    ops: &mut Vec<CompareSwapOp>,
    step: &mut usize,
) {
    if n <= 1 {
        return;
    }

    if n == 2 {
        ops.push(CompareSwapOp {
            i: lo,
            j: lo + step_size,
            step: *step,
        });
        return;
    }

    let half = n / 2;

    // Merge odds
    generate_merge_network(lo, half, step_size * 2, ops, step);

    // Merge evens
    generate_merge_network(lo + step_size, half, step_size * 2, ops, step);

    *step += 1;

    // Adjacent compare-swaps
    for i in 1..n - 1 {
        if i % 2 == 1 {
            ops.push(CompareSwapOp {
                i: lo + i * step_size,
                j: lo + (i + 1) * step_size,
                step: *step,
            });
        }
    }
}

// Odd-even transposition sort (simpler but O(n) depth)
// Different from odd-even merge sort - this is a simple bubble-sort variant
pub fn odd_even_transposition_sort<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    // O(n) rounds, each round has n/2 parallel comparisons
    for _ in 0..n {
        // Odd phase: compare (1,2), (3,4), (5,6), ...
        for i in (1..n - 1).step_by(2) {
            if arr[i] > arr[i + 1] {
                arr.swap(i, i + 1);
            }
        }

        // Even phase: compare (0,1), (2,3), (4,5), ...
        for i in (0..n - 1).step_by(2) {
            if arr[i] > arr[i + 1] {
                arr.swap(i, i + 1);
            }
        }
    }
}

// Complexity analysis
#[derive(Debug)]
pub struct OddEvenMergeComplexity {
    pub n: usize,
    pub total_comparisons: usize,
    pub parallel_depth: usize,
    pub comparisons_per_step: usize,
}

pub fn analyze_odd_even_merge_complexity(n: usize) -> OddEvenMergeComplexity {
    if n == 0 || !n.is_power_of_two() {
        return OddEvenMergeComplexity {
            n,
            total_comparisons: 0,
            parallel_depth: 0,
            comparisons_per_step: 0,
        };
    }

    let network = generate_odd_even_merge_network(n);

    OddEvenMergeComplexity {
        n,
        total_comparisons: network.operations.len(),
        parallel_depth: network.depth,
        comparisons_per_step: n / 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_odd_even_merge_sort_power_of_two() {
        let mut arr = vec![8, 4, 2, 9, 3, 1, 7, 5];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn test_odd_even_merge_sort_already_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5, 6, 7, 8];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_odd_even_merge_sort_reverse_sorted() {
        let mut arr = vec![8, 7, 6, 5, 4, 3, 2, 1];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_odd_even_merge_sort_duplicates() {
        let mut arr = vec![4, 2, 4, 1, 3, 3, 2, 1];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn test_odd_even_merge_sort_single() {
        let mut arr = vec![42];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_odd_even_merge_sort_two() {
        let mut arr = vec![5, 3];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![3, 5]);
    }

    #[test]
    fn test_odd_even_merge_sort_empty() {
        let mut arr: Vec<i32> = vec![];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_odd_even_merge_sort_sixteen() {
        let mut arr: Vec<i32> = (1..=16).rev().collect();
        odd_even_merge_sort_power_of_two(&mut arr);
        let expected: Vec<i32> = (1..=16).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_odd_even_merge_sort_large() {
        let mut arr: Vec<i32> = (0..256).rev().collect();
        odd_even_merge_sort_power_of_two(&mut arr);
        let expected: Vec<i32> = (0..256).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_odd_even_merge_sorted_non_mutating() {
        let arr = vec![8, 4, 2, 9, 3, 1, 7, 5];
        let sorted = odd_even_merge_sorted(&arr);
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 7, 8, 9]);
        // Original unchanged
        assert_eq!(arr, vec![8, 4, 2, 9, 3, 1, 7, 5]);
    }

    #[test]
    fn test_odd_even_merge_sort_by_descending() {
        let mut arr = vec![1, 5, 3, 7, 2, 6, 4, 8];
        odd_even_merge_sort_by(&mut arr, |a, b| b.cmp(a));
        assert_eq!(arr, vec![8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_odd_even_merge_sort_strings() {
        let mut arr = vec![
            "delta", "alpha", "charlie", "bravo", "echo", "foxtrot", "golf", "hotel",
        ];
        odd_even_merge_sort_power_of_two(&mut arr);
        assert_eq!(
            arr,
            vec![
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"
            ]
        );
    }

    #[test]
    fn test_odd_even_transposition_sort() {
        let mut arr = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
        odd_even_transposition_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_odd_even_transposition_sort_any_size() {
        // Unlike merge sort, transposition sort works with any size
        let mut arr = vec![5, 3, 8, 1, 9];
        odd_even_transposition_sort(&mut arr);
        assert_eq!(arr, vec![1, 3, 5, 8, 9]);
    }

    #[test]
    fn test_generate_network() {
        let network = generate_odd_even_merge_network(8);
        assert_eq!(network.size, 8);
        assert!(!network.operations.is_empty());
        assert!(network.depth > 0);

        // Verify all indices are valid
        for op in &network.operations {
            assert!(op.i < 8);
            assert!(op.j < 8);
            assert!(op.i < op.j);
        }
    }

    #[test]
    fn test_analyze_complexity() {
        let comp = analyze_odd_even_merge_complexity(8);
        assert_eq!(comp.n, 8);
        assert!(comp.total_comparisons > 0);
        assert!(comp.parallel_depth > 0);
        assert_eq!(comp.comparisons_per_step, 4);

        let comp16 = analyze_odd_even_merge_complexity(16);
        assert!(comp16.total_comparisons > comp.total_comparisons);
    }

    #[test]
    fn test_all_power_of_two_sizes() {
        for &size in &[2, 4, 8, 16, 32, 64] {
            let mut arr: Vec<i32> = (0..size).rev().collect();
            odd_even_merge_sort_power_of_two(&mut arr);
            let expected: Vec<i32> = (0..size).collect();
            assert_eq!(arr, expected, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_compare_swap() {
        let mut arr = vec![5, 3, 8, 1];

        compare_swap(&mut arr, 0, 1);
        assert_eq!(arr, vec![3, 5, 8, 1]);

        compare_swap(&mut arr, 2, 3);
        assert_eq!(arr, vec![3, 5, 1, 8]);
    }

    #[test]
    fn test_odd_even_merge() {
        // Two sorted halves
        let mut arr = vec![1, 3, 5, 7, 2, 4, 6, 8];
        odd_even_merge(&mut arr, 0, 8, 1);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_sorting_network_correctness() {
        // Verify that applying the network operations sorts any input
        let network = generate_odd_even_merge_network(8);

        let mut arr = vec![8, 4, 2, 9, 3, 1, 7, 5];

        // Group operations by step and apply
        let mut current_step = 0;
        for op in &network.operations {
            if op.step != current_step {
                current_step = op.step;
            }
            if arr[op.i] > arr[op.j] {
                arr.swap(op.i, op.j);
            }
        }

        // Should be sorted
        for i in 1..arr.len() {
            assert!(arr[i] >= arr[i - 1], "Array not sorted at index {}", i);
        }
    }

    #[test]
    fn test_matches_bitonic_sort_result() {
        use crate::bitonic_sort::bitonic_sort_power_of_two;

        let original = vec![8, 4, 2, 9, 3, 1, 7, 5, 6, 0, 11, 10, 15, 14, 13, 12];

        let mut bitonic = original.clone();
        let mut odd_even = original.clone();

        bitonic_sort_power_of_two(&mut bitonic);
        odd_even_merge_sort_power_of_two(&mut odd_even);

        assert_eq!(bitonic, odd_even);
    }

    #[test]
    fn test_odd_even_merge_sort_with_custom_comparator() {
        let mut arr = vec![3.14_f64, 1.41, 2.72, 1.73, 0.58, 2.23, 1.62, 0.69];
        odd_even_merge_sort_by(&mut arr, |a, b| a.partial_cmp(b).unwrap());

        for i in 1..arr.len() {
            assert!(arr[i] >= arr[i - 1]);
        }
    }
}
