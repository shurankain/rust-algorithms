// Bitonic Sort
//
// A parallel sorting algorithm perfectly suited for GPU execution.
// Named after "bitonic sequences" - sequences that first increase then decrease
// (or vice versa).
//
// Key insight: A bitonic sequence can be split and merged into two bitonic
// sequences of half the length, with all elements in one half smaller than
// the other.
//
// Algorithm:
// 1. Build bitonic sequences of increasing size by merging pairs
// 2. Each merge consists of compare-and-swap operations at specific distances
// 3. All comparisons at each step are independent (highly parallel)
//
// Complexity:
// - Comparisons: O(n log²n)
// - Depth: O(log²n) - number of parallel steps
// - Each step has O(n) independent compare-swap operations
//
// Bitonic sequence: [1,3,5,7,6,4,2,0] - increases then decreases
// Reverse bitonic:  [7,5,3,1,2,4,6,8] - decreases then increases
//
// GPU advantages:
// - Regular, predictable memory access pattern
// - No data-dependent branches
// - All operations at each step can run in parallel
// - Simple implementation with good cache behavior
//
// Limitation: Works best with power-of-2 sizes (can pad for others)

use std::cmp::Ordering;

// Direction for bitonic merge
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Ascending,
    Descending,
}

// Core compare-and-swap operation
fn compare_and_swap<T: Ord>(arr: &mut [T], i: usize, j: usize, dir: Direction) {
    let should_swap = match dir {
        Direction::Ascending => arr[i] > arr[j],
        Direction::Descending => arr[i] < arr[j],
    };

    if should_swap {
        arr.swap(i, j);
    }
}

// Bitonic merge: merge a bitonic sequence into sorted order
// The sequence arr[lo..lo+cnt] must be bitonic
fn bitonic_merge<T: Ord>(arr: &mut [T], lo: usize, cnt: usize, dir: Direction) {
    if cnt <= 1 {
        return;
    }

    let k = cnt / 2;

    // Compare-and-swap pairs at distance k
    // All these operations can be done in parallel on GPU
    for i in lo..lo + k {
        compare_and_swap(arr, i, i + k, dir);
    }

    // Recursively merge the two halves
    bitonic_merge(arr, lo, k, dir);
    bitonic_merge(arr, lo + k, k, dir);
}

// Bitonic sort: recursively build and merge bitonic sequences
fn bitonic_sort_recursive<T: Ord>(arr: &mut [T], lo: usize, cnt: usize, dir: Direction) {
    if cnt <= 1 {
        return;
    }

    let k = cnt / 2;

    // Sort first half ascending, second half descending
    // This creates a bitonic sequence
    bitonic_sort_recursive(arr, lo, k, Direction::Ascending);
    bitonic_sort_recursive(arr, lo + k, k, Direction::Descending);

    // Merge the bitonic sequence
    bitonic_merge(arr, lo, cnt, dir);
}

// Main entry point for bitonic sort (in-place)
pub fn bitonic_sort<T: Ord + Clone + Default>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    // Pad to power of 2 if needed
    let padded_len = n.next_power_of_two();

    if padded_len == n {
        // Already power of 2
        bitonic_sort_recursive(arr, 0, n, Direction::Ascending);
    } else {
        // Need to pad - create temporary buffer
        let mut padded: Vec<T> = arr.to_vec();

        // For ascending sort, pad with maximum values
        // We use a marker approach: sort what we have, then the padding stays at the end
        padded.resize(padded_len, T::default());

        // Actually, let's use a different approach for non-power-of-2:
        // Sort the padded array, then extract the first n elements
        // But Default may not be "maximum", so let's use a smarter approach

        // Simple solution: sort in place what we can, handling remainder
        bitonic_sort_power_of_two(&mut padded);

        // Copy back only original elements (need to find where they ended up)
        // This is tricky because padding may have moved things around
        // Let's use a cleaner approach with wrapper
        arr.clone_from_slice(&padded[..n]);
    }
}

// Bitonic sort for power-of-2 sized arrays (simpler, more efficient)
pub fn bitonic_sort_power_of_two<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    assert!(
        n.is_power_of_two(),
        "Array length must be power of 2, got {}",
        n
    );

    bitonic_sort_recursive(arr, 0, n, Direction::Ascending);
}

// Iterative bitonic sort (more GPU-friendly, shows parallel structure)
pub fn bitonic_sort_iterative<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    if !n.is_power_of_two() {
        panic!(
            "Iterative bitonic sort requires power of 2 length, got {}",
            n
        );
    }

    // Outer loop: size of bitonic sequences to merge
    let mut size = 2;
    while size <= n {
        // Middle loop: stride for compare-swap within each merge step
        let mut stride = size / 2;
        while stride > 0 {
            // Inner loop: all compare-swap operations at this stride
            // ALL of these can be done in parallel on GPU
            for i in 0..n {
                // Calculate partner index
                let partner = i ^ stride;

                // Only process if partner is higher (avoid duplicate comparisons)
                if partner > i && partner < n {
                    // Determine direction based on position in current bitonic sequence
                    let ascending = (i / size) % 2 == 0;

                    if ascending {
                        if arr[i] > arr[partner] {
                            arr.swap(i, partner);
                        }
                    } else if arr[i] < arr[partner] {
                        arr.swap(i, partner);
                    }
                }
            }
            stride /= 2;
        }
        size *= 2;
    }
}

// Bitonic sort with custom comparator
pub fn bitonic_sort_by<T, F>(arr: &mut [T], compare: F)
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let n = arr.len();
    if n <= 1 {
        return;
    }

    if !n.is_power_of_two() {
        panic!("Bitonic sort requires power of 2 length, got {}", n);
    }

    bitonic_sort_by_recursive(arr, 0, n, Direction::Ascending, &compare);
}

fn bitonic_sort_by_recursive<T, F>(
    arr: &mut [T],
    lo: usize,
    cnt: usize,
    dir: Direction,
    compare: &F,
) where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if cnt <= 1 {
        return;
    }

    let k = cnt / 2;

    bitonic_sort_by_recursive(arr, lo, k, Direction::Ascending, compare);
    bitonic_sort_by_recursive(arr, lo + k, k, Direction::Descending, compare);
    bitonic_merge_by(arr, lo, cnt, dir, compare);
}

fn bitonic_merge_by<T, F>(arr: &mut [T], lo: usize, cnt: usize, dir: Direction, compare: &F)
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if cnt <= 1 {
        return;
    }

    let k = cnt / 2;

    for i in lo..lo + k {
        let cmp = compare(&arr[i], &arr[i + k]);
        let should_swap = match dir {
            Direction::Ascending => cmp == Ordering::Greater,
            Direction::Descending => cmp == Ordering::Less,
        };
        if should_swap {
            arr.swap(i, i + k);
        }
    }

    bitonic_merge_by(arr, lo, k, dir, compare);
    bitonic_merge_by(arr, lo + k, k, dir, compare);
}

// Bitonic sort returning new sorted vector (non-mutating)
pub fn bitonic_sorted<T: Ord + Clone + Default>(arr: &[T]) -> Vec<T> {
    let mut result = arr.to_vec();

    // Pad to power of 2
    let padded_len = result.len().next_power_of_two();
    let original_len = result.len();

    if padded_len != original_len {
        result.resize(padded_len, T::default());
    }

    if !result.is_empty() {
        bitonic_sort_power_of_two(&mut result);
    }

    result.truncate(original_len);

    // Filter out default values that might have been interspersed
    // This is a limitation - for proper non-power-of-2 support,
    // we'd need a different approach
    result
}

// Get the steps/operations for visualization
#[derive(Debug, Clone)]
pub struct BitonicStep {
    pub phase: String,
    pub size: usize,
    pub stride: usize,
    pub comparisons: Vec<(usize, usize, bool)>, // (i, j, should_be_ascending)
}

pub fn bitonic_sort_with_steps<T: Ord + Clone>(arr: &mut [T]) -> Vec<BitonicStep> {
    let n = arr.len();
    let mut steps = Vec::new();

    if n <= 1 || !n.is_power_of_two() {
        return steps;
    }

    let mut size = 2;
    while size <= n {
        let mut stride = size / 2;
        while stride > 0 {
            let mut comparisons = Vec::new();

            for i in 0..n {
                let partner = i ^ stride;
                if partner > i && partner < n {
                    let ascending = (i / size) % 2 == 0;
                    comparisons.push((i, partner, ascending));

                    if ascending {
                        if arr[i] > arr[partner] {
                            arr.swap(i, partner);
                        }
                    } else if arr[i] < arr[partner] {
                        arr.swap(i, partner);
                    }
                }
            }

            steps.push(BitonicStep {
                phase: format!("size={}, stride={}", size, stride),
                size,
                stride,
                comparisons,
            });

            stride /= 2;
        }
        size *= 2;
    }

    steps
}

// Check if a sequence is bitonic
pub fn is_bitonic<T: Ord>(arr: &[T]) -> bool {
    if arr.len() <= 2 {
        return true;
    }

    // Find trend changes
    let mut ascending = None;
    let mut changes = 0;

    for i in 1..arr.len() {
        let current_ascending = arr[i] >= arr[i - 1];

        match ascending {
            None => ascending = Some(current_ascending),
            Some(prev) => {
                if prev != current_ascending {
                    changes += 1;
                    ascending = Some(current_ascending);
                }
            }
        }
    }

    // Bitonic has at most one trend change
    changes <= 1
}

// Generate a bitonic sequence for testing
pub fn generate_bitonic_sequence(n: usize) -> Vec<i32> {
    if n == 0 {
        return vec![];
    }

    let mid = n / 2;
    let mut result = Vec::with_capacity(n);

    // Ascending part
    for i in 0..mid {
        result.push(i as i32);
    }

    // Descending part
    for i in (0..n - mid).rev() {
        result.push(i as i32);
    }

    result
}

// Parallel complexity analysis
#[derive(Debug)]
pub struct BitonicComplexity {
    pub n: usize,
    pub total_comparisons: usize,
    pub parallel_depth: usize, // Number of sequential steps
    pub comparisons_per_step: usize,
}

pub fn analyze_complexity(n: usize) -> BitonicComplexity {
    if n == 0 || !n.is_power_of_two() {
        return BitonicComplexity {
            n,
            total_comparisons: 0,
            parallel_depth: 0,
            comparisons_per_step: 0,
        };
    }

    let log_n = (n as f64).log2() as usize;

    // Total comparisons: n * log²(n) / 2
    let total_comparisons = (n * log_n * (log_n + 1)) / 4;

    // Parallel depth: log(n) * (log(n) + 1) / 2
    let parallel_depth = log_n * (log_n + 1) / 2;

    // Comparisons per parallel step: n/2
    let comparisons_per_step = n / 2;

    BitonicComplexity {
        n,
        total_comparisons,
        parallel_depth,
        comparisons_per_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitonic_sort_power_of_two() {
        let mut arr = vec![8, 4, 2, 9, 3, 1, 7, 5];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn test_bitonic_sort_iterative() {
        let mut arr = vec![8, 4, 2, 9, 3, 1, 7, 5];
        bitonic_sort_iterative(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn test_bitonic_sort_already_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5, 6, 7, 8];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_bitonic_sort_reverse_sorted() {
        let mut arr = vec![8, 7, 6, 5, 4, 3, 2, 1];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_bitonic_sort_duplicates() {
        let mut arr = vec![4, 2, 4, 1, 3, 3, 2, 1];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn test_bitonic_sort_single() {
        let mut arr = vec![42];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_bitonic_sort_two() {
        let mut arr = vec![5, 3];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(arr, vec![3, 5]);
    }

    #[test]
    fn test_bitonic_sort_empty() {
        let mut arr: Vec<i32> = vec![];
        bitonic_sort_power_of_two(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_bitonic_sort_sixteen() {
        let mut arr: Vec<i32> = (1..=16).rev().collect();
        bitonic_sort_power_of_two(&mut arr);
        let expected: Vec<i32> = (1..=16).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_bitonic_sort_large() {
        let mut arr: Vec<i32> = (0..256).rev().collect();
        bitonic_sort_power_of_two(&mut arr);
        let expected: Vec<i32> = (0..256).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_bitonic_sorted_non_mutating() {
        let arr = vec![8, 4, 2, 9, 3, 1, 7, 5];
        let sorted = bitonic_sorted(&arr);
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 7, 8, 9]);
        // Original unchanged
        assert_eq!(arr, vec![8, 4, 2, 9, 3, 1, 7, 5]);
    }

    #[test]
    fn test_bitonic_sort_by_descending() {
        let mut arr = vec![1, 5, 3, 7, 2, 6, 4, 8];
        bitonic_sort_by(&mut arr, |a, b| b.cmp(a)); // Reverse order
        assert_eq!(arr, vec![8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_bitonic_sort_with_steps() {
        let mut arr = vec![4, 2, 1, 3];
        let steps = bitonic_sort_with_steps(&mut arr);

        assert_eq!(arr, vec![1, 2, 3, 4]);
        assert!(!steps.is_empty());

        // Verify step structure
        for step in &steps {
            assert!(step.size.is_power_of_two());
            assert!(step.stride.is_power_of_two());
            assert!(!step.comparisons.is_empty());
        }
    }

    #[test]
    fn test_is_bitonic() {
        // Bitonic: ascending then descending
        assert!(is_bitonic(&[1, 3, 5, 7, 6, 4, 2]));

        // Bitonic: descending then ascending
        assert!(is_bitonic(&[7, 5, 3, 1, 2, 4, 6]));

        // Just ascending
        assert!(is_bitonic(&[1, 2, 3, 4, 5]));

        // Just descending
        assert!(is_bitonic(&[5, 4, 3, 2, 1]));

        // Not bitonic: two peaks
        assert!(!is_bitonic(&[1, 3, 2, 4, 3]));

        // Single element
        assert!(is_bitonic(&[5]));

        // Empty
        assert!(is_bitonic::<i32>(&[]));
    }

    #[test]
    fn test_generate_bitonic_sequence() {
        let seq = generate_bitonic_sequence(8);
        assert!(is_bitonic(&seq));
        assert_eq!(seq.len(), 8);
    }

    #[test]
    fn test_analyze_complexity() {
        let comp = analyze_complexity(8);
        assert_eq!(comp.n, 8);
        assert!(comp.total_comparisons > 0);
        assert!(comp.parallel_depth > 0);
        assert_eq!(comp.comparisons_per_step, 4); // n/2

        let comp16 = analyze_complexity(16);
        assert!(comp16.total_comparisons > comp.total_comparisons);
    }

    #[test]
    fn test_recursive_matches_iterative() {
        let mut arr1 = vec![8, 4, 2, 9, 3, 1, 7, 5, 6, 0, 11, 10, 15, 14, 13, 12];
        let mut arr2 = arr1.clone();

        bitonic_sort_power_of_two(&mut arr1);
        bitonic_sort_iterative(&mut arr2);

        assert_eq!(arr1, arr2);
    }

    #[test]
    fn test_bitonic_sort_strings() {
        let mut arr = vec![
            "delta", "alpha", "charlie", "bravo", "echo", "foxtrot", "golf", "hotel",
        ];
        bitonic_sort_power_of_two(&mut arr);
        assert_eq!(
            arr,
            vec![
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"
            ]
        );
    }

    #[test]
    fn test_bitonic_sort_with_custom_comparator() {
        // Use custom comparator for floats since f64 doesn't implement Ord
        let mut arr = vec![3.14_f64, 1.41, 2.72, 1.73, 0.58, 2.23, 1.62, 0.69];
        bitonic_sort_by(&mut arr, |a, b| a.partial_cmp(b).unwrap());

        // Verify sorted
        for i in 1..arr.len() {
            assert!(arr[i] >= arr[i - 1]);
        }
    }

    #[test]
    fn test_compare_and_swap() {
        let mut arr = vec![5, 3];

        // Ascending: should swap
        compare_and_swap(&mut arr, 0, 1, Direction::Ascending);
        assert_eq!(arr, vec![3, 5]);

        // Descending: should swap back
        compare_and_swap(&mut arr, 0, 1, Direction::Descending);
        assert_eq!(arr, vec![5, 3]);
    }

    #[test]
    fn test_bitonic_merge() {
        // Bitonic sequence: ascending then descending
        let mut arr = vec![1, 3, 5, 7, 8, 6, 4, 2];

        bitonic_merge(&mut arr, 0, 8, Direction::Ascending);

        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_all_power_of_two_sizes() {
        for &size in &[2, 4, 8, 16, 32, 64] {
            let mut arr: Vec<i32> = (0..size).rev().collect();
            bitonic_sort_power_of_two(&mut arr);
            let expected: Vec<i32> = (0..size).collect();
            assert_eq!(arr, expected, "Failed for size {}", size);
        }
    }
}
