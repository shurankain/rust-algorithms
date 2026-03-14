// Tim Sort - Hybrid stable sorting algorithm
// Combines merge sort and insertion sort
// Used as the default sort in Python and Java
// Time: O(n log n) worst case, O(n) best case (already sorted)
// Space: O(n)
// Stable: Yes

const MIN_MERGE: usize = 32;

/// Calculate minimum run length for Tim Sort
/// Returns a value such that n/minrun is a power of 2 or close to it
fn calc_min_run(mut n: usize) -> usize {
    let mut r = 0;
    while n >= MIN_MERGE {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

/// Tim Sort for i32 slices
/// Time: O(n log n), Space: O(n), Stable
pub fn timsort(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    let min_run = calc_min_run(n);

    // Sort individual runs using insertion sort
    let mut start = 0;
    while start < n {
        let end = (start + min_run).min(n);
        insertion_sort_range(arr, start, end);
        start = end;
    }

    // Merge runs
    let mut size = min_run;
    while size < n {
        let mut left = 0;
        while left < n {
            let mid = (left + size).min(n);
            let right = (left + 2 * size).min(n);

            if mid < right {
                merge(arr, left, mid, right);
            }

            left += 2 * size;
        }
        size *= 2;
    }
}

/// Generic Tim Sort
pub fn timsort_generic<T: Ord + Clone>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    let min_run = calc_min_run(n);

    // Sort individual runs using insertion sort
    let mut start = 0;
    while start < n {
        let end = (start + min_run).min(n);
        insertion_sort_range_generic(arr, start, end);
        start = end;
    }

    // Merge runs
    let mut size = min_run;
    while size < n {
        let mut left = 0;
        while left < n {
            let mid = (left + size).min(n);
            let right = (left + 2 * size).min(n);

            if mid < right {
                merge_generic(arr, left, mid, right);
            }

            left += 2 * size;
        }
        size *= 2;
    }
}

/// Tim Sort with custom comparator
pub fn timsort_by<T: Clone, F>(arr: &mut [T], compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering + Copy,
{
    let n = arr.len();
    if n < 2 {
        return;
    }

    let min_run = calc_min_run(n);

    // Sort individual runs
    let mut start = 0;
    while start < n {
        let end = (start + min_run).min(n);
        insertion_sort_range_by(arr, start, end, compare);
        start = end;
    }

    // Merge runs
    let mut size = min_run;
    while size < n {
        let mut left = 0;
        while left < n {
            let mid = (left + size).min(n);
            let right = (left + 2 * size).min(n);

            if mid < right {
                merge_by(arr, left, mid, right, compare);
            }

            left += 2 * size;
        }
        size *= 2;
    }
}

/// Insertion sort for a range [start, end)
fn insertion_sort_range(arr: &mut [i32], start: usize, end: usize) {
    for i in (start + 1)..end {
        let key = arr[i];
        let mut j = i;

        while j > start && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Generic insertion sort for a range
fn insertion_sort_range_generic<T: Ord + Clone>(arr: &mut [T], start: usize, end: usize) {
    for i in (start + 1)..end {
        let key = arr[i].clone();
        let mut j = i;

        while j > start && arr[j - 1] > key {
            arr[j] = arr[j - 1].clone();
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Insertion sort for a range with comparator
fn insertion_sort_range_by<T: Clone, F>(arr: &mut [T], start: usize, end: usize, compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    for i in (start + 1)..end {
        let key = arr[i].clone();
        let mut j = i;

        while j > start && compare(&arr[j - 1], &key) == std::cmp::Ordering::Greater {
            arr[j] = arr[j - 1].clone();
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Merge two sorted subarrays [left, mid) and [mid, right)
fn merge(arr: &mut [i32], left: usize, mid: usize, right: usize) {
    let left_arr: Vec<i32> = arr[left..mid].to_vec();
    let right_arr: Vec<i32> = arr[mid..right].to_vec();

    let mut i = 0;
    let mut j = 0;
    let mut k = left;

    while i < left_arr.len() && j < right_arr.len() {
        if left_arr[i] <= right_arr[j] {
            arr[k] = left_arr[i];
            i += 1;
        } else {
            arr[k] = right_arr[j];
            j += 1;
        }
        k += 1;
    }

    while i < left_arr.len() {
        arr[k] = left_arr[i];
        i += 1;
        k += 1;
    }

    while j < right_arr.len() {
        arr[k] = right_arr[j];
        j += 1;
        k += 1;
    }
}

/// Generic merge
fn merge_generic<T: Ord + Clone>(arr: &mut [T], left: usize, mid: usize, right: usize) {
    let left_arr: Vec<T> = arr[left..mid].to_vec();
    let right_arr: Vec<T> = arr[mid..right].to_vec();

    let mut i = 0;
    let mut j = 0;
    let mut k = left;

    while i < left_arr.len() && j < right_arr.len() {
        if left_arr[i] <= right_arr[j] {
            arr[k] = left_arr[i].clone();
            i += 1;
        } else {
            arr[k] = right_arr[j].clone();
            j += 1;
        }
        k += 1;
    }

    while i < left_arr.len() {
        arr[k] = left_arr[i].clone();
        i += 1;
        k += 1;
    }

    while j < right_arr.len() {
        arr[k] = right_arr[j].clone();
        j += 1;
        k += 1;
    }
}

/// Merge with comparator
fn merge_by<T: Clone, F>(arr: &mut [T], left: usize, mid: usize, right: usize, compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let left_arr: Vec<T> = arr[left..mid].to_vec();
    let right_arr: Vec<T> = arr[mid..right].to_vec();

    let mut i = 0;
    let mut j = 0;
    let mut k = left;

    while i < left_arr.len() && j < right_arr.len() {
        if compare(&left_arr[i], &right_arr[j]) != std::cmp::Ordering::Greater {
            arr[k] = left_arr[i].clone();
            i += 1;
        } else {
            arr[k] = right_arr[j].clone();
            j += 1;
        }
        k += 1;
    }

    while i < left_arr.len() {
        arr[k] = left_arr[i].clone();
        i += 1;
        k += 1;
    }

    while j < right_arr.len() {
        arr[k] = right_arr[j].clone();
        j += 1;
        k += 1;
    }
}

/// Natural Tim Sort - detects natural runs in the data
/// More efficient when data has existing order
pub fn timsort_natural(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Collect runs (ascending or descending sequences)
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;

    while i < n {
        let run_start = i;

        if i + 1 < n {
            if arr[i] <= arr[i + 1] {
                // Ascending run
                while i + 1 < n && arr[i] <= arr[i + 1] {
                    i += 1;
                }
            } else {
                // Descending run - reverse it
                while i + 1 < n && arr[i] > arr[i + 1] {
                    i += 1;
                }
                // Reverse the descending run to make it ascending
                arr[run_start..=i].reverse();
            }
        }

        i += 1;
        let run_end = i;

        // Extend short runs to minimum run length using insertion sort
        let min_run = calc_min_run(n);
        if run_end - run_start < min_run && run_end < n {
            let extend_to = (run_start + min_run).min(n);
            insertion_sort_range(arr, run_start, extend_to);
            i = extend_to;
        }

        runs.push((run_start, i));
    }

    // Merge runs using a stack-based approach
    merge_runs(arr, &mut runs);
}

/// Merge runs using Tim Sort's merge rules
fn merge_runs(arr: &mut [i32], runs: &mut Vec<(usize, usize)>) {
    while runs.len() > 1 {
        let n = runs.len();

        // Find which runs to merge based on Tim Sort invariants
        let mut merge_idx = n - 2;

        if n >= 3 {
            // Calculate run lengths
            let x_len = runs[n - 3].1 - runs[n - 3].0;
            let y_len = runs[n - 2].1 - runs[n - 2].0;
            let z_len = runs[n - 1].1 - runs[n - 1].0;

            if x_len <= y_len + z_len || y_len <= z_len {
                if x_len < z_len {
                    merge_idx = n - 3;
                }
            } else {
                break;
            }
        }

        // Merge runs[merge_idx] and runs[merge_idx + 1]
        let (left, mid) = runs[merge_idx];
        let (_, right) = runs[merge_idx + 1];

        merge(arr, left, mid, right);

        runs[merge_idx] = (left, right);
        runs.remove(merge_idx + 1);
    }
}

/// Adaptive Tim Sort - adjusts strategy based on data characteristics
pub fn timsort_adaptive<T: Ord + Clone>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // For very small arrays, just use insertion sort
    if n <= MIN_MERGE {
        insertion_sort_range_generic(arr, 0, n);
        return;
    }

    // Detect if array is mostly sorted
    let mut inversions = 0;
    for i in 0..n - 1 {
        if arr[i] > arr[i + 1] {
            inversions += 1;
        }
    }

    // If very few inversions, insertion sort is faster
    if inversions <= n / 8 {
        insertion_sort_range_generic(arr, 0, n);
        return;
    }

    // Otherwise, use standard timsort
    timsort_generic(arr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timsort_basic() {
        let mut arr = [64, 25, 12, 22, 11];
        timsort(&mut arr);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_timsort_already_sorted() {
        let mut arr = [1, 2, 3, 4, 5];
        timsort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_timsort_reversed() {
        let mut arr = [5, 4, 3, 2, 1];
        timsort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_timsort_empty() {
        let mut arr: [i32; 0] = [];
        timsort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_timsort_single() {
        let mut arr = [42];
        timsort(&mut arr);
        assert_eq!(arr, [42]);
    }

    #[test]
    fn test_timsort_duplicates() {
        let mut arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        timsort(&mut arr);
        assert_eq!(arr, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn test_timsort_negative() {
        let mut arr = [-5, 3, -1, 0, 2, -3];
        timsort(&mut arr);
        assert_eq!(arr, [-5, -3, -1, 0, 2, 3]);
    }

    #[test]
    fn test_timsort_generic() {
        let mut arr = ["banana", "apple", "cherry", "date"];
        timsort_generic(&mut arr);
        assert_eq!(arr, ["apple", "banana", "cherry", "date"]);
    }

    #[test]
    fn test_timsort_by_descending() {
        let mut arr = [1, 5, 2, 4, 3];
        timsort_by(&mut arr, |a, b| b.cmp(a));
        assert_eq!(arr, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_timsort_natural() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        timsort_natural(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_timsort_natural_with_runs() {
        // Array with natural ascending runs: [1,2,3], [5,6,7], [4]
        let mut arr = [1, 2, 3, 5, 6, 7, 4];
        timsort_natural(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_timsort_natural_descending_run() {
        // Descending run should be reversed
        let mut arr = [5, 4, 3, 2, 1, 6, 7, 8];
        timsort_natural(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_timsort_adaptive() {
        let mut arr = vec![64, 25, 12, 22, 11, 90, 3, 45, 67];
        timsort_adaptive(&mut arr);
        assert_eq!(arr, vec![3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_timsort_adaptive_nearly_sorted() {
        let mut arr = vec![1, 2, 3, 5, 4, 6, 7, 8, 9, 10];
        timsort_adaptive(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_timsort_stability() {
        // Test that equal elements maintain their relative order
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct Item {
            key: i32,
            value: char,
        }

        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Item {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }

        let mut arr = vec![
            Item { key: 3, value: 'a' },
            Item { key: 1, value: 'b' },
            Item { key: 3, value: 'c' },
            Item { key: 2, value: 'd' },
            Item { key: 3, value: 'e' },
        ];

        timsort_generic(&mut arr);

        // Check stability: items with key=3 should appear in order a, c, e
        let key_3_items: Vec<char> = arr.iter().filter(|x| x.key == 3).map(|x| x.value).collect();
        assert_eq!(key_3_items, vec!['a', 'c', 'e']);
    }

    #[test]
    fn test_large_array() {
        let mut arr: Vec<i32> = (0..1000).rev().collect();
        timsort(&mut arr);

        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_large_array_natural() {
        let mut arr: Vec<i32> = (0..1000).rev().collect();
        timsort_natural(&mut arr);

        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_two_elements() {
        let mut arr = [2, 1];
        timsort(&mut arr);
        assert_eq!(arr, [1, 2]);
    }

    #[test]
    fn test_all_same() {
        let mut arr = [5, 5, 5, 5, 5];
        timsort(&mut arr);
        assert_eq!(arr, [5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_calc_min_run() {
        // calc_min_run divides n until it's below MIN_MERGE, adding remainder bits
        assert_eq!(calc_min_run(64), 16); // 64 -> 32 -> 16
        assert_eq!(calc_min_run(65), 17); // 65 -> 32 (r=1) -> 16 (r=1), 16+1=17
        assert!(calc_min_run(1000) >= 16);
        assert!(calc_min_run(1000) <= 64);
    }

    #[test]
    fn test_random_like_pattern() {
        let mut arr = [23, 67, 12, 89, 45, 78, 34, 56, 90, 11];
        timsort(&mut arr);
        assert_eq!(arr, [11, 12, 23, 34, 45, 56, 67, 78, 89, 90]);
    }
}
