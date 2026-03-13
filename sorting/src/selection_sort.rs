// Selection Sort - Simple comparison-based sorting algorithm
// Time: O(n²) all cases, Space: O(1)
// Not stable, but minimal number of swaps (at most n-1)

/// Selection sort for i32 slices
/// Finds minimum element and places it at the beginning, then repeats
/// Time: O(n²), Space: O(1)
pub fn selection_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    for i in 0..n - 1 {
        // Find minimum element in unsorted portion
        let mut min_idx = i;
        for j in (i + 1)..n {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }

        // Swap minimum with first unsorted element
        if min_idx != i {
            arr.swap(i, min_idx);
        }
    }
}

/// Generic selection sort for any Ord type
pub fn selection_sort_generic<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }

        if min_idx != i {
            arr.swap(i, min_idx);
        }
    }
}

/// Selection sort with custom comparator
pub fn selection_sort_by<T, F>(arr: &mut [T], compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let n = arr.len();
    if n < 2 {
        return;
    }

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if compare(&arr[j], &arr[min_idx]) == std::cmp::Ordering::Less {
                min_idx = j;
            }
        }

        if min_idx != i {
            arr.swap(i, min_idx);
        }
    }
}

/// Selection sort by key extractor
pub fn selection_sort_by_key<T, K, F>(arr: &mut [T], key: F)
where
    K: Ord,
    F: Fn(&T) -> K,
{
    let n = arr.len();
    if n < 2 {
        return;
    }

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if key(&arr[j]) < key(&arr[min_idx]) {
                min_idx = j;
            }
        }

        if min_idx != i {
            arr.swap(i, min_idx);
        }
    }
}

/// Double selection sort - finds both min and max in each pass
/// Slightly more efficient as it reduces passes by half
pub fn double_selection_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    let mut left = 0;
    let mut right = n - 1;

    while left < right {
        let mut min_idx = left;
        let mut max_idx = left;

        // Find both min and max in one pass
        for i in left..=right {
            if arr[i] < arr[min_idx] {
                min_idx = i;
            }
            if arr[i] > arr[max_idx] {
                max_idx = i;
            }
        }

        // Swap minimum to left position
        if min_idx != left {
            arr.swap(left, min_idx);
            // If max was at left, it's now at min_idx
            if max_idx == left {
                max_idx = min_idx;
            }
        }

        // Swap maximum to right position
        if max_idx != right {
            arr.swap(right, max_idx);
        }

        left += 1;
        right -= 1;
    }
}

/// Stable selection sort - maintains relative order of equal elements
/// Uses minimum swaps by finding position and shifting
pub fn stable_selection_sort<T: Ord + Clone>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }

        if min_idx != i {
            // Instead of swapping, shift elements to maintain stability
            let min_val = arr[min_idx].clone();
            // Shift elements right
            for j in (i..min_idx).rev() {
                arr[j + 1] = arr[j].clone();
            }
            arr[i] = min_val;
        }
    }
}

/// Count swaps performed during selection sort
pub fn selection_sort_count_swaps(arr: &mut [i32]) -> usize {
    let n = arr.len();
    if n < 2 {
        return 0;
    }

    let mut swaps = 0;

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }

        if min_idx != i {
            arr.swap(i, min_idx);
            swaps += 1;
        }
    }

    swaps
}

/// Bingo sort - variant of selection sort for arrays with many duplicates
/// Groups equal elements together efficiently
pub fn bingo_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Find the maximum value
    let mut max_val = arr[0];
    for &val in arr.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    let mut start = 0;

    while start < n {
        // Find minimum value in remaining unsorted portion
        let mut bingo = max_val;
        for &val in arr.iter().skip(start) {
            if val < bingo {
                bingo = val;
            }
        }

        // Move all elements equal to bingo to the front
        if bingo == max_val {
            break;
        }

        // Move all elements equal to bingo to sorted position
        let mut i = start;
        while i < n {
            if arr[i] == bingo {
                arr.swap(i, start);
                start += 1;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_sort_basic() {
        let mut arr = [64, 25, 12, 22, 11];
        selection_sort(&mut arr);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_selection_sort_already_sorted() {
        let mut arr = [1, 2, 3, 4, 5];
        selection_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_selection_sort_reversed() {
        let mut arr = [5, 4, 3, 2, 1];
        selection_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_selection_sort_empty() {
        let mut arr: [i32; 0] = [];
        selection_sort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_selection_sort_single() {
        let mut arr = [42];
        selection_sort(&mut arr);
        assert_eq!(arr, [42]);
    }

    #[test]
    fn test_selection_sort_duplicates() {
        let mut arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        selection_sort(&mut arr);
        assert_eq!(arr, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn test_selection_sort_negative() {
        let mut arr = [-5, 3, -1, 0, 2, -3];
        selection_sort(&mut arr);
        assert_eq!(arr, [-5, -3, -1, 0, 2, 3]);
    }

    #[test]
    fn test_selection_sort_generic() {
        let mut arr = ["banana", "apple", "cherry", "date"];
        selection_sort_generic(&mut arr);
        assert_eq!(arr, ["apple", "banana", "cherry", "date"]);
    }

    #[test]
    fn test_selection_sort_generic_integers() {
        let mut arr = [64u64, 25, 12, 22, 11];
        selection_sort_generic(&mut arr);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_selection_sort_by() {
        let mut arr = [1, 5, 2, 4, 3];
        selection_sort_by(&mut arr, |a, b| b.cmp(a)); // Descending
        assert_eq!(arr, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_selection_sort_by_key() {
        let mut arr = [(1, "z"), (3, "a"), (2, "m")];
        selection_sort_by_key(&mut arr, |x| x.0);
        assert_eq!(arr, [(1, "z"), (2, "m"), (3, "a")]);
    }

    #[test]
    fn test_double_selection_sort() {
        let mut arr = [64, 25, 12, 22, 11];
        double_selection_sort(&mut arr);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_double_selection_sort_empty() {
        let mut arr: [i32; 0] = [];
        double_selection_sort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_double_selection_sort_reversed() {
        let mut arr = [5, 4, 3, 2, 1];
        double_selection_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_stable_selection_sort() {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
        struct Item {
            key: i32,
            value: char,
        }

        let mut arr = [
            Item { key: 3, value: 'a' },
            Item { key: 1, value: 'b' },
            Item { key: 3, value: 'c' },
            Item { key: 2, value: 'd' },
        ];

        stable_selection_sort(&mut arr);

        // Should maintain relative order of equal elements
        assert_eq!(arr[0], Item { key: 1, value: 'b' });
        assert_eq!(arr[1], Item { key: 2, value: 'd' });
        assert_eq!(arr[2], Item { key: 3, value: 'a' });
        assert_eq!(arr[3], Item { key: 3, value: 'c' });
    }

    #[test]
    fn test_count_swaps() {
        let mut arr = [5, 4, 3, 2, 1];
        let swaps = selection_sort_count_swaps(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
        // Selection sort does at most n-1 swaps
        assert!(swaps <= 4);
    }

    #[test]
    fn test_count_swaps_sorted() {
        let mut arr = [1, 2, 3, 4, 5];
        let swaps = selection_sort_count_swaps(&mut arr);
        assert_eq!(swaps, 0);
    }

    #[test]
    fn test_bingo_sort() {
        let mut arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        bingo_sort(&mut arr);
        assert_eq!(arr, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn test_bingo_sort_all_same() {
        let mut arr = [5, 5, 5, 5, 5];
        bingo_sort(&mut arr);
        assert_eq!(arr, [5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_bingo_sort_empty() {
        let mut arr: [i32; 0] = [];
        bingo_sort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_all_sorts_consistent() {
        let original = [64, 25, 12, 22, 11, 90, 3];

        let mut arr1 = original;
        let mut arr2 = original;
        let mut arr3 = original;

        selection_sort(&mut arr1);
        double_selection_sort(&mut arr2);
        bingo_sort(&mut arr3);

        assert_eq!(arr1, arr2);
        assert_eq!(arr1, arr3);
    }

    #[test]
    fn test_large_array() {
        let mut arr: Vec<i32> = (0..100).rev().collect();
        selection_sort(&mut arr);

        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(arr, expected);
    }
}
