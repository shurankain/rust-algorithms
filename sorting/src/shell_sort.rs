// Shell Sort - Gap-based insertion sort
// Improves on insertion sort by comparing elements far apart, then reducing the gap
// Time: O(n²) worst case, O(n log n) to O(n^1.5) depending on gap sequence
// Space: O(1)

/// Shell sort using Shell's original gap sequence (n/2, n/4, ..., 1)
/// Time: O(n²) worst case
pub fn shell_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Start with half the array length as gap
    let mut gap = n / 2;

    while gap > 0 {
        // Perform gapped insertion sort
        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            // Shift elements until correct position found
            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }

        gap /= 2;
    }
}

/// Generic shell sort for any Ord type
pub fn shell_sort_generic<T: Ord + Clone>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    let mut gap = n / 2;

    while gap > 0 {
        for i in gap..n {
            let temp = arr[i].clone();
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap].clone();
                j -= gap;
            }
            arr[j] = temp;
        }

        gap /= 2;
    }
}

/// Shell sort using Hibbard's gap sequence (2^k - 1: 1, 3, 7, 15, 31, ...)
/// Time: O(n^1.5) worst case
pub fn shell_sort_hibbard(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Generate Hibbard sequence: 2^k - 1
    let mut gaps = Vec::new();
    let mut k = 1;
    loop {
        let gap = (1 << k) - 1; // 2^k - 1
        if gap >= n {
            break;
        }
        gaps.push(gap);
        k += 1;
    }

    // Process gaps in reverse order (largest to smallest)
    for &gap in gaps.iter().rev() {
        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Shell sort using Sedgewick's gap sequence
/// Time: O(n^(4/3)) worst case
pub fn shell_sort_sedgewick(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Generate Sedgewick sequence: 4^k + 3*2^(k-1) + 1 for k >= 1, and 1
    let mut gaps = vec![1];
    let mut k = 1u32;
    loop {
        // Sedgewick's formula: 4^k + 3*2^(k-1) + 1
        let gap = 4usize.pow(k) + 3 * 2usize.pow(k - 1) + 1;
        if gap >= n {
            break;
        }
        gaps.push(gap);
        k += 1;
    }

    // Process gaps in reverse order
    for &gap in gaps.iter().rev() {
        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Shell sort using Knuth's gap sequence (3^k - 1) / 2: 1, 4, 13, 40, 121, ...
/// Time: O(n^1.5) worst case
pub fn shell_sort_knuth(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Find largest gap in Knuth sequence less than n/3
    let mut gap = 1;
    while gap < n / 3 {
        gap = 3 * gap + 1; // Knuth sequence: h = 3h + 1
    }

    while gap >= 1 {
        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }

        gap /= 3; // Reverse: h = (h - 1) / 3
    }
}

/// Shell sort using Tokuda's gap sequence (empirically good)
/// ceil((9 * (9/4)^k - 4) / 5): 1, 4, 9, 20, 46, 103, ...
pub fn shell_sort_tokuda(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Precomputed Tokuda sequence
    let tokuda_gaps = [
        1, 4, 9, 20, 46, 103, 233, 525, 1182, 2660, 5985, 13467, 30301, 68178, 153401, 345152,
        776591, 1747331, 3931496, 8845866, 19903198,
    ];

    // Find starting index
    let mut start_idx = 0;
    for (i, &gap) in tokuda_gaps.iter().enumerate() {
        if gap >= n {
            break;
        }
        start_idx = i;
    }

    // Process gaps in reverse
    for &gap in tokuda_gaps[..=start_idx].iter().rev() {
        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Shell sort using Ciura's gap sequence (empirically optimal for small/medium arrays)
/// 1, 4, 10, 23, 57, 132, 301, 701, ...
pub fn shell_sort_ciura(arr: &mut [i32]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // Ciura's empirically derived sequence
    let ciura_gaps = [701, 301, 132, 57, 23, 10, 4, 1];

    for &gap in &ciura_gaps {
        if gap >= n {
            continue;
        }

        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Shell sort with custom gap sequence
pub fn shell_sort_custom(arr: &mut [i32], gaps: &[usize]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    for &gap in gaps {
        if gap == 0 || gap >= n {
            continue;
        }

        for i in gap..n {
            let temp = arr[i];
            let mut j = i;

            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Shell sort with comparator
pub fn shell_sort_by<T: Clone, F>(arr: &mut [T], compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let n = arr.len();
    if n < 2 {
        return;
    }

    let mut gap = n / 2;

    while gap > 0 {
        for i in gap..n {
            let temp = arr[i].clone();
            let mut j = i;

            while j >= gap && compare(&arr[j - gap], &temp) == std::cmp::Ordering::Greater {
                arr[j] = arr[j - gap].clone();
                j -= gap;
            }
            arr[j] = temp;
        }

        gap /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_sort_basic() {
        let mut arr = [64, 25, 12, 22, 11];
        shell_sort(&mut arr);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_shell_sort_already_sorted() {
        let mut arr = [1, 2, 3, 4, 5];
        shell_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_shell_sort_reversed() {
        let mut arr = [5, 4, 3, 2, 1];
        shell_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_shell_sort_empty() {
        let mut arr: [i32; 0] = [];
        shell_sort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_shell_sort_single() {
        let mut arr = [42];
        shell_sort(&mut arr);
        assert_eq!(arr, [42]);
    }

    #[test]
    fn test_shell_sort_duplicates() {
        let mut arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        shell_sort(&mut arr);
        assert_eq!(arr, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn test_shell_sort_negative() {
        let mut arr = [-5, 3, -1, 0, 2, -3];
        shell_sort(&mut arr);
        assert_eq!(arr, [-5, -3, -1, 0, 2, 3]);
    }

    #[test]
    fn test_shell_sort_generic() {
        let mut arr = ["banana", "apple", "cherry", "date"];
        shell_sort_generic(&mut arr);
        assert_eq!(arr, ["apple", "banana", "cherry", "date"]);
    }

    #[test]
    fn test_shell_sort_hibbard() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        shell_sort_hibbard(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_shell_sort_sedgewick() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        shell_sort_sedgewick(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_shell_sort_knuth() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        shell_sort_knuth(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_shell_sort_tokuda() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        shell_sort_tokuda(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_shell_sort_ciura() {
        let mut arr = [64, 25, 12, 22, 11, 90, 3, 45, 67];
        shell_sort_ciura(&mut arr);
        assert_eq!(arr, [3, 11, 12, 22, 25, 45, 64, 67, 90]);
    }

    #[test]
    fn test_shell_sort_custom_gaps() {
        let mut arr = [64, 25, 12, 22, 11];
        shell_sort_custom(&mut arr, &[5, 3, 1]);
        assert_eq!(arr, [11, 12, 22, 25, 64]);
    }

    #[test]
    fn test_shell_sort_by_descending() {
        let mut arr = [1, 5, 2, 4, 3];
        shell_sort_by(&mut arr, |a, b| b.cmp(a));
        assert_eq!(arr, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_all_variants_consistent() {
        let original = [64, 25, 12, 22, 11, 90, 3, 45, 67, 78, 1, 99];

        let mut arr1 = original;
        let mut arr2 = original;
        let mut arr3 = original;
        let mut arr4 = original;
        let mut arr5 = original;
        let mut arr6 = original;

        shell_sort(&mut arr1);
        shell_sort_hibbard(&mut arr2);
        shell_sort_sedgewick(&mut arr3);
        shell_sort_knuth(&mut arr4);
        shell_sort_tokuda(&mut arr5);
        shell_sort_ciura(&mut arr6);

        assert_eq!(arr1, arr2);
        assert_eq!(arr1, arr3);
        assert_eq!(arr1, arr4);
        assert_eq!(arr1, arr5);
        assert_eq!(arr1, arr6);
    }

    #[test]
    fn test_large_array() {
        let mut arr: Vec<i32> = (0..1000).rev().collect();
        shell_sort(&mut arr);

        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_large_array_all_variants() {
        let original: Vec<i32> = (0..500).rev().collect();
        let expected: Vec<i32> = (0..500).collect();

        let mut arr1 = original.clone();
        let mut arr2 = original.clone();
        let mut arr3 = original.clone();

        shell_sort(&mut arr1);
        shell_sort_knuth(&mut arr2);
        shell_sort_ciura(&mut arr3);

        assert_eq!(arr1, expected);
        assert_eq!(arr2, expected);
        assert_eq!(arr3, expected);
    }

    #[test]
    fn test_two_elements() {
        let mut arr = [2, 1];
        shell_sort(&mut arr);
        assert_eq!(arr, [1, 2]);
    }

    #[test]
    fn test_all_same() {
        let mut arr = [5, 5, 5, 5, 5];
        shell_sort(&mut arr);
        assert_eq!(arr, [5, 5, 5, 5, 5]);
    }
}
