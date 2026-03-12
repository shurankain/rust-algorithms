// Longest Increasing Subsequence (LIS) - Classic dynamic programming problem

/// Find the length of the longest strictly increasing subsequence
/// Time: O(n²), Space: O(n)
pub fn lis_length(arr: &[i32]) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let n = arr.len();
    // dp[i] = length of LIS ending at index i
    let mut dp = vec![1; n];

    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }

    *dp.iter().max().unwrap()
}

/// Find the length of LIS using binary search
/// Time: O(n log n), Space: O(n)
pub fn lis_length_binary_search(arr: &[i32]) -> usize {
    if arr.is_empty() {
        return 0;
    }

    // tails[i] = smallest ending element of all increasing subsequences of length i+1
    let mut tails: Vec<i32> = Vec::with_capacity(arr.len());

    for &num in arr {
        // Binary search for position to insert/replace
        match tails.binary_search(&num) {
            Ok(_) => {} // Already exists, skip for strictly increasing
            Err(pos) => {
                if pos == tails.len() {
                    tails.push(num);
                } else {
                    tails[pos] = num;
                }
            }
        }
    }

    tails.len()
}

/// Find the actual longest increasing subsequence
/// Time: O(n²), Space: O(n)
pub fn lis(arr: &[i32]) -> Vec<i32> {
    if arr.is_empty() {
        return Vec::new();
    }

    let n = arr.len();
    let mut dp = vec![1; n];
    let mut parent = vec![None; n]; // Track predecessors

    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }

    // Find the index with maximum LIS length
    let mut max_idx = 0;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    // Reconstruct the subsequence
    let mut result = Vec::new();
    let mut idx = Some(max_idx);
    while let Some(i) = idx {
        result.push(arr[i]);
        idx = parent[i];
    }

    result.reverse();
    result
}

/// Find LIS using binary search with reconstruction
/// Time: O(n log n), Space: O(n)
pub fn lis_binary_search(arr: &[i32]) -> Vec<i32> {
    if arr.is_empty() {
        return Vec::new();
    }

    let n = arr.len();
    // tails[i] = index of smallest ending element of LIS of length i+1
    let mut tails_idx: Vec<usize> = Vec::with_capacity(n);
    let mut parent: Vec<Option<usize>> = vec![None; n];

    for i in 0..n {
        // Binary search for position
        let pos = tails_idx
            .binary_search_by(|&j| arr[j].cmp(&arr[i]))
            .unwrap_or_else(|x| x);

        if pos == tails_idx.len() {
            tails_idx.push(i);
        } else {
            tails_idx[pos] = i;
        }

        if pos > 0 {
            parent[i] = Some(tails_idx[pos - 1]);
        }
    }

    // Reconstruct
    let mut result = Vec::new();
    let mut idx = tails_idx.last().copied();
    while let Some(i) = idx {
        result.push(arr[i]);
        idx = parent[i];
    }

    result.reverse();
    result
}

/// Find the longest non-decreasing subsequence (allows equal elements)
pub fn longest_non_decreasing_subsequence(arr: &[i32]) -> Vec<i32> {
    if arr.is_empty() {
        return Vec::new();
    }

    let n = arr.len();
    let mut dp = vec![1; n];
    let mut parent = vec![None; n];

    for i in 1..n {
        for j in 0..i {
            if arr[j] <= arr[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }

    let mut max_idx = 0;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut result = Vec::new();
    let mut idx = Some(max_idx);
    while let Some(i) = idx {
        result.push(arr[i]);
        idx = parent[i];
    }

    result.reverse();
    result
}

/// Find the longest decreasing subsequence
pub fn longest_decreasing_subsequence(arr: &[i32]) -> Vec<i32> {
    if arr.is_empty() {
        return Vec::new();
    }

    // Reverse the comparison logic
    let n = arr.len();
    let mut dp = vec![1; n];
    let mut parent = vec![None; n];

    for i in 1..n {
        for j in 0..i {
            if arr[j] > arr[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }

    let mut max_idx = 0;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut result = Vec::new();
    let mut idx = Some(max_idx);
    while let Some(i) = idx {
        result.push(arr[i]);
        idx = parent[i];
    }

    result.reverse();
    result
}

/// Count the number of LIS (distinct longest increasing subsequences)
pub fn count_lis(arr: &[i32]) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let n = arr.len();
    let mut length = vec![1; n]; // LIS length ending at i
    let mut count = vec![1; n]; // Number of LIS ending at i

    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] {
                if length[j] + 1 > length[i] {
                    length[i] = length[j] + 1;
                    count[i] = count[j];
                } else if length[j] + 1 == length[i] {
                    count[i] += count[j];
                }
            }
        }
    }

    let max_len = *length.iter().max().unwrap();
    length
        .iter()
        .zip(count.iter())
        .filter(|(len, _)| **len == max_len)
        .map(|(_, cnt)| *cnt)
        .sum()
}

/// Find LIS for generic types that implement Ord
pub fn lis_generic<T: Ord + Clone>(arr: &[T]) -> Vec<T> {
    if arr.is_empty() {
        return Vec::new();
    }

    let n = arr.len();
    let mut dp = vec![1; n];
    let mut parent = vec![None; n];

    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }

    let mut max_idx = 0;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut result = Vec::new();
    let mut idx = Some(max_idx);
    while let Some(i) = idx {
        result.push(arr[i].clone());
        idx = parent[i];
    }

    result.reverse();
    result
}

/// Longest Bitonic Subsequence (increases then decreases)
pub fn longest_bitonic_subsequence(arr: &[i32]) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let n = arr.len();

    // LIS ending at each index (left to right)
    let mut lis = vec![1; n];
    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] {
                lis[i] = lis[i].max(lis[j] + 1);
            }
        }
    }

    // LDS starting at each index (right to left, which is LIS from right)
    let mut lds = vec![1; n];
    for i in (0..n - 1).rev() {
        for j in (i + 1..n).rev() {
            if arr[j] < arr[i] {
                lds[i] = lds[i].max(lds[j] + 1);
            }
        }
    }

    // Maximum bitonic length (subtract 1 because peak is counted twice)
    let mut max_bitonic = 0;
    for i in 0..n {
        max_bitonic = max_bitonic.max(lis[i] + lds[i] - 1);
    }

    max_bitonic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lis_length_basic() {
        assert_eq!(lis_length(&[10, 9, 2, 5, 3, 7, 101, 18]), 4);
        assert_eq!(lis_length(&[0, 1, 0, 3, 2, 3]), 4);
        assert_eq!(lis_length(&[7, 7, 7, 7, 7, 7, 7]), 1);
    }

    #[test]
    fn test_lis_length_empty() {
        assert_eq!(lis_length(&[]), 0);
    }

    #[test]
    fn test_lis_length_single() {
        assert_eq!(lis_length(&[5]), 1);
    }

    #[test]
    fn test_lis_length_sorted() {
        assert_eq!(lis_length(&[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn test_lis_length_reverse_sorted() {
        assert_eq!(lis_length(&[5, 4, 3, 2, 1]), 1);
    }

    #[test]
    fn test_lis_length_binary_search() {
        assert_eq!(lis_length_binary_search(&[10, 9, 2, 5, 3, 7, 101, 18]), 4);
        assert_eq!(lis_length_binary_search(&[0, 1, 0, 3, 2, 3]), 4);
        assert_eq!(lis_length_binary_search(&[7, 7, 7, 7, 7, 7, 7]), 1);
        assert_eq!(lis_length_binary_search(&[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn test_lis_reconstruction() {
        let result = lis(&[10, 9, 2, 5, 3, 7, 101, 18]);
        assert_eq!(result.len(), 4);
        // Verify it's strictly increasing
        for i in 1..result.len() {
            assert!(result[i - 1] < result[i]);
        }
    }

    #[test]
    fn test_lis_binary_search_reconstruction() {
        let result = lis_binary_search(&[10, 9, 2, 5, 3, 7, 101, 18]);
        assert_eq!(result.len(), 4);
        for i in 1..result.len() {
            assert!(result[i - 1] < result[i]);
        }
    }

    #[test]
    fn test_lis_empty() {
        assert_eq!(lis(&[]), Vec::<i32>::new());
        assert_eq!(lis_binary_search(&[]), Vec::<i32>::new());
    }

    #[test]
    fn test_longest_non_decreasing() {
        let result = longest_non_decreasing_subsequence(&[1, 2, 2, 3, 3, 3, 4]);
        assert_eq!(result.len(), 7); // All elements can be included

        let result2 = longest_non_decreasing_subsequence(&[5, 4, 3, 2, 1]);
        assert_eq!(result2.len(), 1);
    }

    #[test]
    fn test_longest_decreasing() {
        let result = longest_decreasing_subsequence(&[5, 4, 3, 2, 1]);
        assert_eq!(result.len(), 5);
        assert_eq!(result, vec![5, 4, 3, 2, 1]);

        let result2 = longest_decreasing_subsequence(&[1, 2, 3, 4, 5]);
        assert_eq!(result2.len(), 1);
    }

    #[test]
    fn test_count_lis() {
        assert_eq!(count_lis(&[1, 3, 5, 4, 7]), 2); // [1,3,5,7] and [1,3,4,7]
        assert_eq!(count_lis(&[2, 2, 2, 2, 2]), 5); // Each element alone
    }

    #[test]
    fn test_lis_generic_strings() {
        let strings = vec!["apple", "banana", "cherry", "date"];
        let result = lis_generic(&strings);
        assert_eq!(result.len(), 4); // Already sorted
    }

    #[test]
    fn test_lis_generic_chars() {
        let chars = vec!['b', 'a', 'd', 'c', 'e'];
        let result = lis_generic(&chars);
        assert_eq!(result.len(), 3); // a, c, e or a, d, e or b, d, e or b, c, e
    }

    #[test]
    fn test_longest_bitonic() {
        assert_eq!(longest_bitonic_subsequence(&[1, 11, 2, 10, 4, 5, 2, 1]), 6);
        // Example: 1, 2, 4, 5, 2, 1 (increases then decreases)

        assert_eq!(longest_bitonic_subsequence(&[12, 11, 40, 5, 3, 1]), 5);
        // Example: 12, 40, 5, 3, 1 or 11, 40, 5, 3, 1
    }

    #[test]
    fn test_longest_bitonic_monotonic() {
        // Pure increasing is valid bitonic
        assert_eq!(longest_bitonic_subsequence(&[1, 2, 3, 4, 5]), 5);
        // Pure decreasing is valid bitonic
        assert_eq!(longest_bitonic_subsequence(&[5, 4, 3, 2, 1]), 5);
    }

    #[test]
    fn test_consistency_between_methods() {
        let arr = vec![3, 10, 2, 1, 20];
        assert_eq!(lis_length(&arr), lis_length_binary_search(&arr));
        assert_eq!(lis(&arr).len(), lis_binary_search(&arr).len());
    }
}
