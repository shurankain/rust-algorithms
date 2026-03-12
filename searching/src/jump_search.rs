// Jump Search Algorithm
// A searching algorithm for sorted arrays that jumps ahead by fixed steps
// then performs linear search in the block where the element may exist
// Time: O(√n), Space: O(1)

/// Jump search for finding an element in a sorted array
/// Returns the index of the target if found, None otherwise
/// Time: O(√n), Space: O(1)
pub fn jump_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    // Optimal jump size is √n
    let jump = (n as f64).sqrt() as usize;
    let jump = jump.max(1); // Ensure at least 1

    // Find the block where element may be present
    let mut prev = 0;
    let mut curr = jump;

    // Jump until we find a block where arr[curr] >= target
    while curr < n && arr[curr] < *target {
        prev = curr;
        curr += jump;
    }

    // Linear search within the block [prev, min(curr, n-1)]
    let end = curr.min(n - 1);
    for (i, item) in arr.iter().enumerate().take(end + 1).skip(prev) {
        if item == target {
            return Some(i);
        }
        if item > target {
            break;
        }
    }

    None
}

/// Jump search that returns the first occurrence of the target
pub fn jump_search_first<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    let jump = ((n as f64).sqrt() as usize).max(1);

    let mut prev = 0;
    let mut curr = jump;

    while curr < n && arr[curr] < *target {
        prev = curr;
        curr += jump;
    }

    // Linear search to find first occurrence
    let end = curr.min(n - 1);
    for (i, item) in arr.iter().enumerate().take(end + 1).skip(prev) {
        if item == target {
            return Some(i);
        }
        if item > target {
            break;
        }
    }

    None
}

/// Jump search that returns the last occurrence of the target
pub fn jump_search_last<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    let jump = ((n as f64).sqrt() as usize).max(1);

    let mut prev = 0;
    let mut curr = jump;

    // Jump until we pass the target
    while curr < n && arr[curr] <= *target {
        prev = curr;
        curr += jump;
    }

    // Linear search backwards from curr to find last occurrence
    let end = curr.min(n - 1);
    let mut result = None;

    for (i, item) in arr.iter().enumerate().take(end + 1).skip(prev) {
        if item == target {
            result = Some(i);
        }
        if item > target {
            break;
        }
    }

    result
}

/// Jump search with custom jump size
pub fn jump_search_with_step<T: Ord>(arr: &[T], target: &T, step: usize) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    let jump = step.max(1);

    let mut prev = 0;
    let mut curr = jump;

    while curr < n && arr[curr] < *target {
        prev = curr;
        curr += jump;
    }

    let end = curr.min(n - 1);
    for (i, item) in arr.iter().enumerate().take(end + 1).skip(prev) {
        if item == target {
            return Some(i);
        }
        if item > target {
            break;
        }
    }

    None
}

/// Exponential search - starts with jump of 1 and doubles until we pass target
/// Then binary search in the found range
/// Time: O(log n), Space: O(1)
pub fn exponential_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    // Check first element
    if arr[0] == *target {
        return Some(0);
    }

    // Find range by doubling
    let mut bound = 1;
    while bound < n && arr[bound] < *target {
        bound *= 2;
    }

    // Binary search in range [bound/2, min(bound, n-1)]
    let left = bound / 2;
    let right = bound.min(n - 1);

    binary_search_range(arr, target, left, right)
}

/// Binary search within a specific range
fn binary_search_range<T: Ord>(arr: &[T], target: &T, left: usize, right: usize) -> Option<usize> {
    let mut lo = left;
    let mut hi = right;

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;

        if arr[mid] == *target {
            return Some(mid);
        }

        if arr[mid] < *target {
            lo = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
    }

    None
}

/// Interpolation search - estimates position based on value distribution
/// Best for uniformly distributed sorted arrays
/// Time: O(log log n) average for uniform distribution, O(n) worst case
pub fn interpolation_search(arr: &[i64], target: i64) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    let mut lo = 0;
    let mut hi = n - 1;

    while lo <= hi && target >= arr[lo] && target <= arr[hi] {
        if lo == hi {
            return if arr[lo] == target { Some(lo) } else { None };
        }

        // Interpolation formula: estimate position
        let range = arr[hi] - arr[lo];
        if range == 0 {
            return if arr[lo] == target { Some(lo) } else { None };
        }

        let pos = lo + (((target - arr[lo]) as usize * (hi - lo)) / range as usize);
        let pos = pos.min(hi);

        if arr[pos] == target {
            return Some(pos);
        }

        if arr[pos] < target {
            lo = pos + 1;
        } else {
            if pos == 0 {
                break;
            }
            hi = pos - 1;
        }
    }

    None
}

/// Fibonacci search - uses Fibonacci numbers for division points
/// Time: O(log n), Space: O(1)
pub fn fibonacci_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    // Find smallest Fibonacci number >= n
    let mut fib_m2 = 0; // (m-2)th Fibonacci
    let mut fib_m1 = 1; // (m-1)th Fibonacci
    let mut fib_m = fib_m2 + fib_m1; // mth Fibonacci

    while fib_m < n {
        fib_m2 = fib_m1;
        fib_m1 = fib_m;
        fib_m = fib_m2 + fib_m1;
    }

    // Offset marks the eliminated range from front
    let mut offset = 0;

    while fib_m > 1 {
        // Check if fib_m2 is a valid index
        let i = (offset + fib_m2).min(n - 1);

        if arr[i] < *target {
            // Move one Fibonacci down, reset offset
            fib_m = fib_m1;
            fib_m1 = fib_m2;
            fib_m2 = fib_m - fib_m1;
            offset = i;
        } else if arr[i] > *target {
            // Move two Fibonacci down
            fib_m = fib_m2;
            fib_m1 -= fib_m2;
            fib_m2 = fib_m - fib_m1;
        } else {
            return Some(i);
        }
    }

    // Compare last element
    if fib_m1 == 1 && offset + 1 < n && arr[offset + 1] == *target {
        return Some(offset + 1);
    }

    if offset < n && arr[offset] == *target {
        return Some(offset);
    }

    None
}

/// Ternary search - divides array into three parts
/// Time: O(log₃ n), Space: O(1)
pub fn ternary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let n = arr.len();
    if n == 0 {
        return None;
    }

    let mut lo = 0;
    let mut hi = n - 1;

    while lo <= hi {
        let mid1 = lo + (hi - lo) / 3;
        let mid2 = hi - (hi - lo) / 3;

        if arr[mid1] == *target {
            return Some(mid1);
        }
        if arr[mid2] == *target {
            return Some(mid2);
        }

        if *target < arr[mid1] {
            if mid1 == 0 {
                break;
            }
            hi = mid1 - 1;
        } else if *target > arr[mid2] {
            lo = mid2 + 1;
        } else {
            lo = mid1 + 1;
            hi = mid2 - 1;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_search_basic() {
        let arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(jump_search(&arr, &6), Some(6));
        assert_eq!(jump_search(&arr, &0), Some(0));
        assert_eq!(jump_search(&arr, &10), Some(10));
    }

    #[test]
    fn test_jump_search_not_found() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(jump_search(&arr, &2), None);
        assert_eq!(jump_search(&arr, &0), None);
        assert_eq!(jump_search(&arr, &10), None);
    }

    #[test]
    fn test_jump_search_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(jump_search(&arr, &5), None);
    }

    #[test]
    fn test_jump_search_single() {
        let arr = [5];
        assert_eq!(jump_search(&arr, &5), Some(0));
        assert_eq!(jump_search(&arr, &3), None);
    }

    #[test]
    fn test_jump_search_duplicates() {
        let arr = [1, 2, 2, 2, 3, 4, 4, 5];
        let result = jump_search(&arr, &2);
        assert!(result.is_some());
        assert_eq!(arr[result.unwrap()], 2);
    }

    #[test]
    fn test_jump_search_first() {
        let arr = [1, 2, 2, 2, 3, 4, 4, 5];
        assert_eq!(jump_search_first(&arr, &2), Some(1));
        assert_eq!(jump_search_first(&arr, &4), Some(5));
    }

    #[test]
    fn test_jump_search_last() {
        let arr = [1, 2, 2, 2, 3, 4, 4, 5];
        assert_eq!(jump_search_last(&arr, &2), Some(3));
        assert_eq!(jump_search_last(&arr, &4), Some(6));
    }

    #[test]
    fn test_jump_search_custom_step() {
        let arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(jump_search_with_step(&arr, &7, 2), Some(7));
        assert_eq!(jump_search_with_step(&arr, &7, 5), Some(7));
    }

    #[test]
    fn test_exponential_search_basic() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(exponential_search(&arr, &5), Some(4));
        assert_eq!(exponential_search(&arr, &1), Some(0));
        assert_eq!(exponential_search(&arr, &10), Some(9));
    }

    #[test]
    fn test_exponential_search_not_found() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(exponential_search(&arr, &2), None);
        assert_eq!(exponential_search(&arr, &0), None);
    }

    #[test]
    fn test_exponential_search_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(exponential_search(&arr, &5), None);
    }

    #[test]
    fn test_interpolation_search_basic() {
        let arr: Vec<i64> = (0..100).collect();
        assert_eq!(interpolation_search(&arr, 50), Some(50));
        assert_eq!(interpolation_search(&arr, 0), Some(0));
        assert_eq!(interpolation_search(&arr, 99), Some(99));
    }

    #[test]
    fn test_interpolation_search_not_found() {
        let arr = [10i64, 20, 30, 40, 50];
        assert_eq!(interpolation_search(&arr, 25), None);
        assert_eq!(interpolation_search(&arr, 5), None);
        assert_eq!(interpolation_search(&arr, 55), None);
    }

    #[test]
    fn test_interpolation_search_empty() {
        let arr: [i64; 0] = [];
        assert_eq!(interpolation_search(&arr, 5), None);
    }

    #[test]
    fn test_interpolation_search_uniform() {
        // Interpolation search is optimal for uniform distribution
        let arr: Vec<i64> = (0..1000).map(|x| x * 10).collect();
        assert_eq!(interpolation_search(&arr, 5000), Some(500));
    }

    #[test]
    fn test_fibonacci_search_basic() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(fibonacci_search(&arr, &5), Some(4));
        assert_eq!(fibonacci_search(&arr, &1), Some(0));
        assert_eq!(fibonacci_search(&arr, &10), Some(9));
    }

    #[test]
    fn test_fibonacci_search_not_found() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(fibonacci_search(&arr, &2), None);
        assert_eq!(fibonacci_search(&arr, &0), None);
    }

    #[test]
    fn test_fibonacci_search_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(fibonacci_search(&arr, &5), None);
    }

    #[test]
    fn test_fibonacci_search_single() {
        let arr = [42];
        assert_eq!(fibonacci_search(&arr, &42), Some(0));
        assert_eq!(fibonacci_search(&arr, &1), None);
    }

    #[test]
    fn test_ternary_search_basic() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(ternary_search(&arr, &5), Some(4));
        assert_eq!(ternary_search(&arr, &1), Some(0));
        assert_eq!(ternary_search(&arr, &10), Some(9));
    }

    #[test]
    fn test_ternary_search_not_found() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(ternary_search(&arr, &2), None);
        assert_eq!(ternary_search(&arr, &0), None);
    }

    #[test]
    fn test_ternary_search_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(ternary_search(&arr, &5), None);
    }

    #[test]
    fn test_all_algorithms_consistent() {
        let arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

        for &target in &[1, 5, 11, 19] {
            let jump = jump_search(&arr, &target);
            let exp = exponential_search(&arr, &target);
            let fib = fibonacci_search(&arr, &target);
            let tern = ternary_search(&arr, &target);

            assert_eq!(jump, exp, "Mismatch for target {}", target);
            assert_eq!(jump, fib, "Mismatch for target {}", target);
            assert_eq!(jump, tern, "Mismatch for target {}", target);
        }
    }

    #[test]
    fn test_all_algorithms_not_found() {
        let arr = [2, 4, 6, 8, 10];

        for &target in &[1, 3, 5, 11] {
            assert_eq!(jump_search(&arr, &target), None);
            assert_eq!(exponential_search(&arr, &target), None);
            assert_eq!(fibonacci_search(&arr, &target), None);
            assert_eq!(ternary_search(&arr, &target), None);
        }
    }

    #[test]
    fn test_large_array() {
        let arr: Vec<i32> = (0..10000).collect();

        assert_eq!(jump_search(&arr, &5000), Some(5000));
        assert_eq!(exponential_search(&arr, &5000), Some(5000));
        assert_eq!(fibonacci_search(&arr, &5000), Some(5000));
        assert_eq!(ternary_search(&arr, &5000), Some(5000));
    }

    #[test]
    fn test_strings() {
        let arr = ["apple", "banana", "cherry", "date", "elderberry"];

        assert_eq!(jump_search(&arr, &"cherry"), Some(2));
        assert_eq!(exponential_search(&arr, &"banana"), Some(1));
        assert_eq!(fibonacci_search(&arr, &"date"), Some(3));
        assert_eq!(ternary_search(&arr, &"apple"), Some(0));
    }

    #[test]
    fn test_negative_numbers() {
        let arr = [-10, -5, 0, 5, 10];

        assert_eq!(jump_search(&arr, &-5), Some(1));
        assert_eq!(jump_search(&arr, &0), Some(2));
        assert_eq!(exponential_search(&arr, &-10), Some(0));
        assert_eq!(fibonacci_search(&arr, &10), Some(4));
    }
}
