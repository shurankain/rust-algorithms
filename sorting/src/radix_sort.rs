// Radix Sort for non-negative integers (LSD variant).
// Time complexity: O(n * k), where k is the number of digits.
// Stable sorting per digit is achieved via Counting Sort.
pub fn radix_sort(mut arr: Vec<u32>) -> Vec<u32> {
    if arr.is_empty() {
        return arr;
    }

    let max = *arr.iter().max().unwrap();
    let mut exp = 1;

    while max / exp > 0 {
        arr = counting_sort_by_digit(&arr, exp);
        exp *= 10;
    }

    arr
}

fn counting_sort_by_digit(arr: &[u32], exp: u32) -> Vec<u32> {
    let mut output = vec![0; arr.len()];
    let mut count = [0usize; 10];

    // Count occurrences of each digit
    for &num in arr {
        let digit = (num / exp % 10) as usize;
        count[digit] += 1;
    }

    // Cumulative count
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    // Build output array (stable sort)
    for &num in arr.iter().rev() {
        let digit = (num / exp % 10) as usize;
        count[digit] -= 1;
        output[count[digit]] = num;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_basic() {
        let arr = vec![170, 45, 75, 90, 802, 24, 2, 66];
        let sorted = vec![2, 24, 45, 66, 75, 90, 170, 802];
        assert_eq!(radix_sort(arr), sorted);
    }

    #[test]
    fn test_radix_sort_empty() {
        let arr: Vec<u32> = vec![];
        assert_eq!(radix_sort(arr), vec![]);
    }

    #[test]
    fn test_radix_sort_single_element() {
        let arr = vec![42];
        assert_eq!(radix_sort(arr.clone()), arr);
    }

    #[test]
    fn test_radix_sort_duplicates() {
        let arr = vec![5, 3, 5, 2, 8, 3, 5];
        let sorted = vec![2, 3, 3, 5, 5, 5, 8];
        assert_eq!(radix_sort(arr), sorted);
    }

    #[test]
    fn test_radix_sort_sorted() {
        let arr = vec![1, 2, 3, 4, 5];
        assert_eq!(radix_sort(arr.clone()), arr);
    }

    #[test]
    fn test_radix_sort_reverse() {
        let arr = vec![5, 4, 3, 2, 1];
        assert_eq!(radix_sort(arr), vec![1, 2, 3, 4, 5]);
    }
}
