pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    let mut left = 0;
    let mut right = arr.len() - 1;

    while left <= right {
        let mid = left + (right - left) / 2;
        if arr[mid] == *target {
            return Some(mid);
        }

        if arr[mid] < *target {
            left = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            right = mid - 1;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_numbers() {
        let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let target = 6;

        let result = binary_search(&input, &target);

        assert_eq!(Some(6), result);
    }

    #[test]
    fn test_mixed_numbers() {
        let input = [-3, -1, 0, 2, 7, 11];
        let target = 2;

        let result = binary_search(&input, &target);

        assert_eq!(Some(3), result);
    }

    #[test]
    fn test_not_present() {
        let input = [-3, -1, 0, 2, 7, 11];
        let target = 1;

        let result = binary_search(&input, &target);

        assert_eq!(None, result);
    }

    #[test]
    fn test_single() {
        let input = [5];
        assert_eq!(Some(0), binary_search(&input, &5));
        assert_eq!(None, binary_search(&input, &1));
    }
}
