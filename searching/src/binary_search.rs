// `target` is only used for comparison and not returned, so it doesn't influence the output lifetime
pub fn binary_search<'a, T>(arr: &'a[T], target: &T) -> Result<&'a T, &'static str>
where T: Ord {
    if arr.is_empty() {
        return Result::Err("Array is empty");
    }

    let mut left = 0;
    let mut right = arr.len() - 1;

    while left <= right {
        let mid = left + (right - left) / 2;
        if arr[mid] == *target {
            return Result::Ok(&arr[mid]);
        }

        if arr[mid] < *target {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    Result::Err("Not found")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_positive_numbs() {
        let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let target = 6;

        let result = binary_search(&input, &target).unwrap();

        assert_eq!(6, *result);
    }

    #[test]
    fn test_mixed_numbers() {
        let input = [-3, -1, 0, 2, 7, 11];
        let target = 2;

        let result = binary_search(&input, &target).unwrap();

        assert_eq!(3, *result);
    }

    #[test]
    fn test_not_present() {
        let input = [-3, -1, 0, 2, 7, 11];
        let target = 1;

        let result = binary_search(&input, &target).unwrap_err();

        assert_eq!("Not found", result);
    }
}
