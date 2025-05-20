pub fn insertion_sort(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }

    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j] < arr[j - 1] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsorted_array() {
        let mut input = [1, 7, 3, 55, 12, 14, 2];
        insertion_sort(&mut input);
        assert_eq!(input, [1, 2, 3, 7, 12, 14, 55]);
    }

    #[test]
    fn test_empty_array() {
        let mut input: [i32; 0] = [];
        insertion_sort(&mut input);
        assert_eq!(input, []);
    }

    #[test]
    fn test_single_element() {
        let mut input = [42];
        insertion_sort(&mut input);
        assert_eq!(input, [42]);
    }

    #[test]
    fn test_sorted_array() {
        let mut input = [1, 2, 3, 4, 5];
        insertion_sort(&mut input);
        assert_eq!(input, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted_array() {
        let mut input = [5, 4, 3, 2, 1];
        insertion_sort(&mut input);
        assert_eq!(input, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_all_equal_elements() {
        let mut input = [7, 7, 7, 7, 7];
        insertion_sort(&mut input);
        assert_eq!(input, [7, 7, 7, 7, 7]);
    }

    #[test]
    fn test_with_negative_numbers() {
        let mut input = [3, -1, 4, -5, 0];
        insertion_sort(&mut input);
        assert_eq!(input, [-5, -1, 0, 3, 4]);
    }
}
