pub fn bubble_sort(arr: &mut [i32]) {
    if arr.len() < 2 {
        return;
    }

    let mut last_ind = arr.len() - 1;
    for _ in 0..arr.len() - 1 {
        let mut was_swapped = false;
        for j in 0..last_ind {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                was_swapped = true;
            }
        }
        if !was_swapped {
            break;
        }
        last_ind -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_sort_basic() {
        let mut input = [1, 7, 3, 55, 12, 14, 2];
        bubble_sort(&mut input);
        println!("{:#?}", input);
        assert_eq!([1, 2, 3, 7, 12, 14, 55], input);
    }

    #[test]
    fn already_sorted() {
        let mut input = [1, 2, 3, 4];
        bubble_sort(&mut input);
        assert_eq!([1, 2, 3, 4], input);
    }

    #[test]
    fn reversed() {
        let mut input = [9, 8, 7, 6];
        bubble_sort(&mut input);
        assert_eq!([6, 7, 8, 9], input);
    }

    #[test]
    fn empty() {
        let mut input: [i32; 0] = [];
        bubble_sort(&mut input);
        assert!(input.is_empty());
    }
}
