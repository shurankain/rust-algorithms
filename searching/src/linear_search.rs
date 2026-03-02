pub fn linear_search(data: &[i32], search_val: i32) -> Option<usize> {
    for (i, curr_val) in data.iter().enumerate() {
        if *curr_val == search_val {
            return Option::Some(i);
        }
    }
    Option::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_present() {
        let input = vec![12, 34, 17, 9, 24, 3];
        let result = linear_search(&input, 17).expect("Incorrect result state");

        assert_eq!(2, result);
    }

    #[test]
    fn test_not_present() {
        let input = vec![12, 34, 17, 9, 24, 3];
        let result = linear_search(&input, 11);

        assert_eq!(None, result);
    }

    #[test]
    fn test_empty() {
        let input: Vec<i32> = vec![];
        assert_eq!(None, linear_search(&input, 1));
    }

    #[test]
    fn test_single() {
        let input = vec![42];
        assert_eq!(Some(0), linear_search(&input, 42));
        assert_eq!(None, linear_search(&input, 1));
    }
}
