// Naive implementation
// For "banana": [5, 3, 1, 0, 4, 2]
pub fn build_suffix_array(s: &str) -> Vec<usize> {
    let n = s.len();
    let mut suffixes: Vec<(usize, &str)> = (0..n).map(|i| (i, &s[i..])).collect();

    suffixes.sort_by(|a, b| a.1.cmp(b.1));

    suffixes.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_array_basic() {
        let s = "banana";
        let expected = vec![5, 3, 1, 0, 4, 2];
        assert_eq!(build_suffix_array(s), expected);
    }

    #[test]
    fn test_suffix_array_single_char() {
        let s = "a";
        let expected = vec![0];
        assert_eq!(build_suffix_array(s), expected);
    }

    #[test]
    fn test_suffix_array_repeated_chars() {
        let s = "aaaa";
        let expected = vec![3, 2, 1, 0];
        assert_eq!(build_suffix_array(s), expected);
    }

    #[test]
    fn test_suffix_array_empty() {
        let s = "";
        let expected: Vec<usize> = vec![];
        assert_eq!(build_suffix_array(s), expected);
    }
}
