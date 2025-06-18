use std::collections::HashSet;

pub fn has_duplicates<T: std::hash::Hash + Eq>(items: &[T]) -> bool {
    let mut seen = HashSet::new();
    for item in items {
        if !seen.insert(item) {
            return true; // Already seen
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_duplicates() {
        assert!(!has_duplicates(&[1, 2, 3, 4]));
        assert!(has_duplicates(&[1, 2, 3, 1]));
        assert!(!has_duplicates(&["a", "b", "c"]));
        assert!(has_duplicates(&["a", "b", "a"]));
    }
}