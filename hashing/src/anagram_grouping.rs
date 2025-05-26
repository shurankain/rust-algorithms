use std::collections::HashMap;

pub fn group_anagrams(words: &[&str]) -> Vec<Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for word in words {
        let mut chars: Vec<char> = word.chars().collect();
        chars.sort_unstable(); // faster than sort()
        let key: String = chars.into_iter().collect();

        map.entry(key).or_default().push(word.to_string());
    }

    map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_anagrams_basic() {
        let input = vec!["eat", "tea", "tan", "ate", "nat", "bat"];
        let mut result = group_anagrams(&input);

        for group in &mut result {
            group.sort();
        }

        result.sort(); // sorting externally, for test stability

        let expected = vec![vec!["ate", "eat", "tea"], vec!["bat"], vec!["nat", "tan"]];

        let mut expected_sorted = expected
            .into_iter()
            .map(|mut group| {
                group.sort();
                group
            })
            .collect::<Vec<_>>();
        expected_sorted.sort();

        assert_eq!(result, expected_sorted);
    }

    #[test]
    fn test_group_anagrams_empty() {
        let input: Vec<&str> = vec![];
        let result = group_anagrams(&input);
        assert!(result.is_empty());
    }

    #[test]
    fn test_group_anagrams_single_word() {
        let input = vec!["hello"];
        let result = group_anagrams(&input);
        assert_eq!(result, vec![vec!["hello".to_string()]]);
    }
}
