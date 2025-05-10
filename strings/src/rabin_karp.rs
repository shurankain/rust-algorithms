use std::hash::{DefaultHasher, Hash, Hasher};

pub fn rabin_karp(pattern: &str, text: &str) -> Option<usize> {
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    let pattern_len = pattern_chars.len();
    if pattern.len() > text.len() {
        return None;
    }

    let pattern_hash = get_hash(&pattern_chars);

    for i in 0..=text_chars.len() - pattern_len {
        let window_chars = &text_chars[i..i + pattern_len];
        let window_hash = get_hash(&window_chars);

        if window_hash == pattern_hash && window_chars == pattern_chars {
            return Some(i);
        }
    }
    None
}

fn get_hash<T>(input: &T) -> u64
where
    T: Hash,
{
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::rabin_karp;

    #[test]
    fn test_pattern_present() {
        let pattern = "abc";
        let text = "mfdgsabcarasv";

        let result = rabin_karp(pattern, text);

        assert_eq!(5, result.unwrap());
    }

    #[test]
    fn test_pattern_is_not_present() {
        let pattern = "abc";
        let text = "mfdgsbcarasv";

        let result = rabin_karp(pattern, text);

        assert_eq!(None, result);
    }
}
