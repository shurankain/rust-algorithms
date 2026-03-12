// Boyer-Moore String Search Algorithm
// One of the most efficient string searching algorithms for practical use

use std::collections::HashMap;

/// Find the first occurrence of pattern in text using Boyer-Moore algorithm
/// Returns the starting index of the first match, or None if not found
/// Time: O(n/m) best case, O(nm) worst case (rare with both heuristics)
/// Space: O(m + alphabet_size)
pub fn boyer_moore(pattern: &str, text: &str) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();

    if pattern_chars.len() > text_chars.len() {
        return None;
    }

    let bad_char = build_bad_char_table(&pattern_chars);
    let good_suffix = build_good_suffix_table(&pattern_chars);

    let mut i = pattern_chars.len() - 1; // Position in text

    while i < text_chars.len() {
        let mut j = pattern_chars.len() - 1; // Position in pattern
        let mut k = i; // Current position in text for comparison

        // Match from right to left
        while pattern_chars[j] == text_chars[k] {
            if j == 0 {
                return Some(k); // Full match found
            }
            j -= 1;
            k -= 1;
        }

        // Mismatch: calculate shift using both heuristics
        let bad_char_shift = bad_char_rule(&bad_char, text_chars[k], j);
        let good_suffix_shift = good_suffix[j];

        i += bad_char_shift.max(good_suffix_shift);
    }

    None
}

/// Find all occurrences of pattern in text
pub fn boyer_moore_all(pattern: &str, text: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();

    if pattern_chars.len() > text_chars.len() {
        return Vec::new();
    }

    let bad_char = build_bad_char_table(&pattern_chars);
    let good_suffix = build_good_suffix_table(&pattern_chars);

    let mut results = Vec::new();
    let mut i = pattern_chars.len() - 1;

    while i < text_chars.len() {
        let mut j = pattern_chars.len() - 1;
        let mut k = i;

        while pattern_chars[j] == text_chars[k] {
            if j == 0 {
                results.push(k);
                // Continue searching after this match
                i += 1;
                break;
            }
            j -= 1;
            k -= 1;
        }

        if j > 0 || pattern_chars[j] != text_chars[k] {
            let bad_char_shift = bad_char_rule(&bad_char, text_chars[k], j);
            let good_suffix_shift = good_suffix[j];
            i += bad_char_shift.max(good_suffix_shift).max(1);
        }
    }

    results
}

/// Build the bad character table
/// Maps each character to its rightmost position in the pattern
fn build_bad_char_table(pattern: &[char]) -> HashMap<char, usize> {
    let mut table = HashMap::new();
    for (i, &ch) in pattern.iter().enumerate() {
        table.insert(ch, i);
    }
    table
}

/// Calculate shift based on bad character rule
fn bad_char_rule(table: &HashMap<char, usize>, ch: char, j: usize) -> usize {
    match table.get(&ch) {
        Some(&pos) if pos < j => j - pos,
        Some(_) => 1,  // Character is to the right of mismatch
        None => j + 1, // Character not in pattern
    }
}

/// Build the good suffix table using the standard algorithm
fn build_good_suffix_table(pattern: &[char]) -> Vec<usize> {
    let m = pattern.len();
    let mut shift = vec![0; m];
    let mut border_pos = vec![0; m + 1];

    // Case 1: Suffix appears elsewhere in pattern
    compute_suffix_positions(pattern, &mut shift, &mut border_pos);

    // Case 2: A prefix of the pattern matches a suffix of the matched portion
    compute_prefix_positions(pattern, &mut shift, &border_pos);

    shift
}

/// Compute suffix positions for good suffix rule
fn compute_suffix_positions(pattern: &[char], shift: &mut [usize], border_pos: &mut [usize]) {
    let m = pattern.len();
    let mut i = m;
    let mut j = m + 1;
    border_pos[i] = j;

    while i > 0 {
        while j <= m && pattern[i - 1] != pattern[j - 1] {
            if shift[j - 1] == 0 {
                shift[j - 1] = j - i;
            }
            j = border_pos[j];
        }
        i -= 1;
        j -= 1;
        border_pos[i] = j;
    }
}

/// Compute prefix positions for good suffix rule
fn compute_prefix_positions(pattern: &[char], shift: &mut [usize], border_pos: &[usize]) {
    let m = pattern.len();
    let mut j = border_pos[0];

    for (i, s) in shift.iter_mut().enumerate().take(m) {
        if *s == 0 {
            *s = j;
        }
        if i + 1 == j {
            j = border_pos[j];
        }
    }
}

/// Simplified Boyer-Moore using only bad character rule (Boyer-Moore-Horspool)
/// Simpler and often faster for short patterns
pub fn boyer_moore_horspool(pattern: &str, text: &str) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    let m = pattern_chars.len();
    let n = text_chars.len();

    if m > n {
        return None;
    }

    // Build shift table: for each character, how much to shift
    let mut shift: HashMap<char, usize> = HashMap::new();
    for (idx, &ch) in pattern_chars.iter().enumerate().take(m - 1) {
        shift.insert(ch, m - 1 - idx);
    }

    let mut i = 0;
    while i <= n - m {
        let mut j = m - 1;

        // Match from right to left
        while pattern_chars[j] == text_chars[i + j] {
            if j == 0 {
                return Some(i);
            }
            j -= 1;
        }

        // Shift based on the last character of the window
        let last_char = text_chars[i + m - 1];
        i += shift.get(&last_char).copied().unwrap_or(m);
    }

    None
}

/// Find all occurrences using Horspool variant
pub fn boyer_moore_horspool_all(pattern: &str, text: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    let m = pattern_chars.len();
    let n = text_chars.len();

    if m > n {
        return Vec::new();
    }

    let mut shift: HashMap<char, usize> = HashMap::new();
    for (idx, &ch) in pattern_chars.iter().enumerate().take(m - 1) {
        shift.insert(ch, m - 1 - idx);
    }

    let mut results = Vec::new();
    let mut i = 0;

    while i <= n - m {
        let mut j = m - 1;

        while pattern_chars[j] == text_chars[i + j] {
            if j == 0 {
                results.push(i);
                break;
            }
            j -= 1;
        }

        if i + m - 1 < n {
            let last_char = text_chars[i + m - 1];
            let s = shift.get(&last_char).copied().unwrap_or(m);
            i += if j == 0 { 1 } else { s };
        } else {
            break;
        }
    }

    results
}

/// Sunday's variant of Boyer-Moore (looks at character after the window)
pub fn sunday(pattern: &str, text: &str) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    let m = pattern_chars.len();
    let n = text_chars.len();

    if m > n {
        return None;
    }

    // Build shift table: for each character, how much to shift
    // Default shift is m + 1 (pattern length + 1)
    let mut shift: HashMap<char, usize> = HashMap::new();
    for (i, &ch) in pattern_chars.iter().enumerate() {
        shift.insert(ch, m - i);
    }

    let mut i = 0;
    while i <= n - m {
        let mut j = 0;

        // Match from left to right
        while j < m && pattern_chars[j] == text_chars[i + j] {
            j += 1;
        }

        if j == m {
            return Some(i);
        }

        // Shift based on character just after the window
        if i + m < n {
            let next_char = text_chars[i + m];
            i += shift.get(&next_char).copied().unwrap_or(m + 1);
        } else {
            break;
        }
    }

    None
}

/// Find all occurrences using Sunday's algorithm
pub fn sunday_all(pattern: &str, text: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    let m = pattern_chars.len();
    let n = text_chars.len();

    if m > n {
        return Vec::new();
    }

    let mut shift: HashMap<char, usize> = HashMap::new();
    for (i, &ch) in pattern_chars.iter().enumerate() {
        shift.insert(ch, m - i);
    }

    let mut results = Vec::new();
    let mut i = 0;

    while i <= n - m {
        let mut j = 0;

        while j < m && pattern_chars[j] == text_chars[i + j] {
            j += 1;
        }

        if j == m {
            results.push(i);
        }

        if i + m < n {
            let next_char = text_chars[i + m];
            i += shift.get(&next_char).copied().unwrap_or(m + 1);
        } else {
            break;
        }
    }

    results
}

/// Turbo Boyer-Moore - avoids re-examining characters
/// More efficient for patterns with many repeated characters
pub fn turbo_boyer_moore(pattern: &str, text: &str) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    let m = pattern_chars.len();
    let n = text_chars.len();

    if m > n {
        return None;
    }

    let bad_char = build_bad_char_table(&pattern_chars);
    let good_suffix = build_good_suffix_table(&pattern_chars);

    let mut i = 0; // Starting position in text
    let mut turbo_shift = 0;

    while i <= n - m {
        let mut j = m - 1;

        // Match from right to left, but skip turbo_shift characters
        while j >= turbo_shift && pattern_chars[j] == text_chars[i + j] {
            if j == 0 {
                return Some(i);
            }
            j -= 1;
        }

        if j < turbo_shift {
            // Turbo shift case
            i += turbo_shift + 1;
            turbo_shift = 0;
        } else {
            let bad_char_shift = bad_char_rule(&bad_char, text_chars[i + j], j);
            let gs_shift = good_suffix[j];

            let shift = bad_char_shift.max(gs_shift);
            if shift > gs_shift {
                turbo_shift = 0;
            } else {
                turbo_shift = m - 1 - j;
            }
            i += shift;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boyer_moore_basic() {
        assert_eq!(boyer_moore("abc", "asersvabcgrwgrsabcdfs"), Some(6));
        assert_eq!(boyer_moore("abcd", "asersvabcgrwgrsabcdfs"), Some(15));
    }

    #[test]
    fn test_boyer_moore_not_found() {
        assert_eq!(boyer_moore("xyz", "abcdef"), None);
    }

    #[test]
    fn test_boyer_moore_empty_pattern() {
        assert_eq!(boyer_moore("", "abcdef"), Some(0));
    }

    #[test]
    fn test_boyer_moore_empty_text() {
        assert_eq!(boyer_moore("a", ""), None);
    }

    #[test]
    fn test_boyer_moore_pattern_longer_than_text() {
        assert_eq!(boyer_moore("abcdefg", "abc"), None);
    }

    #[test]
    fn test_boyer_moore_full_match() {
        assert_eq!(boyer_moore("hello", "hello"), Some(0));
    }

    #[test]
    fn test_boyer_moore_at_end() {
        assert_eq!(boyer_moore("world", "hello world"), Some(6));
    }

    #[test]
    fn test_boyer_moore_repeated_pattern() {
        assert_eq!(boyer_moore("aa", "aaaaa"), Some(0));
    }

    #[test]
    fn test_boyer_moore_all_basic() {
        let results = boyer_moore_all("ab", "abcabcab");
        assert_eq!(results, vec![0, 3, 6]);
    }

    #[test]
    fn test_boyer_moore_all_overlapping() {
        let results = boyer_moore_all("aa", "aaaa");
        assert_eq!(results, vec![0, 1, 2]);
    }

    #[test]
    fn test_boyer_moore_all_no_match() {
        let results = boyer_moore_all("xyz", "abcdef");
        assert!(results.is_empty());
    }

    #[test]
    fn test_horspool_basic() {
        assert_eq!(
            boyer_moore_horspool("abc", "asersvabcgrwgrsabcdfs"),
            Some(6)
        );
        assert_eq!(boyer_moore_horspool("xyz", "abcdef"), None);
    }

    #[test]
    fn test_horspool_empty() {
        assert_eq!(boyer_moore_horspool("", "abcdef"), Some(0));
        assert_eq!(boyer_moore_horspool("a", ""), None);
    }

    #[test]
    fn test_horspool_all() {
        let results = boyer_moore_horspool_all("ab", "abcabcab");
        assert_eq!(results, vec![0, 3, 6]);
    }

    #[test]
    fn test_sunday_basic() {
        assert_eq!(sunday("abc", "asersvabcgrwgrsabcdfs"), Some(6));
        assert_eq!(sunday("xyz", "abcdef"), None);
    }

    #[test]
    fn test_sunday_empty() {
        assert_eq!(sunday("", "abcdef"), Some(0));
        assert_eq!(sunday("a", ""), None);
    }

    #[test]
    fn test_sunday_all() {
        let results = sunday_all("ab", "abcabcab");
        assert_eq!(results, vec![0, 3, 6]);
    }

    #[test]
    fn test_turbo_boyer_moore() {
        assert_eq!(turbo_boyer_moore("abc", "asersvabcgrwgrsabcdfs"), Some(6));
        assert_eq!(turbo_boyer_moore("xyz", "abcdef"), None);
    }

    #[test]
    fn test_turbo_empty() {
        assert_eq!(turbo_boyer_moore("", "abcdef"), Some(0));
        assert_eq!(turbo_boyer_moore("a", ""), None);
    }

    #[test]
    fn test_all_algorithms_consistent() {
        let text = "the quick brown fox jumps over the lazy dog";
        let pattern = "fox";

        let bm = boyer_moore(pattern, text);
        let horspool = boyer_moore_horspool(pattern, text);
        let sun = sunday(pattern, text);
        let turbo = turbo_boyer_moore(pattern, text);

        assert_eq!(bm, horspool);
        assert_eq!(bm, sun);
        assert_eq!(bm, turbo);
        assert_eq!(bm, Some(16));
    }

    #[test]
    fn test_all_algorithms_not_found() {
        let text = "abcdefghijklmnop";
        let pattern = "xyz";

        assert_eq!(boyer_moore(pattern, text), None);
        assert_eq!(boyer_moore_horspool(pattern, text), None);
        assert_eq!(sunday(pattern, text), None);
        assert_eq!(turbo_boyer_moore(pattern, text), None);
    }

    #[test]
    fn test_unicode_support() {
        assert_eq!(boyer_moore("日本", "こんにちは日本語"), Some(5));
        assert_eq!(boyer_moore_horspool("日本", "こんにちは日本語"), Some(5));
        assert_eq!(sunday("日本", "こんにちは日本語"), Some(5));
    }

    #[test]
    fn test_long_pattern() {
        let pattern = "abcdefghij";
        let text = "xyzabcdefghijklm";

        assert_eq!(boyer_moore(pattern, text), Some(3));
        assert_eq!(boyer_moore_horspool(pattern, text), Some(3));
        assert_eq!(sunday(pattern, text), Some(3));
        assert_eq!(turbo_boyer_moore(pattern, text), Some(3));
    }

    #[test]
    fn test_repeated_characters() {
        let pattern = "aaa";
        let text = "aaaaaaa";

        assert_eq!(boyer_moore(pattern, text), Some(0));
        assert_eq!(boyer_moore_horspool(pattern, text), Some(0));

        let all_bm = boyer_moore_all(pattern, text);
        let all_horspool = boyer_moore_horspool_all(pattern, text);

        assert_eq!(all_bm, vec![0, 1, 2, 3, 4]);
        assert_eq!(all_horspool, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_bad_character_rule() {
        // Pattern "GCAGAGAG" searching in text with mismatches
        let pattern = "GCAGAGAG";
        let text = "GCATCGCAGAGAGTATACAGTACG";

        assert_eq!(boyer_moore(pattern, text), Some(5));
    }

    #[test]
    fn test_good_suffix_rule() {
        // Pattern with repeated suffix
        let pattern = "ABCAB";
        let text = "ABABCABCAB";

        assert_eq!(boyer_moore(pattern, text), Some(2));
    }
}
