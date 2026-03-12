// Aho-Corasick Algorithm - Efficient multi-pattern string matching
// Finds all occurrences of multiple patterns in a text in a single pass

use std::collections::{HashMap, VecDeque};

/// A match found by the Aho-Corasick algorithm
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// Index of the pattern that matched
    pub pattern_index: usize,
    /// Starting position in the text
    pub start: usize,
    /// Ending position in the text (exclusive)
    pub end: usize,
}

/// Aho-Corasick automaton for multi-pattern matching
#[derive(Debug)]
pub struct AhoCorasick {
    /// Trie nodes: each node maps char -> next state
    goto: Vec<HashMap<char, usize>>,
    /// Failure function: state -> fallback state
    fail: Vec<usize>,
    /// Output function: state -> list of pattern indices that match here
    output: Vec<Vec<usize>>,
    /// The patterns stored for reference
    patterns: Vec<String>,
}

impl AhoCorasick {
    /// Build an Aho-Corasick automaton from a list of patterns
    /// Time: O(sum of pattern lengths)
    /// Space: O(sum of pattern lengths)
    pub fn new(patterns: &[&str]) -> Self {
        let mut ac = AhoCorasick {
            goto: vec![HashMap::new()],
            fail: vec![0],
            output: vec![Vec::new()],
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
        };

        // Build the trie (goto function)
        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            ac.add_pattern(pattern, pattern_idx);
        }

        // Build failure and output functions using BFS
        ac.build_failure_function();

        ac
    }

    /// Add a pattern to the trie
    fn add_pattern(&mut self, pattern: &str, pattern_idx: usize) {
        let mut state = 0;

        for ch in pattern.chars() {
            state = match self.goto[state].get(&ch) {
                Some(&next) => next,
                None => {
                    let new_state = self.goto.len();
                    self.goto[state].insert(ch, new_state);
                    self.goto.push(HashMap::new());
                    self.fail.push(0);
                    self.output.push(Vec::new());
                    new_state
                }
            };
        }

        // Mark this state as accepting for this pattern
        self.output[state].push(pattern_idx);
    }

    /// Build the failure function using BFS
    fn build_failure_function(&mut self) {
        let mut queue = VecDeque::new();

        // Initialize: depth-1 states have failure link to root
        for &next_state in self.goto[0].values() {
            self.fail[next_state] = 0;
            queue.push_back(next_state);
        }

        // BFS to compute failure function for deeper states
        while let Some(state) = queue.pop_front() {
            for (&ch, &next_state) in &self.goto[state] {
                queue.push_back(next_state);

                // Follow failure links to find the longest proper suffix
                let mut failure = self.fail[state];
                while failure != 0 && !self.goto[failure].contains_key(&ch) {
                    failure = self.fail[failure];
                }

                self.fail[next_state] = self.goto[failure].get(&ch).copied().unwrap_or(0);

                // If we land on root and there's a direct edge, take it
                if self.fail[next_state] == 0
                    && self.goto[0].contains_key(&ch)
                    && next_state != self.goto[0][&ch]
                {
                    self.fail[next_state] = self.goto[0][&ch];
                }

                // Merge output from failure state
                let fail_outputs = self.output[self.fail[next_state]].clone();
                self.output[next_state].extend(fail_outputs);
            }
        }
    }

    /// Find all matches in the text
    /// Time: O(text.len() + number of matches)
    pub fn find_all(&self, text: &str) -> Vec<Match> {
        let mut matches = Vec::new();
        let mut state = 0;

        for (i, ch) in text.chars().enumerate() {
            // Follow failure links until we find a valid transition or reach root
            while state != 0 && !self.goto[state].contains_key(&ch) {
                state = self.fail[state];
            }

            // Take the transition if it exists
            state = self.goto[state].get(&ch).copied().unwrap_or(0);

            // Report all patterns that match at this position
            for &pattern_idx in &self.output[state] {
                let pattern_len = self.patterns[pattern_idx].chars().count();
                matches.push(Match {
                    pattern_index: pattern_idx,
                    start: i + 1 - pattern_len,
                    end: i + 1,
                });
            }
        }

        matches
    }

    /// Find the first match in the text
    pub fn find_first(&self, text: &str) -> Option<Match> {
        let mut state = 0;

        for (i, ch) in text.chars().enumerate() {
            while state != 0 && !self.goto[state].contains_key(&ch) {
                state = self.fail[state];
            }

            state = self.goto[state].get(&ch).copied().unwrap_or(0);

            if !self.output[state].is_empty() {
                let pattern_idx = self.output[state][0];
                let pattern_len = self.patterns[pattern_idx].chars().count();
                return Some(Match {
                    pattern_index: pattern_idx,
                    start: i + 1 - pattern_len,
                    end: i + 1,
                });
            }
        }

        None
    }

    /// Check if any pattern matches in the text
    pub fn is_match(&self, text: &str) -> bool {
        self.find_first(text).is_some()
    }

    /// Count total number of matches
    pub fn count_matches(&self, text: &str) -> usize {
        self.find_all(text).len()
    }

    /// Get the number of patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get a pattern by index
    pub fn pattern(&self, index: usize) -> Option<&str> {
        self.patterns.get(index).map(|s| s.as_str())
    }

    /// Replace all matches with a replacement string
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        let matches = self.find_all(text);
        if matches.is_empty() {
            return text.to_string();
        }

        // Sort matches by start position
        let mut sorted_matches = matches;
        sorted_matches.sort_by_key(|m| m.start);

        // Handle overlapping matches by taking the first one
        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();
        let mut last_end = 0;

        for m in sorted_matches {
            if m.start >= last_end {
                // Add text before this match
                result.extend(&chars[last_end..m.start]);
                // Add replacement
                result.push_str(replacement);
                last_end = m.end;
            }
        }

        // Add remaining text
        result.extend(&chars[last_end..]);

        result
    }

    /// Replace matches with pattern-specific replacements
    pub fn replace_all_with(&self, text: &str, replacements: &[&str]) -> String {
        assert_eq!(replacements.len(), self.patterns.len());

        let matches = self.find_all(text);
        if matches.is_empty() {
            return text.to_string();
        }

        let mut sorted_matches = matches;
        sorted_matches.sort_by_key(|m| m.start);

        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();
        let mut last_end = 0;

        for m in sorted_matches {
            if m.start >= last_end {
                result.extend(&chars[last_end..m.start]);
                result.push_str(replacements[m.pattern_index]);
                last_end = m.end;
            }
        }

        result.extend(&chars[last_end..]);

        result
    }
}

/// Builder for Aho-Corasick with additional options
#[derive(Debug, Default)]
pub struct AhoCorasickBuilder {
    patterns: Vec<String>,
    case_insensitive: bool,
}

impl AhoCorasickBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pattern
    pub fn add_pattern(mut self, pattern: &str) -> Self {
        self.patterns.push(pattern.to_string());
        self
    }

    /// Add multiple patterns
    pub fn add_patterns(mut self, patterns: &[&str]) -> Self {
        self.patterns.extend(patterns.iter().map(|s| s.to_string()));
        self
    }

    /// Enable case-insensitive matching
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        self.case_insensitive = yes;
        self
    }

    /// Build the automaton
    pub fn build(self) -> AhoCorasick {
        let patterns: Vec<String> = if self.case_insensitive {
            self.patterns.iter().map(|s| s.to_lowercase()).collect()
        } else {
            self.patterns
        };

        let pattern_refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
        AhoCorasick::new(&pattern_refs)
    }
}

/// Convenience function to find all matches
pub fn find_all_matches(patterns: &[&str], text: &str) -> Vec<Match> {
    let ac = AhoCorasick::new(patterns);
    ac.find_all(text)
}

/// Convenience function to check if any pattern matches
pub fn any_match(patterns: &[&str], text: &str) -> bool {
    let ac = AhoCorasick::new(patterns);
    ac.is_match(text)
}

/// Streaming matcher for processing text incrementally
#[derive(Debug)]
pub struct StreamingMatcher<'a> {
    ac: &'a AhoCorasick,
    state: usize,
    position: usize,
}

impl<'a> StreamingMatcher<'a> {
    pub fn new(ac: &'a AhoCorasick) -> Self {
        StreamingMatcher {
            ac,
            state: 0,
            position: 0,
        }
    }

    /// Process a chunk of text and return matches found
    pub fn process(&mut self, chunk: &str) -> Vec<Match> {
        let mut matches = Vec::new();

        for ch in chunk.chars() {
            while self.state != 0 && !self.ac.goto[self.state].contains_key(&ch) {
                self.state = self.ac.fail[self.state];
            }

            self.state = self.ac.goto[self.state].get(&ch).copied().unwrap_or(0);

            for &pattern_idx in &self.ac.output[self.state] {
                let pattern_len = self.ac.patterns[pattern_idx].chars().count();
                matches.push(Match {
                    pattern_index: pattern_idx,
                    start: self.position + 1 - pattern_len,
                    end: self.position + 1,
                });
            }

            self.position += 1;
        }

        matches
    }

    /// Reset the matcher state
    pub fn reset(&mut self) {
        self.state = 0;
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_matching() {
        let ac = AhoCorasick::new(&["he", "she", "his", "hers"]);
        let matches = ac.find_all("ushers");

        // Should find: "she" at 1, "he" at 2, "hers" at 2
        assert_eq!(matches.len(), 3);

        let patterns_found: Vec<&str> = matches
            .iter()
            .map(|m| ac.pattern(m.pattern_index).unwrap())
            .collect();
        assert!(patterns_found.contains(&"she"));
        assert!(patterns_found.contains(&"he"));
        assert!(patterns_found.contains(&"hers"));
    }

    #[test]
    fn test_single_pattern() {
        let ac = AhoCorasick::new(&["abc"]);
        let matches = ac.find_all("abcabcabc");

        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[1].start, 3);
        assert_eq!(matches[2].start, 6);
    }

    #[test]
    fn test_no_match() {
        let ac = AhoCorasick::new(&["xyz"]);
        let matches = ac.find_all("abcdef");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_empty_patterns() {
        let ac = AhoCorasick::new(&[]);
        let matches = ac.find_all("anything");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_overlapping_patterns() {
        let ac = AhoCorasick::new(&["a", "ab", "abc"]);
        let matches = ac.find_all("abc");

        // Should find all three patterns
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_find_first() {
        let ac = AhoCorasick::new(&["world", "hello"]);
        let first = ac.find_first("hello world");

        assert!(first.is_some());
        let m = first.unwrap();
        assert_eq!(m.start, 0);
        assert_eq!(ac.pattern(m.pattern_index).unwrap(), "hello");
    }

    #[test]
    fn test_is_match() {
        let ac = AhoCorasick::new(&["needle"]);

        assert!(ac.is_match("haystack with needle inside"));
        assert!(!ac.is_match("haystack without"));
    }

    #[test]
    fn test_count_matches() {
        let ac = AhoCorasick::new(&["the"]);
        let count = ac.count_matches("the cat and the dog went to the park");

        assert_eq!(count, 3);
    }

    #[test]
    fn test_replace_all() {
        let ac = AhoCorasick::new(&["cat", "dog"]);
        let result = ac.replace_all("the cat and dog", "animal");

        assert_eq!(result, "the animal and animal");
    }

    #[test]
    fn test_replace_all_with() {
        let ac = AhoCorasick::new(&["cat", "dog"]);
        let result = ac.replace_all_with("the cat and dog", &["feline", "canine"]);

        assert_eq!(result, "the feline and canine");
    }

    #[test]
    fn test_unicode() {
        let ac = AhoCorasick::new(&["日本", "中国"]);
        let matches = ac.find_all("日本と中国");

        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_pattern_count() {
        let ac = AhoCorasick::new(&["a", "b", "c"]);
        assert_eq!(ac.pattern_count(), 3);
    }

    #[test]
    fn test_pattern_retrieval() {
        let ac = AhoCorasick::new(&["foo", "bar"]);

        assert_eq!(ac.pattern(0), Some("foo"));
        assert_eq!(ac.pattern(1), Some("bar"));
        assert_eq!(ac.pattern(2), None);
    }

    #[test]
    fn test_builder() {
        let ac = AhoCorasickBuilder::new()
            .add_pattern("hello")
            .add_patterns(&["world", "foo"])
            .build();

        assert_eq!(ac.pattern_count(), 3);
    }

    #[test]
    fn test_streaming_matcher() {
        let ac = AhoCorasick::new(&["hello", "world"]);
        let mut matcher = StreamingMatcher::new(&ac);

        let matches1 = matcher.process("hel");
        assert!(matches1.is_empty());

        let matches2 = matcher.process("lo wor");
        assert_eq!(matches2.len(), 1);
        assert_eq!(ac.pattern(matches2[0].pattern_index).unwrap(), "hello");

        let matches3 = matcher.process("ld");
        assert_eq!(matches3.len(), 1);
        assert_eq!(ac.pattern(matches3[0].pattern_index).unwrap(), "world");
    }

    #[test]
    fn test_streaming_matcher_reset() {
        let ac = AhoCorasick::new(&["ab"]);
        let mut matcher = StreamingMatcher::new(&ac);

        matcher.process("a");
        matcher.reset();

        // After reset, "b" alone shouldn't match
        let matches = matcher.process("b");
        assert!(matches.is_empty());

        // But "ab" should
        let matches = matcher.process("ab");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_convenience_functions() {
        let matches = find_all_matches(&["foo", "bar"], "foobar");
        assert_eq!(matches.len(), 2);

        assert!(any_match(&["needle"], "haystack needle"));
        assert!(!any_match(&["xyz"], "abc"));
    }

    #[test]
    fn test_match_positions() {
        let ac = AhoCorasick::new(&["abc"]);
        let matches = ac.find_all("xxxabcyyy");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 3);
        assert_eq!(matches[0].end, 6);
    }

    #[test]
    fn test_prefix_patterns() {
        let ac = AhoCorasick::new(&["a", "aa", "aaa"]);
        let matches = ac.find_all("aaa");

        // Should find: "a" at 0, "a" at 1, "aa" at 0, "a" at 2, "aa" at 1, "aaa" at 0
        assert_eq!(matches.len(), 6);
    }

    #[test]
    fn test_suffix_patterns() {
        let ac = AhoCorasick::new(&["abc", "bc", "c"]);
        let matches = ac.find_all("abc");

        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_repeated_pattern() {
        let ac = AhoCorasick::new(&["aa"]);
        let matches = ac.find_all("aaaa");

        // "aa" appears at positions 0, 1, 2
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_long_text() {
        let patterns = ["needle"];
        let text = "a".repeat(10000) + "needle" + &"b".repeat(10000);

        let ac = AhoCorasick::new(&patterns);
        let matches = ac.find_all(&text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 10000);
    }

    #[test]
    fn test_many_patterns() {
        let patterns: Vec<String> = (0..100).map(|i| format!("pattern{}", i)).collect();
        let pattern_refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();

        let ac = AhoCorasick::new(&pattern_refs);
        let text = "pattern42 and pattern99";
        let matches = ac.find_all(text);

        // Should find at least pattern42 and pattern99
        // Also finds pattern4 (prefix of pattern42) and pattern9 (prefix of pattern99)
        assert!(matches.len() >= 2);

        let found_patterns: Vec<&str> = matches
            .iter()
            .map(|m| ac.pattern(m.pattern_index).unwrap())
            .collect();
        assert!(found_patterns.contains(&"pattern42"));
        assert!(found_patterns.contains(&"pattern99"));
    }

    #[test]
    fn test_replace_overlapping() {
        let ac = AhoCorasick::new(&["ab", "bc"]);
        let result = ac.replace_all("abc", "X");

        // "ab" matches first at position 0, so "bc" is partially consumed
        assert_eq!(result, "Xc");
    }

    #[test]
    fn test_empty_text() {
        let ac = AhoCorasick::new(&["abc"]);
        let matches = ac.find_all("");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_classic_example() {
        // The classic Aho-Corasick example from the original paper
        let ac = AhoCorasick::new(&["a", "ab", "bab", "bc", "bca", "c", "caa"]);
        let matches = ac.find_all("abccab");

        // Verify all expected patterns are found
        let found_patterns: Vec<&str> = matches
            .iter()
            .map(|m| ac.pattern(m.pattern_index).unwrap())
            .collect();

        assert!(found_patterns.contains(&"a"));
        assert!(found_patterns.contains(&"ab"));
        assert!(found_patterns.contains(&"bc"));
        assert!(found_patterns.contains(&"c"));
    }
}
