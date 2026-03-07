// Trie (Prefix Tree) - Efficient string storage and prefix matching
//
// A trie stores strings character by character, sharing common prefixes.
// Each node represents a character, and paths from root to marked nodes form words.
//
// Time Complexity:
// - Insert: O(m) where m is the word length
// - Search: O(m)
// - Prefix search: O(m + k) where k is the number of matches
//
// Space Complexity: O(n * m) worst case, but typically much better due to prefix sharing

use std::collections::HashMap;

/// A node in the Trie
#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end_of_word: bool,
    // Optional: store word count for frequency tracking
    word_count: usize,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end_of_word: false,
            word_count: 0,
        }
    }
}

/// Trie (Prefix Tree) for efficient string operations
///
/// Supports insertion, exact search, prefix matching, and autocomplete.
#[derive(Debug, Default)]
pub struct Trie {
    root: TrieNode,
    word_count: usize,
}

impl Trie {
    /// Create a new empty Trie
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            word_count: 0,
        }
    }

    /// Insert a word into the trie
    pub fn insert(&mut self, word: &str) {
        let mut current = &mut self.root;

        for ch in word.chars() {
            current = current.children.entry(ch).or_default();
        }

        if !current.is_end_of_word {
            self.word_count += 1;
        }
        current.is_end_of_word = true;
        current.word_count += 1;
    }

    /// Check if the trie contains an exact word
    pub fn contains(&self, word: &str) -> bool {
        self.find_node(word).is_some_and(|node| node.is_end_of_word)
    }

    /// Check if any word in the trie starts with the given prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.find_node(prefix).is_some()
    }

    /// Get all words that start with the given prefix (autocomplete)
    pub fn autocomplete(&self, prefix: &str) -> Vec<String> {
        let mut results = Vec::new();

        if let Some(node) = self.find_node(prefix) {
            self.collect_words(node, prefix.to_string(), &mut results);
        }

        results
    }

    /// Get the number of times a word was inserted
    pub fn word_frequency(&self, word: &str) -> usize {
        self.find_node(word)
            .filter(|node| node.is_end_of_word)
            .map(|node| node.word_count)
            .unwrap_or(0)
    }

    /// Get the total number of unique words in the trie
    pub fn len(&self) -> usize {
        self.word_count
    }

    /// Check if the trie is empty
    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }

    /// Remove a word from the trie. Returns true if the word was present.
    pub fn remove(&mut self, word: &str) -> bool {
        if !self.contains(word) {
            return false;
        }

        Self::remove_recursive(&mut self.root, word, 0);
        self.word_count -= 1;
        true
    }

    /// Get all words in the trie
    pub fn all_words(&self) -> Vec<String> {
        let mut results = Vec::new();
        self.collect_words(&self.root, String::new(), &mut results);
        results
    }

    /// Find the longest common prefix among all words
    pub fn longest_common_prefix(&self) -> String {
        let mut prefix = String::new();
        let mut current = &self.root;

        loop {
            // Stop if we hit a word ending or multiple children or no children
            if current.is_end_of_word || current.children.len() != 1 {
                break;
            }

            let (ch, next) = current.children.iter().next().unwrap();
            prefix.push(*ch);
            current = next;
        }

        prefix
    }

    // Helper: find the node corresponding to a prefix
    fn find_node(&self, prefix: &str) -> Option<&TrieNode> {
        let mut current = &self.root;

        for ch in prefix.chars() {
            current = current.children.get(&ch)?;
        }

        Some(current)
    }

    // Helper: collect all words from a node
    fn collect_words(&self, node: &TrieNode, current_word: String, results: &mut Vec<String>) {
        if node.is_end_of_word {
            results.push(current_word.clone());
        }

        for (ch, child) in &node.children {
            let mut new_word = current_word.clone();
            new_word.push(*ch);
            Self::collect_words_recursive(child, new_word, results);
        }
    }

    fn collect_words_recursive(node: &TrieNode, current_word: String, results: &mut Vec<String>) {
        if node.is_end_of_word {
            results.push(current_word.clone());
        }

        for (ch, child) in &node.children {
            let mut new_word = current_word.clone();
            new_word.push(*ch);
            Self::collect_words_recursive(child, new_word, results);
        }
    }

    // Helper: recursive removal (static to avoid borrow issues)
    fn remove_recursive(node: &mut TrieNode, word: &str, depth: usize) -> bool {
        let chars: Vec<char> = word.chars().collect();

        if depth == chars.len() {
            node.is_end_of_word = false;
            node.word_count = 0;
            return node.children.is_empty();
        }

        let ch = chars[depth];
        if let Some(child) = node.children.get_mut(&ch)
            && Self::remove_recursive(child, word, depth + 1)
        {
            node.children.remove(&ch);
            return !node.is_end_of_word && node.children.is_empty();
        }

        false
    }
}

/// Compressed Trie (Radix Tree / Patricia Trie)
///
/// Optimizes space by merging chains of single-child nodes into single edges.
/// Stores edge labels as strings instead of single characters.
#[derive(Debug, Default)]
pub struct CompressedTrie {
    root: CompressedTrieNode,
    word_count: usize,
}

#[derive(Debug, Default)]
struct CompressedTrieNode {
    children: HashMap<char, (String, CompressedTrieNode)>,
    is_end_of_word: bool,
}

impl CompressedTrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end_of_word: false,
        }
    }
}

impl CompressedTrie {
    /// Create a new empty compressed trie
    pub fn new() -> Self {
        Self {
            root: CompressedTrieNode::new(),
            word_count: 0,
        }
    }

    /// Insert a word into the compressed trie
    pub fn insert(&mut self, word: &str) {
        if word.is_empty() {
            if !self.root.is_end_of_word {
                self.word_count += 1;
                self.root.is_end_of_word = true;
            }
            return;
        }

        let added = Self::insert_recursive(&mut self.root, word);
        if added {
            self.word_count += 1;
        }
    }

    // Returns true if a new word was added
    fn insert_recursive(node: &mut CompressedTrieNode, remaining: &str) -> bool {
        if remaining.is_empty() {
            if !node.is_end_of_word {
                node.is_end_of_word = true;
                return true;
            }
            return false;
        }

        let first_char = remaining.chars().next().unwrap();

        if let Some((edge_label, child)) = node.children.get_mut(&first_char) {
            // Find common prefix length
            let common_len = edge_label
                .chars()
                .zip(remaining.chars())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == edge_label.len() {
                // Edge is a prefix of remaining, continue down
                let rest = &remaining[common_len..];
                Self::insert_recursive(child, rest)
            } else {
                // Need to split the edge
                let common: String = edge_label.chars().take(common_len).collect();
                let edge_rest: String = edge_label.chars().skip(common_len).collect();
                let remaining_rest: String = remaining.chars().skip(common_len).collect();

                // Create new intermediate node
                let mut new_node = CompressedTrieNode::new();

                // Move existing child under new node
                let old_child = std::mem::replace(child, CompressedTrieNode::new());
                let edge_first = edge_rest.chars().next().unwrap();
                new_node.children.insert(edge_first, (edge_rest, old_child));

                // Insert the new word's remaining part
                if remaining_rest.is_empty() {
                    new_node.is_end_of_word = true;
                } else {
                    let rem_first = remaining_rest.chars().next().unwrap();
                    let mut leaf = CompressedTrieNode::new();
                    leaf.is_end_of_word = true;
                    new_node.children.insert(rem_first, (remaining_rest, leaf));
                }

                // Replace edge with common prefix pointing to new node
                node.children.insert(first_char, (common, new_node));
                true
            }
        } else {
            // No matching edge, create new one
            let mut leaf = CompressedTrieNode::new();
            leaf.is_end_of_word = true;
            node.children
                .insert(first_char, (remaining.to_string(), leaf));
            true
        }
    }

    /// Check if the trie contains an exact word
    pub fn contains(&self, word: &str) -> bool {
        if word.is_empty() {
            return self.root.is_end_of_word;
        }

        self.find_node(word).is_some_and(|node| node.is_end_of_word)
    }

    /// Check if any word starts with the given prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true;
        }

        // Need to check if prefix ends mid-edge or at a node
        let mut current = &self.root;
        let mut remaining = prefix;

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();

            if let Some((edge_label, child)) = current.children.get(&first_char) {
                let common_len = edge_label
                    .chars()
                    .zip(remaining.chars())
                    .take_while(|(a, b)| a == b)
                    .count();

                if common_len == remaining.len() {
                    // Prefix ends within or exactly at this edge
                    return true;
                } else if common_len == edge_label.len() {
                    // Continue to next node
                    remaining = &remaining[common_len..];
                    current = child;
                } else {
                    // Mismatch
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Get the number of unique words
    pub fn len(&self) -> usize {
        self.word_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }

    // Helper: find node at end of word path
    fn find_node(&self, word: &str) -> Option<&CompressedTrieNode> {
        let mut current = &self.root;
        let mut remaining = word;

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();

            let (edge_label, child) = current.children.get(&first_char)?;

            if !remaining.starts_with(edge_label.as_str()) {
                return None;
            }

            remaining = &remaining[edge_label.len()..];
            current = child;
        }

        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic Trie tests
    #[test]
    fn test_trie_insert_and_contains() {
        let mut trie = Trie::new();
        trie.insert("hello");
        trie.insert("world");

        assert!(trie.contains("hello"));
        assert!(trie.contains("world"));
        assert!(!trie.contains("hell"));
        assert!(!trie.contains("helloo"));
    }

    #[test]
    fn test_trie_starts_with() {
        let mut trie = Trie::new();
        trie.insert("hello");
        trie.insert("help");
        trie.insert("helicopter");

        assert!(trie.starts_with("hel"));
        assert!(trie.starts_with("hello"));
        assert!(trie.starts_with("help"));
        assert!(!trie.starts_with("hex"));
    }

    #[test]
    fn test_trie_autocomplete() {
        let mut trie = Trie::new();
        trie.insert("car");
        trie.insert("card");
        trie.insert("care");
        trie.insert("careful");
        trie.insert("cart");

        let mut results = trie.autocomplete("car");
        results.sort();

        assert_eq!(results, vec!["car", "card", "care", "careful", "cart"]);

        let mut results2 = trie.autocomplete("care");
        results2.sort();
        assert_eq!(results2, vec!["care", "careful"]);
    }

    #[test]
    fn test_trie_word_frequency() {
        let mut trie = Trie::new();
        trie.insert("test");
        trie.insert("test");
        trie.insert("test");
        trie.insert("other");

        assert_eq!(trie.word_frequency("test"), 3);
        assert_eq!(trie.word_frequency("other"), 1);
        assert_eq!(trie.word_frequency("missing"), 0);
    }

    #[test]
    fn test_trie_len_and_empty() {
        let mut trie = Trie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);

        trie.insert("a");
        trie.insert("b");
        trie.insert("a"); // duplicate

        assert!(!trie.is_empty());
        assert_eq!(trie.len(), 2); // unique words only
    }

    #[test]
    fn test_trie_remove() {
        let mut trie = Trie::new();
        trie.insert("hello");
        trie.insert("help");

        assert!(trie.contains("hello"));
        assert!(trie.remove("hello"));
        assert!(!trie.contains("hello"));
        assert!(trie.contains("help"));

        assert!(!trie.remove("hello")); // already removed
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn test_trie_all_words() {
        let mut trie = Trie::new();
        trie.insert("apple");
        trie.insert("app");
        trie.insert("banana");

        let mut words = trie.all_words();
        words.sort();

        assert_eq!(words, vec!["app", "apple", "banana"]);
    }

    #[test]
    fn test_trie_longest_common_prefix() {
        let mut trie = Trie::new();
        trie.insert("flower");
        trie.insert("flow");
        trie.insert("flight");

        assert_eq!(trie.longest_common_prefix(), "fl");

        let mut trie2 = Trie::new();
        trie2.insert("dog");
        trie2.insert("racecar");
        trie2.insert("car");

        assert_eq!(trie2.longest_common_prefix(), "");
    }

    #[test]
    fn test_trie_unicode() {
        let mut trie = Trie::new();
        trie.insert("日本語");
        trie.insert("日本");
        trie.insert("こんにちは");

        assert!(trie.contains("日本語"));
        assert!(trie.contains("日本"));
        assert!(trie.starts_with("日"));
        assert!(!trie.contains("日"));

        let results = trie.autocomplete("日");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_trie_empty_string() {
        let mut trie = Trie::new();
        trie.insert("");

        assert!(trie.contains(""));
        assert!(trie.starts_with(""));
        assert_eq!(trie.len(), 1);
    }

    // Compressed Trie tests
    #[test]
    fn test_compressed_trie_basic() {
        let mut trie = CompressedTrie::new();
        trie.insert("test");
        trie.insert("testing");
        trie.insert("tested");

        assert!(trie.contains("test"));
        assert!(trie.contains("testing"));
        assert!(trie.contains("tested"));
        assert!(!trie.contains("tes"));
        assert!(!trie.contains("tester"));
    }

    #[test]
    fn test_compressed_trie_starts_with() {
        let mut trie = CompressedTrie::new();
        trie.insert("algorithm");
        trie.insert("algorithmic");

        assert!(trie.starts_with("alg"));
        assert!(trie.starts_with("algorithm"));
        assert!(trie.starts_with("algorithmi"));
        assert!(!trie.starts_with("alt"));
    }

    #[test]
    fn test_compressed_trie_split() {
        let mut trie = CompressedTrie::new();
        trie.insert("romane");
        trie.insert("romanus");
        trie.insert("romulus");

        assert!(trie.contains("romane"));
        assert!(trie.contains("romanus"));
        assert!(trie.contains("romulus"));
        assert!(!trie.contains("roman"));
        assert!(!trie.contains("rom"));
    }

    #[test]
    fn test_compressed_trie_len() {
        let mut trie = CompressedTrie::new();
        assert!(trie.is_empty());

        trie.insert("a");
        trie.insert("ab");
        trie.insert("abc");

        assert_eq!(trie.len(), 3);
        assert!(!trie.is_empty());
    }

    #[test]
    fn test_compressed_trie_empty_string() {
        let mut trie = CompressedTrie::new();
        trie.insert("");

        assert!(trie.contains(""));
        assert!(trie.starts_with(""));
        assert_eq!(trie.len(), 1);
    }
}
