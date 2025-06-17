use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// Node in Huffman Tree
#[derive(Debug, Eq)]
enum HuffmanNode {
    Leaf {
        ch: char,
        freq: usize,
    },
    Internal {
        freq: usize,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn freq(&self) -> usize {
        match self {
            HuffmanNode::Leaf { freq, .. } => *freq,
            HuffmanNode::Internal { freq, .. } => *freq,
        }
    }
}

// Min-heap comparator
impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse to make min-heap
        other.freq().cmp(&self.freq())
    }
}
impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.freq() == other.freq()
    }
}

// Build Huffman Tree from frequency map
fn build_huffman_tree(freq_map: &HashMap<char, usize>) -> Option<HuffmanNode> {
    let mut heap = BinaryHeap::new();

    for (&ch, &freq) in freq_map.iter() {
        heap.push(HuffmanNode::Leaf { ch, freq });
    }

    while heap.len() > 1 {
        let node1 = heap.pop().unwrap();
        let node2 = heap.pop().unwrap();

        let merged = HuffmanNode::Internal {
            freq: node1.freq() + node2.freq(),
            left: Box::new(node1),
            right: Box::new(node2),
        };

        heap.push(merged);
    }

    heap.pop()
}

// Generate binary codes from tree
fn generate_codes(node: &HuffmanNode, prefix: String, codes: &mut HashMap<char, String>) {
    match node {
        HuffmanNode::Leaf { ch, .. } => {
            codes.insert(*ch, prefix);
        }
        HuffmanNode::Internal { left, right, .. } => {
            generate_codes(left, format!("{}0", prefix), codes);
            generate_codes(right, format!("{}1", prefix), codes);
        }
    }
}

// Encode text using Huffman Coding
pub fn huffman_encode(text: &str) -> (String, HashMap<char, String>) {
    let mut freq_map = HashMap::new();
    for ch in text.chars() {
        *freq_map.entry(ch).or_insert(0) += 1;
    }

    let tree = build_huffman_tree(&freq_map).expect("Empty input");
    let mut codes = HashMap::new();
    generate_codes(&tree, String::new(), &mut codes);

    let encoded: String = text.chars().map(|c| codes[&c].clone()).collect();
    (encoded, codes)
}

// Decode binary string using Huffman codes
pub fn huffman_decode(encoded: &str, codes: &HashMap<char, String>) -> String {
    let mut reverse = HashMap::new();
    for (k, v) in codes.iter() {
        reverse.insert(v.clone(), *k);
    }

    let mut result = String::new();
    let mut buffer = String::new();

    for bit in encoded.chars() {
        buffer.push(bit);
        if let Some(&ch) = reverse.get(&buffer) {
            result.push(ch);
            buffer.clear();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_encode_decode() {
        let text = "banana";
        let (encoded, codes) = huffman_encode(text);
        let decoded = huffman_decode(&encoded, &codes);
        assert_eq!(text, decoded);
    }
}
