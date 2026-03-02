// Naive suffix tree representation in Rust for educational purposes
use std::collections::HashMap;

#[derive(Debug)]
struct Node {
    children: HashMap<char, Edge>,
}

#[derive(Debug)]
struct Edge {
    label: String,
    node: Box<Node>,
}

#[derive(Debug)]
pub struct SuffixTree {
    root: Node,
}

impl SuffixTree {
    pub fn new(text: &str) -> Self {
        let mut tree = SuffixTree {
            root: Node {
                children: HashMap::new(),
            },
        };
        let text = format!("{}$", text);
        for i in 0..text.len() {
            tree.insert_suffix(&text[i..]);
        }
        tree
    }

    fn insert_suffix(&mut self, suffix: &str) {
        let mut node = &mut self.root;
        let mut chars = suffix.chars().peekable();

        while let Some(&c) = chars.peek() {
            if node.children.contains_key(&c) {
                // Extract edge by key and temporarily remove it
                let mut edge = node.children.remove(&c).unwrap();
                let mut label_chars = edge.label.chars();
                let mut match_len = 0;

                while let (Some(&sc), Some(lc)) = (chars.peek(), label_chars.next()) {
                    if sc != lc {
                        break;
                    }
                    chars.next();
                    match_len += 1;
                }

                if match_len == edge.label.len() {
                    // Full match — continue deeper
                    node.children.insert(c, edge); // Return edge back
                    node = node.children.get_mut(&c).unwrap().node.as_mut();
                } else {
                    // Edge splitting
                    let remaining_label = edge.label[match_len..].to_string();
                    let mut split_node = Node {
                        children: HashMap::new(),
                    };

                    split_node.children.insert(
                        remaining_label.chars().next().unwrap(),
                        Edge {
                            label: remaining_label,
                            node: edge.node,
                        },
                    );

                    let new_suffix: String = chars.by_ref().collect();
                    if let Some(nc) = new_suffix.chars().next() {
                        split_node.children.insert(
                            nc,
                            Edge {
                                label: new_suffix,
                                node: Box::new(Node {
                                    children: HashMap::new(),
                                }),
                            },
                        );
                    }

                    edge.label = edge.label[..match_len].to_string();
                    edge.node = Box::new(split_node);
                    node.children.insert(c, edge);
                    return;
                }
            } else {
                // Insert new suffix
                let new_suffix: String = chars.collect();
                node.children.insert(
                    c,
                    Edge {
                        label: new_suffix,
                        node: Box::new(Node {
                            children: HashMap::new(),
                        }),
                    },
                );
                return;
            }
        }
    }

    pub fn contains(&self, pattern: &str) -> bool {
        let mut node = &self.root;
        let mut chars = pattern.chars().peekable();

        while let Some(&c) = chars.peek() {
            match node.children.get(&c) {
                Some(edge) => {
                    let mut label_chars = edge.label.chars();
                    while let (Some(&sc), Some(lc)) = (chars.peek(), label_chars.next()) {
                        if sc != lc {
                            return false;
                        }
                        chars.next();
                    }
                    node = &edge.node;
                }
                None => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_tree_contains() {
        let tree = SuffixTree::new("banana");
        assert!(tree.contains("ban"));
        assert!(tree.contains("ana"));
        assert!(tree.contains("nana"));
        assert!(tree.contains("a"));
        assert!(tree.contains("banana"));
        assert!(!tree.contains("band"));
        assert!(!tree.contains("apple"));
    }
}
