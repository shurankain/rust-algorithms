// Merkle Tree (Hash Tree)
//
// A tree data structure where every leaf node is labeled with the hash of a data block,
// and every non-leaf node is labeled with the cryptographic hash of its children's labels.
// Invented by Ralph Merkle in 1979.
//
// Key properties:
// - Efficient verification: Verify any data block with O(log n) hashes
// - Tamper detection: Any modification changes the root hash
// - Efficient sync: Identify differences between trees in O(log n) comparisons
//
// Used in:
// - Git: Content-addressable storage
// - Bitcoin/Ethereum: Transaction verification (Merkle proofs)
// - Certificate Transparency: Append-only logs
// - Distributed databases: Anti-entropy (Cassandra, DynamoDB)
// - IPFS: Content addressing
// - ZK-proofs: Merkle membership proofs
//
// Variants implemented:
// - Standard binary Merkle tree
// - Merkle proof generation and verification
// - Tree comparison for sync (anti-entropy)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Type alias for a hash value
pub type HashValue = u64;

/// A node in the Merkle tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleNode {
    /// Leaf node containing data hash
    Leaf { hash: HashValue, data_index: usize },
    /// Internal node containing combined hash of children
    Internal {
        hash: HashValue,
        left: Box<MerkleNode>,
        right: Box<MerkleNode>,
    },
    /// Empty placeholder for padding to power of 2
    Empty,
}

impl MerkleNode {
    /// Get the hash of this node
    pub fn hash(&self) -> HashValue {
        match self {
            MerkleNode::Leaf { hash, .. } => *hash,
            MerkleNode::Internal { hash, .. } => *hash,
            MerkleNode::Empty => 0,
        }
    }

    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self, MerkleNode::Leaf { .. })
    }

    /// Check if this node is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, MerkleNode::Empty)
    }
}

/// Direction in a Merkle proof path
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofDirection {
    /// The sibling is on the left
    Left,
    /// The sibling is on the right
    Right,
}

/// A single step in a Merkle proof
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofStep {
    /// Hash of the sibling node
    pub sibling_hash: HashValue,
    /// Direction of the sibling
    pub direction: ProofDirection,
}

/// A Merkle proof for verifying membership
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// The leaf hash being proved
    pub leaf_hash: HashValue,
    /// Index of the leaf in the tree
    pub leaf_index: usize,
    /// Path from leaf to root
    pub path: Vec<ProofStep>,
    /// Expected root hash
    pub root_hash: HashValue,
}

impl MerkleProof {
    /// Verify this proof
    pub fn verify(&self) -> bool {
        let mut current_hash = self.leaf_hash;

        for step in &self.path {
            current_hash = match step.direction {
                ProofDirection::Left => combine_hashes(step.sibling_hash, current_hash),
                ProofDirection::Right => combine_hashes(current_hash, step.sibling_hash),
            };
        }

        current_hash == self.root_hash
    }

    /// Get the proof size (number of hashes)
    pub fn size(&self) -> usize {
        self.path.len()
    }
}

/// A Merkle tree for efficient data verification
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// Root node of the tree
    root: Option<MerkleNode>,
    /// Number of data items (leaves)
    num_leaves: usize,
    /// Original data hashes (for proof generation)
    leaf_hashes: Vec<HashValue>,
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

impl MerkleTree {
    /// Create an empty Merkle tree
    pub fn new() -> Self {
        Self {
            root: None,
            num_leaves: 0,
            leaf_hashes: Vec::new(),
        }
    }

    /// Build a Merkle tree from data items
    pub fn from_data<T: Hash>(data: &[T]) -> Self {
        if data.is_empty() {
            return Self::new();
        }

        let leaf_hashes: Vec<HashValue> = data.iter().map(|item| hash_data(item)).collect();
        Self::from_hashes(&leaf_hashes)
    }

    /// Build a Merkle tree from pre-computed hashes
    pub fn from_hashes(hashes: &[HashValue]) -> Self {
        if hashes.is_empty() {
            return Self::new();
        }

        let leaf_hashes = hashes.to_vec();
        let num_leaves = hashes.len();

        // Build leaves
        let mut nodes: Vec<MerkleNode> = hashes
            .iter()
            .enumerate()
            .map(|(i, &hash)| MerkleNode::Leaf {
                hash,
                data_index: i,
            })
            .collect();

        // Pad to power of 2 if needed
        let target_size = num_leaves.next_power_of_two();
        while nodes.len() < target_size {
            nodes.push(MerkleNode::Empty);
        }

        // Build tree bottom-up
        while nodes.len() > 1 {
            let mut next_level = Vec::with_capacity(nodes.len() / 2);

            for chunk in nodes.chunks(2) {
                let left = chunk[0].clone();
                let right = chunk.get(1).cloned().unwrap_or(MerkleNode::Empty);

                let combined_hash = combine_hashes(left.hash(), right.hash());
                next_level.push(MerkleNode::Internal {
                    hash: combined_hash,
                    left: Box::new(left),
                    right: Box::new(right),
                });
            }

            nodes = next_level;
        }

        Self {
            root: nodes.into_iter().next(),
            num_leaves,
            leaf_hashes,
        }
    }

    /// Get the root hash
    pub fn root_hash(&self) -> Option<HashValue> {
        self.root.as_ref().map(|n| n.hash())
    }

    /// Get the number of leaves (data items)
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.num_leaves == 0
    }

    /// Get the height of the tree
    pub fn height(&self) -> usize {
        if self.num_leaves == 0 {
            0
        } else {
            (self.num_leaves.next_power_of_two() as f64).log2() as usize + 1
        }
    }

    /// Generate a Merkle proof for a leaf at given index
    pub fn generate_proof(&self, leaf_index: usize) -> Option<MerkleProof> {
        if leaf_index >= self.num_leaves {
            return None;
        }

        let root = self.root.as_ref()?;
        let leaf_hash = self.leaf_hashes[leaf_index];

        let mut path = Vec::new();
        let padded_size = self.num_leaves.next_power_of_two();
        let current_index = leaf_index;
        let level_size = padded_size;

        // Traverse from leaf to root, collecting siblings
        fn collect_path(
            node: &MerkleNode,
            target_index: usize,
            level_size: usize,
            path: &mut Vec<ProofStep>,
        ) {
            if let MerkleNode::Internal { left, right, .. } = node {
                let mid = level_size / 2;

                if target_index < mid {
                    // Target is in left subtree, sibling is right
                    path.push(ProofStep {
                        sibling_hash: right.hash(),
                        direction: ProofDirection::Right,
                    });
                    collect_path(left, target_index, mid, path);
                } else {
                    // Target is in right subtree, sibling is left
                    path.push(ProofStep {
                        sibling_hash: left.hash(),
                        direction: ProofDirection::Left,
                    });
                    collect_path(right, target_index - mid, mid, path);
                }
            }
        }

        collect_path(root, current_index, level_size, &mut path);
        path.reverse(); // We collected root-to-leaf, but want leaf-to-root

        Some(MerkleProof {
            leaf_hash,
            leaf_index,
            path,
            root_hash: root.hash(),
        })
    }

    /// Verify a data item exists at the given index
    pub fn verify<T: Hash>(&self, data: &T, index: usize) -> bool {
        if let Some(proof) = self.generate_proof(index) {
            let data_hash = hash_data(data);
            data_hash == proof.leaf_hash && proof.verify()
        } else {
            false
        }
    }

    /// Get leaf hash at index
    pub fn get_leaf_hash(&self, index: usize) -> Option<HashValue> {
        self.leaf_hashes.get(index).copied()
    }

    /// Compare with another tree and find differing leaf indices
    pub fn diff(&self, other: &MerkleTree) -> TreeDiff {
        let mut diff = TreeDiff {
            only_in_self: Vec::new(),
            only_in_other: Vec::new(),
            different: Vec::new(),
            same_root: self.root_hash() == other.root_hash(),
        };

        if diff.same_root {
            return diff;
        }

        // Find differing leaves by comparing subtrees
        fn find_diffs(
            node1: Option<&MerkleNode>,
            node2: Option<&MerkleNode>,
            base_index: usize,
            level_size: usize,
            diff: &mut TreeDiff,
            tree1_leaves: usize,
            tree2_leaves: usize,
        ) {
            match (node1, node2) {
                (Some(n1), Some(n2)) if n1.hash() == n2.hash() => {
                    // Subtrees are identical, skip
                }
                (Some(MerkleNode::Leaf { data_index, .. }), Some(MerkleNode::Leaf { .. })) => {
                    // Different leaf hashes
                    diff.different.push(*data_index);
                }
                (
                    Some(MerkleNode::Internal { left, right, .. }),
                    Some(MerkleNode::Internal {
                        left: left2,
                        right: right2,
                        ..
                    }),
                ) => {
                    let mid = level_size / 2;
                    find_diffs(
                        Some(left),
                        Some(left2),
                        base_index,
                        mid,
                        diff,
                        tree1_leaves,
                        tree2_leaves,
                    );
                    find_diffs(
                        Some(right),
                        Some(right2),
                        base_index + mid,
                        mid,
                        diff,
                        tree1_leaves,
                        tree2_leaves,
                    );
                }
                (Some(MerkleNode::Leaf { data_index, .. }), None)
                | (Some(MerkleNode::Leaf { data_index, .. }), Some(MerkleNode::Empty)) => {
                    diff.only_in_self.push(*data_index);
                }
                (None, Some(MerkleNode::Leaf { data_index, .. }))
                | (Some(MerkleNode::Empty), Some(MerkleNode::Leaf { data_index, .. })) => {
                    diff.only_in_other.push(*data_index);
                }
                _ => {
                    // Mixed internal/leaf or other cases - enumerate leaves
                    for i in base_index..base_index + level_size {
                        if i < tree1_leaves && i >= tree2_leaves {
                            diff.only_in_self.push(i);
                        } else if i >= tree1_leaves && i < tree2_leaves {
                            diff.only_in_other.push(i);
                        } else if i < tree1_leaves && i < tree2_leaves {
                            diff.different.push(i);
                        }
                    }
                }
            }
        }

        let max_leaves = self.num_leaves.max(other.num_leaves);
        let level_size = max_leaves.next_power_of_two();

        find_diffs(
            self.root.as_ref(),
            other.root.as_ref(),
            0,
            level_size,
            &mut diff,
            self.num_leaves,
            other.num_leaves,
        );

        diff
    }

    /// Get audit path for a leaf (alias for generate_proof)
    pub fn audit_path(&self, leaf_index: usize) -> Option<MerkleProof> {
        self.generate_proof(leaf_index)
    }

    /// Get consistency proof between two tree sizes
    /// Used in Certificate Transparency to prove append-only property
    pub fn consistency_proof(&self, old_size: usize) -> Option<ConsistencyProof> {
        if old_size == 0 || old_size > self.num_leaves {
            return None;
        }

        if old_size == self.num_leaves {
            return Some(ConsistencyProof {
                old_size,
                new_size: self.num_leaves,
                path: Vec::new(),
            });
        }

        // Build subtree hashes for consistency proof
        let mut path = Vec::new();

        // Find the path that proves old tree is prefix of new tree
        fn build_consistency_path(
            node: &MerkleNode,
            base_index: usize,
            level_size: usize,
            old_size: usize,
            path: &mut Vec<HashValue>,
            include_self: bool,
        ) {
            match node {
                MerkleNode::Internal { left, right, hash } => {
                    let mid = base_index + level_size / 2;

                    if old_size <= base_index {
                        // This subtree is entirely in the new part
                        if include_self {
                            path.push(*hash);
                        }
                    } else if old_size >= base_index + level_size {
                        // This subtree is entirely in the old part
                        if include_self {
                            path.push(*hash);
                        }
                    } else {
                        // Split point is in this subtree
                        build_consistency_path(
                            left,
                            base_index,
                            level_size / 2,
                            old_size,
                            path,
                            old_size > base_index,
                        );
                        build_consistency_path(
                            right,
                            mid,
                            level_size / 2,
                            old_size,
                            path,
                            old_size < base_index + level_size,
                        );
                    }
                }
                MerkleNode::Leaf { hash, .. } => {
                    if include_self {
                        path.push(*hash);
                    }
                }
                MerkleNode::Empty => {}
            }
        }

        if let Some(root) = &self.root {
            let level_size = self.num_leaves.next_power_of_two();
            build_consistency_path(root, 0, level_size, old_size, &mut path, false);
        }

        Some(ConsistencyProof {
            old_size,
            new_size: self.num_leaves,
            path,
        })
    }
}

/// Difference between two Merkle trees
#[derive(Debug, Clone, Default)]
pub struct TreeDiff {
    /// Leaf indices only in the first tree
    pub only_in_self: Vec<usize>,
    /// Leaf indices only in the second tree
    pub only_in_other: Vec<usize>,
    /// Leaf indices present in both but with different hashes
    pub different: Vec<usize>,
    /// Whether the root hashes are the same
    pub same_root: bool,
}

impl TreeDiff {
    /// Check if trees are identical
    pub fn is_identical(&self) -> bool {
        self.same_root
    }

    /// Total number of differences
    pub fn total_differences(&self) -> usize {
        self.only_in_self.len() + self.only_in_other.len() + self.different.len()
    }
}

/// Consistency proof for append-only logs
#[derive(Debug, Clone)]
pub struct ConsistencyProof {
    /// Size of the old tree
    pub old_size: usize,
    /// Size of the new tree
    pub new_size: usize,
    /// Hashes needed to verify consistency
    pub path: Vec<HashValue>,
}

/// Hash data using default hasher
pub fn hash_data<T: Hash>(data: &T) -> HashValue {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Combine two hashes into one (for internal nodes)
pub fn combine_hashes(left: HashValue, right: HashValue) -> HashValue {
    let mut hasher = DefaultHasher::new();
    left.hash(&mut hasher);
    right.hash(&mut hasher);
    hasher.finish()
}

/// Verify a Merkle proof given leaf hash and root hash
pub fn verify_proof(proof: &MerkleProof) -> bool {
    proof.verify()
}

/// Calculate the expected proof size for a tree with n leaves
pub fn expected_proof_size(num_leaves: usize) -> usize {
    if num_leaves <= 1 {
        0
    } else {
        (num_leaves as f64).log2().ceil() as usize
    }
}

/// Merkle tree statistics
#[derive(Debug, Clone, Default)]
pub struct MerkleStats {
    /// Number of leaves
    pub num_leaves: usize,
    /// Tree height
    pub height: usize,
    /// Total number of nodes (including internal)
    pub total_nodes: usize,
    /// Number of empty padding nodes
    pub empty_nodes: usize,
}

/// Analyze a Merkle tree
pub fn analyze_tree(tree: &MerkleTree) -> MerkleStats {
    fn count_nodes(node: &MerkleNode) -> (usize, usize) {
        match node {
            MerkleNode::Leaf { .. } => (1, 0),
            MerkleNode::Internal { left, right, .. } => {
                let (total_l, empty_l) = count_nodes(left);
                let (total_r, empty_r) = count_nodes(right);
                (1 + total_l + total_r, empty_l + empty_r)
            }
            MerkleNode::Empty => (1, 1),
        }
    }

    let (total_nodes, empty_nodes) = tree.root.as_ref().map(count_nodes).unwrap_or((0, 0));

    MerkleStats {
        num_leaves: tree.num_leaves,
        height: tree.height(),
        total_nodes,
        empty_nodes,
    }
}

/// Sparse Merkle Tree for key-value storage
/// Uses a fixed depth and default empty values
#[derive(Debug, Clone)]
pub struct SparseMerkleTree {
    /// Tree depth (determines key space: 2^depth possible keys)
    depth: usize,
    /// Root hash
    root_hash: HashValue,
    /// Stored leaves: key -> (value_hash, value_data)
    leaves: std::collections::HashMap<u64, HashValue>,
    /// Default hash for empty subtrees at each level
    default_hashes: Vec<HashValue>,
}

impl SparseMerkleTree {
    /// Create a new sparse Merkle tree with given depth
    pub fn new(depth: usize) -> Self {
        // Precompute default hashes for empty subtrees
        let mut default_hashes = vec![0u64; depth + 1];
        default_hashes[0] = 0; // Empty leaf

        for i in 1..=depth {
            default_hashes[i] = combine_hashes(default_hashes[i - 1], default_hashes[i - 1]);
        }

        Self {
            depth,
            root_hash: default_hashes[depth],
            leaves: std::collections::HashMap::new(),
            default_hashes,
        }
    }

    /// Insert or update a key-value pair
    pub fn insert<T: Hash>(&mut self, key: u64, value: &T) {
        let value_hash = hash_data(value);
        self.leaves.insert(key, value_hash);
        self.recompute_root();
    }

    /// Remove a key
    pub fn remove(&mut self, key: u64) -> bool {
        if self.leaves.remove(&key).is_some() {
            self.recompute_root();
            true
        } else {
            false
        }
    }

    /// Get root hash
    pub fn root_hash(&self) -> HashValue {
        self.root_hash
    }

    /// Check if a key exists
    pub fn contains(&self, key: u64) -> bool {
        self.leaves.contains_key(&key)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Recompute root hash (simplified - full recomputation)
    fn recompute_root(&mut self) {
        if self.leaves.is_empty() {
            self.root_hash = self.default_hashes[self.depth];
            return;
        }

        // For simplicity, rebuild the tree
        // A production implementation would use incremental updates
        self.root_hash = self.compute_subtree_hash(0, self.depth);
    }

    /// Compute hash of a subtree
    fn compute_subtree_hash(&self, prefix: u64, remaining_depth: usize) -> HashValue {
        if remaining_depth == 0 {
            // Leaf level
            *self.leaves.get(&prefix).unwrap_or(&0)
        } else {
            let left_prefix = prefix;
            let right_prefix = prefix | (1 << (remaining_depth - 1));

            let left_hash = self.compute_subtree_hash(left_prefix, remaining_depth - 1);
            let right_hash = self.compute_subtree_hash(right_prefix, remaining_depth - 1);

            // Optimize: if both are default, use precomputed default
            if left_hash == self.default_hashes[remaining_depth - 1]
                && right_hash == self.default_hashes[remaining_depth - 1]
            {
                self.default_hashes[remaining_depth]
            } else {
                combine_hashes(left_hash, right_hash)
            }
        }
    }

    /// Generate a proof for a key
    pub fn generate_proof(&self, key: u64) -> SparseMerkleProof {
        let mut path = Vec::with_capacity(self.depth);
        let value_hash = *self.leaves.get(&key).unwrap_or(&0);

        for level in 0..self.depth {
            let bit = (key >> level) & 1;
            let sibling_prefix = key ^ (1 << level);
            let sibling_hash =
                self.compute_subtree_hash(sibling_prefix & ((1 << (level + 1)) - 1), 0);

            // This is simplified - real impl needs proper sibling computation
            path.push(ProofStep {
                sibling_hash: if sibling_hash == 0 {
                    self.default_hashes[0]
                } else {
                    sibling_hash
                },
                direction: if bit == 0 {
                    ProofDirection::Right
                } else {
                    ProofDirection::Left
                },
            });
        }

        SparseMerkleProof {
            key,
            value_hash,
            path,
            root_hash: self.root_hash,
        }
    }
}

/// Proof for sparse Merkle tree
#[derive(Debug, Clone)]
pub struct SparseMerkleProof {
    /// The key being proved
    pub key: u64,
    /// Hash of the value (0 if non-membership)
    pub value_hash: HashValue,
    /// Path from leaf to root
    pub path: Vec<ProofStep>,
    /// Expected root hash
    pub root_hash: HashValue,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.num_leaves(), 0);
        assert_eq!(tree.root_hash(), None);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn test_single_leaf() {
        let tree = MerkleTree::from_data(&["hello"]);
        assert_eq!(tree.num_leaves(), 1);
        assert!(tree.root_hash().is_some());
        assert_eq!(tree.height(), 1);
    }

    #[test]
    fn test_two_leaves() {
        let tree = MerkleTree::from_data(&["hello", "world"]);
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(tree.height(), 2);

        // Root should be hash of combined children
        let h1 = hash_data(&"hello");
        let h2 = hash_data(&"world");
        let expected_root = combine_hashes(h1, h2);
        assert_eq!(tree.root_hash(), Some(expected_root));
    }

    #[test]
    fn test_power_of_two_leaves() {
        let data: Vec<i32> = (0..8).collect();
        let tree = MerkleTree::from_data(&data);
        assert_eq!(tree.num_leaves(), 8);
        assert_eq!(tree.height(), 4); // log2(8) + 1 = 4
    }

    #[test]
    fn test_non_power_of_two() {
        let data: Vec<i32> = (0..5).collect();
        let tree = MerkleTree::from_data(&data);
        assert_eq!(tree.num_leaves(), 5);
        // Padded to 8, so height is 4
        assert_eq!(tree.height(), 4);
    }

    #[test]
    fn test_proof_generation() {
        let data = vec!["a", "b", "c", "d"];
        let tree = MerkleTree::from_data(&data);

        for i in 0..4 {
            let proof = tree.generate_proof(i);
            assert!(proof.is_some());
            let proof = proof.unwrap();
            assert_eq!(proof.leaf_index, i);
            assert!(proof.verify());
        }
    }

    #[test]
    fn test_proof_invalid_index() {
        let data = vec!["a", "b", "c", "d"];
        let tree = MerkleTree::from_data(&data);

        assert!(tree.generate_proof(4).is_none());
        assert!(tree.generate_proof(100).is_none());
    }

    #[test]
    fn test_proof_size() {
        let data: Vec<i32> = (0..16).collect();
        let tree = MerkleTree::from_data(&data);

        let proof = tree.generate_proof(0).unwrap();
        // log2(16) = 4 steps
        assert_eq!(proof.size(), 4);
    }

    #[test]
    fn test_verify_data() {
        let data = vec!["hello", "world", "foo", "bar"];
        let tree = MerkleTree::from_data(&data);

        assert!(tree.verify(&"hello", 0));
        assert!(tree.verify(&"world", 1));
        assert!(tree.verify(&"foo", 2));
        assert!(tree.verify(&"bar", 3));

        // Wrong data at index
        assert!(!tree.verify(&"wrong", 0));
        // Wrong index
        assert!(!tree.verify(&"hello", 1));
    }

    #[test]
    fn test_tamper_detection() {
        let data1 = vec!["a", "b", "c", "d"];
        let data2 = vec!["a", "b", "c", "e"]; // Changed last element

        let tree1 = MerkleTree::from_data(&data1);
        let tree2 = MerkleTree::from_data(&data2);

        assert_ne!(tree1.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_deterministic() {
        let data = vec![1, 2, 3, 4, 5];
        let tree1 = MerkleTree::from_data(&data);
        let tree2 = MerkleTree::from_data(&data);

        assert_eq!(tree1.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_tree_diff_identical() {
        let data = vec!["a", "b", "c", "d"];
        let tree1 = MerkleTree::from_data(&data);
        let tree2 = MerkleTree::from_data(&data);

        let diff = tree1.diff(&tree2);
        assert!(diff.is_identical());
        assert_eq!(diff.total_differences(), 0);
    }

    #[test]
    fn test_tree_diff_different() {
        let data1 = vec!["a", "b", "c", "d"];
        let data2 = vec!["a", "x", "c", "d"]; // Changed index 1

        let tree1 = MerkleTree::from_data(&data1);
        let tree2 = MerkleTree::from_data(&data2);

        let diff = tree1.diff(&tree2);
        assert!(!diff.is_identical());
        assert!(diff.different.contains(&1));
    }

    #[test]
    fn test_from_hashes() {
        let hashes = vec![1u64, 2, 3, 4];
        let tree = MerkleTree::from_hashes(&hashes);

        assert_eq!(tree.num_leaves(), 4);
        assert_eq!(tree.get_leaf_hash(0), Some(1));
        assert_eq!(tree.get_leaf_hash(3), Some(4));
    }

    #[test]
    fn test_combine_hashes_deterministic() {
        let h1 = combine_hashes(100, 200);
        let h2 = combine_hashes(100, 200);
        assert_eq!(h1, h2);

        // Order matters
        let h3 = combine_hashes(200, 100);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_expected_proof_size() {
        assert_eq!(expected_proof_size(1), 0);
        assert_eq!(expected_proof_size(2), 1);
        assert_eq!(expected_proof_size(4), 2);
        assert_eq!(expected_proof_size(8), 3);
        assert_eq!(expected_proof_size(16), 4);
        assert_eq!(expected_proof_size(5), 3); // ceil(log2(5)) = 3
    }

    #[test]
    fn test_analyze_tree() {
        let data: Vec<i32> = (0..4).collect();
        let tree = MerkleTree::from_data(&data);

        let stats = analyze_tree(&tree);
        assert_eq!(stats.num_leaves, 4);
        assert_eq!(stats.height, 3);
        // 4 leaves + 2 internal + 1 root = 7 nodes
        assert_eq!(stats.total_nodes, 7);
        assert_eq!(stats.empty_nodes, 0);
    }

    #[test]
    fn test_analyze_tree_with_padding() {
        let data: Vec<i32> = (0..3).collect();
        let tree = MerkleTree::from_data(&data);

        let stats = analyze_tree(&tree);
        assert_eq!(stats.num_leaves, 3);
        // Padded to 4, so some empty nodes
        assert!(stats.empty_nodes > 0);
    }

    #[test]
    fn test_audit_path() {
        let data = vec!["a", "b", "c", "d"];
        let tree = MerkleTree::from_data(&data);

        let audit = tree.audit_path(2);
        assert!(audit.is_some());

        let audit = audit.unwrap();
        assert!(audit.verify());
    }

    #[test]
    fn test_consistency_proof() {
        let data: Vec<i32> = (0..8).collect();
        let tree = MerkleTree::from_data(&data);

        // Consistency proof from size 4 to size 8
        let proof = tree.consistency_proof(4);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert_eq!(proof.old_size, 4);
        assert_eq!(proof.new_size, 8);
    }

    #[test]
    fn test_consistency_proof_invalid() {
        let data: Vec<i32> = (0..4).collect();
        let tree = MerkleTree::from_data(&data);

        // Invalid: old size > current size
        assert!(tree.consistency_proof(5).is_none());
        // Invalid: old size = 0
        assert!(tree.consistency_proof(0).is_none());
    }

    #[test]
    fn test_sparse_merkle_tree() {
        let mut smt = SparseMerkleTree::new(8);

        assert!(smt.is_empty());
        let empty_root = smt.root_hash();

        smt.insert(5u64, &"hello");
        assert!(!smt.is_empty());
        assert_eq!(smt.len(), 1);
        assert!(smt.contains(5));
        assert!(!smt.contains(6));

        // Root should change
        assert_ne!(smt.root_hash(), empty_root);
    }

    #[test]
    fn test_sparse_merkle_remove() {
        let mut smt = SparseMerkleTree::new(8);
        let empty_root = smt.root_hash();

        smt.insert(10u64, &"value");
        assert!(smt.contains(10));

        let removed = smt.remove(10);
        assert!(removed);
        assert!(!smt.contains(10));

        // Should return to empty root
        assert_eq!(smt.root_hash(), empty_root);
    }

    #[test]
    fn test_sparse_merkle_multiple_inserts() {
        let mut smt = SparseMerkleTree::new(16);

        for i in 0..10u64 {
            smt.insert(i, &format!("value{}", i));
        }

        assert_eq!(smt.len(), 10);

        for i in 0..10u64 {
            assert!(smt.contains(i));
        }
    }

    #[test]
    fn test_large_tree() {
        let data: Vec<i32> = (0..1000).collect();
        let tree = MerkleTree::from_data(&data);

        assert_eq!(tree.num_leaves(), 1000);

        // Verify a few random proofs
        for i in [0, 100, 500, 999] {
            let proof = tree.generate_proof(i).unwrap();
            assert!(proof.verify());
        }
    }

    #[test]
    fn test_proof_step_direction() {
        let data = vec!["a", "b"];
        let tree = MerkleTree::from_data(&data);

        let proof0 = tree.generate_proof(0).unwrap();
        let proof1 = tree.generate_proof(1).unwrap();

        // For a two-leaf tree, the sibling directions should be opposite
        assert_eq!(proof0.path.len(), 1);
        assert_eq!(proof1.path.len(), 1);
        assert_ne!(proof0.path[0].direction, proof1.path[0].direction);
    }

    #[test]
    fn test_hash_data_consistency() {
        let h1 = hash_data(&"test");
        let h2 = hash_data(&"test");
        let h3 = hash_data(&"different");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
