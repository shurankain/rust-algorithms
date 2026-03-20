// Consistent Hashing
//
// Paper: "Consistent Hashing and Random Trees" by Karger et al. (1997)
//
// Key insight: Map both servers and keys to a circular hash space.
// Keys are assigned to the first server found when walking clockwise from the key's position.
//
// How it works:
// 1. Hash each server to one or more positions on a ring [0, 2^32)
// 2. Hash each key to a position on the same ring
// 3. Walk clockwise from the key's position to find the first server
// 4. That server is responsible for the key
//
// Benefits:
// - When a server is added/removed, only K/N keys need to move on average
//   (K = total keys, N = total servers)
// - Standard hashing would require rehashing most keys
// - Used in: Memcached, Redis Cluster, DynamoDB, Cassandra, CDNs
//
// Virtual nodes:
// - Each physical server maps to multiple positions on the ring
// - Improves load distribution when servers have unequal capacity
// - Provides better load balance with few physical servers

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A node (server) in the consistent hash ring
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    /// Unique identifier for the node
    pub id: String,
    /// Optional weight for weighted distribution (higher = more virtual nodes)
    pub weight: u32,
}

impl Node {
    /// Create a new node with default weight
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            weight: 1,
        }
    }

    /// Create a new node with specified weight
    pub fn with_weight(id: impl Into<String>, weight: u32) -> Self {
        Self {
            id: id.into(),
            weight: weight.max(1),
        }
    }
}

/// Configuration for the consistent hash ring
#[derive(Debug, Clone)]
pub struct ConsistentHashConfig {
    /// Number of virtual nodes per physical node (per unit weight)
    pub virtual_nodes: usize,
    /// Hash function seed
    pub seed: u64,
}

impl ConsistentHashConfig {
    /// Create a new config
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            virtual_nodes: virtual_nodes.max(1),
            seed: 0,
        }
    }

    /// Set the hash seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for ConsistentHashConfig {
    fn default() -> Self {
        Self::new(150) // Common default: 100-200 virtual nodes per physical node
    }
}

/// Statistics about the consistent hash ring
#[derive(Debug, Clone, Default)]
pub struct ConsistentHashStats {
    /// Number of physical nodes
    pub physical_nodes: usize,
    /// Total number of virtual nodes on the ring
    pub virtual_nodes: usize,
    /// Number of lookups performed
    pub lookups: usize,
    /// Number of times nodes were added
    pub nodes_added: usize,
    /// Number of times nodes were removed
    pub nodes_removed: usize,
}

/// A consistent hash ring
#[derive(Debug)]
pub struct ConsistentHash {
    config: ConsistentHashConfig,
    /// Maps ring positions to node IDs
    ring: BTreeMap<u64, String>,
    /// Maps node IDs to their physical nodes
    nodes: BTreeMap<String, Node>,
    /// Statistics
    stats: ConsistentHashStats,
}

impl ConsistentHash {
    /// Create a new consistent hash ring
    pub fn new(config: ConsistentHashConfig) -> Self {
        Self {
            config,
            ring: BTreeMap::new(),
            nodes: BTreeMap::new(),
            stats: ConsistentHashStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ConsistentHashConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &ConsistentHashConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsistentHashStats {
        &self.stats
    }

    /// Number of physical nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of virtual nodes on the ring
    pub fn num_virtual_nodes(&self) -> usize {
        self.ring.len()
    }

    /// Check if ring is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Hash a value to a position on the ring
    fn hash_position(&self, key: &str, replica: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.config.seed.hash(&mut hasher);
        key.hash(&mut hasher);
        replica.hash(&mut hasher);
        hasher.finish()
    }

    /// Add a node to the ring
    pub fn add_node(&mut self, node: Node) {
        if self.nodes.contains_key(&node.id) {
            // Node already exists, remove first to update
            self.remove_node(&node.id.clone());
        }

        let num_replicas = self.config.virtual_nodes * node.weight as usize;

        for i in 0..num_replicas {
            let position = self.hash_position(&node.id, i);
            self.ring.insert(position, node.id.clone());
        }

        self.nodes.insert(node.id.clone(), node);
        self.stats.physical_nodes = self.nodes.len();
        self.stats.virtual_nodes = self.ring.len();
        self.stats.nodes_added += 1;
    }

    /// Remove a node from the ring
    pub fn remove_node(&mut self, node_id: &str) -> Option<Node> {
        let node = self.nodes.remove(node_id)?;
        let num_replicas = self.config.virtual_nodes * node.weight as usize;

        for i in 0..num_replicas {
            let position = self.hash_position(node_id, i);
            self.ring.remove(&position);
        }

        self.stats.physical_nodes = self.nodes.len();
        self.stats.virtual_nodes = self.ring.len();
        self.stats.nodes_removed += 1;

        Some(node)
    }

    /// Get the node responsible for a key
    pub fn get_node(&mut self, key: &str) -> Option<&Node> {
        if self.ring.is_empty() {
            return None;
        }

        self.stats.lookups += 1;

        let hash = {
            let mut hasher = DefaultHasher::new();
            self.config.seed.hash(&mut hasher);
            key.hash(&mut hasher);
            hasher.finish()
        };

        // Find the first node at or after this position
        // If none found, wrap around to the first node
        let node_id = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id.clone())?;

        self.nodes.get(&node_id)
    }

    /// Get the node ID responsible for a key (without stats update)
    pub fn get_node_id(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = {
            let mut hasher = DefaultHasher::new();
            self.config.seed.hash(&mut hasher);
            key.hash(&mut hasher);
            hasher.finish()
        };

        let node_id = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id.as_str())?;

        Some(node_id)
    }

    /// Get N nodes for a key (for replication)
    /// Returns up to N distinct physical nodes
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<&Node> {
        if self.ring.is_empty() || n == 0 {
            return vec![];
        }

        let hash = {
            let mut hasher = DefaultHasher::new();
            self.config.seed.hash(&mut hasher);
            key.hash(&mut hasher);
            hasher.finish()
        };

        let mut result = Vec::with_capacity(n.min(self.nodes.len()));
        let mut seen = std::collections::HashSet::new();

        // Iterate from hash position, wrapping around
        let after = self.ring.range(hash..).map(|(_, id)| id);
        let before = self.ring.range(..hash).map(|(_, id)| id);

        for node_id in after.chain(before) {
            if seen.insert(node_id)
                && let Some(node) = self.nodes.get(node_id)
            {
                result.push(node);
                if result.len() >= n {
                    break;
                }
            }
        }

        result
    }

    /// Get all nodes in the ring
    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Calculate load distribution across nodes for a set of keys
    pub fn calculate_distribution(
        &self,
        keys: &[&str],
    ) -> std::collections::HashMap<String, usize> {
        let mut distribution = std::collections::HashMap::new();

        for key in keys {
            if let Some(node_id) = self.get_node_id(key) {
                *distribution.entry(node_id.to_string()).or_insert(0) += 1;
            }
        }

        distribution
    }

    /// Calculate how many keys would need to move if a node is added
    pub fn keys_affected_by_add(&self, node_id: &str, keys: &[&str]) -> usize {
        let mut temp_ring = self.clone();
        temp_ring.add_node(Node::new(node_id));

        let mut affected = 0;
        for key in keys {
            let old_node = self.get_node_id(key);
            let new_node = temp_ring.get_node_id(key);

            if old_node != new_node {
                affected += 1;
            }
        }

        affected
    }

    /// Calculate how many keys would need to move if a node is removed
    pub fn keys_affected_by_remove(&self, node_id: &str, keys: &[&str]) -> usize {
        let mut temp_ring = self.clone();
        temp_ring.remove_node(node_id);

        let mut affected = 0;
        for key in keys {
            let old_node = self.get_node_id(key);
            let new_node = temp_ring.get_node_id(key);

            if old_node != new_node {
                affected += 1;
            }
        }

        affected
    }
}

impl Clone for ConsistentHash {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            ring: self.ring.clone(),
            nodes: self.nodes.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Simple consistent hashing without virtual nodes (for comparison)
pub struct SimpleConsistentHash {
    ring: BTreeMap<u64, String>,
    seed: u64,
}

impl SimpleConsistentHash {
    /// Create a new simple consistent hash ring
    pub fn new() -> Self {
        Self {
            ring: BTreeMap::new(),
            seed: 0,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node_id: &str) {
        let hash = {
            let mut hasher = DefaultHasher::new();
            self.seed.hash(&mut hasher);
            node_id.hash(&mut hasher);
            hasher.finish()
        };
        self.ring.insert(hash, node_id.to_string());
    }

    /// Get node for key
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = {
            let mut hasher = DefaultHasher::new();
            self.seed.hash(&mut hasher);
            key.hash(&mut hasher);
            hasher.finish()
        };

        self.ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id.as_str())
    }
}

impl Default for SimpleConsistentHash {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the standard deviation of load across nodes
pub fn calculate_load_std_dev(distribution: &std::collections::HashMap<String, usize>) -> f64 {
    if distribution.is_empty() {
        return 0.0;
    }

    let values: Vec<f64> = distribution.values().map(|&v| v as f64).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance.sqrt()
}

/// Calculate load imbalance ratio (max/min)
pub fn calculate_load_imbalance(distribution: &std::collections::HashMap<String, usize>) -> f64 {
    if distribution.is_empty() {
        return 1.0;
    }

    let values: Vec<usize> = distribution.values().cloned().collect();
    let min = *values.iter().min().unwrap_or(&1) as f64;
    let max = *values.iter().max().unwrap_or(&1) as f64;

    if min > 0.0 { max / min } else { f64::INFINITY }
}

/// Compare consistent hashing with standard modulo hashing
/// Returns (consistent_moves, modulo_moves) when changing from n to n+1 servers
pub fn compare_with_modulo_hashing(keys: &[&str], initial_servers: usize) -> (usize, usize) {
    // Consistent hashing
    let config = ConsistentHashConfig::new(100);
    let mut ch = ConsistentHash::new(config);

    for i in 0..initial_servers {
        ch.add_node(Node::new(format!("server{}", i)));
    }

    // Get initial assignments (collect to owned strings to avoid borrowing issues)
    let initial_assignments: Vec<Option<String>> = keys
        .iter()
        .map(|k| ch.get_node_id(k).map(|s| s.to_string()))
        .collect();

    // Add one more server
    ch.add_node(Node::new(format!("server{}", initial_servers)));

    // Count moves
    let consistent_moves = keys
        .iter()
        .zip(initial_assignments.iter())
        .filter(|&(k, old)| ch.get_node_id(k).map(|s| s.to_string()) != *old)
        .count();

    // Modulo hashing
    let hash_key = |key: &str| -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    };

    let modulo_moves = keys
        .iter()
        .filter(|k| {
            let h = hash_key(k);
            (h % initial_servers as u64) != (h % (initial_servers + 1) as u64)
        })
        .count();

    (consistent_moves, modulo_moves)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new("server1");
        assert_eq!(node.id, "server1");
        assert_eq!(node.weight, 1);

        let weighted = Node::with_weight("server2", 3);
        assert_eq!(weighted.id, "server2");
        assert_eq!(weighted.weight, 3);
    }

    #[test]
    fn test_config_default() {
        let config = ConsistentHashConfig::default();
        assert_eq!(config.virtual_nodes, 150);
    }

    #[test]
    fn test_config_builder() {
        let config = ConsistentHashConfig::new(100).seed(12345);
        assert_eq!(config.virtual_nodes, 100);
        assert_eq!(config.seed, 12345);
    }

    #[test]
    fn test_empty_ring() {
        let mut ch = ConsistentHash::with_defaults();
        assert!(ch.is_empty());
        assert_eq!(ch.num_nodes(), 0);
        assert!(ch.get_node("key1").is_none());
    }

    #[test]
    fn test_single_node() {
        let mut ch = ConsistentHash::with_defaults();
        ch.add_node(Node::new("server1"));

        assert_eq!(ch.num_nodes(), 1);
        assert!(!ch.is_empty());

        // All keys should go to the single server
        assert_eq!(
            ch.get_node("key1").map(|n| &n.id),
            Some(&"server1".to_string())
        );
        assert_eq!(
            ch.get_node("key2").map(|n| &n.id),
            Some(&"server1".to_string())
        );
        assert_eq!(
            ch.get_node("anything").map(|n| &n.id),
            Some(&"server1".to_string())
        );
    }

    #[test]
    fn test_add_remove_nodes() {
        let mut ch = ConsistentHash::with_defaults();

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        assert_eq!(ch.num_nodes(), 2);

        let removed = ch.remove_node("server1");
        assert!(removed.is_some());
        assert_eq!(ch.num_nodes(), 1);

        // Remaining node should handle all keys
        assert_eq!(
            ch.get_node("key1").map(|n| &n.id),
            Some(&"server2".to_string())
        );
    }

    #[test]
    fn test_deterministic() {
        let config = ConsistentHashConfig::new(100).seed(42);
        let mut ch1 = ConsistentHash::new(config.clone());
        let mut ch2 = ConsistentHash::new(config);

        ch1.add_node(Node::new("a"));
        ch1.add_node(Node::new("b"));

        ch2.add_node(Node::new("a"));
        ch2.add_node(Node::new("b"));

        // Same key should map to same node
        for key in &["key1", "key2", "key3", "test", "hello"] {
            assert_eq!(ch1.get_node_id(key), ch2.get_node_id(key));
        }
    }

    #[test]
    fn test_distribution() {
        let config = ConsistentHashConfig::new(150);
        let mut ch = ConsistentHash::new(config);

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        ch.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let distribution = ch.calculate_distribution(&keys);

        // All servers should get some keys
        assert_eq!(distribution.len(), 3);

        // Distribution should be reasonably balanced (within 50% of ideal)
        let ideal = 1000.0 / 3.0;
        for &count in distribution.values() {
            assert!(
                (count as f64) > ideal * 0.5,
                "Server got {} keys, expected around {}",
                count,
                ideal
            );
            assert!(
                (count as f64) < ideal * 1.5,
                "Server got {} keys, expected around {}",
                count,
                ideal
            );
        }
    }

    #[test]
    fn test_weighted_nodes() {
        let config = ConsistentHashConfig::new(100);
        let mut ch = ConsistentHash::new(config);

        // Server2 has 3x the weight
        ch.add_node(Node::with_weight("server1", 1));
        ch.add_node(Node::with_weight("server2", 3));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let distribution = ch.calculate_distribution(&keys);

        let count1 = *distribution.get("server1").unwrap_or(&0);
        let count2 = *distribution.get("server2").unwrap_or(&0);

        // Server2 should get roughly 3x the keys (with some variance)
        let ratio = count2 as f64 / count1 as f64;
        assert!(
            ratio > 2.0 && ratio < 4.0,
            "Expected ratio around 3, got {}",
            ratio
        );
    }

    #[test]
    fn test_get_nodes_for_replication() {
        let mut ch = ConsistentHash::with_defaults();

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        ch.add_node(Node::new("server3"));

        let replicas = ch.get_nodes("test_key", 2);
        assert_eq!(replicas.len(), 2);

        // Should be distinct nodes
        assert_ne!(replicas[0].id, replicas[1].id);

        // Requesting more than available
        let all = ch.get_nodes("test_key", 5);
        assert_eq!(all.len(), 3); // Only 3 nodes exist
    }

    #[test]
    fn test_minimal_movement_on_add() {
        let config = ConsistentHashConfig::new(100);
        let mut ch = ConsistentHash::new(config);

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        ch.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Record initial assignments (clone to owned strings)
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| ch.get_node_id(k).map(|s| s.to_string()))
            .collect();

        // Add a fourth server
        ch.add_node(Node::new("server4"));

        // Count how many keys moved
        let moved: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| ch.get_node_id(k).map(|s| s.to_string()) != *old)
            .count();

        // Should move approximately K/N keys (250 in this case)
        // Allow some variance
        assert!(
            moved > 150 && moved < 400,
            "Expected ~250 keys to move, got {}",
            moved
        );
    }

    #[test]
    fn test_minimal_movement_on_remove() {
        let config = ConsistentHashConfig::new(100);
        let mut ch = ConsistentHash::new(config);

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        ch.add_node(Node::new("server3"));
        ch.add_node(Node::new("server4"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Record initial assignments (clone to owned strings)
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| ch.get_node_id(k).map(|s| s.to_string()))
            .collect();

        // Remove a server
        ch.remove_node("server2");

        // Count how many keys moved
        let moved: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| ch.get_node_id(k).map(|s| s.to_string()) != *old)
            .count();

        // Keys from removed server should move to others
        // Should be approximately K/N keys (250)
        assert!(
            moved > 150 && moved < 400,
            "Expected ~250 keys to move, got {}",
            moved
        );
    }

    #[test]
    fn test_compare_with_modulo() {
        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let (consistent_moves, modulo_moves) = compare_with_modulo_hashing(&keys, 5);

        // Consistent hashing should move far fewer keys
        assert!(
            consistent_moves < modulo_moves,
            "Consistent: {}, Modulo: {}",
            consistent_moves,
            modulo_moves
        );

        // Modulo should move most keys (roughly 5/6 of them)
        assert!(modulo_moves > 700);

        // Consistent should move roughly 1/6 (adding 6th server)
        assert!(consistent_moves < 300);
    }

    #[test]
    fn test_stats() {
        let mut ch = ConsistentHash::with_defaults();

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));

        assert_eq!(ch.stats().nodes_added, 2);

        ch.get_node("key1");
        ch.get_node("key2");
        ch.get_node("key3");

        assert_eq!(ch.stats().lookups, 3);

        ch.remove_node("server1");
        assert_eq!(ch.stats().nodes_removed, 1);
    }

    #[test]
    fn test_virtual_nodes_count() {
        let config = ConsistentHashConfig::new(100);
        let mut ch = ConsistentHash::new(config);

        ch.add_node(Node::new("server1"));
        assert_eq!(ch.num_virtual_nodes(), 100);

        ch.add_node(Node::with_weight("server2", 2));
        assert_eq!(ch.num_virtual_nodes(), 300); // 100 + 200
    }

    #[test]
    fn test_load_std_dev() {
        let mut distribution = std::collections::HashMap::new();
        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 100);
        distribution.insert("c".to_string(), 100);

        let std_dev = calculate_load_std_dev(&distribution);
        assert!(std_dev < 0.001); // Perfectly balanced

        distribution.insert("a".to_string(), 50);
        distribution.insert("b".to_string(), 100);
        distribution.insert("c".to_string(), 150);

        let std_dev = calculate_load_std_dev(&distribution);
        assert!(std_dev > 40.0); // Imbalanced
    }

    #[test]
    fn test_load_imbalance() {
        let mut distribution = std::collections::HashMap::new();
        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 100);

        let imbalance = calculate_load_imbalance(&distribution);
        assert!((imbalance - 1.0).abs() < 0.001);

        distribution.insert("a".to_string(), 50);
        distribution.insert("b".to_string(), 150);

        let imbalance = calculate_load_imbalance(&distribution);
        assert!((imbalance - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_simple_consistent_hash() {
        let mut sch = SimpleConsistentHash::new();

        sch.add_node("server1");
        sch.add_node("server2");

        // Should deterministically route keys
        let node1 = sch.get_node("key1");
        let node2 = sch.get_node("key1");
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_keys_affected_prediction() {
        let config = ConsistentHashConfig::new(100);
        let mut ch = ConsistentHash::new(config);

        ch.add_node(Node::new("server1"));
        ch.add_node(Node::new("server2"));
        ch.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..100)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Predict affected keys
        let predicted = ch.keys_affected_by_add("server4", &keys);

        // Actually add and verify
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| ch.get_node_id(k).map(|s| s.to_string()))
            .collect();
        ch.add_node(Node::new("server4"));
        let actual: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| ch.get_node_id(k).map(|s| s.to_string()).as_ref() != old.as_ref())
            .count();

        assert_eq!(predicted, actual);
    }

    #[test]
    fn test_all_nodes_iterator() {
        let mut ch = ConsistentHash::with_defaults();

        ch.add_node(Node::new("a"));
        ch.add_node(Node::new("b"));
        ch.add_node(Node::new("c"));

        let nodes: Vec<_> = ch.all_nodes().map(|n| n.id.as_str()).collect();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&"a"));
        assert!(nodes.contains(&"b"));
        assert!(nodes.contains(&"c"));
    }

    #[test]
    fn test_update_existing_node() {
        let mut ch = ConsistentHash::with_defaults();

        ch.add_node(Node::with_weight("server1", 1));
        let initial_vnodes = ch.num_virtual_nodes();

        // Update with higher weight
        ch.add_node(Node::with_weight("server1", 2));

        // Should still have 1 physical node
        assert_eq!(ch.num_nodes(), 1);

        // But more virtual nodes
        assert!(ch.num_virtual_nodes() > initial_vnodes);
    }
}
