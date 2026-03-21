// Rendezvous Hashing (Highest Random Weight - HRW)
//
// A distributed hash table algorithm that provides consistent hashing with
// simpler implementation than ring-based consistent hashing.
//
// Key properties:
// - Each key independently selects its server by computing hash(key, server)
//   for all servers and choosing the one with the highest value
// - Minimal disruption: when a server is added/removed, only keys that would
//   map to that server are affected (approximately K/N keys)
// - No virtual nodes needed for load balancing (natural distribution)
// - O(N) lookup time where N is number of servers (vs O(log N) for ring-based)
// - Simpler to implement and reason about than consistent hashing
//
// Algorithm:
// 1. For a given key, compute weight = hash(key || server_id) for each server
// 2. Select the server with the highest weight
// 3. For replication, select the top-N servers by weight
//
// Advantages over ring-based consistent hashing:
// - Simpler implementation (no ring data structure)
// - Better load distribution without virtual nodes
// - Easier to understand and debug
// - Naturally supports weighted servers
//
// Disadvantages:
// - O(N) lookup vs O(log N) for ring-based
// - For very large N, may need optimizations like skeleton-based HRW
//
// Used in: Microsoft's ROME, Twitter's distributed cache, various CDNs

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A node (server) in the rendezvous hash system
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Unique identifier for the node
    pub id: String,
    /// Weight for weighted distribution (higher = more likely to be chosen)
    pub weight: f64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl Node {
    /// Create a new node with default weight of 1.0
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self {
            id: id.into(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a node with specified weight
    pub fn with_weight<S: Into<String>>(id: S, weight: f64) -> Self {
        Self {
            id: id.into(),
            weight,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the node
    pub fn with_metadata<S: Into<String>>(mut self, key: S, value: S) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Configuration for rendezvous hashing
#[derive(Debug, Clone)]
pub struct RendezvousConfig {
    /// Seed for hash function (for reproducibility)
    pub seed: u64,
    /// Whether to use weighted hashing
    pub use_weights: bool,
}

impl Default for RendezvousConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            use_weights: true,
        }
    }
}

impl RendezvousConfig {
    /// Create new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hash seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable weighted hashing
    pub fn use_weights(mut self, use_weights: bool) -> Self {
        self.use_weights = use_weights;
        self
    }
}

/// Statistics for rendezvous hashing operations
#[derive(Debug, Clone, Default)]
pub struct RendezvousStats {
    /// Number of lookups performed
    pub lookups: usize,
    /// Number of node additions
    pub additions: usize,
    /// Number of node removals
    pub removals: usize,
    /// Total hash computations
    pub hash_computations: usize,
}

/// Rendezvous hashing implementation
#[derive(Debug, Clone)]
pub struct RendezvousHash {
    /// Configuration
    config: RendezvousConfig,
    /// All nodes in the system
    nodes: HashMap<String, Node>,
    /// Statistics
    stats: RendezvousStats,
}

impl RendezvousHash {
    /// Create a new rendezvous hash with the given configuration
    pub fn new(config: RendezvousConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            stats: RendezvousStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RendezvousConfig::default())
    }

    /// Add a node to the system
    pub fn add_node(&mut self, node: Node) {
        self.stats.additions += 1;
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node from the system
    pub fn remove_node(&mut self, node_id: &str) -> Option<Node> {
        self.stats.removals += 1;
        self.nodes.remove(node_id)
    }

    /// Update an existing node's weight
    pub fn update_weight(&mut self, node_id: &str, weight: f64) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.weight = weight;
            true
        } else {
            false
        }
    }

    /// Get a node by its ID
    pub fn get_node_by_id(&self, node_id: &str) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Check if there are no nodes
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all nodes
    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Get statistics
    pub fn stats(&self) -> &RendezvousStats {
        &self.stats
    }

    /// Compute the hash weight for a (key, node) pair
    fn compute_weight(&self, key: &str, node: &Node) -> f64 {
        // Combine key and node ID into a single hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.config.seed.hash(&mut hasher);
        key.hash(&mut hasher);
        node.id.hash(&mut hasher);
        let hash = hasher.finish();

        // Convert to f64 in range [0, 1)
        let base_weight = (hash as f64) / (u64::MAX as f64);

        // Apply node weight if enabled
        // Using -ln(random) / weight gives proper weighted distribution
        // Higher weight = smaller value from -ln(x)/w = higher in max selection
        if self.config.use_weights && node.weight > 0.0 {
            // We want higher weights to win more often
            // Use: hash / (1/weight) = hash * weight, scaled appropriately
            // Or use the proper HRW weighted formula: -ln(hash) / weight
            // Smaller -ln(hash)/weight wins when we want max
            // So we use: weight / -ln(hash) for proper weighting (larger = better)
            let ln_val = -((1.0 - base_weight).max(1e-10)).ln();
            node.weight / ln_val
        } else {
            base_weight
        }
    }

    /// Get the node responsible for a key
    pub fn get_node(&mut self, key: &str) -> Option<&Node> {
        self.stats.lookups += 1;

        if self.nodes.is_empty() {
            return None;
        }

        let mut best_node_id: Option<String> = None;
        let mut best_weight = f64::NEG_INFINITY;

        for node in self.nodes.values() {
            self.stats.hash_computations += 1;
            let weight = self.compute_weight(key, node);
            if weight > best_weight {
                best_weight = weight;
                best_node_id = Some(node.id.clone());
            }
        }

        best_node_id.and_then(|id| self.nodes.get(&id))
    }

    /// Get the node ID responsible for a key (without full node lookup)
    pub fn get_node_id(&mut self, key: &str) -> Option<&str> {
        self.stats.lookups += 1;

        if self.nodes.is_empty() {
            return None;
        }

        let mut best_node_id: Option<&str> = None;
        let mut best_weight = f64::NEG_INFINITY;

        for node in self.nodes.values() {
            self.stats.hash_computations += 1;
            let weight = self.compute_weight(key, node);
            if weight > best_weight {
                best_weight = weight;
                best_node_id = Some(&node.id);
            }
        }

        best_node_id
    }

    /// Get multiple nodes for a key (for replication)
    /// Returns nodes sorted by their weight (highest first)
    pub fn get_nodes(&mut self, key: &str, n: usize) -> Vec<&Node> {
        self.stats.lookups += 1;

        if self.nodes.is_empty() || n == 0 {
            return Vec::new();
        }

        // Compute weights for all nodes - collect first to avoid borrow issues
        let node_data: Vec<(String, f64)> = self
            .nodes
            .values()
            .map(|node| (node.id.clone(), node.weight, node.clone()))
            .map(|(id, _weight, node)| (id, self.compute_weight(key, &node)))
            .collect();

        // Update stats
        self.stats.hash_computations += node_data.len();

        // Sort by weight descending
        let mut weighted = node_data;
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top N nodes
        weighted
            .into_iter()
            .take(n)
            .filter_map(|(id, _)| self.nodes.get(&id))
            .collect()
    }

    /// Calculate load distribution for a set of keys
    pub fn calculate_distribution(&mut self, keys: &[&str]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for key in keys {
            if let Some(node_id) = self.get_node_id(key) {
                *distribution.entry(node_id.to_string()).or_insert(0) += 1;
            }
        }

        distribution
    }

    /// Predict how many keys would be affected by adding a new node
    pub fn keys_affected_by_add(&mut self, new_node_id: &str, keys: &[&str]) -> usize {
        // Create a temporary node to test
        let new_node = Node::new(new_node_id);

        keys.iter()
            .filter(|key| {
                // Get current best weight
                let current_best = self
                    .nodes
                    .values()
                    .map(|n| self.compute_weight(key, n))
                    .fold(f64::NEG_INFINITY, f64::max);

                // Would new node win?
                let new_weight = self.compute_weight(key, &new_node);
                new_weight > current_best
            })
            .count()
    }

    /// Predict how many keys would be affected by removing a node
    pub fn keys_affected_by_remove(&mut self, node_id: &str, keys: &[&str]) -> usize {
        if !self.nodes.contains_key(node_id) {
            return 0;
        }

        keys.iter()
            .filter(|key| {
                // Check if this node currently handles the key
                self.get_node_id(key) == Some(node_id)
            })
            .count()
    }
}

/// Simple wrapper for basic rendezvous hashing use cases
pub struct SimpleRendezvousHash {
    inner: RendezvousHash,
}

impl SimpleRendezvousHash {
    /// Create a new simple rendezvous hash
    pub fn new() -> Self {
        Self {
            inner: RendezvousHash::with_defaults(),
        }
    }

    /// Add a server by name
    pub fn add_server(&mut self, name: &str) {
        self.inner.add_node(Node::new(name));
    }

    /// Remove a server by name
    pub fn remove_server(&mut self, name: &str) -> bool {
        self.inner.remove_node(name).is_some()
    }

    /// Get the server for a key
    pub fn get_server(&mut self, key: &str) -> Option<String> {
        self.inner.get_node_id(key).map(|s| s.to_string())
    }

    /// Get multiple servers for replication
    pub fn get_servers(&mut self, key: &str, n: usize) -> Vec<String> {
        self.inner
            .get_nodes(key, n)
            .into_iter()
            .map(|n| n.id.clone())
            .collect()
    }
}

impl Default for SimpleRendezvousHash {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the load imbalance ratio (max/min load)
pub fn calculate_load_imbalance(distribution: &HashMap<String, usize>) -> f64 {
    if distribution.is_empty() {
        return 1.0;
    }

    let values: Vec<usize> = distribution.values().cloned().collect();
    let min = *values.iter().min().unwrap_or(&1) as f64;
    let max = *values.iter().max().unwrap_or(&1) as f64;

    if min > 0.0 { max / min } else { f64::INFINITY }
}

/// Calculate the standard deviation of load distribution
pub fn calculate_load_std_dev(distribution: &HashMap<String, usize>) -> f64 {
    if distribution.is_empty() {
        return 0.0;
    }

    let values: Vec<f64> = distribution.values().map(|&v| v as f64).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

    variance.sqrt()
}

/// Compare rendezvous hashing with consistent hashing and modulo hashing
/// Returns (rendezvous_moves, modulo_moves) when adding a server
pub fn compare_with_modulo_hashing(keys: &[&str], initial_servers: usize) -> (usize, usize) {
    // Rendezvous hashing
    let config = RendezvousConfig::new().use_weights(false);
    let mut rh = RendezvousHash::new(config);

    for i in 0..initial_servers {
        rh.add_node(Node::new(format!("server{}", i)));
    }

    // Get initial assignments
    let initial_assignments: Vec<Option<String>> = keys
        .iter()
        .map(|k| rh.get_node_id(k).map(|s| s.to_string()))
        .collect();

    // Add one more server
    rh.add_node(Node::new(format!("server{}", initial_servers)));

    // Count moves
    let rendezvous_moves = keys
        .iter()
        .zip(initial_assignments.iter())
        .filter(|&(k, old)| rh.get_node_id(k).map(|s| s.to_string()) != *old)
        .count();

    // Modulo hashing
    let hash_key = |key: &str| -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    };

    let modulo_initial: Vec<usize> = keys
        .iter()
        .map(|k| (hash_key(k) % initial_servers as u64) as usize)
        .collect();

    let modulo_moves = keys
        .iter()
        .zip(modulo_initial.iter())
        .filter(|&(k, &old)| {
            let new = (hash_key(k) % (initial_servers + 1) as u64) as usize;
            new != old
        })
        .count();

    (rendezvous_moves, modulo_moves)
}

/// Compute the expected number of keys that would move when adding a node
/// For N nodes and K keys, expected moves ≈ K / (N + 1)
pub fn expected_keys_moved_on_add(num_keys: usize, num_nodes: usize) -> f64 {
    if num_nodes == 0 {
        return num_keys as f64;
    }
    num_keys as f64 / (num_nodes + 1) as f64
}

/// Compute the expected number of keys that would move when removing a node
/// For N nodes and K keys, expected moves ≈ K / N
pub fn expected_keys_moved_on_remove(num_keys: usize, num_nodes: usize) -> f64 {
    if num_nodes == 0 {
        return 0.0;
    }
    num_keys as f64 / num_nodes as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new("server1");
        assert_eq!(node.id, "server1");
        assert_eq!(node.weight, 1.0);
        assert!(node.metadata.is_empty());

        let weighted = Node::with_weight("server2", 2.0);
        assert_eq!(weighted.weight, 2.0);

        let with_meta = Node::new("server3").with_metadata("dc", "us-east");
        assert_eq!(with_meta.metadata.get("dc"), Some(&"us-east".to_string()));
    }

    #[test]
    fn test_config() {
        let config = RendezvousConfig::new().seed(42).use_weights(false);

        assert_eq!(config.seed, 42);
        assert!(!config.use_weights);
    }

    #[test]
    fn test_empty_ring() {
        let mut rh = RendezvousHash::with_defaults();

        assert!(rh.is_empty());
        assert_eq!(rh.num_nodes(), 0);
        assert!(rh.get_node("key").is_none());
        assert!(rh.get_node_id("key").is_none());
    }

    #[test]
    fn test_single_node() {
        let mut rh = RendezvousHash::with_defaults();
        rh.add_node(Node::new("server1"));

        assert_eq!(rh.num_nodes(), 1);
        assert!(!rh.is_empty());

        // All keys should go to the single server
        assert_eq!(
            rh.get_node("key1").map(|n| &n.id),
            Some(&"server1".to_string())
        );
        assert_eq!(
            rh.get_node("key2").map(|n| &n.id),
            Some(&"server1".to_string())
        );
        assert_eq!(
            rh.get_node("anything").map(|n| &n.id),
            Some(&"server1".to_string())
        );
    }

    #[test]
    fn test_add_remove_nodes() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        assert_eq!(rh.num_nodes(), 2);

        let removed = rh.remove_node("server1");
        assert!(removed.is_some());
        assert_eq!(rh.num_nodes(), 1);

        // Remaining node should handle all keys
        assert_eq!(
            rh.get_node("key1").map(|n| &n.id),
            Some(&"server2".to_string())
        );
    }

    #[test]
    fn test_deterministic() {
        let config = RendezvousConfig::new().seed(42);
        let mut rh1 = RendezvousHash::new(config.clone());
        let mut rh2 = RendezvousHash::new(config);

        rh1.add_node(Node::new("a"));
        rh1.add_node(Node::new("b"));
        rh1.add_node(Node::new("c"));

        rh2.add_node(Node::new("a"));
        rh2.add_node(Node::new("b"));
        rh2.add_node(Node::new("c"));

        // Same keys should map to same nodes
        for i in 0..100 {
            let key = format!("key{}", i);
            assert_eq!(rh1.get_node_id(&key), rh2.get_node_id(&key));
        }
    }

    #[test]
    fn test_distribution() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let distribution = rh.calculate_distribution(&keys);

        // All nodes should have some keys
        assert!(distribution.len() == 3);

        // Distribution should be reasonably balanced
        // With 1000 keys and 3 nodes, expect ~333 each
        for count in distribution.values() {
            assert!(*count > 200, "Too few keys: {}", count);
            assert!(*count < 500, "Too many keys: {}", count);
        }
    }

    #[test]
    fn test_weighted_distribution() {
        let mut rh = RendezvousHash::with_defaults();

        // One node has double weight
        rh.add_node(Node::with_weight("server1", 1.0));
        rh.add_node(Node::with_weight("server2", 2.0));

        let keys: Vec<&str> = (0..3000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let distribution = rh.calculate_distribution(&keys);

        let server1_count = *distribution.get("server1").unwrap_or(&0);
        let server2_count = *distribution.get("server2").unwrap_or(&0);

        // Server2 should have roughly double the keys
        // Allow significant variance due to randomness
        assert!(
            server2_count > server1_count,
            "server2 ({}) should have more keys than server1 ({})",
            server2_count,
            server1_count
        );
    }

    #[test]
    fn test_minimal_movement_on_add() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Record initial assignments
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| rh.get_node_id(k).map(|s| s.to_string()))
            .collect();

        // Add a fourth server
        rh.add_node(Node::new("server4"));

        // Count how many keys moved
        let moved: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| rh.get_node_id(k).map(|s| s.to_string()) != *old)
            .count();

        // Should move approximately K/(N+1) = 1000/4 = 250 keys
        assert!(
            moved > 150 && moved < 400,
            "Expected ~250 keys to move, got {}",
            moved
        );
    }

    #[test]
    fn test_minimal_movement_on_remove() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));
        rh.add_node(Node::new("server4"));

        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Record initial assignments
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| rh.get_node_id(k).map(|s| s.to_string()))
            .collect();

        // Remove a server
        rh.remove_node("server2");

        // Count how many keys moved
        let moved: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| rh.get_node_id(k).map(|s| s.to_string()) != *old)
            .count();

        // Keys from removed server should move to others
        // Should be approximately K/N = 1000/4 = 250
        assert!(
            moved > 150 && moved < 400,
            "Expected ~250 keys to move, got {}",
            moved
        );
    }

    #[test]
    fn test_get_multiple_nodes() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));

        let nodes = rh.get_nodes("key1", 2);
        assert_eq!(nodes.len(), 2);

        // Should return different nodes
        assert_ne!(nodes[0].id, nodes[1].id);

        // Asking for more than available
        let all_nodes = rh.get_nodes("key2", 5);
        assert_eq!(all_nodes.len(), 3);
    }

    #[test]
    fn test_replication_order_stability() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("a"));
        rh.add_node(Node::new("b"));
        rh.add_node(Node::new("c"));

        // Same key should always return same order
        let order1: Vec<String> = rh
            .get_nodes("mykey", 3)
            .iter()
            .map(|n| n.id.clone())
            .collect();
        let order2: Vec<String> = rh
            .get_nodes("mykey", 3)
            .iter()
            .map(|n| n.id.clone())
            .collect();

        assert_eq!(order1, order2);
    }

    #[test]
    fn test_compare_with_modulo() {
        let keys: Vec<&str> = (0..1000)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        let (rendezvous_moves, modulo_moves) = compare_with_modulo_hashing(&keys, 3);

        // Rendezvous should move far fewer keys
        assert!(
            rendezvous_moves < modulo_moves,
            "Rendezvous ({}) should move fewer keys than modulo ({})",
            rendezvous_moves,
            modulo_moves
        );

        // Modulo should move most keys when going from 3 to 4 servers
        assert!(
            modulo_moves > 500,
            "Modulo should move most keys, got {}",
            modulo_moves
        );
    }

    #[test]
    fn test_simple_rendezvous_hash() {
        let mut rh = SimpleRendezvousHash::new();

        rh.add_server("server1");
        rh.add_server("server2");
        rh.add_server("server3");

        let server = rh.get_server("mykey");
        assert!(server.is_some());

        let servers = rh.get_servers("mykey", 2);
        assert_eq!(servers.len(), 2);

        assert!(rh.remove_server("server1"));
        assert!(!rh.remove_server("nonexistent"));
    }

    #[test]
    fn test_load_imbalance() {
        let mut distribution = HashMap::new();
        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 200);

        let imbalance = calculate_load_imbalance(&distribution);
        assert!((imbalance - 2.0).abs() < 0.01);

        // Equal distribution
        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 100);
        let imbalance = calculate_load_imbalance(&distribution);
        assert!((imbalance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_load_std_dev() {
        let mut distribution = HashMap::new();
        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 100);

        let std_dev = calculate_load_std_dev(&distribution);
        assert!(
            std_dev.abs() < 0.01,
            "Equal distribution should have ~0 std dev"
        );

        distribution.insert("a".to_string(), 100);
        distribution.insert("b".to_string(), 200);
        let std_dev = calculate_load_std_dev(&distribution);
        assert!(
            std_dev > 0.0,
            "Unequal distribution should have non-zero std dev"
        );
    }

    #[test]
    fn test_stats() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));

        let _ = rh.get_node("key1");
        let _ = rh.get_node("key2");
        let _ = rh.get_node("key3");

        let stats = rh.stats();
        assert_eq!(stats.additions, 2);
        assert_eq!(stats.lookups, 3);
        assert!(stats.hash_computations >= 6); // At least 2 per lookup
    }

    #[test]
    fn test_update_weight() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::with_weight("server1", 1.0));
        assert!(rh.update_weight("server1", 5.0));
        assert!(!rh.update_weight("nonexistent", 5.0));

        let node = rh.get_node_by_id("server1").unwrap();
        assert_eq!(node.weight, 5.0);
    }

    #[test]
    fn test_all_nodes_iterator() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("a"));
        rh.add_node(Node::new("b"));
        rh.add_node(Node::new("c"));

        let ids: Vec<_> = rh.all_nodes().map(|n| n.id.as_str()).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_keys_affected_prediction_add() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..100)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Predict affected keys
        let predicted = rh.keys_affected_by_add("server4", &keys);

        // Actually add and verify
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| rh.get_node_id(k).map(|s| s.to_string()))
            .collect();
        rh.add_node(Node::new("server4"));
        let actual: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| rh.get_node_id(k).map(|s| s.to_string()).as_ref() != old.as_ref())
            .count();

        assert_eq!(predicted, actual);
    }

    #[test]
    fn test_keys_affected_prediction_remove() {
        let mut rh = RendezvousHash::with_defaults();

        rh.add_node(Node::new("server1"));
        rh.add_node(Node::new("server2"));
        rh.add_node(Node::new("server3"));

        let keys: Vec<&str> = (0..100)
            .map(|i| Box::leak(format!("key{}", i).into_boxed_str()) as &str)
            .collect();

        // Count keys that go to server2
        let predicted = rh.keys_affected_by_remove("server2", &keys);

        // Actually remove and verify
        let initial: Vec<Option<String>> = keys
            .iter()
            .map(|k| rh.get_node_id(k).map(|s| s.to_string()))
            .collect();
        rh.remove_node("server2");
        let actual: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(k, old)| rh.get_node_id(k).map(|s| s.to_string()).as_ref() != old.as_ref())
            .count();

        assert_eq!(predicted, actual);
    }

    #[test]
    fn test_expected_moves() {
        let expected_add = expected_keys_moved_on_add(1000, 3);
        assert!((expected_add - 250.0).abs() < 1.0); // 1000 / 4 = 250

        let expected_remove = expected_keys_moved_on_remove(1000, 4);
        assert!((expected_remove - 250.0).abs() < 1.0); // 1000 / 4 = 250

        // Edge case: no nodes
        assert_eq!(expected_keys_moved_on_add(100, 0), 100.0);
        assert_eq!(expected_keys_moved_on_remove(100, 0), 0.0);
    }

    #[test]
    fn test_different_seeds_different_distribution() {
        let config1 = RendezvousConfig::new().seed(1);
        let config2 = RendezvousConfig::new().seed(2);

        let mut rh1 = RendezvousHash::new(config1);
        let mut rh2 = RendezvousHash::new(config2);

        for i in 0..3 {
            rh1.add_node(Node::new(format!("server{}", i)));
            rh2.add_node(Node::new(format!("server{}", i)));
        }

        // Different seeds should produce different distributions
        let mut different = 0;
        for i in 0..100 {
            let key = format!("key{}", i);
            if rh1.get_node_id(&key) != rh2.get_node_id(&key) {
                different += 1;
            }
        }

        assert!(
            different > 0,
            "Different seeds should produce different distributions"
        );
    }
}
