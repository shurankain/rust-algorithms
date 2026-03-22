// Jump Consistent Hash
//
// A fast, minimal-memory consistent hash algorithm developed at Google.
// Published in "A Fast, Minimal Memory, Consistent Hash Algorithm" (2014)
// by John Lamping and Eric Veach.
//
// Key properties:
// - O(1) memory: no data structures needed, just computation
// - O(ln n) time: expected number of jumps is ln(n)
// - Perfect balance: maps keys uniformly across buckets
// - Minimal disruption: when changing from n to n+1 buckets, only 1/(n+1) keys move
// - Monotonic: if a key maps to bucket b with n buckets, it maps to b or higher with n+1 buckets
//
// Algorithm intuition:
// - For each bucket b from 0 to n-1, we "flip a coin" to decide if we should "jump" to a higher bucket
// - The probability of jumping from bucket b is 1/(b+1)
// - This gives us the property that exactly 1/(n+1) keys move when adding bucket n
//
// Limitations:
// - Only supports numbered buckets (0 to n-1), not named servers
// - Cannot remove arbitrary buckets (only works with contiguous range)
// - For named/weighted servers, use consistent hashing or rendezvous hashing
//
// Best use cases:
// - Sharding data across numbered partitions
// - Load balancing when servers are numbered
// - Any case where buckets are added sequentially and rarely removed

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Jump consistent hash function
/// Maps a key to a bucket in the range [0, num_buckets)
///
/// # Arguments
/// * `key` - The 64-bit key to hash
/// * `num_buckets` - Number of buckets (must be > 0)
///
/// # Returns
/// A bucket number in [0, num_buckets)
///
/// # Panics
/// Panics if num_buckets is 0
pub fn jump_hash(key: u64, num_buckets: u32) -> u32 {
    assert!(num_buckets > 0, "num_buckets must be > 0");

    let mut b: i64 = -1;
    let mut j: i64 = 0;
    let mut key = key;

    while j < num_buckets as i64 {
        b = j;
        // Linear congruential generator for pseudo-random sequence
        key = key.wrapping_mul(2862933555777941757).wrapping_add(1);
        // The magic: probability of jumping from bucket b to a higher bucket
        // is (b+1)/(new_bucket), which gives uniform distribution
        j = ((b + 1) as f64 * (1i64 << 31) as f64 / ((key >> 33) + 1) as f64) as i64;
    }

    b as u32
}

/// Jump hash for any hashable key type
/// Converts the key to u64 using std hash, then applies jump_hash
pub fn jump_hash_key<K: Hash>(key: &K, num_buckets: u32) -> u32 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    let hash = hasher.finish();
    jump_hash(hash, num_buckets)
}

/// Jump hash with a seed for reproducibility
pub fn jump_hash_seeded(key: u64, num_buckets: u32, seed: u64) -> u32 {
    jump_hash(key ^ seed, num_buckets)
}

/// Statistics for jump hash operations
#[derive(Debug, Clone, Default)]
pub struct JumpHashStats {
    /// Total number of hash operations
    pub hash_operations: usize,
    /// Total jumps across all operations (for analyzing algorithm behavior)
    pub total_jumps: usize,
}

/// Wrapper for jump consistent hash with statistics tracking
#[derive(Debug, Clone)]
pub struct JumpConsistentHash {
    /// Number of buckets
    num_buckets: u32,
    /// Optional seed for reproducibility
    seed: u64,
    /// Statistics
    stats: JumpHashStats,
}

impl JumpConsistentHash {
    /// Create a new jump consistent hash with the given number of buckets
    pub fn new(num_buckets: u32) -> Self {
        assert!(num_buckets > 0, "num_buckets must be > 0");
        Self {
            num_buckets,
            seed: 0,
            stats: JumpHashStats::default(),
        }
    }

    /// Create with a specific seed
    pub fn with_seed(num_buckets: u32, seed: u64) -> Self {
        assert!(num_buckets > 0, "num_buckets must be > 0");
        Self {
            num_buckets,
            seed,
            stats: JumpHashStats::default(),
        }
    }

    /// Get the number of buckets
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Set the number of buckets
    pub fn set_num_buckets(&mut self, num_buckets: u32) {
        assert!(num_buckets > 0, "num_buckets must be > 0");
        self.num_buckets = num_buckets;
    }

    /// Add a bucket (increment bucket count)
    pub fn add_bucket(&mut self) {
        self.num_buckets += 1;
    }

    /// Remove the last bucket (decrement bucket count)
    /// Returns false if only one bucket remains
    pub fn remove_bucket(&mut self) -> bool {
        if self.num_buckets > 1 {
            self.num_buckets -= 1;
            true
        } else {
            false
        }
    }

    /// Get the bucket for a u64 key
    pub fn get_bucket(&mut self, key: u64) -> u32 {
        self.stats.hash_operations += 1;
        jump_hash_seeded(key, self.num_buckets, self.seed)
    }

    /// Get the bucket for any hashable key
    pub fn get_bucket_for_key<K: Hash>(&mut self, key: &K) -> u32 {
        self.stats.hash_operations += 1;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        jump_hash_seeded(hash, self.num_buckets, self.seed)
    }

    /// Get statistics
    pub fn stats(&self) -> &JumpHashStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = JumpHashStats::default();
    }

    /// Calculate distribution of keys across buckets
    pub fn calculate_distribution(&mut self, keys: &[u64]) -> HashMap<u32, usize> {
        let mut distribution = HashMap::new();
        for &key in keys {
            let bucket = self.get_bucket(key);
            *distribution.entry(bucket).or_insert(0) += 1;
        }
        distribution
    }

    /// Predict how many keys would move when adding a bucket
    pub fn keys_affected_by_add(&self, keys: &[u64]) -> usize {
        keys.iter()
            .filter(|&&key| {
                let current = jump_hash_seeded(key, self.num_buckets, self.seed);
                let after = jump_hash_seeded(key, self.num_buckets + 1, self.seed);
                current != after
            })
            .count()
    }

    /// Predict how many keys would move when removing a bucket
    pub fn keys_affected_by_remove(&self, keys: &[u64]) -> usize {
        if self.num_buckets <= 1 {
            return 0;
        }
        keys.iter()
            .filter(|&&key| {
                let current = jump_hash_seeded(key, self.num_buckets, self.seed);
                let after = jump_hash_seeded(key, self.num_buckets - 1, self.seed);
                current != after
            })
            .count()
    }
}

/// Calculate the load imbalance ratio (max/min load)
pub fn calculate_load_imbalance(distribution: &HashMap<u32, usize>) -> f64 {
    if distribution.is_empty() {
        return 1.0;
    }

    let values: Vec<usize> = distribution.values().cloned().collect();
    let min = *values.iter().min().unwrap_or(&1) as f64;
    let max = *values.iter().max().unwrap_or(&1) as f64;

    if min > 0.0 { max / min } else { f64::INFINITY }
}

/// Calculate standard deviation of load distribution
pub fn calculate_load_std_dev(distribution: &HashMap<u32, usize>) -> f64 {
    if distribution.is_empty() {
        return 0.0;
    }

    let values: Vec<f64> = distribution.values().map(|&v| v as f64).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

    variance.sqrt()
}

/// Compare jump hash with modulo hashing
/// Returns (jump_moves, modulo_moves) when changing bucket count
pub fn compare_with_modulo_hashing(keys: &[u64], initial_buckets: u32) -> (usize, usize) {
    // Jump hash
    let jump_initial: Vec<u32> = keys
        .iter()
        .map(|&k| jump_hash(k, initial_buckets))
        .collect();

    let jump_moves = keys
        .iter()
        .zip(jump_initial.iter())
        .filter(|&(&k, &old)| jump_hash(k, initial_buckets + 1) != old)
        .count();

    // Modulo hash
    let modulo_initial: Vec<u32> = keys
        .iter()
        .map(|&k| (k % initial_buckets as u64) as u32)
        .collect();

    let modulo_moves = keys
        .iter()
        .zip(modulo_initial.iter())
        .filter(|&(&k, &old)| (k % (initial_buckets + 1) as u64) as u32 != old)
        .count();

    (jump_moves, modulo_moves)
}

/// Verify the monotonicity property of jump hash
/// When adding buckets, keys only move to the new bucket or stay put
pub fn verify_monotonicity(key: u64, max_buckets: u32) -> bool {
    let mut prev_bucket = 0u32;
    for n in 1..=max_buckets {
        let bucket = jump_hash(key, n);
        // Bucket should either stay the same or be the new bucket (n-1)
        if bucket != prev_bucket && bucket != n - 1 {
            return false;
        }
        prev_bucket = bucket;
    }
    true
}

/// Calculate expected keys moved when changing from n to n+1 buckets
/// Theoretical: K/(n+1) keys should move
pub fn expected_keys_moved(num_keys: usize, current_buckets: u32) -> f64 {
    num_keys as f64 / (current_buckets + 1) as f64
}

/// Analyze the jump pattern for a key across bucket counts
/// Returns a vector of (num_buckets, assigned_bucket) pairs where assignment changed
pub fn analyze_jump_pattern(key: u64, max_buckets: u32) -> Vec<(u32, u32)> {
    let mut jumps = Vec::new();
    let mut prev = 0;

    for n in 1..=max_buckets {
        let bucket = jump_hash(key, n);
        if bucket != prev {
            jumps.push((n, bucket));
            prev = bucket;
        }
    }

    jumps
}

/// Virtual bucket mapping for named servers
/// Maps jump hash buckets to server names
#[derive(Debug, Clone)]
pub struct JumpHashRouter {
    /// Ordered list of server names
    servers: Vec<String>,
    /// Inner jump hash
    hasher: JumpConsistentHash,
}

impl JumpHashRouter {
    /// Create a new router with the given servers
    pub fn new(servers: Vec<String>) -> Self {
        let num_buckets = servers.len().max(1) as u32;
        Self {
            servers,
            hasher: JumpConsistentHash::new(num_buckets),
        }
    }

    /// Create with a seed
    pub fn with_seed(servers: Vec<String>, seed: u64) -> Self {
        let num_buckets = servers.len().max(1) as u32;
        Self {
            servers,
            hasher: JumpConsistentHash::with_seed(num_buckets, seed),
        }
    }

    /// Get the number of servers
    pub fn num_servers(&self) -> usize {
        self.servers.len()
    }

    /// Add a server at the end
    pub fn add_server(&mut self, name: String) {
        self.servers.push(name);
        self.hasher.add_bucket();
    }

    /// Remove the last server
    /// Returns the removed server name, or None if only one server remains
    pub fn remove_last_server(&mut self) -> Option<String> {
        if self.servers.len() > 1 && self.hasher.remove_bucket() {
            self.servers.pop()
        } else {
            None
        }
    }

    /// Get the server for a key
    pub fn get_server<K: Hash>(&mut self, key: &K) -> Option<&str> {
        if self.servers.is_empty() {
            return None;
        }
        let bucket = self.hasher.get_bucket_for_key(key);
        self.servers.get(bucket as usize).map(|s| s.as_str())
    }

    /// Get the server index for a key
    pub fn get_server_index<K: Hash>(&mut self, key: &K) -> Option<usize> {
        if self.servers.is_empty() {
            return None;
        }
        Some(self.hasher.get_bucket_for_key(key) as usize)
    }

    /// Get all server names
    pub fn servers(&self) -> &[String] {
        &self.servers
    }

    /// Calculate distribution across servers
    pub fn calculate_distribution<K: Hash>(&mut self, keys: &[K]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for key in keys {
            if let Some(server) = self.get_server(key) {
                *distribution.entry(server.to_string()).or_insert(0) += 1;
            }
        }
        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_jump_hash() {
        // Test basic functionality
        let bucket = jump_hash(0, 10);
        assert!(bucket < 10);

        // Deterministic
        assert_eq!(jump_hash(12345, 100), jump_hash(12345, 100));
    }

    #[test]
    fn test_single_bucket() {
        // All keys should go to bucket 0
        for key in 0..1000u64 {
            assert_eq!(jump_hash(key, 1), 0);
        }
    }

    #[test]
    fn test_distribution() {
        let num_buckets = 10u32;
        let num_keys = 10000usize;

        let mut distribution = HashMap::new();
        for key in 0..num_keys as u64 {
            let bucket = jump_hash(key, num_buckets);
            *distribution.entry(bucket).or_insert(0) += 1;
        }

        // All buckets should have some keys
        assert_eq!(distribution.len(), num_buckets as usize);

        // Distribution should be reasonably balanced
        // Expected: 1000 per bucket, allow ±20%
        for &count in distribution.values() {
            assert!(count > 800, "Too few keys: {}", count);
            assert!(count < 1200, "Too many keys: {}", count);
        }
    }

    #[test]
    fn test_minimal_movement() {
        let keys: Vec<u64> = (0..10000).collect();
        let initial_buckets = 10u32;

        // Count how many keys move when adding a bucket
        let initial: Vec<u32> = keys
            .iter()
            .map(|&k| jump_hash(k, initial_buckets))
            .collect();

        let moved: usize = keys
            .iter()
            .zip(initial.iter())
            .filter(|&(&k, &old)| jump_hash(k, initial_buckets + 1) != old)
            .count();

        // Expected: 10000/11 ≈ 909 keys
        // Allow some variance
        assert!(
            moved > 700 && moved < 1200,
            "Expected ~909 keys to move, got {}",
            moved
        );
    }

    #[test]
    fn test_monotonicity() {
        // Test that keys only move forward (to new buckets)
        for key in 0..100u64 {
            assert!(
                verify_monotonicity(key, 100),
                "Monotonicity violated for key {}",
                key
            );
        }
    }

    #[test]
    fn test_jump_hash_key() {
        let bucket1 = jump_hash_key(&"hello", 10);
        let bucket2 = jump_hash_key(&"hello", 10);
        assert_eq!(bucket1, bucket2);

        let bucket3 = jump_hash_key(&"world", 10);
        // Different keys might map to same bucket, but should be deterministic
        assert_eq!(jump_hash_key(&"world", 10), bucket3);
    }

    #[test]
    fn test_seeded_hash() {
        let _bucket1 = jump_hash_seeded(12345, 10, 0);
        let _bucket2 = jump_hash_seeded(12345, 10, 1);

        // Different seeds might produce different buckets
        // (not guaranteed, but likely for many keys)
        let mut different = 0;
        for key in 0..100u64 {
            if jump_hash_seeded(key, 10, 0) != jump_hash_seeded(key, 10, 1) {
                different += 1;
            }
        }
        assert!(different > 0, "Seeds should affect distribution");
    }

    #[test]
    fn test_jump_consistent_hash_struct() {
        let mut jch = JumpConsistentHash::new(5);

        assert_eq!(jch.num_buckets(), 5);

        let bucket = jch.get_bucket(12345);
        assert!(bucket < 5);

        jch.add_bucket();
        assert_eq!(jch.num_buckets(), 6);

        jch.remove_bucket();
        assert_eq!(jch.num_buckets(), 5);

        // Can't remove below 1
        jch.set_num_buckets(1);
        assert!(!jch.remove_bucket());
    }

    #[test]
    fn test_distribution_calculation() {
        let mut jch = JumpConsistentHash::new(5);
        let keys: Vec<u64> = (0..1000).collect();

        let distribution = jch.calculate_distribution(&keys);
        assert_eq!(distribution.len(), 5);

        let total: usize = distribution.values().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_keys_affected_prediction() {
        let jch = JumpConsistentHash::new(5);
        let keys: Vec<u64> = (0..1000).collect();

        let predicted = jch.keys_affected_by_add(&keys);

        // Expected: 1000/6 ≈ 167
        assert!(
            predicted > 100 && predicted < 250,
            "Expected ~167, got {}",
            predicted
        );
    }

    #[test]
    fn test_compare_with_modulo() {
        let keys: Vec<u64> = (0..10000).collect();

        let (jump_moves, modulo_moves) = compare_with_modulo_hashing(&keys, 10);

        // Jump should move far fewer keys
        assert!(
            jump_moves < modulo_moves,
            "Jump ({}) should move fewer than modulo ({})",
            jump_moves,
            modulo_moves
        );

        // Modulo should move most keys
        assert!(
            modulo_moves > 5000,
            "Modulo should move most keys, got {}",
            modulo_moves
        );
    }

    #[test]
    fn test_load_imbalance() {
        let mut distribution = HashMap::new();
        distribution.insert(0, 100);
        distribution.insert(1, 200);

        let imbalance = calculate_load_imbalance(&distribution);
        assert!((imbalance - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_load_std_dev() {
        let mut distribution = HashMap::new();
        distribution.insert(0, 100);
        distribution.insert(1, 100);

        let std_dev = calculate_load_std_dev(&distribution);
        assert!(
            std_dev.abs() < 0.01,
            "Equal distribution should have ~0 std dev"
        );
    }

    #[test]
    fn test_analyze_jump_pattern() {
        let pattern = analyze_jump_pattern(12345, 20);

        // Should have some jumps
        assert!(!pattern.is_empty());

        // Each jump should be to a new bucket
        for (n, bucket) in &pattern {
            assert!(*bucket < *n);
        }
    }

    #[test]
    fn test_jump_hash_router() {
        let servers = vec![
            "server0".to_string(),
            "server1".to_string(),
            "server2".to_string(),
        ];
        let mut router = JumpHashRouter::new(servers);

        assert_eq!(router.num_servers(), 3);

        // Get server for a key
        let server = router.get_server(&"mykey");
        assert!(server.is_some());

        // Add a server
        router.add_server("server3".to_string());
        assert_eq!(router.num_servers(), 4);

        // Remove last server
        let removed = router.remove_last_server();
        assert_eq!(removed, Some("server3".to_string()));
        assert_eq!(router.num_servers(), 3);
    }

    #[test]
    fn test_router_distribution() {
        let servers = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut router = JumpHashRouter::new(servers);

        let keys: Vec<String> = (0..900).map(|i| format!("key{}", i)).collect();
        let distribution = router.calculate_distribution(&keys);

        // All servers should have some keys
        assert_eq!(distribution.len(), 3);

        // Distribution should be reasonably balanced
        for &count in distribution.values() {
            assert!(count > 200, "Too few keys: {}", count);
            assert!(count < 400, "Too many keys: {}", count);
        }
    }

    #[test]
    fn test_router_deterministic() {
        let servers = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut router1 = JumpHashRouter::with_seed(servers.clone(), 42);
        let mut router2 = JumpHashRouter::with_seed(servers, 42);

        for i in 0..100 {
            let key = format!("key{}", i);
            assert_eq!(router1.get_server(&key), router2.get_server(&key));
        }
    }

    #[test]
    fn test_empty_router() {
        let mut router = JumpHashRouter::new(vec![]);
        assert!(router.get_server(&"key").is_none());
    }

    #[test]
    fn test_single_server_router() {
        let mut router = JumpHashRouter::new(vec!["only".to_string()]);

        for i in 0..100 {
            let key = format!("key{}", i);
            assert_eq!(router.get_server(&key), Some("only"));
        }

        // Can't remove the only server
        assert!(router.remove_last_server().is_none());
    }

    #[test]
    fn test_stats() {
        let mut jch = JumpConsistentHash::new(5);

        jch.get_bucket(1);
        jch.get_bucket(2);
        jch.get_bucket(3);

        assert_eq!(jch.stats().hash_operations, 3);

        jch.reset_stats();
        assert_eq!(jch.stats().hash_operations, 0);
    }

    #[test]
    fn test_expected_keys_moved() {
        let expected = expected_keys_moved(1000, 10);
        // 1000/11 ≈ 90.9
        assert!((expected - 90.9).abs() < 1.0);
    }

    #[test]
    #[should_panic(expected = "num_buckets must be > 0")]
    fn test_zero_buckets_panics() {
        jump_hash(123, 0);
    }

    #[test]
    #[should_panic(expected = "num_buckets must be > 0")]
    fn test_zero_buckets_struct_panics() {
        let _ = JumpConsistentHash::new(0);
    }

    #[test]
    fn test_large_bucket_count() {
        // Test with large number of buckets
        let bucket = jump_hash(12345, 1_000_000);
        assert!(bucket < 1_000_000);

        // Still deterministic
        assert_eq!(jump_hash(12345, 1_000_000), bucket);
    }

    #[test]
    fn test_all_keys_in_range() {
        for num_buckets in [1, 2, 5, 10, 100, 1000] {
            for key in 0..1000u64 {
                let bucket = jump_hash(key, num_buckets);
                assert!(
                    bucket < num_buckets,
                    "bucket {} >= num_buckets {} for key {}",
                    bucket,
                    num_buckets,
                    key
                );
            }
        }
    }
}
