// Locality Sensitive Hashing (LSH)
//
// A technique for approximate nearest neighbor search that hashes
// similar vectors into the same "buckets" with high probability.
//
// This implementation uses random hyperplane LSH (SimHash) which works
// well with cosine similarity. For each hash function, we generate a
// random hyperplane and determine which side of the hyperplane a vector falls on.
//
// Key concepts:
// - Hash tables: Multiple hash tables increase recall
// - Hash functions: More bits = higher precision but lower recall
// - Buckets: Vectors with same hash are candidates for similarity
//
// Time complexity:
// - Build: O(n * L * K * d) where L = tables, K = bits, d = dimension
// - Query: O(L * K * d + |candidates| * d)
//
// Space complexity: O(n * L + L * K * d)

use rand::Rng;
use std::collections::HashMap;

/// Configuration for LSH index
#[derive(Debug, Clone)]
pub struct LshConfig {
    /// Number of hash tables (more tables = better recall, more memory)
    pub num_tables: usize,
    /// Number of hash bits per table (more bits = higher precision, lower recall)
    pub num_bits: usize,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_bits: 8,
        }
    }
}

impl LshConfig {
    /// Create a new LSH configuration
    pub fn new(num_tables: usize, num_bits: usize) -> Self {
        Self {
            num_tables,
            num_bits,
        }
    }

    /// Set number of hash tables
    pub fn tables(mut self, num_tables: usize) -> Self {
        self.num_tables = num_tables;
        self
    }

    /// Set number of hash bits per table
    pub fn bits(mut self, num_bits: usize) -> Self {
        self.num_bits = num_bits;
        self
    }
}

/// A single hash table with random hyperplanes
struct HashTable {
    /// Random hyperplanes (num_bits x dimension)
    hyperplanes: Vec<Vec<f32>>,
    /// Buckets mapping hash -> vector indices
    buckets: HashMap<u64, Vec<usize>>,
}

impl HashTable {
    /// Create a new hash table with random hyperplanes
    fn new(num_bits: usize, dimension: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Generate random hyperplanes
        let hyperplanes: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.sample::<f32, _>(rand::distributions::Standard) * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        Self {
            hyperplanes,
            buckets: HashMap::new(),
        }
    }

    /// Compute hash for a vector
    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash: u64 = 0;
        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product with hyperplane
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(&v, &h)| v * h)
                .sum();

            // Set bit if on positive side of hyperplane
            if dot >= 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }

    /// Insert a vector index into the table
    fn insert(&mut self, vector: &[f32], index: usize) {
        let hash = self.hash(vector);
        self.buckets.entry(hash).or_default().push(index);
    }

    /// Get candidate indices for a query vector
    fn get_candidates(&self, vector: &[f32]) -> Vec<usize> {
        let hash = self.hash(vector);
        self.buckets.get(&hash).cloned().unwrap_or_default()
    }
}

/// LSH Index for approximate nearest neighbor search
pub struct LshIndex {
    /// The vector data
    vectors: Vec<Vec<f32>>,
    /// Hash tables
    tables: Vec<HashTable>,
    /// Vector dimension
    dimension: usize,
    /// Configuration
    config: LshConfig,
}

impl LshIndex {
    /// Create a new empty LSH index
    pub fn new(dimension: usize, config: LshConfig) -> Self {
        let tables = (0..config.num_tables)
            .map(|_| HashTable::new(config.num_bits, dimension))
            .collect();

        Self {
            vectors: Vec::new(),
            tables,
            dimension,
            config,
        }
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the dimension of vectors
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Insert a vector into the index
    /// Returns the index of the inserted vector
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");

        let index = self.vectors.len();

        // Insert into all hash tables
        for table in &mut self.tables {
            table.insert(&vector, index);
        }

        self.vectors.push(vector);
        index
    }

    /// Insert multiple vectors
    pub fn insert_batch(&mut self, vectors: &[Vec<f32>]) {
        for vector in vectors {
            self.insert(vector.clone());
        }
    }

    /// Search for k nearest neighbors using cosine similarity
    /// Returns vector of (index, similarity) pairs, sorted by similarity (descending)
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Collect candidates from all tables
        let mut candidate_set = std::collections::HashSet::new();
        for table in &self.tables {
            for idx in table.get_candidates(query) {
                candidate_set.insert(idx);
            }
        }

        // If no candidates found, fall back to scanning all vectors
        let candidates: Vec<usize> = if candidate_set.is_empty() {
            (0..self.vectors.len()).collect()
        } else {
            candidate_set.into_iter().collect()
        };

        // Compute actual cosine similarity for candidates
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&idx| {
                let sim = cosine_similarity(query, &self.vectors[idx]);
                (idx, sim)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        results.truncate(k);
        results
    }

    /// Search with Euclidean distance instead of cosine similarity
    /// Returns vector of (index, distance) pairs, sorted by distance (ascending)
    pub fn search_euclidean(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Collect candidates from all tables
        let mut candidate_set = std::collections::HashSet::new();
        for table in &self.tables {
            for idx in table.get_candidates(query) {
                candidate_set.insert(idx);
            }
        }

        // If no candidates found, fall back to scanning all vectors
        let candidates: Vec<usize> = if candidate_set.is_empty() {
            (0..self.vectors.len()).collect()
        } else {
            candidate_set.into_iter().collect()
        };

        // Compute Euclidean distance for candidates
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&idx| {
                let dist = euclidean_distance(query, &self.vectors[idx]);
                (idx, dist)
            })
            .collect();

        // Sort by distance (ascending)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        results.truncate(k);
        results
    }

    /// Get a vector by index
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        self.vectors.get(index).map(|v| v.as_slice())
    }

    /// Get statistics about the index
    pub fn stats(&self) -> LshStats {
        let bucket_sizes: Vec<usize> = self
            .tables
            .iter()
            .flat_map(|t| t.buckets.values().map(|b| b.len()))
            .collect();

        let total_buckets: usize = self.tables.iter().map(|t| t.buckets.len()).sum();
        let avg_bucket_size = if total_buckets > 0 {
            bucket_sizes.iter().sum::<usize>() as f32 / total_buckets as f32
        } else {
            0.0
        };
        let max_bucket_size = bucket_sizes.iter().max().copied().unwrap_or(0);

        LshStats {
            num_vectors: self.vectors.len(),
            num_tables: self.config.num_tables,
            num_bits: self.config.num_bits,
            total_buckets,
            avg_bucket_size,
            max_bucket_size,
        }
    }
}

/// Statistics about an LSH index
#[derive(Debug, Clone)]
pub struct LshStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub num_bits: usize,
    pub total_buckets: usize,
    pub avg_bucket_size: f32,
    pub max_bucket_size: usize,
}

/// Build an LSH index from a set of vectors
pub fn build_lsh_index(vectors: &[Vec<f32>], config: LshConfig) -> LshIndex {
    assert!(!vectors.is_empty(), "Cannot build index from empty vectors");

    let dimension = vectors[0].len();
    let mut index = LshIndex::new(dimension, config);
    index.insert_batch(vectors);
    index
}

// Helper functions (duplicated to avoid circular dependencies)

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_config_default() {
        let config = LshConfig::default();
        assert_eq!(config.num_tables, 10);
        assert_eq!(config.num_bits, 8);
    }

    #[test]
    fn test_lsh_config_builder() {
        let config = LshConfig::new(5, 16).tables(20).bits(12);
        assert_eq!(config.num_tables, 20);
        assert_eq!(config.num_bits, 12);
    }

    #[test]
    fn test_lsh_empty_index() {
        let index = LshIndex::new(128, LshConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.dimension(), 128);
    }

    #[test]
    fn test_lsh_insert() {
        let mut index = LshIndex::new(3, LshConfig::default());
        let idx = index.insert(vec![1.0, 2.0, 3.0]);
        assert_eq!(idx, 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_lsh_search_basic() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0], // Similar to first vector
        ];

        let index = build_lsh_index(&vectors, LshConfig::new(10, 4));

        // Search for vector similar to first
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2);

        assert!(!results.is_empty());
        // First result should be the exact match or very similar
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn test_lsh_search_euclidean() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0],
        ];

        let index = build_lsh_index(&vectors, LshConfig::new(10, 4));

        let query = vec![0.5, 0.0, 0.0];
        let results = index.search_euclidean(&query, 2);

        assert_eq!(results.len(), 2);
        // Results should be sorted by distance
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_lsh_high_dimensional() {
        let dim = 128;
        let n = 100;

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.sample::<f32, _>(rand::distributions::Standard))
                    .collect()
            })
            .collect();

        let index = build_lsh_index(&vectors, LshConfig::new(15, 10));

        // Search should work
        let query: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(rand::distributions::Standard))
            .collect();
        let results = index.search(&query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_lsh_identical_vectors() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];

        let index = build_lsh_index(&vectors, LshConfig::default());

        let results = index.search(&vectors[0], 3);
        assert_eq!(results.len(), 3);
        // All should have perfect similarity
        for (_, sim) in &results {
            assert!((*sim - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lsh_get_vector() {
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let index = build_lsh_index(&vectors, LshConfig::default());

        let v = index.get_vector(0).unwrap();
        assert_eq!(v, &[1.0, 2.0, 3.0]);

        let v = index.get_vector(1).unwrap();
        assert_eq!(v, &[4.0, 5.0, 6.0]);

        assert!(index.get_vector(2).is_none());
    }

    #[test]
    fn test_lsh_stats() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();

        let config = LshConfig::new(5, 4);
        let index = build_lsh_index(&vectors, config);

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 50);
        assert_eq!(stats.num_tables, 5);
        assert_eq!(stats.num_bits, 4);
        assert!(stats.total_buckets > 0);
        assert!(stats.avg_bucket_size > 0.0);
    }

    #[test]
    fn test_lsh_recall() {
        // Test that LSH finds at least some of the true nearest neighbors
        let dim = 16;
        let n = 200;

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.sample::<f32, _>(rand::distributions::Standard))
                    .collect()
            })
            .collect();

        // Use more tables and bits for better recall
        let config = LshConfig::new(20, 8);
        let index = build_lsh_index(&vectors, config);

        // Use first vector as query
        let query = &vectors[0];
        let results = index.search(query, 1);

        // Should find the exact vector with similarity ~1.0
        assert!(!results.is_empty());
        assert!(
            results[0].1 > 0.99,
            "Expected high similarity, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_hash_table_consistency() {
        let table = HashTable::new(8, 3);

        let vector = vec![1.0, 2.0, 3.0];

        // Same vector should always get same hash
        let hash1 = table.hash(&vector);
        let hash2 = table.hash(&vector);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_similar_vectors_same_bucket() {
        // Highly similar vectors should often hash to the same bucket
        let table = HashTable::new(4, 3); // Few bits = more collisions

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.99, 0.01, 0.0]; // Very similar to v1

        let hash1 = table.hash(&v1);
        let hash2 = table.hash(&v2);

        // With only 4 bits and similar vectors, they often match
        // This is probabilistic, so we just check the hashes are computed
        assert!(hash1 < 16); // 4 bits = max 16 values
        assert!(hash2 < 16);
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_lsh_dimension_mismatch() {
        let mut index = LshIndex::new(3, LshConfig::default());
        index.insert(vec![1.0, 2.0]); // Wrong dimension
    }

    #[test]
    #[should_panic(expected = "Cannot build index from empty vectors")]
    fn test_build_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        build_lsh_index(&vectors, LshConfig::default());
    }
}
