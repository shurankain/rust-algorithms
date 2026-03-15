// Product Quantization (PQ)
//
// A vector compression technique for approximate nearest neighbor search.
// Splits high-dimensional vectors into M subvectors and quantizes each
// using K centroids (typically 256 for 8-bit codes).
//
// Key concepts:
// - Subvectors: Original D-dimensional vector split into M parts of D/M dimensions
// - Codebook: K centroids per subvector, learned via k-means
// - Codes: Each vector compressed to M bytes (indices into codebooks)
// - ADC: Asymmetric Distance Computation for fast search
//
// Memory reduction: D*4 bytes (f32) -> M bytes (u8 codes)
// Example: 128-dim f32 (512 bytes) -> 8 subvectors (8 bytes) = 64x compression
//
// Time complexity:
// - Training: O(n * iterations * K * D)
// - Encoding: O(D * K)
// - Search (ADC): O(M * K + n * M) per query
//
// Space complexity:
// - Codebooks: O(M * K * D/M) = O(K * D)
// - Codes: O(n * M)

use rand::Rng;
use std::cmp::Ordering;

/// Configuration for Product Quantization
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Number of subvectors (M)
    pub num_subvectors: usize,
    /// Number of centroids per subvector (K), typically 256 for 8-bit
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            num_subvectors: 8,
            num_centroids: 256,
            kmeans_iterations: 20,
        }
    }
}

impl PqConfig {
    /// Create a new PQ configuration
    pub fn new(num_subvectors: usize, num_centroids: usize) -> Self {
        Self {
            num_subvectors,
            num_centroids,
            kmeans_iterations: 20,
        }
    }

    /// Set number of subvectors
    pub fn subvectors(mut self, m: usize) -> Self {
        self.num_subvectors = m;
        self
    }

    /// Set number of centroids
    pub fn centroids(mut self, k: usize) -> Self {
        self.num_centroids = k;
        self
    }

    /// Set k-means iterations
    pub fn iterations(mut self, iters: usize) -> Self {
        self.kmeans_iterations = iters;
        self
    }
}

/// Product Quantization Index
pub struct PqIndex {
    /// Codebooks: M codebooks, each with K centroids of dimension D/M
    /// Shape: [num_subvectors][num_centroids][subvector_dim]
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Encoded vectors: [num_vectors][num_subvectors]
    codes: Vec<Vec<u8>>,
    /// Original vectors (optional, for exact distance fallback)
    vectors: Option<Vec<Vec<f32>>>,
    /// Vector dimension
    dimension: usize,
    /// Configuration
    config: PqConfig,
    /// Subvector dimension (D / M)
    subvector_dim: usize,
}

impl PqIndex {
    /// Train PQ codebooks on a set of training vectors
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vectors");
        let dimension = vectors[0].len();
        assert!(
            dimension % config.num_subvectors == 0,
            "Dimension must be divisible by num_subvectors"
        );

        let subvector_dim = dimension / config.num_subvectors;
        let mut codebooks = Vec::with_capacity(config.num_subvectors);

        // Train a codebook for each subvector position
        for m in 0..config.num_subvectors {
            // Extract subvectors at position m from all training vectors
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| {
                    let start = m * subvector_dim;
                    let end = start + subvector_dim;
                    v[start..end].to_vec()
                })
                .collect();

            // Train k-means to get centroids
            let centroids = kmeans(&subvectors, config.num_centroids, config.kmeans_iterations);
            codebooks.push(centroids);
        }

        Self {
            codebooks,
            codes: Vec::new(),
            vectors: None,
            dimension,
            config,
            subvector_dim,
        }
    }

    /// Encode a single vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");

        let mut codes = Vec::with_capacity(self.config.num_subvectors);

        for m in 0..self.config.num_subvectors {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let nearest = self.codebooks[m]
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, squared_euclidean(subvector, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            codes.push(nearest as u8);
        }

        codes
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: &[Vec<f32>], store_originals: bool) {
        for vector in vectors {
            let code = self.encode(vector);
            self.codes.push(code);
        }

        if store_originals {
            if self.vectors.is_none() {
                self.vectors = Some(Vec::new());
            }
            if let Some(ref mut vecs) = self.vectors {
                vecs.extend(vectors.iter().cloned());
            }
        }
    }

    /// Get number of indexed vectors
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Search using Asymmetric Distance Computation (ADC)
    /// Returns (index, approximate_distance) pairs sorted by distance
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.codes.is_empty() {
            return Vec::new();
        }

        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");

        // Precompute distance table: [num_subvectors][num_centroids]
        // distance_table[m][c] = distance from query subvector m to centroid c
        let distance_table = self.compute_distance_table(query);

        // Compute approximate distances for all vectors
        let mut distances: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(idx, code)| {
                let dist = self.compute_adc_distance(code, &distance_table);
                (idx, dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return top k
        distances.truncate(k);
        distances
    }

    /// Search with re-ranking using original vectors (if stored)
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        candidates: usize,
    ) -> Vec<(usize, f32)> {
        // Get more candidates than needed
        let mut results = self.search(query, candidates.max(k));

        // Re-rank using exact distances if we have original vectors
        if let Some(ref vectors) = self.vectors {
            for (idx, dist) in &mut results {
                *dist = squared_euclidean(query, &vectors[*idx]).sqrt();
            }
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }

        results.truncate(k);
        results
    }

    /// Compute distance table for ADC
    fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.config.num_subvectors);

        for m in 0..self.config.num_subvectors {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            let distances: Vec<f32> = self.codebooks[m]
                .iter()
                .map(|centroid| squared_euclidean(query_sub, centroid))
                .collect();
            table.push(distances);
        }

        table
    }

    /// Compute ADC distance using precomputed table
    fn compute_adc_distance(&self, code: &[u8], distance_table: &[Vec<f32>]) -> f32 {
        code.iter()
            .enumerate()
            .map(|(m, &c)| distance_table[m][c as usize])
            .sum::<f32>()
            .sqrt()
    }

    /// Reconstruct an approximate vector from its code
    pub fn reconstruct(&self, index: usize) -> Option<Vec<f32>> {
        let code = self.codes.get(index)?;
        let mut vector = Vec::with_capacity(self.dimension);

        for (m, &c) in code.iter().enumerate() {
            vector.extend_from_slice(&self.codebooks[m][c as usize]);
        }

        Some(vector)
    }

    /// Get quantization error for a vector
    pub fn quantization_error(&self, vector: &[f32]) -> f32 {
        let code = self.encode(vector);
        let mut error = 0.0;

        for (m, &c) in code.iter().enumerate() {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvector = &vector[start..end];
            let centroid = &self.codebooks[m][c as usize];
            error += squared_euclidean(subvector, centroid);
        }

        error.sqrt()
    }

    /// Get average quantization error over stored vectors
    pub fn average_quantization_error(&self) -> Option<f32> {
        let vectors = self.vectors.as_ref()?;
        if vectors.is_empty() {
            return None;
        }

        let total_error: f32 = vectors.iter().map(|v| self.quantization_error(v)).sum();

        Some(total_error / vectors.len() as f32)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimension * 4; // f32 = 4 bytes
        let compressed_bytes = self.config.num_subvectors; // 1 byte per subvector
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> MemoryUsage {
        let codebook_bytes =
            self.config.num_subvectors * self.config.num_centroids * self.subvector_dim * 4; // f32
        let codes_bytes = self.codes.len() * self.config.num_subvectors;
        let vectors_bytes = self
            .vectors
            .as_ref()
            .map(|v| v.len() * self.dimension * 4)
            .unwrap_or(0);

        MemoryUsage {
            codebooks: codebook_bytes,
            codes: codes_bytes,
            vectors: vectors_bytes,
            total: codebook_bytes + codes_bytes + vectors_bytes,
        }
    }
}

/// Memory usage breakdown
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub codebooks: usize,
    pub codes: usize,
    pub vectors: usize,
    pub total: usize,
}

/// Simple k-means clustering
fn kmeans(data: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    if data.is_empty() {
        return vec![];
    }

    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n); // Can't have more centroids than data points

    let mut rng = rand::thread_rng();

    // Initialize centroids randomly from data points
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|_| {
            let idx = rng.gen_range(0..n);
            data[idx].clone()
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assign each point to nearest centroid
        for (i, point) in data.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, squared_euclidean(point, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            assignments[i] = nearest;
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }

        for (c, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[c] > 0 {
                for val in centroid.iter_mut() {
                    *val /= counts[c] as f32;
                }
            } else {
                // Empty cluster: reinitialize randomly
                let idx = rng.gen_range(0..n);
                *centroid = data[idx].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// Squared Euclidean distance
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Build and train a PQ index
pub fn build_pq_index(
    training_vectors: &[Vec<f32>],
    index_vectors: &[Vec<f32>],
    config: PqConfig,
    store_originals: bool,
) -> PqIndex {
    let mut index = PqIndex::train(training_vectors, config);
    index.add(index_vectors, store_originals);
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.sample::<f32, _>(rand::distributions::Standard))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_pq_config_default() {
        let config = PqConfig::default();
        assert_eq!(config.num_subvectors, 8);
        assert_eq!(config.num_centroids, 256);
        assert_eq!(config.kmeans_iterations, 20);
    }

    #[test]
    fn test_pq_config_builder() {
        let config = PqConfig::new(4, 128)
            .subvectors(16)
            .centroids(64)
            .iterations(10);
        assert_eq!(config.num_subvectors, 16);
        assert_eq!(config.num_centroids, 64);
        assert_eq!(config.kmeans_iterations, 10);
    }

    #[test]
    fn test_pq_train() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 16).iterations(5);
        let index = PqIndex::train(&vectors, config);

        assert_eq!(index.dimension(), 32);
        assert!(index.is_empty());
    }

    #[test]
    fn test_pq_encode() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 16).iterations(5);
        let index = PqIndex::train(&vectors, config);

        let code = index.encode(&vectors[0]);
        assert_eq!(code.len(), 4);
        for &c in &code {
            assert!(c < 16);
        }
    }

    #[test]
    fn test_pq_add() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 16).iterations(5);
        let mut index = PqIndex::train(&vectors, config);

        index.add(&vectors, false);
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_pq_search() {
        let vectors = random_vectors(200, 32);
        let config = PqConfig::new(4, 32).iterations(10);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, false);

        let query = &vectors[0];
        let results = index.search(query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_pq_search_finds_exact() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 32).iterations(10);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, false);

        // Search for first vector - should find itself
        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
        // First result should be index 0 (the query vector itself)
        assert_eq!(results[0].0, 0);
        // Distance should be very small (quantization error only)
        assert!(results[0].1 < 1.0);
    }

    #[test]
    fn test_pq_search_with_rerank() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 16).iterations(5);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, true); // Store originals

        let query = &vectors[0];
        let results = index.search_with_rerank(query, 5, 20);

        assert_eq!(results.len(), 5);
        // First result should be exact match with distance 0
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_pq_reconstruct() {
        let vectors = random_vectors(50, 32);
        let config = PqConfig::new(4, 16).iterations(5);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, false);

        let reconstructed = index.reconstruct(0).unwrap();
        assert_eq!(reconstructed.len(), 32);

        // Reconstructed should be somewhat close to original
        let error = squared_euclidean(&vectors[0], &reconstructed).sqrt();
        assert!(error < 10.0); // Reasonable error bound
    }

    #[test]
    fn test_pq_quantization_error() {
        let vectors = random_vectors(100, 32);
        let config = PqConfig::new(4, 32).iterations(10);
        let index = PqIndex::train(&vectors, config);

        let error = index.quantization_error(&vectors[0]);
        assert!(error >= 0.0);
        assert!(error < 10.0); // Should be reasonable
    }

    #[test]
    fn test_pq_compression_ratio() {
        let vectors = random_vectors(50, 128);
        let config = PqConfig::new(8, 256);
        let index = PqIndex::train(&vectors, config);

        // 128 floats * 4 bytes / 8 codes = 64x compression
        let ratio = index.compression_ratio();
        assert!((ratio - 64.0).abs() < 0.1);
    }

    #[test]
    fn test_pq_memory_usage() {
        let vectors = random_vectors(100, 64);
        let config = PqConfig::new(8, 16).iterations(5);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, true);

        let mem = index.memory_usage();
        assert!(mem.codebooks > 0);
        assert!(mem.codes > 0);
        assert!(mem.vectors > 0);
        assert_eq!(mem.total, mem.codebooks + mem.codes + mem.vectors);
    }

    #[test]
    fn test_pq_high_dimensional() {
        let vectors = random_vectors(200, 128);
        let config = PqConfig::new(16, 64).iterations(5);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, false);

        let query = random_vectors(1, 128).pop().unwrap();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_kmeans_basic() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let centroids = kmeans(&data, 2, 10);
        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0, 0) and (10, 10)
        let mut sorted_centroids = centroids.clone();
        sorted_centroids.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        assert!(sorted_centroids[0][0] < 1.0);
        assert!(sorted_centroids[1][0] > 9.0);
    }

    #[test]
    fn test_build_pq_index() {
        let training = random_vectors(100, 32);
        let index_vecs = random_vectors(50, 32);
        let config = PqConfig::new(4, 16).iterations(5);

        let index = build_pq_index(&training, &index_vecs, config, false);

        assert_eq!(index.len(), 50);
        assert_eq!(index.dimension(), 32);
    }

    #[test]
    #[should_panic(expected = "Dimension must be divisible")]
    fn test_pq_dimension_not_divisible() {
        let vectors = random_vectors(10, 33); // 33 not divisible by 8
        let config = PqConfig::new(8, 16);
        PqIndex::train(&vectors, config);
    }

    #[test]
    #[should_panic(expected = "Cannot train on empty")]
    fn test_pq_train_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let config = PqConfig::default();
        PqIndex::train(&vectors, config);
    }

    #[test]
    fn test_pq_recall() {
        // Test that PQ finds reasonably good results
        let vectors = random_vectors(500, 64);
        let config = PqConfig::new(8, 64).iterations(15);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, true);

        // Search for first vector
        let query = &vectors[0];
        let results = index.search(query, 10);

        // Should find the exact vector in top results
        let found = results.iter().any(|(idx, _)| *idx == 0);
        assert!(found, "Expected to find query vector in results");
    }

    #[test]
    fn test_average_quantization_error() {
        let vectors = random_vectors(50, 32);
        let config = PqConfig::new(4, 32).iterations(10);
        let mut index = PqIndex::train(&vectors, config);
        index.add(&vectors, true);

        let avg_error = index.average_quantization_error().unwrap();
        assert!(avg_error > 0.0);
        assert!(avg_error < 10.0);
    }
}
