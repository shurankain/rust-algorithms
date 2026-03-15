// Inverted File Index (IVF)
//
// A coarse quantization technique for approximate nearest neighbor search.
// Partitions the vector space into clusters using k-means, then at search
// time only examines vectors in the most relevant clusters.
//
// Key concepts:
// - Centroids: Cluster centers learned via k-means
// - Inverted lists: Vectors assigned to each centroid
// - nprobe: Number of clusters to search (tradeoff: speed vs recall)
//
// Variants:
// - IVF-Flat: Exact distances within clusters (this implementation)
// - IVF-PQ: Product quantization within clusters (can combine with PQ)
// - IVF-SQ: Scalar quantization within clusters
//
// Time complexity:
// - Training: O(n * iterations * num_clusters * D)
// - Search: O(num_clusters * D + nprobe * avg_cluster_size * D)
//
// Space complexity: O(n * D + num_clusters * D)

use crate::similarity::DistanceMetric;
use rand::Rng;
use std::cmp::Ordering;

/// Configuration for IVF index
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of clusters (centroids)
    pub num_clusters: usize,
    /// Number of clusters to probe during search
    pub nprobe: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            num_clusters: 100,
            nprobe: 10,
            kmeans_iterations: 20,
            metric: DistanceMetric::Euclidean,
        }
    }
}

impl IvfConfig {
    /// Create a new IVF configuration
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            nprobe: (num_clusters / 10).max(1),
            kmeans_iterations: 20,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set number of clusters to probe
    pub fn nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set k-means iterations
    pub fn iterations(mut self, iters: usize) -> Self {
        self.kmeans_iterations = iters;
        self
    }

    /// Set distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// An entry in the inverted list
#[derive(Debug, Clone)]
struct IvfEntry {
    /// Original index of the vector
    index: usize,
    /// The vector data
    vector: Vec<f32>,
}

/// Inverted File Index
pub struct IvfIndex {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    /// Inverted lists: cluster_id -> list of vectors in that cluster
    inverted_lists: Vec<Vec<IvfEntry>>,
    /// Vector dimension
    dimension: usize,
    /// Configuration
    config: IvfConfig,
    /// Total number of vectors
    num_vectors: usize,
}

impl IvfIndex {
    /// Train IVF centroids on a set of training vectors
    pub fn train(vectors: &[Vec<f32>], config: IvfConfig) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vectors");
        let dimension = vectors[0].len();

        // Train k-means to get centroids
        let centroids = kmeans(
            vectors,
            config.num_clusters,
            config.kmeans_iterations,
            &config.metric,
        );

        // Initialize empty inverted lists
        let inverted_lists = vec![Vec::new(); centroids.len()];

        Self {
            centroids,
            inverted_lists,
            dimension,
            config,
            num_vectors: 0,
        }
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: &[Vec<f32>]) {
        for (i, vector) in vectors.iter().enumerate() {
            assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");

            // Find nearest centroid
            let cluster_id = self.find_nearest_centroid(vector);

            // Add to inverted list
            let entry = IvfEntry {
                index: self.num_vectors + i,
                vector: vector.clone(),
            };
            self.inverted_lists[cluster_id].push(entry);
        }

        self.num_vectors += vectors.len();
    }

    /// Add a single vector
    pub fn add_single(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");

        let index = self.num_vectors;
        let cluster_id = self.find_nearest_centroid(&vector);

        let entry = IvfEntry { index, vector };
        self.inverted_lists[cluster_id].push(entry);

        self.num_vectors += 1;
        index
    }

    /// Get number of indexed vectors
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Find nearest centroid for a vector
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.config.metric.distance(vector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find k nearest centroids for a vector
    fn find_nearest_centroids(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.config.metric.distance(vector, c)))
            .collect();

        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        centroid_dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Search with custom nprobe
    pub fn search_with_nprobe(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        if self.num_vectors == 0 {
            return Vec::new();
        }

        assert_eq!(query.len(), self.dimension, "Query dimension mismatch");

        // Find nearest centroids
        let probe_clusters = self.find_nearest_centroids(query, nprobe.min(self.centroids.len()));

        // Collect candidates from probed clusters
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for cluster_id in probe_clusters {
            for entry in &self.inverted_lists[cluster_id] {
                let dist = self.config.metric.distance(query, &entry.vector);
                candidates.push((entry.index, dist));
            }
        }

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return top k
        candidates.truncate(k);
        candidates
    }

    /// Get cluster statistics
    pub fn cluster_stats(&self) -> ClusterStats {
        let sizes: Vec<usize> = self.inverted_lists.iter().map(|l| l.len()).collect();

        let total: usize = sizes.iter().sum();
        let non_empty = sizes.iter().filter(|&&s| s > 0).count();
        let avg_size = if non_empty > 0 {
            total as f32 / non_empty as f32
        } else {
            0.0
        };
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let min_size = sizes.iter().filter(|&&s| s > 0).min().copied().unwrap_or(0);

        // Compute standard deviation
        let variance = if non_empty > 0 {
            sizes
                .iter()
                .filter(|&&s| s > 0)
                .map(|&s| {
                    let diff = s as f32 - avg_size;
                    diff * diff
                })
                .sum::<f32>()
                / non_empty as f32
        } else {
            0.0
        };

        ClusterStats {
            num_clusters: self.centroids.len(),
            non_empty_clusters: non_empty,
            avg_cluster_size: avg_size,
            max_cluster_size: max_size,
            min_cluster_size: min_size,
            std_dev: variance.sqrt(),
        }
    }

    /// Get vectors in a specific cluster
    pub fn get_cluster_vectors(&self, cluster_id: usize) -> Option<Vec<(usize, &[f32])>> {
        self.inverted_lists.get(cluster_id).map(|list| {
            list.iter()
                .map(|e| (e.index, e.vector.as_slice()))
                .collect()
        })
    }

    /// Get centroid for a cluster
    pub fn get_centroid(&self, cluster_id: usize) -> Option<&[f32]> {
        self.centroids.get(cluster_id).map(|c| c.as_slice())
    }

    /// Estimate recall at given nprobe
    /// Uses a sample of vectors to estimate what fraction would be found
    pub fn estimate_recall(&self, sample_vectors: &[Vec<f32>], nprobe: usize, k: usize) -> f32 {
        if sample_vectors.is_empty() {
            return 0.0;
        }

        let mut total_recall = 0.0;

        for query in sample_vectors {
            // Get approximate results
            let approx_results = self.search_with_nprobe(query, k, nprobe);

            // Get exact results (search all clusters)
            let exact_results = self.search_with_nprobe(query, k, self.centroids.len());

            // Count how many of exact top-k are in approximate results
            let exact_indices: std::collections::HashSet<usize> =
                exact_results.iter().map(|(i, _)| *i).collect();
            let approx_indices: std::collections::HashSet<usize> =
                approx_results.iter().map(|(i, _)| *i).collect();

            let found = exact_indices.intersection(&approx_indices).count();
            total_recall += found as f32 / k.min(exact_indices.len()) as f32;
        }

        total_recall / sample_vectors.len() as f32
    }
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub num_clusters: usize,
    pub non_empty_clusters: usize,
    pub avg_cluster_size: f32,
    pub max_cluster_size: usize,
    pub min_cluster_size: usize,
    pub std_dev: f32,
}

/// Build and populate an IVF index
pub fn build_ivf_index(
    training_vectors: &[Vec<f32>],
    index_vectors: &[Vec<f32>],
    config: IvfConfig,
) -> IvfIndex {
    let mut index = IvfIndex::train(training_vectors, config);
    index.add(index_vectors);
    index
}

/// K-means clustering
fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    iterations: usize,
    metric: &DistanceMetric,
) -> Vec<Vec<f32>> {
    if data.is_empty() {
        return vec![];
    }

    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);

    let mut rng = rand::thread_rng();

    // Initialize centroids using k-means++ style initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // First centroid: random
    centroids.push(data[rng.gen_range(0..n)].clone());

    // Remaining centroids: weighted by distance to nearest existing centroid
    while centroids.len() < k {
        let mut weights: Vec<f32> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| metric.distance(point, c))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(f32::MAX)
            })
            .collect();

        // Square the distances for weighting
        for w in &mut weights {
            *w = *w * *w;
        }

        let total: f32 = weights.iter().sum();
        if total == 0.0 {
            // All points are on existing centroids, just pick random
            centroids.push(data[rng.gen_range(0..n)].clone());
            continue;
        }

        // Sample proportional to weights
        let threshold = rng.sample::<f32, _>(rand::distributions::Standard) * total;
        let mut cumsum = 0.0;
        let mut selected = 0;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }
        centroids.push(data[selected].clone());
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assign each point to nearest centroid
        for (i, point) in data.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, metric.distance(point, c)))
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
                // Empty cluster: reinitialize from random point
                let idx = rng.gen_range(0..n);
                *centroid = data[idx].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
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
    fn test_ivf_config_default() {
        let config = IvfConfig::default();
        assert_eq!(config.num_clusters, 100);
        assert_eq!(config.nprobe, 10);
        assert_eq!(config.kmeans_iterations, 20);
        assert_eq!(config.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_ivf_config_builder() {
        let config = IvfConfig::new(50)
            .nprobe(5)
            .iterations(10)
            .metric(DistanceMetric::Cosine);
        assert_eq!(config.num_clusters, 50);
        assert_eq!(config.nprobe, 5);
        assert_eq!(config.kmeans_iterations, 10);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_ivf_train() {
        let vectors = random_vectors(100, 32);
        let config = IvfConfig::new(10).iterations(5);
        let index = IvfIndex::train(&vectors, config);

        assert_eq!(index.dimension(), 32);
        assert_eq!(index.num_clusters(), 10);
        assert!(index.is_empty());
    }

    #[test]
    fn test_ivf_add() {
        let vectors = random_vectors(100, 32);
        let config = IvfConfig::new(10).iterations(5);
        let mut index = IvfIndex::train(&vectors, config);

        index.add(&vectors);
        assert_eq!(index.len(), 100);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_ivf_add_single() {
        let vectors = random_vectors(50, 32);
        let config = IvfConfig::new(10).iterations(5);
        let mut index = IvfIndex::train(&vectors, config);

        let idx = index.add_single(vec![0.5; 32]);
        assert_eq!(idx, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_ivf_search() {
        let vectors = random_vectors(200, 32);
        let config = IvfConfig::new(20).nprobe(5).iterations(10);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let query = &vectors[0];
        let results = index.search(query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_ivf_search_finds_exact() {
        let vectors = random_vectors(100, 32);
        let config = IvfConfig::new(10).nprobe(10).iterations(10); // High nprobe for exact
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
        // First result should be the query vector itself
        assert_eq!(results[0].0, 0);
        // Distance should be zero
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_ivf_search_with_nprobe() {
        let vectors = random_vectors(200, 32);
        let config = IvfConfig::new(20).iterations(10);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let query = &vectors[0];

        // More nprobe should give better results
        let results_low = index.search_with_nprobe(query, 10, 1);
        let results_high = index.search_with_nprobe(query, 10, 20);

        // Both should return results
        assert!(!results_low.is_empty());
        assert!(!results_high.is_empty());

        // High nprobe should find the exact vector
        assert_eq!(results_high[0].0, 0);
    }

    #[test]
    fn test_ivf_cluster_stats() {
        let vectors = random_vectors(100, 32);
        let config = IvfConfig::new(10).iterations(10);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let stats = index.cluster_stats();
        assert_eq!(stats.num_clusters, 10);
        assert!(stats.non_empty_clusters > 0);
        assert!(stats.avg_cluster_size > 0.0);
        // Average should be approximately 100/10 = 10
        assert!(stats.avg_cluster_size > 5.0 && stats.avg_cluster_size < 20.0);
    }

    #[test]
    fn test_ivf_get_cluster_vectors() {
        let vectors = random_vectors(50, 16);
        let config = IvfConfig::new(5).iterations(5);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        // Find a non-empty cluster
        let stats = index.cluster_stats();
        assert!(stats.non_empty_clusters > 0);

        // At least one cluster should have vectors
        let mut found_vectors = false;
        for i in 0..5 {
            if let Some(vecs) = index.get_cluster_vectors(i) {
                if !vecs.is_empty() {
                    found_vectors = true;
                    // Check vector dimensions
                    for (_, v) in vecs {
                        assert_eq!(v.len(), 16);
                    }
                    break;
                }
            }
        }
        assert!(found_vectors);
    }

    #[test]
    fn test_ivf_get_centroid() {
        let vectors = random_vectors(50, 32);
        let config = IvfConfig::new(10).iterations(5);
        let index = IvfIndex::train(&vectors, config);

        let centroid = index.get_centroid(0).unwrap();
        assert_eq!(centroid.len(), 32);

        assert!(index.get_centroid(100).is_none());
    }

    #[test]
    fn test_build_ivf_index() {
        let training = random_vectors(100, 32);
        let index_vecs = random_vectors(50, 32);
        let config = IvfConfig::new(10).iterations(5);

        let index = build_ivf_index(&training, &index_vecs, config);

        assert_eq!(index.len(), 50);
        assert_eq!(index.dimension(), 32);
        assert_eq!(index.num_clusters(), 10);
    }

    #[test]
    fn test_ivf_cosine_metric() {
        let vectors = random_vectors(100, 32);
        let config = IvfConfig::new(10)
            .nprobe(5)
            .iterations(10)
            .metric(DistanceMetric::Cosine);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
        // First result should be the query vector
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_ivf_high_dimensional() {
        let vectors = random_vectors(200, 128);
        let config = IvfConfig::new(20).nprobe(5).iterations(5);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        let query = random_vectors(1, 128).pop().unwrap();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_ivf_recall_estimation() {
        let vectors = random_vectors(200, 32);
        let config = IvfConfig::new(20).nprobe(5).iterations(10);
        let mut index = IvfIndex::train(&vectors, config);
        index.add(&vectors);

        // Sample some vectors for recall estimation
        let sample: Vec<Vec<f32>> = vectors.iter().take(10).cloned().collect();

        // Low nprobe should have lower recall
        let recall_low = index.estimate_recall(&sample, 1, 10);
        // High nprobe should have higher recall
        let recall_high = index.estimate_recall(&sample, 20, 10);

        assert!(recall_high >= recall_low);
        // With all clusters, should get perfect recall
        assert!((recall_high - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kmeans_basic() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let centroids = kmeans(&data, 2, 10, &DistanceMetric::Euclidean);
        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0, 0) and (10, 10)
        let mut sorted_centroids = centroids.clone();
        sorted_centroids.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        assert!(sorted_centroids[0][0] < 1.0);
        assert!(sorted_centroids[1][0] > 9.0);
    }

    #[test]
    #[should_panic(expected = "Cannot train on empty")]
    fn test_ivf_train_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let config = IvfConfig::default();
        IvfIndex::train(&vectors, config);
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_ivf_dimension_mismatch() {
        let vectors = random_vectors(50, 32);
        let config = IvfConfig::new(10).iterations(5);
        let mut index = IvfIndex::train(&vectors, config);

        index.add(&[vec![1.0, 2.0]]); // Wrong dimension
    }

    #[test]
    fn test_ivf_empty_search() {
        let vectors = random_vectors(50, 32);
        let config = IvfConfig::new(10).iterations(5);
        let index = IvfIndex::train(&vectors, config);

        // Search on empty index
        let results = index.search(&vectors[0], 10);
        assert!(results.is_empty());
    }
}
