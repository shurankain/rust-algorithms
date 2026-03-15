// HNSW (Hierarchical Navigable Small World)
// State-of-the-art algorithm for Approximate Nearest Neighbor (ANN) search
// Used in vector databases: Pinecone, Milvus, pgvector, Qdrant, Weaviate
//
// Key concepts:
// - Multi-layer graph structure (like skip lists)
// - Higher layers have fewer nodes, enabling fast navigation
// - Lower layers have more connections for accurate local search
// - Greedy search from top layer down to layer 0
//
// Time complexity:
// - Build: O(n * log(n) * M)
// - Search: O(log(n) * M)
// where n = number of vectors, M = max connections per node
//
// Space complexity: O(n * M * L) where L = number of layers

use crate::similarity::DistanceMetric;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// Helper trait to work around `gen` being reserved in Rust 2024
#[allow(dead_code)]
trait RngExt {
    fn random_f64(&mut self) -> f64;
    fn random_f32(&mut self) -> f32;
}

impl<R: Rng> RngExt for R {
    fn random_f64(&mut self) -> f64 {
        self.sample(rand::distributions::Standard)
    }
    fn random_f32(&mut self) -> f32 {
        self.sample(rand::distributions::Standard)
    }
}

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node at layer 0
    pub m: usize,
    /// Maximum number of connections per node at layers > 0 (typically m / 2)
    pub m_max0: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search
    pub ef_search: usize,
    /// Level generation factor (typically 1 / ln(M))
    pub ml: f64,
    /// Distance metric to use
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16.0_f64).ln(), // 1/ln(M)
            metric: DistanceMetric::Euclidean,
        }
    }
}

impl HnswConfig {
    /// Create a new config with custom M parameter
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set ef_construction parameter
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search parameter
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

/// A node in the HNSW graph
#[derive(Debug, Clone)]
struct HnswNode {
    /// The vector data
    vector: Vec<f32>,
    /// Connections at each layer (layer -> list of neighbor indices)
    neighbors: Vec<Vec<usize>>,
}

/// Element with distance for priority queue
#[derive(Clone)]
struct DistanceElement {
    distance: f32,
    index: usize,
}

impl PartialEq for DistanceElement {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for DistanceElement {}

impl PartialOrd for DistanceElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (smallest distance first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap element (for keeping furthest candidates)
#[derive(Clone)]
struct MaxDistanceElement {
    distance: f32,
    index: usize,
}

impl PartialEq for MaxDistanceElement {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for MaxDistanceElement {}

impl PartialOrd for MaxDistanceElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxDistanceElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal order for max-heap (largest distance first)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW Index for approximate nearest neighbor search
#[derive(Debug)]
pub struct HnswIndex {
    /// Configuration
    config: HnswConfig,
    /// All nodes in the graph
    nodes: Vec<HnswNode>,
    /// Entry point (index of node at highest layer)
    entry_point: Option<usize>,
    /// Current maximum layer in the graph
    max_layer: usize,
    /// Dimension of vectors
    dimension: usize,
}

impl HnswIndex {
    /// Create a new empty HNSW index
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            dimension,
        }
    }

    /// Create with default configuration
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, HnswConfig::default())
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the dimension of vectors
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Generate a random layer for a new node
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.random_f64() < self.config.ml && level < 32 {
            level += 1;
        }
        level
    }

    /// Compute distance between a query and a node
    fn distance(&self, query: &[f32], node_idx: usize) -> f32 {
        self.config
            .metric
            .distance(query, &self.nodes[node_idx].vector)
    }

    /// Insert a vector into the index
    /// Returns the index of the inserted vector
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimension,
            vector.len()
        );

        let new_idx = self.nodes.len();
        let new_layer = self.random_layer();

        // Create new node with empty neighbor lists
        let mut new_node = HnswNode {
            vector,
            neighbors: vec![Vec::new(); new_layer + 1],
        };

        // Handle first node
        if self.nodes.is_empty() {
            self.nodes.push(new_node);
            self.entry_point = Some(0);
            self.max_layer = new_layer;
            return 0;
        }

        let entry_point = self.entry_point.unwrap();
        let query = &new_node.vector;

        // Search from top layer down to layer new_layer + 1
        let mut current_ep = entry_point;

        for layer in (new_layer + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer);
        }

        // Search from min(new_layer, max_layer) down to 0, collecting neighbors
        let search_start_layer = new_layer.min(self.max_layer);

        for layer in (0..=search_start_layer).rev() {
            let candidates =
                self.search_layer(query, current_ep, self.config.ef_construction, layer);

            // Select neighbors using heuristic
            let max_connections = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors_heuristic(query, &candidates, max_connections);

            // Store neighbors for this layer
            new_node.neighbors[layer] = neighbors.clone();

            // Update entry point for next layer
            if let Some(nearest) = candidates.first() {
                current_ep = nearest.index;
            }
        }

        // Add the new node
        self.nodes.push(new_node);

        // Add bidirectional connections
        for layer in 0..=search_start_layer.min(new_layer) {
            let neighbors = self.nodes[new_idx].neighbors[layer].clone();
            let max_connections = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            for &neighbor_idx in &neighbors {
                self.add_connection(neighbor_idx, new_idx, layer, max_connections);
            }
        }

        // Update entry point if new node is at a higher layer
        if new_layer > self.max_layer {
            self.entry_point = Some(new_idx);
            self.max_layer = new_layer;
        }

        new_idx
    }

    /// Search layer for single nearest neighbor (greedy)
    fn search_layer_single(&self, query: &[f32], entry_point: usize, layer: usize) -> usize {
        let mut current = entry_point;
        let mut current_dist = self.distance(query, current);

        loop {
            let mut changed = false;

            // Check all neighbors at this layer
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    let dist = self.distance(query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search layer for ef nearest neighbors
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<DistanceElement> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<DistanceElement> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxDistanceElement> = BinaryHeap::new();

        let entry_dist = self.distance(query, entry_point);
        visited.insert(entry_point);
        candidates.push(DistanceElement {
            distance: entry_dist,
            index: entry_point,
        });
        results.push(MaxDistanceElement {
            distance: entry_dist,
            index: entry_point,
        });

        while let Some(current) = candidates.pop() {
            // Get furthest result distance
            let furthest_dist = results.peek().map(|r| r.distance).unwrap_or(f32::MAX);

            if current.distance > furthest_dist {
                break;
            }

            // Explore neighbors
            if layer < self.nodes[current.index].neighbors.len() {
                for &neighbor in &self.nodes[current.index].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let dist = self.distance(query, neighbor);
                        let furthest = results.peek().map(|r| r.distance).unwrap_or(f32::MAX);

                        if dist < furthest || results.len() < ef {
                            candidates.push(DistanceElement {
                                distance: dist,
                                index: neighbor,
                            });
                            results.push(MaxDistanceElement {
                                distance: dist,
                                index: neighbor,
                            });

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted vector
        let mut result_vec: Vec<DistanceElement> = results
            .into_iter()
            .map(|r| DistanceElement {
                distance: r.distance,
                index: r.index,
            })
            .collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }

    /// Select neighbors using the heuristic algorithm
    fn select_neighbors_heuristic(
        &self,
        _query: &[f32],
        candidates: &[DistanceElement],
        m: usize,
    ) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|c| c.index).collect();
        }

        let mut selected = Vec::with_capacity(m);
        let mut remaining: Vec<&DistanceElement> = candidates.iter().collect();

        while selected.len() < m && !remaining.is_empty() {
            // Find closest candidate
            let (best_idx, _) = remaining
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap();

            let best = remaining.remove(best_idx);

            // Check if this candidate is closer to query than to any selected node
            let mut is_good = true;
            let best_node: &HnswNode = &self.nodes[best.index];
            for &sel_idx in &selected {
                let sel_node: &HnswNode = &self.nodes[sel_idx];
                let dist_to_selected = self
                    .config
                    .metric
                    .distance(&best_node.vector, &sel_node.vector);
                if dist_to_selected < best.distance {
                    is_good = false;
                    break;
                }
            }

            if is_good {
                selected.push(best.index);
            }
        }

        // If we don't have enough, add remaining by distance
        if selected.len() < m {
            remaining.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            for candidate in remaining {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&candidate.index) {
                    selected.push(candidate.index);
                }
            }
        }

        selected
    }

    /// Add a connection from node to neighbor at given layer
    fn add_connection(
        &mut self,
        node_idx: usize,
        neighbor_idx: usize,
        layer: usize,
        max_connections: usize,
    ) {
        // Ensure the node has neighbors vector for this layer
        while self.nodes[node_idx].neighbors.len() <= layer {
            self.nodes[node_idx].neighbors.push(Vec::new());
        }

        // Don't add duplicate
        if self.nodes[node_idx].neighbors[layer].contains(&neighbor_idx) {
            return;
        }

        self.nodes[node_idx].neighbors[layer].push(neighbor_idx);

        // Prune if too many connections
        if self.nodes[node_idx].neighbors[layer].len() > max_connections {
            // Compute distances and keep closest
            let query = self.nodes[node_idx].vector.clone();
            let neighbor_indices: Vec<usize> = self.nodes[node_idx].neighbors[layer].clone();

            let with_dist: Vec<DistanceElement> = neighbor_indices
                .iter()
                .map(|&idx| DistanceElement {
                    distance: self.config.metric.distance(&query, &self.nodes[idx].vector),
                    index: idx,
                })
                .collect();

            let selected = self.select_neighbors_heuristic(&query, &with_dist, max_connections);
            self.nodes[node_idx].neighbors[layer] = selected;
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension mismatch: expected {}, got {}",
            self.dimension,
            query.len()
        );

        let entry_point = self.entry_point.unwrap();

        // Search from top layer down to layer 1
        let mut current_ep = entry_point;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer);
        }

        // Search layer 0 with ef_search
        let ef = self.config.ef_search.max(k);
        let results = self.search_layer(query, current_ep, ef, 0);

        // Return top k
        results
            .into_iter()
            .take(k)
            .map(|r| (r.index, r.distance))
            .collect()
    }

    /// Search with custom ef parameter
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();

        let mut current_ep = entry_point;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer);
        }

        let results = self.search_layer(query, current_ep, ef.max(k), 0);

        results
            .into_iter()
            .take(k)
            .map(|r| (r.index, r.distance))
            .collect()
    }

    /// Get the vector at a given index
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        self.nodes.get(index).map(|n| n.vector.as_slice())
    }

    /// Bulk insert multiple vectors
    pub fn insert_batch(&mut self, vectors: Vec<Vec<f32>>) -> Vec<usize> {
        vectors.into_iter().map(|v| self.insert(v)).collect()
    }
}

/// Result of a nearest neighbor search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index of the vector in the index
    pub index: usize,
    /// Distance to the query
    pub distance: f32,
}

/// Convenience function to create an HNSW index and insert vectors
pub fn build_hnsw_index(vectors: &[Vec<f32>], config: HnswConfig) -> HnswIndex {
    if vectors.is_empty() {
        panic!("Cannot build index from empty vector set");
    }

    let dimension = vectors[0].len();
    let mut index = HnswIndex::new(dimension, config);

    for vector in vectors {
        index.insert(vector.clone());
    }

    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![2.0, 2.0],
            vec![3.0, 0.0],
            vec![0.0, 3.0],
        ]
    }

    #[test]
    fn test_hnsw_insert_single() {
        let mut index = HnswIndex::with_dimension(2);
        let idx = index.insert(vec![1.0, 2.0]);
        assert_eq!(idx, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hnsw_insert_multiple() {
        let vectors = create_test_vectors();
        let mut index = HnswIndex::with_dimension(2);

        for (i, v) in vectors.iter().enumerate() {
            let idx = index.insert(v.clone());
            assert_eq!(idx, i);
        }

        assert_eq!(index.len(), vectors.len());
    }

    #[test]
    fn test_hnsw_search_exact() {
        let vectors = create_test_vectors();
        let index = build_hnsw_index(&vectors, HnswConfig::default());

        // Search for exact match
        let results = index.search(&[0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // Should find index 0
        assert!(results[0].1 < 1e-6); // Distance should be ~0
    }

    #[test]
    fn test_hnsw_search_k_nearest() {
        let vectors = create_test_vectors();
        let index = build_hnsw_index(&vectors, HnswConfig::default());

        // Search for 3 nearest to origin
        let results = index.search(&[0.0, 0.0], 3);
        assert_eq!(results.len(), 3);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }

        // First result should be exact match
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hnsw_search_all() {
        let vectors = create_test_vectors();
        let index = build_hnsw_index(&vectors, HnswConfig::default());

        // Search for more than available
        let results = index.search(&[0.5, 0.5], 100);
        assert_eq!(results.len(), vectors.len());
    }

    #[test]
    fn test_hnsw_empty_search() {
        let index = HnswIndex::with_dimension(2);
        let results = index.search(&[0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = HnswConfig::default().metric(DistanceMetric::Cosine);
        let mut index = HnswIndex::new(2, config);

        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![1.0, 1.0]);

        // Search with a vector pointing in same direction as [1, 0]
        let results = index.search(&[2.0, 0.0], 1);
        assert_eq!(results[0].0, 0); // Should find [1, 0]
        assert!(results[0].1 < 1e-6); // Cosine distance should be ~0
    }

    #[test]
    fn test_hnsw_get_vector() {
        let mut index = HnswIndex::with_dimension(3);
        index.insert(vec![1.0, 2.0, 3.0]);

        let v = index.get_vector(0);
        assert!(v.is_some());
        assert_eq!(v.unwrap(), &[1.0, 2.0, 3.0]);

        assert!(index.get_vector(1).is_none());
    }

    #[test]
    fn test_hnsw_batch_insert() {
        let vectors = create_test_vectors();
        let mut index = HnswIndex::with_dimension(2);

        let indices = index.insert_batch(vectors.clone());
        assert_eq!(indices.len(), vectors.len());
        assert_eq!(index.len(), vectors.len());
    }

    #[test]
    fn test_hnsw_high_dimensional() {
        let dim = 128;
        let n = 100;

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random_f32()).collect())
            .collect();

        let index = build_hnsw_index(&vectors, HnswConfig::with_m(32));

        // Search should work
        let query: Vec<f32> = (0..dim).map(|_| rng.random_f32()).collect();
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);

        // Results should be sorted
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_hnsw_recall() {
        // Test that HNSW achieves reasonable recall
        let dim = 8;
        let n = 200;

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random_f32()).collect())
            .collect();

        let config = HnswConfig::with_m(16).ef_construction(100).ef_search(50);
        let index = build_hnsw_index(&vectors, config);

        // Use first vector as query
        let query = &vectors[0];
        let results = index.search(query, 1);

        // Should find the exact vector
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_hnsw_search_with_ef() {
        let vectors = create_test_vectors();
        let index = build_hnsw_index(&vectors, HnswConfig::default());

        // Higher ef should give same or better results
        let results_low = index.search_with_ef(&[0.25, 0.25], 3, 10);
        let results_high = index.search_with_ef(&[0.25, 0.25], 3, 100);

        assert_eq!(results_low.len(), 3);
        assert_eq!(results_high.len(), 3);

        // First result distance should be same or better with higher ef
        assert!(results_high[0].1 <= results_low[0].1 + 1e-6);
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_hnsw_dimension_mismatch_insert() {
        let mut index = HnswIndex::with_dimension(3);
        index.insert(vec![1.0, 2.0]); // Wrong dimension
    }

    #[test]
    #[should_panic(expected = "Query dimension mismatch")]
    fn test_hnsw_dimension_mismatch_search() {
        let mut index = HnswIndex::with_dimension(3);
        index.insert(vec![1.0, 2.0, 3.0]);
        index.search(&[1.0, 2.0], 1); // Wrong dimension
    }

    #[test]
    fn test_config_builder() {
        let config = HnswConfig::with_m(32)
            .metric(DistanceMetric::Cosine)
            .ef_construction(300)
            .ef_search(100);

        assert_eq!(config.m, 32);
        assert_eq!(config.m_max0, 64);
        assert_eq!(config.ef_construction, 300);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }
}
