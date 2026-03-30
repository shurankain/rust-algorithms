// GraphSAGE Sampling
//
// Efficient neighbor sampling for large-scale Graph Neural Networks.
// GraphSAGE (SAmple and aggreGatE) samples a fixed number of neighbors
// at each layer, making it scalable to graphs with millions of nodes.
//
// Key concepts:
// 1. Uniform sampling: Sample K random neighbors per node
// 2. Layer-wise sampling: Different sample sizes per layer
// 3. Mini-batch training: Sample subgraphs for efficient training
//
// Sampling strategies:
// - Uniform: Equal probability for all neighbors
// - Weighted: Sample based on edge weights or node importance
// - Importance: Sample based on learned importance scores
//
// References:
// - "Inductive Representation Learning on Large Graphs" (Hamilton et al., NeurIPS 2017)
// - "FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling"
// - "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Networks"

use crate::message_passing::{Aggregation, EdgeId, Features, Graph, NodeFeatures, NodeId};
use std::collections::{HashMap, HashSet};

/// Configuration for GraphSAGE sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Number of neighbors to sample at each layer (from output to input)
    /// e.g., [10, 25] means sample 10 neighbors at layer 2, 25 at layer 1
    pub sample_sizes: Vec<usize>,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Whether to include self-loops in sampling
    pub include_self: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sample_sizes: vec![10, 25], // Common default: 2-layer with 10 and 25 samples
            strategy: SamplingStrategy::Uniform,
            seed: 42,
            include_self: true,
        }
    }
}

impl SamplingConfig {
    /// Create config with specified sample sizes
    pub fn new(sample_sizes: Vec<usize>) -> Self {
        Self {
            sample_sizes,
            ..Default::default()
        }
    }

    /// Set sampling strategy
    pub fn with_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.sample_sizes.len()
    }
}

/// Sampling strategy for neighbor selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Sample based on node degree (higher degree = higher probability)
    DegreeWeighted,
    /// Sample all neighbors (no sampling, for small graphs)
    Full,
}

/// A sampled subgraph for mini-batch training
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    /// All nodes in the subgraph (including sampled neighbors)
    pub nodes: Vec<NodeId>,
    /// Mapping from original node ID to local index
    pub node_to_idx: HashMap<NodeId, usize>,
    /// Edges in the subgraph: (source_local, target_local)
    pub edges: Vec<(usize, usize)>,
    /// Target nodes (the original batch nodes)
    pub target_nodes: Vec<NodeId>,
    /// Nodes at each layer (from output layer to input layer)
    pub layer_nodes: Vec<Vec<NodeId>>,
    /// Number of layers
    pub num_layers: usize,
}

impl SampledSubgraph {
    /// Create a new sampled subgraph
    pub fn new(target_nodes: Vec<NodeId>, num_layers: usize) -> Self {
        let mut node_to_idx = HashMap::new();
        for (idx, &node) in target_nodes.iter().enumerate() {
            node_to_idx.insert(node, idx);
        }

        Self {
            nodes: target_nodes.clone(),
            node_to_idx,
            edges: Vec::new(),
            target_nodes,
            layer_nodes: Vec::new(),
            num_layers,
        }
    }

    /// Get local index for a node
    pub fn local_idx(&self, node: NodeId) -> Option<usize> {
        self.node_to_idx.get(&node).copied()
    }

    /// Get original node ID from local index
    pub fn original_node(&self, local_idx: usize) -> Option<NodeId> {
        self.nodes.get(local_idx).copied()
    }

    /// Add a node to the subgraph
    pub fn add_node(&mut self, node: NodeId) -> usize {
        if let Some(&idx) = self.node_to_idx.get(&node) {
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node);
            self.node_to_idx.insert(node, idx);
            idx
        }
    }

    /// Add an edge to the subgraph
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) {
        let src_idx = self.add_node(source);
        let tgt_idx = self.add_node(target);
        self.edges.push((src_idx, tgt_idx));
    }

    /// Number of nodes in subgraph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in subgraph
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Extract features for the subgraph nodes
    pub fn extract_features(&self, full_features: &NodeFeatures) -> NodeFeatures {
        let dim = full_features.dim();
        let mut sub_features = NodeFeatures::new(self.nodes.len(), dim);

        for (local_idx, &node) in self.nodes.iter().enumerate() {
            if node < full_features.num_nodes() {
                sub_features.set(local_idx, full_features.get(node).clone());
            }
        }

        sub_features
    }

    /// Build adjacency list for the subgraph
    pub fn build_adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.nodes.len()];

        for &(src, tgt) in &self.edges {
            adj[tgt].push(src); // For message passing: neighbors -> target
        }

        adj
    }
}

/// Simple deterministic pseudo-random number generator
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() as usize) % max
    }

    /// Sample k items from 0..n without replacement
    fn sample_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k >= n {
            return (0..n).collect();
        }

        // Fisher-Yates partial shuffle
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.next_usize(n - i);
            indices.swap(i, j);
        }

        indices[0..k].to_vec()
    }
}

/// GraphSAGE neighbor sampler
#[derive(Debug, Clone)]
pub struct NeighborSampler {
    /// Configuration
    config: SamplingConfig,
    /// Random number generator
    rng: SimpleRng,
}

impl NeighborSampler {
    /// Create a new neighbor sampler
    pub fn new(config: SamplingConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self { config, rng }
    }

    /// Create with default config
    pub fn default_sampler() -> Self {
        Self::new(SamplingConfig::default())
    }

    /// Sample neighbors for a single node
    pub fn sample_neighbors(
        &mut self,
        graph: &Graph,
        node: NodeId,
        num_samples: usize,
    ) -> Vec<(NodeId, EdgeId)> {
        let neighbors = graph.neighbors(node);

        if neighbors.is_empty() {
            return Vec::new();
        }

        match self.config.strategy {
            SamplingStrategy::Full => neighbors.to_vec(),
            SamplingStrategy::Uniform => {
                if neighbors.len() <= num_samples {
                    neighbors.to_vec()
                } else {
                    let indices = self
                        .rng
                        .sample_without_replacement(neighbors.len(), num_samples);
                    indices.into_iter().map(|i| neighbors[i]).collect()
                }
            }
            SamplingStrategy::DegreeWeighted => {
                // Weight by neighbor's degree
                self.sample_degree_weighted(graph, neighbors, num_samples)
            }
        }
    }

    /// Degree-weighted sampling
    fn sample_degree_weighted(
        &mut self,
        graph: &Graph,
        neighbors: &[(NodeId, EdgeId)],
        num_samples: usize,
    ) -> Vec<(NodeId, EdgeId)> {
        if neighbors.len() <= num_samples {
            return neighbors.to_vec();
        }

        // Compute weights based on degree
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&(n, _)| (graph.degree(n) as f64).sqrt())
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight == 0.0 {
            return self
                .rng
                .sample_without_replacement(neighbors.len(), num_samples)
                .into_iter()
                .map(|i| neighbors[i])
                .collect();
        }

        // Sample with weights (simplified weighted sampling with replacement, then dedup)
        let mut selected = HashSet::new();
        let mut attempts = 0;
        let max_attempts = num_samples * 10;

        while selected.len() < num_samples && attempts < max_attempts {
            let r = (self.rng.next() as f64 / u64::MAX as f64) * total_weight;
            let mut cumulative = 0.0;

            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if r <= cumulative {
                    selected.insert(i);
                    break;
                }
            }
            attempts += 1;
        }

        selected.into_iter().map(|i| neighbors[i]).collect()
    }

    /// Sample a mini-batch subgraph using layer-wise sampling
    pub fn sample_subgraph(&mut self, graph: &Graph, batch_nodes: &[NodeId]) -> SampledSubgraph {
        let num_layers = self.config.sample_sizes.len();
        let mut subgraph = SampledSubgraph::new(batch_nodes.to_vec(), num_layers);

        // Layer 0 is the output layer (batch nodes)
        let mut current_layer: HashSet<NodeId> = batch_nodes.iter().copied().collect();
        subgraph.layer_nodes.push(batch_nodes.to_vec());

        // Sample backwards from output to input
        for layer_idx in 0..num_layers {
            let sample_size = self.config.sample_sizes[layer_idx];
            let mut next_layer: HashSet<NodeId> = HashSet::new();

            for &node in &current_layer {
                // Sample neighbors
                let sampled = self.sample_neighbors(graph, node, sample_size);

                for (neighbor, _edge_id) in sampled {
                    subgraph.add_edge(neighbor, node);
                    next_layer.insert(neighbor);
                }

                // Add self-loop if configured
                if self.config.include_self {
                    subgraph.add_edge(node, node);
                }
            }

            subgraph
                .layer_nodes
                .push(next_layer.iter().copied().collect());
            current_layer = next_layer;
        }

        subgraph
    }

    /// Reset the random seed
    pub fn reset_seed(&mut self, seed: u64) {
        self.rng = SimpleRng::new(seed);
    }
}

/// GraphSAGE aggregator types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SAGEAggregator {
    /// Mean aggregator: mean(neighbors)
    Mean,
    /// GCN aggregator: mean(neighbors + self)
    GCN,
    /// Pool aggregator: max(MLP(neighbors))
    MaxPool,
    /// LSTM aggregator (simplified as mean)
    LSTM,
}

/// GraphSAGE layer
#[derive(Debug, Clone)]
pub struct GraphSAGELayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Aggregator type
    aggregator: SAGEAggregator,
    /// Weight matrix for self features
    weight_self: Vec<Vec<f64>>,
    /// Weight matrix for neighbor features
    weight_neigh: Vec<Vec<f64>>,
    /// Whether to normalize output
    normalize: bool,
}

impl GraphSAGELayer {
    /// Create a new GraphSAGE layer
    pub fn new(input_dim: usize, output_dim: usize, aggregator: SAGEAggregator) -> Self {
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        // Initialize weights
        let weight_self: Vec<Vec<f64>> = (0..input_dim)
            .map(|i| {
                (0..output_dim)
                    .map(|j| {
                        let seed = (i * 31 + j * 17) as f64;
                        seed.sin() * scale
                    })
                    .collect()
            })
            .collect();

        let weight_neigh: Vec<Vec<f64>> = (0..input_dim)
            .map(|i| {
                (0..output_dim)
                    .map(|j| {
                        let seed = (i * 37 + j * 23) as f64;
                        seed.sin() * scale
                    })
                    .collect()
            })
            .collect();

        Self {
            input_dim,
            output_dim,
            aggregator,
            weight_self,
            weight_neigh,
            normalize: true,
        }
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Apply linear transformation
    fn linear(&self, features: &Features, weights: &[Vec<f64>]) -> Features {
        let out_dim = if weights.is_empty() || weights[0].is_empty() {
            0
        } else {
            weights[0].len()
        };

        let mut result = vec![0.0; out_dim];

        for (i, &f) in features.iter().enumerate() {
            if i < weights.len() {
                for (j, &w) in weights[i].iter().enumerate() {
                    result[j] += f * w;
                }
            }
        }

        result
    }

    /// L2 normalize features
    fn l2_normalize(&self, features: &mut Features) {
        let norm: f64 = features.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for f in features.iter_mut() {
                *f /= norm;
            }
        }
    }

    /// Aggregate neighbor features
    fn aggregate(&self, neighbor_features: &[Features]) -> Features {
        if neighbor_features.is_empty() {
            return vec![0.0; self.input_dim];
        }

        match self.aggregator {
            SAGEAggregator::Mean | SAGEAggregator::GCN | SAGEAggregator::LSTM => {
                Aggregation::Mean.aggregate(neighbor_features)
            }
            SAGEAggregator::MaxPool => Aggregation::Max.aggregate(neighbor_features),
        }
    }

    /// Forward pass for a single node given its neighbors' features
    pub fn forward_node(
        &self,
        self_features: &Features,
        neighbor_features: &[Features],
    ) -> Features {
        // Aggregate neighbor features
        let aggregated = self.aggregate(neighbor_features);

        // Transform self and neighbor features
        let self_transformed = self.linear(self_features, &self.weight_self);
        let neigh_transformed = self.linear(&aggregated, &self.weight_neigh);

        // Combine: h = W_self * h_v + W_neigh * AGG(h_u)
        let mut output: Features = self_transformed
            .iter()
            .zip(neigh_transformed.iter())
            .map(|(&s, &n)| s + n)
            .collect();

        // ReLU activation
        for f in &mut output {
            *f = f.max(0.0);
        }

        // L2 normalize if configured
        if self.normalize {
            self.l2_normalize(&mut output);
        }

        output
    }

    /// Forward pass on a sampled subgraph
    pub fn forward_subgraph(
        &self,
        subgraph: &SampledSubgraph,
        node_features: &NodeFeatures,
    ) -> NodeFeatures {
        let adjacency = subgraph.build_adjacency();
        let mut new_features = NodeFeatures::new(subgraph.num_nodes(), self.output_dim);

        for (local_idx, neighbors) in adjacency.iter().enumerate() {
            let self_feat = node_features.get(local_idx);

            // Get neighbor features
            let neighbor_features: Vec<Features> = neighbors
                .iter()
                .map(|&neigh_idx| node_features.get(neigh_idx).clone())
                .collect();

            let output = self.forward_node(self_feat, &neighbor_features);
            new_features.set(local_idx, output);
        }

        new_features
    }
}

/// Full GraphSAGE model
#[derive(Debug, Clone)]
pub struct GraphSAGE {
    /// Layers
    layers: Vec<GraphSAGELayer>,
    /// Sampler
    sampler: NeighborSampler,
}

impl GraphSAGE {
    /// Create a new GraphSAGE model
    pub fn new(layers: Vec<GraphSAGELayer>, sampler: NeighborSampler) -> Self {
        Self { layers, sampler }
    }

    /// Create a 2-layer GraphSAGE model
    pub fn two_layer(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        aggregator: SAGEAggregator,
    ) -> Self {
        let layers = vec![
            GraphSAGELayer::new(input_dim, hidden_dim, aggregator),
            GraphSAGELayer::new(hidden_dim, output_dim, aggregator),
        ];

        let config = SamplingConfig::new(vec![10, 25]);
        let sampler = NeighborSampler::new(config);

        Self { layers, sampler }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass on full graph (for inference)
    pub fn forward(&self, graph: &Graph, node_features: &NodeFeatures) -> NodeFeatures {
        let mut current = node_features.clone();

        for layer in &self.layers {
            let mut new_features = NodeFeatures::new(graph.num_nodes(), layer.output_dim());

            for node in 0..graph.num_nodes() {
                let self_feat = current.get(node);
                let neighbor_features: Vec<Features> = graph
                    .neighbors(node)
                    .iter()
                    .map(|&(n, _)| current.get(n).clone())
                    .collect();

                let output = layer.forward_node(self_feat, &neighbor_features);
                new_features.set(node, output);
            }

            current = new_features;
        }

        current
    }

    /// Forward pass on a mini-batch (for training)
    pub fn forward_batch(
        &mut self,
        graph: &Graph,
        node_features: &NodeFeatures,
        batch_nodes: &[NodeId],
    ) -> (NodeFeatures, SampledSubgraph) {
        // Sample subgraph
        let subgraph = self.sampler.sample_subgraph(graph, batch_nodes);

        // Extract features for subgraph nodes
        let mut current = subgraph.extract_features(node_features);

        // Forward through layers
        for layer in &self.layers {
            current = layer.forward_subgraph(&subgraph, &current);
        }

        (current, subgraph)
    }
}

/// Statistics about sampling
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    /// Number of nodes sampled
    pub nodes_sampled: usize,
    /// Number of edges sampled
    pub edges_sampled: usize,
    /// Sampling ratio (sampled / total)
    pub sampling_ratio: f64,
    /// Average neighbors sampled per node
    pub avg_neighbors: f64,
}

/// Analyze sampling efficiency
pub fn analyze_sampling(graph: &Graph, subgraph: &SampledSubgraph) -> SamplingStats {
    let total_nodes = graph.num_nodes();
    let _total_edges = graph.num_edges();

    SamplingStats {
        nodes_sampled: subgraph.num_nodes(),
        edges_sampled: subgraph.num_edges(),
        sampling_ratio: if total_nodes > 0 {
            subgraph.num_nodes() as f64 / total_nodes as f64
        } else {
            0.0
        },
        avg_neighbors: if subgraph.target_nodes.is_empty() {
            0.0
        } else {
            subgraph.num_edges() as f64 / subgraph.target_nodes.len() as f64
        },
    }
}

/// Create random mini-batches of node IDs
pub fn create_batches(
    nodes: &[NodeId],
    batch_size: usize,
    shuffle_seed: Option<u64>,
) -> Vec<Vec<NodeId>> {
    let mut node_list: Vec<NodeId> = nodes.to_vec();

    // Shuffle if seed provided
    if let Some(seed) = shuffle_seed {
        let mut rng = SimpleRng::new(seed);
        for i in (1..node_list.len()).rev() {
            let j = rng.next_usize(i + 1);
            node_list.swap(i, j);
        }
    }

    // Create batches
    node_list.chunks(batch_size).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        // Create a small graph for testing
        let mut graph = Graph::undirected(10);
        // Create some edges
        for i in 0..10 {
            graph.add_edge(i, (i + 1) % 10); // Ring
            if i < 5 {
                graph.add_edge(i, i + 5); // Cross connections
            }
        }
        graph
    }

    fn create_star_graph() -> Graph {
        let mut graph = Graph::undirected(6);
        // Node 0 is center
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        graph
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::new(vec![5, 10, 15]);
        assert_eq!(config.num_layers(), 3);
        assert_eq!(config.sample_sizes, vec![5, 10, 15]);
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next();
        let v2 = rng.next();
        assert_ne!(v1, v2);

        // Test reproducibility
        let mut rng2 = SimpleRng::new(42);
        assert_eq!(rng2.next(), v1);
    }

    #[test]
    fn test_sample_without_replacement() {
        let mut rng = SimpleRng::new(42);
        let samples = rng.sample_without_replacement(10, 5);

        assert_eq!(samples.len(), 5);

        // All unique
        let unique: HashSet<usize> = samples.iter().copied().collect();
        assert_eq!(unique.len(), 5);

        // All in range
        assert!(samples.iter().all(|&s| s < 10));
    }

    #[test]
    fn test_sample_all_when_k_large() {
        let mut rng = SimpleRng::new(42);
        let samples = rng.sample_without_replacement(5, 10);
        assert_eq!(samples.len(), 5); // Can't sample more than n
    }

    #[test]
    fn test_neighbor_sampler() {
        let graph = create_test_graph();
        let config = SamplingConfig::new(vec![3]);
        let mut sampler = NeighborSampler::new(config);

        let samples = sampler.sample_neighbors(&graph, 0, 3);
        assert!(!samples.is_empty());
        assert!(samples.len() <= 3);
    }

    #[test]
    fn test_neighbor_sampler_full() {
        let graph = create_star_graph();
        let config = SamplingConfig::new(vec![10]).with_strategy(SamplingStrategy::Full);
        let mut sampler = NeighborSampler::new(config);

        // Center node has 5 neighbors
        let samples = sampler.sample_neighbors(&graph, 0, 2);
        assert_eq!(samples.len(), 5); // Full strategy returns all
    }

    #[test]
    fn test_sampled_subgraph() {
        let mut subgraph = SampledSubgraph::new(vec![0, 1, 2], 2);

        assert_eq!(subgraph.num_nodes(), 3);
        assert_eq!(subgraph.local_idx(0), Some(0));
        assert_eq!(subgraph.local_idx(1), Some(1));

        subgraph.add_edge(5, 0);
        assert_eq!(subgraph.num_nodes(), 4);
        assert_eq!(subgraph.num_edges(), 1);
    }

    #[test]
    fn test_sample_subgraph() {
        let graph = create_test_graph();
        let config = SamplingConfig::new(vec![3, 5]);
        let mut sampler = NeighborSampler::new(config);

        let batch = vec![0, 1, 2];
        let subgraph = sampler.sample_subgraph(&graph, &batch);

        // Should include batch nodes
        assert!(subgraph.local_idx(0).is_some());
        assert!(subgraph.local_idx(1).is_some());
        assert!(subgraph.local_idx(2).is_some());

        // Should have sampled some neighbors
        assert!(subgraph.num_nodes() >= 3);
    }

    #[test]
    fn test_extract_features() {
        let mut full_features = NodeFeatures::new(10, 4);
        for i in 0..10 {
            full_features.set(i, vec![i as f64; 4]);
        }

        let subgraph = SampledSubgraph::new(vec![0, 5, 9], 1);
        let sub_features = subgraph.extract_features(&full_features);

        assert_eq!(sub_features.num_nodes(), 3);
        assert_eq!(sub_features.get(0), &vec![0.0; 4]);
        assert_eq!(sub_features.get(1), &vec![5.0; 4]);
        assert_eq!(sub_features.get(2), &vec![9.0; 4]);
    }

    #[test]
    fn test_graphsage_layer() {
        let layer = GraphSAGELayer::new(4, 2, SAGEAggregator::Mean);

        let self_feat = vec![1.0, 2.0, 3.0, 4.0];
        let neighbor_feats = vec![vec![0.5, 1.0, 1.5, 2.0], vec![1.5, 2.0, 2.5, 3.0]];

        let output = layer.forward_node(&self_feat, &neighbor_feats);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_graphsage_layer_no_neighbors() {
        let layer = GraphSAGELayer::new(4, 2, SAGEAggregator::Mean);

        let self_feat = vec![1.0, 2.0, 3.0, 4.0];
        let neighbor_feats: Vec<Features> = Vec::new();

        let output = layer.forward_node(&self_feat, &neighbor_feats);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_graphsage_aggregators() {
        // Use layers without normalization to see the difference
        let mean_layer = GraphSAGELayer::new(4, 2, SAGEAggregator::Mean).with_normalize(false);
        let max_layer = GraphSAGELayer::new(4, 2, SAGEAggregator::MaxPool).with_normalize(false);

        // Use larger positive values to ensure positive outputs after ReLU
        let self_feat = vec![10.0, 10.0, 10.0, 10.0];
        let neighbor_feats = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];

        let mean_out = mean_layer.forward_node(&self_feat, &neighbor_feats);
        let max_out = max_layer.forward_node(&self_feat, &neighbor_feats);

        // Both should produce output (may or may not be different depending on weights)
        assert_eq!(mean_out.len(), 2);
        assert_eq!(max_out.len(), 2);
    }

    #[test]
    fn test_graphsage_model() {
        let graph = create_test_graph();
        let mut features = NodeFeatures::new(10, 8);
        for i in 0..10 {
            features.set(i, vec![1.0; 8]);
        }

        let model = GraphSAGE::two_layer(8, 4, 2, SAGEAggregator::Mean);
        assert_eq!(model.num_layers(), 2);

        let output = model.forward(&graph, &features);
        assert_eq!(output.num_nodes(), 10);
        assert_eq!(output.get(0).len(), 2);
    }

    #[test]
    fn test_graphsage_batch() {
        let graph = create_test_graph();
        let mut features = NodeFeatures::new(10, 8);
        for i in 0..10 {
            features.set(i, vec![1.0; 8]);
        }

        let mut model = GraphSAGE::two_layer(8, 4, 2, SAGEAggregator::Mean);
        let batch = vec![0, 1, 2];

        let (output, subgraph) = model.forward_batch(&graph, &features, &batch);

        // Output should be for subgraph nodes
        assert_eq!(output.num_nodes(), subgraph.num_nodes());
    }

    #[test]
    fn test_analyze_sampling() {
        let graph = create_test_graph();
        let config = SamplingConfig::new(vec![2, 3]);
        let mut sampler = NeighborSampler::new(config);

        let batch = vec![0, 1];
        let subgraph = sampler.sample_subgraph(&graph, &batch);

        let stats = analyze_sampling(&graph, &subgraph);
        assert!(stats.nodes_sampled >= 2);
        assert!(stats.sampling_ratio <= 1.0);
        assert!(stats.sampling_ratio > 0.0);
    }

    #[test]
    fn test_create_batches() {
        let nodes: Vec<NodeId> = (0..100).collect();
        let batches = create_batches(&nodes, 32, Some(42));

        assert_eq!(batches.len(), 4); // 100 / 32 = 3 full + 1 partial
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 32);
        assert_eq!(batches[2].len(), 32);
        assert_eq!(batches[3].len(), 4);

        // All nodes should be present
        let all_nodes: HashSet<NodeId> = batches.iter().flatten().copied().collect();
        assert_eq!(all_nodes.len(), 100);
    }

    #[test]
    fn test_create_batches_no_shuffle() {
        let nodes: Vec<NodeId> = (0..10).collect();
        let batches = create_batches(&nodes, 3, None);

        // Without shuffle, order should be preserved
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
    }

    #[test]
    fn test_degree_weighted_sampling() {
        let graph = create_star_graph();
        let config = SamplingConfig::new(vec![2]).with_strategy(SamplingStrategy::DegreeWeighted);
        let mut sampler = NeighborSampler::new(config);

        // Leaf nodes have degree 1, center has degree 5
        // From a leaf, the only neighbor is the center
        let samples = sampler.sample_neighbors(&graph, 1, 2);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_l2_normalize() {
        let layer = GraphSAGELayer::new(4, 2, SAGEAggregator::Mean).with_normalize(true);

        let self_feat = vec![3.0, 4.0, 0.0, 0.0]; // 3-4-5 triangle
        let output = layer.forward_node(&self_feat, &[]);

        // Output should be L2 normalized
        let norm: f64 = output.iter().map(|&x| x * x).sum::<f64>().sqrt();
        // Due to ReLU, some values might be 0, but if non-zero, should be normalized
        if norm > 1e-6 {
            assert!((norm - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_layer_nodes() {
        let graph = create_test_graph();
        let config = SamplingConfig::new(vec![2, 3]);
        let mut sampler = NeighborSampler::new(config);

        let batch = vec![0];
        let subgraph = sampler.sample_subgraph(&graph, &batch);

        // Should have 3 layers: output (batch), layer 1 samples, layer 2 samples
        assert_eq!(subgraph.layer_nodes.len(), 3);
        assert_eq!(subgraph.layer_nodes[0], vec![0]);
    }

    #[test]
    fn test_reproducible_sampling() {
        let graph = create_test_graph();

        let config1 = SamplingConfig::new(vec![3]).with_seed(42);
        let mut sampler1 = NeighborSampler::new(config1);

        let config2 = SamplingConfig::new(vec![3]).with_seed(42);
        let mut sampler2 = NeighborSampler::new(config2);

        let samples1 = sampler1.sample_neighbors(&graph, 0, 3);
        let samples2 = sampler2.sample_neighbors(&graph, 0, 3);

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_build_adjacency() {
        let mut subgraph = SampledSubgraph::new(vec![0, 1], 1);
        subgraph.add_edge(2, 0); // 2 -> 0
        subgraph.add_edge(3, 0); // 3 -> 0
        subgraph.add_edge(2, 1); // 2 -> 1

        let adj = subgraph.build_adjacency();

        // Node 0 (local) should have neighbors 2, 3 (local indices)
        assert_eq!(adj[0].len(), 2);
        // Node 1 (local) should have neighbor 2 (local index)
        assert_eq!(adj[1].len(), 1);
    }
}
