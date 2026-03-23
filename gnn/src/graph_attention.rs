// Graph Attention Networks (GAT)
//
// Attention-weighted message aggregation for GNNs.
// Instead of treating all neighbors equally, GAT learns to assign
// different importance weights to different neighbors.
//
// Key concepts:
// 1. Attention coefficients: e_ij = a(W*h_i, W*h_j)
// 2. Softmax normalization: α_ij = softmax_j(e_ij)
// 3. Weighted aggregation: h'_i = σ(Σ α_ij * W * h_j)
//
// Multi-head attention:
// - Run K independent attention mechanisms
// - Concatenate or average the outputs
// - Improves stability and expressive power
//
// References:
// - "Graph Attention Networks" (Veličković et al., ICLR 2018)
// - "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022)
// - GATv2: Fixes the static attention problem of original GAT

use crate::message_passing::{EdgeId, Features, Graph, NodeFeatures, NodeId};

/// Configuration for Graph Attention layer
#[derive(Debug, Clone)]
pub struct GATConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Output feature dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Whether to concatenate heads (true) or average (false)
    pub concat_heads: bool,
    /// Dropout rate for attention coefficients (0.0 = no dropout)
    pub dropout: f64,
    /// Negative slope for LeakyReLU
    pub negative_slope: f64,
    /// Whether to add self-loops
    pub add_self_loops: bool,
    /// Whether to use GATv2 (dynamic attention)
    pub use_gatv2: bool,
}

impl Default for GATConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            output_dim: 64,
            num_heads: 8,
            concat_heads: true,
            dropout: 0.0,
            negative_slope: 0.2,
            add_self_loops: true,
            use_gatv2: false,
        }
    }
}

impl GATConfig {
    /// Create a new config with specified dimensions
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            ..Default::default()
        }
    }

    /// Set number of attention heads
    pub fn with_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set whether to concatenate heads
    pub fn with_concat(mut self, concat: bool) -> Self {
        self.concat_heads = concat;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable GATv2 (dynamic attention)
    pub fn with_gatv2(mut self) -> Self {
        self.use_gatv2 = true;
        self
    }

    /// Get the output dimension considering multi-head
    pub fn effective_output_dim(&self) -> usize {
        if self.concat_heads {
            self.output_dim * self.num_heads
        } else {
            self.output_dim
        }
    }
}

/// A single attention head
#[derive(Debug, Clone)]
pub struct AttentionHead {
    /// Linear transformation weights (input_dim x output_dim)
    weights: Vec<Vec<f64>>,
    /// Attention vector for source nodes
    attention_src: Vec<f64>,
    /// Attention vector for target nodes
    attention_dst: Vec<f64>,
    /// Output dimension
    output_dim: usize,
    /// Negative slope for LeakyReLU
    negative_slope: f64,
    /// Whether to use GATv2
    use_gatv2: bool,
}

impl AttentionHead {
    /// Create a new attention head with deterministic initialization
    pub fn new(input_dim: usize, output_dim: usize, head_idx: usize, use_gatv2: bool) -> Self {
        // Xavier initialization with deterministic "random" values
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|i| {
                (0..output_dim)
                    .map(|j| {
                        let seed = ((i * 31 + j * 17 + head_idx * 13) as f64).sin();
                        seed * scale
                    })
                    .collect()
            })
            .collect();

        // Attention vectors
        let attention_src: Vec<f64> = (0..output_dim)
            .map(|i| {
                let seed = ((i * 23 + head_idx * 7) as f64).sin();
                seed * scale
            })
            .collect();

        let attention_dst: Vec<f64> = (0..output_dim)
            .map(|i| {
                let seed = ((i * 29 + head_idx * 11) as f64).sin();
                seed * scale
            })
            .collect();

        Self {
            weights,
            attention_src,
            attention_dst,
            output_dim,
            negative_slope: 0.2,
            use_gatv2,
        }
    }

    /// Apply linear transformation to features
    fn linear_transform(&self, features: &Features) -> Features {
        let mut result = vec![0.0; self.output_dim];

        for (i, &f) in features.iter().enumerate() {
            if i < self.weights.len() {
                for (j, &w) in self.weights[i].iter().enumerate() {
                    result[j] += f * w;
                }
            }
        }

        result
    }

    /// Compute attention coefficient e_ij (before softmax)
    fn compute_attention_coef(&self, source: &Features, target: &Features) -> f64 {
        if self.use_gatv2 {
            // GATv2: Apply attention after summing transformed features
            // a^T * LeakyReLU(W * [h_i || h_j])
            let sum: f64 = source
                .iter()
                .zip(target.iter())
                .zip(self.attention_src.iter())
                .map(|((&s, &t), &a)| {
                    let combined = s + t;
                    self.leaky_relu(combined) * a
                })
                .sum();
            sum
        } else {
            // Original GAT: LeakyReLU(a^T * [W*h_i || W*h_j])
            let src_attention: f64 = source
                .iter()
                .zip(self.attention_src.iter())
                .map(|(&f, &a)| f * a)
                .sum();

            let dst_attention: f64 = target
                .iter()
                .zip(self.attention_dst.iter())
                .map(|(&f, &a)| f * a)
                .sum();

            self.leaky_relu(src_attention + dst_attention)
        }
    }

    /// LeakyReLU activation
    fn leaky_relu(&self, x: f64) -> f64 {
        if x >= 0.0 { x } else { self.negative_slope * x }
    }

    /// Compute attention for a node given its neighbors
    pub fn compute_node_attention(
        &self,
        node_features: &NodeFeatures,
        node: NodeId,
        neighbors: &[(NodeId, EdgeId)],
        include_self: bool,
    ) -> (Vec<f64>, Vec<Features>) {
        let target = node_features.get(node);
        let target_transformed = self.linear_transform(target);

        // Collect all sources (neighbors + potentially self)
        let mut sources: Vec<(NodeId, Features)> = neighbors
            .iter()
            .map(|&(neighbor, _)| {
                let src = node_features.get(neighbor);
                (neighbor, self.linear_transform(src))
            })
            .collect();

        if include_self {
            sources.push((node, target_transformed.clone()));
        }

        if sources.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Compute attention coefficients
        let attention_coefs: Vec<f64> = sources
            .iter()
            .map(|(_, src_transformed)| {
                self.compute_attention_coef(src_transformed, &target_transformed)
            })
            .collect();

        // Softmax normalization
        let attention_weights = softmax(&attention_coefs);

        let transformed_features: Vec<Features> =
            sources.into_iter().map(|(_, feat)| feat).collect();

        (attention_weights, transformed_features)
    }

    /// Aggregate features using attention weights
    pub fn aggregate(&self, weights: &[f64], features: &[Features]) -> Features {
        if features.is_empty() {
            return vec![0.0; self.output_dim];
        }

        let mut result = vec![0.0; self.output_dim];

        for (weight, feat) in weights.iter().zip(features.iter()) {
            for (i, &f) in feat.iter().enumerate() {
                result[i] += weight * f;
            }
        }

        result
    }
}

/// Graph Attention Layer
#[derive(Debug, Clone)]
pub struct GraphAttentionLayer {
    /// Configuration
    config: GATConfig,
    /// Attention heads
    heads: Vec<AttentionHead>,
}

impl GraphAttentionLayer {
    /// Create a new Graph Attention layer
    pub fn new(config: GATConfig) -> Self {
        let heads: Vec<AttentionHead> = (0..config.num_heads)
            .map(|i| AttentionHead::new(config.input_dim, config.output_dim, i, config.use_gatv2))
            .collect();

        Self { config, heads }
    }

    /// Get configuration
    pub fn config(&self) -> &GATConfig {
        &self.config
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Forward pass for a single node
    fn forward_node(&self, graph: &Graph, node_features: &NodeFeatures, node: NodeId) -> Features {
        let neighbors = graph.neighbors(node);

        // Compute output for each head
        let head_outputs: Vec<Features> = self
            .heads
            .iter()
            .map(|head| {
                let (weights, features) = head.compute_node_attention(
                    node_features,
                    node,
                    neighbors,
                    self.config.add_self_loops,
                );

                if weights.is_empty() {
                    vec![0.0; self.config.output_dim]
                } else {
                    head.aggregate(&weights, &features)
                }
            })
            .collect();

        // Combine heads
        if self.config.concat_heads {
            // Concatenate all head outputs
            head_outputs.into_iter().flatten().collect()
        } else {
            // Average all head outputs
            let num_heads = self.heads.len() as f64;
            let dim = self.config.output_dim;
            let mut result = vec![0.0; dim];

            for head_out in &head_outputs {
                for (i, &v) in head_out.iter().enumerate() {
                    result[i] += v / num_heads;
                }
            }

            result
        }
    }

    /// Forward pass for the entire graph
    pub fn forward(&self, graph: &Graph, node_features: &NodeFeatures) -> NodeFeatures {
        let num_nodes = graph.num_nodes();
        let output_dim = self.config.effective_output_dim();
        let mut new_features = NodeFeatures::new(num_nodes, output_dim);

        for node in 0..num_nodes {
            let output = self.forward_node(graph, node_features, node);
            new_features.set(node, output);
        }

        new_features
    }

    /// Forward pass returning attention weights for analysis
    pub fn forward_with_attention(
        &self,
        graph: &Graph,
        node_features: &NodeFeatures,
    ) -> (NodeFeatures, AttentionWeights) {
        let num_nodes = graph.num_nodes();
        let output_dim = self.config.effective_output_dim();
        let mut new_features = NodeFeatures::new(num_nodes, output_dim);
        let mut attention_weights = AttentionWeights::new(num_nodes, self.config.num_heads);

        for node in 0..num_nodes {
            let neighbors = graph.neighbors(node);

            // Compute for each head
            let head_outputs: Vec<(Features, Vec<(NodeId, f64)>)> = self
                .heads
                .iter()
                .map(|head| {
                    let (weights, features) = head.compute_node_attention(
                        node_features,
                        node,
                        neighbors,
                        self.config.add_self_loops,
                    );

                    // Track which nodes got which weights
                    let mut node_weights: Vec<(NodeId, f64)> = neighbors
                        .iter()
                        .zip(weights.iter())
                        .map(|(&(neighbor, _), &w)| (neighbor, w))
                        .collect();

                    if self.config.add_self_loops && !weights.is_empty() {
                        node_weights.push((node, *weights.last().unwrap()));
                    }

                    let aggregated = if weights.is_empty() {
                        vec![0.0; self.config.output_dim]
                    } else {
                        head.aggregate(&weights, &features)
                    };

                    (aggregated, node_weights)
                })
                .collect();

            // Store attention weights
            for (head_idx, (_, node_weights)) in head_outputs.iter().enumerate() {
                for &(neighbor, weight) in node_weights {
                    attention_weights.set(node, neighbor, head_idx, weight);
                }
            }

            // Combine head outputs
            let output = if self.config.concat_heads {
                head_outputs.into_iter().flat_map(|(out, _)| out).collect()
            } else {
                let num_heads = self.heads.len() as f64;
                let dim = self.config.output_dim;
                let mut result = vec![0.0; dim];

                for (head_out, _) in &head_outputs {
                    for (i, &v) in head_out.iter().enumerate() {
                        result[i] += v / num_heads;
                    }
                }

                result
            };

            new_features.set(node, output);
        }

        (new_features, attention_weights)
    }
}

/// Storage for attention weights (for analysis/visualization)
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Weights: (source, target, head) -> weight
    /// Stored as: weights[target][head] = Vec<(source, weight)>
    weights: Vec<Vec<Vec<(NodeId, f64)>>>,
    /// Number of heads
    num_heads: usize,
}

impl AttentionWeights {
    /// Create new attention weight storage
    pub fn new(num_nodes: usize, num_heads: usize) -> Self {
        Self {
            weights: vec![vec![Vec::new(); num_heads]; num_nodes],
            num_heads,
        }
    }

    /// Set attention weight from source to target for a head
    pub fn set(&mut self, target: NodeId, source: NodeId, head: usize, weight: f64) {
        if target < self.weights.len() && head < self.num_heads {
            self.weights[target][head].push((source, weight));
        }
    }

    /// Get attention weights for a target node and head
    pub fn get(&self, target: NodeId, head: usize) -> &[(NodeId, f64)] {
        if target < self.weights.len() && head < self.num_heads {
            &self.weights[target][head]
        } else {
            &[]
        }
    }

    /// Get average attention weight from source to target across all heads
    pub fn average_weight(&self, target: NodeId, source: NodeId) -> f64 {
        if target >= self.weights.len() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for head in 0..self.num_heads {
            for &(src, weight) in &self.weights[target][head] {
                if src == source {
                    sum += weight;
                    count += 1;
                }
            }
        }

        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    /// Get the most attended neighbor for a node (averaged across heads)
    pub fn most_attended(&self, target: NodeId) -> Option<(NodeId, f64)> {
        if target >= self.weights.len() {
            return None;
        }

        let mut total_weights: std::collections::HashMap<NodeId, f64> =
            std::collections::HashMap::new();

        for head in 0..self.num_heads {
            for &(src, weight) in &self.weights[target][head] {
                *total_weights.entry(src).or_insert(0.0) += weight;
            }
        }

        total_weights
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(node, weight)| (node, weight / self.num_heads as f64))
    }
}

/// Multi-layer Graph Attention Network
#[derive(Debug, Clone)]
pub struct GATNetwork {
    /// Layers
    layers: Vec<GraphAttentionLayer>,
}

impl GATNetwork {
    /// Create a new GAT network
    pub fn new(layers: Vec<GraphAttentionLayer>) -> Self {
        Self { layers }
    }

    /// Create a simple 2-layer GAT
    pub fn two_layer(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_heads: usize,
    ) -> Self {
        let layer1_config = GATConfig::new(input_dim, hidden_dim)
            .with_heads(num_heads)
            .with_concat(true);

        let layer2_config = GATConfig::new(hidden_dim * num_heads, output_dim)
            .with_heads(1)
            .with_concat(false);

        Self {
            layers: vec![
                GraphAttentionLayer::new(layer1_config),
                GraphAttentionLayer::new(layer2_config),
            ],
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass through all layers
    pub fn forward(&self, graph: &Graph, initial_features: &NodeFeatures) -> NodeFeatures {
        let mut current = initial_features.clone();

        for layer in &self.layers {
            current = layer.forward(graph, &current);
        }

        current
    }

    /// Forward pass returning intermediate representations
    pub fn forward_all(&self, graph: &Graph, initial_features: &NodeFeatures) -> Vec<NodeFeatures> {
        let mut results = vec![initial_features.clone()];
        let mut current = initial_features.clone();

        for layer in &self.layers {
            current = layer.forward(graph, &current);
            results.push(current.clone());
        }

        results
    }
}

/// Softmax function
fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();

    if sum > 0.0 {
        exp_values.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / values.len() as f64; values.len()]
    }
}

/// ELU activation function
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// Apply ELU activation to features
pub fn apply_elu(features: &mut Features, alpha: f64) {
    for f in features.iter_mut() {
        *f = elu(*f, alpha);
    }
}

/// Statistics about attention patterns
#[derive(Debug, Clone, Default)]
pub struct AttentionStats {
    /// Average attention entropy (higher = more uniform attention)
    pub avg_entropy: f64,
    /// Maximum attention weight observed
    pub max_attention: f64,
    /// Minimum attention weight observed
    pub min_attention: f64,
    /// Average number of attended neighbors
    pub avg_attended: f64,
}

/// Analyze attention patterns
pub fn analyze_attention(attention_weights: &AttentionWeights, num_nodes: usize) -> AttentionStats {
    let mut total_entropy = 0.0;
    let mut max_attention = f64::NEG_INFINITY;
    let mut min_attention = f64::INFINITY;
    let mut total_attended = 0usize;
    let mut count = 0usize;

    for node in 0..num_nodes {
        for head in 0..attention_weights.num_heads {
            let weights = attention_weights.get(node, head);
            if weights.is_empty() {
                continue;
            }

            // Entropy
            let entropy: f64 = weights
                .iter()
                .filter(|(_, w)| *w > 0.0)
                .map(|(_, w)| -w * w.ln())
                .sum();
            total_entropy += entropy;

            // Min/max
            for &(_, w) in weights {
                if w > max_attention {
                    max_attention = w;
                }
                if w < min_attention {
                    min_attention = w;
                }
            }

            total_attended += weights.len();
            count += 1;
        }
    }

    AttentionStats {
        avg_entropy: if count > 0 {
            total_entropy / count as f64
        } else {
            0.0
        },
        max_attention: if max_attention > f64::NEG_INFINITY {
            max_attention
        } else {
            0.0
        },
        min_attention: if min_attention < f64::INFINITY {
            min_attention
        } else {
            0.0
        },
        avg_attended: if count > 0 {
            total_attended as f64 / count as f64
        } else {
            0.0
        },
    }
}

/// Compute attention entropy for a distribution (higher = more uniform)
pub fn attention_entropy(weights: &[f64]) -> f64 {
    weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| -w * w.ln())
        .sum()
}

/// Check if attention is "peaked" (one neighbor dominates)
pub fn is_attention_peaked(weights: &[f64], threshold: f64) -> bool {
    weights.iter().any(|&w| w > threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_passing::Graph;

    fn create_simple_graph() -> Graph {
        let mut graph = Graph::undirected(4);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph
    }

    fn create_star_graph() -> Graph {
        // Node 0 is the center, connected to 1, 2, 3, 4
        let mut graph = Graph::undirected(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(0, 4);
        graph
    }

    #[test]
    fn test_gat_config() {
        let config = GATConfig::new(64, 32).with_heads(4).with_concat(true);

        assert_eq!(config.input_dim, 64);
        assert_eq!(config.output_dim, 32);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.effective_output_dim(), 128); // 32 * 4
    }

    #[test]
    fn test_gat_config_average() {
        let config = GATConfig::new(64, 32).with_heads(4).with_concat(false);

        assert_eq!(config.effective_output_dim(), 32); // Average, not concat
    }

    #[test]
    fn test_attention_head() {
        let head = AttentionHead::new(4, 2, 0, false);

        assert_eq!(head.output_dim, 2);

        // Test linear transform
        let features = vec![1.0, 2.0, 3.0, 4.0];
        let transformed = head.linear_transform(&features);
        assert_eq!(transformed.len(), 2);
    }

    #[test]
    fn test_leaky_relu() {
        let head = AttentionHead::new(4, 2, 0, false);

        assert_eq!(head.leaky_relu(1.0), 1.0);
        assert_eq!(head.leaky_relu(0.0), 0.0);
        assert!((head.leaky_relu(-1.0) - (-0.2)).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = softmax(&values);

        // Should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher values should have higher probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let values: Vec<f64> = Vec::new();
        let result = softmax(&values);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gat_layer_creation() {
        let config = GATConfig::new(4, 2).with_heads(3);
        let layer = GraphAttentionLayer::new(config);

        assert_eq!(layer.num_heads(), 3);
    }

    #[test]
    fn test_gat_layer_forward() {
        let graph = create_simple_graph();
        let mut node_features = NodeFeatures::new(4, 4);
        node_features.set(0, vec![1.0, 0.0, 0.0, 0.0]);
        node_features.set(1, vec![0.0, 1.0, 0.0, 0.0]);
        node_features.set(2, vec![0.0, 0.0, 1.0, 0.0]);
        node_features.set(3, vec![0.0, 0.0, 0.0, 1.0]);

        let config = GATConfig::new(4, 2).with_heads(2).with_concat(true);
        let layer = GraphAttentionLayer::new(config);

        let output = layer.forward(&graph, &node_features);

        assert_eq!(output.num_nodes(), 4);
        assert_eq!(output.get(0).len(), 4); // 2 * 2 heads concatenated
    }

    #[test]
    fn test_gat_layer_average_heads() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 4);

        let config = GATConfig::new(4, 2).with_heads(2).with_concat(false);
        let layer = GraphAttentionLayer::new(config);

        let output = layer.forward(&graph, &node_features);

        assert_eq!(output.get(0).len(), 2); // Averaged, not concatenated
    }

    #[test]
    fn test_attention_weights_storage() {
        let mut weights = AttentionWeights::new(4, 2);

        weights.set(0, 1, 0, 0.5);
        weights.set(0, 2, 0, 0.5);
        weights.set(0, 1, 1, 0.3);
        weights.set(0, 2, 1, 0.7);

        let head0 = weights.get(0, 0);
        assert_eq!(head0.len(), 2);

        let avg = weights.average_weight(0, 1);
        assert!((avg - 0.4).abs() < 1e-6); // (0.5 + 0.3) / 2
    }

    #[test]
    fn test_most_attended() {
        let mut weights = AttentionWeights::new(4, 2);

        weights.set(0, 1, 0, 0.2);
        weights.set(0, 2, 0, 0.8);
        weights.set(0, 1, 1, 0.3);
        weights.set(0, 2, 1, 0.7);

        let (node, _) = weights.most_attended(0).unwrap();
        assert_eq!(node, 2); // Node 2 has higher average weight
    }

    #[test]
    fn test_forward_with_attention() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 4);

        let config = GATConfig::new(4, 2).with_heads(2);
        let layer = GraphAttentionLayer::new(config);

        let (output, attention) = layer.forward_with_attention(&graph, &node_features);

        assert_eq!(output.num_nodes(), 4);

        // Check attention weights are stored
        let weights = attention.get(0, 0);
        assert!(!weights.is_empty());

        // Attention weights should sum to ~1
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gat_network() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 8);

        let network = GATNetwork::two_layer(8, 4, 2, 2);

        assert_eq!(network.num_layers(), 2);

        let output = network.forward(&graph, &node_features);
        assert_eq!(output.num_nodes(), 4);
        assert_eq!(output.get(0).len(), 2); // Final output dim
    }

    #[test]
    fn test_gat_network_forward_all() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 8);

        let network = GATNetwork::two_layer(8, 4, 2, 2);
        let all_outputs = network.forward_all(&graph, &node_features);

        assert_eq!(all_outputs.len(), 3); // Initial + 2 layers
    }

    #[test]
    fn test_gatv2() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 4);

        let config = GATConfig::new(4, 2).with_heads(2).with_gatv2();
        let layer = GraphAttentionLayer::new(config);

        let output = layer.forward(&graph, &node_features);
        assert_eq!(output.num_nodes(), 4);
    }

    #[test]
    fn test_elu() {
        assert_eq!(elu(1.0, 1.0), 1.0);
        assert_eq!(elu(0.0, 1.0), 0.0);
        assert!(elu(-1.0, 1.0) < 0.0);
        assert!(elu(-1.0, 1.0) > -1.0);
    }

    #[test]
    fn test_apply_elu() {
        let mut features = vec![1.0, 0.0, -1.0];
        apply_elu(&mut features, 1.0);

        assert_eq!(features[0], 1.0);
        assert_eq!(features[1], 0.0);
        assert!(features[2] < 0.0);
    }

    #[test]
    fn test_attention_entropy() {
        // Uniform distribution has higher entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let peaked = vec![0.9, 0.05, 0.03, 0.02];

        let entropy_uniform = attention_entropy(&uniform);
        let entropy_peaked = attention_entropy(&peaked);

        assert!(entropy_uniform > entropy_peaked);
    }

    #[test]
    fn test_is_attention_peaked() {
        let peaked = vec![0.9, 0.05, 0.05];
        let uniform = vec![0.33, 0.33, 0.34];

        assert!(is_attention_peaked(&peaked, 0.8));
        assert!(!is_attention_peaked(&uniform, 0.8));
    }

    #[test]
    fn test_star_graph_attention() {
        let graph = create_star_graph();
        let mut node_features = NodeFeatures::new(5, 4);

        // Make center node different from others
        node_features.set(0, vec![1.0, 1.0, 1.0, 1.0]);
        for i in 1..5 {
            node_features.set(i, vec![0.1, 0.1, 0.1, 0.1]);
        }

        let config = GATConfig::new(4, 2).with_heads(1);
        let layer = GraphAttentionLayer::new(config);

        let (_, attention) = layer.forward_with_attention(&graph, &node_features);

        // Center node (0) attends to all 4 neighbors + self
        let center_attention = attention.get(0, 0);
        assert!(!center_attention.is_empty());
    }

    #[test]
    fn test_analyze_attention() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(4, 4);

        let config = GATConfig::new(4, 2).with_heads(2);
        let layer = GraphAttentionLayer::new(config);

        let (_, attention) = layer.forward_with_attention(&graph, &node_features);
        let stats = analyze_attention(&attention, 4);

        assert!(stats.avg_entropy >= 0.0);
        assert!(stats.max_attention <= 1.0);
        assert!(stats.min_attention >= 0.0);
        assert!(stats.avg_attended > 0.0);
    }

    #[test]
    fn test_dropout_config() {
        let config = GATConfig::new(4, 2).with_dropout(0.5);
        assert!((config.dropout - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_no_neighbors() {
        // Node 0 has no neighbors in a disconnected graph
        let graph = Graph::undirected(2); // No edges
        let node_features = NodeFeatures::new(2, 4);

        let config = GATConfig::new(4, 2).with_heads(1);
        let layer = GraphAttentionLayer::new(config);

        let output = layer.forward(&graph, &node_features);

        // Should still produce output (from self-loop)
        assert_eq!(output.num_nodes(), 2);
    }
}
