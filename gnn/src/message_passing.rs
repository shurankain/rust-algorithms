// Message Passing Framework for Graph Neural Networks
//
// Message Passing Neural Networks (MPNNs) are the foundation of most GNN architectures.
// The framework follows three phases:
//
// 1. Message: Compute messages from neighboring nodes
//    m_{v->u} = MESSAGE(h_v, h_u, e_{v,u})
//
// 2. Aggregate: Combine messages from all neighbors
//    m_u = AGGREGATE({m_{v->u} : v in N(u)})
//
// 3. Update: Update node representation using aggregated message
//    h_u' = UPDATE(h_u, m_u)
//
// This abstraction encompasses:
// - GCN (Graph Convolutional Networks)
// - GraphSAGE
// - GAT (Graph Attention Networks)
// - GIN (Graph Isomorphism Networks)
// - EdgeConv, PointNet++, etc.
//
// References:
// - "Neural Message Passing for Quantum Chemistry" (Gilmer et al., 2017)
// - "A Comprehensive Survey on Graph Neural Networks" (Wu et al., 2020)

use std::collections::HashMap;

/// Node identifier type
pub type NodeId = usize;

/// Edge identifier type
pub type EdgeId = usize;

/// A simple graph structure for GNN operations
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of nodes
    num_nodes: usize,
    /// Adjacency list: node -> list of (neighbor, edge_id)
    adjacency: Vec<Vec<(NodeId, EdgeId)>>,
    /// Edge list: (source, target)
    edges: Vec<(NodeId, NodeId)>,
    /// Whether the graph is directed
    directed: bool,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(num_nodes: usize, directed: bool) -> Self {
        Self {
            num_nodes,
            adjacency: vec![Vec::new(); num_nodes],
            edges: Vec::new(),
            directed,
        }
    }

    /// Create an undirected graph
    pub fn undirected(num_nodes: usize) -> Self {
        Self::new(num_nodes, false)
    }

    /// Create a directed graph
    pub fn directed(num_nodes: usize) -> Self {
        Self::new(num_nodes, true)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> EdgeId {
        let edge_id = self.edges.len();
        self.edges.push((source, target));
        self.adjacency[source].push((target, edge_id));

        if !self.directed && source != target {
            self.adjacency[target].push((source, edge_id));
        }

        edge_id
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get neighbors of a node with their edge IDs
    pub fn neighbors(&self, node: NodeId) -> &[(NodeId, EdgeId)] {
        &self.adjacency[node]
    }

    /// Get the degree of a node
    pub fn degree(&self, node: NodeId) -> usize {
        self.adjacency[node].len()
    }

    /// Get edge endpoints
    pub fn edge(&self, edge_id: EdgeId) -> (NodeId, NodeId) {
        self.edges[edge_id]
    }

    /// Check if graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> {
        0..self.num_nodes
    }

    /// Get all edges as (source, target, edge_id)
    pub fn edges_iter(&self) -> impl Iterator<Item = (NodeId, NodeId, EdgeId)> + '_ {
        self.edges
            .iter()
            .enumerate()
            .map(|(id, &(s, t))| (s, t, id))
    }
}

/// Feature vector type (simple f64 vector)
pub type Features = Vec<f64>;

/// Node features for a graph
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    /// Features for each node: node_id -> feature vector
    features: Vec<Features>,
    /// Feature dimension
    dim: usize,
}

impl NodeFeatures {
    /// Create node features with given dimension
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        Self {
            features: vec![vec![0.0; dim]; num_nodes],
            dim,
        }
    }

    /// Create from existing features
    pub fn from_vec(features: Vec<Features>) -> Self {
        let dim = features.first().map(|f| f.len()).unwrap_or(0);
        Self { features, dim }
    }

    /// Get features for a node
    pub fn get(&self, node: NodeId) -> &Features {
        &self.features[node]
    }

    /// Get mutable features for a node
    pub fn get_mut(&mut self, node: NodeId) -> &mut Features {
        &mut self.features[node]
    }

    /// Set features for a node
    pub fn set(&mut self, node: NodeId, features: Features) {
        self.features[node] = features;
    }

    /// Get feature dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.features.len()
    }

    /// Iterate over all features
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Features)> {
        self.features.iter().enumerate()
    }
}

/// Edge features for a graph
#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    /// Features for each edge
    features: Vec<Features>,
    /// Feature dimension
    dim: usize,
}

impl EdgeFeatures {
    /// Create edge features with given dimension
    pub fn new(num_edges: usize, dim: usize) -> Self {
        Self {
            features: vec![vec![0.0; dim]; num_edges],
            dim,
        }
    }

    /// Create from existing features
    pub fn from_vec(features: Vec<Features>) -> Self {
        let dim = features.first().map(|f| f.len()).unwrap_or(0);
        Self { features, dim }
    }

    /// Get features for an edge
    pub fn get(&self, edge: EdgeId) -> &Features {
        &self.features[edge]
    }

    /// Set features for an edge
    pub fn set(&mut self, edge: EdgeId, features: Features) {
        self.features[edge] = features;
    }

    /// Get feature dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Aggregation methods for combining neighbor messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Sum of all messages
    Sum,
    /// Mean of all messages
    Mean,
    /// Maximum across each dimension
    Max,
    /// Minimum across each dimension
    Min,
}

impl Aggregation {
    /// Aggregate a list of messages
    pub fn aggregate(&self, messages: &[Features]) -> Features {
        if messages.is_empty() {
            return Vec::new();
        }

        let dim = messages[0].len();
        let mut result = vec![0.0; dim];

        match self {
            Aggregation::Sum => {
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        result[i] += v;
                    }
                }
            }
            Aggregation::Mean => {
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        result[i] += v;
                    }
                }
                let n = messages.len() as f64;
                for v in &mut result {
                    *v /= n;
                }
            }
            Aggregation::Max => {
                result = vec![f64::NEG_INFINITY; dim];
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        if v > result[i] {
                            result[i] = v;
                        }
                    }
                }
            }
            Aggregation::Min => {
                result = vec![f64::INFINITY; dim];
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        if v < result[i] {
                            result[i] = v;
                        }
                    }
                }
            }
        }

        result
    }
}

/// Message function types for computing messages between nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageFunction {
    /// Just use source node features: m = h_source
    Identity,
    /// Use source features scaled by edge weight
    Scaled,
    /// Concatenate source and target features
    Concat,
    /// Add source and target features
    Add,
    /// Multiply elementwise (Hadamard product)
    Hadamard,
}

/// Update function types for updating node features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateFunction {
    /// Replace with aggregated message: h' = m
    Replace,
    /// Add message to features: h' = h + m
    Residual,
    /// Concatenate: h' = [h || m]
    Concat,
    /// Gated update: h' = gate * h + (1-gate) * m
    GRU,
}

/// Configuration for a message passing layer
#[derive(Debug, Clone)]
pub struct MessagePassingConfig {
    /// Aggregation method
    pub aggregation: Aggregation,
    /// Message function
    pub message_fn: MessageFunction,
    /// Update function
    pub update_fn: UpdateFunction,
    /// Whether to add self-loops
    pub add_self_loops: bool,
    /// Whether to normalize by degree
    pub normalize: bool,
}

impl Default for MessagePassingConfig {
    fn default() -> Self {
        Self {
            aggregation: Aggregation::Sum,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Replace,
            add_self_loops: true,
            normalize: false,
        }
    }
}

impl MessagePassingConfig {
    /// Create GCN-style config (mean aggregation with normalization)
    pub fn gcn() -> Self {
        Self {
            aggregation: Aggregation::Mean,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Replace,
            add_self_loops: true,
            normalize: true,
        }
    }

    /// Create GraphSAGE-style config (mean aggregation, concat update)
    pub fn graphsage() -> Self {
        Self {
            aggregation: Aggregation::Mean,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Concat,
            add_self_loops: false,
            normalize: false,
        }
    }

    /// Create GIN-style config (sum aggregation)
    pub fn gin() -> Self {
        Self {
            aggregation: Aggregation::Sum,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Residual,
            add_self_loops: true,
            normalize: false,
        }
    }

    /// Create max-pool config
    pub fn max_pool() -> Self {
        Self {
            aggregation: Aggregation::Max,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Replace,
            add_self_loops: false,
            normalize: false,
        }
    }
}

/// A message passing layer for GNNs
#[derive(Debug, Clone)]
pub struct MessagePassingLayer {
    /// Configuration
    config: MessagePassingConfig,
    /// Optional linear transformation weights (input_dim x output_dim)
    weights: Option<Vec<Vec<f64>>>,
    /// Optional bias
    bias: Option<Vec<f64>>,
}

impl MessagePassingLayer {
    /// Create a new message passing layer
    pub fn new(config: MessagePassingConfig) -> Self {
        Self {
            config,
            weights: None,
            bias: None,
        }
    }

    /// Create with linear transformation
    pub fn with_linear(config: MessagePassingConfig, input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with small random values (simplified Xavier init)
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|i| {
                (0..output_dim)
                    .map(|j| {
                        // Simple deterministic "random" for reproducibility
                        let seed = (i * 31 + j * 17) as f64;
                        seed.sin() * scale
                    })
                    .collect()
            })
            .collect();

        let bias = vec![0.0; output_dim];

        Self {
            config,
            weights: Some(weights),
            bias: Some(bias),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &MessagePassingConfig {
        &self.config
    }

    /// Compute message from source to target node
    fn compute_message(
        &self,
        source_features: &Features,
        target_features: &Features,
        _edge_features: Option<&Features>,
    ) -> Features {
        match self.config.message_fn {
            MessageFunction::Identity => source_features.clone(),
            MessageFunction::Scaled => {
                // Scale by 1.0 if no edge features (simplified)
                source_features.clone()
            }
            MessageFunction::Concat => {
                let mut result = source_features.clone();
                result.extend(target_features.iter().cloned());
                result
            }
            MessageFunction::Add => source_features
                .iter()
                .zip(target_features.iter())
                .map(|(&s, &t)| s + t)
                .collect(),
            MessageFunction::Hadamard => source_features
                .iter()
                .zip(target_features.iter())
                .map(|(&s, &t)| s * t)
                .collect(),
        }
    }

    /// Update node features with aggregated message
    fn update_features(&self, current: &Features, message: &Features) -> Features {
        match self.config.update_fn {
            UpdateFunction::Replace => message.clone(),
            UpdateFunction::Residual => current
                .iter()
                .zip(message.iter())
                .map(|(&c, &m)| c + m)
                .collect(),
            UpdateFunction::Concat => {
                let mut result = current.clone();
                result.extend(message.iter().cloned());
                result
            }
            UpdateFunction::GRU => {
                // Simplified GRU: 0.5 * current + 0.5 * message
                current
                    .iter()
                    .zip(message.iter())
                    .map(|(&c, &m)| 0.5 * c + 0.5 * m)
                    .collect()
            }
        }
    }

    /// Apply linear transformation to features
    fn linear_transform(&self, features: &Features) -> Features {
        match (&self.weights, &self.bias) {
            (Some(weights), Some(bias)) => {
                let mut result = bias.clone();

                for (i, &f) in features.iter().enumerate() {
                    if i < weights.len() {
                        for (j, &w) in weights[i].iter().enumerate() {
                            result[j] += f * w;
                        }
                    }
                }

                result
            }
            _ => features.clone(),
        }
    }

    /// Forward pass: propagate messages through the graph
    pub fn forward(
        &self,
        graph: &Graph,
        node_features: &NodeFeatures,
        edge_features: Option<&EdgeFeatures>,
    ) -> NodeFeatures {
        let num_nodes = graph.num_nodes();
        let mut new_features = NodeFeatures::new(num_nodes, node_features.dim());

        // For each node, collect messages from neighbors and update
        for node in 0..num_nodes {
            let mut messages: Vec<Features> = Vec::new();
            let target_features = node_features.get(node);

            // Collect messages from neighbors
            for &(neighbor, edge_id) in graph.neighbors(node) {
                let source_features = node_features.get(neighbor);
                let edge_feat = edge_features.map(|ef| ef.get(edge_id));

                let message = self.compute_message(source_features, target_features, edge_feat);
                messages.push(message);
            }

            // Add self-loop if configured
            if self.config.add_self_loops {
                messages.push(target_features.clone());
            }

            // Aggregate messages
            let aggregated = if messages.is_empty() {
                vec![0.0; node_features.dim()]
            } else {
                let mut agg = self.config.aggregation.aggregate(&messages);

                // Normalize by degree if configured
                if self.config.normalize && !messages.is_empty() {
                    let degree = messages.len() as f64;
                    let norm = 1.0 / degree.sqrt();
                    for v in &mut agg {
                        *v *= norm;
                    }
                }

                agg
            };

            // Update features
            let updated = self.update_features(target_features, &aggregated);

            // Apply linear transformation if present
            let final_features = self.linear_transform(&updated);

            new_features.set(node, final_features);
        }

        new_features
    }
}

/// Statistics about message passing
#[derive(Debug, Clone, Default)]
pub struct MessagePassingStats {
    /// Total messages computed
    pub messages_computed: usize,
    /// Total aggregations performed
    pub aggregations: usize,
    /// Average messages per node
    pub avg_messages_per_node: f64,
}

/// Analyze message passing on a graph
pub fn analyze_message_passing(graph: &Graph) -> MessagePassingStats {
    let total_messages: usize = graph.nodes().map(|n| graph.degree(n)).sum();
    let num_nodes = graph.num_nodes();

    MessagePassingStats {
        messages_computed: total_messages,
        aggregations: num_nodes,
        avg_messages_per_node: if num_nodes > 0 {
            total_messages as f64 / num_nodes as f64
        } else {
            0.0
        },
    }
}

/// Multi-layer message passing network
#[derive(Debug, Clone)]
pub struct MessagePassingNetwork {
    /// Layers in the network
    layers: Vec<MessagePassingLayer>,
}

impl MessagePassingNetwork {
    /// Create a new network with given layers
    pub fn new(layers: Vec<MessagePassingLayer>) -> Self {
        Self { layers }
    }

    /// Create a network with uniform configuration
    pub fn uniform(config: MessagePassingConfig, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| MessagePassingLayer::new(config.clone()))
            .collect();
        Self { layers }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass through all layers
    pub fn forward(
        &self,
        graph: &Graph,
        initial_features: &NodeFeatures,
        edge_features: Option<&EdgeFeatures>,
    ) -> NodeFeatures {
        let mut current = initial_features.clone();

        for layer in &self.layers {
            current = layer.forward(graph, &current, edge_features);
        }

        current
    }

    /// Forward pass returning all intermediate representations
    pub fn forward_all(
        &self,
        graph: &Graph,
        initial_features: &NodeFeatures,
        edge_features: Option<&EdgeFeatures>,
    ) -> Vec<NodeFeatures> {
        let mut results = vec![initial_features.clone()];
        let mut current = initial_features.clone();

        for layer in &self.layers {
            current = layer.forward(graph, &current, edge_features);
            results.push(current.clone());
        }

        results
    }
}

/// Compute graph-level representation by pooling node features
pub fn global_pool(node_features: &NodeFeatures, method: Aggregation) -> Features {
    let all_features: Vec<Features> = node_features.iter().map(|(_, f)| f.clone()).collect();
    method.aggregate(&all_features)
}

/// Compute attention weights between nodes (simplified)
pub fn compute_attention_weights(query: &Features, keys: &[Features], _dim: usize) -> Vec<f64> {
    if keys.is_empty() {
        return Vec::new();
    }

    // Compute dot product attention scores
    let scores: Vec<f64> = keys
        .iter()
        .map(|key| {
            query
                .iter()
                .zip(key.iter())
                .map(|(&q, &k)| q * k)
                .sum::<f64>()
        })
        .collect();

    // Softmax
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();

    exp_scores.iter().map(|&e| e / sum_exp).collect()
}

/// K-hop neighborhood aggregation
pub fn k_hop_neighbors(graph: &Graph, node: NodeId, k: usize) -> HashMap<NodeId, usize> {
    let mut distances: HashMap<NodeId, usize> = HashMap::new();
    distances.insert(node, 0);

    let mut frontier = vec![node];

    for hop in 1..=k {
        let mut next_frontier = Vec::new();

        for &current in &frontier {
            for &(neighbor, _) in graph.neighbors(current) {
                if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                    e.insert(hop);
                    next_frontier.push(neighbor);
                }
            }
        }

        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }

    distances
}

/// Compute receptive field size for k layers of message passing
pub fn receptive_field_size(graph: &Graph, node: NodeId, k: usize) -> usize {
    k_hop_neighbors(graph, node, k).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_graph() -> Graph {
        // Create a simple triangle graph: 0 -- 1 -- 2 -- 0
        let mut graph = Graph::undirected(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        graph
    }

    fn create_line_graph() -> Graph {
        // Create a line: 0 -- 1 -- 2 -- 3
        let mut graph = Graph::undirected(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_simple_graph();
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
        assert!(!graph.is_directed());
    }

    #[test]
    fn test_graph_neighbors() {
        let graph = create_simple_graph();

        // Each node in triangle has 2 neighbors
        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(2), 2);
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::directed(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert!(graph.is_directed());
        assert_eq!(graph.degree(0), 1); // Only outgoing
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 0); // No outgoing edges
    }

    #[test]
    fn test_node_features() {
        let mut features = NodeFeatures::new(3, 4);

        features.set(0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(features.get(0), &vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(features.dim(), 4);
        assert_eq!(features.num_nodes(), 3);
    }

    #[test]
    fn test_aggregation_sum() {
        let messages = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = Aggregation::Sum.aggregate(&messages);
        assert_eq!(result, vec![9.0, 12.0]);
    }

    #[test]
    fn test_aggregation_mean() {
        let messages = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = Aggregation::Mean.aggregate(&messages);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_aggregation_max() {
        let messages = vec![vec![1.0, 6.0], vec![3.0, 2.0], vec![5.0, 4.0]];

        let result = Aggregation::Max.aggregate(&messages);
        assert_eq!(result, vec![5.0, 6.0]);
    }

    #[test]
    fn test_aggregation_min() {
        let messages = vec![vec![1.0, 6.0], vec![3.0, 2.0], vec![5.0, 4.0]];

        let result = Aggregation::Min.aggregate(&messages);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_aggregation_empty() {
        let messages: Vec<Features> = Vec::new();
        let result = Aggregation::Sum.aggregate(&messages);
        assert!(result.is_empty());
    }

    #[test]
    fn test_message_passing_layer() {
        let graph = create_simple_graph();
        let mut node_features = NodeFeatures::new(3, 2);
        node_features.set(0, vec![1.0, 0.0]);
        node_features.set(1, vec![0.0, 1.0]);
        node_features.set(2, vec![1.0, 1.0]);

        let config = MessagePassingConfig::default();
        let layer = MessagePassingLayer::new(config);

        let output = layer.forward(&graph, &node_features, None);

        // Check output dimensions
        assert_eq!(output.num_nodes(), 3);
        assert_eq!(output.dim(), 2);
    }

    #[test]
    fn test_gcn_config() {
        let config = MessagePassingConfig::gcn();
        assert_eq!(config.aggregation, Aggregation::Mean);
        assert!(config.add_self_loops);
        assert!(config.normalize);
    }

    #[test]
    fn test_graphsage_config() {
        let config = MessagePassingConfig::graphsage();
        assert_eq!(config.aggregation, Aggregation::Mean);
        assert_eq!(config.update_fn, UpdateFunction::Concat);
        assert!(!config.add_self_loops);
    }

    #[test]
    fn test_message_passing_network() {
        let graph = create_simple_graph();
        let mut node_features = NodeFeatures::new(3, 2);
        node_features.set(0, vec![1.0, 0.0]);
        node_features.set(1, vec![0.0, 1.0]);
        node_features.set(2, vec![1.0, 1.0]);

        let config = MessagePassingConfig::default();
        let network = MessagePassingNetwork::uniform(config, 2);

        assert_eq!(network.num_layers(), 2);

        let output = network.forward(&graph, &node_features, None);
        assert_eq!(output.num_nodes(), 3);
    }

    #[test]
    fn test_forward_all() {
        let graph = create_simple_graph();
        let node_features = NodeFeatures::new(3, 2);

        let config = MessagePassingConfig::default();
        let network = MessagePassingNetwork::uniform(config, 3);

        let all_outputs = network.forward_all(&graph, &node_features, None);

        // Should have initial + 3 layer outputs
        assert_eq!(all_outputs.len(), 4);
    }

    #[test]
    fn test_global_pool() {
        let mut node_features = NodeFeatures::new(3, 2);
        node_features.set(0, vec![1.0, 2.0]);
        node_features.set(1, vec![3.0, 4.0]);
        node_features.set(2, vec![5.0, 6.0]);

        let pooled = global_pool(&node_features, Aggregation::Mean);
        assert_eq!(pooled, vec![3.0, 4.0]);
    }

    #[test]
    fn test_attention_weights() {
        let query = vec![1.0, 0.0, 1.0];
        let keys = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];

        let weights = compute_attention_weights(&query, &keys, 3);

        // Should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // First and third should have higher weights (similar to query)
        assert!(weights[0] > weights[1]);
        assert!(weights[2] > weights[1]);
    }

    #[test]
    fn test_k_hop_neighbors() {
        let graph = create_line_graph();

        // 1-hop from node 0
        let one_hop = k_hop_neighbors(&graph, 0, 1);
        assert_eq!(one_hop.len(), 2); // 0 and 1
        assert_eq!(one_hop.get(&0), Some(&0));
        assert_eq!(one_hop.get(&1), Some(&1));

        // 2-hop from node 0
        let two_hop = k_hop_neighbors(&graph, 0, 2);
        assert_eq!(two_hop.len(), 3); // 0, 1, 2
        assert_eq!(two_hop.get(&2), Some(&2));
    }

    #[test]
    fn test_receptive_field() {
        let graph = create_line_graph();

        assert_eq!(receptive_field_size(&graph, 0, 1), 2);
        assert_eq!(receptive_field_size(&graph, 0, 2), 3);
        assert_eq!(receptive_field_size(&graph, 0, 3), 4);
        assert_eq!(receptive_field_size(&graph, 0, 4), 4); // Can't grow beyond graph
    }

    #[test]
    fn test_analyze_message_passing() {
        let graph = create_simple_graph();
        let stats = analyze_message_passing(&graph);

        // Triangle: each node has 2 neighbors
        assert_eq!(stats.messages_computed, 6);
        assert_eq!(stats.aggregations, 3);
        assert!((stats.avg_messages_per_node - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_residual_update() {
        let graph = create_simple_graph();
        let mut node_features = NodeFeatures::new(3, 2);
        node_features.set(0, vec![1.0, 1.0]);
        node_features.set(1, vec![1.0, 1.0]);
        node_features.set(2, vec![1.0, 1.0]);

        let config = MessagePassingConfig {
            aggregation: Aggregation::Sum,
            message_fn: MessageFunction::Identity,
            update_fn: UpdateFunction::Residual,
            add_self_loops: false,
            normalize: false,
        };
        let layer = MessagePassingLayer::new(config);

        let output = layer.forward(&graph, &node_features, None);

        // Each node receives sum of 2 neighbors (2.0, 2.0) + original (1.0, 1.0)
        assert_eq!(output.get(0), &vec![3.0, 3.0]);
    }

    #[test]
    fn test_layer_with_linear() {
        let graph = create_simple_graph();
        let mut node_features = NodeFeatures::new(3, 4);
        node_features.set(0, vec![1.0, 2.0, 3.0, 4.0]);
        node_features.set(1, vec![1.0, 2.0, 3.0, 4.0]);
        node_features.set(2, vec![1.0, 2.0, 3.0, 4.0]);

        let config = MessagePassingConfig::default();
        let layer = MessagePassingLayer::with_linear(config, 4, 2);

        let output = layer.forward(&graph, &node_features, None);

        // Linear transform is applied after update, output dim matches linear output
        // But NodeFeatures stores features that went through linear_transform
        // which produces 2D output from 4D input
        assert_eq!(output.get(0).len(), 2);
    }

    #[test]
    fn test_edge_features() {
        let mut edge_features = EdgeFeatures::new(3, 2);
        edge_features.set(0, vec![1.0, 2.0]);

        assert_eq!(edge_features.get(0), &vec![1.0, 2.0]);
        assert_eq!(edge_features.dim(), 2);
    }

    #[test]
    fn test_self_loops() {
        let graph = create_line_graph();
        let mut node_features = NodeFeatures::new(4, 1);
        for i in 0..4 {
            node_features.set(i, vec![1.0]);
        }

        // Without self-loops
        let config_no_self = MessagePassingConfig {
            add_self_loops: false,
            ..MessagePassingConfig::default()
        };
        let layer_no_self = MessagePassingLayer::new(config_no_self);
        let out_no_self = layer_no_self.forward(&graph, &node_features, None);

        // With self-loops
        let config_with_self = MessagePassingConfig {
            add_self_loops: true,
            ..MessagePassingConfig::default()
        };
        let layer_with_self = MessagePassingLayer::new(config_with_self);
        let out_with_self = layer_with_self.forward(&graph, &node_features, None);

        // Node 0 has 1 neighbor, so without self-loop sum=1, with self-loop sum=2
        assert_eq!(out_no_self.get(0), &vec![1.0]);
        assert_eq!(out_with_self.get(0), &vec![2.0]);
    }

    #[test]
    fn test_concat_message() {
        let source = vec![1.0, 2.0];
        let target = vec![3.0, 4.0];

        let _graph = Graph::undirected(2);
        let mut features = NodeFeatures::new(2, 2);
        features.set(0, source);
        features.set(1, target);

        let config = MessagePassingConfig {
            message_fn: MessageFunction::Concat,
            update_fn: UpdateFunction::Replace,
            add_self_loops: false,
            ..MessagePassingConfig::default()
        };

        let layer = MessagePassingLayer::new(config);

        // The message function concatenates, but we need an edge to test
        // For now just verify config is set
        assert_eq!(layer.config().message_fn, MessageFunction::Concat);
    }

    #[test]
    fn test_gin_config() {
        let config = MessagePassingConfig::gin();
        assert_eq!(config.aggregation, Aggregation::Sum);
        assert_eq!(config.update_fn, UpdateFunction::Residual);
        assert!(config.add_self_loops);
    }

    #[test]
    fn test_max_pool_config() {
        let config = MessagePassingConfig::max_pool();
        assert_eq!(config.aggregation, Aggregation::Max);
        assert!(!config.add_self_loops);
    }
}
