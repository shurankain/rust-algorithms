// Graph Neural Network Primitives
// Foundational algorithms for GNN implementations

pub mod message_passing;

pub use message_passing::{
    Aggregation, EdgeFeatures, EdgeId, Features, Graph, MessageFunction, MessagePassingConfig,
    MessagePassingLayer, MessagePassingNetwork, MessagePassingStats, NodeFeatures, NodeId,
    UpdateFunction, analyze_message_passing, compute_attention_weights, global_pool,
    k_hop_neighbors, receptive_field_size,
};
