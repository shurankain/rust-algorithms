// Graph Neural Network Primitives
// Foundational algorithms for GNN implementations

pub mod graph_attention;
pub mod graphsage_sampling;
pub mod message_passing;

pub use message_passing::{
    Aggregation, EdgeFeatures, EdgeId, Features, Graph, MessageFunction, MessagePassingConfig,
    MessagePassingLayer, MessagePassingNetwork, MessagePassingStats, NodeFeatures, NodeId,
    UpdateFunction, analyze_message_passing, compute_attention_weights, global_pool,
    k_hop_neighbors, receptive_field_size,
};

pub use graph_attention::{
    AttentionHead, AttentionStats, AttentionWeights, GATConfig, GATNetwork, GraphAttentionLayer,
    analyze_attention, apply_elu, attention_entropy, elu, is_attention_peaked,
};

pub use graphsage_sampling::{
    GraphSAGE, GraphSAGELayer, NeighborSampler, SAGEAggregator, SampledSubgraph, SamplingConfig,
    SamplingStats, SamplingStrategy, analyze_sampling, create_batches,
};
