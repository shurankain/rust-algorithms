// Graph Neural Network Primitives
// Foundational algorithms for GNN implementations

pub mod graph_attention;
pub mod graphsage_sampling;
pub mod laplacian;
pub mod message_passing;
pub mod pagerank;
pub mod random_walk;

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

pub use random_walk::{
    RandomWalkConfig, RandomWalkWithRestart, RandomWalker, SimpleRng, Walk, WalkStats,
    WalkStrategy, analyze_walks, compute_cooccurrence, compute_transition_matrix,
    compute_visit_frequency, generate_skipgram_pairs, negative_sample,
};

pub use pagerank::{
    ApproximatePPR, PageRank, PageRankConfig, PageRankResult, PageRankStats, PersonalizedPageRank,
    TopicSensitivePageRank, analyze_pagerank, pagerank_similarity, structural_similarity,
};

pub use laplacian::{
    ChebyshevFilter, DenseMatrix, GraphLaplacian, LaplacianConfig, LaplacianStats, LaplacianType,
    SparseEntry, SparseMatrix, analyze_laplacian, apply_laplacian, compute_laplacian_eigenvalues,
    dirichlet_energy, estimate_algebraic_connectivity, feature_smoothness, heat_diffusion,
};
