// Vector Search Algorithms
// Approximate Nearest Neighbor (ANN) search for high-dimensional vectors
// Critical for RAG systems, vector databases, and LLM applications

pub mod hnsw;
pub mod lsh;
pub mod similarity;

pub use hnsw::{HnswConfig, HnswIndex, SearchResult, build_hnsw_index};
pub use lsh::{LshConfig, LshIndex, LshStats, build_lsh_index};
pub use similarity::{
    DistanceMetric, cosine_distance, cosine_similarity, dot_product, dot_product_distance,
    euclidean_distance, euclidean_distance_squared, manhattan_distance, normalize,
};
