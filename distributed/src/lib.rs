// Distributed Systems Algorithms
// Essential algorithms for building scalable distributed systems

pub mod consistent_hashing;
pub mod jump_consistent_hash;
pub mod merkle_tree;
pub mod raft;
pub mod rendezvous_hashing;
pub mod vector_clock;

pub use consistent_hashing::{
    ConsistentHash, ConsistentHashConfig, ConsistentHashStats, Node, SimpleConsistentHash,
    calculate_load_imbalance, calculate_load_std_dev, compare_with_modulo_hashing,
};

pub use rendezvous_hashing::{
    Node as RendezvousNode, RendezvousConfig, RendezvousHash, RendezvousStats,
    SimpleRendezvousHash, calculate_load_imbalance as rendezvous_load_imbalance,
    calculate_load_std_dev as rendezvous_load_std_dev,
    compare_with_modulo_hashing as rendezvous_compare_with_modulo, expected_keys_moved_on_add,
    expected_keys_moved_on_remove,
};

pub use jump_consistent_hash::{
    JumpConsistentHash, JumpHashRouter, JumpHashStats, analyze_jump_pattern,
    calculate_load_imbalance as jump_load_imbalance, calculate_load_std_dev as jump_load_std_dev,
    compare_with_modulo_hashing as jump_compare_with_modulo, expected_keys_moved, jump_hash,
    jump_hash_key, jump_hash_seeded, verify_monotonicity,
};

pub use raft::{
    AppendEntries, AppendEntriesResponse, LeaderState, LogEntry, LogIndex, Message, NodeId,
    NodeState, PersistentState, RaftCluster, RaftConfig, RaftNode, RaftStats, RequestVote,
    RequestVoteResponse, Term, VolatileState,
};

pub use vector_clock::{
    CausalOrder, DottedVersionVector, Event, Process, ProcessId, Timestamp, VectorClock,
    VectorClockStats, analyze_events, build_causality_graph, causal_sort, compare_events,
    find_concurrent,
};

pub use merkle_tree::{
    ConsistencyProof, HashValue, MerkleNode, MerkleProof, MerkleStats, MerkleTree, ProofDirection,
    ProofStep, SparseMerkleProof, SparseMerkleTree, TreeDiff, analyze_tree, combine_hashes,
    expected_proof_size, hash_data, verify_proof,
};
