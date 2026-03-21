// Distributed Systems Algorithms
// Essential algorithms for building scalable distributed systems

pub mod consistent_hashing;
pub mod rendezvous_hashing;

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
