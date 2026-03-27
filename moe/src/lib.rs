// Mixture of Experts (MoE) Routing Algorithms
// Hot topic in 2025 with DeepSeek, Qwen3, LLaMA-4

pub mod expert_choice;
pub mod load_balancing;
pub mod optimal_transport;
pub mod top_k;

pub use top_k::{
    ExpertAssignment, RoutingDecision, TopKRouter, TopKRouterConfig, TopKRoutingResult,
    compute_router_probabilities, noisy_top_k, softmax, sparse_softmax, top_k_indices,
};

pub use expert_choice::{
    ExpertChoiceConfig, ExpertChoiceResult, ExpertChoiceRouter, TokenAssignment,
    compute_expert_scores, expert_capacity,
};

pub use optimal_transport::{
    OTConfig, OTRouter, OTRoutingResult, SinkhornResult, balanced_assignment,
    compute_transport_plan, sinkhorn_iterations,
};

pub use load_balancing::{
    LoadBalancingConfig, LoadBalancingLoss, LoadBalancingStats, compute_auxiliary_loss,
    compute_load_balance_loss, compute_router_z_loss, expert_utilization, routing_entropy,
};
