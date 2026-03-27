// Load Balancing Losses for Mixture of Experts
//
// Auxiliary losses that encourage balanced expert utilization and prevent
// routing collapse (where all tokens go to a few experts). Essential for
// stable MoE training.
//
// Key loss types:
// - Load Balance Loss: Penalizes uneven token distribution
// - Router Z-Loss: Regularizes router logits to prevent instability
// - Importance Loss: Penalizes experts with disproportionate importance
//
// References:
// - Fedus et al. "Switch Transformers" (2022)
// - Lepikhin et al. "GShard" (2020)
// - Riquelme et al. "Scaling Vision with Sparse Mixture of Experts" (2021)

/// Configuration for load balancing loss
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Weight for load balance loss
    pub load_balance_weight: f64,
    /// Weight for router z-loss
    pub z_loss_weight: f64,
    /// Target load fraction per expert (default: 1/num_experts)
    pub target_load: Option<f64>,
    /// Whether to use importance weighting
    pub use_importance: bool,
    /// Smoothing factor for importance calculation
    pub importance_smoothing: f64,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            load_balance_weight: 0.01,
            z_loss_weight: 0.001,
            target_load: None,
            use_importance: true,
            importance_smoothing: 1e-6,
        }
    }
}

impl LoadBalancingConfig {
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            ..Default::default()
        }
    }

    pub fn with_load_balance_weight(mut self, weight: f64) -> Self {
        self.load_balance_weight = weight;
        self
    }

    pub fn with_z_loss_weight(mut self, weight: f64) -> Self {
        self.z_loss_weight = weight;
        self
    }

    pub fn with_target_load(mut self, target: f64) -> Self {
        self.target_load = Some(target);
        self
    }
}

/// Statistics from load balancing loss computation
#[derive(Debug, Clone, Default)]
pub struct LoadBalancingStats {
    /// Load balance loss value
    pub load_balance_loss: f64,
    /// Router z-loss value
    pub z_loss: f64,
    /// Total auxiliary loss (weighted sum)
    pub total_loss: f64,
    /// Per-expert load fractions
    pub expert_loads: Vec<f64>,
    /// Per-expert importance scores
    pub expert_importance: Vec<f64>,
    /// Load variance (lower is better)
    pub load_variance: f64,
    /// Load entropy (higher means more balanced)
    pub load_entropy: f64,
    /// Gini coefficient (0 = perfect equality)
    pub gini_coefficient: f64,
}

/// Load Balancing Loss Calculator
///
/// Computes auxiliary losses for MoE training.
#[derive(Debug, Clone)]
pub struct LoadBalancingLoss {
    pub config: LoadBalancingConfig,
}

impl LoadBalancingLoss {
    pub fn new(config: LoadBalancingConfig) -> Self {
        Self { config }
    }

    /// Compute all load balancing losses
    ///
    /// # Arguments
    /// * `router_probs` - Router probabilities [num_tokens, num_experts]
    /// * `expert_mask` - Binary mask of selected experts [num_tokens, num_experts]
    /// * `router_logits` - Raw router logits (for z-loss)
    ///
    /// # Returns
    /// Load balancing statistics including loss values
    pub fn compute(
        &self,
        router_probs: &[Vec<f64>],
        expert_mask: &[Vec<f64>],
        router_logits: Option<&[Vec<f64>]>,
    ) -> LoadBalancingStats {
        let num_tokens = router_probs.len();
        if num_tokens == 0 {
            return LoadBalancingStats::default();
        }

        // Compute load balance loss
        let (load_balance_loss, expert_loads, expert_importance) =
            self.compute_load_balance_loss(router_probs, expert_mask);

        // Compute router z-loss if logits provided
        let z_loss = router_logits.map_or(0.0, compute_router_z_loss);

        // Total weighted loss
        let total_loss = self.config.load_balance_weight * load_balance_loss
            + self.config.z_loss_weight * z_loss;

        // Compute additional statistics
        let load_variance = self.compute_load_variance(&expert_loads);
        let load_entropy = routing_entropy(&expert_loads);
        let gini_coefficient = self.compute_gini(&expert_loads);

        LoadBalancingStats {
            load_balance_loss,
            z_loss,
            total_loss,
            expert_loads,
            expert_importance,
            load_variance,
            load_entropy,
            gini_coefficient,
        }
    }

    /// Compute load balance loss (Switch Transformer style)
    ///
    /// Loss = num_experts * sum_i (f_i * P_i)
    /// where f_i = fraction of tokens routed to expert i
    /// and P_i = mean router probability for expert i
    fn compute_load_balance_loss(
        &self,
        router_probs: &[Vec<f64>],
        expert_mask: &[Vec<f64>],
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let num_tokens = router_probs.len() as f64;
        let num_experts = self.config.num_experts;

        // Compute fraction of tokens routed to each expert (f_i)
        let expert_loads: Vec<f64> = (0..num_experts)
            .map(|e| expert_mask.iter().map(|mask| mask[e]).sum::<f64>() / num_tokens)
            .collect();

        // Compute mean router probability for each expert (P_i)
        let expert_importance: Vec<f64> = (0..num_experts)
            .map(|e| router_probs.iter().map(|probs| probs[e]).sum::<f64>() / num_tokens)
            .collect();

        // Load balance loss
        let loss: f64 = expert_loads
            .iter()
            .zip(expert_importance.iter())
            .map(|(f, p)| f * p)
            .sum::<f64>()
            * num_experts as f64;

        (loss, expert_loads, expert_importance)
    }

    fn compute_load_variance(&self, loads: &[f64]) -> f64 {
        if loads.is_empty() {
            return 0.0;
        }

        let mean = loads.iter().sum::<f64>() / loads.len() as f64;
        loads.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / loads.len() as f64
    }

    fn compute_gini(&self, loads: &[f64]) -> f64 {
        if loads.is_empty() {
            return 0.0;
        }

        let n = loads.len() as f64;
        let mean = loads.iter().sum::<f64>() / n;

        if mean == 0.0 {
            return 0.0;
        }

        let mut sorted = loads.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum_diff: f64 = (0..loads.len())
            .map(|i| (2.0 * (i + 1) as f64 - n - 1.0) * sorted[i])
            .sum();

        sum_diff / (n * n * mean)
    }
}

/// Compute load balance loss directly from router probabilities and masks
///
/// This is the standard Switch Transformer load balance loss:
/// L = alpha * num_experts * sum_i(f_i * P_i)
///
/// # Arguments
/// * `router_probs` - Router probabilities [num_tokens, num_experts]
/// * `expert_mask` - Binary selection mask [num_tokens, num_experts]
/// * `num_experts` - Number of experts
///
/// # Returns
/// Load balance loss value (before weighting by alpha)
pub fn compute_load_balance_loss(
    router_probs: &[Vec<f64>],
    expert_mask: &[Vec<f64>],
    num_experts: usize,
) -> f64 {
    let num_tokens = router_probs.len() as f64;
    if num_tokens == 0.0 {
        return 0.0;
    }

    let mut loss = 0.0;

    for e in 0..num_experts {
        // f_i: fraction of tokens routed to expert e
        let f_i: f64 = expert_mask.iter().map(|mask| mask[e]).sum::<f64>() / num_tokens;

        // P_i: mean probability of routing to expert e
        let p_i: f64 = router_probs.iter().map(|probs| probs[e]).sum::<f64>() / num_tokens;

        loss += f_i * p_i;
    }

    loss * num_experts as f64
}

/// Compute router z-loss for logit regularization
///
/// L_z = (1/n) * sum_i log(sum_j exp(x_ij))^2
///
/// This penalizes large router logits which can cause instability.
pub fn compute_router_z_loss(router_logits: &[Vec<f64>]) -> f64 {
    if router_logits.is_empty() {
        return 0.0;
    }

    let num_tokens = router_logits.len() as f64;

    let z_loss: f64 = router_logits
        .iter()
        .map(|logits| {
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let logsumexp = max_logit
                + logits
                    .iter()
                    .map(|&x| (x - max_logit).exp())
                    .sum::<f64>()
                    .ln();
            logsumexp.powi(2)
        })
        .sum();

    z_loss / num_tokens
}

/// Compute combined auxiliary loss
///
/// # Arguments
/// * `router_probs` - Router probabilities
/// * `expert_mask` - Expert selection mask
/// * `router_logits` - Router logits (optional)
/// * `num_experts` - Number of experts
/// * `load_balance_weight` - Weight for load balance loss
/// * `z_loss_weight` - Weight for z-loss
pub fn compute_auxiliary_loss(
    router_probs: &[Vec<f64>],
    expert_mask: &[Vec<f64>],
    router_logits: Option<&[Vec<f64>]>,
    num_experts: usize,
    load_balance_weight: f64,
    z_loss_weight: f64,
) -> f64 {
    let lb_loss = compute_load_balance_loss(router_probs, expert_mask, num_experts);
    let z_loss = router_logits.map_or(0.0, compute_router_z_loss);

    load_balance_weight * lb_loss + z_loss_weight * z_loss
}

/// Compute expert utilization metrics
pub fn expert_utilization(expert_loads: &[f64]) -> ExpertUtilizationStats {
    if expert_loads.is_empty() {
        return ExpertUtilizationStats::default();
    }

    let num_experts = expert_loads.len();
    let total: f64 = expert_loads.iter().sum();

    // Normalize to get distribution
    let distribution: Vec<f64> = if total > 0.0 {
        expert_loads.iter().map(|&l| l / total).collect()
    } else {
        vec![0.0; num_experts]
    };

    // Effective number of experts (exp of entropy)
    let entropy = routing_entropy(&distribution);
    let effective_experts = entropy.exp();

    // Active experts (non-zero load)
    let active_experts = expert_loads.iter().filter(|&&l| l > 1e-10).count();

    // Max/min ratio
    let max_load = expert_loads.iter().cloned().fold(0.0, f64::max);
    let min_load = expert_loads.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_min_ratio = if min_load > 1e-10 {
        max_load / min_load
    } else {
        f64::INFINITY
    };

    // Coefficient of variation
    let mean = total / num_experts as f64;
    let variance = expert_loads
        .iter()
        .map(|&l| (l - mean).powi(2))
        .sum::<f64>()
        / num_experts as f64;
    let cv = if mean > 1e-10 {
        variance.sqrt() / mean
    } else {
        0.0
    };

    ExpertUtilizationStats {
        distribution,
        effective_experts,
        active_experts,
        max_min_ratio,
        coefficient_of_variation: cv,
        entropy,
    }
}

/// Expert utilization statistics
#[derive(Debug, Clone, Default)]
pub struct ExpertUtilizationStats {
    /// Normalized load distribution
    pub distribution: Vec<f64>,
    /// Effective number of experts (exp of entropy)
    pub effective_experts: f64,
    /// Number of experts with non-zero load
    pub active_experts: usize,
    /// Ratio of max to min load
    pub max_min_ratio: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_of_variation: f64,
    /// Entropy of load distribution
    pub entropy: f64,
}

/// Compute entropy of routing distribution
pub fn routing_entropy(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Compute importance-weighted load balance loss
///
/// Similar to standard load balance but weights experts by their
/// cumulative importance across the batch.
pub fn importance_weighted_loss(
    router_probs: &[Vec<f64>],
    expert_mask: &[Vec<f64>],
    num_experts: usize,
) -> (f64, Vec<f64>) {
    let num_tokens = router_probs.len() as f64;
    if num_tokens == 0.0 {
        return (0.0, vec![0.0; num_experts]);
    }

    // Compute importance (sum of probabilities) for each expert
    let importance: Vec<f64> = (0..num_experts)
        .map(|e| router_probs.iter().map(|probs| probs[e]).sum::<f64>())
        .collect();

    let total_importance: f64 = importance.iter().sum();
    let normalized_importance: Vec<f64> = if total_importance > 0.0 {
        importance.iter().map(|&i| i / total_importance).collect()
    } else {
        vec![1.0 / num_experts as f64; num_experts]
    };

    // Compute load for each expert
    let load: Vec<f64> = (0..num_experts)
        .map(|e| expert_mask.iter().map(|mask| mask[e]).sum::<f64>() / num_tokens)
        .collect();

    // Loss is variance between normalized importance and load
    let loss: f64 = normalized_importance
        .iter()
        .zip(load.iter())
        .map(|(i, l)| (i - l).powi(2))
        .sum();

    (loss, normalized_importance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_balance_loss_balanced() {
        // Perfectly balanced: each expert gets 1/4 of tokens
        let router_probs = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];

        let expert_mask = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let loss = compute_load_balance_loss(&router_probs, &expert_mask, 4);

        // With perfect balance: f_i = 0.25, P_i = 0.25
        // Loss = 4 * (4 * 0.25 * 0.25) = 4 * 0.25 = 1.0
        assert!((loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_balance_loss_imbalanced() {
        // Imbalanced: all tokens go to expert 0
        let router_probs = vec![
            vec![0.7, 0.1, 0.1, 0.1],
            vec![0.7, 0.1, 0.1, 0.1],
            vec![0.7, 0.1, 0.1, 0.1],
            vec![0.7, 0.1, 0.1, 0.1],
        ];

        let expert_mask = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];

        let loss = compute_load_balance_loss(&router_probs, &expert_mask, 4);

        // f_0 = 1.0, P_0 = 0.7, others = 0
        // Loss = 4 * (1.0 * 0.7) = 2.8
        assert!((loss - 2.8).abs() < 1e-10);
    }

    #[test]
    fn test_router_z_loss() {
        let logits = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let z_loss = compute_router_z_loss(&logits);

        // Should be positive
        assert!(z_loss > 0.0);

        // Larger logits should give larger z-loss
        let large_logits = vec![vec![10.0, 20.0, 30.0], vec![20.0, 30.0, 40.0]];
        let large_z_loss = compute_router_z_loss(&large_logits);
        assert!(large_z_loss > z_loss);
    }

    #[test]
    fn test_router_z_loss_small() {
        // Small logits should give small z-loss
        let logits = vec![vec![0.1, 0.1, 0.1]];
        let z_loss = compute_router_z_loss(&logits);

        // logsumexp([0.1, 0.1, 0.1]) = 0.1 + log(3) ≈ 1.2
        // z_loss = 1.2^2 ≈ 1.4
        assert!(z_loss < 2.0);
    }

    #[test]
    fn test_compute_auxiliary_loss() {
        let router_probs = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let expert_mask = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let router_logits = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

        let loss = compute_auxiliary_loss(
            &router_probs,
            &expert_mask,
            Some(&router_logits),
            2,
            0.01,
            0.001,
        );

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_routing_entropy() {
        // Uniform distribution has max entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = routing_entropy(&uniform);

        // Peaked distribution has low entropy
        let peaked = vec![0.9, 0.05, 0.025, 0.025];
        let peaked_entropy = routing_entropy(&peaked);

        assert!(uniform_entropy > peaked_entropy);

        // Max entropy for 4 experts = ln(4) ≈ 1.386
        assert!((uniform_entropy - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_routing_entropy_degenerate() {
        // All mass on one expert
        let degenerate = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = routing_entropy(&degenerate);

        // Entropy should be 0
        assert!(entropy.abs() < 1e-10);
    }

    #[test]
    fn test_expert_utilization() {
        let loads = vec![25.0, 25.0, 25.0, 25.0];
        let stats = expert_utilization(&loads);

        // All 4 experts active
        assert_eq!(stats.active_experts, 4);

        // Effective experts should be 4 (uniform)
        assert!((stats.effective_experts - 4.0).abs() < 0.1);

        // Max/min ratio should be 1 (balanced)
        assert!((stats.max_min_ratio - 1.0).abs() < 1e-10);

        // CV should be 0 (no variation)
        assert!(stats.coefficient_of_variation.abs() < 1e-10);
    }

    #[test]
    fn test_expert_utilization_imbalanced() {
        let loads = vec![90.0, 5.0, 3.0, 2.0];
        let stats = expert_utilization(&loads);

        // All active
        assert_eq!(stats.active_experts, 4);

        // Effective experts should be low (concentrated)
        assert!(stats.effective_experts < 2.0);

        // High max/min ratio
        assert!(stats.max_min_ratio > 10.0);

        // High CV
        assert!(stats.coefficient_of_variation > 1.0);
    }

    #[test]
    fn test_load_balancing_loss_full() {
        let config = LoadBalancingConfig::new(4);
        let loss_fn = LoadBalancingLoss::new(config);

        let router_probs = vec![
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.1, 0.4, 0.3, 0.2],
            vec![0.2, 0.1, 0.4, 0.3],
            vec![0.3, 0.2, 0.1, 0.4],
        ];

        let expert_mask = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let router_logits = vec![
            vec![2.0, 1.5, 1.0, 0.5],
            vec![0.5, 2.0, 1.5, 1.0],
            vec![1.0, 0.5, 2.0, 1.5],
            vec![1.5, 1.0, 0.5, 2.0],
        ];

        let stats = loss_fn.compute(&router_probs, &expert_mask, Some(&router_logits));

        // All metrics should be computed
        assert!(stats.load_balance_loss.is_finite());
        assert!(stats.z_loss.is_finite());
        assert!(stats.total_loss.is_finite());
        assert_eq!(stats.expert_loads.len(), 4);
        assert_eq!(stats.expert_importance.len(), 4);

        // Gini should be between 0 and 1
        assert!(stats.gini_coefficient >= 0.0 && stats.gini_coefficient <= 1.0);
    }

    #[test]
    fn test_load_variance() {
        let config = LoadBalancingConfig::new(4);
        let loss_fn = LoadBalancingLoss::new(config);

        // Zero variance (uniform)
        let uniform_loads = vec![0.25, 0.25, 0.25, 0.25];
        let variance = loss_fn.compute_load_variance(&uniform_loads);
        assert!(variance.abs() < 1e-10);

        // High variance
        let imbalanced_loads = vec![0.9, 0.05, 0.03, 0.02];
        let high_variance = loss_fn.compute_load_variance(&imbalanced_loads);
        assert!(high_variance > 0.1);
    }

    #[test]
    fn test_gini_coefficient() {
        let config = LoadBalancingConfig::new(4);
        let loss_fn = LoadBalancingLoss::new(config);

        // Perfect equality
        let equal = vec![25.0, 25.0, 25.0, 25.0];
        let gini_equal = loss_fn.compute_gini(&equal);
        assert!(gini_equal.abs() < 1e-10);

        // High inequality
        let unequal = vec![100.0, 0.0, 0.0, 0.0];
        let gini_unequal = loss_fn.compute_gini(&unequal);
        assert!(gini_unequal > 0.5);
    }

    #[test]
    fn test_importance_weighted_loss() {
        let router_probs = vec![
            vec![0.6, 0.4],
            vec![0.6, 0.4],
            vec![0.4, 0.6],
            vec![0.4, 0.6],
        ];

        let expert_mask = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];

        let (loss, importance) = importance_weighted_loss(&router_probs, &expert_mask, 2);

        // Importance should sum to 1
        let importance_sum: f64 = importance.iter().sum();
        assert!((importance_sum - 1.0).abs() < 1e-10);

        // With balanced routing matching importance, loss should be low
        assert!(loss < 0.1);
    }

    #[test]
    fn test_empty_inputs() {
        let config = LoadBalancingConfig::new(4);
        let loss_fn = LoadBalancingLoss::new(config);

        let stats = loss_fn.compute(&[], &[], None);

        assert_eq!(stats.load_balance_loss, 0.0);
        assert_eq!(stats.z_loss, 0.0);
        assert!(stats.expert_loads.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = LoadBalancingConfig::new(8)
            .with_load_balance_weight(0.02)
            .with_z_loss_weight(0.005)
            .with_target_load(0.125);

        assert_eq!(config.num_experts, 8);
        assert_eq!(config.load_balance_weight, 0.02);
        assert_eq!(config.z_loss_weight, 0.005);
        assert_eq!(config.target_load, Some(0.125));
    }

    #[test]
    fn test_stats_consistency() {
        let config = LoadBalancingConfig::new(4);
        let loss_fn = LoadBalancingLoss::new(config);

        let router_probs = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];

        let expert_mask = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let stats = loss_fn.compute(&router_probs, &expert_mask, None);

        // Total loss should equal weighted sum
        let expected_total = stats.load_balance_loss * 0.01 + stats.z_loss * 0.001;
        assert!((stats.total_loss - expected_total).abs() < 1e-10);

        // Expert loads should sum to 1
        let load_sum: f64 = stats.expert_loads.iter().sum();
        assert!((load_sum - 1.0).abs() < 1e-10);
    }
}
