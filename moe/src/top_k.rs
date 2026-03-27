// Top-K Routing for Mixture of Experts
//
// The foundational routing mechanism used in modern MoE models like
// GPT-4, Mixtral, DeepSeek, and LLaMA-4. Each token is routed to the
// top-K experts based on router logits.
//
// Key concepts:
// - Router: A small network (often linear) that produces logits for each expert
// - Gating: Softmax over router logits to get routing probabilities
// - Top-K Selection: Only activate K experts (typically 1-2) per token
// - Capacity Factor: Limit tokens per expert to prevent imbalance
//
// References:
// - Shazeer et al. "Outrageously Large Neural Networks" (Switch Transformer)
// - Fedus et al. "Switch Transformers" (Top-1 routing)
// - Lepikhin et al. "GShard" (Top-2 routing)

use std::collections::HashMap;

/// Configuration for Top-K routing
#[derive(Debug, Clone)]
pub struct TopKRouterConfig {
    /// Number of experts to route to per token
    pub top_k: usize,
    /// Total number of experts
    pub num_experts: usize,
    /// Capacity factor (tokens per expert = capacity_factor * tokens / num_experts)
    pub capacity_factor: f64,
    /// Whether to add noise during training for exploration
    pub add_noise: bool,
    /// Noise standard deviation for exploration
    pub noise_std: f64,
    /// Whether to normalize gate values after top-k selection
    pub normalize_gate: bool,
    /// Minimum probability threshold (experts below this are dropped)
    pub min_probability: Option<f64>,
}

impl Default for TopKRouterConfig {
    fn default() -> Self {
        Self {
            top_k: 2,
            num_experts: 8,
            capacity_factor: 1.25,
            add_noise: false,
            noise_std: 1.0,
            normalize_gate: false,
            min_probability: None,
        }
    }
}

impl TopKRouterConfig {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            ..Default::default()
        }
    }

    pub fn with_capacity_factor(mut self, factor: f64) -> Self {
        self.capacity_factor = factor;
        self
    }

    pub fn with_noise(mut self, std: f64) -> Self {
        self.add_noise = true;
        self.noise_std = std;
        self
    }

    pub fn with_normalize_gate(mut self, normalize: bool) -> Self {
        self.normalize_gate = normalize;
        self
    }
}

/// Assignment of a token to an expert
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertAssignment {
    /// Token index in the batch
    pub token_idx: usize,
    /// Expert index
    pub expert_idx: usize,
    /// Gating weight for this assignment
    pub weight: f64,
    /// Rank (0 = top expert, 1 = second, etc.)
    pub rank: usize,
}

/// Result of routing for a single token
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Token index
    pub token_idx: usize,
    /// Selected expert indices (in order of preference)
    pub expert_indices: Vec<usize>,
    /// Gating weights for selected experts
    pub weights: Vec<f64>,
    /// Original router probabilities (softmax output)
    pub probabilities: Vec<f64>,
    /// Whether token was dropped due to capacity
    pub dropped: bool,
}

/// Result of top-k routing for a batch
#[derive(Debug, Clone)]
pub struct TopKRoutingResult {
    /// All routing decisions for the batch
    pub decisions: Vec<RoutingDecision>,
    /// Assignments grouped by expert (expert_idx -> assignments)
    pub expert_assignments: Vec<Vec<ExpertAssignment>>,
    /// Number of tokens assigned to each expert
    pub expert_loads: Vec<usize>,
    /// Number of tokens dropped due to capacity limits
    pub num_dropped: usize,
    /// Average routing probability of selected experts
    pub avg_confidence: f64,
    /// Router probabilities matrix [num_tokens, num_experts]
    pub router_probs: Vec<Vec<f64>>,
}

/// Top-K Router for Mixture of Experts
///
/// Routes each token to the top-K experts based on router logits.
/// Optionally enforces capacity limits to prevent load imbalance.
#[derive(Debug, Clone)]
pub struct TopKRouter {
    pub config: TopKRouterConfig,
}

impl TopKRouter {
    pub fn new(config: TopKRouterConfig) -> Self {
        Self { config }
    }

    /// Route a batch of tokens to experts
    ///
    /// # Arguments
    /// * `router_logits` - Router logits [num_tokens, num_experts]
    /// * `enforce_capacity` - Whether to enforce capacity limits
    ///
    /// # Returns
    /// Routing result with assignments and statistics
    pub fn route(&self, router_logits: &[Vec<f64>], enforce_capacity: bool) -> TopKRoutingResult {
        let num_tokens = router_logits.len();
        if num_tokens == 0 {
            return TopKRoutingResult {
                decisions: vec![],
                expert_assignments: vec![vec![]; self.config.num_experts],
                expert_loads: vec![0; self.config.num_experts],
                num_dropped: 0,
                avg_confidence: 0.0,
                router_probs: vec![],
            };
        }

        // Compute router probabilities (optionally with noise)
        let router_probs: Vec<Vec<f64>> = if self.config.add_noise {
            router_logits
                .iter()
                .enumerate()
                .map(|(i, logits)| {
                    let noisy = noisy_top_k(logits, self.config.top_k, self.config.noise_std, i);
                    softmax(&noisy)
                })
                .collect()
        } else {
            router_logits.iter().map(|logits| softmax(logits)).collect()
        };

        // Calculate expert capacity
        let expert_capacity = if enforce_capacity {
            ((num_tokens as f64 * self.config.capacity_factor) / self.config.num_experts as f64)
                .ceil() as usize
        } else {
            usize::MAX
        };

        // Track expert loads
        let mut expert_loads = vec![0usize; self.config.num_experts];
        let mut expert_assignments: Vec<Vec<ExpertAssignment>> =
            vec![vec![]; self.config.num_experts];
        let mut decisions = Vec::with_capacity(num_tokens);
        let mut num_dropped = 0;
        let mut total_confidence = 0.0;
        let mut total_assigned = 0;

        // Process each token
        for (token_idx, probs) in router_probs.iter().enumerate() {
            // Get top-k experts
            let top_k = top_k_indices(probs, self.config.top_k);

            let mut selected_experts = Vec::new();
            let mut selected_weights = Vec::new();
            let mut token_dropped = false;

            for (rank, &(expert_idx, prob)) in top_k.iter().enumerate() {
                // Check minimum probability threshold
                if let Some(min_prob) = self.config.min_probability
                    && prob < min_prob
                {
                    continue;
                }

                // Check capacity
                if expert_loads[expert_idx] >= expert_capacity {
                    if rank == 0 {
                        token_dropped = true;
                    }
                    continue;
                }

                selected_experts.push(expert_idx);
                selected_weights.push(prob);
                expert_loads[expert_idx] += 1;

                expert_assignments[expert_idx].push(ExpertAssignment {
                    token_idx,
                    expert_idx,
                    weight: prob,
                    rank,
                });

                total_confidence += prob;
                total_assigned += 1;
            }

            // Normalize gate values if configured
            if self.config.normalize_gate && !selected_weights.is_empty() {
                let sum: f64 = selected_weights.iter().sum();
                if sum > 0.0 {
                    for w in &mut selected_weights {
                        *w /= sum;
                    }
                }
            }

            if token_dropped || selected_experts.is_empty() {
                num_dropped += 1;
            }

            decisions.push(RoutingDecision {
                token_idx,
                expert_indices: selected_experts,
                weights: selected_weights,
                probabilities: probs.clone(),
                dropped: token_dropped,
            });
        }

        let avg_confidence = if total_assigned > 0 {
            total_confidence / total_assigned as f64
        } else {
            0.0
        };

        TopKRoutingResult {
            decisions,
            expert_assignments,
            expert_loads,
            num_dropped,
            avg_confidence,
            router_probs,
        }
    }

    /// Route with auxiliary loss computation
    pub fn route_with_aux_loss(&self, router_logits: &[Vec<f64>]) -> (TopKRoutingResult, f64) {
        let result = self.route(router_logits, true);

        // Compute load balancing auxiliary loss
        let num_tokens = router_logits.len() as f64;
        let num_experts = self.config.num_experts as f64;

        // Fraction of tokens routed to each expert
        let fraction_routed: Vec<f64> = result
            .expert_loads
            .iter()
            .map(|&load| load as f64 / num_tokens)
            .collect();

        // Average routing probability to each expert
        let avg_prob: Vec<f64> = (0..self.config.num_experts)
            .map(|e| {
                result
                    .router_probs
                    .iter()
                    .map(|probs| probs[e])
                    .sum::<f64>()
                    / num_tokens
            })
            .collect();

        // Load balance loss: sum(fraction * avg_prob) * num_experts
        let aux_loss: f64 = fraction_routed
            .iter()
            .zip(avg_prob.iter())
            .map(|(f, p)| f * p)
            .sum::<f64>()
            * num_experts;

        (result, aux_loss)
    }
}

/// Compute softmax over logits
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

    logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect()
}

/// Compute router probabilities with optional temperature
pub fn compute_router_probabilities(logits: &[f64], temperature: f64) -> Vec<f64> {
    if temperature == 1.0 {
        softmax(logits)
    } else {
        let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();
        softmax(&scaled)
    }
}

/// Get indices of top-k elements (returns (index, value) pairs sorted by value descending)
pub fn top_k_indices(values: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// Sparse softmax: only compute softmax over top-k elements
pub fn sparse_softmax(logits: &[f64], k: usize) -> Vec<f64> {
    let top_k = top_k_indices(logits, k);
    let mut result = vec![0.0; logits.len()];

    if top_k.is_empty() {
        return result;
    }

    let max_logit = top_k[0].1;
    let exp_sum: f64 = top_k.iter().map(|(_, v)| (v - max_logit).exp()).sum();

    for (idx, value) in top_k {
        result[idx] = (value - max_logit).exp() / exp_sum;
    }

    result
}

/// Add noise to logits for exploration (used during training)
/// Uses a simple deterministic noise based on position for reproducibility
pub fn noisy_top_k(logits: &[f64], k: usize, noise_std: f64, seed: usize) -> Vec<f64> {
    let top_k = top_k_indices(logits, k);
    let threshold = if top_k.len() >= k {
        top_k[k - 1].1
    } else if !top_k.is_empty() {
        top_k[top_k.len() - 1].1
    } else {
        f64::NEG_INFINITY
    };

    // Add noise only to logits near the threshold
    logits
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            // Simple deterministic pseudo-noise for educational purposes
            let noise = ((seed * 31 + i * 17) % 100) as f64 / 100.0 - 0.5;
            x + noise * noise_std * (1.0 / (1.0 + (threshold - x).abs()))
        })
        .collect()
}

/// Compute expert utilization statistics
pub fn compute_expert_utilization(
    routing_result: &TopKRoutingResult,
    num_experts: usize,
) -> HashMap<String, f64> {
    let total_capacity: f64 = routing_result.expert_loads.iter().sum::<usize>() as f64;
    let num_tokens = routing_result.decisions.len() as f64;

    let mut stats = HashMap::new();

    // Average load per expert
    let avg_load = total_capacity / num_experts as f64;
    stats.insert("avg_load".to_string(), avg_load);

    // Load variance
    let load_variance = routing_result
        .expert_loads
        .iter()
        .map(|&load| (load as f64 - avg_load).powi(2))
        .sum::<f64>()
        / num_experts as f64;
    stats.insert("load_variance".to_string(), load_variance);

    // Load imbalance (max/min ratio)
    let max_load = *routing_result.expert_loads.iter().max().unwrap_or(&0) as f64;
    let min_load = *routing_result.expert_loads.iter().min().unwrap_or(&0) as f64;
    let imbalance = if min_load > 0.0 {
        max_load / min_load
    } else {
        f64::INFINITY
    };
    stats.insert("load_imbalance".to_string(), imbalance);

    // Drop rate
    let drop_rate = routing_result.num_dropped as f64 / num_tokens.max(1.0);
    stats.insert("drop_rate".to_string(), drop_rate);

    // Active experts (experts with at least one token)
    let active_experts = routing_result
        .expert_loads
        .iter()
        .filter(|&&load| load > 0)
        .count();
    stats.insert("active_experts".to_string(), active_experts as f64);

    // Expert utilization (fraction of experts used)
    stats.insert(
        "expert_utilization".to_string(),
        active_experts as f64 / num_experts as f64,
    );

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Check sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ordering (higher logit -> higher prob)
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large logits should not overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_top_k_indices() {
        let values = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let top_2 = top_k_indices(&values, 2);

        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, 3); // index of 0.8
        assert_eq!(top_2[1].0, 1); // index of 0.5
    }

    #[test]
    fn test_top_k_indices_k_larger_than_len() {
        let values = vec![0.1, 0.5];
        let top_5 = top_k_indices(&values, 5);

        assert_eq!(top_5.len(), 2);
    }

    #[test]
    fn test_sparse_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sparse = sparse_softmax(&logits, 2);

        // Only top 2 should be non-zero
        assert!(sparse[0].abs() < 1e-10);
        assert!(sparse[1].abs() < 1e-10);
        assert!(sparse[2].abs() < 1e-10);
        assert!(sparse[3] > 0.0);
        assert!(sparse[4] > 0.0);

        // Should sum to 1 (considering only non-zero)
        let sum: f64 = sparse.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_basic_top_k_routing() {
        let config = TopKRouterConfig::new(4, 2);
        let router = TopKRouter::new(config);

        // 3 tokens, 4 experts
        let logits = vec![
            vec![1.0, 2.0, 0.5, 0.1], // Token 0: prefers expert 1, then 0
            vec![0.1, 0.5, 3.0, 2.0], // Token 1: prefers expert 2, then 3
            vec![2.0, 1.0, 1.0, 0.5], // Token 2: prefers expert 0, then 1
        ];

        let result = router.route(&logits, false);

        assert_eq!(result.decisions.len(), 3);

        // Token 0 should route to experts 1 and 0
        assert_eq!(result.decisions[0].expert_indices.len(), 2);
        assert_eq!(result.decisions[0].expert_indices[0], 1);

        // Token 1 should route to experts 2 and 3
        assert_eq!(result.decisions[1].expert_indices[0], 2);

        // Token 2 should route to experts 0 and 1
        assert_eq!(result.decisions[2].expert_indices[0], 0);
    }

    #[test]
    fn test_routing_with_capacity() {
        let config = TopKRouterConfig {
            top_k: 1,
            num_experts: 2,
            capacity_factor: 0.5, // Very low capacity
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        // 4 tokens all prefer expert 0
        let logits = vec![
            vec![10.0, 1.0],
            vec![10.0, 1.0],
            vec![10.0, 1.0],
            vec![10.0, 1.0],
        ];

        let result = router.route(&logits, true);

        // With capacity_factor=0.5, capacity = ceil(4 * 0.5 / 2) = 1
        // So only 1 token can go to each expert
        assert!(result.num_dropped > 0);
        assert!(result.expert_loads[0] <= 1);
    }

    #[test]
    fn test_routing_without_capacity() {
        let config = TopKRouterConfig {
            top_k: 1,
            num_experts: 2,
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        // 4 tokens all prefer expert 0
        let logits = vec![
            vec![10.0, 1.0],
            vec![10.0, 1.0],
            vec![10.0, 1.0],
            vec![10.0, 1.0],
        ];

        let result = router.route(&logits, false);

        // Without capacity limits, all should go to expert 0
        assert_eq!(result.num_dropped, 0);
        assert_eq!(result.expert_loads[0], 4);
        assert_eq!(result.expert_loads[1], 0);
    }

    #[test]
    fn test_gate_normalization() {
        let config = TopKRouterConfig {
            top_k: 2,
            num_experts: 4,
            normalize_gate: true,
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        let logits = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let result = router.route(&logits, false);

        // Weights should sum to 1 after normalization
        let weight_sum: f64 = result.decisions[0].weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_probability_threshold() {
        let config = TopKRouterConfig {
            top_k: 4,
            num_experts: 4,
            min_probability: Some(0.2),
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        // Expert 3 has very low probability
        let logits = vec![vec![2.0, 2.0, 2.0, -10.0]];
        let result = router.route(&logits, false);

        // Expert 3 should be filtered out due to min probability
        assert!(!result.decisions[0].expert_indices.contains(&3));
    }

    #[test]
    fn test_aux_loss() {
        let config = TopKRouterConfig::new(4, 2);
        let router = TopKRouter::new(config);

        // Balanced routing
        let logits = vec![
            vec![2.0, 1.0, 0.5, 0.1],
            vec![1.0, 2.0, 0.5, 0.1],
            vec![0.5, 0.1, 2.0, 1.0],
            vec![0.5, 0.1, 1.0, 2.0],
        ];

        let (result, aux_loss) = router.route_with_aux_loss(&logits);

        // Aux loss should be finite and positive
        assert!(aux_loss.is_finite());
        assert!(aux_loss >= 0.0);

        // With balanced routing, aux loss should be relatively low
        // Perfect balance would give aux_loss = 1.0 (when k=1)
        assert!(aux_loss < 4.0);

        // Check result is valid
        assert_eq!(result.decisions.len(), 4);
    }

    #[test]
    fn test_expert_assignments() {
        let config = TopKRouterConfig::new(3, 2);
        let router = TopKRouter::new(config);

        let logits = vec![
            vec![3.0, 2.0, 1.0], // Token 0: experts 0, 1
            vec![1.0, 3.0, 2.0], // Token 1: experts 1, 2
            vec![2.0, 1.0, 3.0], // Token 2: experts 2, 0
        ];

        let result = router.route(&logits, false);

        // Check expert 0 assignments
        let expert_0_tokens: Vec<usize> = result.expert_assignments[0]
            .iter()
            .map(|a| a.token_idx)
            .collect();
        assert!(expert_0_tokens.contains(&0)); // Token 0's first choice
        assert!(expert_0_tokens.contains(&2)); // Token 2's second choice

        // Check expert 1 assignments
        let expert_1_tokens: Vec<usize> = result.expert_assignments[1]
            .iter()
            .map(|a| a.token_idx)
            .collect();
        assert!(expert_1_tokens.contains(&0)); // Token 0's second choice
        assert!(expert_1_tokens.contains(&1)); // Token 1's first choice
    }

    #[test]
    fn test_empty_batch() {
        let config = TopKRouterConfig::new(4, 2);
        let router = TopKRouter::new(config);

        let result = router.route(&[], false);

        assert!(result.decisions.is_empty());
        assert_eq!(result.num_dropped, 0);
        assert_eq!(result.avg_confidence, 0.0);
    }

    #[test]
    fn test_compute_expert_utilization() {
        let config = TopKRouterConfig::new(4, 2);
        let router = TopKRouter::new(config);

        // Each token routes to 2 experts, ensuring all 4 experts are used
        let logits = vec![
            vec![3.0, 2.0, 0.5, 0.1], // Token 0: experts 0, 1
            vec![0.5, 0.1, 3.0, 2.0], // Token 1: experts 2, 3
            vec![2.0, 3.0, 0.5, 0.1], // Token 2: experts 1, 0
            vec![0.1, 0.5, 2.0, 3.0], // Token 3: experts 3, 2
        ];

        let result = router.route(&logits, false);
        let stats = compute_expert_utilization(&result, 4);

        assert!(stats.contains_key("avg_load"));
        assert!(stats.contains_key("load_variance"));
        assert!(stats.contains_key("load_imbalance"));
        assert!(stats.contains_key("drop_rate"));
        assert!(stats.contains_key("active_experts"));
        assert!(stats.contains_key("expert_utilization"));

        // All experts should be active
        assert_eq!(stats["active_experts"] as usize, 4);
        // No drops without capacity limit
        assert_eq!(stats["drop_rate"], 0.0);
    }

    #[test]
    fn test_noisy_top_k() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let noisy = noisy_top_k(&logits, 2, 0.1, 42);

        // Output should have same length
        assert_eq!(noisy.len(), logits.len());

        // Values should be similar to original (small noise)
        for (orig, noised) in logits.iter().zip(noisy.iter()) {
            assert!((orig - noised).abs() < 1.0);
        }
    }

    #[test]
    fn test_router_probabilities_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0];

        let probs_t1 = compute_router_probabilities(&logits, 1.0);
        let probs_t2 = compute_router_probabilities(&logits, 2.0);
        let probs_t05 = compute_router_probabilities(&logits, 0.5);

        // Higher temperature -> more uniform
        let entropy_t1: f64 = -probs_t1.iter().map(|&p| p * p.ln()).sum::<f64>();
        let entropy_t2: f64 = -probs_t2.iter().map(|&p| p * p.ln()).sum::<f64>();
        let entropy_t05: f64 = -probs_t05.iter().map(|&p| p * p.ln()).sum::<f64>();

        assert!(entropy_t2 > entropy_t1);
        assert!(entropy_t1 > entropy_t05);
    }

    #[test]
    fn test_routing_decision_weights() {
        let config = TopKRouterConfig {
            top_k: 2,
            num_experts: 4,
            normalize_gate: false,
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        let logits = vec![vec![0.0, 0.0, 0.0, 0.0]]; // Uniform
        let result = router.route(&logits, false);

        // With uniform logits, weights should be equal
        let weights = &result.decisions[0].weights;
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - weights[1]).abs() < 1e-10);
    }

    #[test]
    fn test_large_batch() {
        let config = TopKRouterConfig {
            top_k: 2,
            num_experts: 8,
            capacity_factor: 1.5,
            ..Default::default()
        };
        let router = TopKRouter::new(config);

        // 100 tokens with varying preferences
        let logits: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let mut l = vec![0.0; 8];
                l[i % 8] = 3.0; // Primary expert
                l[(i + 1) % 8] = 2.0; // Secondary expert
                l
            })
            .collect();

        let result = router.route(&logits, true);

        assert_eq!(result.decisions.len(), 100);

        // With round-robin pattern, load should be relatively balanced
        let max_load = *result.expert_loads.iter().max().unwrap();
        let min_load = *result.expert_loads.iter().min().unwrap();
        assert!(max_load as f64 / min_load.max(1) as f64 <= 2.0);
    }
}
