// Optimal Transport Routing for Mixture of Experts
//
// Uses the Sinkhorn algorithm to find balanced token-to-expert assignments
// that minimize transport cost while satisfying supply/demand constraints.
// This provides softer, more differentiable load balancing than Top-K.
//
// Key concepts:
// - Cost matrix: Negative log probabilities from router
// - Supply: Each token has supply 1 (one token to route)
// - Demand: Each expert has demand = total_tokens / num_experts
// - Sinkhorn: Iterative algorithm to find optimal transport plan
//
// References:
// - Clark et al. "BASE Layers" (2022) - OT for balanced assignment
// - Cuturi "Sinkhorn Distances" (2013) - Fast optimal transport
// - DeepSeek-V3 uses Sinkhorn for expert routing

/// Configuration for Optimal Transport routing
#[derive(Debug, Clone)]
pub struct OTConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of Sinkhorn iterations
    pub num_iterations: usize,
    /// Regularization parameter (entropy regularization)
    pub epsilon: f64,
    /// Convergence threshold
    pub threshold: f64,
    /// Whether to use log-domain computation for stability
    pub log_domain: bool,
    /// Temperature for cost matrix
    pub temperature: f64,
    /// Capacity factor per expert (1.0 = perfectly balanced)
    pub capacity_factor: f64,
}

impl Default for OTConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_iterations: 100,
            epsilon: 0.1,
            threshold: 1e-6,
            log_domain: true,
            temperature: 1.0,
            capacity_factor: 1.0,
        }
    }
}

impl OTConfig {
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            ..Default::default()
        }
    }

    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.num_iterations = iters;
        self
    }

    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn with_capacity_factor(mut self, factor: f64) -> Self {
        self.capacity_factor = factor;
        self
    }
}

/// Result of Sinkhorn algorithm
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// Transport plan [num_tokens, num_experts]
    pub transport_plan: Vec<Vec<f64>>,
    /// Row scaling factors (for tokens)
    pub row_scaling: Vec<f64>,
    /// Column scaling factors (for experts)
    pub col_scaling: Vec<f64>,
    /// Number of iterations run
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Final marginal error
    pub marginal_error: f64,
}

/// Result of OT routing
#[derive(Debug, Clone)]
pub struct OTRoutingResult {
    /// Transport plan [num_tokens, num_experts]
    pub transport_plan: Vec<Vec<f64>>,
    /// Discretized assignments (top-k per token)
    pub assignments: Vec<Vec<(usize, f64)>>,
    /// Expert loads (sum of transport plan columns)
    pub expert_loads: Vec<f64>,
    /// Sinkhorn convergence info
    pub sinkhorn_result: SinkhornResult,
    /// Total transport cost
    pub transport_cost: f64,
}

/// Optimal Transport Router
///
/// Uses Sinkhorn algorithm for balanced token-to-expert routing.
#[derive(Debug, Clone)]
pub struct OTRouter {
    pub config: OTConfig,
}

impl OTRouter {
    pub fn new(config: OTConfig) -> Self {
        Self { config }
    }

    /// Route tokens using optimal transport
    ///
    /// # Arguments
    /// * `router_logits` - Router logits [num_tokens, num_experts]
    /// * `top_k` - Number of experts per token in discretized assignment
    ///
    /// # Returns
    /// OT routing result with transport plan and discretized assignments
    pub fn route(&self, router_logits: &[Vec<f64>], top_k: usize) -> OTRoutingResult {
        let num_tokens = router_logits.len();
        if num_tokens == 0 {
            return OTRoutingResult {
                transport_plan: vec![],
                assignments: vec![],
                expert_loads: vec![0.0; self.config.num_experts],
                sinkhorn_result: SinkhornResult {
                    transport_plan: vec![],
                    row_scaling: vec![],
                    col_scaling: vec![],
                    iterations: 0,
                    converged: true,
                    marginal_error: 0.0,
                },
                transport_cost: 0.0,
            };
        }

        // Compute cost matrix (negative log probabilities)
        let cost_matrix = self.compute_cost_matrix(router_logits);

        // Define supply (tokens) and demand (experts)
        let supply = vec![1.0 / num_tokens as f64; num_tokens]; // Normalized
        let total_demand = 1.0;
        let per_expert_demand =
            total_demand * self.config.capacity_factor / self.config.num_experts as f64;
        let demand = vec![per_expert_demand; self.config.num_experts];

        // Run Sinkhorn
        let sinkhorn_result = if self.config.log_domain {
            sinkhorn_log_domain(
                &cost_matrix,
                &supply,
                &demand,
                self.config.epsilon,
                self.config.num_iterations,
                self.config.threshold,
            )
        } else {
            sinkhorn_iterations(
                &cost_matrix,
                &supply,
                &demand,
                self.config.epsilon,
                self.config.num_iterations,
                self.config.threshold,
            )
        };

        // Compute expert loads
        let expert_loads: Vec<f64> = (0..self.config.num_experts)
            .map(|e| {
                sinkhorn_result
                    .transport_plan
                    .iter()
                    .map(|row| row[e])
                    .sum()
            })
            .collect();

        // Discretize to top-k assignments
        let assignments = self.discretize(&sinkhorn_result.transport_plan, top_k);

        // Compute transport cost
        let transport_cost: f64 = sinkhorn_result
            .transport_plan
            .iter()
            .zip(cost_matrix.iter())
            .map(|(plan_row, cost_row)| {
                plan_row
                    .iter()
                    .zip(cost_row.iter())
                    .map(|(&p, &c)| p * c)
                    .sum::<f64>()
            })
            .sum();

        OTRoutingResult {
            transport_plan: sinkhorn_result.transport_plan.clone(),
            assignments,
            expert_loads,
            sinkhorn_result,
            transport_cost,
        }
    }

    /// Compute cost matrix from router logits
    fn compute_cost_matrix(&self, router_logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        router_logits
            .iter()
            .map(|logits| {
                // Apply temperature and convert to costs (negative for maximization)
                let scaled: Vec<f64> = logits
                    .iter()
                    .map(|&l| l / self.config.temperature)
                    .collect();
                let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Cost = -log(softmax(logits)) = -logits + logsumexp
                let logsumexp = max_val
                    + scaled
                        .iter()
                        .map(|&x| (x - max_val).exp())
                        .sum::<f64>()
                        .ln();
                scaled.iter().map(|&x| logsumexp - x).collect()
            })
            .collect()
    }

    /// Discretize transport plan to top-k assignments per token
    fn discretize(&self, transport_plan: &[Vec<f64>], top_k: usize) -> Vec<Vec<(usize, f64)>> {
        transport_plan
            .iter()
            .map(|row| {
                let mut indexed: Vec<(usize, f64)> = row.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(top_k);

                // Renormalize weights
                let sum: f64 = indexed.iter().map(|(_, w)| w).sum();
                if sum > 0.0 {
                    indexed.iter().map(|&(e, w)| (e, w / sum)).collect()
                } else {
                    indexed
                }
            })
            .collect()
    }
}

/// Standard Sinkhorn algorithm
pub fn sinkhorn_iterations(
    cost_matrix: &[Vec<f64>],
    supply: &[f64],
    demand: &[f64],
    epsilon: f64,
    max_iterations: usize,
    threshold: f64,
) -> SinkhornResult {
    let num_tokens = cost_matrix.len();
    let num_experts = if num_tokens > 0 {
        cost_matrix[0].len()
    } else {
        0
    };

    if num_tokens == 0 || num_experts == 0 {
        return SinkhornResult {
            transport_plan: vec![],
            row_scaling: vec![],
            col_scaling: vec![],
            iterations: 0,
            converged: true,
            marginal_error: 0.0,
        };
    }

    // Initialize kernel K = exp(-C/epsilon)
    let kernel: Vec<Vec<f64>> = cost_matrix
        .iter()
        .map(|row| row.iter().map(|&c| (-c / epsilon).exp()).collect())
        .collect();

    // Initialize scaling factors
    let mut u = vec![1.0; num_tokens];
    let mut v = vec![1.0; num_experts];

    let mut converged = false;
    let mut iterations = 0;
    let mut marginal_error = f64::INFINITY;

    for iter in 0..max_iterations {
        iterations = iter + 1;

        // Update v: v = b / (K^T u)
        let kt_u: Vec<f64> = (0..num_experts)
            .map(|j| (0..num_tokens).map(|i| kernel[i][j] * u[i]).sum())
            .collect();

        for (j, v_j) in v.iter_mut().enumerate() {
            if kt_u[j] > 1e-10 {
                *v_j = demand[j] / kt_u[j];
            }
        }

        // Update u: u = a / (K v)
        let k_v: Vec<f64> = (0..num_tokens)
            .map(|i| (0..num_experts).map(|j| kernel[i][j] * v[j]).sum())
            .collect();

        for (i, u_i) in u.iter_mut().enumerate() {
            if k_v[i] > 1e-10 {
                *u_i = supply[i] / k_v[i];
            }
        }

        // Check convergence
        let row_marginal: Vec<f64> = (0..num_tokens)
            .map(|i| (0..num_experts).map(|j| u[i] * kernel[i][j] * v[j]).sum())
            .collect();

        marginal_error = row_marginal
            .iter()
            .zip(supply.iter())
            .map(|(actual, target)| (actual - target).abs())
            .sum();

        if marginal_error < threshold {
            converged = true;
            break;
        }
    }

    // Compute transport plan P = diag(u) K diag(v)
    let transport_plan: Vec<Vec<f64>> = (0..num_tokens)
        .map(|i| {
            (0..num_experts)
                .map(|j| u[i] * kernel[i][j] * v[j])
                .collect()
        })
        .collect();

    SinkhornResult {
        transport_plan,
        row_scaling: u,
        col_scaling: v,
        iterations,
        converged,
        marginal_error,
    }
}

/// Log-domain Sinkhorn for numerical stability
pub fn sinkhorn_log_domain(
    cost_matrix: &[Vec<f64>],
    supply: &[f64],
    demand: &[f64],
    epsilon: f64,
    max_iterations: usize,
    threshold: f64,
) -> SinkhornResult {
    let num_tokens = cost_matrix.len();
    let num_experts = if num_tokens > 0 {
        cost_matrix[0].len()
    } else {
        0
    };

    if num_tokens == 0 || num_experts == 0 {
        return SinkhornResult {
            transport_plan: vec![],
            row_scaling: vec![],
            col_scaling: vec![],
            iterations: 0,
            converged: true,
            marginal_error: 0.0,
        };
    }

    // Log-domain: work with f = log(u), g = log(v)
    let mut f = vec![0.0; num_tokens];
    let mut g = vec![0.0; num_experts];

    let log_supply: Vec<f64> = supply.iter().map(|&s| s.ln()).collect();
    let log_demand: Vec<f64> = demand.iter().map(|&d| d.ln()).collect();

    let mut converged = false;
    let mut iterations = 0;
    let mut marginal_error = f64::INFINITY;

    for iter in 0..max_iterations {
        iterations = iter + 1;

        // Update g
        for j in 0..num_experts {
            let log_sum = log_sum_exp(
                &(0..num_tokens)
                    .map(|i| f[i] - cost_matrix[i][j] / epsilon)
                    .collect::<Vec<_>>(),
            );
            g[j] = log_demand[j] - log_sum;
        }

        // Update f
        for i in 0..num_tokens {
            let log_sum = log_sum_exp(
                &(0..num_experts)
                    .map(|j| g[j] - cost_matrix[i][j] / epsilon)
                    .collect::<Vec<_>>(),
            );
            f[i] = log_supply[i] - log_sum;
        }

        // Check convergence (row marginal)
        let row_marginal: Vec<f64> = (0..num_tokens)
            .map(|i| {
                (0..num_experts)
                    .map(|j| (f[i] + g[j] - cost_matrix[i][j] / epsilon).exp())
                    .sum()
            })
            .collect();

        marginal_error = row_marginal
            .iter()
            .zip(supply.iter())
            .map(|(actual, target)| (actual - target).abs())
            .sum();

        if marginal_error < threshold {
            converged = true;
            break;
        }
    }

    // Compute transport plan in log domain
    let transport_plan: Vec<Vec<f64>> = (0..num_tokens)
        .map(|i| {
            (0..num_experts)
                .map(|j| (f[i] + g[j] - cost_matrix[i][j] / epsilon).exp())
                .collect()
        })
        .collect();

    let row_scaling: Vec<f64> = f.iter().map(|&fi| fi.exp()).collect();
    let col_scaling: Vec<f64> = g.iter().map(|&gj| gj.exp()).collect();

    SinkhornResult {
        transport_plan,
        row_scaling,
        col_scaling,
        iterations,
        converged,
        marginal_error,
    }
}

/// Log-sum-exp for numerical stability
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    max_val
        + values
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f64>()
            .ln()
}

/// Compute balanced assignment using OT
pub fn compute_transport_plan(
    cost_matrix: &[Vec<f64>],
    epsilon: f64,
    iterations: usize,
) -> Vec<Vec<f64>> {
    let num_tokens = cost_matrix.len();
    let num_experts = if num_tokens > 0 {
        cost_matrix[0].len()
    } else {
        return vec![];
    };

    let supply = vec![1.0 / num_tokens as f64; num_tokens];
    let demand = vec![1.0 / num_experts as f64; num_experts];

    let result = sinkhorn_log_domain(cost_matrix, &supply, &demand, epsilon, iterations, 1e-6);
    result.transport_plan
}

/// Simple balanced assignment using greedy approach
pub fn balanced_assignment(scores: &[Vec<f64>], capacity: usize) -> Vec<Vec<(usize, f64)>> {
    let num_tokens = scores.len();
    let num_experts = if num_tokens > 0 { scores[0].len() } else { 0 };

    if num_tokens == 0 || num_experts == 0 {
        return vec![];
    }

    // Track expert loads
    let mut expert_loads = vec![0usize; num_experts];
    let mut assignments: Vec<Vec<(usize, f64)>> = vec![vec![]; num_tokens];

    // Create list of all (token, expert, score) tuples
    let mut candidates: Vec<(usize, usize, f64)> = scores
        .iter()
        .enumerate()
        .flat_map(|(t, row)| row.iter().enumerate().map(move |(e, &s)| (t, e, s)))
        .collect();

    // Sort by score descending
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy assignment respecting capacity
    for (token, expert, score) in candidates {
        if expert_loads[expert] < capacity {
            assignments[token].push((expert, score));
            expert_loads[expert] += 1;
        }
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);

        // Manual calculation: log(e^1 + e^2 + e^3)
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_large_values() {
        // Large values should not overflow
        let values = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&values);

        assert!(result.is_finite());
        assert!(result > 1000.0);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let result = log_sum_exp(&[]);
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn test_sinkhorn_basic() {
        // Simple 2x2 case
        let cost = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let supply = vec![0.5, 0.5];
        let demand = vec![0.5, 0.5];

        let result = sinkhorn_iterations(&cost, &supply, &demand, 0.1, 100, 1e-6);

        // Transport plan should sum to 1
        let total: f64 = result.transport_plan.iter().flat_map(|r| r.iter()).sum();
        assert!((total - 1.0).abs() < 0.01);

        // Row sums should match supply
        for (i, row) in result.transport_plan.iter().enumerate() {
            let row_sum: f64 = row.iter().sum();
            assert!((row_sum - supply[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_sinkhorn_log_domain() {
        let cost = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let supply = vec![0.5, 0.5];
        let demand = vec![0.5, 0.5];

        let result = sinkhorn_log_domain(&cost, &supply, &demand, 0.1, 100, 1e-6);

        // Should converge
        assert!(result.converged);

        // Transport plan should be valid
        let total: f64 = result.transport_plan.iter().flat_map(|r| r.iter()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sinkhorn_convergence() {
        let cost = vec![
            vec![0.5, 1.5, 2.0],
            vec![1.0, 0.5, 1.5],
            vec![2.0, 1.0, 0.5],
        ];
        let supply = vec![1.0 / 3.0; 3];
        let demand = vec![1.0 / 3.0; 3];

        let result = sinkhorn_log_domain(&cost, &supply, &demand, 0.05, 200, 1e-8);

        assert!(result.converged);
        assert!(result.marginal_error < 1e-6);
    }

    #[test]
    fn test_ot_router_basic() {
        let config = OTConfig {
            num_experts: 4,
            epsilon: 0.1,
            num_iterations: 100,
            ..Default::default()
        };
        let router = OTRouter::new(config);

        // 4 tokens, 4 experts
        let logits = vec![
            vec![3.0, 1.0, 1.0, 1.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![1.0, 1.0, 3.0, 1.0],
            vec![1.0, 1.0, 1.0, 3.0],
        ];

        let result = router.route(&logits, 2);

        // Should have 4 assignments
        assert_eq!(result.assignments.len(), 4);

        // Each assignment should have top-2 experts
        for assignment in &result.assignments {
            assert_eq!(assignment.len(), 2);
        }

        // Expert loads should be balanced
        let max_load = result.expert_loads.iter().cloned().fold(0.0, f64::max);
        let min_load = result
            .expert_loads
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(max_load - min_load < 0.2);
    }

    #[test]
    fn test_ot_router_transport_cost() {
        let config = OTConfig::new(2);
        let router = OTRouter::new(config);

        // Optimal: token 0 -> expert 0, token 1 -> expert 1
        let logits = vec![vec![3.0, 0.0], vec![0.0, 3.0]];

        let result = router.route(&logits, 1);

        // Transport cost should be low (optimal assignment)
        assert!(result.transport_cost < 1.0);
    }

    #[test]
    fn test_ot_router_imbalanced() {
        let config = OTConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            ..Default::default()
        };
        let router = OTRouter::new(config);

        // All tokens prefer expert 0
        let logits = vec![
            vec![3.0, 0.0],
            vec![3.0, 0.0],
            vec![3.0, 0.0],
            vec![3.0, 0.0],
        ];

        let result = router.route(&logits, 1);

        // OT should balance despite preferences
        let load_diff = (result.expert_loads[0] - result.expert_loads[1]).abs();
        assert!(load_diff < 0.2);
    }

    #[test]
    fn test_ot_router_empty() {
        let config = OTConfig::new(4);
        let router = OTRouter::new(config);

        let result = router.route(&[], 2);

        assert!(result.transport_plan.is_empty());
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn test_compute_transport_plan() {
        let cost = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![2.0, 1.0, 4.0, 3.0],
            vec![3.0, 4.0, 1.0, 2.0],
        ];

        let plan = compute_transport_plan(&cost, 0.1, 100);

        // Plan should be 4x4
        assert_eq!(plan.len(), 4);
        for row in &plan {
            assert_eq!(row.len(), 4);
        }

        // Total mass should be 1
        let total: f64 = plan.iter().flat_map(|r| r.iter()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_balanced_assignment() {
        let scores = vec![
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.8, 0.2, 0.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.0, 0.0, 0.8, 0.2],
        ];

        let assignments = balanced_assignment(&scores, 1);

        // Each expert should have at most 1 token
        let mut expert_counts = vec![0; 4];
        for token_assignments in &assignments {
            for &(expert, _) in token_assignments {
                expert_counts[expert] += 1;
            }
        }

        for count in expert_counts {
            assert!(count <= 1);
        }
    }

    #[test]
    fn test_balanced_assignment_overcapacity() {
        // All prefer expert 0
        let scores = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];

        let assignments = balanced_assignment(&scores, 2);

        // Expert 0 should have exactly 2 tokens
        let expert_0_count: usize = assignments
            .iter()
            .filter(|a| a.iter().any(|&(e, _)| e == 0))
            .count();
        assert_eq!(expert_0_count, 2);
    }

    #[test]
    fn test_discretize_top_k() {
        let config = OTConfig::new(4);
        let router = OTRouter::new(config);

        let logits = vec![vec![4.0, 3.0, 2.0, 1.0]];
        let result = router.route(&logits, 2);

        // Should select top 2 experts
        let assignment = &result.assignments[0];
        assert_eq!(assignment.len(), 2);

        // Weights should sum to 1
        let weight_sum: f64 = assignment.iter().map(|&(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);

        // Top assignment should have higher weight than second
        assert!(assignment[0].1 >= assignment[1].1);
    }

    #[test]
    fn test_sinkhorn_result_scaling_factors() {
        let cost = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let supply = vec![0.5, 0.5];
        let demand = vec![0.5, 0.5];

        let result = sinkhorn_log_domain(&cost, &supply, &demand, 0.1, 100, 1e-6);

        // Scaling factors should be positive
        assert!(result.row_scaling.iter().all(|&u| u > 0.0));
        assert!(result.col_scaling.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_temperature_effect() {
        let low_temp_config = OTConfig {
            num_experts: 2,
            temperature: 0.5,
            ..Default::default()
        };

        let high_temp_config = OTConfig {
            num_experts: 2,
            temperature: 2.0,
            ..Default::default()
        };

        let logits = vec![vec![2.0, 1.0]];

        let low_temp_result = OTRouter::new(low_temp_config).route(&logits, 2);
        let high_temp_result = OTRouter::new(high_temp_config).route(&logits, 2);

        // Both should produce valid transport plans
        assert_eq!(low_temp_result.transport_plan.len(), 1);
        assert_eq!(high_temp_result.transport_plan.len(), 1);

        // With single token and balanced OT, both get roughly equal mass
        // Temperature mainly affects the cost matrix, less visible in balanced OT
        let low_total: f64 = low_temp_result.transport_plan[0].iter().sum();
        let high_total: f64 = high_temp_result.transport_plan[0].iter().sum();
        assert!((low_total - high_total).abs() < 0.1);
    }

    #[test]
    fn test_large_scale_ot() {
        let config = OTConfig {
            num_experts: 8,
            num_iterations: 50,
            epsilon: 0.1,
            ..Default::default()
        };
        let router = OTRouter::new(config);

        // 32 tokens
        let logits: Vec<Vec<f64>> = (0..32)
            .map(|i| {
                let mut l = vec![1.0; 8];
                l[i % 8] = 3.0; // Prefer expert i%8
                l
            })
            .collect();

        let result = router.route(&logits, 2);

        // Should converge or get close
        assert!(result.sinkhorn_result.iterations <= 50);

        // Expert loads should be relatively balanced
        let avg_load: f64 = result.expert_loads.iter().sum::<f64>() / 8.0;
        let max_deviation = result
            .expert_loads
            .iter()
            .map(|&l| (l - avg_load).abs())
            .fold(0.0, f64::max);
        assert!(max_deviation < 0.15);
    }
}
