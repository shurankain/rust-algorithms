// Direct Preference Optimization (DPO)
// RLHF without explicit reward model - optimizes directly from preferences
//
// DPO reformulates RLHF as a classification problem on human preferences.
// Instead of: 1) Train reward model, 2) Optimize policy with RL
// DPO directly optimizes: π_θ(y_w|x) / π_θ(y_l|x) using preference data
//
// Key formula:
// L_DPO(π_θ; π_ref) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
//
// References:
// - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
//   (Rafailov et al., NeurIPS 2023)
// - https://arxiv.org/abs/2305.18290

use crate::policy_gradient::SoftmaxPolicy;

/// A preference pair: (prompt, chosen_response, rejected_response)
/// In our simplified setting: (state, preferred_action, rejected_action)
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// The input state/prompt
    pub state: Vec<f64>,
    /// The preferred/chosen action
    pub chosen_action: usize,
    /// The rejected action
    pub rejected_action: usize,
    /// Optional: margin for preference strength (default 1.0)
    pub margin: f64,
}

impl PreferencePair {
    pub fn new(state: Vec<f64>, chosen_action: usize, rejected_action: usize) -> Self {
        Self {
            state,
            chosen_action,
            rejected_action,
            margin: 1.0,
        }
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }
}

/// Configuration for DPO training
#[derive(Debug, Clone)]
pub struct DPOConfig {
    /// Beta parameter controlling deviation from reference policy
    /// Higher beta = stay closer to reference, more conservative
    /// Lower beta = allow more deviation, more aggressive optimization
    pub beta: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Whether to use label smoothing (helps with overconfident preferences)
    pub label_smoothing: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f64>,
    /// Number of training epochs per batch
    pub num_epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            learning_rate: 1e-4,
            label_smoothing: 0.0,
            max_grad_norm: Some(1.0),
            num_epochs: 1,
            batch_size: 32,
        }
    }
}

impl DPOConfig {
    /// Conservative config (stays close to reference)
    pub fn conservative() -> Self {
        Self {
            beta: 0.5,
            learning_rate: 5e-5,
            label_smoothing: 0.1,
            max_grad_norm: Some(0.5),
            num_epochs: 1,
            batch_size: 32,
        }
    }

    /// Aggressive config (allows more deviation)
    pub fn aggressive() -> Self {
        Self {
            beta: 0.01,
            learning_rate: 1e-3,
            label_smoothing: 0.0,
            max_grad_norm: Some(2.0),
            num_epochs: 3,
            batch_size: 64,
        }
    }
}

/// DPO trainer
#[derive(Debug)]
pub struct DPO {
    /// The policy being trained
    pub policy: SoftmaxPolicy,
    /// Reference policy (frozen copy of initial policy)
    pub reference_policy: SoftmaxPolicy,
    /// Configuration
    pub config: DPOConfig,
}

impl DPO {
    pub fn new(policy: SoftmaxPolicy, config: DPOConfig) -> Self {
        let reference_policy = policy.clone();
        Self {
            policy,
            reference_policy,
            config,
        }
    }

    pub fn with_default_config(state_dim: usize, num_actions: usize) -> Self {
        let policy = SoftmaxPolicy::new(state_dim, num_actions);
        Self::new(policy, DPOConfig::default())
    }

    /// Compute log probability ratio: log π_θ(a|s) - log π_ref(a|s)
    fn log_ratio(&self, state: &[f64], action: usize) -> f64 {
        let policy_log_prob = self.policy.log_prob(state, action);
        let ref_log_prob = self.reference_policy.log_prob(state, action);
        policy_log_prob - ref_log_prob
    }

    /// Compute DPO loss for a single preference pair
    /// L = -log σ(β * (log_ratio(chosen) - log_ratio(rejected)))
    pub fn compute_loss(&self, pair: &PreferencePair) -> f64 {
        let chosen_log_ratio = self.log_ratio(&pair.state, pair.chosen_action);
        let rejected_log_ratio = self.log_ratio(&pair.state, pair.rejected_action);

        let logit = self.config.beta * (chosen_log_ratio - rejected_log_ratio);

        // Apply label smoothing if configured
        let target = if self.config.label_smoothing > 0.0 {
            1.0 - self.config.label_smoothing
        } else {
            1.0
        };

        // Binary cross-entropy: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
        // For y=1: -log(σ(x)) = log(1 + exp(-x))
        // For smoothed y: -y*log(σ(x)) - (1-y)*log(1-σ(x))
        if self.config.label_smoothing > 0.0 {
            let sigmoid = sigmoid_stable(logit);
            -target * sigmoid.ln() - (1.0 - target) * (1.0 - sigmoid).ln()
        } else {
            // Numerically stable: log(1 + exp(-x)) for positive x
            // log(1 + exp(x)) - x for negative x
            if logit >= 0.0 {
                (1.0 + (-logit).exp()).ln()
            } else {
                -logit + (1.0 + logit.exp()).ln()
            }
        }
    }

    /// Compute gradient of DPO loss w.r.t. policy weights
    fn compute_gradient(&self, pair: &PreferencePair) -> Vec<Vec<f64>> {
        let chosen_log_ratio = self.log_ratio(&pair.state, pair.chosen_action);
        let rejected_log_ratio = self.log_ratio(&pair.state, pair.rejected_action);

        let logit = self.config.beta * (chosen_log_ratio - rejected_log_ratio);
        let sigmoid_val = sigmoid_stable(logit);

        // Gradient: -β * (1 - σ(logit)) * (∇log π_θ(chosen) - ∇log π_θ(rejected))
        let scale = -self.config.beta * (1.0 - sigmoid_val);

        let chosen_grad = self
            .policy
            .log_prob_gradient(&pair.state, pair.chosen_action);
        let rejected_grad = self
            .policy
            .log_prob_gradient(&pair.state, pair.rejected_action);

        let state_dim = self.policy.state_dim;
        let num_actions = self.policy.num_actions;

        let mut gradient = vec![vec![0.0; num_actions]; state_dim];
        for i in 0..state_dim {
            for a in 0..num_actions {
                gradient[i][a] = scale * (chosen_grad[i][a] - rejected_grad[i][a]);
            }
        }

        gradient
    }

    /// Update policy on a batch of preference pairs
    pub fn update(&mut self, preferences: &[PreferencePair]) -> DPOStats {
        if preferences.is_empty() {
            return DPOStats::default();
        }

        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut total_chosen_reward = 0.0;
        let mut total_rejected_reward = 0.0;

        for _epoch in 0..self.config.num_epochs {
            let num_batches = preferences.len().div_ceil(self.config.batch_size);

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(preferences.len());
                let batch = &preferences[start..end];

                let mut batch_gradient =
                    vec![vec![0.0; self.policy.num_actions]; self.policy.state_dim];

                for pair in batch {
                    // Compute loss
                    let loss = self.compute_loss(pair);
                    total_loss += loss;

                    // Compute implicit rewards
                    let chosen_reward = self.implicit_reward(&pair.state, pair.chosen_action);
                    let rejected_reward = self.implicit_reward(&pair.state, pair.rejected_action);
                    total_chosen_reward += chosen_reward;
                    total_rejected_reward += rejected_reward;

                    // Check accuracy (did we predict correctly?)
                    if chosen_reward > rejected_reward {
                        total_accuracy += 1.0;
                    }

                    // Accumulate gradient
                    let grad = self.compute_gradient(pair);
                    for i in 0..self.policy.state_dim {
                        for a in 0..self.policy.num_actions {
                            batch_gradient[i][a] += grad[i][a];
                        }
                    }
                }

                // Normalize gradient by batch size
                let batch_len = batch.len() as f64;
                for row in &mut batch_gradient {
                    for g in row.iter_mut() {
                        *g /= batch_len;
                    }
                }

                // Gradient clipping
                let grad_norm: f64 = batch_gradient
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|&g| g * g)
                    .sum::<f64>()
                    .sqrt();

                if let Some(max_norm) = self.config.max_grad_norm
                    && grad_norm > max_norm
                {
                    let clip_ratio = max_norm / grad_norm;
                    for row in &mut batch_gradient {
                        for g in row.iter_mut() {
                            *g *= clip_ratio;
                        }
                    }
                }

                // Update (note: gradient is already negative for loss minimization)
                // But we computed -gradient, so we add
                self.policy
                    .update_weights(&batch_gradient, -self.config.learning_rate);
            }
        }

        let n = preferences.len() as f64;
        let epochs = self.config.num_epochs as f64;

        DPOStats {
            loss: total_loss / (n * epochs),
            accuracy: total_accuracy / (n * epochs),
            chosen_reward: total_chosen_reward / (n * epochs),
            rejected_reward: total_rejected_reward / (n * epochs),
            reward_margin: (total_chosen_reward - total_rejected_reward) / (n * epochs),
        }
    }

    /// Compute implicit reward: r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
    pub fn implicit_reward(&self, state: &[f64], action: usize) -> f64 {
        self.config.beta * self.log_ratio(state, action)
    }

    /// Get the reward margin between chosen and rejected
    pub fn reward_margin(&self, pair: &PreferencePair) -> f64 {
        let chosen_reward = self.implicit_reward(&pair.state, pair.chosen_action);
        let rejected_reward = self.implicit_reward(&pair.state, pair.rejected_action);
        chosen_reward - rejected_reward
    }

    /// Predict which action would be preferred
    pub fn predict_preference(&self, state: &[f64], action_a: usize, action_b: usize) -> usize {
        let reward_a = self.implicit_reward(state, action_a);
        let reward_b = self.implicit_reward(state, action_b);
        if reward_a >= reward_b {
            action_a
        } else {
            action_b
        }
    }

    /// Get probability that action_a is preferred over action_b
    pub fn preference_probability(&self, state: &[f64], chosen: usize, rejected: usize) -> f64 {
        let chosen_log_ratio = self.log_ratio(state, chosen);
        let rejected_log_ratio = self.log_ratio(state, rejected);
        let logit = self.config.beta * (chosen_log_ratio - rejected_log_ratio);
        sigmoid_stable(logit)
    }

    /// Reset reference policy to current policy (for iterative DPO)
    pub fn update_reference(&mut self) {
        self.reference_policy = self.policy.clone();
    }

    /// Compute KL divergence from reference policy
    pub fn kl_from_reference(&self, state: &[f64]) -> f64 {
        let policy_probs = self.policy.action_probs(state);
        let ref_probs = self.reference_policy.action_probs(state);

        policy_probs
            .iter()
            .zip(ref_probs.iter())
            .filter(|&(&p, &r)| p > 1e-10 && r > 1e-10)
            .map(|(&p, &r)| p * (p / r).ln())
            .sum()
    }
}

/// Statistics from DPO training
#[derive(Debug, Clone, Default)]
pub struct DPOStats {
    /// Average DPO loss
    pub loss: f64,
    /// Accuracy on preference pairs (did chosen have higher reward?)
    pub accuracy: f64,
    /// Average implicit reward for chosen actions
    pub chosen_reward: f64,
    /// Average implicit reward for rejected actions
    pub rejected_reward: f64,
    /// Average reward margin (chosen - rejected)
    pub reward_margin: f64,
}

/// Identity Preference Optimization (IPO) - a variant of DPO
/// Uses a different loss that doesn't saturate as quickly
#[derive(Debug)]
pub struct IPO {
    pub policy: SoftmaxPolicy,
    pub reference_policy: SoftmaxPolicy,
    /// Tau parameter (similar role to beta in DPO)
    pub tau: f64,
    pub learning_rate: f64,
}

impl IPO {
    pub fn new(policy: SoftmaxPolicy, tau: f64, learning_rate: f64) -> Self {
        let reference_policy = policy.clone();
        Self {
            policy,
            reference_policy,
            tau,
            learning_rate,
        }
    }

    /// IPO loss: (log_ratio_diff - 1/(2τ))²
    pub fn compute_loss(&self, pair: &PreferencePair) -> f64 {
        let chosen_log_ratio = self.log_ratio(&pair.state, pair.chosen_action);
        let rejected_log_ratio = self.log_ratio(&pair.state, pair.rejected_action);
        let diff = chosen_log_ratio - rejected_log_ratio;
        let target = 1.0 / (2.0 * self.tau);
        (diff - target).powi(2)
    }

    fn log_ratio(&self, state: &[f64], action: usize) -> f64 {
        self.policy.log_prob(state, action) - self.reference_policy.log_prob(state, action)
    }
}

/// Kahneman-Tversky Optimization (KTO) - DPO variant for unpaired preferences
/// Works with just "good" and "bad" examples, doesn't need paired comparisons
#[derive(Debug)]
pub struct KTO {
    pub policy: SoftmaxPolicy,
    pub reference_policy: SoftmaxPolicy,
    pub beta: f64,
    pub learning_rate: f64,
    /// Desirability weight for positive examples
    pub lambda_d: f64,
    /// Undesirability weight for negative examples
    pub lambda_u: f64,
}

impl KTO {
    pub fn new(policy: SoftmaxPolicy, beta: f64, learning_rate: f64) -> Self {
        let reference_policy = policy.clone();
        Self {
            policy,
            reference_policy,
            beta,
            learning_rate,
            lambda_d: 1.0,
            lambda_u: 1.0,
        }
    }

    /// KTO uses separate losses for desirable and undesirable outputs
    pub fn compute_loss_desirable(&self, state: &[f64], action: usize, kl_estimate: f64) -> f64 {
        let log_ratio = self.log_ratio(state, action);
        let v = self.beta * log_ratio - kl_estimate;
        self.lambda_d * (1.0 - sigmoid_stable(v))
    }

    pub fn compute_loss_undesirable(&self, state: &[f64], action: usize, kl_estimate: f64) -> f64 {
        let log_ratio = self.log_ratio(state, action);
        let v = kl_estimate - self.beta * log_ratio;
        self.lambda_u * (1.0 - sigmoid_stable(v))
    }

    fn log_ratio(&self, state: &[f64], action: usize) -> f64 {
        self.policy.log_prob(state, action) - self.reference_policy.log_prob(state, action)
    }
}

/// Sigmoid function with numerical stability
pub fn sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Log sigmoid with numerical stability
pub fn log_sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Generate synthetic preference pairs from a reward function
pub fn generate_preferences_from_reward<F>(
    states: &[Vec<f64>],
    num_actions: usize,
    reward_fn: F,
    pairs_per_state: usize,
    rng_seed: u64,
) -> Vec<PreferencePair>
where
    F: Fn(&[f64], usize) -> f64,
{
    let mut preferences = Vec::new();
    let mut rng_state = rng_seed;

    for state in states {
        for _ in 0..pairs_per_state {
            // Simple LCG for deterministic random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let action_a = (rng_state as usize) % num_actions;

            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut action_b = (rng_state as usize) % num_actions;
            if action_b == action_a {
                action_b = (action_b + 1) % num_actions;
            }

            let reward_a = reward_fn(state, action_a);
            let reward_b = reward_fn(state, action_b);

            let (chosen, rejected) = if reward_a >= reward_b {
                (action_a, action_b)
            } else {
                (action_b, action_a)
            };

            let margin = (reward_a - reward_b).abs();
            preferences
                .push(PreferencePair::new(state.clone(), chosen, rejected).with_margin(margin));
        }
    }

    preferences
}

/// Analyze DPO training progress
pub fn analyze_dpo_training(stats_history: &[DPOStats]) -> DPOAnalysis {
    if stats_history.is_empty() {
        return DPOAnalysis::default();
    }

    let n = stats_history.len() as f64;

    let mean_loss = stats_history.iter().map(|s| s.loss).sum::<f64>() / n;
    let mean_accuracy = stats_history.iter().map(|s| s.accuracy).sum::<f64>() / n;
    let mean_margin = stats_history.iter().map(|s| s.reward_margin).sum::<f64>() / n;

    // Check for improving trend
    let improving = if stats_history.len() >= 2 {
        let first_half_loss: f64 = stats_history[..stats_history.len() / 2]
            .iter()
            .map(|s| s.loss)
            .sum::<f64>()
            / (stats_history.len() / 2) as f64;
        let second_half_loss: f64 = stats_history[stats_history.len() / 2..]
            .iter()
            .map(|s| s.loss)
            .sum::<f64>()
            / (stats_history.len() - stats_history.len() / 2) as f64;
        second_half_loss < first_half_loss
    } else {
        true
    };

    DPOAnalysis {
        mean_loss,
        mean_accuracy,
        mean_reward_margin: mean_margin,
        is_improving: improving,
        final_accuracy: stats_history.last().map_or(0.0, |s| s.accuracy),
    }
}

/// Analysis of DPO training
#[derive(Debug, Clone, Default)]
pub struct DPOAnalysis {
    pub mean_loss: f64,
    pub mean_accuracy: f64,
    pub mean_reward_margin: f64,
    pub is_improving: bool,
    pub final_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_preferences() -> Vec<PreferencePair> {
        vec![
            PreferencePair::new(vec![1.0, 0.0], 0, 1),
            PreferencePair::new(vec![0.0, 1.0], 1, 0),
            PreferencePair::new(vec![0.5, 0.5], 0, 1),
        ]
    }

    #[test]
    fn test_preference_pair_creation() {
        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1);
        assert_eq!(pair.chosen_action, 0);
        assert_eq!(pair.rejected_action, 1);
        assert!((pair.margin - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_preference_pair_with_margin() {
        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1).with_margin(2.5);
        assert!((pair.margin - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_dpo_config_default() {
        let config = DPOConfig::default();
        assert!(config.beta > 0.0);
        assert!(config.learning_rate > 0.0);
        assert!(config.label_smoothing >= 0.0);
    }

    #[test]
    fn test_dpo_config_variants() {
        let conservative = DPOConfig::conservative();
        let aggressive = DPOConfig::aggressive();

        // Conservative has higher beta (stays closer to reference)
        assert!(conservative.beta > aggressive.beta);
    }

    #[test]
    fn test_dpo_creation() {
        let dpo = DPO::with_default_config(4, 3);
        assert_eq!(dpo.policy.state_dim, 4);
        assert_eq!(dpo.policy.num_actions, 3);
    }

    #[test]
    fn test_dpo_log_ratio_initial() {
        let dpo = DPO::with_default_config(2, 2);

        // Initially, policy == reference, so log ratio should be 0
        let ratio = dpo.log_ratio(&[1.0, 0.0], 0);
        assert!(ratio.abs() < 1e-10);
    }

    #[test]
    fn test_dpo_compute_loss() {
        let dpo = DPO::with_default_config(2, 2);
        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1);

        let loss = dpo.compute_loss(&pair);

        // Loss should be finite and non-negative (it's a negative log likelihood)
        assert!(loss.is_finite());
        // Initial loss should be log(2) ≈ 0.693 (when logit = 0, sigmoid = 0.5)
        assert!((loss - 2.0_f64.ln()).abs() < 0.1);
    }

    #[test]
    fn test_dpo_update() {
        let mut dpo = DPO::with_default_config(2, 2);
        let preferences = create_simple_preferences();

        let initial_loss = dpo.compute_loss(&preferences[0]);
        let stats = dpo.update(&preferences);

        // Should have computed stats
        assert!(stats.loss.is_finite());
        assert!(stats.accuracy >= 0.0 && stats.accuracy <= 1.0);

        // After update, loss might change
        let final_loss = dpo.compute_loss(&preferences[0]);
        assert!(final_loss.is_finite());

        // Accuracy should be reported
        assert!(stats.accuracy >= 0.0);
    }

    #[test]
    fn test_dpo_implicit_reward() {
        let dpo = DPO::with_default_config(2, 2);
        let state = vec![1.0, 0.0];

        let reward = dpo.implicit_reward(&state, 0);
        // Initially should be 0 (policy == reference)
        assert!(reward.abs() < 1e-10);
    }

    #[test]
    fn test_dpo_reward_margin() {
        let dpo = DPO::with_default_config(2, 2);
        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1);

        let margin = dpo.reward_margin(&pair);
        // Initially should be 0
        assert!(margin.abs() < 1e-10);
    }

    #[test]
    fn test_dpo_predict_preference() {
        let mut dpo = DPO::with_default_config(2, 2);
        let preferences = vec![
            PreferencePair::new(vec![1.0, 0.0], 0, 1),
            PreferencePair::new(vec![1.0, 0.0], 0, 1),
            PreferencePair::new(vec![1.0, 0.0], 0, 1),
        ];

        // Train to prefer action 0 for state [1.0, 0.0]
        for _ in 0..10 {
            dpo.update(&preferences);
        }

        // Should predict action 0 as preferred
        let predicted = dpo.predict_preference(&[1.0, 0.0], 0, 1);
        assert_eq!(predicted, 0);
    }

    #[test]
    fn test_dpo_preference_probability() {
        let dpo = DPO::with_default_config(2, 2);
        let state = vec![1.0, 0.0];

        let prob = dpo.preference_probability(&state, 0, 1);
        // Initially should be 0.5 (equal)
        assert!((prob - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_dpo_update_reference() {
        let mut dpo = DPO::with_default_config(2, 2);
        let preferences = create_simple_preferences();

        // Train policy
        dpo.update(&preferences);

        // Update reference to current policy
        dpo.update_reference();

        // Now log ratios should be 0 again
        let ratio = dpo.log_ratio(&[1.0, 0.0], 0);
        assert!(ratio.abs() < 1e-10);
    }

    #[test]
    fn test_dpo_kl_from_reference() {
        let mut dpo = DPO::with_default_config(2, 2);

        // Initially KL should be 0
        let kl_initial = dpo.kl_from_reference(&[1.0, 0.0]);
        assert!(kl_initial.abs() < 1e-10);

        // After training, KL should be positive
        let preferences = create_simple_preferences();
        for _ in 0..5 {
            dpo.update(&preferences);
        }

        let kl_after = dpo.kl_from_reference(&[1.0, 0.0]);
        assert!(kl_after >= 0.0);
    }

    #[test]
    fn test_sigmoid_stable() {
        // Test at 0
        let s0 = sigmoid_stable(0.0);
        assert!((s0 - 0.5).abs() < 1e-10);

        // Test large positive
        let s_pos = sigmoid_stable(100.0);
        assert!((s_pos - 1.0).abs() < 1e-10);

        // Test large negative
        let s_neg = sigmoid_stable(-100.0);
        assert!(s_neg.abs() < 1e-10);

        // Test symmetry
        let s1 = sigmoid_stable(2.0);
        let s2 = sigmoid_stable(-2.0);
        assert!((s1 + s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sigmoid_stable() {
        // log(sigmoid(0)) = log(0.5) ≈ -0.693
        let ls0 = log_sigmoid_stable(0.0);
        assert!((ls0 - (-2.0_f64.ln())).abs() < 1e-10);

        // Large positive should give ~0
        let ls_pos = log_sigmoid_stable(100.0);
        assert!(ls_pos.abs() < 1e-10);

        // Should not overflow for large negative
        let ls_neg = log_sigmoid_stable(-100.0);
        assert!(ls_neg.is_finite());
    }

    #[test]
    fn test_ipo_creation() {
        let policy = SoftmaxPolicy::new(4, 3);
        let ipo = IPO::new(policy, 0.1, 1e-4);
        assert_eq!(ipo.policy.state_dim, 4);
        assert!((ipo.tau - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_ipo_loss() {
        let policy = SoftmaxPolicy::new(2, 2);
        let ipo = IPO::new(policy, 0.5, 1e-4);
        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1);

        let loss = ipo.compute_loss(&pair);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_kto_creation() {
        let policy = SoftmaxPolicy::new(4, 3);
        let kto = KTO::new(policy, 0.1, 1e-4);
        assert_eq!(kto.policy.state_dim, 4);
    }

    #[test]
    fn test_kto_losses() {
        let policy = SoftmaxPolicy::new(2, 2);
        let kto = KTO::new(policy, 0.1, 1e-4);

        let state = vec![1.0, 0.0];
        let kl_estimate = 0.1;

        let loss_d = kto.compute_loss_desirable(&state, 0, kl_estimate);
        let loss_u = kto.compute_loss_undesirable(&state, 1, kl_estimate);

        assert!(loss_d.is_finite());
        assert!(loss_u.is_finite());
        assert!(loss_d >= 0.0);
        assert!(loss_u >= 0.0);
    }

    #[test]
    fn test_generate_preferences_from_reward() {
        let states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        // Reward function: prefer action 0
        let reward_fn = |_state: &[f64], action: usize| if action == 0 { 1.0 } else { 0.0 };

        let preferences = generate_preferences_from_reward(&states, 2, reward_fn, 3, 42);

        assert_eq!(preferences.len(), 6); // 2 states * 3 pairs

        // All should prefer action 0
        for pref in &preferences {
            assert_eq!(pref.chosen_action, 0);
        }
    }

    #[test]
    fn test_analyze_dpo_training_empty() {
        let analysis = analyze_dpo_training(&[]);
        assert_eq!(analysis.mean_loss, 0.0);
    }

    #[test]
    fn test_analyze_dpo_training() {
        let stats = vec![
            DPOStats {
                loss: 0.7,
                accuracy: 0.5,
                chosen_reward: 0.1,
                rejected_reward: -0.1,
                reward_margin: 0.2,
            },
            DPOStats {
                loss: 0.5,
                accuracy: 0.7,
                chosen_reward: 0.2,
                rejected_reward: -0.2,
                reward_margin: 0.4,
            },
        ];

        let analysis = analyze_dpo_training(&stats);

        assert!((analysis.mean_loss - 0.6).abs() < 1e-10);
        assert!((analysis.mean_accuracy - 0.6).abs() < 1e-10);
        assert!(analysis.is_improving); // Loss decreased
        assert!((analysis.final_accuracy - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_dpo_training_improves_accuracy() {
        let mut dpo = DPO::with_default_config(2, 2);

        // Create consistent preferences
        let preferences: Vec<PreferencePair> = (0..20)
            .map(|_| PreferencePair::new(vec![1.0, 0.0], 0, 1))
            .collect();

        let mut stats_history = Vec::new();
        for _ in 0..20 {
            let stats = dpo.update(&preferences);
            stats_history.push(stats);
        }

        // Final accuracy should be high
        let final_stats = stats_history.last().unwrap();
        assert!(final_stats.accuracy >= 0.9);
    }

    #[test]
    fn test_dpo_with_label_smoothing() {
        let config = DPOConfig {
            label_smoothing: 0.1,
            ..DPOConfig::default()
        };
        let policy = SoftmaxPolicy::new(2, 2);
        let dpo = DPO::new(policy, config);

        let pair = PreferencePair::new(vec![1.0, 0.0], 0, 1);
        let loss = dpo.compute_loss(&pair);

        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_dpo_stats_default() {
        let stats = DPOStats::default();
        assert_eq!(stats.loss, 0.0);
        assert_eq!(stats.accuracy, 0.0);
    }
}
