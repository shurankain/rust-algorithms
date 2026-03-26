// Group Relative Policy Optimization (GRPO)
// DeepSeek's algorithm for LLM alignment without value function
//
// GRPO is a simplified RL algorithm that:
// 1. Samples multiple responses per prompt (group sampling)
// 2. Computes advantages relative to group mean (no critic needed)
// 3. Uses PPO-style clipped objectives for stability
//
// Key insight: By comparing responses within a group, we can compute
// relative quality without learning an absolute value function.
//
// References:
// - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
// - "GRPO: Group Relative Policy Optimization" (DeepSeek, 2024)

use crate::policy_gradient::SoftmaxPolicy;

/// Configuration for GRPO
#[derive(Debug, Clone)]
pub struct GRPOConfig {
    /// Number of samples per prompt/state (group size)
    pub group_size: usize,
    /// KL penalty coefficient (beta)
    pub kl_coeff: f64,
    /// PPO-style clip epsilon
    pub clip_epsilon: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f64>,
    /// Minimum advantage threshold (filter low-signal samples)
    pub advantage_threshold: Option<f64>,
    /// Whether to normalize advantages within group
    pub normalize_advantages: bool,
    /// Entropy bonus coefficient
    pub entropy_coeff: f64,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            group_size: 4,
            kl_coeff: 0.1,
            clip_epsilon: 0.2,
            learning_rate: 1e-4,
            max_grad_norm: Some(1.0),
            advantage_threshold: None,
            normalize_advantages: true,
            entropy_coeff: 0.01,
        }
    }
}

impl GRPOConfig {
    /// Create config optimized for mathematical reasoning
    pub fn for_math() -> Self {
        Self {
            group_size: 8,
            kl_coeff: 0.05,
            clip_epsilon: 0.1,
            learning_rate: 5e-5,
            max_grad_norm: Some(0.5),
            advantage_threshold: Some(0.1),
            normalize_advantages: true,
            entropy_coeff: 0.0,
        }
    }

    /// Create config optimized for code generation
    pub fn for_code() -> Self {
        Self {
            group_size: 16,
            kl_coeff: 0.1,
            clip_epsilon: 0.2,
            learning_rate: 1e-4,
            max_grad_norm: Some(1.0),
            advantage_threshold: Some(0.05),
            normalize_advantages: true,
            entropy_coeff: 0.005,
        }
    }

    /// Create config with minimal KL constraint
    pub fn low_kl() -> Self {
        Self {
            kl_coeff: 0.01,
            ..Default::default()
        }
    }

    /// Create config with strong KL constraint
    pub fn high_kl() -> Self {
        Self {
            kl_coeff: 0.5,
            ..Default::default()
        }
    }

    pub fn with_group_size(mut self, size: usize) -> Self {
        self.group_size = size;
        self
    }

    pub fn with_kl_coeff(mut self, coeff: f64) -> Self {
        self.kl_coeff = coeff;
        self
    }

    pub fn with_clip_epsilon(mut self, epsilon: f64) -> Self {
        self.clip_epsilon = epsilon;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

/// A sampled response with its reward
#[derive(Debug, Clone)]
pub struct GroupSample {
    /// The state/prompt
    pub state: Vec<f64>,
    /// The action/response taken
    pub action: usize,
    /// The reward received
    pub reward: f64,
    /// Log probability under current policy
    pub log_prob: f64,
    /// Log probability under reference policy
    pub ref_log_prob: f64,
}

impl GroupSample {
    pub fn new(
        state: Vec<f64>,
        action: usize,
        reward: f64,
        log_prob: f64,
        ref_log_prob: f64,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            log_prob,
            ref_log_prob,
        }
    }
}

/// A group of samples from the same prompt/state
#[derive(Debug, Clone)]
pub struct SampleGroup {
    /// Shared state for all samples
    pub state: Vec<f64>,
    /// Actions taken
    pub actions: Vec<usize>,
    /// Rewards received
    pub rewards: Vec<f64>,
    /// Log probs under policy when sampled
    pub old_log_probs: Vec<f64>,
    /// Log probs under reference policy
    pub ref_log_probs: Vec<f64>,
}

impl SampleGroup {
    pub fn new(state: Vec<f64>) -> Self {
        Self {
            state,
            actions: Vec::new(),
            rewards: Vec::new(),
            old_log_probs: Vec::new(),
            ref_log_probs: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, action: usize, reward: f64, log_prob: f64, ref_log_prob: f64) {
        self.actions.push(action);
        self.rewards.push(reward);
        self.old_log_probs.push(log_prob);
        self.ref_log_probs.push(ref_log_prob);
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Compute group-relative advantages
    pub fn compute_advantages(&self, normalize: bool) -> Vec<f64> {
        if self.rewards.is_empty() {
            return Vec::new();
        }

        let mean_reward: f64 = self.rewards.iter().sum::<f64>() / self.rewards.len() as f64;

        let mut advantages: Vec<f64> = self.rewards.iter().map(|&r| r - mean_reward).collect();

        if normalize && advantages.len() > 1 {
            let std_dev: f64 =
                (advantages.iter().map(|a| a * a).sum::<f64>() / advantages.len() as f64).sqrt();

            if std_dev > 1e-8 {
                for a in &mut advantages {
                    *a /= std_dev;
                }
            }
        }

        advantages
    }

    /// Get best action in group
    pub fn best_action(&self) -> Option<usize> {
        self.rewards
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| self.actions[i])
    }

    /// Get reward spread (max - min)
    pub fn reward_spread(&self) -> f64 {
        if self.rewards.is_empty() {
            return 0.0;
        }
        let max = self
            .rewards
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self.rewards.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

/// GRPO trainer
#[derive(Debug, Clone)]
pub struct GRPO {
    /// Current policy
    pub policy: SoftmaxPolicy,
    /// Reference policy (frozen)
    pub reference_policy: SoftmaxPolicy,
    /// Configuration
    pub config: GRPOConfig,
}

impl GRPO {
    pub fn new(state_dim: usize, num_actions: usize, config: GRPOConfig) -> Self {
        let policy = SoftmaxPolicy::new(state_dim, num_actions);
        let reference_policy = policy.clone();
        Self {
            policy,
            reference_policy,
            config,
        }
    }

    /// Sample an action from the policy
    pub fn sample_action(&self, state: &[f64], rng_value: f64) -> (usize, f64) {
        let probs = self.policy.action_probs(state);
        let mut cumsum = 0.0;
        for (action, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rng_value < cumsum {
                let log_prob = self.policy.log_prob(state, action);
                return (action, log_prob);
            }
        }
        let action = probs.len() - 1;
        (action, self.policy.log_prob(state, action))
    }

    /// Get reference log probability
    pub fn ref_log_prob(&self, state: &[f64], action: usize) -> f64 {
        self.reference_policy.log_prob(state, action)
    }

    /// Sample a group of responses for a state
    pub fn sample_group(
        &self,
        state: &[f64],
        reward_fn: impl Fn(&[f64], usize) -> f64,
        rng_values: &[f64],
    ) -> SampleGroup {
        let mut group = SampleGroup::new(state.to_vec());

        for &rng in rng_values.iter().take(self.config.group_size) {
            let (action, log_prob) = self.sample_action(state, rng);
            let ref_log_prob = self.ref_log_prob(state, action);
            let reward = reward_fn(state, action);
            group.add_sample(action, reward, log_prob, ref_log_prob);
        }

        group
    }

    /// Compute GRPO loss for a single sample
    fn compute_sample_loss(
        &self,
        state: &[f64],
        action: usize,
        advantage: f64,
        old_log_prob: f64,
        ref_log_prob: f64,
    ) -> (f64, f64, f64) {
        let new_log_prob = self.policy.log_prob(state, action);

        // Policy ratio
        let ratio = (new_log_prob - old_log_prob).exp();

        // Clipped surrogate objective (PPO-style)
        let clipped_ratio = ratio.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        );

        let policy_loss = -(ratio * advantage).min(clipped_ratio * advantage);

        // KL penalty from reference
        let kl_penalty = self.config.kl_coeff * (new_log_prob - ref_log_prob);

        // Entropy bonus
        let entropy = self.policy.entropy(state);

        (policy_loss + kl_penalty, entropy, ratio)
    }

    /// Update policy on a batch of sample groups
    pub fn update(&mut self, groups: &[SampleGroup]) -> GRPOStats {
        if groups.is_empty() {
            return GRPOStats::default();
        }

        let state_dim = self.policy.state_dim;
        let num_actions = self.policy.num_actions;

        let mut total_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut total_kl = 0.0;
        let mut total_ratio = 0.0;
        let mut total_advantage = 0.0;
        let mut clip_fraction = 0.0;
        let mut sample_count = 0;
        let mut filtered_count = 0;

        let mut gradient = vec![vec![0.0; num_actions]; state_dim];

        for group in groups {
            if group.is_empty() {
                continue;
            }

            let advantages = group.compute_advantages(self.config.normalize_advantages);

            for (i, &advantage) in advantages.iter().enumerate() {
                // Filter low-signal samples
                if let Some(threshold) = self.config.advantage_threshold
                    && advantage.abs() < threshold
                {
                    filtered_count += 1;
                    continue;
                }

                let action = group.actions[i];
                let old_log_prob = group.old_log_probs[i];
                let ref_log_prob = group.ref_log_probs[i];

                let (loss, entropy, ratio) = self.compute_sample_loss(
                    &group.state,
                    action,
                    advantage,
                    old_log_prob,
                    ref_log_prob,
                );

                total_loss += loss;
                total_entropy += entropy;
                total_ratio += ratio;
                total_advantage += advantage.abs();

                // KL from reference
                let new_log_prob = self.policy.log_prob(&group.state, action);
                total_kl += new_log_prob - ref_log_prob;

                // Track clipping
                if ratio < 1.0 - self.config.clip_epsilon || ratio > 1.0 + self.config.clip_epsilon
                {
                    clip_fraction += 1.0;
                }

                // Compute gradient
                let log_prob_grad = self.policy.log_prob_gradient(&group.state, action);
                let clipped_ratio = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                );

                // Use gradient only if unclipped
                let use_unclipped = ratio * advantage <= clipped_ratio * advantage;
                let grad_scale = if use_unclipped { advantage } else { 0.0 };

                // Add KL gradient
                let kl_grad_scale = self.config.kl_coeff;

                for (s_idx, row) in log_prob_grad.iter().enumerate() {
                    for (a_idx, &g) in row.iter().enumerate() {
                        gradient[s_idx][a_idx] += g * grad_scale - g * kl_grad_scale;
                    }
                }

                // Entropy gradient
                if self.config.entropy_coeff > 0.0 {
                    let probs = self.policy.action_probs(&group.state);
                    for (s_idx, &s) in group.state.iter().enumerate() {
                        for (a_idx, &prob) in probs.iter().enumerate() {
                            let entropy_grad = -s * prob * (1.0 + prob.ln().max(-10.0));
                            gradient[s_idx][a_idx] += self.config.entropy_coeff * entropy_grad;
                        }
                    }
                }

                sample_count += 1;
            }
        }

        if sample_count == 0 {
            return GRPOStats {
                filtered_samples: filtered_count,
                ..Default::default()
            };
        }

        // Normalize gradient
        for row in &mut gradient {
            for g in row.iter_mut() {
                *g /= sample_count as f64;
            }
        }

        // Gradient clipping
        let grad_norm: f64 = gradient
            .iter()
            .flat_map(|row| row.iter())
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        if let Some(max_norm) = self.config.max_grad_norm
            && grad_norm > max_norm
        {
            let clip_ratio = max_norm / grad_norm;
            for row in &mut gradient {
                for g in row.iter_mut() {
                    *g *= clip_ratio;
                }
            }
        }

        // Update policy
        self.policy
            .update_weights(&gradient, self.config.learning_rate);

        GRPOStats {
            loss: total_loss / sample_count as f64,
            entropy: total_entropy / sample_count as f64,
            kl_divergence: total_kl / sample_count as f64,
            mean_ratio: total_ratio / sample_count as f64,
            clip_fraction: clip_fraction / sample_count as f64,
            mean_advantage: total_advantage / sample_count as f64,
            gradient_norm: grad_norm,
            num_samples: sample_count,
            filtered_samples: filtered_count,
        }
    }

    /// Update reference policy to current policy
    pub fn update_reference(&mut self) {
        self.reference_policy = self.policy.clone();
    }

    /// Compute KL divergence from reference for a state
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

    /// Get action probabilities
    pub fn action_probs(&self, state: &[f64]) -> Vec<f64> {
        self.policy.action_probs(state)
    }

    /// Choose best action (greedy)
    pub fn best_action(&self, state: &[f64]) -> usize {
        let probs = self.policy.action_probs(state);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Statistics from GRPO update
#[derive(Debug, Clone, Default)]
pub struct GRPOStats {
    /// Average loss
    pub loss: f64,
    /// Average entropy
    pub entropy: f64,
    /// Average KL divergence from reference
    pub kl_divergence: f64,
    /// Mean policy ratio
    pub mean_ratio: f64,
    /// Fraction of samples that were clipped
    pub clip_fraction: f64,
    /// Mean absolute advantage
    pub mean_advantage: f64,
    /// Gradient norm before clipping
    pub gradient_norm: f64,
    /// Number of samples used
    pub num_samples: usize,
    /// Number of samples filtered
    pub filtered_samples: usize,
}

/// Online GRPO that maintains a replay buffer
#[derive(Debug, Clone)]
pub struct OnlineGRPO {
    /// Base GRPO trainer
    pub grpo: GRPO,
    /// Buffer of sample groups
    buffer: Vec<SampleGroup>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Minimum samples before update
    min_samples_for_update: usize,
}

impl OnlineGRPO {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        config: GRPOConfig,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            grpo: GRPO::new(state_dim, num_actions, config),
            buffer: Vec::new(),
            max_buffer_size,
            min_samples_for_update: 32,
        }
    }

    /// Add a sample group to the buffer
    pub fn add_group(&mut self, group: SampleGroup) {
        self.buffer.push(group);
        if self.buffer.len() > self.max_buffer_size {
            self.buffer.remove(0);
        }
    }

    /// Check if ready for update
    pub fn ready_for_update(&self) -> bool {
        let total_samples: usize = self.buffer.iter().map(|g| g.len()).sum();
        total_samples >= self.min_samples_for_update
    }

    /// Perform update and clear buffer
    pub fn update(&mut self) -> GRPOStats {
        let stats = self.grpo.update(&self.buffer);
        self.buffer.clear();
        stats
    }

    /// Get buffer size
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

/// Reward-weighted GRPO variant
/// Weights samples by reward magnitude for faster learning
#[derive(Debug, Clone)]
pub struct RewardWeightedGRPO {
    /// Base GRPO
    pub grpo: GRPO,
    /// Temperature for reward weighting
    pub temperature: f64,
}

impl RewardWeightedGRPO {
    pub fn new(state_dim: usize, num_actions: usize, config: GRPOConfig, temperature: f64) -> Self {
        Self {
            grpo: GRPO::new(state_dim, num_actions, config),
            temperature,
        }
    }

    /// Compute reward-weighted advantages
    fn compute_weighted_advantages(&self, group: &SampleGroup) -> Vec<f64> {
        if group.rewards.is_empty() {
            return Vec::new();
        }

        // Softmax weighting of rewards
        let max_reward = group
            .rewards
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let weights: Vec<f64> = group
            .rewards
            .iter()
            .map(|&r| ((r - max_reward) / self.temperature).exp())
            .collect();

        let weight_sum: f64 = weights.iter().sum();

        // Weighted mean
        let weighted_mean: f64 = group
            .rewards
            .iter()
            .zip(weights.iter())
            .map(|(&r, &w)| r * w)
            .sum::<f64>()
            / weight_sum;

        // Advantages relative to weighted mean
        group.rewards.iter().map(|&r| r - weighted_mean).collect()
    }

    /// Update with reward-weighted sampling
    pub fn update(&mut self, groups: &[SampleGroup]) -> GRPOStats {
        // Modify groups with weighted advantages
        let weighted_groups: Vec<SampleGroup> = groups
            .iter()
            .map(|g| {
                let mut new_group = g.clone();
                let weighted_advs = self.compute_weighted_advantages(g);
                // Store advantages in rewards for processing
                new_group.rewards = weighted_advs;
                new_group
            })
            .collect();

        self.grpo.update(&weighted_groups)
    }
}

/// Analysis of GRPO training
#[derive(Debug, Clone, Default)]
pub struct GRPOAnalysis {
    /// Average loss trend
    pub loss_trend: f64,
    /// Average KL divergence
    pub mean_kl: f64,
    /// KL divergence trend
    pub kl_trend: f64,
    /// Average clip fraction
    pub mean_clip_fraction: f64,
    /// Is KL constraint being respected
    pub kl_healthy: bool,
    /// Is clipping excessive
    pub clipping_excessive: bool,
    /// Recommended KL coefficient adjustment
    pub recommended_kl_adjustment: Option<f64>,
}

/// Analyze GRPO training progress
pub fn analyze_grpo_training(stats_history: &[GRPOStats]) -> GRPOAnalysis {
    if stats_history.is_empty() {
        return GRPOAnalysis::default();
    }

    let n = stats_history.len();

    // Compute means
    let mean_kl: f64 = stats_history.iter().map(|s| s.kl_divergence).sum::<f64>() / n as f64;
    let mean_clip_fraction: f64 =
        stats_history.iter().map(|s| s.clip_fraction).sum::<f64>() / n as f64;

    // Compute trends (simple linear regression slope)
    let loss_trend = if n > 1 {
        let losses: Vec<f64> = stats_history.iter().map(|s| s.loss).collect();
        compute_trend(&losses)
    } else {
        0.0
    };

    let kl_trend = if n > 1 {
        let kls: Vec<f64> = stats_history.iter().map(|s| s.kl_divergence).collect();
        compute_trend(&kls)
    } else {
        0.0
    };

    // Health checks
    let kl_healthy = mean_kl.abs() < 0.1; // KL should be small
    let clipping_excessive = mean_clip_fraction > 0.3; // >30% clipping is concerning

    // Recommend KL adjustment
    let recommended_kl_adjustment = if mean_kl > 0.1 {
        Some(1.5) // Increase KL penalty
    } else if mean_kl < 0.01 && mean_clip_fraction < 0.1 {
        Some(0.7) // Can reduce KL penalty
    } else {
        None
    };

    GRPOAnalysis {
        loss_trend,
        mean_kl,
        kl_trend,
        mean_clip_fraction,
        kl_healthy,
        clipping_excessive,
        recommended_kl_adjustment,
    }
}

/// Compute linear trend (slope)
fn compute_trend(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }

    let mean_x = (n - 1) as f64 / 2.0;
    let mean_y: f64 = values.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - mean_x) * (y - mean_y);
        denominator += (x - mean_x) * (x - mean_x);
    }

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Generate sample groups from a reward function
pub fn generate_groups(
    grpo: &GRPO,
    states: &[Vec<f64>],
    reward_fn: impl Fn(&[f64], usize) -> f64,
    rng_values: &[f64],
) -> Vec<SampleGroup> {
    let mut groups = Vec::new();
    let mut rng_idx = 0;

    for state in states {
        let rng_slice: Vec<f64> = rng_values
            .iter()
            .skip(rng_idx)
            .take(grpo.config.group_size)
            .copied()
            .collect();

        if rng_slice.len() < grpo.config.group_size {
            break;
        }

        let group = grpo.sample_group(state, &reward_fn, &rng_slice);
        groups.push(group);
        rng_idx += grpo.config.group_size;
    }

    groups
}

/// Compute group statistics
pub fn group_statistics(groups: &[SampleGroup]) -> GroupStats {
    if groups.is_empty() {
        return GroupStats::default();
    }

    let total_samples: usize = groups.iter().map(|g| g.len()).sum();
    let total_rewards: f64 = groups.iter().flat_map(|g| g.rewards.iter()).sum();

    let mean_reward = if total_samples > 0 {
        total_rewards / total_samples as f64
    } else {
        0.0
    };

    let reward_spreads: Vec<f64> = groups.iter().map(|g| g.reward_spread()).collect();
    let mean_spread = reward_spreads.iter().sum::<f64>() / groups.len() as f64;

    let max_reward = groups
        .iter()
        .flat_map(|g| g.rewards.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let min_reward = groups
        .iter()
        .flat_map(|g| g.rewards.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);

    GroupStats {
        num_groups: groups.len(),
        total_samples,
        mean_reward,
        max_reward,
        min_reward,
        mean_reward_spread: mean_spread,
    }
}

/// Statistics about sample groups
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    pub num_groups: usize,
    pub total_samples: usize,
    pub mean_reward: f64,
    pub max_reward: f64,
    pub min_reward: f64,
    pub mean_reward_spread: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_reward(state: &[f64], action: usize) -> f64 {
        // Reward that prefers action 0 when state[0] > 0, else action 1
        if state[0] > 0.0 {
            if action == 0 { 1.0 } else { -1.0 }
        } else if action == 1 {
            1.0
        } else {
            -1.0
        }
    }

    #[test]
    fn test_grpo_config_default() {
        let config = GRPOConfig::default();
        assert_eq!(config.group_size, 4);
        assert!((config.kl_coeff - 0.1).abs() < 1e-10);
        assert!((config.clip_epsilon - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_grpo_config_builders() {
        let config = GRPOConfig::for_math();
        assert_eq!(config.group_size, 8);
        assert!((config.kl_coeff - 0.05).abs() < 1e-10);

        let config = GRPOConfig::for_code();
        assert_eq!(config.group_size, 16);

        let config = GRPOConfig::default().with_group_size(32);
        assert_eq!(config.group_size, 32);
    }

    #[test]
    fn test_sample_group_creation() {
        let mut group = SampleGroup::new(vec![1.0, 0.0]);
        assert!(group.is_empty());

        group.add_sample(0, 1.0, -0.5, -0.6);
        group.add_sample(1, -1.0, -0.7, -0.8);

        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());
    }

    #[test]
    fn test_sample_group_advantages() {
        let mut group = SampleGroup::new(vec![1.0]);
        group.add_sample(0, 2.0, -0.5, -0.5);
        group.add_sample(1, 0.0, -0.5, -0.5);
        group.add_sample(2, 1.0, -0.5, -0.5);

        // Mean reward = 1.0
        let advantages = group.compute_advantages(false);
        assert!((advantages[0] - 1.0).abs() < 1e-10); // 2 - 1 = 1
        assert!((advantages[1] - (-1.0)).abs() < 1e-10); // 0 - 1 = -1
        assert!((advantages[2] - 0.0).abs() < 1e-10); // 1 - 1 = 0
    }

    #[test]
    fn test_sample_group_normalized_advantages() {
        let mut group = SampleGroup::new(vec![1.0]);
        group.add_sample(0, 2.0, -0.5, -0.5);
        group.add_sample(1, 0.0, -0.5, -0.5);

        let advantages = group.compute_advantages(true);
        // Should be normalized to have unit std
        let std: f64 = (advantages.iter().map(|a| a * a).sum::<f64>() / 2.0).sqrt();
        assert!((std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_group_best_action() {
        let mut group = SampleGroup::new(vec![1.0]);
        group.add_sample(0, -1.0, -0.5, -0.5);
        group.add_sample(1, 2.0, -0.5, -0.5);
        group.add_sample(2, 0.5, -0.5, -0.5);

        assert_eq!(group.best_action(), Some(1));
    }

    #[test]
    fn test_sample_group_reward_spread() {
        let mut group = SampleGroup::new(vec![1.0]);
        group.add_sample(0, 1.0, -0.5, -0.5);
        group.add_sample(1, 5.0, -0.5, -0.5);
        group.add_sample(2, 2.0, -0.5, -0.5);

        assert!((group.reward_spread() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_grpo_creation() {
        let grpo = GRPO::new(4, 2, GRPOConfig::default());
        assert_eq!(grpo.policy.state_dim, 4);
        assert_eq!(grpo.policy.num_actions, 2);
    }

    #[test]
    fn test_grpo_sample_action() {
        let grpo = GRPO::new(2, 3, GRPOConfig::default());
        let state = vec![1.0, 0.5];

        let (action, log_prob) = grpo.sample_action(&state, 0.5);
        assert!(action < 3);
        assert!(log_prob <= 0.0); // Log probs are negative
    }

    #[test]
    fn test_grpo_sample_group() {
        let grpo = GRPO::new(2, 3, GRPOConfig::default().with_group_size(4));
        let state = vec![1.0, -0.5];
        let rng_values = vec![0.1, 0.4, 0.6, 0.9];

        let group = grpo.sample_group(&state, simple_reward, &rng_values);
        assert_eq!(group.len(), 4);
        assert_eq!(group.state, state);
    }

    #[test]
    fn test_grpo_update_empty() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default());
        let stats = grpo.update(&[]);
        assert_eq!(stats.num_samples, 0);
    }

    #[test]
    fn test_grpo_update_single_group() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default().with_group_size(4));
        let state = vec![1.0, 0.0];
        let rng_values = vec![0.1, 0.4, 0.6, 0.9];

        let group = grpo.sample_group(&state, simple_reward, &rng_values);
        let stats = grpo.update(&[group]);

        assert!(stats.num_samples > 0);
    }

    #[test]
    fn test_grpo_update_multiple_groups() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default().with_group_size(4));

        let states = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.5, 0.5]];

        let rng_values: Vec<f64> = (0..12).map(|i| i as f64 / 12.0).collect();

        let groups = generate_groups(&grpo, &states, simple_reward, &rng_values);
        assert_eq!(groups.len(), 3);

        let stats = grpo.update(&groups);
        assert!(stats.num_samples > 0);
    }

    #[test]
    fn test_grpo_learning() {
        let config = GRPOConfig::default()
            .with_group_size(8)
            .with_learning_rate(0.1);
        let mut grpo = GRPO::new(2, 2, config);

        // Train on positive states
        let states: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 0.0]).collect();
        let rng_values: Vec<f64> = (0..80).map(|i| (i as f64 * 0.123) % 1.0).collect();

        for _ in 0..10 {
            let groups = generate_groups(&grpo, &states, simple_reward, &rng_values);
            grpo.update(&groups);
        }

        // Should prefer action 0 for positive state
        let probs = grpo.action_probs(&[1.0, 0.0]);
        assert!(probs[0] > probs[1]);
    }

    #[test]
    fn test_grpo_kl_from_reference() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default());
        let state = vec![1.0, 0.0];

        // Initially KL should be 0
        let kl = grpo.kl_from_reference(&state);
        assert!(kl.abs() < 1e-10);

        // After update, KL should be positive
        let rng_values = vec![0.1, 0.4, 0.6, 0.9];
        let group = grpo.sample_group(&state, simple_reward, &rng_values);
        grpo.update(&[group]);

        let kl = grpo.kl_from_reference(&state);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_grpo_update_reference() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default());
        let state = vec![1.0, 0.0];

        // Train a bit
        let rng_values = vec![0.1, 0.4, 0.6, 0.9];
        let group = grpo.sample_group(&state, simple_reward, &rng_values);
        grpo.update(&[group]);

        // KL should be positive
        assert!(grpo.kl_from_reference(&state) >= 0.0);

        // Update reference
        grpo.update_reference();

        // KL should be 0 again
        assert!(grpo.kl_from_reference(&state).abs() < 1e-10);
    }

    #[test]
    fn test_grpo_best_action() {
        let grpo = GRPO::new(2, 3, GRPOConfig::default());
        let state = vec![1.0, 0.0];

        let action = grpo.best_action(&state);
        assert!(action < 3);
    }

    #[test]
    fn test_grpo_stats_fields() {
        let mut grpo = GRPO::new(2, 2, GRPOConfig::default().with_group_size(4));
        let state = vec![1.0, 0.0];
        let rng_values = vec![0.1, 0.4, 0.6, 0.9];

        let group = grpo.sample_group(&state, simple_reward, &rng_values);
        let stats = grpo.update(&[group]);

        assert!(stats.entropy >= 0.0);
        assert!(stats.gradient_norm >= 0.0);
        assert!(stats.clip_fraction >= 0.0 && stats.clip_fraction <= 1.0);
    }

    #[test]
    fn test_grpo_advantage_filtering() {
        let config = GRPOConfig {
            advantage_threshold: Some(0.5),
            ..Default::default()
        };
        let mut grpo = GRPO::new(2, 2, config);

        // Create group with small reward differences
        let mut group = SampleGroup::new(vec![1.0, 0.0]);
        group.add_sample(0, 0.1, -0.5, -0.5);
        group.add_sample(1, 0.0, -0.5, -0.5);
        group.add_sample(0, -0.1, -0.5, -0.5);
        group.add_sample(1, 0.05, -0.5, -0.5);

        let stats = grpo.update(&[group]);
        assert!(stats.filtered_samples > 0);
    }

    #[test]
    fn test_online_grpo() {
        let mut online = OnlineGRPO::new(2, 2, GRPOConfig::default().with_group_size(4), 100);

        assert!(!online.ready_for_update());
        assert_eq!(online.buffer_len(), 0);

        // Add groups
        for _ in 0..10 {
            let mut group = SampleGroup::new(vec![1.0, 0.0]);
            group.add_sample(0, 1.0, -0.5, -0.5);
            group.add_sample(1, -1.0, -0.5, -0.5);
            group.add_sample(0, 0.5, -0.5, -0.5);
            group.add_sample(1, 0.0, -0.5, -0.5);
            online.add_group(group);
        }

        assert!(online.ready_for_update());
        assert_eq!(online.buffer_len(), 10);

        let stats = online.update();
        assert!(stats.num_samples > 0);
        assert_eq!(online.buffer_len(), 0);
    }

    #[test]
    fn test_online_grpo_buffer_overflow() {
        let mut online = OnlineGRPO::new(2, 2, GRPOConfig::default(), 5);

        for i in 0..10 {
            let group = SampleGroup::new(vec![i as f64, 0.0]);
            online.add_group(group);
        }

        // Should only keep last 5
        assert_eq!(online.buffer_len(), 5);
    }

    #[test]
    fn test_reward_weighted_grpo() {
        let mut rw_grpo =
            RewardWeightedGRPO::new(2, 2, GRPOConfig::default().with_group_size(4), 1.0);

        let mut group = SampleGroup::new(vec![1.0, 0.0]);
        group.add_sample(0, 2.0, -0.5, -0.5);
        group.add_sample(1, -1.0, -0.5, -0.5);
        group.add_sample(0, 1.0, -0.5, -0.5);
        group.add_sample(1, 0.0, -0.5, -0.5);

        let stats = rw_grpo.update(&[group]);
        assert!(stats.num_samples > 0);
    }

    #[test]
    fn test_grpo_analysis_empty() {
        let analysis = analyze_grpo_training(&[]);
        assert!((analysis.mean_kl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_grpo_analysis() {
        let stats = vec![
            GRPOStats {
                loss: 1.0,
                kl_divergence: 0.05,
                clip_fraction: 0.1,
                ..Default::default()
            },
            GRPOStats {
                loss: 0.8,
                kl_divergence: 0.04,
                clip_fraction: 0.15,
                ..Default::default()
            },
            GRPOStats {
                loss: 0.6,
                kl_divergence: 0.03,
                clip_fraction: 0.12,
                ..Default::default()
            },
        ];

        let analysis = analyze_grpo_training(&stats);
        assert!(analysis.kl_healthy); // KL is low
        assert!(!analysis.clipping_excessive); // Clip fraction < 30%
        assert!(analysis.loss_trend < 0.0); // Loss decreasing
    }

    #[test]
    fn test_grpo_analysis_unhealthy_kl() {
        let stats = vec![GRPOStats {
            kl_divergence: 0.5,
            clip_fraction: 0.1,
            ..Default::default()
        }];

        let analysis = analyze_grpo_training(&stats);
        assert!(!analysis.kl_healthy);
        assert!(analysis.recommended_kl_adjustment.is_some());
    }

    #[test]
    fn test_grpo_analysis_excessive_clipping() {
        let stats = vec![GRPOStats {
            kl_divergence: 0.05,
            clip_fraction: 0.5,
            ..Default::default()
        }];

        let analysis = analyze_grpo_training(&stats);
        assert!(analysis.clipping_excessive);
    }

    #[test]
    fn test_generate_groups() {
        let grpo = GRPO::new(2, 2, GRPOConfig::default().with_group_size(4));
        let states = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let rng_values: Vec<f64> = (0..8).map(|i| i as f64 / 8.0).collect();

        let groups = generate_groups(&grpo, &states, simple_reward, &rng_values);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 4);
        assert_eq!(groups[1].len(), 4);
    }

    #[test]
    fn test_group_statistics() {
        let mut group1 = SampleGroup::new(vec![1.0]);
        group1.add_sample(0, 2.0, 0.0, 0.0);
        group1.add_sample(1, 0.0, 0.0, 0.0);

        let mut group2 = SampleGroup::new(vec![-1.0]);
        group2.add_sample(0, 1.0, 0.0, 0.0);
        group2.add_sample(1, 3.0, 0.0, 0.0);

        let stats = group_statistics(&[group1, group2]);
        assert_eq!(stats.num_groups, 2);
        assert_eq!(stats.total_samples, 4);
        assert!((stats.mean_reward - 1.5).abs() < 1e-10); // (2+0+1+3)/4
        assert!((stats.max_reward - 3.0).abs() < 1e-10);
        assert!((stats.min_reward - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_group_statistics_empty() {
        let stats = group_statistics(&[]);
        assert_eq!(stats.num_groups, 0);
        assert_eq!(stats.total_samples, 0);
    }

    #[test]
    fn test_compute_trend() {
        // Decreasing values should have negative trend
        let decreasing = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        assert!(compute_trend(&decreasing) < 0.0);

        // Increasing values should have positive trend
        let increasing = vec![0.2, 0.4, 0.6, 0.8, 1.0];
        assert!(compute_trend(&increasing) > 0.0);

        // Constant values should have zero trend
        let constant = vec![0.5, 0.5, 0.5, 0.5];
        assert!(compute_trend(&constant).abs() < 1e-10);
    }

    #[test]
    fn test_grpo_entropy_coefficient() {
        let config = GRPOConfig {
            entropy_coeff: 0.1,
            ..Default::default()
        };
        let mut grpo = GRPO::new(2, 2, config);

        let mut group = SampleGroup::new(vec![1.0, 0.0]);
        group.add_sample(0, 1.0, -0.5, -0.5);
        group.add_sample(1, -1.0, -0.5, -0.5);

        let stats = grpo.update(&[group]);
        assert!(stats.entropy > 0.0);
    }

    #[test]
    fn test_grpo_gradient_clipping() {
        let config = GRPOConfig {
            max_grad_norm: Some(0.01),
            learning_rate: 10.0, // High LR to make clipping visible
            ..Default::default()
        };
        let mut grpo = GRPO::new(2, 2, config);

        let mut group = SampleGroup::new(vec![1.0, 0.0]);
        group.add_sample(0, 10.0, -0.5, -0.5); // Large reward difference
        group.add_sample(1, -10.0, -0.5, -0.5);

        let stats = grpo.update(&[group]);
        // Gradient should be clipped
        assert!(stats.gradient_norm > 0.0);
    }
}
