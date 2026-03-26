// Proximal Policy Optimization (PPO)
// Industry standard algorithm for RLHF and policy optimization

use crate::policy_gradient::{GAE, LinearValueFunction, SoftmaxPolicy, Step, Trajectory};

/// PPO configuration parameters
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Learning rate for policy updates
    pub policy_learning_rate: f64,
    /// Learning rate for value function updates
    pub value_learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f64,
    /// GAE lambda parameter
    pub gae_lambda: f64,
    /// Clipping parameter epsilon for policy ratio
    pub clip_epsilon: f64,
    /// Coefficient for value function loss
    pub value_loss_coef: f64,
    /// Coefficient for entropy bonus
    pub entropy_coef: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f64>,
    /// Number of epochs per update
    pub num_epochs: usize,
    /// Mini-batch size for updates
    pub batch_size: usize,
    /// Whether to normalize advantages
    pub normalize_advantages: bool,
    /// Target KL divergence for early stopping (None to disable)
    pub target_kl: Option<f64>,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            policy_learning_rate: 3e-4,
            value_learning_rate: 1e-3,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: Some(0.5),
            num_epochs: 4,
            batch_size: 64,
            normalize_advantages: true,
            target_kl: Some(0.02),
        }
    }
}

/// Experience buffer for PPO training
#[derive(Debug, Clone)]
pub struct PPOBuffer {
    /// States
    pub states: Vec<Vec<f64>>,
    /// Actions taken
    pub actions: Vec<usize>,
    /// Rewards received
    pub rewards: Vec<f64>,
    /// Log probabilities under old policy
    pub old_log_probs: Vec<f64>,
    /// Value estimates
    pub values: Vec<f64>,
    /// Done flags
    pub dones: Vec<bool>,
    /// Computed advantages
    pub advantages: Vec<f64>,
    /// Computed returns (targets for value function)
    pub returns: Vec<f64>,
}

impl PPOBuffer {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            old_log_probs: Vec::new(),
            values: Vec::new(),
            dones: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    /// Add a single step to the buffer
    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        log_prob: f64,
        value: f64,
        done: bool,
    ) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.old_log_probs.push(log_prob);
        self.values.push(value);
        self.dones.push(done);
    }

    /// Add data from a trajectory
    pub fn add_trajectory(&mut self, trajectory: &Trajectory, values: &[f64]) {
        for (step, &value) in trajectory.steps.iter().zip(values.iter()) {
            self.add(
                step.state.clone(),
                step.action,
                step.reward,
                step.log_prob,
                value,
                step.done,
            );
        }
    }

    /// Compute advantages and returns using GAE
    pub fn compute_advantages(&mut self, gamma: f64, gae_lambda: f64, last_value: f64) {
        let n = self.rewards.len();
        if n == 0 {
            return;
        }

        // Extend values with bootstrap value
        let mut values = self.values.clone();
        values.push(last_value);

        // Compute GAE
        let gae = GAE::new(gae_lambda, gamma);
        self.advantages = gae.compute_advantages(&self.rewards, &values, &self.dones);

        // Compute returns as advantage + value
        self.returns = self
            .advantages
            .iter()
            .zip(self.values.iter())
            .map(|(&a, &v)| a + v)
            .collect();
    }

    /// Normalize advantages (zero mean, unit variance)
    pub fn normalize_advantages(&mut self) {
        if self.advantages.is_empty() {
            return;
        }

        let mean: f64 = self.advantages.iter().sum::<f64>() / self.advantages.len() as f64;
        let variance: f64 = self
            .advantages
            .iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f64>()
            / self.advantages.len() as f64;
        let std = (variance + 1e-8).sqrt();

        for a in &mut self.advantages {
            *a = (*a - mean) / std;
        }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.old_log_probs.clear();
        self.values.clear();
        self.dones.clear();
        self.advantages.clear();
        self.returns.clear();
    }

    /// Get a mini-batch of indices
    pub fn get_batch_indices(&self, batch_size: usize, batch_idx: usize) -> Vec<usize> {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.len());
        (start..end).collect()
    }

    /// Get number of batches
    pub fn num_batches(&self, batch_size: usize) -> usize {
        self.len().div_ceil(batch_size)
    }
}

impl Default for PPOBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// PPO algorithm implementation
#[derive(Debug)]
pub struct PPO {
    /// Policy network (actor)
    pub policy: SoftmaxPolicy,
    /// Value function (critic)
    pub value_fn: LinearValueFunction,
    /// Configuration
    pub config: PPOConfig,
}

impl PPO {
    pub fn new(state_dim: usize, num_actions: usize, config: PPOConfig) -> Self {
        let policy = SoftmaxPolicy::new(state_dim, num_actions);
        let value_fn = LinearValueFunction::new(state_dim, config.value_learning_rate);
        Self {
            policy,
            value_fn,
            config,
        }
    }

    pub fn with_default_config(state_dim: usize, num_actions: usize) -> Self {
        Self::new(state_dim, num_actions, PPOConfig::default())
    }

    /// Get action and log probability for a state
    pub fn get_action(&self, state: &[f64], rng_value: f64) -> (usize, f64) {
        self.policy.sample_action(state, rng_value)
    }

    /// Get value estimate for a state
    pub fn get_value(&self, state: &[f64]) -> f64 {
        self.value_fn.predict(state)
    }

    /// Compute PPO clipped objective
    fn compute_policy_loss(
        &self,
        state: &[f64],
        action: usize,
        advantage: f64,
        old_log_prob: f64,
    ) -> (f64, f64, f64) {
        let new_log_prob = self.policy.log_prob(state, action);
        let ratio = (new_log_prob - old_log_prob).exp();

        // Clipped objective
        let clipped_ratio = ratio.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        );

        let policy_loss1 = ratio * advantage;
        let policy_loss2 = clipped_ratio * advantage;

        // PPO uses minimum to create pessimistic bound
        let policy_loss = policy_loss1.min(policy_loss2);

        // Entropy for exploration
        let entropy = self.policy.entropy(state);

        // KL divergence approximation
        let kl = old_log_prob - new_log_prob;

        (policy_loss, entropy, kl)
    }

    /// Compute value function loss (MSE)
    fn compute_value_loss(&self, state: &[f64], target_return: f64) -> f64 {
        let value = self.value_fn.predict(state);
        let error = target_return - value;
        0.5 * error * error
    }

    /// Update policy using PPO with a buffer of experiences
    pub fn update(&mut self, buffer: &mut PPOBuffer) -> PPOStats {
        if buffer.is_empty() {
            return PPOStats::default();
        }

        // Compute advantages if not already done
        if buffer.advantages.is_empty() {
            // Assume last state was terminal for simplicity
            buffer.compute_advantages(self.config.gamma, self.config.gae_lambda, 0.0);
        }

        // Normalize advantages
        if self.config.normalize_advantages {
            buffer.normalize_advantages();
        }

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut total_kl = 0.0;
        let mut update_count = 0;
        let mut early_stop = false;

        // Multiple epochs of training
        for _epoch in 0..self.config.num_epochs {
            if early_stop {
                break;
            }

            let num_batches = buffer.num_batches(self.config.batch_size);

            for batch_idx in 0..num_batches {
                let indices = buffer.get_batch_indices(self.config.batch_size, batch_idx);

                let mut batch_policy_grad =
                    vec![vec![0.0; self.policy.num_actions]; self.policy.state_dim];
                let mut batch_policy_loss = 0.0;
                let mut batch_value_loss = 0.0;
                let mut batch_entropy = 0.0;
                let mut batch_kl = 0.0;

                for &i in &indices {
                    let state = &buffer.states[i];
                    let action = buffer.actions[i];
                    let advantage = buffer.advantages[i];
                    let old_log_prob = buffer.old_log_probs[i];
                    let target_return = buffer.returns[i];

                    // Compute losses
                    let (policy_loss, entropy, kl) =
                        self.compute_policy_loss(state, action, advantage, old_log_prob);
                    let value_loss = self.compute_value_loss(state, target_return);

                    batch_policy_loss += policy_loss;
                    batch_value_loss += value_loss;
                    batch_entropy += entropy;
                    batch_kl += kl;

                    // Compute policy gradient
                    let log_prob_grad = self.policy.log_prob_gradient(state, action);
                    let new_log_prob = self.policy.log_prob(state, action);
                    let ratio = (new_log_prob - old_log_prob).exp();
                    let clipped_ratio = ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    );

                    // Use gradient only if unclipped
                    let use_unclipped = ratio * advantage <= clipped_ratio * advantage;
                    let grad_scale = if use_unclipped { advantage } else { 0.0 };

                    for (i, row) in log_prob_grad.iter().enumerate() {
                        for (a, &g) in row.iter().enumerate() {
                            batch_policy_grad[i][a] += g * grad_scale;
                        }
                    }

                    // Update value function
                    self.value_fn.update(state, target_return);
                }

                let batch_len = indices.len() as f64;

                // Normalize batch gradient
                for row in &mut batch_policy_grad {
                    for g in row.iter_mut() {
                        *g /= batch_len;
                    }
                }

                // Add entropy gradient
                if self.config.entropy_coef > 0.0 {
                    for &i in &indices {
                        let state = &buffer.states[i];
                        let probs = self.policy.action_probs(state);
                        for (j, &s) in state.iter().enumerate() {
                            for (a, &prob) in probs.iter().enumerate() {
                                let entropy_grad = -s * prob * (1.0 + prob.ln().max(-10.0));
                                batch_policy_grad[j][a] +=
                                    self.config.entropy_coef * entropy_grad / batch_len;
                            }
                        }
                    }
                }

                // Gradient clipping
                let grad_norm: f64 = batch_policy_grad
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|&g| g * g)
                    .sum::<f64>()
                    .sqrt();

                if let Some(max_norm) = self.config.max_grad_norm
                    && grad_norm > max_norm
                {
                    let clip_ratio = max_norm / grad_norm;
                    for row in &mut batch_policy_grad {
                        for g in row.iter_mut() {
                            *g *= clip_ratio;
                        }
                    }
                }

                // Update policy
                self.policy
                    .update_weights(&batch_policy_grad, self.config.policy_learning_rate);

                // Accumulate statistics
                total_policy_loss += batch_policy_loss / batch_len;
                total_value_loss += batch_value_loss / batch_len;
                total_entropy += batch_entropy / batch_len;
                total_kl += batch_kl / batch_len;
                update_count += 1;

                // Check for early stopping based on KL divergence
                if let Some(target_kl) = self.config.target_kl {
                    let mean_kl = batch_kl / batch_len;
                    if mean_kl > target_kl * 1.5 {
                        early_stop = true;
                        break;
                    }
                }
            }
        }

        if update_count == 0 {
            return PPOStats::default();
        }

        let count = update_count as f64;
        PPOStats {
            policy_loss: -total_policy_loss / count, // Negate because we maximize
            value_loss: total_value_loss / count,
            entropy: total_entropy / count,
            approx_kl: total_kl / count,
            clip_fraction: 0.0, // Would need to track this separately
            explained_variance: self.compute_explained_variance(buffer),
        }
    }

    /// Compute explained variance of value function
    fn compute_explained_variance(&self, buffer: &PPOBuffer) -> f64 {
        if buffer.returns.is_empty() {
            return 0.0;
        }

        let predictions: Vec<f64> = buffer
            .states
            .iter()
            .map(|s| self.value_fn.predict(s))
            .collect();

        let mean_return: f64 = buffer.returns.iter().sum::<f64>() / buffer.returns.len() as f64;
        let var_return: f64 = buffer
            .returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / buffer.returns.len() as f64;

        if var_return < 1e-10 {
            return 1.0;
        }

        let mse: f64 = buffer
            .returns
            .iter()
            .zip(predictions.iter())
            .map(|(&r, &p)| (r - p).powi(2))
            .sum::<f64>()
            / buffer.returns.len() as f64;

        1.0 - mse / var_return
    }

    /// Collect a trajectory using the current policy
    pub fn collect_trajectory(
        &self,
        initial_state: Vec<f64>,
        step_fn: impl Fn(&[f64], usize) -> (Vec<f64>, f64, bool),
        max_steps: usize,
        rng_values: &[f64],
    ) -> (Trajectory, Vec<f64>) {
        let mut trajectory = Trajectory::new();
        let mut values = Vec::new();
        let mut state = initial_state;

        for &rng in rng_values.iter().take(max_steps) {
            let value = self.get_value(&state);
            values.push(value);

            let (action, log_prob) = self.get_action(&state, rng);
            let (next_state, reward, done) = step_fn(&state, action);

            trajectory.add_step(Step::new(state.clone(), action, reward, log_prob, done));

            if done {
                break;
            }

            state = next_state;
        }

        (trajectory, values)
    }
}

/// Statistics from PPO update
#[derive(Debug, Clone, Default)]
pub struct PPOStats {
    /// Policy loss (negative of objective)
    pub policy_loss: f64,
    /// Value function loss
    pub value_loss: f64,
    /// Mean policy entropy
    pub entropy: f64,
    /// Approximate KL divergence
    pub approx_kl: f64,
    /// Fraction of clipped ratios
    pub clip_fraction: f64,
    /// Explained variance of value function
    pub explained_variance: f64,
}

/// PPO with clipped value function (PPO2 style)
#[derive(Debug)]
pub struct PPOClippedValue {
    pub policy: SoftmaxPolicy,
    pub value_fn: LinearValueFunction,
    pub config: PPOConfig,
}

impl PPOClippedValue {
    pub fn new(state_dim: usize, num_actions: usize, config: PPOConfig) -> Self {
        let policy = SoftmaxPolicy::new(state_dim, num_actions);
        let value_fn = LinearValueFunction::new(state_dim, config.value_learning_rate);
        Self {
            policy,
            value_fn,
            config,
        }
    }

    /// Compute clipped value loss
    pub fn compute_clipped_value_loss(
        &self,
        state: &[f64],
        old_value: f64,
        target_return: f64,
    ) -> f64 {
        let value = self.value_fn.predict(state);

        // Clipped value estimate
        let clipped_value = old_value
            + (value - old_value).clamp(-self.config.clip_epsilon, self.config.clip_epsilon);

        // Loss is max of unclipped and clipped
        let loss_unclipped = (value - target_return).powi(2);
        let loss_clipped = (clipped_value - target_return).powi(2);

        0.5 * loss_unclipped.max(loss_clipped)
    }
}

/// Rollout buffer for PPO with GAE
#[derive(Debug, Clone)]
pub struct RolloutBuffer {
    states: Vec<Vec<f64>>,
    actions: Vec<usize>,
    rewards: Vec<f64>,
    log_probs: Vec<f64>,
    values: Vec<f64>,
    dones: Vec<bool>,
    advantages: Vec<f64>,
    returns: Vec<f64>,
    pos: usize,
    full: bool,
    buffer_size: usize,
}

impl RolloutBuffer {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            states: Vec::with_capacity(buffer_size),
            actions: Vec::with_capacity(buffer_size),
            rewards: Vec::with_capacity(buffer_size),
            log_probs: Vec::with_capacity(buffer_size),
            values: Vec::with_capacity(buffer_size),
            dones: Vec::with_capacity(buffer_size),
            advantages: Vec::with_capacity(buffer_size),
            returns: Vec::with_capacity(buffer_size),
            pos: 0,
            full: false,
            buffer_size,
        }
    }

    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        log_prob: f64,
        value: f64,
        done: bool,
    ) {
        if self.states.len() < self.buffer_size {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.log_probs.push(log_prob);
            self.values.push(value);
            self.dones.push(done);
        } else {
            self.states[self.pos] = state;
            self.actions[self.pos] = action;
            self.rewards[self.pos] = reward;
            self.log_probs[self.pos] = log_prob;
            self.values[self.pos] = value;
            self.dones[self.pos] = done;
        }

        self.pos = (self.pos + 1) % self.buffer_size;
        if self.pos == 0 {
            self.full = true;
        }
    }

    pub fn size(&self) -> usize {
        if self.full {
            self.buffer_size
        } else {
            self.pos
        }
    }

    pub fn is_ready(&self) -> bool {
        self.full
    }

    /// Compute GAE and store advantages/returns
    pub fn compute_returns_and_advantage(&mut self, last_value: f64, gamma: f64, gae_lambda: f64) {
        let n = self.size();
        if n == 0 {
            return;
        }

        self.advantages = vec![0.0; n];
        self.returns = vec![0.0; n];

        let mut last_gae_lam = 0.0;

        for t in (0..n).rev() {
            let next_value = if t == n - 1 {
                if self.dones[t] { 0.0 } else { last_value }
            } else if self.dones[t] {
                0.0
            } else {
                self.values[t + 1]
            };

            let delta = self.rewards[t] + gamma * next_value - self.values[t];
            let next_gae = if self.dones[t] { 0.0 } else { last_gae_lam };
            last_gae_lam = delta + gamma * gae_lambda * next_gae;

            self.advantages[t] = last_gae_lam;
            self.returns[t] = last_gae_lam + self.values[t];
        }
    }

    /// Get experience at index
    pub fn get(&self, idx: usize) -> Option<RolloutSample> {
        if idx >= self.size() {
            return None;
        }

        Some(RolloutSample {
            state: self.states[idx].clone(),
            action: self.actions[idx],
            reward: self.rewards[idx],
            log_prob: self.log_probs[idx],
            value: self.values[idx],
            done: self.dones[idx],
            advantage: self.advantages.get(idx).copied().unwrap_or(0.0),
            returns: self.returns.get(idx).copied().unwrap_or(0.0),
        })
    }

    pub fn reset(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.log_probs.clear();
        self.values.clear();
        self.dones.clear();
        self.advantages.clear();
        self.returns.clear();
        self.pos = 0;
        self.full = false;
    }
}

/// Single sample from rollout buffer
#[derive(Debug, Clone)]
pub struct RolloutSample {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub log_prob: f64,
    pub value: f64,
    pub done: bool,
    pub advantage: f64,
    pub returns: f64,
}

/// Compute ratio for PPO objective
pub fn compute_ratio(new_log_prob: f64, old_log_prob: f64) -> f64 {
    (new_log_prob - old_log_prob).exp()
}

/// Compute clipped surrogate objective
pub fn clipped_surrogate(ratio: f64, advantage: f64, clip_epsilon: f64) -> f64 {
    let clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
    (ratio * advantage).min(clipped_ratio * advantage)
}

/// Analyze PPO training statistics
pub fn analyze_ppo_training(stats_history: &[PPOStats]) -> PPOAnalysis {
    if stats_history.is_empty() {
        return PPOAnalysis::default();
    }

    let n = stats_history.len() as f64;

    let mean_policy_loss = stats_history.iter().map(|s| s.policy_loss).sum::<f64>() / n;
    let mean_value_loss = stats_history.iter().map(|s| s.value_loss).sum::<f64>() / n;
    let mean_entropy = stats_history.iter().map(|s| s.entropy).sum::<f64>() / n;
    let mean_kl = stats_history.iter().map(|s| s.approx_kl).sum::<f64>() / n;

    // Check for training stability
    let entropy_trend: f64 = if stats_history.len() > 1 {
        let first_half: f64 = stats_history[..stats_history.len() / 2]
            .iter()
            .map(|s| s.entropy)
            .sum::<f64>()
            / (stats_history.len() / 2) as f64;
        let second_half: f64 = stats_history[stats_history.len() / 2..]
            .iter()
            .map(|s| s.entropy)
            .sum::<f64>()
            / (stats_history.len() - stats_history.len() / 2) as f64;
        second_half - first_half
    } else {
        0.0
    };

    let is_stable = mean_kl < 0.05 && mean_entropy > 0.1;

    PPOAnalysis {
        mean_policy_loss,
        mean_value_loss,
        mean_entropy,
        mean_kl,
        entropy_trend,
        is_stable,
    }
}

/// Analysis of PPO training run
#[derive(Debug, Clone, Default)]
pub struct PPOAnalysis {
    /// Mean policy loss
    pub mean_policy_loss: f64,
    /// Mean value loss
    pub mean_value_loss: f64,
    /// Mean entropy
    pub mean_entropy: f64,
    /// Mean approximate KL
    pub mean_kl: f64,
    /// Entropy trend (positive = increasing)
    pub entropy_trend: f64,
    /// Whether training is stable
    pub is_stable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_config_default() {
        let config = PPOConfig::default();

        assert!(config.policy_learning_rate > 0.0);
        assert!(config.gamma > 0.0 && config.gamma <= 1.0);
        assert!(config.clip_epsilon > 0.0 && config.clip_epsilon < 1.0);
        assert!(config.num_epochs > 0);
    }

    #[test]
    fn test_ppo_buffer_creation() {
        let buffer = PPOBuffer::new();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_ppo_buffer_add() {
        let mut buffer = PPOBuffer::new();

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, true);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.states.len(), 2);
        assert_eq!(buffer.actions, vec![0, 1]);
    }

    #[test]
    fn test_ppo_buffer_compute_advantages() {
        let mut buffer = PPOBuffer::new();

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, true);

        buffer.compute_advantages(0.99, 0.95, 0.0);

        assert_eq!(buffer.advantages.len(), 2);
        assert_eq!(buffer.returns.len(), 2);
    }

    #[test]
    fn test_ppo_buffer_normalize_advantages() {
        let mut buffer = PPOBuffer::new();

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, false);
        buffer.add(vec![0.5, 0.5], 0, 3.0, -0.4, 0.6, true);

        buffer.compute_advantages(0.99, 0.95, 0.0);
        buffer.normalize_advantages();

        // Mean should be ~0
        let mean: f64 = buffer.advantages.iter().sum::<f64>() / buffer.advantages.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_ppo_buffer_batches() {
        let mut buffer = PPOBuffer::new();

        for i in 0..10 {
            buffer.add(vec![i as f64], i % 2, 1.0, -0.5, 0.5, false);
        }

        assert_eq!(buffer.num_batches(3), 4); // ceil(10/3)
        assert_eq!(buffer.get_batch_indices(3, 0), vec![0, 1, 2]);
        assert_eq!(buffer.get_batch_indices(3, 3), vec![9]);
    }

    #[test]
    fn test_ppo_creation() {
        let ppo = PPO::with_default_config(4, 3);

        assert_eq!(ppo.policy.state_dim, 4);
        assert_eq!(ppo.policy.num_actions, 3);
    }

    #[test]
    fn test_ppo_get_action() {
        let ppo = PPO::with_default_config(4, 2);
        let state = vec![1.0, 0.0, -0.5, 0.3];

        let (action, log_prob) = ppo.get_action(&state, 0.5);

        assert!(action < 2);
        assert!(log_prob.is_finite());
        assert!(log_prob <= 0.0); // Log probability is always <= 0
    }

    #[test]
    fn test_ppo_get_value() {
        let ppo = PPO::with_default_config(4, 2);
        let state = vec![1.0, 0.0, -0.5, 0.3];

        let value = ppo.get_value(&state);
        assert!(value.is_finite());
    }

    #[test]
    fn test_ppo_update_empty_buffer() {
        let mut ppo = PPO::with_default_config(4, 2);
        let mut buffer = PPOBuffer::new();

        let stats = ppo.update(&mut buffer);

        assert_eq!(stats.policy_loss, 0.0);
        assert_eq!(stats.value_loss, 0.0);
    }

    #[test]
    fn test_ppo_update() {
        let mut ppo = PPO::with_default_config(2, 2);
        let mut buffer = PPOBuffer::new();

        // Add some experiences
        for _ in 0..10 {
            buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        }
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, true);

        let stats = ppo.update(&mut buffer);

        assert!(stats.entropy >= 0.0);
    }

    #[test]
    fn test_compute_ratio() {
        // Same probability gives ratio of 1
        let ratio = compute_ratio(-0.5, -0.5);
        assert!((ratio - 1.0).abs() < 1e-10);

        // Higher new probability gives ratio > 1
        let ratio = compute_ratio(-0.3, -0.5);
        assert!(ratio > 1.0);

        // Lower new probability gives ratio < 1
        let ratio = compute_ratio(-0.7, -0.5);
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_clipped_surrogate() {
        let clip_epsilon = 0.2;

        // When ratio is within bounds, use unclipped
        let loss = clipped_surrogate(1.1, 1.0, clip_epsilon);
        assert!((loss - 1.1).abs() < 1e-10);

        // When ratio exceeds upper bound, clip
        let loss = clipped_surrogate(1.5, 1.0, clip_epsilon);
        assert!((loss - 1.2).abs() < 1e-10); // min(1.5, 1.2) = 1.2

        // When ratio is below lower bound with positive advantage
        let loss = clipped_surrogate(0.5, 1.0, clip_epsilon);
        assert!((loss - 0.5).abs() < 1e-10); // min(0.5, 0.8) = 0.5

        // Negative advantage changes the clipping behavior
        let loss = clipped_surrogate(1.5, -1.0, clip_epsilon);
        assert!((loss - -1.5).abs() < 1e-10); // min(-1.5, -1.2) = -1.5
    }

    #[test]
    fn test_rollout_buffer() {
        let mut buffer = RolloutBuffer::new(100);

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, true);

        assert_eq!(buffer.size(), 2);
        assert!(!buffer.is_ready());

        let sample = buffer.get(0).unwrap();
        assert_eq!(sample.action, 0);
        assert!((sample.reward - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rollout_buffer_compute_returns() {
        let mut buffer = RolloutBuffer::new(100);

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 0.8, true);

        buffer.compute_returns_and_advantage(0.0, 0.99, 0.95);

        let sample = buffer.get(1).unwrap();
        // Last step advantage = reward - value = 2.0 - 0.8 = 1.2
        assert!((sample.advantage - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_rollout_buffer_reset() {
        let mut buffer = RolloutBuffer::new(100);

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 0.5, false);
        assert_eq!(buffer.size(), 1);

        buffer.reset();
        assert_eq!(buffer.size(), 0);
        assert!(buffer.get(0).is_none());
    }

    #[test]
    fn test_ppo_clipped_value() {
        let config = PPOConfig::default();
        let ppo = PPOClippedValue::new(4, 2, config);

        let state = vec![1.0, 0.0, -0.5, 0.3];
        let old_value = 0.5;
        let target_return = 1.0;

        let loss = ppo.compute_clipped_value_loss(&state, old_value, target_return);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_analyze_ppo_training_empty() {
        let analysis = analyze_ppo_training(&[]);
        assert_eq!(analysis.mean_policy_loss, 0.0);
    }

    #[test]
    fn test_analyze_ppo_training() {
        let stats = vec![
            PPOStats {
                policy_loss: 0.1,
                value_loss: 0.2,
                entropy: 0.5,
                approx_kl: 0.01,
                clip_fraction: 0.1,
                explained_variance: 0.8,
            },
            PPOStats {
                policy_loss: 0.08,
                value_loss: 0.15,
                entropy: 0.45,
                approx_kl: 0.02,
                clip_fraction: 0.15,
                explained_variance: 0.85,
            },
        ];

        let analysis = analyze_ppo_training(&stats);

        assert!((analysis.mean_policy_loss - 0.09).abs() < 1e-10);
        assert!(analysis.mean_entropy > 0.0);
        assert!(analysis.is_stable);
    }

    #[test]
    fn test_ppo_buffer_add_trajectory() {
        let mut buffer = PPOBuffer::new();
        let mut trajectory = Trajectory::new();

        trajectory.add_step(Step::new(vec![1.0, 0.0], 0, 1.0, -0.5, false));
        trajectory.add_step(Step::new(vec![0.0, 1.0], 1, 2.0, -0.3, true));

        let values = vec![0.5, 0.8];
        buffer.add_trajectory(&trajectory, &values);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.actions, vec![0, 1]);
        assert_eq!(buffer.values, vec![0.5, 0.8]);
    }

    #[test]
    fn test_ppo_explained_variance() {
        let ppo = PPO::with_default_config(2, 2);
        let mut buffer = PPOBuffer::new();

        buffer.add(vec![1.0, 0.0], 0, 1.0, -0.5, 1.0, false);
        buffer.add(vec![0.0, 1.0], 1, 2.0, -0.3, 2.0, true);
        buffer.compute_advantages(0.99, 0.95, 0.0);

        let ev = ppo.compute_explained_variance(&buffer);
        // With initial zero weights, predictions are 0, so EV will be low
        assert!(ev.is_finite());
    }

    #[test]
    fn test_ppo_early_stopping() {
        let config = PPOConfig {
            target_kl: Some(0.001), // Very tight KL constraint
            num_epochs: 10,
            ..PPOConfig::default()
        };
        let mut ppo = PPO::new(2, 2, config);
        let mut buffer = PPOBuffer::new();

        // Add experiences with different old_log_probs to cause KL divergence
        for i in 0..20 {
            buffer.add(
                vec![(i as f64) * 0.1, 1.0 - (i as f64) * 0.1],
                i % 2,
                1.0,
                -2.0, // Very different from what policy would give
                0.5,
                i == 19,
            );
        }

        let stats = ppo.update(&mut buffer);
        // Should have done some updates but potentially stopped early
        assert!(stats.entropy >= 0.0);
    }
}
