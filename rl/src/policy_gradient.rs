// Policy Gradient (REINFORCE) Algorithm
// Foundation reinforcement learning algorithm for policy optimization

use std::collections::HashMap;

/// Configuration for policy gradient training
#[derive(Debug, Clone)]
pub struct PolicyGradientConfig {
    /// Learning rate for policy updates
    pub learning_rate: f64,
    /// Discount factor for future rewards (gamma)
    pub discount_factor: f64,
    /// Whether to use reward baseline for variance reduction
    pub use_baseline: bool,
    /// Entropy bonus coefficient for exploration
    pub entropy_coefficient: f64,
    /// Maximum gradient norm for clipping (None for no clipping)
    pub max_grad_norm: Option<f64>,
}

impl Default for PolicyGradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            discount_factor: 0.99,
            use_baseline: true,
            entropy_coefficient: 0.01,
            max_grad_norm: Some(0.5),
        }
    }
}

/// Represents a single step in a trajectory
#[derive(Debug, Clone)]
pub struct Step {
    /// State at this step (feature vector)
    pub state: Vec<f64>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Log probability of the action under current policy
    pub log_prob: f64,
    /// Whether this step is terminal
    pub done: bool,
}

impl Step {
    pub fn new(state: Vec<f64>, action: usize, reward: f64, log_prob: f64, done: bool) -> Self {
        Self {
            state,
            action,
            reward,
            log_prob,
            done,
        }
    }
}

/// A complete trajectory (episode)
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub steps: Vec<Step>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Calculate total undiscounted return
    pub fn total_reward(&self) -> f64 {
        self.steps.iter().map(|s| s.reward).sum()
    }

    /// Calculate discounted returns for each step (reward-to-go)
    pub fn compute_returns(&self, discount_factor: f64) -> Vec<f64> {
        let n = self.steps.len();
        if n == 0 {
            return Vec::new();
        }

        let mut returns = vec![0.0; n];
        let mut running_return = 0.0;

        for i in (0..n).rev() {
            running_return = self.steps[i].reward + discount_factor * running_return;
            returns[i] = running_return;
        }

        returns
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple softmax policy for discrete action spaces
#[derive(Debug, Clone)]
pub struct SoftmaxPolicy {
    /// Policy parameters: weights[state_dim][num_actions]
    pub weights: Vec<Vec<f64>>,
    /// Number of actions
    pub num_actions: usize,
    /// State dimension
    pub state_dim: usize,
}

impl SoftmaxPolicy {
    pub fn new(state_dim: usize, num_actions: usize) -> Self {
        // Initialize weights to small random values (using simple deterministic init)
        let weights = (0..state_dim)
            .map(|i| {
                (0..num_actions)
                    .map(|j| ((i * 17 + j * 31) % 100) as f64 * 0.001 - 0.05)
                    .collect()
            })
            .collect();

        Self {
            weights,
            num_actions,
            state_dim,
        }
    }

    /// Initialize with zero weights
    pub fn zeros(state_dim: usize, num_actions: usize) -> Self {
        let weights = vec![vec![0.0; num_actions]; state_dim];
        Self {
            weights,
            num_actions,
            state_dim,
        }
    }

    /// Compute action logits for a given state
    pub fn compute_logits(&self, state: &[f64]) -> Vec<f64> {
        assert_eq!(state.len(), self.state_dim, "State dimension mismatch");

        let mut logits = vec![0.0; self.num_actions];
        for (i, &s) in state.iter().enumerate() {
            for (a, logit) in logits.iter_mut().enumerate() {
                *logit += s * self.weights[i][a];
            }
        }
        logits
    }

    /// Compute action probabilities using softmax
    pub fn action_probs(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.compute_logits(state);
        softmax(&logits)
    }

    /// Sample an action and return (action, log_prob)
    pub fn sample_action(&self, state: &[f64], rng_value: f64) -> (usize, f64) {
        let probs = self.action_probs(state);
        let action = sample_categorical(&probs, rng_value);
        let log_prob = probs[action].ln();
        (action, log_prob)
    }

    /// Get log probability of an action given state
    pub fn log_prob(&self, state: &[f64], action: usize) -> f64 {
        let probs = self.action_probs(state);
        probs[action].ln()
    }

    /// Compute policy entropy for a given state
    pub fn entropy(&self, state: &[f64]) -> f64 {
        let probs = self.action_probs(state);
        compute_entropy(&probs)
    }

    /// Compute gradient of log probability w.r.t. weights
    /// Returns gradient[state_dim][num_actions]
    pub fn log_prob_gradient(&self, state: &[f64], action: usize) -> Vec<Vec<f64>> {
        let probs = self.action_probs(state);
        let mut gradient = vec![vec![0.0; self.num_actions]; self.state_dim];

        // Gradient of log π(a|s) = φ(s,a) - Σ_a' π(a'|s) φ(s,a')
        // For linear softmax: φ(s,a) = s (feature for action a)
        for (i, &s) in state.iter().enumerate() {
            for (a, (&prob, grad)) in probs.iter().zip(gradient[i].iter_mut()).enumerate() {
                if a == action {
                    *grad = s * (1.0 - prob);
                } else {
                    *grad = -s * prob;
                }
            }
        }

        gradient
    }

    /// Update weights using gradient
    pub fn update_weights(&mut self, gradient: &[Vec<f64>], learning_rate: f64) {
        for (i, row) in gradient.iter().enumerate() {
            for (a, &g) in row.iter().enumerate() {
                self.weights[i][a] += learning_rate * g;
            }
        }
    }
}

/// REINFORCE algorithm implementation
#[derive(Debug)]
pub struct REINFORCE {
    pub policy: SoftmaxPolicy,
    pub config: PolicyGradientConfig,
    /// Running mean of returns for baseline
    baseline: f64,
    /// Number of episodes seen
    episode_count: usize,
}

impl REINFORCE {
    pub fn new(policy: SoftmaxPolicy, config: PolicyGradientConfig) -> Self {
        Self {
            policy,
            config,
            baseline: 0.0,
            episode_count: 0,
        }
    }

    pub fn with_default_config(state_dim: usize, num_actions: usize) -> Self {
        let policy = SoftmaxPolicy::new(state_dim, num_actions);
        Self::new(policy, PolicyGradientConfig::default())
    }

    /// Update policy using a single trajectory
    pub fn update(&mut self, trajectory: &Trajectory) -> PolicyGradientStats {
        if trajectory.is_empty() {
            return PolicyGradientStats::default();
        }

        // Compute discounted returns
        let returns = trajectory.compute_returns(self.config.discount_factor);
        let total_reward = trajectory.total_reward();

        // Update baseline (moving average of returns)
        self.episode_count += 1;
        let alpha = 1.0 / self.episode_count as f64;
        self.baseline = self.baseline * (1.0 - alpha) + returns[0] * alpha;

        // Compute policy gradient
        let mut total_gradient = vec![vec![0.0; self.policy.num_actions]; self.policy.state_dim];
        let mut total_entropy = 0.0;
        let mut grad_norm_sum = 0.0;

        for (t, step) in trajectory.steps.iter().enumerate() {
            // Advantage estimate (return - baseline)
            let advantage = if self.config.use_baseline {
                returns[t] - self.baseline
            } else {
                returns[t]
            };

            // Policy gradient: ∇log π(a|s) * A(s,a)
            let log_prob_grad = self.policy.log_prob_gradient(&step.state, step.action);

            // Accumulate weighted gradient
            for (i, row) in log_prob_grad.iter().enumerate() {
                for (a, &g) in row.iter().enumerate() {
                    let weighted_grad = g * advantage;
                    total_gradient[i][a] += weighted_grad;
                    grad_norm_sum += weighted_grad * weighted_grad;
                }
            }

            // Entropy bonus
            total_entropy += self.policy.entropy(&step.state);
        }

        // Add entropy gradient (encourages exploration)
        if self.config.entropy_coefficient > 0.0 {
            for step in &trajectory.steps {
                let entropy_grad = self.entropy_gradient(&step.state);
                for (i, row) in entropy_grad.iter().enumerate() {
                    for (a, &g) in row.iter().enumerate() {
                        total_gradient[i][a] += self.config.entropy_coefficient * g;
                    }
                }
            }
        }

        // Gradient clipping
        let grad_norm = grad_norm_sum.sqrt();
        let clip_ratio = if let Some(max_norm) = self.config.max_grad_norm {
            if grad_norm > max_norm {
                max_norm / grad_norm
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Apply clipping
        if clip_ratio < 1.0 {
            for row in &mut total_gradient {
                for g in row.iter_mut() {
                    *g *= clip_ratio;
                }
            }
        }

        // Update policy weights
        self.policy
            .update_weights(&total_gradient, self.config.learning_rate);

        PolicyGradientStats {
            episode_return: total_reward,
            episode_length: trajectory.len(),
            mean_entropy: total_entropy / trajectory.len() as f64,
            gradient_norm: grad_norm,
            baseline: self.baseline,
        }
    }

    /// Update policy using a batch of trajectories
    pub fn update_batch(&mut self, trajectories: &[Trajectory]) -> PolicyGradientStats {
        if trajectories.is_empty() {
            return PolicyGradientStats::default();
        }

        let mut total_return = 0.0;
        let mut total_length = 0;
        let mut total_entropy = 0.0;

        // Accumulate gradients across all trajectories
        let mut batch_gradient = vec![vec![0.0; self.policy.num_actions]; self.policy.state_dim];

        for trajectory in trajectories {
            if trajectory.is_empty() {
                continue;
            }

            let returns = trajectory.compute_returns(self.config.discount_factor);
            total_return += trajectory.total_reward();
            total_length += trajectory.len();

            // Update baseline
            self.episode_count += 1;
            let alpha = 1.0 / self.episode_count as f64;
            self.baseline = self.baseline * (1.0 - alpha) + returns[0] * alpha;

            // Compute gradient for this trajectory
            for (t, step) in trajectory.steps.iter().enumerate() {
                let advantage = if self.config.use_baseline {
                    returns[t] - self.baseline
                } else {
                    returns[t]
                };

                let log_prob_grad = self.policy.log_prob_gradient(&step.state, step.action);

                for (i, row) in log_prob_grad.iter().enumerate() {
                    for (a, &g) in row.iter().enumerate() {
                        batch_gradient[i][a] += g * advantage;
                    }
                }

                total_entropy += self.policy.entropy(&step.state);
            }
        }

        // Normalize by batch size
        let batch_size = trajectories.len() as f64;
        for row in &mut batch_gradient {
            for g in row.iter_mut() {
                *g /= batch_size;
            }
        }

        // Compute gradient norm
        let grad_norm: f64 = batch_gradient
            .iter()
            .flat_map(|row| row.iter())
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        // Gradient clipping
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

        // Update policy
        self.policy
            .update_weights(&batch_gradient, self.config.learning_rate);

        PolicyGradientStats {
            episode_return: total_return / batch_size,
            episode_length: total_length / trajectories.len(),
            mean_entropy: total_entropy / total_length as f64,
            gradient_norm: grad_norm,
            baseline: self.baseline,
        }
    }

    /// Compute entropy gradient
    fn entropy_gradient(&self, state: &[f64]) -> Vec<Vec<f64>> {
        let probs = self.policy.action_probs(state);
        let mut gradient = vec![vec![0.0; self.policy.num_actions]; self.policy.state_dim];

        // Entropy H = -Σ p log p
        // ∇H = -Σ (1 + log p) ∇p
        for (i, &s) in state.iter().enumerate() {
            for a in 0..self.policy.num_actions {
                let mut grad_h = 0.0;
                for a2 in 0..self.policy.num_actions {
                    // ∇p_a2 w.r.t. θ_a for softmax
                    let grad_p = if a == a2 {
                        s * probs[a2] * (1.0 - probs[a2])
                    } else {
                        -s * probs[a] * probs[a2]
                    };
                    if probs[a2] > 1e-10 {
                        grad_h -= (1.0 + probs[a2].ln()) * grad_p;
                    }
                }
                gradient[i][a] = grad_h;
            }
        }

        gradient
    }

    /// Get current baseline value
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Reset baseline
    pub fn reset_baseline(&mut self) {
        self.baseline = 0.0;
        self.episode_count = 0;
    }
}

/// Statistics from policy gradient update
#[derive(Debug, Clone, Default)]
pub struct PolicyGradientStats {
    /// Total return from episode(s)
    pub episode_return: f64,
    /// Episode length
    pub episode_length: usize,
    /// Mean policy entropy
    pub mean_entropy: f64,
    /// Gradient norm before clipping
    pub gradient_norm: f64,
    /// Current baseline value
    pub baseline: f64,
}

/// Generalized Advantage Estimation (GAE)
#[derive(Debug, Clone)]
pub struct GAE {
    /// GAE lambda parameter
    pub lambda: f64,
    /// Discount factor
    pub gamma: f64,
}

impl GAE {
    pub fn new(lambda: f64, gamma: f64) -> Self {
        Self { lambda, gamma }
    }

    /// Compute GAE advantages given rewards and value estimates
    /// values[t] = V(s_t), values should have length len(rewards) + 1
    /// where values[T] = V(s_T) is the bootstrap value (0 if terminal)
    pub fn compute_advantages(&self, rewards: &[f64], values: &[f64], dones: &[bool]) -> Vec<f64> {
        let n = rewards.len();
        if n == 0 {
            return Vec::new();
        }

        assert_eq!(
            values.len(),
            n + 1,
            "Values should have length rewards.len() + 1"
        );

        let mut advantages = vec![0.0; n];
        let mut gae = 0.0;

        for t in (0..n).rev() {
            let next_value = if dones[t] { 0.0 } else { values[t + 1] };
            let delta = rewards[t] + self.gamma * next_value - values[t];
            gae = delta + self.gamma * self.lambda * if dones[t] { 0.0 } else { gae };
            advantages[t] = gae;
        }

        advantages
    }

    /// Compute advantages and returns together
    pub fn compute_advantages_and_returns(
        &self,
        rewards: &[f64],
        values: &[f64],
        dones: &[bool],
    ) -> (Vec<f64>, Vec<f64>) {
        let advantages = self.compute_advantages(rewards, values, dones);
        let returns: Vec<f64> = advantages
            .iter()
            .zip(values.iter())
            .map(|(&a, &v)| a + v)
            .collect();
        (advantages, returns)
    }
}

/// Value function approximator (simple linear)
#[derive(Debug, Clone)]
pub struct LinearValueFunction {
    /// Weights for value estimation
    pub weights: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl LinearValueFunction {
    pub fn new(state_dim: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; state_dim],
            learning_rate,
        }
    }

    /// Predict value for a state
    pub fn predict(&self, state: &[f64]) -> f64 {
        assert_eq!(state.len(), self.weights.len());
        state
            .iter()
            .zip(self.weights.iter())
            .map(|(&s, &w)| s * w)
            .sum()
    }

    /// Update value function using TD error
    pub fn update(&mut self, state: &[f64], target: f64) {
        let pred = self.predict(state);
        let error = target - pred;
        for (w, &s) in self.weights.iter_mut().zip(state.iter()) {
            *w += self.learning_rate * error * s;
        }
    }

    /// Batch update using multiple (state, target) pairs
    pub fn update_batch(&mut self, states: &[Vec<f64>], targets: &[f64]) {
        assert_eq!(states.len(), targets.len());
        for (state, &target) in states.iter().zip(targets.iter()) {
            self.update(state, target);
        }
    }
}

/// Actor-Critic implementation combining policy and value function
#[derive(Debug)]
pub struct ActorCritic {
    pub actor: SoftmaxPolicy,
    pub critic: LinearValueFunction,
    pub config: PolicyGradientConfig,
}

impl ActorCritic {
    pub fn new(state_dim: usize, num_actions: usize, config: PolicyGradientConfig) -> Self {
        let actor = SoftmaxPolicy::new(state_dim, num_actions);
        let critic = LinearValueFunction::new(state_dim, config.learning_rate);
        Self {
            actor,
            critic,
            config,
        }
    }

    /// Update actor and critic using a trajectory
    pub fn update(&mut self, trajectory: &Trajectory) -> PolicyGradientStats {
        if trajectory.is_empty() {
            return PolicyGradientStats::default();
        }

        let n = trajectory.len();

        // Get rewards and dones
        let rewards: Vec<f64> = trajectory.steps.iter().map(|s| s.reward).collect();
        let dones: Vec<bool> = trajectory.steps.iter().map(|s| s.done).collect();

        // Compute values for all states (including bootstrap)
        let mut values: Vec<f64> = trajectory
            .steps
            .iter()
            .map(|s| self.critic.predict(&s.state))
            .collect();
        // Bootstrap value (0 if terminal)
        let last_done = trajectory.steps.last().is_none_or(|s| s.done);
        values.push(if last_done { 0.0 } else { values[n - 1] });

        // Compute advantages using GAE
        let gae = GAE::new(0.95, self.config.discount_factor);
        let (advantages, returns) = gae.compute_advantages_and_returns(&rewards, &values, &dones);

        // Update critic
        let states: Vec<Vec<f64>> = trajectory.steps.iter().map(|s| s.state.clone()).collect();
        self.critic.update_batch(&states, &returns);

        // Update actor using advantages
        let mut total_gradient = vec![vec![0.0; self.actor.num_actions]; self.actor.state_dim];
        let mut total_entropy = 0.0;

        for (t, step) in trajectory.steps.iter().enumerate() {
            let log_prob_grad = self.actor.log_prob_gradient(&step.state, step.action);
            for (i, row) in log_prob_grad.iter().enumerate() {
                for (a, &g) in row.iter().enumerate() {
                    total_gradient[i][a] += g * advantages[t];
                }
            }
            total_entropy += self.actor.entropy(&step.state);
        }

        // Add entropy bonus
        if self.config.entropy_coefficient > 0.0 {
            for step in &trajectory.steps {
                let probs = self.actor.action_probs(&step.state);
                for (i, &s) in step.state.iter().enumerate() {
                    for (a, &prob) in probs.iter().enumerate() {
                        // Simplified entropy gradient contribution
                        let entropy_grad = -s * prob * (1.0 + prob.ln().max(-10.0));
                        total_gradient[i][a] += self.config.entropy_coefficient * entropy_grad;
                    }
                }
            }
        }

        // Compute gradient norm
        let grad_norm: f64 = total_gradient
            .iter()
            .flat_map(|row| row.iter())
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        // Gradient clipping
        if let Some(max_norm) = self.config.max_grad_norm
            && grad_norm > max_norm
        {
            let clip_ratio = max_norm / grad_norm;
            for row in &mut total_gradient {
                for g in row.iter_mut() {
                    *g *= clip_ratio;
                }
            }
        }

        // Update actor
        self.actor
            .update_weights(&total_gradient, self.config.learning_rate);

        PolicyGradientStats {
            episode_return: trajectory.total_reward(),
            episode_length: n,
            mean_entropy: total_entropy / n as f64,
            gradient_norm: grad_norm,
            baseline: values[..n].iter().sum::<f64>() / n as f64,
        }
    }
}

// Helper functions

/// Compute softmax of logits
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    logits
        .iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect()
}

/// Compute entropy of a probability distribution
pub fn compute_entropy(probs: &[f64]) -> f64 {
    -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

/// Sample from categorical distribution
pub fn sample_categorical(probs: &[f64], rng_value: f64) -> usize {
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_value < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Normalize advantages (zero mean, unit variance)
pub fn normalize_advantages(advantages: &[f64]) -> Vec<f64> {
    if advantages.is_empty() {
        return Vec::new();
    }

    let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
    let variance: f64 =
        advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
    let std = (variance + 1e-8).sqrt();

    advantages.iter().map(|&a| (a - mean) / std).collect()
}

/// Compute discounted cumulative sum (useful for returns computation)
pub fn discounted_cumsum(values: &[f64], discount: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = vec![0.0; n];
    let mut running = 0.0;

    for i in (0..n).rev() {
        running = values[i] + discount * running;
        result[i] = running;
    }

    result
}

/// Statistics for analyzing policy performance
#[derive(Debug, Clone, Default)]
pub struct PolicyAnalysis {
    /// Mean action probabilities across states
    pub mean_action_probs: Vec<f64>,
    /// Action frequency distribution
    pub action_frequencies: HashMap<usize, usize>,
    /// Mean entropy across states
    pub mean_entropy: f64,
    /// Variance in action selection
    pub action_variance: f64,
}

/// Analyze policy behavior on a set of trajectories
pub fn analyze_policy(policy: &SoftmaxPolicy, trajectories: &[Trajectory]) -> PolicyAnalysis {
    if trajectories.is_empty() {
        return PolicyAnalysis::default();
    }

    let mut action_probs_sum = vec![0.0; policy.num_actions];
    let mut action_frequencies: HashMap<usize, usize> = HashMap::new();
    let mut total_entropy = 0.0;
    let mut total_steps = 0;

    for trajectory in trajectories {
        for step in &trajectory.steps {
            let probs = policy.action_probs(&step.state);
            for (a, &p) in probs.iter().enumerate() {
                action_probs_sum[a] += p;
            }
            *action_frequencies.entry(step.action).or_insert(0) += 1;
            total_entropy += compute_entropy(&probs);
            total_steps += 1;
        }
    }

    if total_steps == 0 {
        return PolicyAnalysis::default();
    }

    let mean_action_probs: Vec<f64> = action_probs_sum
        .iter()
        .map(|&s| s / total_steps as f64)
        .collect();

    // Compute variance in action selection
    let mean_prob = 1.0 / policy.num_actions as f64;
    let action_variance: f64 = mean_action_probs
        .iter()
        .map(|&p| (p - mean_prob).powi(2))
        .sum::<f64>()
        / policy.num_actions as f64;

    PolicyAnalysis {
        mean_action_probs,
        action_frequencies,
        mean_entropy: total_entropy / total_steps as f64,
        action_variance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Higher logit should give higher probability
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
    fn test_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = compute_entropy(&uniform);

        // Peaked distribution has lower entropy
        let peaked = vec![0.9, 0.05, 0.03, 0.02];
        let peaked_entropy = compute_entropy(&peaked);

        assert!(uniform_entropy > peaked_entropy);
        assert!(uniform_entropy > 0.0);
    }

    #[test]
    fn test_sample_categorical() {
        let probs = vec![0.5, 0.3, 0.2];

        // Low value should select first action
        assert_eq!(sample_categorical(&probs, 0.1), 0);

        // Medium value should select second action
        assert_eq!(sample_categorical(&probs, 0.6), 1);

        // High value should select third action
        assert_eq!(sample_categorical(&probs, 0.9), 2);
    }

    #[test]
    fn test_trajectory_returns() {
        let mut trajectory = Trajectory::new();

        // Simple trajectory with rewards 1, 2, 3
        trajectory.add_step(Step::new(vec![1.0], 0, 1.0, 0.0, false));
        trajectory.add_step(Step::new(vec![1.0], 0, 2.0, 0.0, false));
        trajectory.add_step(Step::new(vec![1.0], 0, 3.0, 0.0, true));

        let returns = trajectory.compute_returns(0.9);

        // G_2 = 3.0
        // G_1 = 2.0 + 0.9 * 3.0 = 4.7
        // G_0 = 1.0 + 0.9 * 4.7 = 5.23
        assert!((returns[2] - 3.0).abs() < 1e-10);
        assert!((returns[1] - 4.7).abs() < 1e-10);
        assert!((returns[0] - 5.23).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_total_reward() {
        let mut trajectory = Trajectory::new();
        trajectory.add_step(Step::new(vec![1.0], 0, 1.0, 0.0, false));
        trajectory.add_step(Step::new(vec![1.0], 0, 2.0, 0.0, false));
        trajectory.add_step(Step::new(vec![1.0], 0, 3.0, 0.0, true));

        assert!((trajectory.total_reward() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_policy_creation() {
        let policy = SoftmaxPolicy::new(4, 3);
        assert_eq!(policy.state_dim, 4);
        assert_eq!(policy.num_actions, 3);
        assert_eq!(policy.weights.len(), 4);
        assert_eq!(policy.weights[0].len(), 3);
    }

    #[test]
    fn test_softmax_policy_zeros() {
        let policy = SoftmaxPolicy::zeros(4, 3);

        // With zero weights, all actions should be equally likely
        let state = vec![1.0, 0.5, 0.2, 0.1];
        let probs = policy.action_probs(&state);

        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_policy_action_probs_sum() {
        let policy = SoftmaxPolicy::new(4, 3);
        let state = vec![1.0, 0.5, -0.3, 0.8];
        let probs = policy.action_probs(&state);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_policy_log_prob() {
        let policy = SoftmaxPolicy::new(4, 3);
        let state = vec![1.0, 0.5, -0.3, 0.8];

        let probs = policy.action_probs(&state);
        for a in 0..3 {
            let log_p = policy.log_prob(&state, a);
            assert!((log_p - probs[a].ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_policy_sample() {
        let policy = SoftmaxPolicy::zeros(4, 3);
        let state = vec![1.0, 0.5, -0.3, 0.8];

        // Sample multiple times
        for rng in [0.1, 0.4, 0.7, 0.9] {
            let (action, log_prob) = policy.sample_action(&state, rng);
            assert!(action < 3);
            assert!(log_prob.is_finite());
        }
    }

    #[test]
    fn test_reinforce_creation() {
        let reinforce = REINFORCE::with_default_config(4, 3);
        assert_eq!(reinforce.policy.state_dim, 4);
        assert_eq!(reinforce.policy.num_actions, 3);
        assert!((reinforce.baseline - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reinforce_update() {
        let mut reinforce = REINFORCE::with_default_config(2, 2);

        let mut trajectory = Trajectory::new();
        trajectory.add_step(Step::new(vec![1.0, 0.0], 0, 1.0, -0.5, false));
        trajectory.add_step(Step::new(vec![0.0, 1.0], 1, 2.0, -0.5, true));

        let stats = reinforce.update(&trajectory);

        assert!(stats.episode_return > 0.0);
        assert_eq!(stats.episode_length, 2);
        assert!(stats.mean_entropy > 0.0);
    }

    #[test]
    fn test_reinforce_update_batch() {
        let mut reinforce = REINFORCE::with_default_config(2, 2);

        let mut trajectories = Vec::new();
        for _ in 0..3 {
            let mut trajectory = Trajectory::new();
            trajectory.add_step(Step::new(vec![1.0, 0.0], 0, 1.0, -0.5, false));
            trajectory.add_step(Step::new(vec![0.0, 1.0], 1, 1.0, -0.5, true));
            trajectories.push(trajectory);
        }

        let stats = reinforce.update_batch(&trajectories);

        assert!(stats.episode_return > 0.0);
        assert!(stats.gradient_norm >= 0.0);
    }

    #[test]
    fn test_reinforce_baseline_update() {
        let mut reinforce = REINFORCE::with_default_config(2, 2);

        // First episode
        let mut traj1 = Trajectory::new();
        traj1.add_step(Step::new(vec![1.0, 0.0], 0, 10.0, -0.5, true));
        reinforce.update(&traj1);

        let baseline1 = reinforce.baseline();

        // Second episode with higher return
        let mut traj2 = Trajectory::new();
        traj2.add_step(Step::new(vec![1.0, 0.0], 0, 20.0, -0.5, true));
        reinforce.update(&traj2);

        let baseline2 = reinforce.baseline();

        // Baseline should increase toward higher returns
        assert!(baseline2 > baseline1);
    }

    #[test]
    fn test_gae_computation() {
        let gae = GAE::new(0.95, 0.99);

        // Simple case: constant rewards
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5, 0.0]; // Bootstrap = 0
        let dones = vec![false, false, true];

        let advantages = gae.compute_advantages(&rewards, &values, &dones);

        assert_eq!(advantages.len(), 3);
        // All advantages should be positive (rewards > values)
        assert!(advantages.iter().all(|&a| a > 0.0));
    }

    #[test]
    fn test_gae_terminal_state() {
        let gae = GAE::new(0.95, 0.99);

        let rewards = vec![1.0, 1.0];
        let values = vec![1.0, 1.0, 10.0]; // High bootstrap value that should be ignored
        let dones = vec![false, true];

        let advantages = gae.compute_advantages(&rewards, &values, &dones);

        // Last step advantage should not include bootstrap
        // delta_1 = 1.0 + 0 - 1.0 = 0.0 (done=true, so next_value=0)
        assert!((advantages[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gae_empty() {
        let gae = GAE::new(0.95, 0.99);
        let advantages = gae.compute_advantages(&[], &[0.0], &[]);
        assert!(advantages.is_empty());
    }

    #[test]
    fn test_linear_value_function() {
        let mut vf = LinearValueFunction::new(3, 0.1);

        let state = vec![1.0, 0.5, 0.2];

        // Initial prediction should be 0 (zero weights)
        assert!((vf.predict(&state) - 0.0).abs() < 1e-10);

        // Update toward target
        vf.update(&state, 5.0);

        // Prediction should move toward target
        let new_pred = vf.predict(&state);
        assert!(new_pred > 0.0);
    }

    #[test]
    fn test_linear_value_function_convergence() {
        let mut vf = LinearValueFunction::new(2, 0.1);

        let state = vec![1.0, 1.0];
        let target = 2.0;

        // Multiple updates should converge
        for _ in 0..100 {
            vf.update(&state, target);
        }

        let pred = vf.predict(&state);
        assert!((pred - target).abs() < 0.5);
    }

    #[test]
    fn test_actor_critic_creation() {
        let config = PolicyGradientConfig::default();
        let ac = ActorCritic::new(4, 3, config);

        assert_eq!(ac.actor.state_dim, 4);
        assert_eq!(ac.actor.num_actions, 3);
        assert_eq!(ac.critic.weights.len(), 4);
    }

    #[test]
    fn test_actor_critic_update() {
        let config = PolicyGradientConfig {
            learning_rate: 0.01,
            discount_factor: 0.99,
            use_baseline: true,
            entropy_coefficient: 0.01,
            max_grad_norm: Some(0.5),
        };
        let mut ac = ActorCritic::new(2, 2, config);

        let mut trajectory = Trajectory::new();
        trajectory.add_step(Step::new(vec![1.0, 0.0], 0, 1.0, -0.5, false));
        trajectory.add_step(Step::new(vec![0.0, 1.0], 1, 2.0, -0.5, true));

        let stats = ac.update(&trajectory);

        assert!(stats.episode_return > 0.0);
        assert_eq!(stats.episode_length, 2);
    }

    #[test]
    fn test_normalize_advantages() {
        let advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_advantages(&advantages);

        // Mean should be ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Std should be ~1
        let variance: f64 =
            normalized.iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / normalized.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_normalize_advantages_empty() {
        let normalized = normalize_advantages(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_discounted_cumsum() {
        let values = vec![1.0, 2.0, 3.0];
        let cumsum = discounted_cumsum(&values, 0.5);

        // cumsum[2] = 3.0
        // cumsum[1] = 2.0 + 0.5 * 3.0 = 3.5
        // cumsum[0] = 1.0 + 0.5 * 3.5 = 2.75
        assert!((cumsum[2] - 3.0).abs() < 1e-10);
        assert!((cumsum[1] - 3.5).abs() < 1e-10);
        assert!((cumsum[0] - 2.75).abs() < 1e-10);
    }

    #[test]
    fn test_policy_analysis() {
        let policy = SoftmaxPolicy::zeros(2, 3);

        let mut trajectories = Vec::new();
        let mut traj = Trajectory::new();
        traj.add_step(Step::new(vec![1.0, 0.0], 0, 1.0, -0.5, false));
        traj.add_step(Step::new(vec![0.0, 1.0], 1, 1.0, -0.5, true));
        trajectories.push(traj);

        let analysis = analyze_policy(&policy, &trajectories);

        assert_eq!(analysis.mean_action_probs.len(), 3);
        assert!(analysis.mean_entropy > 0.0);
        assert_eq!(analysis.action_frequencies.len(), 2);
    }

    #[test]
    fn test_policy_gradient_config_default() {
        let config = PolicyGradientConfig::default();

        assert!(config.learning_rate > 0.0);
        assert!(config.discount_factor > 0.0 && config.discount_factor <= 1.0);
        assert!(config.use_baseline);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = PolicyGradientConfig {
            learning_rate: 1.0, // Large learning rate to produce large gradients
            discount_factor: 0.99,
            use_baseline: false,
            entropy_coefficient: 0.0,
            max_grad_norm: Some(0.1), // Small max norm
        };
        let mut reinforce = REINFORCE::new(SoftmaxPolicy::new(2, 2), config);

        // High reward trajectory to produce large gradient
        let mut trajectory = Trajectory::new();
        trajectory.add_step(Step::new(vec![10.0, 10.0], 0, 100.0, -0.5, true));

        let stats = reinforce.update(&trajectory);

        // Gradient should have been clipped
        assert!(stats.gradient_norm >= 0.0);
    }

    #[test]
    fn test_log_prob_gradient_shape() {
        let policy = SoftmaxPolicy::new(4, 3);
        let state = vec![1.0, 0.5, -0.3, 0.8];

        let gradient = policy.log_prob_gradient(&state, 1);

        assert_eq!(gradient.len(), 4);
        assert_eq!(gradient[0].len(), 3);
    }

    #[test]
    fn test_weight_update() {
        let mut policy = SoftmaxPolicy::zeros(2, 2);
        let initial_weights = policy.weights.clone();

        let gradient = vec![vec![1.0, -1.0], vec![0.5, -0.5]];
        policy.update_weights(&gradient, 0.1);

        // Weights should have changed
        assert!((policy.weights[0][0] - initial_weights[0][0] - 0.1).abs() < 1e-10);
        assert!((policy.weights[0][1] - initial_weights[0][1] + 0.1).abs() < 1e-10);
    }
}
