// Q-Learning
// Classic value-based reinforcement learning algorithm
//
// Q-Learning learns an action-value function Q(s,a) that estimates
// the expected cumulative reward from taking action a in state s.
//
// Update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
//
// Key features:
// - Off-policy: learns optimal policy while following exploratory policy
// - Model-free: doesn't require environment dynamics
// - Tabular: works with discrete state/action spaces
//
// References:
// - "Learning from Delayed Rewards" (Watkins, 1989)
// - "Q-Learning" (Watkins & Dayan, 1992)
// - Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)

use std::collections::HashMap;

/// Configuration for Q-Learning
#[derive(Debug, Clone)]
pub struct QLearningConfig {
    /// Learning rate (alpha)
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Initial Q-value for unseen state-action pairs
    pub initial_q_value: f64,
    /// Exploration rate for epsilon-greedy (epsilon)
    pub epsilon: f64,
    /// Minimum epsilon for decay
    pub epsilon_min: f64,
    /// Epsilon decay rate per episode
    pub epsilon_decay: f64,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            discount_factor: 0.99,
            initial_q_value: 0.0,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
        }
    }
}

impl QLearningConfig {
    /// Config optimized for fast learning
    pub fn fast_learning() -> Self {
        Self {
            learning_rate: 0.5,
            discount_factor: 0.95,
            initial_q_value: 0.0,
            epsilon: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.99,
        }
    }

    /// Config optimized for stable learning
    pub fn stable() -> Self {
        Self {
            learning_rate: 0.01,
            discount_factor: 0.99,
            initial_q_value: 0.0,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.999,
        }
    }

    /// Config with optimistic initialization
    pub fn optimistic(initial_value: f64) -> Self {
        Self {
            initial_q_value: initial_value,
            epsilon: 0.1, // Less exploration needed with optimistic init
            ..Default::default()
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_discount_factor(mut self, gamma: f64) -> Self {
        self.discount_factor = gamma;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// State representation for tabular Q-learning
/// Can be a discrete state index or a discretized continuous state
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiscreteState(pub Vec<i64>);

impl DiscreteState {
    pub fn new(values: Vec<i64>) -> Self {
        Self(values)
    }

    pub fn from_index(index: usize) -> Self {
        Self(vec![index as i64])
    }

    pub fn from_continuous(values: &[f64], bins: &[usize], ranges: &[(f64, f64)]) -> Self {
        let discretized: Vec<i64> = values
            .iter()
            .zip(bins.iter())
            .zip(ranges.iter())
            .map(|((&v, &num_bins), &(min, max))| {
                let normalized = (v - min) / (max - min);
                let bin = (normalized * num_bins as f64).floor() as i64;
                bin.clamp(0, num_bins as i64 - 1)
            })
            .collect();
        Self(discretized)
    }
}

/// Q-Table: maps (state, action) pairs to Q-values
#[derive(Debug, Clone)]
pub struct QTable {
    /// Q-values stored as state -> action -> value
    values: HashMap<DiscreteState, Vec<f64>>,
    /// Number of actions
    num_actions: usize,
    /// Default Q-value for unseen pairs
    default_value: f64,
}

impl QTable {
    pub fn new(num_actions: usize, default_value: f64) -> Self {
        Self {
            values: HashMap::new(),
            num_actions,
            default_value,
        }
    }

    /// Get Q-value for state-action pair
    pub fn get(&self, state: &DiscreteState, action: usize) -> f64 {
        self.values
            .get(state)
            .and_then(|actions| actions.get(action).copied())
            .unwrap_or(self.default_value)
    }

    /// Set Q-value for state-action pair
    pub fn set(&mut self, state: &DiscreteState, action: usize, value: f64) {
        let actions = self
            .values
            .entry(state.clone())
            .or_insert_with(|| vec![self.default_value; self.num_actions]);
        if action < actions.len() {
            actions[action] = value;
        }
    }

    /// Get all Q-values for a state
    pub fn get_state_values(&self, state: &DiscreteState) -> Vec<f64> {
        self.values
            .get(state)
            .cloned()
            .unwrap_or_else(|| vec![self.default_value; self.num_actions])
    }

    /// Get maximum Q-value for a state
    pub fn max_value(&self, state: &DiscreteState) -> f64 {
        self.get_state_values(state)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get action with maximum Q-value (greedy)
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        let values = self.get_state_values(state);
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get number of states in table
    pub fn num_states(&self) -> usize {
        self.values.len()
    }

    /// Get number of actions
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Clear all Q-values
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Get all states in the table
    pub fn states(&self) -> impl Iterator<Item = &DiscreteState> {
        self.values.keys()
    }
}

/// Tabular Q-Learning agent
#[derive(Debug, Clone)]
pub struct QLearning {
    /// Q-table
    pub q_table: QTable,
    /// Configuration
    pub config: QLearningConfig,
    /// Current epsilon (for decay)
    pub epsilon: f64,
    /// Number of updates performed
    pub update_count: usize,
}

impl QLearning {
    pub fn new(num_actions: usize, config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let q_table = QTable::new(num_actions, config.initial_q_value);
        Self {
            q_table,
            config,
            epsilon,
            update_count: 0,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            // Random exploration
            (rng_value / self.epsilon * self.q_table.num_actions() as f64).floor() as usize
                % self.q_table.num_actions()
        } else {
            // Greedy exploitation
            self.q_table.best_action(state)
        }
    }

    /// Select best action (greedy)
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        self.q_table.best_action(state)
    }

    /// Perform Q-learning update
    /// Returns the TD error
    pub fn update(
        &mut self,
        state: &DiscreteState,
        action: usize,
        reward: f64,
        next_state: &DiscreteState,
        done: bool,
    ) -> f64 {
        let current_q = self.q_table.get(state, action);

        // Target: r + γ max_a' Q(s', a') if not done, else just r
        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.q_table.max_value(next_state)
        };

        // TD error
        let td_error = target - current_q;

        // Update Q-value
        let new_q = current_q + self.config.learning_rate * td_error;
        self.q_table.set(state, action, new_q);

        self.update_count += 1;

        td_error
    }

    /// Decay epsilon after episode
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// Get current Q-value
    pub fn get_q_value(&self, state: &DiscreteState, action: usize) -> f64 {
        self.q_table.get(state, action)
    }

    /// Get state value V(s) = max_a Q(s,a)
    pub fn get_state_value(&self, state: &DiscreteState) -> f64 {
        self.q_table.max_value(state)
    }

    /// Reset epsilon to initial value
    pub fn reset_epsilon(&mut self) {
        self.epsilon = self.config.epsilon;
    }

    /// Get action probabilities (epsilon-greedy)
    pub fn action_probs(&self, state: &DiscreteState) -> Vec<f64> {
        let num_actions = self.q_table.num_actions();
        let best = self.q_table.best_action(state);

        let mut probs = vec![self.epsilon / num_actions as f64; num_actions];
        probs[best] += 1.0 - self.epsilon;
        probs
    }
}

/// SARSA: On-policy TD control
/// Update rule: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
#[derive(Debug, Clone)]
pub struct SARSA {
    /// Q-table
    pub q_table: QTable,
    /// Configuration
    pub config: QLearningConfig,
    /// Current epsilon
    pub epsilon: f64,
    /// Update count
    pub update_count: usize,
}

impl SARSA {
    pub fn new(num_actions: usize, config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let q_table = QTable::new(num_actions, config.initial_q_value);
        Self {
            q_table,
            config,
            epsilon,
            update_count: 0,
        }
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_table.num_actions() as f64).floor() as usize
                % self.q_table.num_actions()
        } else {
            self.q_table.best_action(state)
        }
    }

    /// SARSA update (on-policy)
    pub fn update(
        &mut self,
        state: &DiscreteState,
        action: usize,
        reward: f64,
        next_state: &DiscreteState,
        next_action: usize,
        done: bool,
    ) -> f64 {
        let current_q = self.q_table.get(state, action);

        // Target uses actual next action (on-policy)
        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.q_table.get(next_state, next_action)
        };

        let td_error = target - current_q;
        let new_q = current_q + self.config.learning_rate * td_error;
        self.q_table.set(state, action, new_q);

        self.update_count += 1;
        td_error
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// Get best action
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        self.q_table.best_action(state)
    }
}

/// Expected SARSA: Uses expected value over all actions
/// Update: Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',a')] - Q(s,a)]
#[derive(Debug, Clone)]
pub struct ExpectedSARSA {
    /// Q-table
    pub q_table: QTable,
    /// Configuration
    pub config: QLearningConfig,
    /// Current epsilon
    pub epsilon: f64,
    /// Update count
    pub update_count: usize,
}

impl ExpectedSARSA {
    pub fn new(num_actions: usize, config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let q_table = QTable::new(num_actions, config.initial_q_value);
        Self {
            q_table,
            config,
            epsilon,
            update_count: 0,
        }
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_table.num_actions() as f64).floor() as usize
                % self.q_table.num_actions()
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Compute expected Q-value under epsilon-greedy policy
    fn expected_q(&self, state: &DiscreteState) -> f64 {
        let q_values = self.q_table.get_state_values(state);
        let num_actions = q_values.len();
        let best_action = self.q_table.best_action(state);

        let explore_prob = self.epsilon / num_actions as f64;
        let exploit_prob = 1.0 - self.epsilon + explore_prob;

        q_values
            .iter()
            .enumerate()
            .map(|(a, &q)| {
                let prob = if a == best_action {
                    exploit_prob
                } else {
                    explore_prob
                };
                prob * q
            })
            .sum()
    }

    /// Expected SARSA update
    pub fn update(
        &mut self,
        state: &DiscreteState,
        action: usize,
        reward: f64,
        next_state: &DiscreteState,
        done: bool,
    ) -> f64 {
        let current_q = self.q_table.get(state, action);

        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.expected_q(next_state)
        };

        let td_error = target - current_q;
        let new_q = current_q + self.config.learning_rate * td_error;
        self.q_table.set(state, action, new_q);

        self.update_count += 1;
        td_error
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// Get best action
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        self.q_table.best_action(state)
    }
}

/// Double Q-Learning: Reduces overestimation bias
/// Maintains two Q-tables and randomly selects which to update
#[derive(Debug, Clone)]
pub struct DoubleQLearning {
    /// First Q-table
    pub q_table_a: QTable,
    /// Second Q-table
    pub q_table_b: QTable,
    /// Configuration
    pub config: QLearningConfig,
    /// Current epsilon
    pub epsilon: f64,
    /// Update count
    pub update_count: usize,
}

impl DoubleQLearning {
    pub fn new(num_actions: usize, config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let q_table_a = QTable::new(num_actions, config.initial_q_value);
        let q_table_b = QTable::new(num_actions, config.initial_q_value);
        Self {
            q_table_a,
            q_table_b,
            config,
            epsilon,
            update_count: 0,
        }
    }

    /// Get combined Q-value (average of both tables)
    pub fn get_q_value(&self, state: &DiscreteState, action: usize) -> f64 {
        (self.q_table_a.get(state, action) + self.q_table_b.get(state, action)) / 2.0
    }

    /// Get best action using combined Q-values
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        let num_actions = self.q_table_a.num_actions();
        (0..num_actions)
            .max_by(|&a, &b| {
                let qa = self.get_q_value(state, a);
                let qb = self.get_q_value(state, b);
                qa.partial_cmp(&qb).unwrap()
            })
            .unwrap_or(0)
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_table_a.num_actions() as f64).floor() as usize
                % self.q_table_a.num_actions()
        } else {
            self.best_action(state)
        }
    }

    /// Double Q-Learning update
    /// update_a: if true, update table A using B for target; else vice versa
    pub fn update(
        &mut self,
        state: &DiscreteState,
        action: usize,
        reward: f64,
        next_state: &DiscreteState,
        done: bool,
        update_a: bool,
    ) -> f64 {
        let td_error = if update_a {
            // Update Q_A, use Q_A to select action, Q_B to evaluate
            let current_q = self.q_table_a.get(state, action);
            let best_action_a = self.q_table_a.best_action(next_state);

            let target = if done {
                reward
            } else {
                reward + self.config.discount_factor * self.q_table_b.get(next_state, best_action_a)
            };

            let td_error = target - current_q;
            let new_q = current_q + self.config.learning_rate * td_error;
            self.q_table_a.set(state, action, new_q);
            td_error
        } else {
            // Update Q_B, use Q_B to select action, Q_A to evaluate
            let current_q = self.q_table_b.get(state, action);
            let best_action_b = self.q_table_b.best_action(next_state);

            let target = if done {
                reward
            } else {
                reward + self.config.discount_factor * self.q_table_a.get(next_state, best_action_b)
            };

            let td_error = target - current_q;
            let new_q = current_q + self.config.learning_rate * td_error;
            self.q_table_b.set(state, action, new_q);
            td_error
        };

        self.update_count += 1;
        td_error
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }
}

/// Statistics from Q-Learning training
#[derive(Debug, Clone, Default)]
pub struct QLearningStats {
    /// Total reward in episode
    pub episode_reward: f64,
    /// Episode length
    pub episode_length: usize,
    /// Mean TD error
    pub mean_td_error: f64,
    /// Max TD error
    pub max_td_error: f64,
    /// Current epsilon
    pub epsilon: f64,
    /// Number of states visited
    pub states_visited: usize,
}

/// Run Q-learning episode and return stats
pub fn run_episode<S>(
    agent: &mut QLearning,
    initial_state: S,
    step_fn: impl Fn(&S, usize) -> (S, f64, bool),
    state_to_discrete: impl Fn(&S) -> DiscreteState,
    max_steps: usize,
    rng_values: &[f64],
) -> QLearningStats
where
    S: Clone,
{
    let mut state = initial_state;
    let mut total_reward = 0.0;
    let mut td_errors = Vec::new();
    let mut steps = 0;

    for &rng in rng_values.iter().take(max_steps) {
        let discrete_state = state_to_discrete(&state);
        let action = agent.select_action(&discrete_state, rng);

        let (next_state, reward, done) = step_fn(&state, action);
        let discrete_next = state_to_discrete(&next_state);

        let td_error = agent.update(&discrete_state, action, reward, &discrete_next, done);
        td_errors.push(td_error.abs());

        total_reward += reward;
        steps += 1;
        state = next_state;

        if done {
            break;
        }
    }

    agent.decay_epsilon();

    let mean_td = if td_errors.is_empty() {
        0.0
    } else {
        td_errors.iter().sum::<f64>() / td_errors.len() as f64
    };

    let max_td = td_errors.iter().cloned().fold(0.0, f64::max);

    QLearningStats {
        episode_reward: total_reward,
        episode_length: steps,
        mean_td_error: mean_td,
        max_td_error: max_td,
        epsilon: agent.epsilon,
        states_visited: agent.q_table.num_states(),
    }
}

/// Analysis of Q-learning training
#[derive(Debug, Clone, Default)]
pub struct QLearningAnalysis {
    /// Mean episode reward
    pub mean_reward: f64,
    /// Reward trend (slope)
    pub reward_trend: f64,
    /// Mean episode length
    pub mean_length: f64,
    /// Mean TD error
    pub mean_td_error: f64,
    /// TD error trend
    pub td_error_trend: f64,
    /// Is learning converging
    pub is_converging: bool,
    /// Final epsilon
    pub final_epsilon: f64,
}

/// Analyze Q-learning training progress
pub fn analyze_q_learning(stats_history: &[QLearningStats]) -> QLearningAnalysis {
    if stats_history.is_empty() {
        return QLearningAnalysis::default();
    }

    let n = stats_history.len();

    let mean_reward: f64 = stats_history.iter().map(|s| s.episode_reward).sum::<f64>() / n as f64;
    let mean_length: f64 = stats_history
        .iter()
        .map(|s| s.episode_length)
        .sum::<usize>() as f64
        / n as f64;
    let mean_td_error: f64 = stats_history.iter().map(|s| s.mean_td_error).sum::<f64>() / n as f64;

    let reward_trend = if n > 1 {
        let rewards: Vec<f64> = stats_history.iter().map(|s| s.episode_reward).collect();
        compute_trend(&rewards)
    } else {
        0.0
    };

    let td_error_trend = if n > 1 {
        let errors: Vec<f64> = stats_history.iter().map(|s| s.mean_td_error).collect();
        compute_trend(&errors)
    } else {
        0.0
    };

    // Converging if TD error is decreasing and reward is increasing
    let is_converging = td_error_trend < 0.0 && reward_trend > 0.0;

    let final_epsilon = stats_history.last().map(|s| s.epsilon).unwrap_or(1.0);

    QLearningAnalysis {
        mean_reward,
        reward_trend,
        mean_length,
        mean_td_error,
        td_error_trend,
        is_converging,
        final_epsilon,
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

/// N-step Q-Learning
/// Uses n-step returns for potentially faster learning
#[derive(Debug, Clone)]
pub struct NStepQLearning {
    /// Q-table
    pub q_table: QTable,
    /// Configuration
    pub config: QLearningConfig,
    /// Number of steps
    pub n_steps: usize,
    /// Current epsilon
    pub epsilon: f64,
    /// Buffer for n-step transitions
    buffer: Vec<(DiscreteState, usize, f64)>,
}

impl NStepQLearning {
    pub fn new(num_actions: usize, config: QLearningConfig, n_steps: usize) -> Self {
        let epsilon = config.epsilon;
        let q_table = QTable::new(num_actions, config.initial_q_value);
        Self {
            q_table,
            config,
            n_steps,
            epsilon,
            buffer: Vec::new(),
        }
    }

    /// Add transition to buffer
    pub fn add_transition(&mut self, state: DiscreteState, action: usize, reward: f64) {
        self.buffer.push((state, action, reward));
    }

    /// Perform n-step update when episode ends or buffer is full
    pub fn update(&mut self, final_state: &DiscreteState, done: bool) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        // Compute n-step return
        let gamma = self.config.discount_factor;
        let mut g = if done {
            0.0
        } else {
            self.q_table.max_value(final_state)
        };

        let mut total_td_error = 0.0;

        // Process buffer backwards
        for (state, action, reward) in self.buffer.drain(..).rev() {
            g = reward + gamma * g;
            let current_q = self.q_table.get(&state, action);
            let td_error = g - current_q;
            let new_q = current_q + self.config.learning_rate * td_error;
            self.q_table.set(&state, action, new_q);
            total_td_error += td_error.abs();
        }

        total_td_error
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_table.num_actions() as f64).floor() as usize
                % self.q_table.num_actions()
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// Get best action
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        self.q_table.best_action(state)
    }

    /// Check if buffer is full
    pub fn buffer_full(&self) -> bool {
        self.buffer.len() >= self.n_steps
    }

    /// Clear buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }
}

/// Eligibility traces for TD(λ)
#[derive(Debug, Clone)]
pub struct EligibilityTraces {
    /// Traces: state -> action -> trace value
    traces: HashMap<DiscreteState, Vec<f64>>,
    /// Number of actions
    num_actions: usize,
    /// Lambda (trace decay)
    lambda: f64,
    /// Gamma (discount factor)
    gamma: f64,
}

impl EligibilityTraces {
    pub fn new(num_actions: usize, lambda: f64, gamma: f64) -> Self {
        Self {
            traces: HashMap::new(),
            num_actions,
            lambda,
            gamma,
        }
    }

    /// Decay all traces
    pub fn decay(&mut self) {
        let decay = self.gamma * self.lambda;
        for trace_vec in self.traces.values_mut() {
            for t in trace_vec.iter_mut() {
                *t *= decay;
            }
        }
    }

    /// Update trace for state-action pair (replacing traces)
    pub fn update(&mut self, state: &DiscreteState, action: usize) {
        // First decay all traces
        self.decay();

        // Then set current trace to 1
        let traces = self
            .traces
            .entry(state.clone())
            .or_insert_with(|| vec![0.0; self.num_actions]);
        traces[action] = 1.0;
    }

    /// Get trace value
    pub fn get(&self, state: &DiscreteState, action: usize) -> f64 {
        self.traces
            .get(state)
            .and_then(|t| t.get(action).copied())
            .unwrap_or(0.0)
    }

    /// Reset all traces
    pub fn reset(&mut self) {
        self.traces.clear();
    }

    /// Iterate over all non-zero traces
    pub fn iter(&self) -> impl Iterator<Item = (&DiscreteState, usize, f64)> {
        self.traces.iter().flat_map(|(state, traces)| {
            traces
                .iter()
                .enumerate()
                .filter(|&(_, t)| *t > 1e-10)
                .map(move |(a, t)| (state, a, *t))
        })
    }
}

/// Q(λ) with eligibility traces
#[derive(Debug, Clone)]
pub struct QLambda {
    /// Q-table
    pub q_table: QTable,
    /// Eligibility traces
    pub traces: EligibilityTraces,
    /// Configuration
    pub config: QLearningConfig,
    /// Lambda for traces
    pub lambda: f64,
    /// Current epsilon
    pub epsilon: f64,
}

impl QLambda {
    pub fn new(num_actions: usize, config: QLearningConfig, lambda: f64) -> Self {
        let epsilon = config.epsilon;
        let gamma = config.discount_factor;
        let q_table = QTable::new(num_actions, config.initial_q_value);
        let traces = EligibilityTraces::new(num_actions, lambda, gamma);
        Self {
            q_table,
            traces,
            config,
            lambda,
            epsilon,
        }
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &DiscreteState, rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_table.num_actions() as f64).floor() as usize
                % self.q_table.num_actions()
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Q(λ) update
    pub fn update(
        &mut self,
        state: &DiscreteState,
        action: usize,
        reward: f64,
        next_state: &DiscreteState,
        done: bool,
    ) -> f64 {
        // Update trace for current state-action
        self.traces.update(state, action);

        // Compute TD error
        let current_q = self.q_table.get(state, action);
        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.q_table.max_value(next_state)
        };
        let td_error = target - current_q;

        // Update all Q-values using traces
        let updates: Vec<(DiscreteState, usize, f64)> = self
            .traces
            .iter()
            .map(|(s, a, trace)| {
                let old_q = self.q_table.get(s, a);
                let new_q = old_q + self.config.learning_rate * td_error * trace;
                (s.clone(), a, new_q)
            })
            .collect();

        for (s, a, new_q) in updates {
            self.q_table.set(&s, a, new_q);
        }

        if done {
            self.traces.reset();
        }

        td_error
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// Get best action
    pub fn best_action(&self, state: &DiscreteState) -> usize {
        self.q_table.best_action(state)
    }

    /// Reset traces (call at episode start)
    pub fn reset_traces(&mut self) {
        self.traces.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_env() -> impl Fn(&DiscreteState, usize) -> (DiscreteState, f64, bool) {
        // Simple grid: 5 states, 2 actions (left/right)
        // Goal is state 4, start is state 0
        move |state: &DiscreteState, action: usize| {
            let pos = state.0[0];
            let new_pos = if action == 0 {
                (pos - 1).max(0)
            } else {
                (pos + 1).min(4)
            };
            let done = new_pos == 4;
            let reward = if done { 1.0 } else { -0.1 };
            (DiscreteState::from_index(new_pos as usize), reward, done)
        }
    }

    #[test]
    fn test_config_default() {
        let config = QLearningConfig::default();
        assert!((config.learning_rate - 0.1).abs() < 1e-10);
        assert!((config.discount_factor - 0.99).abs() < 1e-10);
        assert!((config.epsilon - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builders() {
        let config = QLearningConfig::fast_learning();
        assert!((config.learning_rate - 0.5).abs() < 1e-10);

        let config = QLearningConfig::stable();
        assert!((config.learning_rate - 0.01).abs() < 1e-10);

        let config = QLearningConfig::optimistic(10.0);
        assert!((config.initial_q_value - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_state() {
        let state = DiscreteState::new(vec![1, 2, 3]);
        assert_eq!(state.0, vec![1, 2, 3]);

        let state = DiscreteState::from_index(5);
        assert_eq!(state.0, vec![5]);
    }

    #[test]
    fn test_discrete_state_from_continuous() {
        let values = [0.5, 0.75];
        let bins = [10, 4];
        let ranges = [(0.0, 1.0), (0.0, 1.0)];

        let state = DiscreteState::from_continuous(&values, &bins, &ranges);
        assert_eq!(state.0[0], 5); // 0.5 * 10 = 5
        assert_eq!(state.0[1], 3); // 0.75 * 4 = 3
    }

    #[test]
    fn test_q_table_basic() {
        let mut table = QTable::new(3, 0.0);
        let state = DiscreteState::from_index(0);

        assert!((table.get(&state, 0) - 0.0).abs() < 1e-10);

        table.set(&state, 1, 5.0);
        assert!((table.get(&state, 1) - 5.0).abs() < 1e-10);
        assert!((table.get(&state, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_table_best_action() {
        let mut table = QTable::new(3, 0.0);
        let state = DiscreteState::from_index(0);

        table.set(&state, 2, 10.0);
        assert_eq!(table.best_action(&state), 2);
    }

    #[test]
    fn test_q_table_max_value() {
        let mut table = QTable::new(3, 0.0);
        let state = DiscreteState::from_index(0);

        table.set(&state, 1, 5.0);
        table.set(&state, 2, 3.0);

        assert!((table.max_value(&state) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_learning_creation() {
        let agent = QLearning::new(4, QLearningConfig::default());
        assert_eq!(agent.q_table.num_actions(), 4);
        assert!((agent.epsilon - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_learning_select_action() {
        let agent = QLearning::new(4, QLearningConfig::default().with_epsilon(0.5));

        // Should be deterministic for same rng
        let state = DiscreteState::from_index(0);
        let action1 = agent.select_action(&state, 0.3);
        let action2 = agent.select_action(&state, 0.3);
        assert_eq!(action1, action2);
    }

    #[test]
    fn test_q_learning_update() {
        let mut agent = QLearning::new(2, QLearningConfig::default().with_learning_rate(0.5));

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        let td_error = agent.update(&state, 1, 1.0, &next_state, false);
        assert!(td_error > 0.0);

        // Q-value should have increased
        assert!(agent.get_q_value(&state, 1) > 0.0);
    }

    #[test]
    fn test_q_learning_terminal_update() {
        let mut agent = QLearning::new(2, QLearningConfig::default().with_learning_rate(0.5));

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        agent.update(&state, 0, 10.0, &next_state, true);

        // Q(s,a) = 0 + 0.5 * (10 - 0) = 5.0
        assert!((agent.get_q_value(&state, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_learning_decay_epsilon() {
        let mut agent = QLearning::new(2, QLearningConfig::default());
        let initial_epsilon = agent.epsilon;

        agent.decay_epsilon();

        assert!(agent.epsilon < initial_epsilon);
    }

    #[test]
    fn test_q_learning_learning() {
        let config = QLearningConfig::fast_learning().with_epsilon(0.3);
        let mut agent = QLearning::new(2, config);
        let env = create_simple_env();

        // Train for several episodes
        for _ in 0..50 {
            let mut state = DiscreteState::from_index(0);
            for step in 0..20 {
                let rng = (step as f64 * 0.1) % 1.0;
                let action = agent.select_action(&state, rng);
                let (next_state, reward, done) = env(&state, action);
                agent.update(&state, action, reward, &next_state, done);
                state = next_state;
                if done {
                    break;
                }
            }
            agent.decay_epsilon();
        }

        // Should learn to go right (action 1)
        let start = DiscreteState::from_index(0);
        assert_eq!(agent.best_action(&start), 1);
    }

    #[test]
    fn test_sarsa_creation() {
        let agent = SARSA::new(4, QLearningConfig::default());
        assert_eq!(agent.q_table.num_actions(), 4);
    }

    #[test]
    fn test_sarsa_update() {
        let mut agent = SARSA::new(2, QLearningConfig::default().with_learning_rate(0.5));

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        let td_error = agent.update(&state, 0, 1.0, &next_state, 1, false);
        assert!(td_error > 0.0);
    }

    #[test]
    fn test_expected_sarsa_creation() {
        let agent = ExpectedSARSA::new(4, QLearningConfig::default());
        assert_eq!(agent.q_table.num_actions(), 4);
    }

    #[test]
    fn test_expected_sarsa_update() {
        let mut agent = ExpectedSARSA::new(2, QLearningConfig::default().with_learning_rate(0.5));

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        let td_error = agent.update(&state, 0, 1.0, &next_state, false);
        assert!(td_error > 0.0);
    }

    #[test]
    fn test_double_q_learning_creation() {
        let agent = DoubleQLearning::new(4, QLearningConfig::default());
        assert_eq!(agent.q_table_a.num_actions(), 4);
        assert_eq!(agent.q_table_b.num_actions(), 4);
    }

    #[test]
    fn test_double_q_learning_update() {
        let mut agent = DoubleQLearning::new(2, QLearningConfig::default().with_learning_rate(0.5));

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        // Update table A
        let td_a = agent.update(&state, 0, 1.0, &next_state, false, true);
        // Update table B
        let td_b = agent.update(&state, 0, 1.0, &next_state, false, false);

        assert!(td_a > 0.0);
        assert!(td_b > 0.0);
    }

    #[test]
    fn test_double_q_learning_combined_value() {
        let mut agent = DoubleQLearning::new(2, QLearningConfig::default());
        let state = DiscreteState::from_index(0);

        agent.q_table_a.set(&state, 0, 4.0);
        agent.q_table_b.set(&state, 0, 6.0);

        assert!((agent.get_q_value(&state, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_n_step_q_learning() {
        let mut agent = NStepQLearning::new(2, QLearningConfig::fast_learning(), 3);

        let state0 = DiscreteState::from_index(0);
        let state1 = DiscreteState::from_index(1);
        let state2 = DiscreteState::from_index(2);

        agent.add_transition(state0, 1, 0.0);
        agent.add_transition(state1, 1, 0.0);
        agent.add_transition(state2.clone(), 1, 1.0);

        assert!(agent.buffer_full());

        let final_state = DiscreteState::from_index(3);
        let td_error = agent.update(&final_state, true);

        assert!(td_error > 0.0);
    }

    #[test]
    fn test_eligibility_traces() {
        let mut traces = EligibilityTraces::new(3, 0.9, 0.99);
        let state = DiscreteState::from_index(0);

        traces.update(&state, 1);
        assert!((traces.get(&state, 1) - 1.0).abs() < 1e-10);

        traces.decay();
        assert!(traces.get(&state, 1) < 1.0);
    }

    #[test]
    fn test_q_lambda() {
        let mut agent = QLambda::new(2, QLearningConfig::fast_learning(), 0.9);

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        let td_error = agent.update(&state, 0, 1.0, &next_state, false);
        assert!(td_error > 0.0);
    }

    #[test]
    fn test_q_lambda_trace_reset() {
        let mut agent = QLambda::new(2, QLearningConfig::default(), 0.9);

        let state = DiscreteState::from_index(0);
        let next_state = DiscreteState::from_index(1);

        agent.update(&state, 0, 1.0, &next_state, true);

        // Traces should be reset after terminal
        assert!(agent.traces.get(&state, 0).abs() < 1e-10);
    }

    #[test]
    fn test_run_episode() {
        let mut agent = QLearning::new(2, QLearningConfig::fast_learning());

        let stats = run_episode(
            &mut agent,
            0usize,
            |&state, action| {
                let new_state = if action == 1 { state + 1 } else { state };
                let done = new_state >= 4;
                let reward = if done { 1.0 } else { -0.1 };
                (new_state.min(4), reward, done)
            },
            |&state| DiscreteState::from_index(state),
            100,
            &[0.6, 0.7, 0.8, 0.9, 0.95, 0.99], // High values = exploit
        );

        assert!(stats.episode_length > 0);
        assert!(stats.states_visited > 0);
    }

    #[test]
    fn test_analyze_q_learning_empty() {
        let analysis = analyze_q_learning(&[]);
        assert!((analysis.mean_reward - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_q_learning() {
        let stats = vec![
            QLearningStats {
                episode_reward: 1.0,
                episode_length: 10,
                mean_td_error: 0.5,
                max_td_error: 1.0,
                epsilon: 0.5,
                states_visited: 5,
            },
            QLearningStats {
                episode_reward: 2.0,
                episode_length: 8,
                mean_td_error: 0.3,
                max_td_error: 0.5,
                epsilon: 0.4,
                states_visited: 6,
            },
            QLearningStats {
                episode_reward: 3.0,
                episode_length: 5,
                mean_td_error: 0.1,
                max_td_error: 0.2,
                epsilon: 0.3,
                states_visited: 7,
            },
        ];

        let analysis = analyze_q_learning(&stats);
        assert!((analysis.mean_reward - 2.0).abs() < 1e-10);
        assert!(analysis.reward_trend > 0.0); // Reward increasing
        assert!(analysis.td_error_trend < 0.0); // TD error decreasing
        assert!(analysis.is_converging);
    }

    #[test]
    fn test_action_probs() {
        let agent = QLearning::new(3, QLearningConfig::default().with_epsilon(0.3));
        let state = DiscreteState::from_index(0);

        let probs = agent.action_probs(&state);
        assert_eq!(probs.len(), 3);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_trend() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(compute_trend(&increasing) > 0.0);

        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(compute_trend(&decreasing) < 0.0);

        let constant = vec![3.0, 3.0, 3.0];
        assert!(compute_trend(&constant).abs() < 1e-10);
    }
}
