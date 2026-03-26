// Deep Q-Network (DQN) Components
// Building blocks for Deep Q-Learning
//
// Key components:
// 1. Experience Replay Buffer - stores and samples transitions
// 2. Target Network - provides stable Q-value targets
// 3. Epsilon-Greedy Exploration - balances exploration/exploitation
// 4. Prioritized Experience Replay - samples important transitions more often
//
// References:
// - "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
// - "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
// - "Prioritized Experience Replay" (Schaul et al., 2015)
// - "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017)

use std::collections::VecDeque;

/// A single transition (s, a, r, s', done)
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current state
    pub state: Vec<f64>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: Vec<f64>,
    /// Whether episode ended
    pub done: bool,
}

impl Transition {
    pub fn new(
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Configuration for experience replay
#[derive(Debug, Clone)]
pub struct ReplayBufferConfig {
    /// Maximum buffer capacity
    pub capacity: usize,
    /// Minimum samples before training can start
    pub min_size: usize,
    /// Batch size for sampling
    pub batch_size: usize,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            min_size: 1000,
            batch_size: 32,
        }
    }
}

impl ReplayBufferConfig {
    pub fn small() -> Self {
        Self {
            capacity: 10_000,
            min_size: 100,
            batch_size: 32,
        }
    }

    pub fn large() -> Self {
        Self {
            capacity: 1_000_000,
            min_size: 10_000,
            batch_size: 64,
        }
    }

    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Experience Replay Buffer
/// Stores transitions and provides uniform random sampling
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    /// Stored transitions
    buffer: VecDeque<Transition>,
    /// Configuration
    config: ReplayBufferConfig,
}

impl ReplayBuffer {
    pub fn new(config: ReplayBufferConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.capacity),
            config,
        }
    }

    /// Add a transition to the buffer
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.config.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    /// Add transition from components
    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.push(Transition::new(state, action, reward, next_state, done));
    }

    /// Check if buffer has enough samples for training
    pub fn can_sample(&self) -> bool {
        self.buffer.len() >= self.config.min_size
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Sample a batch of transitions uniformly
    /// rng_values should have batch_size elements in [0, 1)
    pub fn sample(&self, rng_values: &[f64]) -> Vec<Transition> {
        let n = self.buffer.len();
        if n == 0 {
            return Vec::new();
        }

        rng_values
            .iter()
            .take(self.config.batch_size)
            .map(|&r| {
                let idx = (r * n as f64).floor() as usize % n;
                self.buffer[idx].clone()
            })
            .collect()
    }

    /// Sample indices for a batch
    pub fn sample_indices(&self, rng_values: &[f64]) -> Vec<usize> {
        let n = self.buffer.len();
        if n == 0 {
            return Vec::new();
        }

        rng_values
            .iter()
            .take(self.config.batch_size)
            .map(|&r| (r * n as f64).floor() as usize % n)
            .collect()
    }

    /// Get transition by index
    pub fn get(&self, idx: usize) -> Option<&Transition> {
        self.buffer.get(idx)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new(ReplayBufferConfig::default())
    }
}

/// Prioritized Experience Replay Buffer
/// Samples transitions based on TD-error priority
#[derive(Debug, Clone)]
pub struct PrioritizedReplayBuffer {
    /// Stored transitions with priorities
    buffer: VecDeque<(Transition, f64)>,
    /// Configuration
    config: PrioritizedReplayConfig,
    /// Sum of priorities (for sampling)
    priority_sum: f64,
    /// Maximum priority seen
    max_priority: f64,
}

/// Configuration for prioritized replay
#[derive(Debug, Clone)]
pub struct PrioritizedReplayConfig {
    /// Base replay config
    pub base: ReplayBufferConfig,
    /// Priority exponent (alpha): 0 = uniform, 1 = full prioritization
    pub alpha: f64,
    /// Importance sampling exponent (beta): 0 = no correction, 1 = full correction
    pub beta: f64,
    /// Beta annealing rate per step
    pub beta_annealing: f64,
    /// Small constant to ensure non-zero priority
    pub epsilon: f64,
}

impl Default for PrioritizedReplayConfig {
    fn default() -> Self {
        Self {
            base: ReplayBufferConfig::default(),
            alpha: 0.6,
            beta: 0.4,
            beta_annealing: 1e-6,
            epsilon: 1e-6,
        }
    }
}

impl PrioritizedReplayConfig {
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }
}

impl PrioritizedReplayBuffer {
    pub fn new(config: PrioritizedReplayConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.base.capacity),
            config,
            priority_sum: 0.0,
            max_priority: 1.0,
        }
    }

    /// Add transition with maximum priority
    pub fn push(&mut self, transition: Transition) {
        let priority = self.max_priority.powf(self.config.alpha);

        if self.buffer.len() >= self.config.base.capacity
            && let Some((_, old_priority)) = self.buffer.pop_front()
        {
            self.priority_sum -= old_priority;
        }

        self.priority_sum += priority;
        self.buffer.push_back((transition, priority));
    }

    /// Add transition from components
    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.push(Transition::new(state, action, reward, next_state, done));
    }

    /// Update priority for a transition
    pub fn update_priority(&mut self, idx: usize, td_error: f64) {
        if idx >= self.buffer.len() {
            return;
        }

        let new_priority = (td_error.abs() + self.config.epsilon).powf(self.config.alpha);
        let old_priority = self.buffer[idx].1;

        self.priority_sum = self.priority_sum - old_priority + new_priority;
        self.buffer[idx].1 = new_priority;

        self.max_priority = self.max_priority.max(td_error.abs() + self.config.epsilon);
    }

    /// Update priorities for multiple transitions
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
            self.update_priority(idx, td_error);
        }
    }

    /// Sample based on priorities
    /// Returns (transitions, indices, importance weights)
    pub fn sample(&self, rng_values: &[f64]) -> (Vec<Transition>, Vec<usize>, Vec<f64>) {
        let n = self.buffer.len();
        if n == 0 || self.priority_sum <= 0.0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let batch_size = self.config.base.batch_size.min(n);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        // Segment-based sampling for better coverage
        let segment_size = self.priority_sum / batch_size as f64;

        for (i, &rng) in rng_values.iter().take(batch_size).enumerate() {
            let target = (i as f64 + rng) * segment_size;
            let idx = self.find_priority_index(target);

            let (transition, priority) = &self.buffer[idx];
            transitions.push(transition.clone());
            indices.push(idx);

            // Importance sampling weight
            let prob = priority / self.priority_sum;
            let weight = (n as f64 * prob).powf(-self.config.beta);
            weights.push(weight);
        }

        // Normalize weights
        let max_weight = weights.iter().cloned().fold(0.0, f64::max);
        if max_weight > 0.0 {
            for w in &mut weights {
                *w /= max_weight;
            }
        }

        (transitions, indices, weights)
    }

    /// Find index for priority-based sampling
    fn find_priority_index(&self, target: f64) -> usize {
        let mut cumsum = 0.0;
        for (i, (_, priority)) in self.buffer.iter().enumerate() {
            cumsum += priority;
            if cumsum >= target {
                return i;
            }
        }
        self.buffer.len().saturating_sub(1)
    }

    /// Anneal beta towards 1.0
    pub fn anneal_beta(&mut self) {
        self.config.beta = (self.config.beta + self.config.beta_annealing).min(1.0);
    }

    /// Check if buffer can sample
    pub fn can_sample(&self) -> bool {
        self.buffer.len() >= self.config.base.min_size
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get current beta
    pub fn beta(&self) -> f64 {
        self.config.beta
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.priority_sum = 0.0;
        self.max_priority = 1.0;
    }
}

/// Simple neural network for Q-value approximation
/// Uses a linear layer for simplicity (can be extended)
#[derive(Debug, Clone)]
pub struct QNetwork {
    /// Input dimension (state size)
    pub input_dim: usize,
    /// Number of actions (output dimension)
    pub num_actions: usize,
    /// Weights: [input_dim x num_actions]
    pub weights: Vec<Vec<f64>>,
    /// Biases: [num_actions]
    pub biases: Vec<f64>,
}

impl QNetwork {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_dim + num_actions) as f64).sqrt();
        let weights = (0..input_dim)
            .map(|i| {
                (0..num_actions)
                    .map(|j| {
                        // Deterministic pseudo-random initialization
                        let seed = (i * 31 + j * 17) as f64;
                        (seed.sin() * 43758.5453).fract() * scale
                    })
                    .collect()
            })
            .collect();

        let biases = vec![0.0; num_actions];

        Self {
            input_dim,
            num_actions,
            weights,
            biases,
        }
    }

    /// Compute Q-values for a state
    pub fn forward(&self, state: &[f64]) -> Vec<f64> {
        let mut q_values = self.biases.clone();

        for (i, &s) in state.iter().enumerate().take(self.input_dim) {
            for (a, q) in q_values.iter_mut().enumerate() {
                *q += s * self.weights[i][a];
            }
        }

        q_values
    }

    /// Get best action (argmax Q)
    pub fn best_action(&self, state: &[f64]) -> usize {
        let q_values = self.forward(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get maximum Q-value
    pub fn max_q(&self, state: &[f64]) -> f64 {
        let q_values = self.forward(state);
        q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Copy weights from another network
    pub fn copy_from(&mut self, other: &QNetwork) {
        self.weights = other.weights.clone();
        self.biases = other.biases.clone();
    }

    /// Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
    pub fn soft_update(&mut self, local: &QNetwork, tau: f64) {
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (a, w) in row.iter_mut().enumerate() {
                *w = tau * local.weights[i][a] + (1.0 - tau) * *w;
            }
        }
        for (a, b) in self.biases.iter_mut().enumerate() {
            *b = tau * local.biases[a] + (1.0 - tau) * *b;
        }
    }

    /// Update weights using gradient
    pub fn update(&mut self, gradients: &QNetworkGradient, learning_rate: f64) {
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (a, w) in row.iter_mut().enumerate() {
                *w += learning_rate * gradients.weights[i][a];
            }
        }
        for (a, b) in self.biases.iter_mut().enumerate() {
            *b += learning_rate * gradients.biases[a];
        }
    }
}

/// Gradient for Q-network updates
#[derive(Debug, Clone)]
pub struct QNetworkGradient {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl QNetworkGradient {
    pub fn zeros(input_dim: usize, num_actions: usize) -> Self {
        Self {
            weights: vec![vec![0.0; num_actions]; input_dim],
            biases: vec![0.0; num_actions],
        }
    }

    /// Add gradient contribution from a single sample
    pub fn add_sample(&mut self, state: &[f64], action: usize, td_error: f64) {
        // Gradient of Q(s,a) w.r.t. weights is just the state features
        for (i, &s) in state.iter().enumerate() {
            self.weights[i][action] += td_error * s;
        }
        self.biases[action] += td_error;
    }

    /// Scale gradients
    pub fn scale(&mut self, factor: f64) {
        for row in &mut self.weights {
            for w in row.iter_mut() {
                *w *= factor;
            }
        }
        for b in &mut self.biases {
            *b *= factor;
        }
    }

    /// Clip gradients by norm
    pub fn clip_by_norm(&mut self, max_norm: f64) {
        let norm: f64 = self
            .weights
            .iter()
            .flat_map(|row| row.iter())
            .chain(self.biases.iter())
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        if norm > max_norm {
            let scale = max_norm / norm;
            self.scale(scale);
        }
    }
}

/// DQN Configuration
#[derive(Debug, Clone)]
pub struct DQNConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f64,
    /// Initial exploration rate
    pub epsilon_start: f64,
    /// Final exploration rate
    pub epsilon_end: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Target network update frequency (steps)
    pub target_update_freq: usize,
    /// Soft update coefficient (tau)
    pub tau: f64,
    /// Use soft update instead of hard copy
    pub use_soft_update: bool,
    /// Gradient clipping norm
    pub max_grad_norm: Option<f64>,
    /// Use Double DQN
    pub double_dqn: bool,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_freq: 1000,
            tau: 0.005,
            use_soft_update: true,
            max_grad_norm: Some(10.0),
            double_dqn: true,
        }
    }
}

impl DQNConfig {
    pub fn with_double_dqn(mut self, use_double: bool) -> Self {
        self.double_dqn = use_double;
        self
    }

    pub fn with_soft_update(mut self, tau: f64) -> Self {
        self.use_soft_update = true;
        self.tau = tau;
        self
    }

    pub fn with_hard_update(mut self, freq: usize) -> Self {
        self.use_soft_update = false;
        self.target_update_freq = freq;
        self
    }
}

/// DQN Agent
#[derive(Debug, Clone)]
pub struct DQN {
    /// Online (local) Q-network
    pub q_network: QNetwork,
    /// Target Q-network
    pub target_network: QNetwork,
    /// Experience replay buffer
    pub replay_buffer: ReplayBuffer,
    /// Configuration
    pub config: DQNConfig,
    /// Current epsilon
    pub epsilon: f64,
    /// Step counter
    pub step_count: usize,
}

impl DQN {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        config: DQNConfig,
        buffer_config: ReplayBufferConfig,
    ) -> Self {
        let q_network = QNetwork::new(state_dim, num_actions);
        let target_network = q_network.clone();
        let replay_buffer = ReplayBuffer::new(buffer_config);
        let epsilon = config.epsilon_start;

        Self {
            q_network,
            target_network,
            replay_buffer,
            config,
            epsilon,
            step_count: 0,
        }
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &[f64], rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            // Random action
            (rng_value / self.epsilon * self.q_network.num_actions as f64).floor() as usize
                % self.q_network.num_actions
        } else {
            // Greedy action
            self.q_network.best_action(state)
        }
    }

    /// Store transition in replay buffer
    pub fn store_transition(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.replay_buffer
            .add(state, action, reward, next_state, done);
    }

    /// Perform one training step
    /// Returns mean TD error if training occurred
    pub fn train_step(&mut self, rng_values: &[f64]) -> Option<f64> {
        if !self.replay_buffer.can_sample() {
            return None;
        }

        // Sample batch
        let batch = self.replay_buffer.sample(rng_values);

        // Compute TD errors and gradients
        let mut gradient =
            QNetworkGradient::zeros(self.q_network.input_dim, self.q_network.num_actions);
        let mut total_td_error = 0.0;

        for transition in &batch {
            let td_error = self.compute_td_error(transition);
            gradient.add_sample(&transition.state, transition.action, td_error);
            total_td_error += td_error.abs();
        }

        // Average gradient
        gradient.scale(1.0 / batch.len() as f64);

        // Clip gradient
        if let Some(max_norm) = self.config.max_grad_norm {
            gradient.clip_by_norm(max_norm);
        }

        // Update Q-network
        self.q_network.update(&gradient, self.config.learning_rate);

        // Update target network
        self.step_count += 1;
        if self.config.use_soft_update {
            self.target_network
                .soft_update(&self.q_network, self.config.tau);
        } else if self.step_count % self.config.target_update_freq == 0 {
            self.target_network.copy_from(&self.q_network);
        }

        Some(total_td_error / batch.len() as f64)
    }

    /// Compute TD error for a transition
    fn compute_td_error(&self, transition: &Transition) -> f64 {
        let current_q = self.q_network.forward(&transition.state)[transition.action];

        let target_q = if transition.done {
            transition.reward
        } else if self.config.double_dqn {
            // Double DQN: use online network to select action, target network to evaluate
            let best_action = self.q_network.best_action(&transition.next_state);
            let next_q = self.target_network.forward(&transition.next_state)[best_action];
            transition.reward + self.config.gamma * next_q
        } else {
            // Standard DQN: use target network for both
            let next_q = self.target_network.max_q(&transition.next_state);
            transition.reward + self.config.gamma * next_q
        };

        target_q - current_q
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    /// Get current Q-values for a state
    pub fn get_q_values(&self, state: &[f64]) -> Vec<f64> {
        self.q_network.forward(state)
    }

    /// Get best action
    pub fn best_action(&self, state: &[f64]) -> usize {
        self.q_network.best_action(state)
    }
}

/// DQN with Prioritized Experience Replay
#[derive(Debug, Clone)]
pub struct PrioritizedDQN {
    /// Online Q-network
    pub q_network: QNetwork,
    /// Target Q-network
    pub target_network: QNetwork,
    /// Prioritized replay buffer
    pub replay_buffer: PrioritizedReplayBuffer,
    /// Configuration
    pub config: DQNConfig,
    /// Current epsilon
    pub epsilon: f64,
    /// Step counter
    pub step_count: usize,
}

impl PrioritizedDQN {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        config: DQNConfig,
        buffer_config: PrioritizedReplayConfig,
    ) -> Self {
        let q_network = QNetwork::new(state_dim, num_actions);
        let target_network = q_network.clone();
        let replay_buffer = PrioritizedReplayBuffer::new(buffer_config);
        let epsilon = config.epsilon_start;

        Self {
            q_network,
            target_network,
            replay_buffer,
            config,
            epsilon,
            step_count: 0,
        }
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &[f64], rng_value: f64) -> usize {
        if rng_value < self.epsilon {
            (rng_value / self.epsilon * self.q_network.num_actions as f64).floor() as usize
                % self.q_network.num_actions
        } else {
            self.q_network.best_action(state)
        }
    }

    /// Store transition
    pub fn store_transition(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.replay_buffer
            .add(state, action, reward, next_state, done);
    }

    /// Perform training step with prioritized sampling
    pub fn train_step(&mut self, rng_values: &[f64]) -> Option<f64> {
        if !self.replay_buffer.can_sample() {
            return None;
        }

        // Sample with priorities
        let (batch, indices, weights) = self.replay_buffer.sample(rng_values);

        // Compute TD errors and gradients
        let mut gradient =
            QNetworkGradient::zeros(self.q_network.input_dim, self.q_network.num_actions);
        let mut td_errors = Vec::with_capacity(batch.len());
        let mut total_td_error = 0.0;

        for (i, transition) in batch.iter().enumerate() {
            let td_error = self.compute_td_error(transition);
            td_errors.push(td_error);

            // Weight gradient by importance sampling weight
            let weighted_error = td_error * weights[i];
            gradient.add_sample(&transition.state, transition.action, weighted_error);
            total_td_error += td_error.abs();
        }

        // Update priorities
        self.replay_buffer.update_priorities(&indices, &td_errors);

        // Anneal beta
        self.replay_buffer.anneal_beta();

        // Average gradient
        gradient.scale(1.0 / batch.len() as f64);

        // Clip gradient
        if let Some(max_norm) = self.config.max_grad_norm {
            gradient.clip_by_norm(max_norm);
        }

        // Update Q-network
        self.q_network.update(&gradient, self.config.learning_rate);

        // Update target network
        self.step_count += 1;
        if self.config.use_soft_update {
            self.target_network
                .soft_update(&self.q_network, self.config.tau);
        } else if self.step_count % self.config.target_update_freq == 0 {
            self.target_network.copy_from(&self.q_network);
        }

        Some(total_td_error / batch.len() as f64)
    }

    /// Compute TD error
    fn compute_td_error(&self, transition: &Transition) -> f64 {
        let current_q = self.q_network.forward(&transition.state)[transition.action];

        let target_q = if transition.done {
            transition.reward
        } else if self.config.double_dqn {
            let best_action = self.q_network.best_action(&transition.next_state);
            let next_q = self.target_network.forward(&transition.next_state)[best_action];
            transition.reward + self.config.gamma * next_q
        } else {
            let next_q = self.target_network.max_q(&transition.next_state);
            transition.reward + self.config.gamma * next_q
        };

        target_q - current_q
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    /// Get best action
    pub fn best_action(&self, state: &[f64]) -> usize {
        self.q_network.best_action(state)
    }
}

/// Statistics from DQN training
#[derive(Debug, Clone, Default)]
pub struct DQNStats {
    /// Mean TD error
    pub mean_td_error: f64,
    /// Episode reward
    pub episode_reward: f64,
    /// Episode length
    pub episode_length: usize,
    /// Current epsilon
    pub epsilon: f64,
    /// Buffer size
    pub buffer_size: usize,
}

/// Dueling DQN Network
/// Separates state value and advantage estimation
#[derive(Debug, Clone)]
pub struct DuelingQNetwork {
    /// Input dimension
    pub input_dim: usize,
    /// Number of actions
    pub num_actions: usize,
    /// Shared features to value stream weights
    pub value_weights: Vec<f64>,
    /// Value stream bias
    pub value_bias: f64,
    /// Shared features to advantage stream weights
    pub advantage_weights: Vec<Vec<f64>>,
    /// Advantage stream biases
    pub advantage_biases: Vec<f64>,
}

impl DuelingQNetwork {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();

        let value_weights: Vec<f64> = (0..input_dim)
            .map(|i| {
                let seed = (i * 41) as f64;
                (seed.sin() * 43758.5453).fract() * scale
            })
            .collect();

        let advantage_weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|i| {
                (0..num_actions)
                    .map(|j| {
                        let seed = (i * 37 + j * 19) as f64;
                        (seed.sin() * 43758.5453).fract() * scale
                    })
                    .collect()
            })
            .collect();

        Self {
            input_dim,
            num_actions,
            value_weights,
            value_bias: 0.0,
            advantage_weights,
            advantage_biases: vec![0.0; num_actions],
        }
    }

    /// Forward pass with dueling architecture
    /// Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    pub fn forward(&self, state: &[f64]) -> Vec<f64> {
        // Compute value
        let mut value = self.value_bias;
        for (i, &s) in state.iter().enumerate().take(self.input_dim) {
            value += s * self.value_weights[i];
        }

        // Compute advantages
        let mut advantages = self.advantage_biases.clone();
        for (i, &s) in state.iter().enumerate().take(self.input_dim) {
            for (a, adv) in advantages.iter_mut().enumerate() {
                *adv += s * self.advantage_weights[i][a];
            }
        }

        // Mean advantage
        let mean_adv: f64 = advantages.iter().sum::<f64>() / self.num_actions as f64;

        // Q = V + A - mean(A)
        advantages.iter().map(|&a| value + a - mean_adv).collect()
    }

    /// Get best action
    pub fn best_action(&self, state: &[f64]) -> usize {
        let q_values = self.forward(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Copy from another network
    pub fn copy_from(&mut self, other: &DuelingQNetwork) {
        self.value_weights = other.value_weights.clone();
        self.value_bias = other.value_bias;
        self.advantage_weights = other.advantage_weights.clone();
        self.advantage_biases = other.advantage_biases.clone();
    }

    /// Soft update
    pub fn soft_update(&mut self, local: &DuelingQNetwork, tau: f64) {
        for (i, w) in self.value_weights.iter_mut().enumerate() {
            *w = tau * local.value_weights[i] + (1.0 - tau) * *w;
        }
        self.value_bias = tau * local.value_bias + (1.0 - tau) * self.value_bias;

        for (i, row) in self.advantage_weights.iter_mut().enumerate() {
            for (a, w) in row.iter_mut().enumerate() {
                *w = tau * local.advantage_weights[i][a] + (1.0 - tau) * *w;
            }
        }
        for (a, b) in self.advantage_biases.iter_mut().enumerate() {
            *b = tau * local.advantage_biases[a] + (1.0 - tau) * *b;
        }
    }
}

/// N-step DQN returns computation
pub fn compute_n_step_returns(rewards: &[f64], next_q: f64, gamma: f64, done: bool) -> f64 {
    let mut g = if done { 0.0 } else { next_q };

    for &r in rewards.iter().rev() {
        g = r + gamma * g;
    }

    g
}

/// Compute Huber loss (smooth L1)
pub fn huber_loss(td_error: f64, delta: f64) -> f64 {
    let abs_error = td_error.abs();
    if abs_error <= delta {
        0.5 * td_error * td_error
    } else {
        delta * (abs_error - 0.5 * delta)
    }
}

/// Compute Huber loss gradient
pub fn huber_loss_grad(td_error: f64, delta: f64) -> f64 {
    if td_error.abs() <= delta {
        td_error
    } else {
        delta * td_error.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_creation() {
        let t = Transition::new(vec![1.0, 2.0], 0, 1.0, vec![2.0, 3.0], false);
        assert_eq!(t.state, vec![1.0, 2.0]);
        assert_eq!(t.action, 0);
        assert!((t.reward - 1.0).abs() < 1e-10);
        assert!(!t.done);
    }

    #[test]
    fn test_replay_buffer_creation() {
        let buffer = ReplayBuffer::new(ReplayBufferConfig::default());
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.can_sample());
    }

    #[test]
    fn test_replay_buffer_push() {
        let mut buffer = ReplayBuffer::new(ReplayBufferConfig::small());

        for i in 0..100 {
            buffer.add(vec![i as f64], 0, 1.0, vec![(i + 1) as f64], false);
        }

        assert_eq!(buffer.len(), 100);
        assert!(buffer.can_sample());
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let config = ReplayBufferConfig::default().with_capacity(10);
        let mut buffer = ReplayBuffer::new(config);

        for i in 0..20 {
            buffer.add(vec![i as f64], 0, 1.0, vec![(i + 1) as f64], false);
        }

        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(ReplayBufferConfig::small().with_capacity(100));

        for i in 0..100 {
            buffer.add(vec![i as f64], i % 2, 1.0, vec![(i + 1) as f64], false);
        }

        let rng_values: Vec<f64> = (0..32).map(|i| (i as f64 * 0.03) % 1.0).collect();
        let batch = buffer.sample(&rng_values);

        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_prioritized_buffer_creation() {
        let buffer = PrioritizedReplayBuffer::new(PrioritizedReplayConfig::default());
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_prioritized_buffer_push() {
        let mut buffer = PrioritizedReplayBuffer::new(PrioritizedReplayConfig::default());

        buffer.add(vec![1.0], 0, 1.0, vec![2.0], false);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_prioritized_buffer_update_priority() {
        let mut config = PrioritizedReplayConfig::default();
        config.base.min_size = 1;
        let mut buffer = PrioritizedReplayBuffer::new(config);

        buffer.add(vec![1.0], 0, 1.0, vec![2.0], false);
        buffer.add(vec![2.0], 1, 2.0, vec![3.0], false);

        buffer.update_priority(0, 5.0);
        buffer.update_priority(1, 1.0);

        // Higher priority transition should be sampled more often
        assert!(buffer.can_sample());
    }

    #[test]
    fn test_prioritized_buffer_sample() {
        let mut config = PrioritizedReplayConfig::default();
        config.base.min_size = 10;
        config.base.batch_size = 5;
        let mut buffer = PrioritizedReplayBuffer::new(config);

        for i in 0..20 {
            buffer.add(vec![i as f64], 0, 1.0, vec![(i + 1) as f64], false);
        }

        let rng_values: Vec<f64> = (0..5).map(|i| i as f64 * 0.15).collect();
        let (transitions, indices, weights) = buffer.sample(&rng_values);

        assert_eq!(transitions.len(), 5);
        assert_eq!(indices.len(), 5);
        assert_eq!(weights.len(), 5);

        // All weights should be normalized
        for &w in &weights {
            assert!(w <= 1.0);
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_q_network_creation() {
        let net = QNetwork::new(4, 2);
        assert_eq!(net.input_dim, 4);
        assert_eq!(net.num_actions, 2);
        assert_eq!(net.weights.len(), 4);
        assert_eq!(net.biases.len(), 2);
    }

    #[test]
    fn test_q_network_forward() {
        let net = QNetwork::new(4, 2);
        let state = vec![1.0, 0.0, 0.0, 0.0];

        let q_values = net.forward(&state);
        assert_eq!(q_values.len(), 2);
    }

    #[test]
    fn test_q_network_best_action() {
        let mut net = QNetwork::new(2, 3);

        // Set weights to prefer action 1
        net.weights[0][1] = 10.0;
        net.weights[1][1] = 10.0;

        let state = vec![1.0, 1.0];
        assert_eq!(net.best_action(&state), 1);
    }

    #[test]
    fn test_q_network_copy() {
        let net1 = QNetwork::new(4, 2);
        let mut net2 = QNetwork::new(4, 2);

        net2.copy_from(&net1);

        for i in 0..4 {
            for a in 0..2 {
                assert!((net1.weights[i][a] - net2.weights[i][a]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_q_network_soft_update() {
        let local = QNetwork::new(2, 2);
        let mut target = QNetwork::new(2, 2);

        let old_weights = target.weights.clone();
        target.soft_update(&local, 0.5);

        // Weights should be average of local and old target
        for i in 0..2 {
            for a in 0..2 {
                let expected = 0.5 * local.weights[i][a] + 0.5 * old_weights[i][a];
                assert!((target.weights[i][a] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_q_network_gradient() {
        let mut gradient = QNetworkGradient::zeros(4, 2);

        gradient.add_sample(&[1.0, 0.5, 0.0, 0.0], 0, 2.0);

        assert!((gradient.weights[0][0] - 2.0).abs() < 1e-10);
        assert!((gradient.weights[1][0] - 1.0).abs() < 1e-10);
        assert!((gradient.biases[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dqn_config_default() {
        let config = DQNConfig::default();
        assert!(config.double_dqn);
        assert!(config.use_soft_update);
    }

    #[test]
    fn test_dqn_creation() {
        let dqn = DQN::new(4, 2, DQNConfig::default(), ReplayBufferConfig::small());
        assert_eq!(dqn.q_network.input_dim, 4);
        assert_eq!(dqn.q_network.num_actions, 2);
    }

    #[test]
    fn test_dqn_select_action() {
        let dqn = DQN::new(4, 2, DQNConfig::default(), ReplayBufferConfig::small());
        let state = vec![1.0, 0.0, 0.0, 0.0];

        // Should explore with high epsilon
        let action = dqn.select_action(&state, 0.5);
        assert!(action < 2);
    }

    #[test]
    fn test_dqn_store_transition() {
        let mut dqn = DQN::new(4, 2, DQNConfig::default(), ReplayBufferConfig::small());

        dqn.store_transition(
            vec![1.0, 0.0, 0.0, 0.0],
            0,
            1.0,
            vec![0.0, 1.0, 0.0, 0.0],
            false,
        );

        assert_eq!(dqn.replay_buffer.len(), 1);
    }

    #[test]
    fn test_dqn_train_step_no_samples() {
        let mut dqn = DQN::new(4, 2, DQNConfig::default(), ReplayBufferConfig::small());

        let result = dqn.train_step(&[0.1, 0.2, 0.3]);
        assert!(result.is_none());
    }

    #[test]
    fn test_dqn_train_step() {
        let mut config = ReplayBufferConfig::small();
        config.min_size = 10;
        config.batch_size = 5;

        let mut dqn = DQN::new(4, 2, DQNConfig::default(), config);

        // Fill buffer
        for i in 0..20 {
            dqn.store_transition(
                vec![i as f64, 0.0, 0.0, 0.0],
                i % 2,
                1.0,
                vec![(i + 1) as f64, 0.0, 0.0, 0.0],
                i == 19,
            );
        }

        let rng_values: Vec<f64> = (0..10).map(|i| (i as f64 * 0.08) % 1.0).collect();
        let result = dqn.train_step(&rng_values);

        assert!(result.is_some());
    }

    #[test]
    fn test_dqn_decay_epsilon() {
        let mut dqn = DQN::new(4, 2, DQNConfig::default(), ReplayBufferConfig::small());
        let initial = dqn.epsilon;

        dqn.decay_epsilon();

        assert!(dqn.epsilon < initial);
        assert!(dqn.epsilon >= dqn.config.epsilon_end);
    }

    #[test]
    fn test_prioritized_dqn_creation() {
        let dqn = PrioritizedDQN::new(
            4,
            2,
            DQNConfig::default(),
            PrioritizedReplayConfig::default(),
        );
        assert_eq!(dqn.q_network.input_dim, 4);
    }

    #[test]
    fn test_dueling_network_creation() {
        let net = DuelingQNetwork::new(4, 2);
        assert_eq!(net.input_dim, 4);
        assert_eq!(net.num_actions, 2);
    }

    #[test]
    fn test_dueling_network_forward() {
        let net = DuelingQNetwork::new(4, 2);
        let state = vec![1.0, 0.0, 0.0, 0.0];

        let q_values = net.forward(&state);
        assert_eq!(q_values.len(), 2);
    }

    #[test]
    fn test_dueling_network_mean_advantage() {
        let mut net = DuelingQNetwork::new(2, 3);

        // Set up network where all advantages are equal
        net.advantage_weights = vec![vec![0.0; 3]; 2];
        net.advantage_biases = vec![1.0, 1.0, 1.0];
        net.value_weights = vec![1.0, 1.0];
        net.value_bias = 0.0;

        let state = vec![1.0, 1.0];
        let q_values = net.forward(&state);

        // All Q-values should be equal (V + A - mean(A) = V when A is constant)
        assert!((q_values[0] - q_values[1]).abs() < 1e-10);
        assert!((q_values[1] - q_values[2]).abs() < 1e-10);
    }

    #[test]
    fn test_n_step_returns() {
        let rewards = vec![1.0, 1.0, 1.0];
        let next_q = 5.0;
        let gamma = 0.9;

        let g = compute_n_step_returns(&rewards, next_q, gamma, false);

        // G = 1 + 0.9 * (1 + 0.9 * (1 + 0.9 * 5))
        let expected = 1.0 + 0.9 * (1.0 + 0.9 * (1.0 + 0.9 * 5.0));
        assert!((g - expected).abs() < 1e-10);
    }

    #[test]
    fn test_n_step_returns_terminal() {
        let rewards = vec![1.0, 1.0, 1.0];
        let next_q = 5.0;
        let gamma = 0.9;

        let g = compute_n_step_returns(&rewards, next_q, gamma, true);

        // G = 1 + 0.9 * (1 + 0.9 * 1) when done (no bootstrap)
        let expected = 1.0 + 0.9 * (1.0 + 0.9 * 1.0);
        assert!((g - expected).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss() {
        // Quadratic region
        let loss = huber_loss(0.5, 1.0);
        assert!((loss - 0.125).abs() < 1e-10);

        // Linear region
        let loss = huber_loss(2.0, 1.0);
        assert!((loss - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_grad() {
        // Quadratic region
        let grad = huber_loss_grad(0.5, 1.0);
        assert!((grad - 0.5).abs() < 1e-10);

        // Linear region
        let grad = huber_loss_grad(2.0, 1.0);
        assert!((grad - 1.0).abs() < 1e-10);

        let grad = huber_loss_grad(-2.0, 1.0);
        assert!((grad - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_clip_by_norm() {
        let mut gradient = QNetworkGradient::zeros(2, 2);
        gradient.weights[0][0] = 10.0;
        gradient.weights[1][1] = 10.0;

        gradient.clip_by_norm(1.0);

        let norm: f64 = gradient
            .weights
            .iter()
            .flat_map(|row| row.iter())
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-10);
    }
}
