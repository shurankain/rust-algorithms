// Reinforcement Learning Algorithms
// Core RL algorithms relevant for RLHF and agent training

pub mod dpo;
pub mod dqn;
pub mod grpo;
pub mod policy_gradient;
pub mod ppo;
pub mod q_learning;

pub use policy_gradient::{
    ActorCritic, GAE, LinearValueFunction, PolicyAnalysis, PolicyGradientConfig,
    PolicyGradientStats, REINFORCE, SoftmaxPolicy, Step, Trajectory, analyze_policy,
    compute_entropy, discounted_cumsum, normalize_advantages, sample_categorical, softmax,
};

pub use ppo::{
    PPO, PPOAnalysis, PPOBuffer, PPOClippedValue, PPOConfig, PPOStats, RolloutBuffer,
    RolloutSample, analyze_ppo_training, clipped_surrogate, compute_ratio,
};

pub use dpo::{
    DPO, DPOAnalysis, DPOConfig, DPOStats, IPO, KTO, PreferencePair, analyze_dpo_training,
    generate_preferences_from_reward, log_sigmoid_stable, sigmoid_stable,
};

pub use grpo::{
    GRPO, GRPOAnalysis, GRPOConfig, GRPOStats, GroupSample, GroupStats, OnlineGRPO,
    RewardWeightedGRPO, SampleGroup, analyze_grpo_training, generate_groups, group_statistics,
};

pub use q_learning::{
    DiscreteState, DoubleQLearning, EligibilityTraces, ExpectedSARSA, NStepQLearning, QLambda,
    QLearning, QLearningAnalysis, QLearningConfig, QLearningStats, QTable, SARSA,
    analyze_q_learning, run_episode,
};

pub use dqn::{
    DQN, DQNConfig, DQNStats, DuelingQNetwork, PrioritizedDQN, PrioritizedReplayBuffer,
    PrioritizedReplayConfig, QNetwork, QNetworkGradient, ReplayBuffer, ReplayBufferConfig,
    Transition, compute_n_step_returns, huber_loss, huber_loss_grad,
};
