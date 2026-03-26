// Reinforcement Learning Algorithms
// Core RL algorithms relevant for RLHF and agent training

pub mod policy_gradient;
pub mod ppo;

pub use policy_gradient::{
    ActorCritic, GAE, LinearValueFunction, PolicyAnalysis, PolicyGradientConfig,
    PolicyGradientStats, REINFORCE, SoftmaxPolicy, Step, Trajectory, analyze_policy,
    compute_entropy, discounted_cumsum, normalize_advantages, sample_categorical, softmax,
};

pub use ppo::{
    PPO, PPOAnalysis, PPOBuffer, PPOClippedValue, PPOConfig, PPOStats, RolloutBuffer,
    RolloutSample, analyze_ppo_training, clipped_surrogate, compute_ratio,
};
