// Speculative Decoding
//
// Paper: "Fast Inference from Transformers via Speculative Decoding"
// Authors: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google, 2023)
//
// Also: "Accelerating Large Language Model Decoding with Speculative Sampling"
// Authors: Charlie Chen, Sebastian Borgeaud, et al. (DeepMind, 2023)
//
// Key insight: Use a smaller, faster "draft" model to generate candidate tokens,
// then verify them in parallel with the larger "target" model.
//
// How it works:
// 1. Draft model generates K candidate tokens autoregressively (fast)
// 2. Target model evaluates all K+1 positions in a single forward pass (parallel)
// 3. Accept/reject each token based on probability ratio
// 4. If rejected at position i, resample from adjusted distribution
// 5. Always get at least 1 token, potentially K+1 tokens per iteration
//
// Benefits:
// - 2-3x speedup without quality loss (lossless when done correctly)
// - Works with any draft/target model pair
// - No training required
//
// The acceptance probability is:
// P(accept) = min(1, p_target(x) / p_draft(x))
//
// If rejected, sample from the residual distribution:
// p_residual(x) = max(0, p_target(x) - p_draft(x)) / Z
// where Z is the normalization factor

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of draft tokens to generate per iteration (K)
    pub num_speculative_tokens: usize,
    /// Temperature for sampling (1.0 = no change)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Top-k sampling threshold (0 = disabled)
    pub top_k: usize,
}

impl SpeculativeConfig {
    /// Create a new speculative decoding config
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            num_speculative_tokens: 4,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
        }
    }

    /// Set number of speculative tokens (K)
    pub fn num_speculative_tokens(mut self, k: usize) -> Self {
        self.num_speculative_tokens = k;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-p (nucleus) sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set top-k sampling
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self::new(32000) // Common vocab size
    }
}

/// Statistics for speculative decoding
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total decoding iterations
    pub iterations: usize,
    /// Total tokens generated
    pub tokens_generated: usize,
    /// Draft tokens proposed
    pub draft_tokens_proposed: usize,
    /// Draft tokens accepted
    pub draft_tokens_accepted: usize,
    /// Target model forward passes
    pub target_forward_passes: usize,
    /// Draft model forward passes
    pub draft_forward_passes: usize,
}

impl SpeculativeStats {
    /// Acceptance rate of draft tokens
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens_proposed > 0 {
            self.draft_tokens_accepted as f32 / self.draft_tokens_proposed as f32
        } else {
            0.0
        }
    }

    /// Average tokens per iteration
    pub fn tokens_per_iteration(&self) -> f32 {
        if self.iterations > 0 {
            self.tokens_generated as f32 / self.iterations as f32
        } else {
            0.0
        }
    }

    /// Speedup factor estimate (compared to autoregressive)
    /// Assumes target model is ~5x slower than draft model
    pub fn estimated_speedup(&self, draft_to_target_ratio: f32) -> f32 {
        if self.target_forward_passes == 0 {
            return 1.0;
        }

        // Without speculation: tokens_generated target forward passes
        // With speculation: target_forward_passes + draft_forward_passes / ratio
        let baseline_cost = self.tokens_generated as f32;
        let speculative_cost = self.target_forward_passes as f32
            + self.draft_forward_passes as f32 * draft_to_target_ratio;

        if speculative_cost > 0.0 {
            baseline_cost / speculative_cost
        } else {
            1.0
        }
    }
}

/// Result of a single speculative decoding iteration
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Accepted tokens (always at least 1)
    pub tokens: Vec<usize>,
    /// Number of draft tokens that were accepted
    pub accepted_draft_tokens: usize,
    /// Whether all draft tokens were accepted (bonus token case)
    pub all_accepted: bool,
}

/// Speculative decoding engine
#[derive(Debug)]
pub struct SpeculativeDecoder {
    config: SpeculativeConfig,
    stats: SpeculativeStats,
    rng_state: u64,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            stats: SpeculativeStats::default(),
            rng_state: 42,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Generate a random number in [0, 1)
    fn random(&mut self) -> f32 {
        let mut hasher = DefaultHasher::new();
        self.rng_state.hash(&mut hasher);
        self.rng_state = hasher.finish();
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    /// Apply temperature and optional sampling modifications
    fn apply_temperature(&self, logits: &[f32]) -> Vec<f32> {
        let temp = self.config.temperature.max(1e-8);
        logits.iter().map(|&l| l / temp).collect()
    }

    /// Convert logits to probabilities via softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

        logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Apply top-k filtering to logits
    fn apply_top_k(&self, logits: &[f32]) -> Vec<f32> {
        if self.config.top_k == 0 || self.config.top_k >= logits.len() {
            return logits.to_vec();
        }

        // Find the k-th largest value
        let mut sorted_logits: Vec<f32> = logits.to_vec();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted_logits[self.config.top_k - 1];

        // Mask out values below threshold
        logits
            .iter()
            .map(|&l| if l >= threshold { l } else { f32::NEG_INFINITY })
            .collect()
    }

    /// Apply top-p (nucleus) filtering to probabilities
    fn apply_top_p(&self, probs: &[f32]) -> Vec<f32> {
        if self.config.top_p >= 1.0 {
            return probs.to_vec();
        }

        // Sort probabilities in descending order with indices
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum > self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Create mask
        let mut result = vec![0.0; probs.len()];
        for &(idx, p) in indexed.iter().take(cutoff_idx) {
            result[idx] = p;
        }

        // Renormalize
        let sum: f32 = result.iter().sum();
        if sum > 0.0 {
            for p in &mut result {
                *p /= sum;
            }
        }

        result
    }

    /// Sample a token from a probability distribution
    fn sample(&mut self, probs: &[f32]) -> usize {
        let r = self.random();
        let mut cumsum = 0.0;

        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return idx;
            }
        }

        // Fallback to last token (shouldn't happen with proper normalization)
        probs.len() - 1
    }

    /// Process logits through temperature, top-k, top-p and return probabilities
    fn logits_to_probs(&self, logits: &[f32]) -> Vec<f32> {
        let scaled = self.apply_temperature(logits);
        let filtered = self.apply_top_k(&scaled);
        let probs = self.softmax(&filtered);
        self.apply_top_p(&probs)
    }

    /// Perform one iteration of speculative decoding
    ///
    /// Arguments:
    /// - `draft_logits`: Logits from draft model for K positions [K, vocab_size]
    /// - `target_logits`: Logits from target model for K+1 positions [K+1, vocab_size]
    /// - `draft_tokens`: The K tokens sampled from draft model
    ///
    /// Returns: SpeculativeResult with accepted tokens
    pub fn verify_and_accept(
        &mut self,
        draft_logits: &[Vec<f32>],
        target_logits: &[Vec<f32>],
        draft_tokens: &[usize],
    ) -> SpeculativeResult {
        let k = draft_tokens.len();
        assert_eq!(draft_logits.len(), k);
        assert_eq!(target_logits.len(), k + 1);

        self.stats.iterations += 1;
        self.stats.draft_tokens_proposed += k;
        self.stats.target_forward_passes += 1;
        self.stats.draft_forward_passes += k;

        let mut accepted_tokens = Vec::new();
        let mut accepted_count = 0;

        // Check each draft token
        for i in 0..k {
            let token = draft_tokens[i];

            // Get probabilities
            let draft_probs = self.logits_to_probs(&draft_logits[i]);
            let target_probs = self.logits_to_probs(&target_logits[i]);

            let p_draft = draft_probs.get(token).copied().unwrap_or(0.0);
            let p_target = target_probs.get(token).copied().unwrap_or(0.0);

            // Acceptance probability: min(1, p_target / p_draft)
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 0.0 {
                1.0 // Draft assigned 0 probability but target didn't - accept
            } else {
                0.0 // Both assigned 0 - reject
            };

            let r = self.random();

            if r < accept_prob {
                // Accept this token
                accepted_tokens.push(token);
                accepted_count += 1;
            } else {
                // Reject - sample from residual distribution
                let residual = self.compute_residual(&target_probs, &draft_probs);
                let resampled = self.sample(&residual);
                accepted_tokens.push(resampled);

                self.stats.tokens_generated += accepted_tokens.len();
                self.stats.draft_tokens_accepted += accepted_count;

                return SpeculativeResult {
                    tokens: accepted_tokens,
                    accepted_draft_tokens: accepted_count,
                    all_accepted: false,
                };
            }
        }

        // All draft tokens accepted! Sample bonus token from target's last position
        let bonus_probs = self.logits_to_probs(&target_logits[k]);
        let bonus_token = self.sample(&bonus_probs);
        accepted_tokens.push(bonus_token);

        self.stats.tokens_generated += accepted_tokens.len();
        self.stats.draft_tokens_accepted += accepted_count;

        SpeculativeResult {
            tokens: accepted_tokens,
            accepted_draft_tokens: accepted_count,
            all_accepted: true,
        }
    }

    /// Compute the residual distribution for rejection sampling
    /// p_residual(x) = max(0, p_target(x) - p_draft(x)) / Z
    fn compute_residual(&self, target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
        let mut residual: Vec<f32> = target_probs
            .iter()
            .zip(draft_probs.iter())
            .map(|(&t, &d)| (t - d).max(0.0))
            .collect();

        // Normalize
        let sum: f32 = residual.iter().sum();
        if sum > 0.0 {
            for p in &mut residual {
                *p /= sum;
            }
        } else {
            // Fallback to target distribution if residual is all zeros
            return target_probs.to_vec();
        }

        residual
    }

    /// Simplified interface: generate tokens given model outputs
    ///
    /// This simulates what would happen with actual models by taking
    /// pre-computed logits and performing the verification step.
    pub fn decode_step(
        &mut self,
        draft_logits: &[Vec<f32>],
        target_logits: &[Vec<f32>],
    ) -> SpeculativeResult {
        // First, sample draft tokens from draft logits
        let draft_tokens: Vec<usize> = draft_logits
            .iter()
            .map(|logits| {
                let probs = self.logits_to_probs(logits);
                self.sample(&probs)
            })
            .collect();

        // Then verify with target
        self.verify_and_accept(draft_logits, target_logits, &draft_tokens)
    }
}

/// Compute acceptance rate given draft and target probability distributions
/// This is useful for analyzing model compatibility
pub fn compute_expected_acceptance_rate(draft_probs: &[f32], target_probs: &[f32]) -> f32 {
    // Expected acceptance rate = sum over x of: p_draft(x) * min(1, p_target(x) / p_draft(x))
    // = sum over x of: min(p_draft(x), p_target(x))

    draft_probs
        .iter()
        .zip(target_probs.iter())
        .map(|(&d, &t)| d.min(t))
        .sum()
}

/// Analyze the quality of a draft model for speculative decoding
/// Higher values indicate better draft model alignment with target
pub fn analyze_draft_quality(
    draft_distributions: &[Vec<f32>],
    target_distributions: &[Vec<f32>],
) -> DraftQualityAnalysis {
    assert_eq!(draft_distributions.len(), target_distributions.len());

    if draft_distributions.is_empty() {
        return DraftQualityAnalysis {
            mean_acceptance_rate: 0.0,
            min_acceptance_rate: 0.0,
            max_acceptance_rate: 0.0,
            kl_divergence: 0.0,
            expected_tokens_per_iteration: 0.0,
        };
    }

    let mut acceptance_rates = Vec::with_capacity(draft_distributions.len());
    let mut total_kl = 0.0;

    for (draft, target) in draft_distributions.iter().zip(target_distributions.iter()) {
        let rate = compute_expected_acceptance_rate(draft, target);
        acceptance_rates.push(rate);

        // Compute KL divergence: sum p_target * log(p_target / p_draft)
        let kl: f32 = target
            .iter()
            .zip(draft.iter())
            .filter(|&(&t, &d)| t > 1e-10 && d > 1e-10)
            .map(|(&t, &d)| t * (t / d).ln())
            .sum();
        total_kl += kl;
    }

    let mean_rate: f32 = acceptance_rates.iter().sum::<f32>() / acceptance_rates.len() as f32;
    let min_rate = acceptance_rates
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_rate = acceptance_rates
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Expected tokens per iteration with K draft tokens
    // E[tokens] = sum_{i=0}^{K} (i+1) * P(accept exactly i then reject at i+1)
    //           + (K+1) * P(accept all K)
    // Simplified: approximately 1 / (1 - mean_rate) capped at K+1
    let k = draft_distributions.len();
    let expected_tokens = if mean_rate < 1.0 {
        (1.0 / (1.0 - mean_rate)).min((k + 1) as f32)
    } else {
        (k + 1) as f32
    };

    DraftQualityAnalysis {
        mean_acceptance_rate: mean_rate,
        min_acceptance_rate: min_rate,
        max_acceptance_rate: max_rate,
        kl_divergence: total_kl / draft_distributions.len() as f32,
        expected_tokens_per_iteration: expected_tokens,
    }
}

/// Analysis of draft model quality for speculative decoding
#[derive(Debug, Clone)]
pub struct DraftQualityAnalysis {
    /// Mean acceptance rate across all positions
    pub mean_acceptance_rate: f32,
    /// Minimum acceptance rate
    pub min_acceptance_rate: f32,
    /// Maximum acceptance rate
    pub max_acceptance_rate: f32,
    /// Average KL divergence from target to draft
    pub kl_divergence: f32,
    /// Expected number of tokens per iteration
    pub expected_tokens_per_iteration: f32,
}

/// Tree-based speculative decoding (Medusa-style)
/// Instead of linear speculation, generate a tree of candidates
#[derive(Debug, Clone)]
pub struct SpeculativeTree {
    /// Token at this node
    pub token: usize,
    /// Probability of this token
    pub prob: f32,
    /// Children (candidate continuations)
    pub children: Vec<SpeculativeTree>,
}

impl SpeculativeTree {
    /// Create a leaf node
    pub fn leaf(token: usize, prob: f32) -> Self {
        Self {
            token,
            prob,
            children: vec![],
        }
    }

    /// Create a node with children
    pub fn node(token: usize, prob: f32, children: Vec<SpeculativeTree>) -> Self {
        Self {
            token,
            prob,
            children,
        }
    }

    /// Total number of nodes in the tree
    pub fn size(&self) -> usize {
        1 + self.children.iter().map(|c| c.size()).sum::<usize>()
    }

    /// Maximum depth of the tree
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Get all paths from root to leaves
    pub fn all_paths(&self) -> Vec<Vec<usize>> {
        if self.children.is_empty() {
            vec![vec![self.token]]
        } else {
            let mut paths = Vec::new();
            for child in &self.children {
                for mut path in child.all_paths() {
                    path.insert(0, self.token);
                    paths.push(path);
                }
            }
            paths
        }
    }
}

/// Build a speculative tree from top-k candidates at each level
pub fn build_speculative_tree(
    logits_per_level: &[Vec<f32>],
    top_k_per_level: &[usize],
    temperature: f32,
) -> Vec<SpeculativeTree> {
    if logits_per_level.is_empty() || top_k_per_level.is_empty() {
        return vec![];
    }

    let temp = temperature.max(1e-8);

    // Get top-k for first level
    let first_logits = &logits_per_level[0];
    let k = top_k_per_level[0].min(first_logits.len());

    let probs = softmax_with_temp(first_logits, temp);
    let top_k_indices = top_k_indices(&probs, k);

    top_k_indices
        .into_iter()
        .map(|idx| {
            let children = if logits_per_level.len() > 1 && top_k_per_level.len() > 1 {
                build_speculative_tree(&logits_per_level[1..], &top_k_per_level[1..], temperature)
            } else {
                vec![]
            };
            SpeculativeTree::node(idx, probs[idx], children)
        })
        .collect()
}

/// Helper: softmax with temperature
fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scaled.iter().map(|&l| (l - max_val).exp()).sum();
    scaled
        .iter()
        .map(|&l| (l - max_val).exp() / exp_sum)
        .collect()
}

/// Helper: get indices of top-k elements
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
        let mut result = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let val = (hash as f32 / u64::MAX as f32) * 10.0 - 5.0; // [-5, 5]
            result.push(val);
        }
        result
    }

    fn peaked_logits(vocab_size: usize, peak_idx: usize, peak_value: f32) -> Vec<f32> {
        let mut logits = vec![0.0; vocab_size];
        logits[peak_idx] = peak_value;
        logits
    }

    #[test]
    fn test_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.num_speculative_tokens, 4);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_config_builder() {
        let config = SpeculativeConfig::new(50000)
            .num_speculative_tokens(8)
            .temperature(0.7)
            .top_p(0.9)
            .top_k(50);

        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.num_speculative_tokens, 8);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 50);
    }

    #[test]
    fn test_softmax() {
        let config = SpeculativeConfig::new(4);
        let decoder = SpeculativeDecoder::new(config);

        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = decoder.softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Probabilities should be in increasing order
        for i in 0..probs.len() - 1 {
            assert!(probs[i] < probs[i + 1]);
        }
    }

    #[test]
    fn test_temperature() {
        let config = SpeculativeConfig::new(4).temperature(0.5);
        let decoder = SpeculativeDecoder::new(config);

        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let scaled = decoder.apply_temperature(&logits);

        // Should be doubled due to temperature 0.5
        for i in 0..logits.len() {
            assert!((scaled[i] - logits[i] * 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_top_k() {
        let config = SpeculativeConfig::new(5).top_k(2);
        let decoder = SpeculativeDecoder::new(config);

        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let filtered = decoder.apply_top_k(&logits);

        // Only indices 1 and 3 should be kept (values 5 and 4)
        assert_eq!(filtered[1], 5.0);
        assert_eq!(filtered[3], 4.0);
        assert_eq!(filtered[0], f32::NEG_INFINITY);
        assert_eq!(filtered[2], f32::NEG_INFINITY);
        assert_eq!(filtered[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p() {
        let config = SpeculativeConfig::new(4).top_p(0.8);
        let decoder = SpeculativeDecoder::new(config);

        // Probability distribution: [0.1, 0.2, 0.3, 0.4]
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let filtered = decoder.apply_top_p(&probs);

        // Should keep top tokens until cumsum > 0.8
        // Sorted: 0.4, 0.3, 0.2, 0.1 -> cumsum: 0.4, 0.7, 0.9
        // Should keep indices 3, 2, 1 (up to 0.9 > 0.8)
        assert!(filtered[3] > 0.0);
        assert!(filtered[2] > 0.0);
        assert!(filtered[1] > 0.0);
        // Index 0 might be 0 depending on exact cutoff
    }

    #[test]
    fn test_sample_deterministic() {
        let config = SpeculativeConfig::new(4);
        let mut decoder = SpeculativeDecoder::new(config);
        decoder.set_seed(12345);

        // Very peaked distribution
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let token = decoder.sample(&probs);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_basic_acceptance() {
        let vocab_size = 10;
        let config = SpeculativeConfig::new(vocab_size);
        let mut decoder = SpeculativeDecoder::new(config);
        decoder.set_seed(42);

        // Create aligned draft and target (same distribution -> high acceptance)
        let logits = peaked_logits(vocab_size, 5, 10.0);

        let draft_logits = vec![logits.clone(), logits.clone()];
        let target_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let draft_tokens = vec![5, 5]; // Most likely tokens

        let result = decoder.verify_and_accept(&draft_logits, &target_logits, &draft_tokens);

        // With identical distributions and matching tokens, should accept all
        assert!(!result.tokens.is_empty());
        assert!(result.tokens.len() >= 1);
    }

    #[test]
    fn test_rejection_sampling() {
        let vocab_size = 10;
        let config = SpeculativeConfig::new(vocab_size);
        let mut decoder = SpeculativeDecoder::new(config);
        decoder.set_seed(123);

        // Draft prefers token 0, target prefers token 5
        let draft_logits = vec![peaked_logits(vocab_size, 0, 10.0)];
        let target_logits = vec![
            peaked_logits(vocab_size, 5, 10.0), // Different preference
            random_logits(vocab_size, 1),
        ];
        let draft_tokens = vec![0]; // Draft's choice

        let result = decoder.verify_and_accept(&draft_logits, &target_logits, &draft_tokens);

        // Should produce exactly 1 token (either accepted or resampled)
        assert_eq!(result.tokens.len(), 1);
    }

    #[test]
    fn test_stats() {
        let config = SpeculativeConfig::new(10);
        let mut decoder = SpeculativeDecoder::new(config);

        let logits = random_logits(10, 0);
        let draft_logits = vec![logits.clone(); 3];
        let target_logits = vec![logits.clone(); 4];

        decoder.decode_step(&draft_logits, &target_logits);

        assert_eq!(decoder.stats().iterations, 1);
        assert!(decoder.stats().tokens_generated >= 1);
        assert_eq!(decoder.stats().draft_tokens_proposed, 3);
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        let stats = SpeculativeStats {
            iterations: 10,
            tokens_generated: 35,
            draft_tokens_proposed: 40,
            draft_tokens_accepted: 25,
            target_forward_passes: 10,
            draft_forward_passes: 40,
        };

        assert!((stats.acceptance_rate() - 0.625).abs() < 1e-5);
        assert!((stats.tokens_per_iteration() - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_expected_acceptance() {
        // Identical distributions -> acceptance rate = 1.0
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let rate = compute_expected_acceptance_rate(&probs, &probs);
        assert!((rate - 1.0).abs() < 1e-5);

        // Disjoint distributions -> acceptance rate = 0.0
        let draft = vec![1.0, 0.0, 0.0, 0.0];
        let target = vec![0.0, 0.0, 0.0, 1.0];
        let rate = compute_expected_acceptance_rate(&draft, &target);
        assert!(rate.abs() < 1e-5);
    }

    #[test]
    fn test_draft_quality_analysis() {
        // Good draft: similar to target
        let draft = vec![vec![0.4, 0.3, 0.2, 0.1]];
        let target = vec![vec![0.35, 0.35, 0.2, 0.1]];

        let analysis = analyze_draft_quality(&draft, &target);

        assert!(analysis.mean_acceptance_rate > 0.5);
        assert!(analysis.kl_divergence < 1.0);
    }

    #[test]
    fn test_residual_distribution() {
        let config = SpeculativeConfig::new(4);
        let decoder = SpeculativeDecoder::new(config);

        let target = vec![0.5, 0.3, 0.1, 0.1];
        let draft = vec![0.2, 0.4, 0.2, 0.2];

        let residual = decoder.compute_residual(&target, &draft);

        // Residual should be normalized
        let sum: f32 = residual.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Should prefer tokens where target > draft
        assert!(residual[0] > residual[1]); // target[0] - draft[0] = 0.3 > 0
    }

    #[test]
    fn test_speculative_tree_leaf() {
        let tree = SpeculativeTree::leaf(42, 0.8);
        assert_eq!(tree.token, 42);
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.depth(), 1);
        assert_eq!(tree.all_paths(), vec![vec![42]]);
    }

    #[test]
    fn test_speculative_tree_node() {
        let tree = SpeculativeTree::node(
            1,
            0.5,
            vec![SpeculativeTree::leaf(2, 0.3), SpeculativeTree::leaf(3, 0.2)],
        );

        assert_eq!(tree.size(), 3);
        assert_eq!(tree.depth(), 2);

        let paths = tree.all_paths();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&vec![1, 2]));
        assert!(paths.contains(&vec![1, 3]));
    }

    #[test]
    fn test_build_speculative_tree() {
        let logits = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Level 0
            vec![2.0, 1.0, 3.0, 0.0], // Level 1
        ];
        let top_k = vec![2, 2];

        let trees = build_speculative_tree(&logits, &top_k, 1.0);

        // Should have 2 root nodes (top-2 from first level)
        assert_eq!(trees.len(), 2);

        // Each should have 2 children
        for tree in &trees {
            assert_eq!(tree.children.len(), 2);
        }
    }

    #[test]
    fn test_multiple_iterations() {
        let vocab_size = 100;
        let config = SpeculativeConfig::new(vocab_size).num_speculative_tokens(4);
        let mut decoder = SpeculativeDecoder::new(config);
        decoder.set_seed(999);

        // Run multiple iterations
        for i in 0..10 {
            let draft_logits: Vec<Vec<f32>> = (0..4)
                .map(|j| random_logits(vocab_size, i * 10 + j))
                .collect();
            let target_logits: Vec<Vec<f32>> = (0..5)
                .map(|j| random_logits(vocab_size, i * 10 + j + 100))
                .collect();

            let result = decoder.decode_step(&draft_logits, &target_logits);
            assert!(!result.tokens.is_empty());
            assert!(result.tokens.len() <= 5); // At most K+1
        }

        assert_eq!(decoder.stats().iterations, 10);
        assert!(decoder.stats().tokens_generated >= 10);
    }

    #[test]
    fn test_all_draft_accepted_bonus() {
        let vocab_size = 10;
        let config = SpeculativeConfig::new(vocab_size);
        let mut decoder = SpeculativeDecoder::new(config);
        decoder.set_seed(42);

        // Perfect alignment: draft == target
        let logits = peaked_logits(vocab_size, 7, 20.0);

        let k = 3;
        let draft_logits = vec![logits.clone(); k];
        let target_logits = vec![logits.clone(); k + 1];
        let draft_tokens = vec![7; k]; // The peaked token

        let result = decoder.verify_and_accept(&draft_logits, &target_logits, &draft_tokens);

        // Should get K+1 tokens (all accepted + bonus)
        assert!(result.all_accepted);
        assert_eq!(result.tokens.len(), k + 1);
        assert_eq!(result.accepted_draft_tokens, k);
    }

    #[test]
    fn test_estimated_speedup() {
        let stats = SpeculativeStats {
            iterations: 100,
            tokens_generated: 350, // 3.5 tokens/iteration on average
            draft_tokens_proposed: 400,
            draft_tokens_accepted: 250,
            target_forward_passes: 100,
            draft_forward_passes: 400,
        };

        // With draft:target ratio of 0.1 (draft is 10x faster)
        let speedup = stats.estimated_speedup(0.1);

        // Baseline: 350 forward passes
        // Speculative: 100 + 400*0.1 = 140 effective passes
        // Speedup: 350/140 = 2.5x
        assert!((speedup - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_empty_input() {
        let analysis = analyze_draft_quality(&[], &[]);
        assert_eq!(analysis.mean_acceptance_rate, 0.0);
        assert_eq!(analysis.expected_tokens_per_iteration, 0.0);
    }

    #[test]
    fn test_tree_deep() {
        let tree = SpeculativeTree::node(
            0,
            1.0,
            vec![SpeculativeTree::node(
                1,
                0.5,
                vec![SpeculativeTree::node(
                    2,
                    0.25,
                    vec![SpeculativeTree::leaf(3, 0.1)],
                )],
            )],
        );

        assert_eq!(tree.depth(), 4);
        assert_eq!(tree.size(), 4);
        assert_eq!(tree.all_paths(), vec![vec![0, 1, 2, 3]]);
    }
}
