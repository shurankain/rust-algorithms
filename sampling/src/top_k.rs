// Top-K Sampling
//
// Token sampling strategy for LLM inference. Limits sampling to the K most
// probable tokens, then samples from this reduced distribution.
//
// Algorithm:
// 1. Sort tokens by probability (descending)
// 2. Keep only top K tokens
// 3. Renormalize probabilities to sum to 1
// 4. Sample from the filtered distribution
//
// Why use Top-K:
// - Prevents sampling very unlikely tokens (reduces nonsense)
// - More diverse than greedy decoding (always picking max)
// - Controllable via K parameter
//
// Common K values: 10-100 for most applications
// K=1 is equivalent to greedy decoding
//
// Time: O(n log n) for sorting, or O(n log k) with partial sort
// Space: O(k) for filtered tokens

use std::cmp::Ordering;

// Token with its probability/logit
#[derive(Clone, Debug)]
pub struct TokenProb {
    pub token_id: usize,
    pub prob: f64,
}

// Result of top-k filtering
#[derive(Clone, Debug)]
pub struct TopKResult {
    pub tokens: Vec<TokenProb>,
    pub original_mass: f64, // probability mass of selected tokens before renormalization
}

// Filter to top-k tokens and renormalize
pub fn top_k_filter(probs: &[f64], k: usize) -> TopKResult {
    if probs.is_empty() || k == 0 {
        return TopKResult {
            tokens: vec![],
            original_mass: 0.0,
        };
    }

    let k = k.min(probs.len());

    // Create (index, prob) pairs and sort by probability descending
    let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Take top k
    let top_k: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();

    // Calculate original probability mass
    let original_mass: f64 = top_k.iter().map(|(_, p)| p).sum();

    // Renormalize
    let tokens: Vec<TokenProb> = if original_mass > 0.0 {
        top_k
            .into_iter()
            .map(|(token_id, prob)| TokenProb {
                token_id,
                prob: prob / original_mass,
            })
            .collect()
    } else {
        // All zeros - uniform distribution over top k
        let uniform = 1.0 / k as f64;
        top_k
            .into_iter()
            .map(|(token_id, _)| TokenProb {
                token_id,
                prob: uniform,
            })
            .collect()
    };

    TopKResult {
        tokens,
        original_mass,
    }
}

// Sample a token from the filtered distribution
// Returns token_id
pub fn top_k_sample(probs: &[f64], k: usize, random: f64) -> Option<usize> {
    let filtered = top_k_filter(probs, k);

    if filtered.tokens.is_empty() {
        return None;
    }

    // Use random value [0, 1) to select token
    let random = random.clamp(0.0, 0.9999999);
    let mut cumulative = 0.0;

    for token in &filtered.tokens {
        cumulative += token.prob;
        if random < cumulative {
            return Some(token.token_id);
        }
    }

    // Fallback to last token (shouldn't happen with proper normalization)
    Some(filtered.tokens.last().unwrap().token_id)
}

// Convenience function: top-k with greedy selection (k=1 equivalent)
pub fn greedy_sample(probs: &[f64]) -> Option<usize> {
    if probs.is_empty() {
        return None;
    }

    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
}

// Top-K from logits (unnormalized log probabilities)
// Applies softmax first, then top-k filtering
pub fn top_k_from_logits(logits: &[f64], k: usize) -> TopKResult {
    let probs = softmax(logits);
    top_k_filter(&probs, k)
}

// Softmax: convert logits to probabilities
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    // Subtract max for numerical stability
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exp_values: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp_values.iter().sum();

    if sum > 0.0 {
        exp_values.iter().map(|&x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f64; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_top_k_filter_basic() {
        let probs = vec![0.1, 0.3, 0.2, 0.4];
        let result = top_k_filter(&probs, 2);

        assert_eq!(result.tokens.len(), 2);
        // Top 2 should be indices 3 (0.4) and 1 (0.3)
        assert_eq!(result.tokens[0].token_id, 3);
        assert_eq!(result.tokens[1].token_id, 1);

        // Original mass = 0.4 + 0.3 = 0.7
        assert!(approx_eq(result.original_mass, 0.7));

        // Renormalized: 0.4/0.7 and 0.3/0.7
        assert!(approx_eq(result.tokens[0].prob, 0.4 / 0.7));
        assert!(approx_eq(result.tokens[1].prob, 0.3 / 0.7));
    }

    #[test]
    fn test_top_k_filter_k_larger_than_vocab() {
        let probs = vec![0.5, 0.3, 0.2];
        let result = top_k_filter(&probs, 10);

        // Should include all tokens
        assert_eq!(result.tokens.len(), 3);
        assert!(approx_eq(result.original_mass, 1.0));
    }

    #[test]
    fn test_top_k_filter_k_one() {
        let probs = vec![0.1, 0.6, 0.3];
        let result = top_k_filter(&probs, 1);

        assert_eq!(result.tokens.len(), 1);
        assert_eq!(result.tokens[0].token_id, 1); // highest prob
        assert!(approx_eq(result.tokens[0].prob, 1.0)); // renormalized to 1
    }

    #[test]
    fn test_top_k_sample_deterministic() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // With random=0, should pick first token in filtered list
        let token = top_k_sample(&probs, 2, 0.0).unwrap();
        assert_eq!(token, 3); // highest prob token

        // With random close to 1, should pick last token in filtered list
        let token = top_k_sample(&probs, 2, 0.99).unwrap();
        assert_eq!(token, 2); // second highest
    }

    #[test]
    fn test_greedy_sample() {
        let probs = vec![0.1, 0.5, 0.2, 0.2];
        let token = greedy_sample(&probs).unwrap();
        assert_eq!(token, 1); // highest probability
    }

    #[test]
    fn test_greedy_sample_empty() {
        let probs: Vec<f64> = vec![];
        assert!(greedy_sample(&probs).is_none());
    }

    #[test]
    fn test_top_k_from_logits() {
        // Logits: higher = more likely
        let logits = vec![1.0, 2.0, 0.5, 3.0];
        let result = top_k_from_logits(&logits, 2);

        assert_eq!(result.tokens.len(), 2);
        // Top 2 by logit: indices 3 (3.0) and 1 (2.0)
        assert_eq!(result.tokens[0].token_id, 3);
        assert_eq!(result.tokens[1].token_id, 1);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0));

        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits - should not overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        let sum: f64 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_top_k_empty() {
        let probs: Vec<f64> = vec![];
        let result = top_k_filter(&probs, 5);
        assert!(result.tokens.is_empty());

        let sample = top_k_sample(&probs, 5, 0.5);
        assert!(sample.is_none());
    }

    #[test]
    fn test_top_k_zero() {
        let probs = vec![0.5, 0.5];
        let result = top_k_filter(&probs, 0);
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_top_k_with_ties() {
        // Equal probabilities - order determined by sort stability
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = top_k_filter(&probs, 2);

        assert_eq!(result.tokens.len(), 2);
        assert!(approx_eq(result.original_mass, 0.5));
        // Each selected token should have prob 0.5 after renormalization
        assert!(approx_eq(result.tokens[0].prob, 0.5));
        assert!(approx_eq(result.tokens[1].prob, 0.5));
    }

    #[test]
    fn test_sampling_distribution() {
        // Test that sampling roughly follows the distribution
        let probs = vec![0.0, 0.0, 0.5, 0.5]; // only tokens 2 and 3 have probability
        let k = 4;

        let mut counts = [0; 4];
        let samples = 1000;

        for i in 0..samples {
            let random = (i as f64) / (samples as f64);
            if let Some(token) = top_k_sample(&probs, k, random) {
                counts[token] += 1;
            }
        }

        // Tokens 0 and 1 should never be sampled
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);

        // Tokens 2 and 3 should be sampled roughly equally
        assert!(counts[2] > 400);
        assert!(counts[3] > 400);
    }
}
