// Nucleus Sampling (Top-P Sampling)
//
// Dynamic token sampling strategy for LLM inference. Unlike Top-K which uses
// a fixed number of tokens, Nucleus sampling selects the smallest set of tokens
// whose cumulative probability exceeds threshold P.
//
// Algorithm:
// 1. Sort tokens by probability (descending)
// 2. Compute cumulative probabilities
// 3. Find smallest set where cumulative prob >= P (the "nucleus")
// 4. Renormalize and sample from nucleus
//
// Why Nucleus over Top-K:
// - Adapts to probability distribution shape
// - When model is confident: nucleus is small (few tokens)
// - When model is uncertain: nucleus is large (many tokens)
// - Avoids arbitrary K cutoff that may exclude likely tokens
//
// Common P values: 0.9-0.95 for most applications
// P=1.0 includes all tokens (no filtering)
// P=0.0 is undefined (use greedy instead)
//
// Time: O(n log n) for sorting
// Space: O(n) for sorted tokens

use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct TokenProb {
    pub token_id: usize,
    pub prob: f64,
}

#[derive(Clone, Debug)]
pub struct NucleusResult {
    pub tokens: Vec<TokenProb>,
    pub nucleus_mass: f64,   // cumulative probability of nucleus before renorm
    pub nucleus_size: usize, // number of tokens in nucleus
}

// Filter to nucleus (top-p) and renormalize
pub fn nucleus_filter(probs: &[f64], p: f64) -> NucleusResult {
    if probs.is_empty() || p <= 0.0 {
        return NucleusResult {
            tokens: vec![],
            nucleus_mass: 0.0,
            nucleus_size: 0,
        };
    }

    // Sort by probability descending
    let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Find nucleus: smallest set with cumulative prob >= p
    let mut cumulative = 0.0;
    let mut nucleus_end = 0;

    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        nucleus_end = i + 1;
        if cumulative >= p {
            break;
        }
    }

    // Take nucleus tokens
    let nucleus: Vec<(usize, f64)> = indexed.into_iter().take(nucleus_end).collect();
    let nucleus_mass: f64 = nucleus.iter().map(|(_, p)| p).sum();
    let nucleus_size = nucleus.len();

    // Renormalize
    let tokens: Vec<TokenProb> = if nucleus_mass > 0.0 {
        nucleus
            .into_iter()
            .map(|(token_id, prob)| TokenProb {
                token_id,
                prob: prob / nucleus_mass,
            })
            .collect()
    } else {
        // Edge case: all zeros, uniform over nucleus
        let uniform = 1.0 / nucleus_size as f64;
        nucleus
            .into_iter()
            .map(|(token_id, _)| TokenProb {
                token_id,
                prob: uniform,
            })
            .collect()
    };

    NucleusResult {
        tokens,
        nucleus_mass,
        nucleus_size,
    }
}

// Sample a token from the nucleus
pub fn nucleus_sample(probs: &[f64], p: f64, random: f64) -> Option<usize> {
    let filtered = nucleus_filter(probs, p);

    if filtered.tokens.is_empty() {
        return None;
    }

    let random = random.clamp(0.0, 0.9999999);
    let mut cumulative = 0.0;

    for token in &filtered.tokens {
        cumulative += token.prob;
        if random < cumulative {
            return Some(token.token_id);
        }
    }

    Some(filtered.tokens.last().unwrap().token_id)
}

// Nucleus sampling from logits
pub fn nucleus_from_logits(logits: &[f64], p: f64) -> NucleusResult {
    let probs = softmax(logits);
    nucleus_filter(&probs, p)
}

// Combined Top-K and Nucleus (common in practice)
// First apply top-k, then nucleus within those k tokens
pub fn top_k_nucleus_filter(probs: &[f64], k: usize, p: f64) -> NucleusResult {
    if probs.is_empty() || k == 0 || p <= 0.0 {
        return NucleusResult {
            tokens: vec![],
            nucleus_mass: 0.0,
            nucleus_size: 0,
        };
    }

    // Sort by probability descending
    let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // First limit to top-k
    let top_k: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();

    // Renormalize top-k
    let top_k_mass: f64 = top_k.iter().map(|(_, prob)| prob).sum();
    let top_k_normalized: Vec<(usize, f64)> = if top_k_mass > 0.0 {
        top_k
            .into_iter()
            .map(|(id, prob)| (id, prob / top_k_mass))
            .collect()
    } else {
        let uniform = 1.0 / k.min(probs.len()) as f64;
        top_k.into_iter().map(|(id, _)| (id, uniform)).collect()
    };

    // Now apply nucleus within top-k
    let mut cumulative = 0.0;
    let mut nucleus_end = 0;

    for (i, (_, prob)) in top_k_normalized.iter().enumerate() {
        cumulative += prob;
        nucleus_end = i + 1;
        if cumulative >= p {
            break;
        }
    }

    let nucleus: Vec<(usize, f64)> = top_k_normalized.into_iter().take(nucleus_end).collect();
    let nucleus_mass: f64 = nucleus.iter().map(|(_, p)| p).sum();
    let nucleus_size = nucleus.len();

    // Final renormalization
    let tokens: Vec<TokenProb> = if nucleus_mass > 0.0 {
        nucleus
            .into_iter()
            .map(|(token_id, prob)| TokenProb {
                token_id,
                prob: prob / nucleus_mass,
            })
            .collect()
    } else {
        let uniform = 1.0 / nucleus_size as f64;
        nucleus
            .into_iter()
            .map(|(token_id, _)| TokenProb {
                token_id,
                prob: uniform,
            })
            .collect()
    };

    NucleusResult {
        tokens,
        nucleus_mass,
        nucleus_size,
    }
}

// Softmax for logits conversion
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

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
    fn test_nucleus_filter_basic() {
        // Probs: [0.1, 0.2, 0.3, 0.4] - cumulative from top: 0.4, 0.7, 0.9, 1.0
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // p=0.5: need tokens until cumsum >= 0.5
        // Token 3 (0.4) alone is not enough, need token 1 (0.2) too: 0.4 + 0.3 = 0.7 >= 0.5
        let result = nucleus_filter(&probs, 0.5);
        assert_eq!(result.nucleus_size, 2);
        assert_eq!(result.tokens[0].token_id, 3); // highest
        assert_eq!(result.tokens[1].token_id, 2); // second highest
    }

    #[test]
    fn test_nucleus_filter_high_p() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // p=0.95: need most tokens
        let result = nucleus_filter(&probs, 0.95);
        assert_eq!(result.nucleus_size, 4); // need all to reach 0.95
    }

    #[test]
    fn test_nucleus_filter_low_p() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // p=0.3: just the top token (0.4 >= 0.3)
        let result = nucleus_filter(&probs, 0.3);
        assert_eq!(result.nucleus_size, 1);
        assert_eq!(result.tokens[0].token_id, 3);
    }

    #[test]
    fn test_nucleus_filter_renormalization() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let result = nucleus_filter(&probs, 0.5);

        // Nucleus has tokens 3 and 2 with probs 0.4 and 0.3
        // Renormalized: 0.4/0.7 and 0.3/0.7
        let sum: f64 = result.tokens.iter().map(|t| t.prob).sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_nucleus_sample_deterministic() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // With random=0, should pick highest prob token
        let token = nucleus_sample(&probs, 0.9, 0.0).unwrap();
        assert_eq!(token, 3);

        // With random close to 1, should pick from tail of nucleus
        let token = nucleus_sample(&probs, 0.99, 0.99).unwrap();
        // Should be one of the lower prob tokens in nucleus
        assert!(token <= 3);
    }

    #[test]
    fn test_nucleus_from_logits() {
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        let result = nucleus_from_logits(&logits, 0.5);

        // Highest logit (3.0) should be first
        assert_eq!(result.tokens[0].token_id, 3);
    }

    #[test]
    fn test_nucleus_empty() {
        let probs: Vec<f64> = vec![];
        let result = nucleus_filter(&probs, 0.9);
        assert!(result.tokens.is_empty());
        assert_eq!(result.nucleus_size, 0);
    }

    #[test]
    fn test_nucleus_p_zero() {
        let probs = vec![0.5, 0.5];
        let result = nucleus_filter(&probs, 0.0);
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_nucleus_p_one() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let result = nucleus_filter(&probs, 1.0);

        // p=1.0 should include all tokens
        assert_eq!(result.nucleus_size, 4);
    }

    #[test]
    fn test_nucleus_adapts_to_distribution() {
        // Peaked distribution: one dominant token
        let peaked = vec![0.01, 0.01, 0.01, 0.97];
        let result_peaked = nucleus_filter(&peaked, 0.9);
        assert_eq!(result_peaked.nucleus_size, 1); // just the dominant token

        // Flat distribution: many tokens needed
        let flat = vec![0.25, 0.25, 0.25, 0.25];
        let result_flat = nucleus_filter(&flat, 0.9);
        assert_eq!(result_flat.nucleus_size, 4); // need most tokens
    }

    #[test]
    fn test_top_k_nucleus_combined() {
        let probs = vec![0.05, 0.1, 0.15, 0.2, 0.5];

        // Top-3 then nucleus p=0.8
        let result = top_k_nucleus_filter(&probs, 3, 0.8);

        // Top-3 are: token 4 (0.5), token 3 (0.2), token 2 (0.15)
        // Renormalized in top-3: 0.5/0.85, 0.2/0.85, 0.15/0.85
        // Token 4 alone: 0.5/0.85 ≈ 0.588 < 0.8
        // Tokens 4+3: (0.5+0.2)/0.85 ≈ 0.824 >= 0.8
        assert_eq!(result.nucleus_size, 2);
        assert_eq!(result.tokens[0].token_id, 4);
        assert_eq!(result.tokens[1].token_id, 3);
    }

    #[test]
    fn test_top_k_nucleus_k_limits() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // k=2, p=1.0: should only have 2 tokens despite p=1.0
        let result = top_k_nucleus_filter(&probs, 2, 1.0);
        assert_eq!(result.nucleus_size, 2);
    }

    #[test]
    fn test_sampling_respects_nucleus() {
        // Only tokens 2 and 3 should ever be sampled with this distribution and p
        let probs = vec![0.05, 0.05, 0.4, 0.5];

        let mut seen = [false; 4];
        for i in 0..100 {
            let random = (i as f64) / 100.0;
            if let Some(token) = nucleus_sample(&probs, 0.85, random) {
                seen[token] = true;
            }
        }

        // Tokens 0 and 1 should not be in nucleus (0.5 + 0.4 = 0.9 >= 0.85)
        assert!(!seen[0]);
        assert!(!seen[1]);
        // Tokens 2 and 3 should be sampled
        assert!(seen[2]);
        assert!(seen[3]);
    }
}
