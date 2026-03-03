// Temperature Scaling
//
// Controls randomness in LLM output by scaling logits before softmax.
// Temperature modifies the "sharpness" of the probability distribution.
//
// Formula: P(token_i) = softmax(logit_i / T)
//
// Temperature effects:
// - T = 1.0: Standard softmax, no modification
// - T < 1.0: Sharper distribution, more deterministic (peaks become higher)
// - T > 1.0: Flatter distribution, more random (uniform-like)
// - T → 0: Approaches greedy/argmax (one-hot distribution)
// - T → ∞: Approaches uniform distribution
//
// Common values:
// - T = 0.7-0.9: Slightly more focused, good for factual responses
// - T = 1.0: Default, balanced creativity/coherence
// - T = 1.2-1.5: More creative/diverse, good for brainstorming
//
// Time: O(n) for scaling
// Space: O(n) for output probabilities

use std::cmp::Ordering;

// Apply temperature scaling to logits and return probabilities
pub fn apply_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    // Handle edge cases
    if temperature <= 0.0 {
        // T → 0: return one-hot for max logit (greedy)
        return greedy_distribution(logits);
    }

    if temperature == 1.0 {
        // No scaling needed, just softmax
        return softmax(logits);
    }

    // Scale logits by temperature
    let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();
    softmax(&scaled)
}

// Get scaled logits without converting to probabilities
pub fn scale_logits(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() || temperature <= 0.0 {
        return logits.to_vec();
    }

    logits.iter().map(|&x| x / temperature).collect()
}

// Sample from temperature-scaled distribution
pub fn temperature_sample(logits: &[f64], temperature: f64, random: f64) -> Option<usize> {
    let probs = apply_temperature(logits, temperature);

    if probs.is_empty() {
        return None;
    }

    let random = random.clamp(0.0, 0.9999999);
    let mut cumulative = 0.0;

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if random < cumulative {
            return Some(i);
        }
    }

    Some(probs.len() - 1)
}

// Convenience: get the "effective" number of choices at a temperature
// Uses perplexity: 2^entropy, measures how "spread out" the distribution is
pub fn effective_choices(logits: &[f64], temperature: f64) -> f64 {
    let probs = apply_temperature(logits, temperature);

    if probs.is_empty() {
        return 0.0;
    }

    // Calculate entropy: H = -sum(p * log2(p))
    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();

    // Perplexity = 2^entropy
    2.0_f64.powf(entropy)
}

// Compare distributions at different temperatures
#[derive(Debug, Clone)]
pub struct TemperatureComparison {
    pub temperature: f64,
    pub entropy: f64,
    pub effective_choices: f64,
    pub max_prob: f64,
    pub min_prob: f64,
}

pub fn analyze_temperature(logits: &[f64], temperature: f64) -> TemperatureComparison {
    let probs = apply_temperature(logits, temperature);

    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);

    let min_prob = probs.iter().cloned().fold(1.0_f64, f64::min);

    TemperatureComparison {
        temperature,
        entropy,
        effective_choices: 2.0_f64.powf(entropy),
        max_prob,
        min_prob,
    }
}

// Helper: create one-hot distribution for greedy (T→0)
fn greedy_distribution(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    let max_idx = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut result = vec![0.0; logits.len()];
    result[max_idx] = 1.0;
    result
}

// Softmax with numerical stability
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

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_temperature_one_is_standard_softmax() {
        let logits = vec![1.0, 2.0, 3.0];

        let t1 = apply_temperature(&logits, 1.0);
        let standard = softmax(&logits);

        for (a, b) in t1.iter().zip(standard.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_low_temperature_sharpens() {
        let logits = vec![1.0, 2.0, 3.0];

        let t_high = apply_temperature(&logits, 2.0);
        let t_low = apply_temperature(&logits, 0.5);

        // Low temperature should have higher max probability
        let max_high = t_high.iter().cloned().fold(0.0_f64, f64::max);
        let max_low = t_low.iter().cloned().fold(0.0_f64, f64::max);

        assert!(max_low > max_high);
    }

    #[test]
    fn test_high_temperature_flattens() {
        let logits = vec![1.0, 2.0, 3.0];

        let t1 = apply_temperature(&logits, 1.0);
        let t_high = apply_temperature(&logits, 5.0);

        // High temperature should be more uniform (lower max)
        let max_t1 = t1.iter().cloned().fold(0.0_f64, f64::max);
        let max_high = t_high.iter().cloned().fold(0.0_f64, f64::max);

        assert!(max_high < max_t1);

        // Should approach uniform (1/3 ≈ 0.33)
        let uniform = 1.0 / 3.0;
        for &p in &t_high {
            assert!(approx_eq(p, uniform, 0.15)); // within 0.15 of uniform
        }
    }

    #[test]
    fn test_zero_temperature_is_greedy() {
        let logits = vec![1.0, 3.0, 2.0];

        let probs = apply_temperature(&logits, 0.0);

        // Should be one-hot at max logit (index 1)
        assert!(approx_eq(probs[0], 0.0, 1e-10));
        assert!(approx_eq(probs[1], 1.0, 1e-10));
        assert!(approx_eq(probs[2], 0.0, 1e-10));
    }

    #[test]
    fn test_negative_temperature_is_greedy() {
        let logits = vec![1.0, 3.0, 2.0];

        let probs = apply_temperature(&logits, -0.5);

        // Negative temperature treated as greedy
        assert!(approx_eq(probs[1], 1.0, 1e-10));
    }

    #[test]
    fn test_scale_logits() {
        let logits = vec![2.0, 4.0, 6.0];

        let scaled = scale_logits(&logits, 2.0);

        assert!(approx_eq(scaled[0], 1.0, 1e-10));
        assert!(approx_eq(scaled[1], 2.0, 1e-10));
        assert!(approx_eq(scaled[2], 3.0, 1e-10));
    }

    #[test]
    fn test_temperature_sample_deterministic() {
        let logits = vec![1.0, 2.0, 3.0];

        // With very low temperature, distribution peaks at max logit
        // random=0.0 picks first token with nonzero probability
        // At T=0.1, token 2 (logit=3) has nearly all probability mass
        let probs_low = apply_temperature(&logits, 0.1);
        assert!(probs_low[2] > 0.99); // token 2 dominates

        // Sample should pick token 2 for most random values
        let token = temperature_sample(&logits, 0.1, 0.5).unwrap();
        assert_eq!(token, 2); // highest logit

        // With high temperature, distribution flattens
        let probs_high = apply_temperature(&logits, 10.0);
        // All probabilities should be close to 1/3
        for &p in &probs_high {
            assert!(p > 0.2 && p < 0.5);
        }
    }

    #[test]
    fn test_effective_choices() {
        // Uniform distribution over 4 tokens
        let uniform_logits = vec![0.0, 0.0, 0.0, 0.0];
        let choices = effective_choices(&uniform_logits, 1.0);
        assert!(approx_eq(choices, 4.0, 0.01)); // should be ~4

        // Very peaked distribution
        let peaked_logits = vec![0.0, 0.0, 0.0, 100.0];
        let choices = effective_choices(&peaked_logits, 1.0);
        assert!(choices < 1.1); // should be ~1
    }

    #[test]
    fn test_analyze_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let low = analyze_temperature(&logits, 0.5);
        let high = analyze_temperature(&logits, 2.0);

        // Lower temperature = lower entropy = fewer effective choices
        assert!(low.entropy < high.entropy);
        assert!(low.effective_choices < high.effective_choices);
        assert!(low.max_prob > high.max_prob);
    }

    #[test]
    fn test_empty_logits() {
        let logits: Vec<f64> = vec![];

        assert!(apply_temperature(&logits, 1.0).is_empty());
        assert!(scale_logits(&logits, 1.0).is_empty());
        assert!(temperature_sample(&logits, 1.0, 0.5).is_none());
        assert!(approx_eq(effective_choices(&logits, 1.0), 0.0, 1e-10));
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let probs = apply_temperature(&logits, temp);
            let sum: f64 = probs.iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-9), "T={} sum={}", temp, sum);
        }
    }

    #[test]
    fn test_temperature_preserves_ordering() {
        let logits = vec![1.0, 4.0, 2.0, 3.0];

        for temp in [0.5, 1.0, 2.0] {
            let probs = apply_temperature(&logits, temp);

            // Token 1 (logit=4) should always have highest prob
            // Token 0 (logit=1) should always have lowest prob
            assert!(probs[1] > probs[3]); // 4 > 3
            assert!(probs[3] > probs[2]); // 3 > 2
            assert!(probs[2] > probs[0]); // 2 > 1
        }
    }

    #[test]
    fn test_numerical_stability() {
        // Large logits should not overflow
        let large = vec![1000.0, 1001.0, 1002.0];
        let probs = apply_temperature(&large, 1.0);
        assert!(probs.iter().all(|&p| p.is_finite()));

        let sum: f64 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-9));

        // Small temperature with large logits
        let probs_low = apply_temperature(&large, 0.1);
        assert!(probs_low.iter().all(|&p| p.is_finite()));
    }
}
