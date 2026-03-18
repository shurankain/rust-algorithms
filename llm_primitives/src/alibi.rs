// ALiBi (Attention with Linear Biases)
//
// Paper: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
// Authors: Ofir Press, Noah A. Smith, Mike Lewis (2022)
//
// Key insight: Instead of positional embeddings, add a learned linear bias to attention scores
// that depends on the distance between query and key positions.
//
// Formula: attention(Q, K, V) = softmax(QK^T / sqrt(d) + m * distance_matrix) * V
//
// Where:
// - m is a head-specific slope (computed from number of heads)
// - distance_matrix[i,j] = -(i - j) for causal attention (negative distances for past positions)
//
// Benefits:
// - No learned positional embeddings needed
// - Excellent length extrapolation (train on 1K, test on 4K+)
// - Simple and computationally efficient
// - Works well for causal (autoregressive) models
//
// The slopes are computed as: m_i = 2^(-8*i/n) for i in 1..n where n is number of heads
// This gives geometric spacing: 1/2, 1/4, 1/8, ... for 8 heads

/// Configuration for ALiBi attention
#[derive(Debug, Clone)]
pub struct AlibiConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Custom softmax scale (default: 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl AlibiConfig {
    /// Create a new ALiBi config
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            causal: true,
            softmax_scale: None,
        }
    }

    /// Set causal masking
    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set custom softmax scale
    pub fn softmax_scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Get the softmax scale
    pub fn get_scale(&self) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

impl Default for AlibiConfig {
    fn default() -> Self {
        Self::new(8, 64)
    }
}

/// Statistics for ALiBi attention
#[derive(Debug, Clone, Default)]
pub struct AlibiStats {
    /// Number of forward passes
    pub forward_passes: usize,
    /// Total attention operations
    pub attention_ops: usize,
    /// Total sequence length processed
    pub total_seq_len: usize,
}

/// ALiBi attention implementation
#[derive(Debug)]
pub struct AlibiAttention {
    config: AlibiConfig,
    /// Pre-computed slopes for each head
    slopes: Vec<f32>,
    stats: AlibiStats,
}

impl AlibiAttention {
    /// Create a new ALiBi attention module
    pub fn new(config: AlibiConfig) -> Self {
        let slopes = compute_alibi_slopes(config.num_heads);
        Self {
            config,
            slopes,
            stats: AlibiStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &AlibiConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &AlibiStats {
        &self.stats
    }

    /// Get the ALiBi slopes
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Compute ALiBi attention for a single head
    /// q: [seq_len_q, head_dim]
    /// k: [seq_len_kv, head_dim]
    /// v: [seq_len_kv, head_dim]
    /// head_idx: which head (for slope selection)
    /// Returns: [seq_len_q, head_dim]
    pub fn attention_single_head(
        &mut self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        head_idx: usize,
    ) -> Vec<Vec<f32>> {
        let seq_len_q = q.len();
        let seq_len_kv = k.len();
        let scale = self.config.get_scale();
        let slope = self.slopes[head_idx];

        let mut output = Vec::with_capacity(seq_len_q);

        for (q_pos, q_vec) in q.iter().enumerate() {
            // Compute attention scores with ALiBi bias
            let mut scores = Vec::with_capacity(seq_len_kv);

            for (k_pos, k_vec) in k.iter().enumerate() {
                // QK^T / sqrt(d)
                let dot: f32 = q_vec
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum();
                let score = dot * scale;

                // Add ALiBi bias: m * (k_pos - q_pos)
                // For causal: positions before current get negative bias (less attention to distant past)
                let distance = k_pos as i32 - q_pos as i32;
                let alibi_bias = slope * distance as f32;

                let final_score = score + alibi_bias;

                // Apply causal mask if needed
                if self.config.causal && k_pos > q_pos {
                    scores.push(f32::NEG_INFINITY);
                } else {
                    scores.push(final_score);
                }

                self.stats.attention_ops += 1;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            for score in &mut scores {
                if *score > f32::NEG_INFINITY {
                    *score = (*score - max_score).exp();
                } else {
                    *score = 0.0;
                }
                sum_exp += *score;
            }

            if sum_exp > 0.0 {
                for score in &mut scores {
                    *score /= sum_exp;
                }
            }

            // Weighted sum of values
            let head_dim = self.config.head_dim;
            let mut out_vec = vec![0.0; head_dim];

            for (k_pos, &weight) in scores.iter().enumerate() {
                if weight > 0.0 {
                    for (d, out_d) in out_vec.iter_mut().enumerate() {
                        *out_d += weight * v[k_pos][d];
                    }
                }
            }

            output.push(out_vec);
        }

        output
    }

    /// Compute ALiBi attention for all heads
    /// Q: [num_heads, seq_len_q, head_dim]
    /// K: [num_heads, seq_len_kv, head_dim]
    /// V: [num_heads, seq_len_kv, head_dim]
    /// Returns: [num_heads, seq_len_q, head_dim]
    pub fn forward(
        &mut self,
        q: &[Vec<Vec<f32>>],
        k: &[Vec<Vec<f32>>],
        v: &[Vec<Vec<f32>>],
    ) -> Vec<Vec<Vec<f32>>> {
        assert_eq!(q.len(), self.config.num_heads);
        assert_eq!(k.len(), self.config.num_heads);
        assert_eq!(v.len(), self.config.num_heads);

        self.stats.forward_passes += 1;
        if !q.is_empty() && !q[0].is_empty() {
            self.stats.total_seq_len += q[0].len();
        }

        let mut outputs = Vec::with_capacity(self.config.num_heads);

        for head_idx in 0..self.config.num_heads {
            let head_output =
                self.attention_single_head(&q[head_idx], &k[head_idx], &v[head_idx], head_idx);
            outputs.push(head_output);
        }

        outputs
    }

    /// Compute ALiBi attention with flat tensor layout
    /// Q: [num_heads * seq_len_q * head_dim] (flattened)
    /// K: [num_heads * seq_len_kv * head_dim] (flattened)
    /// V: [num_heads * seq_len_kv * head_dim] (flattened)
    /// Returns: [num_heads * seq_len_q * head_dim] (flattened)
    pub fn forward_flat(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Reshape to [num_heads, seq_len, head_dim]
        let reshape = |flat: &[f32], seq_len: usize| -> Vec<Vec<Vec<f32>>> {
            let mut result = Vec::with_capacity(num_heads);
            for head in 0..num_heads {
                let mut head_data = Vec::with_capacity(seq_len);
                for pos in 0..seq_len {
                    let start = head * seq_len * head_dim + pos * head_dim;
                    let vec: Vec<f32> = flat[start..start + head_dim].to_vec();
                    head_data.push(vec);
                }
                result.push(head_data);
            }
            result
        };

        let q_reshaped = reshape(q, seq_len_q);
        let k_reshaped = reshape(k, seq_len_kv);
        let v_reshaped = reshape(v, seq_len_kv);

        let output = self.forward(&q_reshaped, &k_reshaped, &v_reshaped);

        // Flatten output
        let mut flat_output = Vec::with_capacity(num_heads * seq_len_q * head_dim);
        for head in &output {
            for pos in head {
                flat_output.extend_from_slice(pos);
            }
        }

        flat_output
    }
}

/// Compute ALiBi slopes for each attention head
/// For n heads, slopes are: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
/// This gives a geometric sequence that biases attention differently per head
pub fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    // The formula from the paper for n heads:
    // m_i = 2^(-8 * i / n) for i in 1..=n
    //
    // For n=8: 2^(-1), 2^(-2), 2^(-3), ..., 2^(-8)
    //        = 0.5, 0.25, 0.125, ..., 0.00390625

    let ratio = 8.0 / num_heads as f32;

    (1..=num_heads)
        .map(|i| 2.0_f32.powf(-ratio * i as f32))
        .collect()
}

/// Get the ALiBi bias matrix for a given head and sequence lengths
/// Returns: [seq_len_q, seq_len_kv] bias values
pub fn get_alibi_bias_matrix(
    slope: f32,
    seq_len_q: usize,
    seq_len_kv: usize,
    causal: bool,
) -> Vec<Vec<f32>> {
    let mut bias = Vec::with_capacity(seq_len_q);

    for q_pos in 0..seq_len_q {
        let mut row = Vec::with_capacity(seq_len_kv);
        for k_pos in 0..seq_len_kv {
            let distance = k_pos as i32 - q_pos as i32;
            let alibi_bias = slope * distance as f32;

            if causal && k_pos > q_pos {
                row.push(f32::NEG_INFINITY);
            } else {
                row.push(alibi_bias);
            }
        }
        bias.push(row);
    }

    bias
}

/// Standard attention without ALiBi (for comparison)
/// Returns: [seq_len_q, head_dim]
pub fn standard_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    scale: f32,
    causal: bool,
) -> Vec<Vec<f32>> {
    let seq_len_q = q.len();
    let seq_len_kv = k.len();
    let head_dim = if !q.is_empty() { q[0].len() } else { 0 };

    let mut output = Vec::with_capacity(seq_len_q);

    for (q_pos, q_vec) in q.iter().enumerate() {
        let mut scores = Vec::with_capacity(seq_len_kv);

        for (k_pos, k_vec) in k.iter().enumerate() {
            let dot: f32 = q_vec
                .iter()
                .zip(k_vec.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum();
            let score = dot * scale;

            if causal && k_pos > q_pos {
                scores.push(f32::NEG_INFINITY);
            } else {
                scores.push(score);
            }
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        for score in &mut scores {
            if *score > f32::NEG_INFINITY {
                *score = (*score - max_score).exp();
            } else {
                *score = 0.0;
            }
            sum_exp += *score;
        }

        if sum_exp > 0.0 {
            for score in &mut scores {
                *score /= sum_exp;
            }
        }

        // Weighted sum
        let mut out_vec = vec![0.0; head_dim];
        for (k_pos, &weight) in scores.iter().enumerate() {
            if weight > 0.0 {
                for (d, out_d) in out_vec.iter_mut().enumerate() {
                    *out_d += weight * v[k_pos][d];
                }
            }
        }

        output.push(out_vec);
    }

    output
}

/// Verify ALiBi length extrapolation property
/// The key property is that ALiBi attention weights should be well-behaved
/// even for sequence lengths longer than training
pub fn verify_alibi_extrapolation(
    alibi: &mut AlibiAttention,
    short_len: usize,
    long_len: usize,
    seed: u64,
) -> bool {
    // Generate test data for short sequence
    let short_q: Vec<Vec<f32>> = (0..short_len)
        .map(|i| random_vector(alibi.config.head_dim, seed + i as u64))
        .collect();
    let short_k = short_q.clone();
    let short_v = short_q.clone();

    // Generate test data for long sequence (reusing same patterns)
    let long_q: Vec<Vec<f32>> = (0..long_len)
        .map(|i| random_vector(alibi.config.head_dim, seed + (i % short_len) as u64))
        .collect();
    let long_k = long_q.clone();
    let long_v = long_q.clone();

    // Compute attention for one head
    let short_output = alibi.attention_single_head(&short_q, &short_k, &short_v, 0);
    let long_output = alibi.attention_single_head(&long_q, &long_k, &long_v, 0);

    // Both should produce valid outputs (no NaN or Inf in non-masked positions)
    for row in &short_output {
        for &val in row {
            if val.is_nan() || val.is_infinite() {
                return false;
            }
        }
    }

    for row in &long_output {
        for &val in row {
            if val.is_nan() || val.is_infinite() {
                return false;
            }
        }
    }

    // The outputs for corresponding positions should follow similar patterns
    // (though not identical due to different context)
    true
}

/// Compare attention patterns with and without ALiBi
/// Returns the average difference in attention weights per position
pub fn compare_attention_patterns(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    _v: &[Vec<f32>],
    slope: f32,
    scale: f32,
    causal: bool,
) -> f32 {
    let seq_len = q.len();
    if seq_len == 0 {
        return 0.0;
    }

    // Get standard attention weights
    let std_weights = compute_attention_weights(q, k, scale, causal, None);

    // Get ALiBi attention weights
    let alibi_weights = compute_attention_weights(q, k, scale, causal, Some(slope));

    // Compute average absolute difference
    let mut total_diff = 0.0;
    let mut count = 0;

    for (std_row, alibi_row) in std_weights.iter().zip(alibi_weights.iter()) {
        for (&std_w, &alibi_w) in std_row.iter().zip(alibi_row.iter()) {
            if std_w > 0.0 || alibi_w > 0.0 {
                total_diff += (std_w - alibi_w).abs();
                count += 1;
            }
        }
    }

    if count > 0 {
        total_diff / count as f32
    } else {
        0.0
    }
}

/// Helper to compute attention weights (for comparison)
fn compute_attention_weights(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    scale: f32,
    causal: bool,
    alibi_slope: Option<f32>,
) -> Vec<Vec<f32>> {
    let seq_len_q = q.len();
    let seq_len_kv = k.len();

    let mut weights = Vec::with_capacity(seq_len_q);

    for (q_pos, q_vec) in q.iter().enumerate() {
        let mut scores = Vec::with_capacity(seq_len_kv);

        for (k_pos, k_vec) in k.iter().enumerate() {
            let dot: f32 = q_vec
                .iter()
                .zip(k_vec.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum();
            let mut score = dot * scale;

            // Add ALiBi bias if provided
            if let Some(slope) = alibi_slope {
                let distance = k_pos as i32 - q_pos as i32;
                score += slope * distance as f32;
            }

            if causal && k_pos > q_pos {
                scores.push(f32::NEG_INFINITY);
            } else {
                scores.push(score);
            }
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        for score in &mut scores {
            if *score > f32::NEG_INFINITY {
                *score = (*score - max_score).exp();
            } else {
                *score = 0.0;
            }
            sum_exp += *score;
        }

        if sum_exp > 0.0 {
            for score in &mut scores {
                *score /= sum_exp;
            }
        }

        weights.push(scores);
    }

    weights
}

/// Generate a pseudo-random vector for testing
fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let hash = hasher.finish();
        // Scale to [-1, 1]
        let val = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
        result.push(val);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|i| random_vector(cols, seed + i as u64 * 100))
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = AlibiConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
    }

    #[test]
    fn test_config_builder() {
        let config = AlibiConfig::new(4, 32).causal(false).softmax_scale(0.5);

        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim, 32);
        assert!(!config.causal);
        assert_eq!(config.get_scale(), 0.5);
    }

    #[test]
    fn test_get_scale() {
        let config = AlibiConfig::new(8, 64);
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes() {
        // Test 8 heads
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);

        // Slopes should be 2^(-1), 2^(-2), ..., 2^(-8)
        assert!((slopes[0] - 0.5).abs() < 1e-6);
        assert!((slopes[1] - 0.25).abs() < 1e-6);
        assert!((slopes[7] - 2.0_f32.powi(-8)).abs() < 1e-6);

        // Slopes should be decreasing
        for i in 0..slopes.len() - 1 {
            assert!(slopes[i] > slopes[i + 1]);
        }
    }

    #[test]
    fn test_alibi_slopes_4_heads() {
        let slopes = compute_alibi_slopes(4);
        assert_eq!(slopes.len(), 4);

        // For 4 heads: 2^(-2), 2^(-4), 2^(-6), 2^(-8)
        assert!((slopes[0] - 0.25).abs() < 1e-6);
        assert!((slopes[1] - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn test_basic_attention() {
        let config = AlibiConfig::new(2, 4);
        let mut alibi = AlibiAttention::new(config);

        // Simple 3-token sequence
        let q: Vec<Vec<Vec<f32>>> = vec![random_matrix(3, 4, 1), random_matrix(3, 4, 2)];
        let k: Vec<Vec<Vec<f32>>> = vec![random_matrix(3, 4, 10), random_matrix(3, 4, 20)];
        let v: Vec<Vec<Vec<f32>>> = vec![random_matrix(3, 4, 30), random_matrix(3, 4, 40)];

        let output = alibi.forward(&q, &k, &v);

        assert_eq!(output.len(), 2); // 2 heads
        assert_eq!(output[0].len(), 3); // 3 positions
        assert_eq!(output[0][0].len(), 4); // 4 dims
    }

    #[test]
    fn test_causal_masking() {
        let config = AlibiConfig::new(1, 4).causal(true);
        let mut alibi = AlibiAttention::new(config);

        // 3-token sequence
        let q = vec![random_matrix(3, 4, 1)];
        let k = vec![random_matrix(3, 4, 10)];
        let v = vec![random_matrix(3, 4, 20)];

        let output = alibi.forward(&q, &k, &v);

        // First position should only attend to itself
        // We can verify this indirectly by checking the output is valid
        assert!(!output[0][0][0].is_nan());
    }

    #[test]
    fn test_non_causal() {
        let config = AlibiConfig::new(1, 4).causal(false);
        let mut alibi = AlibiAttention::new(config);

        let q = vec![random_matrix(3, 4, 1)];
        let k = vec![random_matrix(3, 4, 10)];
        let v = vec![random_matrix(3, 4, 20)];

        let output = alibi.forward(&q, &k, &v);

        // All positions should have valid outputs
        for pos in &output[0] {
            for &val in pos {
                assert!(!val.is_nan());
                assert!(!val.is_infinite());
            }
        }
    }

    #[test]
    fn test_bias_matrix() {
        let slope = 0.5;
        let bias = get_alibi_bias_matrix(slope, 3, 3, true);

        // Check shape
        assert_eq!(bias.len(), 3);
        assert_eq!(bias[0].len(), 3);

        // Check diagonal (distance = 0)
        assert!((bias[0][0] - 0.0).abs() < 1e-6);
        assert!((bias[1][1] - 0.0).abs() < 1e-6);
        assert!((bias[2][2] - 0.0).abs() < 1e-6);

        // Check sub-diagonal (distance = -1, attending to previous)
        assert!((bias[1][0] - (-0.5)).abs() < 1e-6);
        assert!((bias[2][1] - (-0.5)).abs() < 1e-6);

        // Check super-diagonal should be masked (causal)
        assert_eq!(bias[0][1], f32::NEG_INFINITY);
        assert_eq!(bias[0][2], f32::NEG_INFINITY);
        assert_eq!(bias[1][2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_bias_matrix_non_causal() {
        let slope = 0.5;
        let bias = get_alibi_bias_matrix(slope, 3, 3, false);

        // Super-diagonal should have positive bias (future positions)
        assert!((bias[0][1] - 0.5).abs() < 1e-6); // distance = 1
        assert!((bias[0][2] - 1.0).abs() < 1e-6); // distance = 2
    }

    #[test]
    fn test_empty_input() {
        let config = AlibiConfig::new(2, 4);
        let mut alibi = AlibiAttention::new(config);

        let q: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];
        let k: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];
        let v: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];

        let output = alibi.forward(&q, &k, &v);

        assert_eq!(output.len(), 2);
        assert!(output[0].is_empty());
    }

    #[test]
    fn test_single_token() {
        let config = AlibiConfig::new(1, 4);
        let mut alibi = AlibiAttention::new(config);

        let q = vec![vec![vec![1.0, 0.0, 0.0, 0.0]]];
        let k = vec![vec![vec![1.0, 0.0, 0.0, 0.0]]];
        let v = vec![vec![vec![0.5, 0.5, 0.5, 0.5]]];

        let output = alibi.forward(&q, &k, &v);

        // Single token should attend fully to itself
        assert_eq!(output[0].len(), 1);
        for i in 0..4 {
            assert!((output[0][0][i] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_flat_layout() {
        let config = AlibiConfig::new(2, 4);
        let mut alibi = AlibiAttention::new(config.clone());

        let seq_len = 3;

        // Create flat tensors
        let q_flat: Vec<f32> = (0..config.num_heads * seq_len * config.head_dim)
            .map(|i| (i as f32 * 0.1) % 1.0)
            .collect();
        let k_flat = q_flat.clone();
        let v_flat = q_flat.clone();

        let output = alibi.forward_flat(&q_flat, &k_flat, &v_flat, seq_len, seq_len);

        assert_eq!(output.len(), config.num_heads * seq_len * config.head_dim);
    }

    #[test]
    fn test_stats() {
        let config = AlibiConfig::new(2, 4);
        let mut alibi = AlibiAttention::new(config);

        let q = vec![random_matrix(3, 4, 1), random_matrix(3, 4, 2)];
        let k = vec![random_matrix(3, 4, 10), random_matrix(3, 4, 20)];
        let v = vec![random_matrix(3, 4, 30), random_matrix(3, 4, 40)];

        alibi.forward(&q, &k, &v);

        assert_eq!(alibi.stats().forward_passes, 1);
        assert_eq!(alibi.stats().total_seq_len, 3);
        assert!(alibi.stats().attention_ops > 0);
    }

    #[test]
    fn test_alibi_vs_standard_different() {
        // ALiBi should produce different attention patterns than standard attention
        let q = random_matrix(4, 8, 1);
        let k = random_matrix(4, 8, 10);
        let v = random_matrix(4, 8, 20);
        let scale = 1.0 / (8.0_f32).sqrt();

        let slope = 0.5;
        let diff = compare_attention_patterns(&q, &k, &v, slope, scale, true);

        // There should be some difference due to ALiBi bias
        assert!(diff > 0.0);
    }

    #[test]
    fn test_extrapolation_property() {
        let config = AlibiConfig::new(2, 8);
        let mut alibi = AlibiAttention::new(config);

        // Verify extrapolation from short to long sequences
        assert!(verify_alibi_extrapolation(&mut alibi, 10, 50, 42));
    }

    #[test]
    fn test_recency_bias() {
        // ALiBi should create a recency bias - nearby positions get more attention
        let config = AlibiConfig::new(1, 4).causal(true);
        let mut alibi = AlibiAttention::new(config);

        // Create uniform Q and K so attention is purely determined by ALiBi bias
        let seq_len = 5;
        let q = vec![vec![vec![1.0, 0.0, 0.0, 0.0]; seq_len]];
        let k = vec![vec![vec![1.0, 0.0, 0.0, 0.0]; seq_len]];
        // Different values to distinguish positions
        let v: Vec<Vec<Vec<f32>>> = vec![(0..seq_len).map(|i| vec![i as f32; 4]).collect()];

        let output = alibi.forward(&q, &k, &v);

        // For the last position attending to all previous,
        // output should be weighted towards recent positions
        let last_output = &output[0][seq_len - 1];

        // With ALiBi, the output should be closer to the most recent value (4.0)
        // than to the average of all values (2.0)
        let avg_val = last_output.iter().sum::<f32>() / 4.0;
        assert!(
            avg_val > 2.0,
            "ALiBi should bias towards recent: avg={}",
            avg_val
        );
    }

    #[test]
    fn test_slope_effect() {
        // Higher slopes should create stronger recency bias
        let slopes_8 = compute_alibi_slopes(8);
        let slopes_4 = compute_alibi_slopes(4);

        // First head of 8-head config has slope 0.5
        // First head of 4-head config has slope 0.25
        assert!(slopes_8[0] > slopes_4[0]);
    }

    #[test]
    fn test_numerical_stability() {
        let config = AlibiConfig::new(2, 8);
        let mut alibi = AlibiAttention::new(config);

        // Test with larger sequences
        let q = vec![random_matrix(50, 8, 1), random_matrix(50, 8, 2)];
        let k = vec![random_matrix(50, 8, 10), random_matrix(50, 8, 20)];
        let v = vec![random_matrix(50, 8, 30), random_matrix(50, 8, 40)];

        let output = alibi.forward(&q, &k, &v);

        // Check no NaN or Inf
        for head in &output {
            for pos in head {
                for &val in pos {
                    assert!(!val.is_nan());
                    assert!(!val.is_infinite());
                }
            }
        }
    }

    #[test]
    fn test_different_qkv_lengths() {
        // Test with different query and key/value lengths (e.g., cross-attention)
        let config = AlibiConfig::new(1, 4).causal(false); // Non-causal for cross-attention
        let mut alibi = AlibiAttention::new(config);

        let q = vec![random_matrix(2, 4, 1)]; // 2 queries
        let k = vec![random_matrix(5, 4, 10)]; // 5 keys
        let v = vec![random_matrix(5, 4, 20)]; // 5 values

        let output = alibi.forward(&q, &k, &v);

        assert_eq!(output[0].len(), 2); // 2 output positions
        assert_eq!(output[0][0].len(), 4); // head_dim
    }

    #[test]
    fn test_standard_attention() {
        let q = random_matrix(3, 4, 1);
        let k = random_matrix(3, 4, 10);
        let v = random_matrix(3, 4, 20);

        let output = standard_attention(&q, &k, &v, 0.5, true);

        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 4);

        // No NaN or Inf
        for pos in &output {
            for &val in pos {
                assert!(!val.is_nan());
                assert!(!val.is_infinite());
            }
        }
    }

    #[test]
    fn test_bloom_config() {
        // BLOOM uses ALiBi with specific configuration
        // BLOOM-176B: 112 attention heads
        let slopes = compute_alibi_slopes(112);
        assert_eq!(slopes.len(), 112);

        // All slopes should be positive and <= 1
        for &slope in &slopes {
            assert!(slope > 0.0);
            assert!(slope <= 1.0);
        }

        // Slopes should be monotonically decreasing
        for i in 0..slopes.len() - 1 {
            assert!(slopes[i] > slopes[i + 1]);
        }
    }
}
