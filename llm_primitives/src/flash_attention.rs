// Flash Attention
//
// IO-aware exact attention algorithm that achieves 2-4x speedup over standard attention
// by reducing memory reads/writes through tiling. From FlashAttention paper (Dao et al., 2022).
//
// Key concepts:
// - Tiling: Process attention in blocks that fit in fast memory (SRAM/cache)
// - Online softmax: Compute softmax incrementally without materializing full attention matrix
// - Recomputation: Trade compute for memory - recompute attention during backward pass
//
// Memory complexity:
// - Standard attention: O(N²) for N sequence length
// - Flash attention: O(N) - linear in sequence length!
//
// Algorithm overview:
// 1. Split Q, K, V into blocks
// 2. For each Q block:
//    a. Initialize output accumulator and softmax statistics
//    b. For each K, V block:
//       - Compute block attention scores (Q_block @ K_block^T)
//       - Update running max and softmax denominator (online softmax)
//       - Accumulate weighted values
//    c. Rescale output by final softmax denominator
//
// This implementation is a CPU reference - actual speedups require GPU with SRAM tiling

/// Configuration for Flash Attention
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for Q (number of query tokens per block)
    pub block_size_q: usize,
    /// Block size for KV (number of key/value tokens per block)
    pub block_size_kv: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Softmax scale (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl FlashAttentionConfig {
    /// Create a new configuration
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            num_heads,
            head_dim,
            causal: false,
            softmax_scale: None,
        }
    }

    /// Set block size for Q
    pub fn block_q(mut self, size: usize) -> Self {
        self.block_size_q = size;
        self
    }

    /// Set block size for KV
    pub fn block_kv(mut self, size: usize) -> Self {
        self.block_size_kv = size;
        self
    }

    /// Enable causal masking
    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set custom softmax scale
    pub fn scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Get the softmax scale (default: 1/sqrt(head_dim))
    pub fn get_scale(&self) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            num_heads: 8,
            head_dim: 64,
            causal: false,
            softmax_scale: None,
        }
    }
}

/// Statistics from Flash Attention computation
#[derive(Debug, Clone, Default)]
pub struct FlashAttentionStats {
    /// Number of Q blocks processed
    pub q_blocks: usize,
    /// Number of KV blocks processed
    pub kv_blocks: usize,
    /// Total block operations (Q_blocks * KV_blocks)
    pub total_block_ops: usize,
    /// Memory saved vs standard attention (approximate)
    pub memory_saved_ratio: f32,
}

/// Online softmax state for a single position
/// Tracks running max and sum for numerically stable softmax
#[derive(Debug, Clone)]
struct OnlineSoftmaxState {
    /// Running maximum score
    max_score: f32,
    /// Running sum of exp(score - max)
    sum_exp: f32,
    /// Accumulated weighted output
    output: Vec<f32>,
}

impl OnlineSoftmaxState {
    fn new(dim: usize) -> Self {
        Self {
            max_score: f32::NEG_INFINITY,
            sum_exp: 0.0,
            output: vec![0.0; dim],
        }
    }

    /// Update state with new scores and values
    /// scores: attention scores for this block
    /// values: value vectors for this block
    fn update(&mut self, scores: &[f32], values: &[Vec<f32>]) {
        if scores.is_empty() {
            return;
        }

        // Find new max
        let block_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute new max and correction factor
        let new_max = self.max_score.max(block_max);

        // Rescale previous sum and output
        if self.sum_exp > 0.0 {
            let correction = (self.max_score - new_max).exp();
            self.sum_exp *= correction;
            for o in &mut self.output {
                *o *= correction;
            }
        }

        // Add new contributions
        for (i, &score) in scores.iter().enumerate() {
            let weight = (score - new_max).exp();
            self.sum_exp += weight;

            // Accumulate weighted value
            for (j, &v) in values[i].iter().enumerate() {
                self.output[j] += weight * v;
            }
        }

        self.max_score = new_max;
    }

    /// Finalize and return normalized output
    fn finalize(self) -> Vec<f32> {
        if self.sum_exp == 0.0 {
            return self.output;
        }
        self.output.into_iter().map(|o| o / self.sum_exp).collect()
    }
}

/// Flash Attention computation
///
/// Computes attention with tiled algorithm for memory efficiency.
///
/// Input shapes:
/// - Q: [batch_size, num_heads, seq_len_q, head_dim]
/// - K: [batch_size, num_heads, seq_len_kv, head_dim]
/// - V: [batch_size, num_heads, seq_len_kv, head_dim]
///
/// Output shape: [batch_size, num_heads, seq_len_q, head_dim]
pub struct FlashAttention {
    config: FlashAttentionConfig,
    stats: FlashAttentionStats,
}

impl FlashAttention {
    /// Create a new Flash Attention instance
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self {
            config,
            stats: FlashAttentionStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Get statistics from last computation
    pub fn stats(&self) -> &FlashAttentionStats {
        &self.stats
    }

    /// Compute attention for a single head
    /// Q: [seq_len_q, head_dim]
    /// K: [seq_len_kv, head_dim]
    /// V: [seq_len_kv, head_dim]
    /// Returns: [seq_len_q, head_dim]
    pub fn forward_single_head(
        &mut self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let seq_len_q = q.len();
        let seq_len_kv = k.len();
        let head_dim = self.config.head_dim;
        let scale = self.config.get_scale();

        if seq_len_q == 0 || seq_len_kv == 0 {
            return vec![];
        }

        // Verify dimensions
        assert!(q.iter().all(|qv| qv.len() == head_dim));
        assert!(k.iter().all(|kv| kv.len() == head_dim));
        assert!(v.iter().all(|vv| vv.len() == head_dim));

        // Initialize output states for each query position
        let mut states: Vec<OnlineSoftmaxState> = (0..seq_len_q)
            .map(|_| OnlineSoftmaxState::new(head_dim))
            .collect();

        // Process in blocks
        let num_q_blocks = seq_len_q.div_ceil(self.config.block_size_q);
        let num_kv_blocks = seq_len_kv.div_ceil(self.config.block_size_kv);

        self.stats.q_blocks = num_q_blocks;
        self.stats.kv_blocks = num_kv_blocks;
        self.stats.total_block_ops = num_q_blocks * num_kv_blocks;

        // For each Q block
        for q_block_idx in 0..num_q_blocks {
            let q_start = q_block_idx * self.config.block_size_q;
            let q_end = (q_start + self.config.block_size_q).min(seq_len_q);

            // For each KV block
            for kv_block_idx in 0..num_kv_blocks {
                let kv_start = kv_block_idx * self.config.block_size_kv;
                let kv_end = (kv_start + self.config.block_size_kv).min(seq_len_kv);

                // Process each query in this Q block
                for q_idx in q_start..q_end {
                    let q_vec = &q[q_idx];

                    // Compute scores for this KV block
                    let mut scores = Vec::with_capacity(kv_end - kv_start);
                    let mut values_block = Vec::with_capacity(kv_end - kv_start);

                    for kv_idx in kv_start..kv_end {
                        // Apply causal mask
                        if self.config.causal && kv_idx > q_idx {
                            // Skip future positions
                            continue;
                        }

                        // Compute attention score: Q @ K^T * scale
                        let score: f32 = q_vec
                            .iter()
                            .zip(k[kv_idx].iter())
                            .map(|(&qi, &ki)| qi * ki)
                            .sum::<f32>()
                            * scale;

                        scores.push(score);
                        values_block.push(v[kv_idx].clone());
                    }

                    // Update online softmax state
                    states[q_idx].update(&scores, &values_block);
                }
            }
        }

        // Compute memory savings
        let standard_memory = seq_len_q * seq_len_kv; // Full attention matrix
        let flash_memory = self.config.block_size_q * self.config.block_size_kv; // One block at a time
        self.stats.memory_saved_ratio = if flash_memory > 0 {
            1.0 - (flash_memory as f32 / standard_memory as f32)
        } else {
            0.0
        };

        // Finalize outputs
        states.into_iter().map(|s| s.finalize()).collect()
    }

    /// Compute attention for multiple heads (batched)
    /// Q: [num_heads, seq_len_q, head_dim]
    /// K: [num_heads, seq_len_kv, head_dim]
    /// V: [num_heads, seq_len_kv, head_dim]
    /// Returns: [num_heads, seq_len_q, head_dim]
    pub fn forward_multi_head(
        &mut self,
        q: &[Vec<Vec<f32>>],
        k: &[Vec<Vec<f32>>],
        v: &[Vec<Vec<f32>>],
    ) -> Vec<Vec<Vec<f32>>> {
        assert_eq!(q.len(), k.len());
        assert_eq!(k.len(), v.len());

        let mut outputs = Vec::with_capacity(q.len());

        for head in 0..q.len() {
            let head_output = self.forward_single_head(&q[head], &k[head], &v[head]);
            outputs.push(head_output);
        }

        outputs
    }

    /// Compute attention with flat tensor layout
    /// Q: [seq_len_q * head_dim] (flattened)
    /// K: [seq_len_kv * head_dim] (flattened)
    /// V: [seq_len_kv * head_dim] (flattened)
    /// Returns: [seq_len_q * head_dim] (flattened)
    pub fn forward_flat(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;

        // Reshape to 2D
        let q_2d: Vec<Vec<f32>> = q.chunks(head_dim).map(|c| c.to_vec()).collect();
        let k_2d: Vec<Vec<f32>> = k.chunks(head_dim).map(|c| c.to_vec()).collect();
        let v_2d: Vec<Vec<f32>> = v.chunks(head_dim).map(|c| c.to_vec()).collect();

        assert_eq!(q_2d.len(), seq_len_q);
        assert_eq!(k_2d.len(), seq_len_kv);
        assert_eq!(v_2d.len(), seq_len_kv);

        // Compute attention
        let output_2d = self.forward_single_head(&q_2d, &k_2d, &v_2d);

        // Flatten output
        output_2d.into_iter().flatten().collect()
    }
}

/// Standard (non-tiled) attention for comparison
/// This materializes the full N×N attention matrix
pub fn standard_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    scale: f32,
    causal: bool,
) -> Vec<Vec<f32>> {
    let seq_len_q = q.len();
    let seq_len_kv = k.len();

    if seq_len_q == 0 || seq_len_kv == 0 {
        return vec![];
    }

    let head_dim = q[0].len();

    // Compute full attention matrix
    let mut attention_scores: Vec<Vec<f32>> = vec![vec![0.0; seq_len_kv]; seq_len_q];

    for (i, q_vec) in q.iter().enumerate() {
        for (j, k_vec) in k.iter().enumerate() {
            if causal && j > i {
                attention_scores[i][j] = f32::NEG_INFINITY;
            } else {
                let score: f32 = q_vec
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&qi, &kj)| qi * kj)
                    .sum::<f32>()
                    * scale;
                attention_scores[i][j] = score;
            }
        }
    }

    // Softmax over each row
    for row in &mut attention_scores {
        let max_score = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;

        for score in row.iter_mut() {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }

        if sum_exp > 0.0 {
            for score in row.iter_mut() {
                *score /= sum_exp;
            }
        }
    }

    // Compute output: attention @ V
    let mut output = vec![vec![0.0; head_dim]; seq_len_q];

    for (i, attn_row) in attention_scores.iter().enumerate() {
        for (j, v_vec) in v.iter().enumerate() {
            let weight = attn_row[j];
            for (d, &v_val) in v_vec.iter().enumerate() {
                output[i][d] += weight * v_val;
            }
        }
    }

    output
}

/// Verify that Flash Attention and standard attention produce the same results
pub fn verify_equivalence(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    config: &FlashAttentionConfig,
    tolerance: f32,
) -> bool {
    let scale = config.get_scale();

    // Standard attention
    let standard_output = standard_attention(q, k, v, scale, config.causal);

    // Flash attention
    let mut flash = FlashAttention::new(config.clone());
    let flash_output = flash.forward_single_head(q, k, v);

    // Compare
    if standard_output.len() != flash_output.len() {
        return false;
    }

    for (std_row, flash_row) in standard_output.iter().zip(flash_output.iter()) {
        if std_row.len() != flash_row.len() {
            return false;
        }
        for (&s, &f) in std_row.iter().zip(flash_row.iter()) {
            if (s - f).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_matrix(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| {
                        let mut hasher = DefaultHasher::new();
                        (seed * 1000 + i * 100 + j).hash(&mut hasher);
                        ((hasher.finish() % 10000) as f32 / 5000.0) - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 64);
        assert!(!config.causal);
    }

    #[test]
    fn test_config_builder() {
        let config = FlashAttentionConfig::new(8, 64)
            .block_q(32)
            .block_kv(32)
            .causal(true)
            .scale(0.125);

        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.block_size_q, 32);
        assert_eq!(config.block_size_kv, 32);
        assert!(config.causal);
        assert_eq!(config.softmax_scale, Some(0.125));
    }

    #[test]
    fn test_get_scale() {
        let config = FlashAttentionConfig::new(8, 64);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected_scale).abs() < 1e-6);

        let config_custom = config.scale(0.1);
        assert_eq!(config_custom.get_scale(), 0.1);
    }

    #[test]
    fn test_online_softmax_single_update() {
        let mut state = OnlineSoftmaxState::new(4);
        let scores = vec![1.0, 2.0, 3.0];
        let values = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        state.update(&scores, &values);
        let output = state.finalize();

        assert_eq!(output.len(), 4);

        // Verify it's a valid probability-weighted sum
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // First 3 dims should sum to ~1
    }

    #[test]
    fn test_online_softmax_incremental() {
        // Test that incremental computation matches batch computation
        let scores1 = vec![1.0, 2.0];
        let scores2 = vec![3.0, 4.0];
        let values1 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let values2 = vec![vec![0.5, 0.5], vec![0.0, 0.0]];

        // Incremental
        let mut state = OnlineSoftmaxState::new(2);
        state.update(&scores1, &values1);
        state.update(&scores2, &values2);
        let incremental_output = state.finalize();

        // Batch (all at once)
        let all_scores = vec![1.0, 2.0, 3.0, 4.0];
        let all_values = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
            vec![0.0, 0.0],
        ];
        let mut batch_state = OnlineSoftmaxState::new(2);
        batch_state.update(&all_scores, &all_values);
        let batch_output = batch_state.finalize();

        // Should be equal
        for (inc, batch) in incremental_output.iter().zip(batch_output.iter()) {
            assert!((inc - batch).abs() < 1e-5);
        }
    }

    #[test]
    fn test_flash_attention_basic() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let q = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let output = flash.forward_single_head(&q, &k, &v);

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_flash_vs_standard_small() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);

        let q = random_matrix(4, 4, 1);
        let k = random_matrix(4, 4, 2);
        let v = random_matrix(4, 4, 3);

        assert!(verify_equivalence(&q, &k, &v, &config, 1e-4));
    }

    #[test]
    fn test_flash_vs_standard_larger() {
        let config = FlashAttentionConfig::new(1, 8).block_q(4).block_kv(4);

        let q = random_matrix(16, 8, 10);
        let k = random_matrix(16, 8, 20);
        let v = random_matrix(16, 8, 30);

        assert!(verify_equivalence(&q, &k, &v, &config, 1e-4));
    }

    #[test]
    fn test_flash_vs_standard_causal() {
        let config = FlashAttentionConfig::new(1, 4)
            .block_q(2)
            .block_kv(2)
            .causal(true);

        let q = random_matrix(4, 4, 1);
        let k = random_matrix(4, 4, 2);
        let v = random_matrix(4, 4, 3);

        assert!(verify_equivalence(&q, &k, &v, &config, 1e-4));
    }

    #[test]
    fn test_flash_causal_masking() {
        let config = FlashAttentionConfig::new(1, 4)
            .block_q(4)
            .block_kv(4)
            .causal(true);
        let mut flash = FlashAttention::new(config);

        // Query at position 0 should only attend to position 0
        let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let v = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let output = flash.forward_single_head(&q, &k, &v);

        // Output should be v[0] since q[0] can only attend to k[0]
        assert!((output[0][0] - 1.0).abs() < 1e-5);
        assert!(output[0][1].abs() < 1e-5);
    }

    #[test]
    fn test_flash_multi_head() {
        let config = FlashAttentionConfig::new(2, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let q = vec![random_matrix(3, 4, 1), random_matrix(3, 4, 2)];
        let k = vec![random_matrix(3, 4, 3), random_matrix(3, 4, 4)];
        let v = vec![random_matrix(3, 4, 5), random_matrix(3, 4, 6)];

        let output = flash.forward_multi_head(&q, &k, &v);

        assert_eq!(output.len(), 2); // 2 heads
        assert_eq!(output[0].len(), 3); // seq_len
        assert_eq!(output[0][0].len(), 4); // head_dim
    }

    #[test]
    fn test_flash_flat_layout() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config.clone());

        let seq_len = 3;
        let head_dim = 4;

        // Create flat tensors
        let q_flat: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k_flat: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i + 10) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i + 20) as f32 * 0.1)
            .collect();

        let output = flash.forward_flat(&q_flat, &k_flat, &v_flat, seq_len, seq_len);

        assert_eq!(output.len(), seq_len * head_dim);
    }

    #[test]
    fn test_stats() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let q = random_matrix(6, 4, 1);
        let k = random_matrix(8, 4, 2);
        let v = random_matrix(8, 4, 3);

        flash.forward_single_head(&q, &k, &v);

        let stats = flash.stats();
        assert_eq!(stats.q_blocks, 3); // ceil(6/2)
        assert_eq!(stats.kv_blocks, 4); // ceil(8/2)
        assert_eq!(stats.total_block_ops, 12);
        assert!(stats.memory_saved_ratio > 0.0);
    }

    #[test]
    fn test_empty_input() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let empty: Vec<Vec<f32>> = vec![];
        let output = flash.forward_single_head(&empty, &empty, &empty);

        assert!(output.is_empty());
    }

    #[test]
    fn test_single_token() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0]];

        let output = flash.forward_single_head(&q, &k, &v);

        // With single token, output should equal v (attention weight = 1)
        assert_eq!(output.len(), 1);
        for (out, val) in output[0].iter().zip(v[0].iter()) {
            assert!((out - val).abs() < 1e-5);
        }
    }

    #[test]
    fn test_different_q_kv_lengths() {
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config);

        let q = random_matrix(3, 4, 1);
        let k = random_matrix(5, 4, 2);
        let v = random_matrix(5, 4, 3);

        let output = flash.forward_single_head(&q, &k, &v);

        assert_eq!(output.len(), 3); // Same as Q length
        assert_eq!(output[0].len(), 4); // head_dim
    }

    #[test]
    fn test_standard_attention_causal() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = q.clone();
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let output = standard_attention(&q, &k, &v, 1.0, true);

        // Position 0 should only attend to position 0
        assert!((output[0][0] - 1.0).abs() < 1e-5);
        assert!(output[0][1].abs() < 1e-5);

        // Position 1 attends to positions 0 and 1
        // Position 2 attends to all positions
    }

    #[test]
    fn test_large_sequence_memory_savings() {
        let config = FlashAttentionConfig::new(1, 64).block_q(64).block_kv(64);
        let mut flash = FlashAttention::new(config);

        let seq_len = 512;
        let q = random_matrix(seq_len, 64, 1);
        let k = random_matrix(seq_len, 64, 2);
        let v = random_matrix(seq_len, 64, 3);

        flash.forward_single_head(&q, &k, &v);

        // Memory savings should be significant for long sequences
        assert!(flash.stats().memory_saved_ratio > 0.9);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with large values that could cause overflow in naive softmax
        let config = FlashAttentionConfig::new(1, 4).block_q(2).block_kv(2);
        let mut flash = FlashAttention::new(config.clone());

        let q = vec![vec![100.0, 100.0, 100.0, 100.0]];
        let k = vec![
            vec![100.0, 100.0, 100.0, 100.0],
            vec![-100.0, -100.0, -100.0, -100.0],
        ];
        let v = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let output = flash.forward_single_head(&q, &k, &v);

        // Should not have NaN or Inf
        assert!(output[0].iter().all(|x| x.is_finite()));

        // High score token (k[0]) should dominate
        assert!(output[0][0] > 0.99);
    }
}
