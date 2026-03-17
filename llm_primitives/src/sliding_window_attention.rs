// Sliding Window Attention
//
// Efficient attention mechanism for long sequences, popularized by Mistral and Longformer.
// Instead of attending to all positions (O(N²)), each position attends only to a fixed
// window of recent positions.
//
// Key concepts:
// - Window size: Maximum number of previous tokens each position can attend to
// - Sparse attention: Only compute attention within the window, not full N×N matrix
// - Memory efficiency: O(N × W) instead of O(N²) where W is window size
// - Streaming-friendly: Can process sequences longer than context with rolling window
//
// Algorithm overview:
// 1. For each query position i:
//    - Determine window: positions max(0, i - window_size + 1) to i (inclusive)
//    - Compute attention scores only within window
//    - Apply softmax and compute weighted sum of values
//
// Variants:
// - Symmetric window (Longformer): attend to positions [i - w/2, i + w/2]
// - Causal window (Mistral): attend to positions [i - w + 1, i]
// - Global tokens (Longformer): some positions attend to all tokens
//
// This implementation focuses on causal sliding window (Mistral-style).

/// Configuration for Sliding Window Attention
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Window size (number of tokens each position can attend to)
    pub window_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Softmax scale (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
    /// Use symmetric window (attend both directions) vs causal (only past)
    pub symmetric: bool,
    /// Global attention positions (these attend to all tokens)
    pub global_positions: Vec<usize>,
}

impl SlidingWindowConfig {
    /// Create a new configuration
    pub fn new(window_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            num_heads,
            head_dim,
            softmax_scale: None,
            symmetric: false,
            global_positions: vec![],
        }
    }

    /// Set custom softmax scale
    pub fn scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Enable symmetric window (Longformer-style)
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Set global attention positions
    pub fn global_positions(mut self, positions: Vec<usize>) -> Self {
        self.global_positions = positions;
        self
    }

    /// Get the softmax scale (default: 1/sqrt(head_dim))
    pub fn get_scale(&self) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            num_heads: 8,
            head_dim: 64,
            softmax_scale: None,
            symmetric: false,
            global_positions: vec![],
        }
    }
}

/// Statistics from Sliding Window Attention computation
#[derive(Debug, Clone, Default)]
pub struct SlidingWindowStats {
    /// Number of attention computations performed
    pub attention_ops: usize,
    /// Average window size used (can be smaller at sequence start)
    pub avg_window_size: f32,
    /// Memory savings vs full attention (approximate)
    pub memory_saved_ratio: f32,
    /// Number of global attention positions
    pub global_positions_count: usize,
}

/// Sliding Window Attention computation
///
/// Computes attention where each position only attends to a fixed window
/// of nearby positions, enabling efficient long-context modeling.
pub struct SlidingWindowAttention {
    config: SlidingWindowConfig,
    stats: SlidingWindowStats,
}

impl SlidingWindowAttention {
    /// Create a new Sliding Window Attention instance
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            stats: SlidingWindowStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }

    /// Get statistics from last computation
    pub fn stats(&self) -> &SlidingWindowStats {
        &self.stats
    }

    /// Check if a position has global attention
    fn is_global(&self, pos: usize) -> bool {
        self.config.global_positions.contains(&pos)
    }

    /// Get the window range for a given query position
    /// Returns (start, end) inclusive range of KV positions to attend to
    fn get_window_range(&self, q_pos: usize, seq_len_kv: usize) -> (usize, usize) {
        if self.is_global(q_pos) {
            // Global position attends to everything
            return (0, seq_len_kv.saturating_sub(1));
        }

        if self.config.symmetric {
            // Symmetric window: attend to [pos - w/2, pos + w/2]
            let half_window = self.config.window_size / 2;
            let start = q_pos.saturating_sub(half_window);
            let end = (q_pos + half_window).min(seq_len_kv.saturating_sub(1));
            (start, end)
        } else {
            // Causal window: attend to [pos - w + 1, pos]
            let start = q_pos.saturating_sub(self.config.window_size - 1);
            let end = q_pos.min(seq_len_kv.saturating_sub(1));
            (start, end)
        }
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

        let mut output = Vec::with_capacity(seq_len_q);
        let mut total_attention_ops = 0usize;
        let mut total_window_size = 0usize;

        for (q_idx, q_vec) in q.iter().enumerate() {
            let (window_start, window_end) = self.get_window_range(q_idx, seq_len_kv);
            let window_len = window_end - window_start + 1;
            total_window_size += window_len;

            // Compute attention scores within window
            let mut scores = Vec::with_capacity(window_len);
            for k_vec in k.iter().take(window_end + 1).skip(window_start) {
                let score: f32 = q_vec
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
                total_attention_ops += 1;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let exp_scores: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let e = (s - max_score).exp();
                    sum_exp += e;
                    e
                })
                .collect();

            // Weighted sum of values
            let mut out_vec = vec![0.0f32; head_dim];
            for (i, kv_idx) in (window_start..=window_end).enumerate() {
                let weight = if sum_exp > 0.0 {
                    exp_scores[i] / sum_exp
                } else {
                    0.0
                };
                for (d, &v_val) in v[kv_idx].iter().enumerate() {
                    out_vec[d] += weight * v_val;
                }
            }

            output.push(out_vec);
        }

        // Update stats
        self.stats.attention_ops = total_attention_ops;
        self.stats.avg_window_size = if seq_len_q > 0 {
            total_window_size as f32 / seq_len_q as f32
        } else {
            0.0
        };
        self.stats.global_positions_count = self.config.global_positions.len();

        // Memory savings: compared to full attention N×N, we use N×W
        let full_ops = seq_len_q * seq_len_kv;
        self.stats.memory_saved_ratio = if full_ops > 0 {
            1.0 - (total_attention_ops as f32 / full_ops as f32)
        } else {
            0.0
        };

        output
    }

    /// Compute attention for multiple heads
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

/// Standard sliding window attention for comparison
/// Materializes the sparse attention matrix for verification
pub fn standard_sliding_window(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    window_size: usize,
    scale: f32,
    symmetric: bool,
) -> Vec<Vec<f32>> {
    let seq_len_q = q.len();
    let seq_len_kv = k.len();

    if seq_len_q == 0 || seq_len_kv == 0 {
        return vec![];
    }

    let head_dim = q[0].len();
    let mut output = vec![vec![0.0; head_dim]; seq_len_q];

    for (i, q_vec) in q.iter().enumerate() {
        // Compute window range
        let (start, end) = if symmetric {
            let half = window_size / 2;
            (
                i.saturating_sub(half),
                (i + half).min(seq_len_kv.saturating_sub(1)),
            )
        } else {
            (
                i.saturating_sub(window_size - 1),
                i.min(seq_len_kv.saturating_sub(1)),
            )
        };

        // Compute scores within window
        let mut scores = Vec::new();
        for k_vec in k.iter().take(end + 1).skip(start) {
            let score: f32 = q_vec
                .iter()
                .zip(k_vec.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum::<f32>()
                * scale;
            scores.push(score);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        // Weighted sum
        for (j, kv_idx) in (start..=end).enumerate() {
            let weight = if sum_exp > 0.0 {
                exp_scores[j] / sum_exp
            } else {
                0.0
            };
            for (d, &v_val) in v[kv_idx].iter().enumerate() {
                output[i][d] += weight * v_val;
            }
        }
    }

    output
}

/// Verify that our implementation matches the reference
pub fn verify_equivalence(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    config: &SlidingWindowConfig,
    tolerance: f32,
) -> bool {
    let scale = config.get_scale();

    // Reference implementation
    let reference = standard_sliding_window(q, k, v, config.window_size, scale, config.symmetric);

    // Our implementation
    let mut swa = SlidingWindowAttention::new(config.clone());
    let result = swa.forward_single_head(q, k, v);

    // Compare
    if reference.len() != result.len() {
        return false;
    }

    for (ref_row, res_row) in reference.iter().zip(result.iter()) {
        if ref_row.len() != res_row.len() {
            return false;
        }
        for (&r, &s) in ref_row.iter().zip(res_row.iter()) {
            if (r - s).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Streaming sliding window attention
/// Processes sequences in chunks, useful for very long sequences
pub struct StreamingSlidingWindow {
    config: SlidingWindowConfig,
    /// Cached keys from previous chunks (up to window_size)
    cached_keys: Vec<Vec<f32>>,
    /// Cached values from previous chunks (up to window_size)
    cached_values: Vec<Vec<f32>>,
    /// Total tokens processed so far
    tokens_processed: usize,
}

impl StreamingSlidingWindow {
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            cached_keys: Vec::new(),
            cached_values: Vec::new(),
            tokens_processed: 0,
        }
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.cached_keys.clear();
        self.cached_values.clear();
        self.tokens_processed = 0;
    }

    /// Get number of tokens processed
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        self.cached_keys.len()
    }

    /// Process a chunk of tokens
    /// Q: [chunk_size, head_dim]
    /// K: [chunk_size, head_dim]
    /// V: [chunk_size, head_dim]
    /// Returns: [chunk_size, head_dim]
    pub fn process_chunk(
        &mut self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let chunk_size = q.len();
        let head_dim = self.config.head_dim;
        let scale = self.config.get_scale();
        let window_size = self.config.window_size;

        if chunk_size == 0 {
            return vec![];
        }

        let mut output = Vec::with_capacity(chunk_size);

        for (i, q_vec) in q.iter().enumerate() {
            // Build effective KV context: cached + current chunk up to position i
            let mut effective_k: Vec<&Vec<f32>> = Vec::new();
            let mut effective_v: Vec<&Vec<f32>> = Vec::new();

            // Add relevant cached tokens
            let cache_start = if self.cached_keys.len() > window_size - 1 - i {
                self.cached_keys.len() - (window_size - 1 - i)
            } else {
                0
            };

            for j in cache_start..self.cached_keys.len() {
                effective_k.push(&self.cached_keys[j]);
                effective_v.push(&self.cached_values[j]);
            }

            // Add current chunk tokens up to and including position i
            for j in 0..=i {
                effective_k.push(&k[j]);
                effective_v.push(&v[j]);
            }

            // Only keep last window_size tokens
            let context_start = effective_k.len().saturating_sub(window_size);
            let effective_k: Vec<&Vec<f32>> = effective_k[context_start..].to_vec();
            let effective_v: Vec<&Vec<f32>> = effective_v[context_start..].to_vec();

            // Compute attention scores
            let mut scores: Vec<f32> = effective_k
                .iter()
                .map(|k_vec| {
                    q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f32>()
                        * scale
                })
                .collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }

            // Weighted sum
            let mut out_vec = vec![0.0f32; head_dim];
            for (j, v_ref) in effective_v.iter().enumerate() {
                let weight = if sum_exp > 0.0 {
                    scores[j] / sum_exp
                } else {
                    0.0
                };
                for (d, &v_val) in v_ref.iter().enumerate() {
                    out_vec[d] += weight * v_val;
                }
            }

            output.push(out_vec);
        }

        // Update cache: keep last window_size - 1 tokens
        // (the -1 is because the current query position is always included)
        for i in 0..chunk_size {
            self.cached_keys.push(k[i].clone());
            self.cached_values.push(v[i].clone());
        }

        // Trim cache to window_size - 1
        while self.cached_keys.len() > window_size - 1 {
            self.cached_keys.remove(0);
            self.cached_values.remove(0);
        }

        self.tokens_processed += chunk_size;

        output
    }
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
        let config = SlidingWindowConfig::default();
        assert_eq!(config.window_size, 4096);
        assert_eq!(config.num_heads, 8);
        assert!(!config.symmetric);
    }

    #[test]
    fn test_config_builder() {
        let config = SlidingWindowConfig::new(128, 4, 32)
            .scale(0.125)
            .symmetric(true)
            .global_positions(vec![0, 10]);

        assert_eq!(config.window_size, 128);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim, 32);
        assert!(config.symmetric);
        assert_eq!(config.global_positions, vec![0, 10]);
    }

    #[test]
    fn test_get_scale() {
        let config = SlidingWindowConfig::new(64, 8, 64);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected_scale).abs() < 1e-6);

        let config_custom = config.scale(0.1);
        assert_eq!(config_custom.get_scale(), 0.1);
    }

    #[test]
    fn test_window_range_causal() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let swa = SlidingWindowAttention::new(config);

        // Position 0: can only see position 0
        assert_eq!(swa.get_window_range(0, 10), (0, 0));

        // Position 2: can see positions 0, 1, 2
        assert_eq!(swa.get_window_range(2, 10), (0, 2));

        // Position 5 with window 4: can see positions 2, 3, 4, 5
        assert_eq!(swa.get_window_range(5, 10), (2, 5));
    }

    #[test]
    fn test_window_range_symmetric() {
        let config = SlidingWindowConfig::new(4, 1, 4).symmetric(true);
        let swa = SlidingWindowAttention::new(config);

        // Position 0: can see positions 0, 1, 2 (half_window = 2)
        assert_eq!(swa.get_window_range(0, 10), (0, 2));

        // Position 5: can see positions 3, 4, 5, 6, 7 (5-2 to 5+2)
        assert_eq!(swa.get_window_range(5, 10), (3, 7));

        // Position 8: can see positions 6, 7, 8, 9 (clamped to seq_len)
        assert_eq!(swa.get_window_range(8, 10), (6, 9));
    }

    #[test]
    fn test_basic_attention() {
        let config = SlidingWindowConfig::new(3, 1, 4);
        let mut swa = SlidingWindowAttention::new(config);

        let q = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let output = swa.forward_single_head(&q, &k, &v);

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_verify_equivalence() {
        let config = SlidingWindowConfig::new(4, 1, 8);

        let q = random_matrix(8, 8, 1);
        let k = random_matrix(8, 8, 2);
        let v = random_matrix(8, 8, 3);

        assert!(verify_equivalence(&q, &k, &v, &config, 1e-5));
    }

    #[test]
    fn test_causal_windowing() {
        // With window_size = 2, position 3 should only attend to positions 2, 3
        let config = SlidingWindowConfig::new(2, 1, 4);
        let mut swa = SlidingWindowAttention::new(config.clone());

        let q = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let k = vec![
            vec![1.0, 0.0, 0.0, 0.0], // high score
            vec![0.0, 1.0, 0.0, 0.0], // low score
            vec![0.0, 1.0, 0.0, 0.0], // low score
            vec![0.0, 1.0, 0.0, 0.0], // low score
        ];
        let v = vec![
            vec![100.0, 0.0, 0.0, 0.0], // Should NOT appear in output[3]
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        let output = swa.forward_single_head(&q, &k, &v);

        // Position 3 should not see position 0 (outside window)
        // So output[3] should NOT have the 100.0 from v[0]
        assert!(output[3][0].abs() < 1e-5);
    }

    #[test]
    fn test_memory_savings() {
        let window_size = 64;
        let seq_len = 1024;
        let config = SlidingWindowConfig::new(window_size, 1, 32);
        let mut swa = SlidingWindowAttention::new(config);

        let q = random_matrix(seq_len, 32, 1);
        let k = random_matrix(seq_len, 32, 2);
        let v = random_matrix(seq_len, 32, 3);

        swa.forward_single_head(&q, &k, &v);

        // Memory savings should be significant
        // Full attention: 1024 * 1024 = 1M operations
        // Sliding window: 1024 * 64 = 64K operations (roughly)
        assert!(swa.stats().memory_saved_ratio > 0.9);
    }

    #[test]
    fn test_global_positions() {
        // Position 0 has global attention
        let config = SlidingWindowConfig::new(2, 1, 4).global_positions(vec![0]);
        let swa = SlidingWindowAttention::new(config);

        // Position 0 should attend to all positions
        assert_eq!(swa.get_window_range(0, 10), (0, 9));

        // Position 1 should have normal window
        assert_eq!(swa.get_window_range(1, 10), (0, 1));
    }

    #[test]
    fn test_multi_head() {
        let config = SlidingWindowConfig::new(4, 2, 4);
        let mut swa = SlidingWindowAttention::new(config);

        let q = vec![random_matrix(3, 4, 1), random_matrix(3, 4, 2)];
        let k = vec![random_matrix(3, 4, 3), random_matrix(3, 4, 4)];
        let v = vec![random_matrix(3, 4, 5), random_matrix(3, 4, 6)];

        let output = swa.forward_multi_head(&q, &k, &v);

        assert_eq!(output.len(), 2); // 2 heads
        assert_eq!(output[0].len(), 3); // seq_len
        assert_eq!(output[0][0].len(), 4); // head_dim
    }

    #[test]
    fn test_flat_layout() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut swa = SlidingWindowAttention::new(config);

        let seq_len = 6;
        let head_dim = 4;

        let q_flat: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k_flat: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i + 10) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i + 20) as f32 * 0.1)
            .collect();

        let output = swa.forward_flat(&q_flat, &k_flat, &v_flat, seq_len, seq_len);

        assert_eq!(output.len(), seq_len * head_dim);
    }

    #[test]
    fn test_empty_input() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut swa = SlidingWindowAttention::new(config);

        let empty: Vec<Vec<f32>> = vec![];
        let output = swa.forward_single_head(&empty, &empty, &empty);

        assert!(output.is_empty());
    }

    #[test]
    fn test_single_token() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut swa = SlidingWindowAttention::new(config);

        let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0]];

        let output = swa.forward_single_head(&q, &k, &v);

        // Single token attention = output equals value
        assert_eq!(output.len(), 1);
        for (out, val) in output[0].iter().zip(v[0].iter()) {
            assert!((out - val).abs() < 1e-5);
        }
    }

    #[test]
    fn test_streaming_basic() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut streaming = StreamingSlidingWindow::new(config);

        // First chunk
        let q1 = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let k1 = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let v1 = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let output1 = streaming.process_chunk(&q1, &k1, &v1);

        assert_eq!(output1.len(), 2);
        assert_eq!(streaming.tokens_processed(), 2);

        // Second chunk
        let q2 = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k2 = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v2 = vec![vec![9.0, 10.0, 11.0, 12.0]];

        let output2 = streaming.process_chunk(&q2, &k2, &v2);

        assert_eq!(output2.len(), 1);
        assert_eq!(streaming.tokens_processed(), 3);
    }

    #[test]
    fn test_streaming_matches_batch() {
        let config = SlidingWindowConfig::new(4, 1, 4);

        // Full sequence
        let q = random_matrix(6, 4, 1);
        let k = random_matrix(6, 4, 2);
        let v = random_matrix(6, 4, 3);

        // Batch computation
        let mut batch_swa = SlidingWindowAttention::new(config.clone());
        let batch_output = batch_swa.forward_single_head(&q, &k, &v);

        // Streaming computation (process in chunks of 2)
        let mut streaming = StreamingSlidingWindow::new(config);
        let mut streaming_output = Vec::new();

        for chunk_start in (0..6).step_by(2) {
            let chunk_end = (chunk_start + 2).min(6);
            let q_chunk: Vec<Vec<f32>> = q[chunk_start..chunk_end].to_vec();
            let k_chunk: Vec<Vec<f32>> = k[chunk_start..chunk_end].to_vec();
            let v_chunk: Vec<Vec<f32>> = v[chunk_start..chunk_end].to_vec();

            let chunk_output = streaming.process_chunk(&q_chunk, &k_chunk, &v_chunk);
            streaming_output.extend(chunk_output);
        }

        // Outputs should match
        assert_eq!(batch_output.len(), streaming_output.len());
        for (batch_row, stream_row) in batch_output.iter().zip(streaming_output.iter()) {
            for (&b, &s) in batch_row.iter().zip(stream_row.iter()) {
                assert!((b - s).abs() < 1e-5, "Mismatch: batch={}, stream={}", b, s);
            }
        }
    }

    #[test]
    fn test_streaming_cache_eviction() {
        let config = SlidingWindowConfig::new(3, 1, 4);
        let mut streaming = StreamingSlidingWindow::new(config);

        // Process 5 tokens one at a time
        for i in 0..5 {
            let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
            let k = vec![vec![i as f32, 0.0, 0.0, 0.0]];
            let v = vec![vec![i as f32 * 10.0, 0.0, 0.0, 0.0]];

            streaming.process_chunk(&q, &k, &v);
        }

        // Cache should contain at most window_size - 1 = 2 tokens
        assert!(streaming.cache_size() <= 2);
        assert_eq!(streaming.tokens_processed(), 5);
    }

    #[test]
    fn test_streaming_reset() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut streaming = StreamingSlidingWindow::new(config);

        // Process some tokens
        let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0]];
        streaming.process_chunk(&q, &k, &v);

        assert_eq!(streaming.tokens_processed(), 1);

        // Reset
        streaming.reset();

        assert_eq!(streaming.tokens_processed(), 0);
        assert_eq!(streaming.cache_size(), 0);
    }

    #[test]
    fn test_symmetric_window() {
        let config = SlidingWindowConfig::new(4, 1, 4).symmetric(true);

        let q = random_matrix(8, 4, 1);
        let k = random_matrix(8, 4, 2);
        let v = random_matrix(8, 4, 3);

        assert!(verify_equivalence(&q, &k, &v, &config, 1e-5));
    }

    #[test]
    fn test_stats() {
        let config = SlidingWindowConfig::new(4, 1, 4).global_positions(vec![0]);
        let mut swa = SlidingWindowAttention::new(config);

        let q = random_matrix(10, 4, 1);
        let k = random_matrix(10, 4, 2);
        let v = random_matrix(10, 4, 3);

        swa.forward_single_head(&q, &k, &v);

        let stats = swa.stats();
        assert!(stats.attention_ops > 0);
        assert!(stats.avg_window_size > 0.0);
        assert_eq!(stats.global_positions_count, 1);
    }

    #[test]
    fn test_numerical_stability() {
        let config = SlidingWindowConfig::new(4, 1, 4);
        let mut swa = SlidingWindowAttention::new(config);

        // Large values that could cause overflow in naive softmax
        let q = vec![vec![100.0, 100.0, 100.0, 100.0]];
        let k = vec![
            vec![100.0, 100.0, 100.0, 100.0],
            vec![-100.0, -100.0, -100.0, -100.0],
        ];
        let v = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let output = swa.forward_single_head(&q, &k, &v);

        // Should not have NaN or Inf
        assert!(output[0].iter().all(|x| x.is_finite()));
    }
}
