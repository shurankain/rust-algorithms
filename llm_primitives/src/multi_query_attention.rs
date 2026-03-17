// Multi-Query Attention (MQA)
//
// Memory-efficient attention variant from "Fast Transformer Decoding" (Shazeer, 2019).
// Instead of having separate K, V projections per head, MQA shares a single K, V
// across all query heads.
//
// Key concepts:
// - Standard Multi-Head Attention (MHA): Q, K, V all have `num_heads` sets of projections
// - Multi-Query Attention (MQA): Q has `num_heads` projections, K, V have just 1
// - This reduces KV cache size by `num_heads` times!
//
// Memory comparison for KV cache:
// - MHA: 2 × num_heads × seq_len × head_dim
// - MQA: 2 × 1 × seq_len × head_dim (num_heads times smaller)
//
// Used in: PaLM, Falcon, StarCoder, and others
//
// Trade-offs:
// - Pros: Massive memory savings, faster inference (less KV cache reads)
// - Cons: Slightly lower quality than MHA (mitigated by GQA middle ground)
//
// Algorithm:
// 1. Each query head attends to the SAME single K, V
// 2. Attention scores computed independently per head
// 3. Output concatenated across heads as usual

/// Configuration for Multi-Query Attention
#[derive(Debug, Clone)]
pub struct MqaConfig {
    /// Number of query heads
    pub num_query_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Softmax scale (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl MqaConfig {
    /// Create a new configuration
    pub fn new(num_query_heads: usize, head_dim: usize) -> Self {
        Self {
            num_query_heads,
            head_dim,
            causal: false,
            softmax_scale: None,
        }
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

    /// Calculate memory savings vs MHA
    pub fn memory_savings_ratio(&self) -> f32 {
        // MHA uses num_query_heads sets of K, V
        // MQA uses 1 set of K, V
        1.0 - (1.0 / self.num_query_heads as f32)
    }
}

impl Default for MqaConfig {
    fn default() -> Self {
        Self {
            num_query_heads: 8,
            head_dim: 64,
            causal: false,
            softmax_scale: None,
        }
    }
}

/// Statistics from MQA computation
#[derive(Debug, Clone, Default)]
pub struct MqaStats {
    /// Sequence length processed
    pub seq_len_q: usize,
    /// KV sequence length
    pub seq_len_kv: usize,
    /// Number of attention operations
    pub attention_ops: usize,
    /// Memory savings vs MHA (ratio)
    pub memory_saved_ratio: f32,
}

/// Multi-Query Attention computation
///
/// Computes attention where all query heads share a single set of keys and values.
///
/// Input shapes:
/// - Q: [num_query_heads, seq_len_q, head_dim]
/// - K: [1, seq_len_kv, head_dim] (single head, shared across queries)
/// - V: [1, seq_len_kv, head_dim] (single head, shared across queries)
///
/// Output shape: [num_query_heads, seq_len_q, head_dim]
pub struct MultiQueryAttention {
    config: MqaConfig,
    stats: MqaStats,
}

impl MultiQueryAttention {
    /// Create a new Multi-Query Attention instance
    pub fn new(config: MqaConfig) -> Self {
        Self {
            config,
            stats: MqaStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MqaConfig {
        &self.config
    }

    /// Get statistics from last computation
    pub fn stats(&self) -> &MqaStats {
        &self.stats
    }

    /// Compute attention for a single query head using shared K, V
    /// Q: [seq_len_q, head_dim]
    /// K: [seq_len_kv, head_dim] (shared)
    /// V: [seq_len_kv, head_dim] (shared)
    /// Returns: [seq_len_q, head_dim]
    fn attention_single_head(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        scale: f32,
    ) -> Vec<Vec<f32>> {
        let seq_len_q = q.len();
        let seq_len_kv = k.len();
        let head_dim = self.config.head_dim;

        if seq_len_q == 0 || seq_len_kv == 0 {
            return vec![];
        }

        let mut output = Vec::with_capacity(seq_len_q);

        for (q_idx, q_vec) in q.iter().enumerate() {
            // Compute attention scores
            let mut scores: Vec<f32> = Vec::with_capacity(seq_len_kv);

            for (kv_idx, k_vec) in k.iter().enumerate() {
                // Apply causal mask
                if self.config.causal && kv_idx > q_idx {
                    scores.push(f32::NEG_INFINITY);
                } else {
                    let score: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }
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
            for (j, v_vec) in v.iter().enumerate() {
                let weight = if sum_exp > 0.0 {
                    exp_scores[j] / sum_exp
                } else {
                    0.0
                };
                for (d, &v_val) in v_vec.iter().enumerate() {
                    out_vec[d] += weight * v_val;
                }
            }

            output.push(out_vec);
        }

        output
    }

    /// Compute Multi-Query Attention
    /// Q: [num_query_heads, seq_len_q, head_dim]
    /// K: [seq_len_kv, head_dim] (single set, shared across all heads)
    /// V: [seq_len_kv, head_dim] (single set, shared across all heads)
    /// Returns: [num_query_heads, seq_len_q, head_dim]
    pub fn forward(
        &mut self,
        q: &[Vec<Vec<f32>>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<Vec<Vec<f32>>> {
        let num_heads = q.len();
        let seq_len_q = if num_heads > 0 { q[0].len() } else { 0 };
        let seq_len_kv = k.len();
        let scale = self.config.get_scale();

        // Update stats
        self.stats.seq_len_q = seq_len_q;
        self.stats.seq_len_kv = seq_len_kv;
        self.stats.attention_ops = num_heads * seq_len_q * seq_len_kv;
        self.stats.memory_saved_ratio = self.config.memory_savings_ratio();

        // Compute attention for each query head using shared K, V
        let mut outputs = Vec::with_capacity(num_heads);

        for head_q in q {
            let head_output = self.attention_single_head(head_q, k, v, scale);
            outputs.push(head_output);
        }

        outputs
    }

    /// Compute attention with flat tensor layout
    /// Q: [num_query_heads * seq_len_q * head_dim] (flattened)
    /// K: [seq_len_kv * head_dim] (flattened, single head)
    /// V: [seq_len_kv * head_dim] (flattened, single head)
    /// Returns: [num_query_heads * seq_len_q * head_dim] (flattened)
    pub fn forward_flat(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_query_heads;

        // Reshape Q to [num_heads, seq_len_q, head_dim]
        let q_3d: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|h| {
                let head_start = h * seq_len_q * head_dim;
                (0..seq_len_q)
                    .map(|s| {
                        let start = head_start + s * head_dim;
                        q[start..start + head_dim].to_vec()
                    })
                    .collect()
            })
            .collect();

        // Reshape K, V to [seq_len_kv, head_dim]
        let k_2d: Vec<Vec<f32>> = k.chunks(head_dim).map(|c| c.to_vec()).collect();
        let v_2d: Vec<Vec<f32>> = v.chunks(head_dim).map(|c| c.to_vec()).collect();

        assert_eq!(k_2d.len(), seq_len_kv);
        assert_eq!(v_2d.len(), seq_len_kv);

        // Compute attention
        let output_3d = self.forward(&q_3d, &k_2d, &v_2d);

        // Flatten output
        output_3d
            .into_iter()
            .flat_map(|head| head.into_iter().flatten())
            .collect()
    }
}

/// Standard Multi-Head Attention for comparison
/// Each head has its own K, V (not shared)
pub fn standard_mha(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<Vec<f32>>],
    v: &[Vec<Vec<f32>>],
    scale: f32,
    causal: bool,
) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(q.len(), k.len());
    assert_eq!(k.len(), v.len());

    let num_heads = q.len();
    let mut outputs = Vec::with_capacity(num_heads);

    for head in 0..num_heads {
        let head_q = &q[head];
        let head_k = &k[head];
        let head_v = &v[head];

        let seq_len_q = head_q.len();
        let seq_len_kv = head_k.len();
        let head_dim = if seq_len_q > 0 { head_q[0].len() } else { 0 };

        let mut head_output = Vec::with_capacity(seq_len_q);

        for (q_idx, q_vec) in head_q.iter().enumerate() {
            let mut scores: Vec<f32> = Vec::with_capacity(seq_len_kv);

            for (kv_idx, k_vec) in head_k.iter().enumerate() {
                if causal && kv_idx > q_idx {
                    scores.push(f32::NEG_INFINITY);
                } else {
                    let score: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted sum
            let mut out_vec = vec![0.0f32; head_dim];
            for (j, v_vec) in head_v.iter().enumerate() {
                let weight = if sum_exp > 0.0 {
                    exp_scores[j] / sum_exp
                } else {
                    0.0
                };
                for (d, &v_val) in v_vec.iter().enumerate() {
                    out_vec[d] += weight * v_val;
                }
            }

            head_output.push(out_vec);
        }

        outputs.push(head_output);
    }

    outputs
}

/// Verify MQA produces same output as MHA when K, V are replicated
pub fn verify_mqa_mha_equivalence(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    config: &MqaConfig,
    tolerance: f32,
) -> bool {
    let scale = config.get_scale();
    let num_heads = q.len();

    // MQA computation
    let mut mqa = MultiQueryAttention::new(config.clone());
    let mqa_output = mqa.forward(q, k, v);

    // MHA computation with replicated K, V
    let k_replicated: Vec<Vec<Vec<f32>>> = (0..num_heads).map(|_| k.to_vec()).collect();
    let v_replicated: Vec<Vec<Vec<f32>>> = (0..num_heads).map(|_| v.to_vec()).collect();
    let mha_output = standard_mha(q, &k_replicated, &v_replicated, scale, config.causal);

    // Compare
    if mqa_output.len() != mha_output.len() {
        return false;
    }

    for (mqa_head, mha_head) in mqa_output.iter().zip(mha_output.iter()) {
        if mqa_head.len() != mha_head.len() {
            return false;
        }
        for (mqa_row, mha_row) in mqa_head.iter().zip(mha_head.iter()) {
            if mqa_row.len() != mha_row.len() {
                return false;
            }
            for (&m, &h) in mqa_row.iter().zip(mha_row.iter()) {
                if (m - h).abs() > tolerance {
                    return false;
                }
            }
        }
    }

    true
}

/// Calculate KV cache size for MHA vs MQA
pub struct KvCacheComparison {
    pub mha_size: usize,
    pub mqa_size: usize,
    pub savings_ratio: f32,
    pub savings_factor: f32,
}

/// Compare KV cache memory requirements
pub fn compare_kv_cache_size(
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    bytes_per_element: usize,
) -> KvCacheComparison {
    // MHA: 2 (K+V) × num_heads × seq_len × head_dim × bytes
    let mha_size = 2 * num_heads * seq_len * head_dim * bytes_per_element;

    // MQA: 2 (K+V) × 1 × seq_len × head_dim × bytes
    let mqa_size = 2 * seq_len * head_dim * bytes_per_element;

    KvCacheComparison {
        mha_size,
        mqa_size,
        savings_ratio: 1.0 - (mqa_size as f32 / mha_size as f32),
        savings_factor: mha_size as f32 / mqa_size as f32,
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
        let config = MqaConfig::default();
        assert_eq!(config.num_query_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(!config.causal);
    }

    #[test]
    fn test_config_builder() {
        let config = MqaConfig::new(16, 128).causal(true).scale(0.0625);

        assert_eq!(config.num_query_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal);
        assert_eq!(config.softmax_scale, Some(0.0625));
    }

    #[test]
    fn test_get_scale() {
        let config = MqaConfig::new(8, 64);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected_scale).abs() < 1e-6);

        let config_custom = config.scale(0.1);
        assert_eq!(config_custom.get_scale(), 0.1);
    }

    #[test]
    fn test_memory_savings_ratio() {
        let config = MqaConfig::new(8, 64);
        // 8 heads -> 1/8 = 0.125 -> savings = 1 - 0.125 = 0.875
        assert!((config.memory_savings_ratio() - 0.875).abs() < 1e-6);

        let config_32 = MqaConfig::new(32, 64);
        // 32 heads -> 1/32 = 0.03125 -> savings = 0.96875
        assert!((config_32.memory_savings_ratio() - 0.96875).abs() < 1e-6);
    }

    #[test]
    fn test_basic_attention() {
        let config = MqaConfig::new(2, 4);
        let mut mqa = MultiQueryAttention::new(config);

        // 2 query heads, each with 2 queries of dim 4
        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        ];

        // Shared K, V with 2 positions
        let k = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let output = mqa.forward(&q, &k, &v);

        assert_eq!(output.len(), 2); // 2 heads
        assert_eq!(output[0].len(), 2); // 2 query positions
        assert_eq!(output[0][0].len(), 4); // head_dim
    }

    #[test]
    fn test_mqa_mha_equivalence() {
        let config = MqaConfig::new(4, 8);

        // 4 query heads
        let q: Vec<Vec<Vec<f32>>> = (0..4).map(|h| random_matrix(5, 8, h + 1)).collect();

        // Shared K, V
        let k = random_matrix(5, 8, 10);
        let v = random_matrix(5, 8, 20);

        assert!(verify_mqa_mha_equivalence(&q, &k, &v, &config, 1e-5));
    }

    #[test]
    fn test_causal_masking() {
        let config = MqaConfig::new(1, 4).causal(true);
        let mut mqa = MultiQueryAttention::new(config);

        // Single query head, position 0
        let q = vec![vec![vec![1.0, 0.0, 0.0, 0.0]]];

        // 3 KV positions
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

        let output = mqa.forward(&q, &k, &v);

        // Position 0 should only attend to position 0 (causal mask)
        // So output should be v[0]
        assert!((output[0][0][0] - 1.0).abs() < 1e-5);
        assert!(output[0][0][1].abs() < 1e-5);
    }

    #[test]
    fn test_flat_layout() {
        let config = MqaConfig::new(2, 4);
        let mut mqa = MultiQueryAttention::new(config);

        let seq_len_q = 3;
        let seq_len_kv = 4;
        let num_heads = 2;
        let head_dim = 4;

        // Flattened Q: [num_heads * seq_len_q * head_dim]
        let q_flat: Vec<f32> = (0..num_heads * seq_len_q * head_dim)
            .map(|i| i as f32 * 0.1)
            .collect();

        // Flattened K, V: [seq_len_kv * head_dim]
        let k_flat: Vec<f32> = (0..seq_len_kv * head_dim)
            .map(|i| (i + 50) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len_kv * head_dim)
            .map(|i| (i + 100) as f32 * 0.1)
            .collect();

        let output = mqa.forward_flat(&q_flat, &k_flat, &v_flat, seq_len_q, seq_len_kv);

        assert_eq!(output.len(), num_heads * seq_len_q * head_dim);
    }

    #[test]
    fn test_empty_input() {
        let config = MqaConfig::new(2, 4);
        let mut mqa = MultiQueryAttention::new(config);

        let q: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];
        let k: Vec<Vec<f32>> = vec![];
        let v: Vec<Vec<f32>> = vec![];

        let output = mqa.forward(&q, &k, &v);

        assert_eq!(output.len(), 2);
        assert!(output[0].is_empty());
        assert!(output[1].is_empty());
    }

    #[test]
    fn test_single_token() {
        let config = MqaConfig::new(1, 4);
        let mut mqa = MultiQueryAttention::new(config);

        let q = vec![vec![vec![1.0, 0.0, 0.0, 0.0]]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v = vec![vec![1.0, 2.0, 3.0, 4.0]];

        let output = mqa.forward(&q, &k, &v);

        // Single token attention = output equals value
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        for (out, val) in output[0][0].iter().zip(v[0].iter()) {
            assert!((out - val).abs() < 1e-5);
        }
    }

    #[test]
    fn test_stats() {
        let config = MqaConfig::new(8, 64);
        let mut mqa = MultiQueryAttention::new(config.clone());

        let q: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(10, 64, h)).collect();
        let k = random_matrix(20, 64, 100);
        let v = random_matrix(20, 64, 200);

        mqa.forward(&q, &k, &v);

        let stats = mqa.stats();
        assert_eq!(stats.seq_len_q, 10);
        assert_eq!(stats.seq_len_kv, 20);
        assert_eq!(stats.attention_ops, 8 * 10 * 20);
        assert!((stats.memory_saved_ratio - 0.875).abs() < 1e-5);
    }

    #[test]
    fn test_compare_kv_cache_size() {
        let comparison = compare_kv_cache_size(32, 4096, 128, 2); // fp16

        // MHA: 2 × 32 × 4096 × 128 × 2 = 67,108,864 bytes = 64 MB
        assert_eq!(comparison.mha_size, 67_108_864);

        // MQA: 2 × 1 × 4096 × 128 × 2 = 2,097,152 bytes = 2 MB
        assert_eq!(comparison.mqa_size, 2_097_152);

        // 32x smaller
        assert!((comparison.savings_factor - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_stability() {
        let config = MqaConfig::new(2, 4);
        let mut mqa = MultiQueryAttention::new(config);

        // Large values that could cause overflow in naive softmax
        let q = vec![vec![vec![100.0, 100.0, 100.0, 100.0]]];
        let k = vec![
            vec![100.0, 100.0, 100.0, 100.0],
            vec![-100.0, -100.0, -100.0, -100.0],
        ];
        let v = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let output = mqa.forward(&q, &k, &v);

        // Should not have NaN or Inf
        assert!(output[0][0].iter().all(|x| x.is_finite()));

        // High score token (k[0]) should dominate
        assert!(output[0][0][0] > 0.99);
    }

    #[test]
    fn test_multiple_heads_different_outputs() {
        let config = MqaConfig::new(2, 4);
        let mut mqa = MultiQueryAttention::new(config);

        // Two different query heads
        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0]], // Attends more to k[0]
            vec![vec![0.0, 1.0, 0.0, 0.0]], // Attends more to k[1]
        ];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let v = vec![vec![10.0, 0.0, 0.0, 0.0], vec![0.0, 10.0, 0.0, 0.0]];

        let output = mqa.forward(&q, &k, &v);

        // Head 0 should favor v[0]
        assert!(output[0][0][0] > output[0][0][1]);

        // Head 1 should favor v[1]
        assert!(output[1][0][1] > output[1][0][0]);
    }

    #[test]
    fn test_standard_mha() {
        let num_heads = 2;
        let q: Vec<Vec<Vec<f32>>> = (0..num_heads).map(|h| random_matrix(3, 4, h + 1)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|h| random_matrix(3, 4, h + 10))
            .collect();
        let v: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|h| random_matrix(3, 4, h + 20))
            .collect();

        let output = standard_mha(&q, &k, &v, 0.5, false);

        assert_eq!(output.len(), num_heads);
        assert_eq!(output[0].len(), 3);
        assert_eq!(output[0][0].len(), 4);
    }

    #[test]
    fn test_causal_mha_equivalence() {
        let config = MqaConfig::new(4, 8).causal(true);

        let q: Vec<Vec<Vec<f32>>> = (0..4).map(|h| random_matrix(6, 8, h + 1)).collect();
        let k = random_matrix(6, 8, 50);
        let v = random_matrix(6, 8, 100);

        assert!(verify_mqa_mha_equivalence(&q, &k, &v, &config, 1e-5));
    }

    #[test]
    fn test_many_heads() {
        let config = MqaConfig::new(32, 64);
        let mut mqa = MultiQueryAttention::new(config);

        let q: Vec<Vec<Vec<f32>>> = (0..32).map(|h| random_matrix(8, 64, h)).collect();
        let k = random_matrix(16, 64, 100);
        let v = random_matrix(16, 64, 200);

        let output = mqa.forward(&q, &k, &v);

        assert_eq!(output.len(), 32);
        assert_eq!(output[0].len(), 8);
        assert_eq!(output[0][0].len(), 64);

        // Verify memory savings stat
        assert!((mqa.stats().memory_saved_ratio - 0.96875).abs() < 1e-5);
    }
}
