// Grouped-Query Attention (GQA)
//
// Efficient attention mechanism that bridges Multi-Head Attention (MHA) and
// Multi-Query Attention (MQA). From "GQA: Training Generalized Multi-Query
// Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023).
//
// Key concepts:
// - MHA: Each query head has its own K, V head (num_kv_heads = num_query_heads)
// - MQA: All query heads share a single K, V head (num_kv_heads = 1)
// - GQA: Groups of query heads share K, V heads (num_kv_heads = num_groups)
//
// Memory comparison for KV cache (for num_query_heads=32):
// - MHA: 32 KV heads → full memory
// - GQA-8: 8 KV heads → 4x smaller than MHA
// - GQA-4: 4 KV heads → 8x smaller than MHA
// - MQA: 1 KV head → 32x smaller than MHA
//
// Used in: LLaMA 2 (70B), Mistral, Mixtral, and many others
//
// Trade-offs:
// - GQA balances quality (closer to MHA) with efficiency (closer to MQA)
// - GQA-8 is a common choice: good quality with 4x memory savings
//
// Algorithm:
// 1. Divide query heads into groups (num_query_heads / num_kv_heads per group)
// 2. Each group shares a single K, V head
// 3. All query heads in a group attend to the same K, V

/// Configuration for Grouped-Query Attention
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads
    pub num_query_heads: usize,
    /// Number of KV heads (groups)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Softmax scale (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
}

impl GqaConfig {
    /// Create a new configuration
    pub fn new(num_query_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        assert!(
            num_query_heads % num_kv_heads == 0,
            "num_query_heads must be divisible by num_kv_heads"
        );
        Self {
            num_query_heads,
            num_kv_heads,
            head_dim,
            causal: false,
            softmax_scale: None,
        }
    }

    /// Create MHA configuration (num_kv_heads = num_query_heads)
    pub fn mha(num_heads: usize, head_dim: usize) -> Self {
        Self::new(num_heads, num_heads, head_dim)
    }

    /// Create MQA configuration (num_kv_heads = 1)
    pub fn mqa(num_query_heads: usize, head_dim: usize) -> Self {
        Self::new(num_query_heads, 1, head_dim)
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

    /// Number of query heads per KV head (group size)
    pub fn queries_per_kv(&self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    /// Calculate memory savings vs MHA
    pub fn memory_savings_ratio(&self) -> f32 {
        // MHA uses num_query_heads KV heads
        // GQA uses num_kv_heads KV heads
        1.0 - (self.num_kv_heads as f32 / self.num_query_heads as f32)
    }

    /// Check if this is effectively MHA
    pub fn is_mha(&self) -> bool {
        self.num_query_heads == self.num_kv_heads
    }

    /// Check if this is effectively MQA
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }
}

impl Default for GqaConfig {
    fn default() -> Self {
        // Default: 32 query heads, 8 KV heads (GQA-8, like LLaMA 2 70B)
        Self {
            num_query_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            causal: false,
            softmax_scale: None,
        }
    }
}

/// Statistics from GQA computation
#[derive(Debug, Clone, Default)]
pub struct GqaStats {
    /// Sequence length processed
    pub seq_len_q: usize,
    /// KV sequence length
    pub seq_len_kv: usize,
    /// Number of attention operations
    pub attention_ops: usize,
    /// Memory savings vs MHA (ratio)
    pub memory_saved_ratio: f32,
    /// Effective attention type
    pub attention_type: AttentionType,
}

/// Type of attention based on configuration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AttentionType {
    /// Multi-Head Attention (1:1 query to KV)
    Mha,
    /// Multi-Query Attention (N:1 query to KV)
    Mqa,
    /// Grouped-Query Attention (G:1 query to KV)
    #[default]
    Gqa,
}

/// Grouped-Query Attention computation
///
/// Computes attention where groups of query heads share KV heads.
///
/// Input shapes:
/// - Q: [num_query_heads, seq_len_q, head_dim]
/// - K: [num_kv_heads, seq_len_kv, head_dim]
/// - V: [num_kv_heads, seq_len_kv, head_dim]
///
/// Output shape: [num_query_heads, seq_len_q, head_dim]
pub struct GroupedQueryAttention {
    config: GqaConfig,
    stats: GqaStats,
}

impl GroupedQueryAttention {
    /// Create a new Grouped-Query Attention instance
    pub fn new(config: GqaConfig) -> Self {
        Self {
            config,
            stats: GqaStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &GqaConfig {
        &self.config
    }

    /// Get statistics from last computation
    pub fn stats(&self) -> &GqaStats {
        &self.stats
    }

    /// Get the KV head index for a query head
    fn kv_head_for_query(&self, query_head: usize) -> usize {
        query_head / self.config.queries_per_kv()
    }

    /// Compute attention for a single query head using its corresponding KV head
    /// Q: [seq_len_q, head_dim]
    /// K: [seq_len_kv, head_dim]
    /// V: [seq_len_kv, head_dim]
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

    /// Compute Grouped-Query Attention
    /// Q: [num_query_heads, seq_len_q, head_dim]
    /// K: [num_kv_heads, seq_len_kv, head_dim]
    /// V: [num_kv_heads, seq_len_kv, head_dim]
    /// Returns: [num_query_heads, seq_len_q, head_dim]
    pub fn forward(
        &mut self,
        q: &[Vec<Vec<f32>>],
        k: &[Vec<Vec<f32>>],
        v: &[Vec<Vec<f32>>],
    ) -> Vec<Vec<Vec<f32>>> {
        assert_eq!(q.len(), self.config.num_query_heads);
        assert_eq!(k.len(), self.config.num_kv_heads);
        assert_eq!(v.len(), self.config.num_kv_heads);

        let seq_len_q = if !q.is_empty() { q[0].len() } else { 0 };
        let seq_len_kv = if !k.is_empty() { k[0].len() } else { 0 };
        let scale = self.config.get_scale();

        // Update stats
        self.stats.seq_len_q = seq_len_q;
        self.stats.seq_len_kv = seq_len_kv;
        self.stats.attention_ops = self.config.num_query_heads * seq_len_q * seq_len_kv;
        self.stats.memory_saved_ratio = self.config.memory_savings_ratio();
        self.stats.attention_type = if self.config.is_mha() {
            AttentionType::Mha
        } else if self.config.is_mqa() {
            AttentionType::Mqa
        } else {
            AttentionType::Gqa
        };

        // Compute attention for each query head
        let mut outputs = Vec::with_capacity(self.config.num_query_heads);

        for (query_head, q_head) in q.iter().enumerate() {
            let kv_head = self.kv_head_for_query(query_head);
            let head_output = self.attention_single_head(q_head, &k[kv_head], &v[kv_head], scale);
            outputs.push(head_output);
        }

        outputs
    }

    /// Compute attention with flat tensor layout
    /// Q: [num_query_heads * seq_len_q * head_dim] (flattened)
    /// K: [num_kv_heads * seq_len_kv * head_dim] (flattened)
    /// V: [num_kv_heads * seq_len_kv * head_dim] (flattened)
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
        let num_query_heads = self.config.num_query_heads;
        let num_kv_heads = self.config.num_kv_heads;

        // Reshape Q to [num_query_heads, seq_len_q, head_dim]
        let q_3d: Vec<Vec<Vec<f32>>> = (0..num_query_heads)
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

        // Reshape K to [num_kv_heads, seq_len_kv, head_dim]
        let k_3d: Vec<Vec<Vec<f32>>> = (0..num_kv_heads)
            .map(|h| {
                let head_start = h * seq_len_kv * head_dim;
                (0..seq_len_kv)
                    .map(|s| {
                        let start = head_start + s * head_dim;
                        k[start..start + head_dim].to_vec()
                    })
                    .collect()
            })
            .collect();

        // Reshape V to [num_kv_heads, seq_len_kv, head_dim]
        let v_3d: Vec<Vec<Vec<f32>>> = (0..num_kv_heads)
            .map(|h| {
                let head_start = h * seq_len_kv * head_dim;
                (0..seq_len_kv)
                    .map(|s| {
                        let start = head_start + s * head_dim;
                        v[start..start + head_dim].to_vec()
                    })
                    .collect()
            })
            .collect();

        // Compute attention
        let output_3d = self.forward(&q_3d, &k_3d, &v_3d);

        // Flatten output
        output_3d
            .into_iter()
            .flat_map(|head| head.into_iter().flatten())
            .collect()
    }
}

/// Calculate KV cache size for different attention configurations
pub struct GqaKvCacheComparison {
    pub mha_size: usize,
    pub gqa_size: usize,
    pub mqa_size: usize,
    pub gqa_savings_vs_mha: f32,
    pub mqa_savings_vs_mha: f32,
}

/// Compare KV cache memory requirements across MHA, GQA, and MQA
pub fn compare_attention_kv_cache(
    num_query_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    bytes_per_element: usize,
) -> GqaKvCacheComparison {
    // MHA: 2 (K+V) × num_query_heads × seq_len × head_dim × bytes
    let mha_size = 2 * num_query_heads * seq_len * head_dim * bytes_per_element;

    // GQA: 2 (K+V) × num_kv_heads × seq_len × head_dim × bytes
    let gqa_size = 2 * num_kv_heads * seq_len * head_dim * bytes_per_element;

    // MQA: 2 (K+V) × 1 × seq_len × head_dim × bytes
    let mqa_size = 2 * seq_len * head_dim * bytes_per_element;

    GqaKvCacheComparison {
        mha_size,
        gqa_size,
        mqa_size,
        gqa_savings_vs_mha: 1.0 - (gqa_size as f32 / mha_size as f32),
        mqa_savings_vs_mha: 1.0 - (mqa_size as f32 / mha_size as f32),
    }
}

/// Verify GQA produces same output as MHA when using same number of KV heads
pub fn verify_gqa_mha_equivalence(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<Vec<f32>>],
    v: &[Vec<Vec<f32>>],
    tolerance: f32,
) -> bool {
    let num_heads = q.len();
    assert_eq!(k.len(), num_heads);
    assert_eq!(v.len(), num_heads);

    let head_dim = if !q.is_empty() && !q[0].is_empty() {
        q[0][0].len()
    } else {
        return true;
    };

    // GQA with num_kv_heads = num_query_heads (should be equivalent to MHA)
    let config = GqaConfig::mha(num_heads, head_dim);
    let mut gqa = GroupedQueryAttention::new(config);
    let gqa_output = gqa.forward(q, k, v);

    // Compare outputs
    if gqa_output.len() != num_heads {
        return false;
    }

    for head in 0..num_heads {
        if gqa_output[head].len() != q[head].len() {
            return false;
        }
        for (gqa_row, q_row_idx) in gqa_output[head].iter().zip(0..q[head].len()) {
            // We need to compute expected MHA output
            let seq_len_kv = k[head].len();
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Compute MHA attention for this position
            let q_vec = &q[head][q_row_idx];
            let mut scores: Vec<f32> = Vec::with_capacity(seq_len_kv);

            for k_vec in &k[head] {
                let score: f32 = q_vec
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }

            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            let mut expected = vec![0.0f32; head_dim];
            for (j, v_vec) in v[head].iter().enumerate() {
                let weight = if sum_exp > 0.0 {
                    exp_scores[j] / sum_exp
                } else {
                    0.0
                };
                for (d, &v_val) in v_vec.iter().enumerate() {
                    expected[d] += weight * v_val;
                }
            }

            // Compare
            for (&g, &e) in gqa_row.iter().zip(expected.iter()) {
                if (g - e).abs() > tolerance {
                    return false;
                }
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
        let config = GqaConfig::default();
        assert_eq!(config.num_query_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert!(!config.causal);
    }

    #[test]
    fn test_config_builder() {
        let config = GqaConfig::new(16, 4, 64).causal(true).scale(0.0625);

        assert_eq!(config.num_query_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
        assert_eq!(config.softmax_scale, Some(0.0625));
        assert_eq!(config.queries_per_kv(), 4);
    }

    #[test]
    fn test_config_mha() {
        let config = GqaConfig::mha(8, 64);
        assert_eq!(config.num_query_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert!(config.is_mha());
        assert!(!config.is_mqa());
        assert_eq!(config.queries_per_kv(), 1);
        assert!((config.memory_savings_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_mqa() {
        let config = GqaConfig::mqa(32, 64);
        assert_eq!(config.num_query_heads, 32);
        assert_eq!(config.num_kv_heads, 1);
        assert!(!config.is_mha());
        assert!(config.is_mqa());
        assert_eq!(config.queries_per_kv(), 32);
        assert!((config.memory_savings_ratio() - 0.96875).abs() < 1e-6);
    }

    #[test]
    fn test_get_scale() {
        let config = GqaConfig::new(8, 2, 64);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected_scale).abs() < 1e-6);

        let config_custom = config.scale(0.1);
        assert_eq!(config_custom.get_scale(), 0.1);
    }

    #[test]
    fn test_memory_savings_ratio() {
        // GQA-8 with 32 query heads
        let config = GqaConfig::new(32, 8, 64);
        // 8/32 = 0.25 -> savings = 0.75
        assert!((config.memory_savings_ratio() - 0.75).abs() < 1e-6);

        // GQA-4 with 32 query heads
        let config_4 = GqaConfig::new(32, 4, 64);
        // 4/32 = 0.125 -> savings = 0.875
        assert!((config_4.memory_savings_ratio() - 0.875).abs() < 1e-6);
    }

    #[test]
    fn test_basic_attention() {
        let config = GqaConfig::new(4, 2, 4); // 4 query heads, 2 KV heads
        let mut gqa = GroupedQueryAttention::new(config);

        // 4 query heads, each with 2 queries of dim 4
        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
            vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]],
            vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]],
        ];

        // 2 KV heads with 2 positions each
        let k = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        ];
        let v = vec![
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            vec![vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
        ];

        let output = gqa.forward(&q, &k, &v);

        assert_eq!(output.len(), 4); // 4 query heads
        assert_eq!(output[0].len(), 2); // 2 query positions
        assert_eq!(output[0][0].len(), 4); // head_dim
    }

    #[test]
    fn test_kv_head_mapping() {
        let config = GqaConfig::new(8, 2, 64);
        let gqa = GroupedQueryAttention::new(config);

        // Query heads 0-3 should use KV head 0
        assert_eq!(gqa.kv_head_for_query(0), 0);
        assert_eq!(gqa.kv_head_for_query(1), 0);
        assert_eq!(gqa.kv_head_for_query(2), 0);
        assert_eq!(gqa.kv_head_for_query(3), 0);

        // Query heads 4-7 should use KV head 1
        assert_eq!(gqa.kv_head_for_query(4), 1);
        assert_eq!(gqa.kv_head_for_query(5), 1);
        assert_eq!(gqa.kv_head_for_query(6), 1);
        assert_eq!(gqa.kv_head_for_query(7), 1);
    }

    #[test]
    fn test_gqa_mha_equivalence() {
        let num_heads = 4;
        let q: Vec<Vec<Vec<f32>>> = (0..num_heads).map(|h| random_matrix(5, 8, h + 1)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|h| random_matrix(5, 8, h + 10))
            .collect();
        let v: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|h| random_matrix(5, 8, h + 20))
            .collect();

        assert!(verify_gqa_mha_equivalence(&q, &k, &v, 1e-5));
    }

    #[test]
    fn test_grouped_sharing() {
        // Test that query heads in the same group produce different outputs
        // when they have different queries but same K, V
        let config = GqaConfig::new(4, 2, 4); // 2 groups of 2
        let mut gqa = GroupedQueryAttention::new(config);

        // Different queries for heads 0 and 1 (same group, share KV head 0)
        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0]], // Head 0: attends more to k[0][0]
            vec![vec![0.0, 1.0, 0.0, 0.0]], // Head 1: attends more to k[0][1]
            vec![vec![0.0, 0.0, 1.0, 0.0]], // Head 2: uses KV head 1
            vec![vec![0.0, 0.0, 0.0, 1.0]], // Head 3: uses KV head 1
        ];

        let k = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]], // KV head 0
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]], // KV head 1
        ];
        let v = vec![
            vec![vec![10.0, 0.0, 0.0, 0.0], vec![0.0, 10.0, 0.0, 0.0]], // KV head 0
            vec![vec![0.0, 0.0, 10.0, 0.0], vec![0.0, 0.0, 0.0, 10.0]], // KV head 1
        ];

        let output = gqa.forward(&q, &k, &v);

        // Heads 0 and 1 should have different outputs (different queries, same K,V)
        assert!(output[0][0][0] > output[0][0][1]); // Head 0 favors v[0][0]
        assert!(output[1][0][1] > output[1][0][0]); // Head 1 favors v[0][1]

        // Heads 2 and 3 use KV head 1
        assert!(output[2][0][2] > output[2][0][3]); // Head 2 favors v[1][0]
        assert!(output[3][0][3] > output[3][0][2]); // Head 3 favors v[1][1]
    }

    #[test]
    fn test_causal_masking() {
        let config = GqaConfig::new(2, 1, 4).causal(true);
        let mut gqa = GroupedQueryAttention::new(config);

        // 2 query heads, position 0
        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0]],
            vec![vec![0.0, 1.0, 0.0, 0.0]],
        ];

        // 1 KV head with 3 positions
        let k = vec![vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]];
        let v = vec![vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]];

        let output = gqa.forward(&q, &k, &v);

        // Position 0 should only attend to position 0 (causal mask)
        assert!((output[0][0][0] - 1.0).abs() < 1e-5);
        assert!(output[0][0][1].abs() < 1e-5);
    }

    #[test]
    fn test_flat_layout() {
        let config = GqaConfig::new(4, 2, 4);
        let mut gqa = GroupedQueryAttention::new(config);

        let seq_len_q = 3;
        let seq_len_kv = 4;
        let num_query_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;

        // Flattened Q: [num_query_heads * seq_len_q * head_dim]
        let q_flat: Vec<f32> = (0..num_query_heads * seq_len_q * head_dim)
            .map(|i| i as f32 * 0.1)
            .collect();

        // Flattened K, V: [num_kv_heads * seq_len_kv * head_dim]
        let k_flat: Vec<f32> = (0..num_kv_heads * seq_len_kv * head_dim)
            .map(|i| (i + 50) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..num_kv_heads * seq_len_kv * head_dim)
            .map(|i| (i + 100) as f32 * 0.1)
            .collect();

        let output = gqa.forward_flat(&q_flat, &k_flat, &v_flat, seq_len_q, seq_len_kv);

        assert_eq!(output.len(), num_query_heads * seq_len_q * head_dim);
    }

    #[test]
    fn test_empty_input() {
        let config = GqaConfig::new(4, 2, 4);
        let mut gqa = GroupedQueryAttention::new(config);

        let q: Vec<Vec<Vec<f32>>> = vec![vec![], vec![], vec![], vec![]];
        let k: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];
        let v: Vec<Vec<Vec<f32>>> = vec![vec![], vec![]];

        let output = gqa.forward(&q, &k, &v);

        assert_eq!(output.len(), 4);
        assert!(output[0].is_empty());
    }

    #[test]
    fn test_single_token() {
        let config = GqaConfig::new(2, 1, 4);
        let mut gqa = GroupedQueryAttention::new(config);

        let q = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0]],
            vec![vec![0.0, 1.0, 0.0, 0.0]],
        ];
        let k = vec![vec![vec![1.0, 0.0, 0.0, 0.0]]];
        let v = vec![vec![vec![1.0, 2.0, 3.0, 4.0]]];

        let output = gqa.forward(&q, &k, &v);

        // Single token attention = output equals value for all heads
        assert_eq!(output.len(), 2);
        for head in &output {
            assert_eq!(head.len(), 1);
            for (out, val) in head[0].iter().zip(v[0][0].iter()) {
                assert!((out - val).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_stats() {
        let config = GqaConfig::new(32, 8, 64);
        let mut gqa = GroupedQueryAttention::new(config.clone());

        let q: Vec<Vec<Vec<f32>>> = (0..32).map(|h| random_matrix(10, 64, h)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(20, 64, h + 100)).collect();
        let v: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(20, 64, h + 200)).collect();

        gqa.forward(&q, &k, &v);

        let stats = gqa.stats();
        assert_eq!(stats.seq_len_q, 10);
        assert_eq!(stats.seq_len_kv, 20);
        assert_eq!(stats.attention_ops, 32 * 10 * 20);
        assert!((stats.memory_saved_ratio - 0.75).abs() < 1e-5);
        assert_eq!(stats.attention_type, AttentionType::Gqa);
    }

    #[test]
    fn test_attention_type_detection() {
        // GQA
        let config_gqa = GqaConfig::new(32, 8, 64);
        let mut gqa = GroupedQueryAttention::new(config_gqa);
        let q: Vec<Vec<Vec<f32>>> = (0..32).map(|h| random_matrix(2, 64, h)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h + 100)).collect();
        let v: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h + 200)).collect();
        gqa.forward(&q, &k, &v);
        assert_eq!(gqa.stats().attention_type, AttentionType::Gqa);

        // MHA
        let config_mha = GqaConfig::mha(8, 64);
        let mut mha = GroupedQueryAttention::new(config_mha);
        let q: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h + 100)).collect();
        let v: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h + 200)).collect();
        mha.forward(&q, &k, &v);
        assert_eq!(mha.stats().attention_type, AttentionType::Mha);

        // MQA
        let config_mqa = GqaConfig::mqa(8, 64);
        let mut mqa = GroupedQueryAttention::new(config_mqa);
        let q: Vec<Vec<Vec<f32>>> = (0..8).map(|h| random_matrix(2, 64, h)).collect();
        let k: Vec<Vec<Vec<f32>>> = (0..1).map(|h| random_matrix(2, 64, h + 100)).collect();
        let v: Vec<Vec<Vec<f32>>> = (0..1).map(|h| random_matrix(2, 64, h + 200)).collect();
        mqa.forward(&q, &k, &v);
        assert_eq!(mqa.stats().attention_type, AttentionType::Mqa);
    }

    #[test]
    fn test_compare_kv_cache() {
        let comparison = compare_attention_kv_cache(32, 8, 4096, 128, 2); // fp16

        // MHA: 2 × 32 × 4096 × 128 × 2 = 67,108,864 bytes
        assert_eq!(comparison.mha_size, 67_108_864);

        // GQA-8: 2 × 8 × 4096 × 128 × 2 = 16,777,216 bytes
        assert_eq!(comparison.gqa_size, 16_777_216);

        // MQA: 2 × 1 × 4096 × 128 × 2 = 2,097,152 bytes
        assert_eq!(comparison.mqa_size, 2_097_152);

        // GQA-8 saves 75% vs MHA
        assert!((comparison.gqa_savings_vs_mha - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_stability() {
        let config = GqaConfig::new(4, 2, 4);
        let mut gqa = GroupedQueryAttention::new(config);

        // Large values
        let q = vec![
            vec![vec![100.0, 100.0, 100.0, 100.0]],
            vec![vec![100.0, 100.0, 100.0, 100.0]],
            vec![vec![-100.0, -100.0, -100.0, -100.0]],
            vec![vec![-100.0, -100.0, -100.0, -100.0]],
        ];
        let k = vec![
            vec![
                vec![100.0, 100.0, 100.0, 100.0],
                vec![-100.0, -100.0, -100.0, -100.0],
            ],
            vec![
                vec![100.0, 100.0, 100.0, 100.0],
                vec![-100.0, -100.0, -100.0, -100.0],
            ],
        ];
        let v = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        ];

        let output = gqa.forward(&q, &k, &v);

        // Should not have NaN or Inf
        for head in &output {
            for row in head {
                assert!(row.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn test_llama2_70b_config() {
        // LLaMA 2 70B uses: 64 query heads, 8 KV heads
        let config = GqaConfig::new(64, 8, 128);

        assert_eq!(config.queries_per_kv(), 8);
        assert!((config.memory_savings_ratio() - 0.875).abs() < 1e-6); // 87.5% savings
        assert!(!config.is_mha());
        assert!(!config.is_mqa());
    }

    #[test]
    #[should_panic(expected = "num_query_heads must be divisible by num_kv_heads")]
    fn test_invalid_config() {
        // 7 is not divisible by 3
        let _ = GqaConfig::new(7, 3, 64);
    }
}
