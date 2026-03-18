// RoPE (Rotary Position Embedding)
//
// Dominant position encoding method in modern LLMs, introduced in RoFormer
// (Su et al., 2021). Used in LLaMA, Mistral, GPT-NeoX, and many others.
//
// Key concepts:
// - Encodes position by rotating pairs of dimensions in the embedding space
// - Relative position information emerges naturally from the rotation angles
// - No learned parameters - purely mathematical construction
// - Excellent length extrapolation properties
//
// Algorithm:
// For each pair of dimensions (2i, 2i+1) at position m:
//   θ_i = 10000^(-2i/d) where d is head dimension
//   [x_{2i}, x_{2i+1}] → [x_{2i}cos(mθ_i) - x_{2i+1}sin(mθ_i),
//                        x_{2i}sin(mθ_i) + x_{2i+1}cos(mθ_i)]
//
// This is equivalent to multiplying by a rotation matrix:
//   [cos(mθ)  -sin(mθ)]   [x_{2i}  ]
//   [sin(mθ)   cos(mθ)] × [x_{2i+1}]
//
// Key insight: When computing Q @ K^T, the rotation matrices combine:
//   Q_rotated @ K_rotated^T encodes relative position (m - n)
//
// Variants:
// - Standard RoPE: θ_i = 10000^(-2i/d)
// - LLaMA-style: interleaved or sequential dimension pairing
// - NTK-aware scaling: for extended context lengths
// - YaRN: improved length extrapolation

use std::f32::consts::PI;

/// Multi-head tensor: [num_heads, seq_len, head_dim]
pub type MultiHeadTensor = Vec<Vec<Vec<f32>>>;

/// Configuration for RoPE
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Head dimension (must be even)
    pub head_dim: usize,
    /// Base frequency for position encoding (default: 10000.0)
    pub base: f32,
    /// Maximum sequence length for precomputing frequencies
    pub max_seq_len: usize,
    /// Scaling factor for extended context (NTK-aware scaling)
    pub scaling_factor: f32,
    /// Whether to use interleaved dimension pairing (like LLaMA)
    pub interleaved: bool,
}

impl RopeConfig {
    /// Create a new configuration
    pub fn new(head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        Self {
            head_dim,
            base: 10000.0,
            max_seq_len: 4096,
            scaling_factor: 1.0,
            interleaved: false,
        }
    }

    /// Set base frequency
    pub fn base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set maximum sequence length
    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set scaling factor for extended context
    pub fn scaling_factor(mut self, factor: f32) -> Self {
        self.scaling_factor = factor;
        self
    }

    /// Use interleaved dimension pairing (LLaMA-style)
    pub fn interleaved(mut self, interleaved: bool) -> Self {
        self.interleaved = interleaved;
        self
    }

    /// Compute inverse frequencies: θ_i = base^(-2i/d)
    fn compute_inv_freq(&self) -> Vec<f32> {
        let half_dim = self.head_dim / 2;
        (0..half_dim)
            .map(|i| {
                let exp = -2.0 * (i as f32) / (self.head_dim as f32);
                self.base.powf(exp)
            })
            .collect()
    }
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 64,
            base: 10000.0,
            max_seq_len: 4096,
            scaling_factor: 1.0,
            interleaved: false,
        }
    }
}

/// Precomputed RoPE frequencies for efficient application
#[derive(Debug, Clone)]
pub struct RopeFrequencies {
    /// Cosine values: [max_seq_len, head_dim/2]
    cos_cache: Vec<Vec<f32>>,
    /// Sine values: [max_seq_len, head_dim/2]
    sin_cache: Vec<Vec<f32>>,
    /// Configuration
    config: RopeConfig,
}

impl RopeFrequencies {
    /// Create and precompute RoPE frequencies
    pub fn new(config: RopeConfig) -> Self {
        let inv_freq = config.compute_inv_freq();
        let half_dim = config.head_dim / 2;

        let mut cos_cache = Vec::with_capacity(config.max_seq_len);
        let mut sin_cache = Vec::with_capacity(config.max_seq_len);

        for pos in 0..config.max_seq_len {
            let scaled_pos = pos as f32 / config.scaling_factor;
            let mut cos_row = Vec::with_capacity(half_dim);
            let mut sin_row = Vec::with_capacity(half_dim);

            for &freq in &inv_freq {
                let angle = scaled_pos * freq;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }

            cos_cache.push(cos_row);
            sin_cache.push(sin_row);
        }

        Self {
            cos_cache,
            sin_cache,
            config,
        }
    }

    /// Get config
    pub fn config(&self) -> &RopeConfig {
        &self.config
    }

    /// Get cos values for a position
    pub fn cos(&self, pos: usize) -> &[f32] {
        &self.cos_cache[pos.min(self.config.max_seq_len - 1)]
    }

    /// Get sin values for a position
    pub fn sin(&self, pos: usize) -> &[f32] {
        &self.sin_cache[pos.min(self.config.max_seq_len - 1)]
    }

    /// Apply RoPE to a single vector at a given position
    /// x: [head_dim]
    /// Returns: rotated vector [head_dim]
    pub fn apply(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let half_dim = head_dim / 2;

        assert_eq!(x.len(), head_dim);

        let cos = self.cos(pos);
        let sin = self.sin(pos);

        let mut rotated = vec![0.0f32; head_dim];

        if self.config.interleaved {
            // Interleaved: pairs are (0,1), (2,3), (4,5), ...
            for i in 0..half_dim {
                let x0 = x[2 * i];
                let x1 = x[2 * i + 1];
                rotated[2 * i] = x0 * cos[i] - x1 * sin[i];
                rotated[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
            }
        } else {
            // Sequential: pairs are (0, half), (1, half+1), ...
            for i in 0..half_dim {
                let x0 = x[i];
                let x1 = x[i + half_dim];
                rotated[i] = x0 * cos[i] - x1 * sin[i];
                rotated[i + half_dim] = x0 * sin[i] + x1 * cos[i];
            }
        }

        rotated
    }

    /// Apply RoPE to a batch of vectors
    /// x: [seq_len, head_dim]
    /// start_pos: starting position (for incremental decoding)
    /// Returns: rotated vectors [seq_len, head_dim]
    pub fn apply_batch(&self, x: &[Vec<f32>], start_pos: usize) -> Vec<Vec<f32>> {
        x.iter()
            .enumerate()
            .map(|(i, vec)| self.apply(vec, start_pos + i))
            .collect()
    }

    /// Apply RoPE to queries and keys together
    /// Useful for attention computation
    /// q: [num_heads, seq_len, head_dim]
    /// k: [num_kv_heads, seq_len, head_dim]
    /// start_pos: starting position
    /// Returns: (rotated_q, rotated_k)
    pub fn apply_qk(
        &self,
        q: &[Vec<Vec<f32>>],
        k: &[Vec<Vec<f32>>],
        start_pos: usize,
    ) -> (MultiHeadTensor, MultiHeadTensor) {
        let rotated_q: MultiHeadTensor = q
            .iter()
            .map(|head| self.apply_batch(head, start_pos))
            .collect();

        let rotated_k: MultiHeadTensor = k
            .iter()
            .map(|head| self.apply_batch(head, start_pos))
            .collect();

        (rotated_q, rotated_k)
    }

    /// Apply RoPE with flat tensor layout
    /// x: [num_heads * seq_len * head_dim]
    /// num_heads: number of attention heads
    /// seq_len: sequence length
    /// start_pos: starting position
    /// Returns: rotated tensor [num_heads * seq_len * head_dim]
    pub fn apply_flat(
        &self,
        x: &[f32],
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let mut result = Vec::with_capacity(x.len());

        for head in 0..num_heads {
            for pos in 0..seq_len {
                let offset = head * seq_len * head_dim + pos * head_dim;
                let vec = &x[offset..offset + head_dim];
                let rotated = self.apply(vec, start_pos + pos);
                result.extend(rotated);
            }
        }

        result
    }
}

/// Compute RoPE angle for a given position and dimension
/// This is the basic building block: θ_i * position
pub fn rope_angle(pos: usize, dim_idx: usize, head_dim: usize, base: f32) -> f32 {
    let exp = -2.0 * (dim_idx as f32) / (head_dim as f32);
    let inv_freq = base.powf(exp);
    (pos as f32) * inv_freq
}

/// Apply RoPE rotation to a pair of values
/// Returns (x0_rotated, x1_rotated)
pub fn rotate_pair(x0: f32, x1: f32, cos: f32, sin: f32) -> (f32, f32) {
    (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
}

/// NTK-aware scaling for extended context lengths
/// From "NTK-Aware Scaled RoPE"
pub fn ntk_scaled_base(original_base: f32, original_max_len: usize, new_max_len: usize) -> f32 {
    let scale = new_max_len as f32 / original_max_len as f32;
    original_base * scale.powf(original_max_len as f32 / (original_max_len as f32 - 2.0))
}

/// YaRN (Yet another RoPE extensioN) scaling
/// Combines NTK scaling with attention scaling for better length extrapolation
pub struct YarnConfig {
    /// Original max context length
    pub original_max_len: usize,
    /// New extended max context length
    pub extended_max_len: usize,
    /// Attention scaling factor
    pub attention_factor: f32,
    /// Beta fast for interpolation
    pub beta_fast: f32,
    /// Beta slow for interpolation
    pub beta_slow: f32,
}

impl YarnConfig {
    pub fn new(original_max_len: usize, extended_max_len: usize) -> Self {
        let scale = extended_max_len as f32 / original_max_len as f32;
        Self {
            original_max_len,
            extended_max_len,
            attention_factor: 0.1 * scale.ln() + 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// Compute interpolation factor for a given dimension
    /// Returns a factor between 1.0 (full positional interpolation) and scale (no interpolation/NTK)
    pub fn interpolation_factor(&self, dim: usize, head_dim: usize) -> f32 {
        let scale = self.extended_max_len as f32 / self.original_max_len as f32;

        // Compute wavelength for this dimension
        // Lower dimensions have longer wavelengths
        let base = 10000.0_f32;
        let dim_ratio = (2 * dim) as f32 / head_dim as f32;
        let wavelength = 2.0 * PI * base.powf(dim_ratio);

        // Low and high frequency thresholds based on beta parameters
        let low_freq_wavelen = self.original_max_len as f32 / self.beta_fast;
        let high_freq_wavelen = self.original_max_len as f32 / self.beta_slow;

        if wavelength < high_freq_wavelen {
            // High frequency (short wavelength): no scaling needed
            1.0
        } else if wavelength > low_freq_wavelen {
            // Low frequency (long wavelength): full NTK scaling
            scale
        } else {
            // Medium frequency: linear interpolation
            let smooth = (self.original_max_len as f32 / wavelength - self.beta_slow)
                / (self.beta_fast - self.beta_slow);
            1.0 + (1.0 - smooth) * (scale - 1.0)
        }
    }
}

/// Verify that RoPE preserves relative position information
/// The dot product of two rotated vectors should depend on their relative position
pub fn verify_relative_position_encoding(
    q: &[f32],
    k: &[f32],
    pos_q: usize,
    pos_k: usize,
    rope: &RopeFrequencies,
) -> f32 {
    let q_rot = rope.apply(q, pos_q);
    let k_rot = rope.apply(k, pos_k);

    // Dot product
    q_rot.iter().zip(k_rot.iter()).map(|(&a, &b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..dim)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (seed * 100 + i).hash(&mut hasher);
                ((hasher.finish() % 10000) as f32 / 5000.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = RopeConfig::default();
        assert_eq!(config.head_dim, 64);
        assert!((config.base - 10000.0).abs() < 1e-6);
        assert_eq!(config.max_seq_len, 4096);
        assert!(!config.interleaved);
    }

    #[test]
    fn test_config_builder() {
        let config = RopeConfig::new(128)
            .base(10000.0)
            .max_seq_len(8192)
            .scaling_factor(2.0)
            .interleaved(true);

        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_seq_len, 8192);
        assert!((config.scaling_factor - 2.0).abs() < 1e-6);
        assert!(config.interleaved);
    }

    #[test]
    fn test_inv_freq_computation() {
        let config = RopeConfig::new(8).base(10000.0);
        let inv_freq = config.compute_inv_freq();

        // θ_0 = 10000^0 = 1.0
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);

        // θ_1 = 10000^(-2/8) = 10000^(-0.25)
        let expected = 10000.0_f32.powf(-0.25);
        assert!((inv_freq[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_frequencies_caching() {
        let config = RopeConfig::new(8).max_seq_len(100);
        let rope = RopeFrequencies::new(config);

        // Check that cos and sin are cached for all positions
        for pos in 0..100 {
            let cos = rope.cos(pos);
            let sin = rope.sin(pos);
            assert_eq!(cos.len(), 4); // head_dim / 2
            assert_eq!(sin.len(), 4);
        }
    }

    #[test]
    fn test_cos_sin_values() {
        let config = RopeConfig::new(4).base(10000.0).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        // At position 0, angle = 0, so cos = 1, sin = 0
        let cos_0 = rope.cos(0);
        let sin_0 = rope.sin(0);
        assert!((cos_0[0] - 1.0).abs() < 1e-6);
        assert!(sin_0[0].abs() < 1e-6);
    }

    #[test]
    fn test_apply_single_vector() {
        let config = RopeConfig::new(4).base(10000.0).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        let x = vec![1.0, 0.0, 1.0, 0.0];
        let rotated = rope.apply(&x, 0);

        // At position 0, no rotation should occur (angle = 0)
        for (orig, rot) in x.iter().zip(rotated.iter()) {
            assert!((orig - rot).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_rotation_changes_vector() {
        let config = RopeConfig::new(4).base(10.0).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        let x = vec![1.0, 0.0, 1.0, 0.0];
        let rotated_0 = rope.apply(&x, 0);
        let rotated_5 = rope.apply(&x, 5);

        // Rotation at position 5 should be different from position 0
        let diff: f32 = rotated_0
            .iter()
            .zip(rotated_5.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let config = RopeConfig::new(8).base(10000.0).max_seq_len(100);
        let rope = RopeFrequencies::new(config);

        let x = random_vector(8, 42);
        let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        for pos in [0, 10, 50, 99] {
            let rotated = rope.apply(&x, pos);
            let rotated_norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();

            // Rotation should preserve norm
            assert!(
                (original_norm - rotated_norm).abs() < 1e-5,
                "Norm not preserved at pos {}: {} vs {}",
                pos,
                original_norm,
                rotated_norm
            );
        }
    }

    #[test]
    fn test_apply_batch() {
        let config = RopeConfig::new(4).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        let batch = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]];

        let rotated = rope.apply_batch(&batch, 0);

        assert_eq!(rotated.len(), 2);
        assert_eq!(rotated[0].len(), 4);
    }

    #[test]
    fn test_apply_qk() {
        let config = RopeConfig::new(4).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        // 2 query heads, 2 positions
        let q = vec![
            vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]],
            vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]],
        ];

        // 2 KV heads, 2 positions
        let k = vec![
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]],
            vec![vec![0.5, 0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5, 0.5]],
        ];

        let (rotated_q, rotated_k) = rope.apply_qk(&q, &k, 0);

        assert_eq!(rotated_q.len(), 2);
        assert_eq!(rotated_k.len(), 2);
        assert_eq!(rotated_q[0].len(), 2);
        assert_eq!(rotated_k[0].len(), 2);
    }

    #[test]
    fn test_relative_position_encoding() {
        let config = RopeConfig::new(8).base(100.0).max_seq_len(100);
        let rope = RopeFrequencies::new(config);

        let q = random_vector(8, 1);
        let k = random_vector(8, 2);

        // Same relative position should give same dot product
        let dot_0_0 = verify_relative_position_encoding(&q, &k, 0, 0, &rope);
        let dot_10_10 = verify_relative_position_encoding(&q, &k, 10, 10, &rope);
        let dot_50_50 = verify_relative_position_encoding(&q, &k, 50, 50, &rope);

        // All should be approximately equal (same relative position = 0)
        assert!(
            (dot_0_0 - dot_10_10).abs() < 1e-4,
            "Relative position not encoded correctly"
        );
        assert!(
            (dot_0_0 - dot_50_50).abs() < 1e-4,
            "Relative position not encoded correctly"
        );

        // Different relative positions should give different dot products
        let dot_0_1 = verify_relative_position_encoding(&q, &k, 0, 1, &rope);
        let dot_0_5 = verify_relative_position_encoding(&q, &k, 0, 5, &rope);

        assert!(
            (dot_0_0 - dot_0_1).abs() > 1e-5,
            "Different relative positions should differ"
        );
        assert!(
            (dot_0_1 - dot_0_5).abs() > 1e-5,
            "Different relative positions should differ"
        );
    }

    #[test]
    fn test_interleaved_mode() {
        let config_seq = RopeConfig::new(8).interleaved(false);
        let config_int = RopeConfig::new(8).interleaved(true);

        let rope_seq = RopeFrequencies::new(config_seq);
        let rope_int = RopeFrequencies::new(config_int);

        let x = random_vector(8, 42);

        let rotated_seq = rope_seq.apply(&x, 5);
        let rotated_int = rope_int.apply(&x, 5);

        // Interleaved and sequential should give different results
        let diff: f32 = rotated_seq
            .iter()
            .zip(rotated_int.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);

        // But both should preserve norm
        let norm_seq: f32 = rotated_seq.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_int: f32 = rotated_int.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!((norm_seq - norm_orig).abs() < 1e-5);
        assert!((norm_int - norm_orig).abs() < 1e-5);
    }

    #[test]
    fn test_apply_flat() {
        let config = RopeConfig::new(4).max_seq_len(10);
        let rope = RopeFrequencies::new(config);

        // 2 heads, 3 positions, head_dim=4
        let flat: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();

        let rotated = rope.apply_flat(&flat, 2, 3, 0);

        assert_eq!(rotated.len(), 24);
    }

    #[test]
    fn test_scaling_factor() {
        let config_no_scale = RopeConfig::new(8).max_seq_len(100).scaling_factor(1.0);
        let config_scaled = RopeConfig::new(8).max_seq_len(100).scaling_factor(2.0);

        let rope_no_scale = RopeFrequencies::new(config_no_scale);
        let rope_scaled = RopeFrequencies::new(config_scaled);

        // Scaled position 20 should equal unscaled position 10
        let cos_unscaled_10 = rope_no_scale.cos(10);
        let cos_scaled_20 = rope_scaled.cos(20);

        for (a, b) in cos_unscaled_10.iter().zip(cos_scaled_20.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_angle() {
        // At dim 0, inv_freq = 1, so angle = position
        let angle_0 = rope_angle(5, 0, 8, 10000.0);
        assert!((angle_0 - 5.0).abs() < 1e-5);

        // At higher dims, frequency decreases
        let angle_1 = rope_angle(5, 1, 8, 10000.0);
        assert!(angle_1 < angle_0);
    }

    #[test]
    fn test_rotate_pair() {
        // 90 degree rotation
        let (x, y) = rotate_pair(1.0, 0.0, 0.0, 1.0); // cos=0, sin=1
        assert!((x - 0.0).abs() < 1e-5);
        assert!((y - 1.0).abs() < 1e-5);

        // 180 degree rotation
        let (x, y) = rotate_pair(1.0, 0.0, -1.0, 0.0); // cos=-1, sin=0
        assert!((x - (-1.0)).abs() < 1e-5);
        assert!(y.abs() < 1e-5);
    }

    #[test]
    fn test_ntk_scaled_base() {
        let new_base = ntk_scaled_base(10000.0, 4096, 8192);

        // Should be larger than original for extended context
        assert!(new_base > 10000.0);
    }

    #[test]
    fn test_yarn_config() {
        let yarn = YarnConfig::new(4096, 32768);

        assert_eq!(yarn.original_max_len, 4096);
        assert_eq!(yarn.extended_max_len, 32768);
        assert!(yarn.attention_factor > 1.0);
    }

    #[test]
    fn test_yarn_interpolation_factor() {
        let yarn = YarnConfig::new(4096, 8192);
        let scale = 8192.0 / 4096.0; // 2.0

        // Test that interpolation factors are in valid range [1.0, scale]
        for dim in 0..64 {
            let factor = yarn.interpolation_factor(dim, 64);
            assert!(factor >= 1.0 - 1e-5, "dim {} factor {} < 1.0", dim, factor);
            assert!(
                factor <= scale + 1e-5,
                "dim {} factor {} > scale {}",
                dim,
                factor,
                scale
            );
        }

        // Lower dimensions (higher frequencies) should have factors closer to 1.0
        // Higher dimensions (lower frequencies) should have factors closer to scale
        let factor_low = yarn.interpolation_factor(0, 64);
        let factor_high = yarn.interpolation_factor(31, 64);
        // Both should be valid
        assert!(factor_low >= 1.0 - 1e-5);
        assert!(factor_high >= 1.0 - 1e-5);
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn test_odd_head_dim_panics() {
        let _ = RopeConfig::new(7);
    }

    #[test]
    fn test_llama_style_config() {
        // LLaMA 7B/13B: head_dim=128, base=10000
        let config = RopeConfig::new(128).base(10000.0).max_seq_len(4096);
        let rope = RopeFrequencies::new(config);

        let x = random_vector(128, 42);
        let rotated = rope.apply(&x, 100);

        // Should preserve norm
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_rot: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_x - norm_rot).abs() < 1e-4);
    }

    #[test]
    fn test_consistency_across_applications() {
        let config = RopeConfig::new(8).max_seq_len(100);
        let rope = RopeFrequencies::new(config);

        let x = random_vector(8, 42);

        // Multiple applications at same position should give same result
        let rot1 = rope.apply(&x, 50);
        let rot2 = rope.apply(&x, 50);

        for (a, b) in rot1.iter().zip(rot2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_incremental_decoding() {
        let config = RopeConfig::new(8).max_seq_len(100);
        let rope = RopeFrequencies::new(config);

        // Simulate incremental decoding
        // First, full sequence encoding
        let seq = vec![
            random_vector(8, 0),
            random_vector(8, 1),
            random_vector(8, 2),
        ];
        let full_encoded = rope.apply_batch(&seq, 0);

        // Then, incremental: encode one token at a time
        let inc_0 = rope.apply(&seq[0], 0);
        let inc_1 = rope.apply(&seq[1], 1);
        let inc_2 = rope.apply(&seq[2], 2);

        // Should match
        for (a, b) in full_encoded[0].iter().zip(inc_0.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for (a, b) in full_encoded[1].iter().zip(inc_1.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for (a, b) in full_encoded[2].iter().zip(inc_2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
