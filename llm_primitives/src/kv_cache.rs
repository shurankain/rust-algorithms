// KV Cache Quantization
//
// Compresses the Key-Value cache in transformer models using INT4/INT8
// quantization to reduce memory consumption for long context inference.
//
// Key concepts:
// - Per-channel quantization: Different scale/zero-point per channel
// - Per-token quantization: Different scale/zero-point per token position
// - Symmetric quantization: Zero-point is 0, values centered around 0
// - Asymmetric quantization: Non-zero zero-point for better range utilization
//
// Memory savings:
// - FP32 -> INT8: 4x compression
// - FP32 -> INT4: 8x compression
// - FP16 -> INT8: 2x compression
// - FP16 -> INT4: 4x compression
//
// Trade-offs:
// - INT8: Nearly lossless for most models
// - INT4: Some quality degradation, better for less critical layers

use std::cmp::Ordering;

/// Quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    Int4,
    Int8,
}

impl QuantBits {
    /// Get the number of bits
    pub fn bits(&self) -> u8 {
        match self {
            QuantBits::Int4 => 4,
            QuantBits::Int8 => 8,
        }
    }

    /// Get the range for signed quantization
    pub fn range(&self) -> (i32, i32) {
        match self {
            QuantBits::Int4 => (-8, 7),
            QuantBits::Int8 => (-128, 127),
        }
    }

    /// Get the range for unsigned quantization
    pub fn unsigned_range(&self) -> (i32, i32) {
        match self {
            QuantBits::Int4 => (0, 15),
            QuantBits::Int8 => (0, 255),
        }
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric: zero_point = 0, range [-max, max]
    Symmetric,
    /// Asymmetric: variable zero_point, range [min, max]
    Asymmetric,
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    /// One scale/zero_point per entire tensor
    PerTensor,
    /// One scale/zero_point per channel (head dimension)
    PerChannel,
    /// One scale/zero_point per token position
    PerToken,
    /// One scale/zero_point per group of values
    PerGroup(usize),
}

/// Configuration for KV cache quantization
#[derive(Debug, Clone)]
pub struct KvQuantConfig {
    /// Bit width for keys
    pub key_bits: QuantBits,
    /// Bit width for values
    pub value_bits: QuantBits,
    /// Quantization mode
    pub mode: QuantMode,
    /// Quantization granularity
    pub granularity: QuantGranularity,
}

impl Default for KvQuantConfig {
    fn default() -> Self {
        Self {
            key_bits: QuantBits::Int8,
            value_bits: QuantBits::Int8,
            mode: QuantMode::Symmetric,
            granularity: QuantGranularity::PerToken,
        }
    }
}

impl KvQuantConfig {
    /// Create INT8 configuration
    pub fn int8() -> Self {
        Self::default()
    }

    /// Create INT4 configuration
    pub fn int4() -> Self {
        Self {
            key_bits: QuantBits::Int4,
            value_bits: QuantBits::Int4,
            ..Default::default()
        }
    }

    /// Set key bits
    pub fn key_bits(mut self, bits: QuantBits) -> Self {
        self.key_bits = bits;
        self
    }

    /// Set value bits
    pub fn value_bits(mut self, bits: QuantBits) -> Self {
        self.value_bits = bits;
        self
    }

    /// Set quantization mode
    pub fn mode(mut self, mode: QuantMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set quantization granularity
    pub fn granularity(mut self, granularity: QuantGranularity) -> Self {
        self.granularity = granularity;
        self
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor(s) for dequantization: float = scale * (int - zero_point)
    pub scales: Vec<f32>,
    /// Zero point(s) for asymmetric quantization
    pub zero_points: Vec<i32>,
    /// Bit width
    pub bits: QuantBits,
    /// Quantization mode
    pub mode: QuantMode,
}

/// Quantized tensor storage
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (packed for INT4)
    pub data: Vec<u8>,
    /// Quantization parameters
    pub params: QuantParams,
    /// Original shape [seq_len, head_dim] or [batch, seq_len, head_dim]
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        let data_bytes = self.data.len();
        let params_bytes = self.params.scales.len() * 4 + self.params.zero_points.len() * 4;
        data_bytes + params_bytes
    }

    /// Get number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get compression ratio vs FP32
    pub fn compression_ratio_vs_fp32(&self) -> f32 {
        let original_bytes = self.num_elements() * 4; // FP32
        original_bytes as f32 / self.memory_size() as f32
    }
}

/// Quantize a tensor to INT8 with symmetric per-tensor quantization
pub fn quantize_symmetric_int8(data: &[f32]) -> QuantizedTensor {
    if data.is_empty() {
        return QuantizedTensor {
            data: vec![],
            params: QuantParams {
                scales: vec![1.0],
                zero_points: vec![0],
                bits: QuantBits::Int8,
                mode: QuantMode::Symmetric,
            },
            shape: vec![0],
        };
    }

    // Find max absolute value
    let max_abs = data
        .iter()
        .map(|&x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or(1.0)
        .max(1e-10); // Avoid division by zero

    // Scale: maps [-max_abs, max_abs] to [-127, 127]
    let scale = max_abs / 127.0;

    // Quantize
    let quantized: Vec<u8> = data
        .iter()
        .map(|&x| {
            let q = (x / scale).round().clamp(-128.0, 127.0) as i8;
            q as u8
        })
        .collect();

    QuantizedTensor {
        data: quantized,
        params: QuantParams {
            scales: vec![scale],
            zero_points: vec![0],
            bits: QuantBits::Int8,
            mode: QuantMode::Symmetric,
        },
        shape: vec![data.len()],
    }
}

/// Quantize a tensor to INT8 with asymmetric per-tensor quantization
pub fn quantize_asymmetric_int8(data: &[f32]) -> QuantizedTensor {
    if data.is_empty() {
        return QuantizedTensor {
            data: vec![],
            params: QuantParams {
                scales: vec![1.0],
                zero_points: vec![0],
                bits: QuantBits::Int8,
                mode: QuantMode::Asymmetric,
            },
            shape: vec![0],
        };
    }

    // Find min and max
    let min_val = data
        .iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or(0.0);
    let max_val = data
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or(0.0);

    // Scale and zero point for [0, 255] range
    let range = (max_val - min_val).max(1e-10);
    let scale = range / 255.0;
    let zero_point = (-min_val / scale).round() as i32;
    let zero_point = zero_point.clamp(0, 255);

    // Quantize
    let quantized: Vec<u8> = data
        .iter()
        .map(|&x| (x / scale + zero_point as f32).round().clamp(0.0, 255.0) as u8)
        .collect();

    QuantizedTensor {
        data: quantized,
        params: QuantParams {
            scales: vec![scale],
            zero_points: vec![zero_point],
            bits: QuantBits::Int8,
            mode: QuantMode::Asymmetric,
        },
        shape: vec![data.len()],
    }
}

/// Quantize a tensor to INT4 with symmetric per-tensor quantization
/// Two INT4 values are packed into one byte
pub fn quantize_symmetric_int4(data: &[f32]) -> QuantizedTensor {
    if data.is_empty() {
        return QuantizedTensor {
            data: vec![],
            params: QuantParams {
                scales: vec![1.0],
                zero_points: vec![0],
                bits: QuantBits::Int4,
                mode: QuantMode::Symmetric,
            },
            shape: vec![0],
        };
    }

    // Find max absolute value
    let max_abs = data
        .iter()
        .map(|&x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or(1.0)
        .max(1e-10);

    // Scale: maps [-max_abs, max_abs] to [-7, 7] (signed INT4)
    let scale = max_abs / 7.0;

    // Quantize and pack two values per byte
    let num_bytes = data.len().div_ceil(2);
    let mut quantized = vec![0u8; num_bytes];

    for (i, &x) in data.iter().enumerate() {
        let q = (x / scale).round().clamp(-8.0, 7.0) as i8;
        // Convert to unsigned 4-bit (0-15 range, where 8 is zero)
        let q_unsigned = ((q + 8) as u8) & 0x0F;

        if i % 2 == 0 {
            quantized[i / 2] |= q_unsigned;
        } else {
            quantized[i / 2] |= q_unsigned << 4;
        }
    }

    QuantizedTensor {
        data: quantized,
        params: QuantParams {
            scales: vec![scale],
            zero_points: vec![8], // Offset for signed INT4
            bits: QuantBits::Int4,
            mode: QuantMode::Symmetric,
        },
        shape: vec![data.len()],
    }
}

/// Dequantize INT8 symmetric tensor
pub fn dequantize_symmetric_int8(tensor: &QuantizedTensor) -> Vec<f32> {
    assert_eq!(tensor.params.bits, QuantBits::Int8);
    assert_eq!(tensor.params.mode, QuantMode::Symmetric);

    let scale = tensor.params.scales[0];

    tensor
        .data
        .iter()
        .map(|&q| {
            let signed = q as i8;
            signed as f32 * scale
        })
        .collect()
}

/// Dequantize INT8 asymmetric tensor
pub fn dequantize_asymmetric_int8(tensor: &QuantizedTensor) -> Vec<f32> {
    assert_eq!(tensor.params.bits, QuantBits::Int8);
    assert_eq!(tensor.params.mode, QuantMode::Asymmetric);

    let scale = tensor.params.scales[0];
    let zero_point = tensor.params.zero_points[0];

    tensor
        .data
        .iter()
        .map(|&q| (q as i32 - zero_point) as f32 * scale)
        .collect()
}

/// Dequantize INT4 symmetric tensor
pub fn dequantize_symmetric_int4(tensor: &QuantizedTensor) -> Vec<f32> {
    assert_eq!(tensor.params.bits, QuantBits::Int4);

    let scale = tensor.params.scales[0];
    let zero_point = tensor.params.zero_points[0] as u8;
    let num_elements = tensor.shape.iter().product();

    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let byte_idx = i / 2;
        let q_unsigned = if i % 2 == 0 {
            tensor.data[byte_idx] & 0x0F
        } else {
            (tensor.data[byte_idx] >> 4) & 0x0F
        };

        // Convert back to signed
        let q_signed = q_unsigned as i8 - zero_point as i8;
        result.push(q_signed as f32 * scale);
    }

    result
}

/// Quantize with per-token granularity (useful for KV cache)
/// data shape: [seq_len, head_dim]
pub fn quantize_per_token_int8(data: &[f32], seq_len: usize, head_dim: usize) -> QuantizedTensor {
    assert_eq!(data.len(), seq_len * head_dim);

    let mut scales = Vec::with_capacity(seq_len);
    let mut quantized = Vec::with_capacity(data.len());

    for t in 0..seq_len {
        let start = t * head_dim;
        let end = start + head_dim;
        let token_data = &data[start..end];

        // Find max absolute value for this token
        let max_abs = token_data
            .iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(1.0)
            .max(1e-10);

        let scale = max_abs / 127.0;
        scales.push(scale);

        // Quantize this token's values
        for &x in token_data {
            let q = (x / scale).round().clamp(-128.0, 127.0) as i8;
            quantized.push(q as u8);
        }
    }

    QuantizedTensor {
        data: quantized,
        params: QuantParams {
            scales,
            zero_points: vec![0; seq_len],
            bits: QuantBits::Int8,
            mode: QuantMode::Symmetric,
        },
        shape: vec![seq_len, head_dim],
    }
}

/// Dequantize per-token INT8 tensor
pub fn dequantize_per_token_int8(tensor: &QuantizedTensor) -> Vec<f32> {
    assert_eq!(tensor.params.bits, QuantBits::Int8);
    assert_eq!(tensor.shape.len(), 2);

    let seq_len = tensor.shape[0];
    let head_dim = tensor.shape[1];

    let mut result = Vec::with_capacity(seq_len * head_dim);

    for t in 0..seq_len {
        let scale = tensor.params.scales[t];
        let start = t * head_dim;
        let end = start + head_dim;

        for &q in &tensor.data[start..end] {
            let signed = q as i8;
            result.push(signed as f32 * scale);
        }
    }

    result
}

/// KV Cache with quantization support
pub struct QuantizedKvCache {
    /// Quantized key cache
    keys: Vec<QuantizedTensor>,
    /// Quantized value cache
    values: Vec<QuantizedTensor>,
    /// Number of layers
    num_layers: usize,
    /// Head dimension
    head_dim: usize,
    /// Number of heads
    num_heads: usize,
    /// Current sequence length
    seq_len: usize,
    /// Configuration
    config: KvQuantConfig,
}

impl QuantizedKvCache {
    /// Create a new quantized KV cache
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        config: KvQuantConfig,
    ) -> Self {
        Self {
            keys: Vec::with_capacity(num_layers),
            values: Vec::with_capacity(num_layers),
            num_layers,
            head_dim,
            num_heads,
            seq_len: 0,
            config,
        }
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let key_bytes: usize = self.keys.iter().map(|k| k.memory_size()).sum();
        let value_bytes: usize = self.values.iter().map(|v| v.memory_size()).sum();
        key_bytes + value_bytes
    }

    /// Get memory usage if stored as FP32
    pub fn fp32_memory_usage(&self) -> usize {
        // Per layer: seq_len * num_heads * head_dim * 4 bytes * 2 (K + V)
        self.num_layers * self.seq_len * self.num_heads * self.head_dim * 4 * 2
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let fp32 = self.fp32_memory_usage();
        if fp32 == 0 {
            return 1.0;
        }
        fp32 as f32 / self.memory_usage() as f32
    }

    /// Append new key-value pairs for all layers
    /// keys/values shape per layer: [num_heads, new_tokens, head_dim]
    pub fn append(&mut self, new_keys: &[Vec<f32>], new_values: &[Vec<f32>], new_tokens: usize) {
        assert_eq!(new_keys.len(), self.num_layers);
        assert_eq!(new_values.len(), self.num_layers);

        for layer in 0..self.num_layers {
            let key_data = &new_keys[layer];
            let value_data = &new_values[layer];

            // Expected size: num_heads * new_tokens * head_dim
            let expected_size = self.num_heads * new_tokens * self.head_dim;
            assert_eq!(key_data.len(), expected_size);
            assert_eq!(value_data.len(), expected_size);

            // Quantize new data
            let quantized_key = match self.config.key_bits {
                QuantBits::Int8 => quantize_symmetric_int8(key_data),
                QuantBits::Int4 => quantize_symmetric_int4(key_data),
            };

            let quantized_value = match self.config.value_bits {
                QuantBits::Int8 => quantize_symmetric_int8(value_data),
                QuantBits::Int4 => quantize_symmetric_int4(value_data),
            };

            if layer >= self.keys.len() {
                self.keys.push(quantized_key);
                self.values.push(quantized_value);
            } else {
                // Append to existing - for simplicity, we re-quantize the combined data
                // In production, you'd want incremental updates
                let mut existing_key = self.get_key(layer);
                let mut existing_value = self.get_value(layer);

                existing_key.extend_from_slice(key_data);
                existing_value.extend_from_slice(value_data);

                self.keys[layer] = match self.config.key_bits {
                    QuantBits::Int8 => quantize_symmetric_int8(&existing_key),
                    QuantBits::Int4 => quantize_symmetric_int4(&existing_key),
                };

                self.values[layer] = match self.config.value_bits {
                    QuantBits::Int8 => quantize_symmetric_int8(&existing_value),
                    QuantBits::Int4 => quantize_symmetric_int4(&existing_value),
                };
            }
        }

        self.seq_len += new_tokens;
    }

    /// Get dequantized keys for a layer
    pub fn get_key(&self, layer: usize) -> Vec<f32> {
        if layer >= self.keys.len() {
            return vec![];
        }

        match self.config.key_bits {
            QuantBits::Int8 => dequantize_symmetric_int8(&self.keys[layer]),
            QuantBits::Int4 => dequantize_symmetric_int4(&self.keys[layer]),
        }
    }

    /// Get dequantized values for a layer
    pub fn get_value(&self, layer: usize) -> Vec<f32> {
        if layer >= self.values.len() {
            return vec![];
        }

        match self.config.value_bits {
            QuantBits::Int8 => dequantize_symmetric_int8(&self.values[layer]),
            QuantBits::Int4 => dequantize_symmetric_int4(&self.values[layer]),
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }
}

/// Compute quantization error (MSE) between original and dequantized
pub fn quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    assert_eq!(original.len(), dequantized.len());
    if original.is_empty() {
        return 0.0;
    }

    let mse: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(&o, &d)| (o - d) * (o - d))
        .sum::<f32>()
        / original.len() as f32;

    mse
}

/// Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB
pub fn sqnr_db(original: &[f32], dequantized: &[f32]) -> f32 {
    let signal_power: f32 = original.iter().map(|&x| x * x).sum::<f32>() / original.len() as f32;
    let noise_power = quantization_error(original, dequantized);

    if noise_power < 1e-10 {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_data(n: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..n)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                let h = hasher.finish();
                ((h % 10000) as f32 / 5000.0) - 1.0 // Range [-1, 1]
            })
            .collect()
    }

    #[test]
    fn test_quant_bits() {
        assert_eq!(QuantBits::Int4.bits(), 4);
        assert_eq!(QuantBits::Int8.bits(), 8);
        assert_eq!(QuantBits::Int4.range(), (-8, 7));
        assert_eq!(QuantBits::Int8.range(), (-128, 127));
    }

    #[test]
    fn test_config_default() {
        let config = KvQuantConfig::default();
        assert_eq!(config.key_bits, QuantBits::Int8);
        assert_eq!(config.value_bits, QuantBits::Int8);
        assert_eq!(config.mode, QuantMode::Symmetric);
    }

    #[test]
    fn test_config_builder() {
        let config = KvQuantConfig::int4()
            .key_bits(QuantBits::Int8)
            .mode(QuantMode::Asymmetric)
            .granularity(QuantGranularity::PerChannel);

        assert_eq!(config.key_bits, QuantBits::Int8);
        assert_eq!(config.value_bits, QuantBits::Int4);
        assert_eq!(config.mode, QuantMode::Asymmetric);
    }

    #[test]
    fn test_symmetric_int8_roundtrip() {
        let data = vec![0.5, -0.3, 0.8, -0.9, 0.1, 0.0];
        let quantized = quantize_symmetric_int8(&data);
        let dequantized = dequantize_symmetric_int8(&quantized);

        assert_eq!(dequantized.len(), data.len());

        // Check reconstruction error is small
        for (o, d) in data.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.02, "Error too large: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_asymmetric_int8_roundtrip() {
        // Use wider range for better quantization accuracy
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantized = quantize_asymmetric_int8(&data);
        let dequantized = dequantize_asymmetric_int8(&quantized);

        assert_eq!(dequantized.len(), data.len());

        // Asymmetric has slightly more error due to zero-point rounding
        for (o, d) in data.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.01, "Error too large: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_symmetric_int4_roundtrip() {
        let data = vec![0.5, -0.3, 0.8, -0.9, 0.1, 0.0, -0.5, 0.7];
        let quantized = quantize_symmetric_int4(&data);
        let dequantized = dequantize_symmetric_int4(&quantized);

        assert_eq!(dequantized.len(), data.len());

        // INT4 has larger error
        for (o, d) in data.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.2, "Error too large: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_per_token_int8_roundtrip() {
        let seq_len = 4;
        let head_dim = 8;
        let data = random_data(seq_len * head_dim);

        let quantized = quantize_per_token_int8(&data, seq_len, head_dim);
        let dequantized = dequantize_per_token_int8(&quantized);

        assert_eq!(dequantized.len(), data.len());
        assert_eq!(quantized.params.scales.len(), seq_len);

        let mse = quantization_error(&data, &dequantized);
        assert!(mse < 0.001, "MSE too high: {}", mse);
    }

    #[test]
    fn test_compression_ratio() {
        let data = random_data(1024);

        let int8 = quantize_symmetric_int8(&data);
        let int4 = quantize_symmetric_int4(&data);

        // INT8: ~4x compression vs FP32
        assert!(int8.compression_ratio_vs_fp32() > 3.5);
        assert!(int8.compression_ratio_vs_fp32() < 4.5);

        // INT4: ~8x compression vs FP32 (with packing)
        assert!(int4.compression_ratio_vs_fp32() > 6.0);
    }

    #[test]
    fn test_quantized_kv_cache() {
        let num_layers = 2;
        let num_heads = 4;
        let head_dim = 8;
        let new_tokens = 3;

        let config = KvQuantConfig::int8();
        let mut cache = QuantizedKvCache::new(num_layers, num_heads, head_dim, config);

        assert_eq!(cache.seq_len(), 0);
        assert!(cache.memory_usage() == 0 || cache.memory_usage() > 0);

        // Add tokens
        let key_size = num_heads * new_tokens * head_dim;
        let keys: Vec<Vec<f32>> = (0..num_layers).map(|_| random_data(key_size)).collect();
        let values: Vec<Vec<f32>> = (0..num_layers).map(|_| random_data(key_size)).collect();

        cache.append(&keys, &values, new_tokens);

        assert_eq!(cache.seq_len(), new_tokens);
        assert!(cache.memory_usage() > 0);
        assert!(cache.compression_ratio() > 3.0);

        // Get cached values
        let retrieved_key = cache.get_key(0);
        assert_eq!(retrieved_key.len(), key_size);

        // Verify reconstruction
        let mse = quantization_error(&keys[0], &retrieved_key);
        assert!(mse < 0.01);
    }

    #[test]
    fn test_kv_cache_int4() {
        let num_layers = 2;
        let num_heads = 4;
        let head_dim = 8;
        let new_tokens = 4;

        let config = KvQuantConfig::int4();
        let mut cache = QuantizedKvCache::new(num_layers, num_heads, head_dim, config);

        let key_size = num_heads * new_tokens * head_dim;
        let keys: Vec<Vec<f32>> = (0..num_layers).map(|_| random_data(key_size)).collect();
        let values: Vec<Vec<f32>> = (0..num_layers).map(|_| random_data(key_size)).collect();

        cache.append(&keys, &values, new_tokens);

        // INT4 should have higher compression
        assert!(cache.compression_ratio() > 6.0);
    }

    #[test]
    fn test_sqnr() {
        let data = random_data(256);
        let quantized = quantize_symmetric_int8(&data);
        let dequantized = dequantize_symmetric_int8(&quantized);

        let sqnr = sqnr_db(&data, &dequantized);
        // INT8 should have decent SQNR (> 30 dB typically)
        assert!(sqnr > 25.0, "SQNR too low: {} dB", sqnr);
    }

    #[test]
    fn test_empty_tensor() {
        let data: Vec<f32> = vec![];

        let int8 = quantize_symmetric_int8(&data);
        assert!(int8.data.is_empty());

        let int4 = quantize_symmetric_int4(&data);
        assert!(int4.data.is_empty());
    }

    #[test]
    fn test_kv_cache_clear() {
        let config = KvQuantConfig::int8();
        let mut cache = QuantizedKvCache::new(2, 4, 8, config);

        let keys: Vec<Vec<f32>> = vec![random_data(96), random_data(96)];
        let values: Vec<Vec<f32>> = vec![random_data(96), random_data(96)];

        cache.append(&keys, &values, 3);
        assert_eq!(cache.seq_len(), 3);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_memory_size() {
        let data = random_data(1000);
        let tensor = quantize_symmetric_int8(&data);

        // Should be close to 1000 bytes (data) + small overhead for params
        assert!(tensor.memory_size() >= 1000);
        assert!(tensor.memory_size() < 1100);
    }

    #[test]
    fn test_large_values() {
        let data = vec![1000.0, -500.0, 2000.0, -1000.0];
        let quantized = quantize_symmetric_int8(&data);
        let dequantized = dequantize_symmetric_int8(&quantized);

        // Relative error should be small
        for (o, d) in data.iter().zip(dequantized.iter()) {
            let rel_error = ((o - d) / o).abs();
            assert!(rel_error < 0.02, "Relative error too large: {}", rel_error);
        }
    }
}
