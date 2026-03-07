// HyperLogLog - Cardinality estimation for large datasets
//
// HyperLogLog (HLL) is a probabilistic algorithm for estimating the number of
// distinct elements (cardinality) in a multiset using very little memory.
// It can count billions of distinct items using only a few kilobytes.
//
// How it works:
// 1. Hash each element to get a uniform random bit string
// 2. Use first p bits to select one of 2^p registers (buckets)
// 3. Count leading zeros in remaining bits, store max seen per register
// 4. Estimate cardinality from harmonic mean of 2^(register values)
//
// Key insight: If we see a hash with k leading zeros, we've likely seen ~2^k elements
// By using multiple registers, we reduce variance through averaging.
//
// Visualization (p=4, so 16 registers):
//
//   Element "apple" → hash: 0110|00001011...
//                          ^^^^  ^^^^^
//                          register=6  leading_zeros=4
//
//   Registers: [1, 3, 2, 0, 1, 2, 4, 1, 0, 2, 3, 1, 2, 1, 0, 1]
//                                  ^
//                                  register 6 now stores max(old, 4)
//
// Estimation formula:
//   E = α_m * m^2 * (sum of 2^(-register[i]))^(-1)
//
// where α_m is a bias correction constant depending on m = 2^p
//
// Error rate: ~1.04 / sqrt(m) standard error
// With m=2048 registers (11 bits): ~2.3% error
// With m=16384 registers (14 bits): ~0.8% error
//
// Memory: O(m) = O(2^p) bytes (typically 1-16 KB)
// Time: O(1) per insert, O(m) for cardinality estimate
//
// Applications:
// - Counting unique visitors to a website
// - Database query optimization (distinct count estimation)
// - Network traffic analysis (unique IPs)
// - Big data analytics (Spark, Redis use HLL)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

// HyperLogLog implementation
pub struct HyperLogLog<T> {
    registers: Vec<u8>,
    precision: usize,     // p: number of bits for register selection
    num_registers: usize, // m = 2^p
    _marker: PhantomData<T>,
}

impl<T: Hash> HyperLogLog<T> {
    // Create with specified precision (p)
    // Higher precision = more accuracy but more memory
    // Typical values: 4-18, recommended: 10-14
    pub fn new(precision: usize) -> Self {
        assert!(
            (4..=18).contains(&precision),
            "Precision must be between 4 and 18"
        );

        let num_registers = 1 << precision;

        Self {
            registers: vec![0; num_registers],
            precision,
            num_registers,
            _marker: PhantomData,
        }
    }

    // Create with target error rate
    // error ≈ 1.04 / sqrt(m), so m ≈ (1.04 / error)^2
    pub fn with_error_rate(error_rate: f64) -> Self {
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );

        // m = (1.04 / error)^2
        let m = ((1.04 / error_rate).powi(2)).ceil() as usize;

        // Find precision p such that 2^p >= m
        let precision = (m as f64).log2().ceil() as usize;
        let precision = precision.clamp(4, 18);

        Self::new(precision)
    }

    // Add an element
    pub fn insert(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        // Use first p bits for register index
        let register_idx = (hash >> (64 - self.precision)) as usize;

        // Count leading zeros in remaining bits (after the p bits used for index)
        let remaining_bits = if self.precision < 64 {
            hash << self.precision
        } else {
            0
        };

        // Add 1 because we're counting position of first 1, not number of zeros
        let leading_zeros = remaining_bits.leading_zeros() as u8 + 1;

        // Store maximum
        if leading_zeros > self.registers[register_idx] {
            self.registers[register_idx] = leading_zeros;
        }
    }

    // Estimate cardinality (number of distinct elements)
    pub fn cardinality(&self) -> f64 {
        // Compute harmonic mean
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powf(-(r as f64)))
            .sum();

        // Raw estimate: E = α_m * m^2 / sum
        let m = self.num_registers as f64;
        let alpha_m = self.alpha();
        let raw_estimate = alpha_m * m * m / sum;

        // Apply corrections for small and large cardinalities
        self.apply_corrections(raw_estimate)
    }

    // Bias correction constant
    fn alpha(&self) -> f64 {
        match self.num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / self.num_registers as f64),
        }
    }

    // Apply small/large range corrections
    fn apply_corrections(&self, raw_estimate: f64) -> f64 {
        let m = self.num_registers as f64;

        // Small range correction (linear counting)
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                // Linear counting
                return m * (m / zeros as f64).ln();
            }
        }

        // Large range correction (for hash collisions)
        let two_32 = 2.0_f64.powi(32);
        if raw_estimate > two_32 / 30.0 {
            return -two_32 * (1.0 - raw_estimate / two_32).ln();
        }

        raw_estimate
    }

    // Merge two HyperLogLog structures (must have same precision)
    pub fn merge(&self, other: &Self) -> Option<Self> {
        if self.precision != other.precision {
            return None;
        }

        let mut result = Self::new(self.precision);
        for i in 0..self.num_registers {
            result.registers[i] = self.registers[i].max(other.registers[i]);
        }
        Some(result)
    }

    // Merge in place
    pub fn merge_in_place(&mut self, other: &Self) -> bool {
        if self.precision != other.precision {
            return false;
        }

        for i in 0..self.num_registers {
            self.registers[i] = self.registers[i].max(other.registers[i]);
        }
        true
    }

    // Get precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    // Get number of registers
    pub fn num_registers(&self) -> usize {
        self.num_registers
    }

    // Get theoretical standard error
    pub fn standard_error(&self) -> f64 {
        1.04 / (self.num_registers as f64).sqrt()
    }

    // Clear all registers
    pub fn clear(&mut self) {
        self.registers.fill(0);
    }

    // Check if empty
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&r| r == 0)
    }

    // Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.registers.len()
    }
}

// HyperLogLog++ (improved version with better small cardinality estimation)
pub struct HyperLogLogPlus<T> {
    registers: Vec<u8>,
    precision: usize,
    num_registers: usize,
    // Sparse representation for small cardinalities
    sparse_set: Option<Vec<u64>>,
    sparse_threshold: usize,
    _marker: PhantomData<T>,
}

impl<T: Hash> HyperLogLogPlus<T> {
    pub fn new(precision: usize) -> Self {
        assert!(
            (4..=18).contains(&precision),
            "Precision must be between 4 and 18"
        );

        let num_registers = 1 << precision;
        // Use sparse representation while cardinality is low
        let sparse_threshold = num_registers * 6;

        Self {
            registers: vec![0; num_registers],
            precision,
            num_registers,
            sparse_set: Some(Vec::new()),
            sparse_threshold,
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        // If still in sparse mode
        if let Some(ref mut sparse) = self.sparse_set {
            if !sparse.contains(&hash) {
                sparse.push(hash);
            }

            // Switch to dense if sparse gets too large
            if sparse.len() >= self.sparse_threshold {
                self.convert_to_dense();
            }
        } else {
            // Dense mode
            self.insert_dense(hash);
        }
    }

    fn insert_dense(&mut self, hash: u64) {
        let register_idx = (hash >> (64 - self.precision)) as usize;
        let remaining_bits = if self.precision < 64 {
            hash << self.precision
        } else {
            0
        };
        let leading_zeros = remaining_bits.leading_zeros() as u8 + 1;

        if leading_zeros > self.registers[register_idx] {
            self.registers[register_idx] = leading_zeros;
        }
    }

    fn convert_to_dense(&mut self) {
        if let Some(sparse) = self.sparse_set.take() {
            for hash in sparse {
                self.insert_dense(hash);
            }
        }
    }

    pub fn cardinality(&self) -> f64 {
        // If in sparse mode, return exact count
        if let Some(ref sparse) = self.sparse_set {
            return sparse.len() as f64;
        }

        // Dense mode: use HLL estimation
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powf(-(r as f64)))
            .sum();

        let m = self.num_registers as f64;
        let alpha_m = match self.num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / self.num_registers as f64),
        };

        let raw_estimate = alpha_m * m * m / sum;

        // Apply bias correction using empirical data
        self.apply_bias_correction(raw_estimate)
    }

    fn apply_bias_correction(&self, raw_estimate: f64) -> f64 {
        let m = self.num_registers as f64;

        // Small range correction
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }

        // Large range correction
        let two_32 = 2.0_f64.powi(32);
        if raw_estimate > two_32 / 30.0 {
            return -two_32 * (1.0 - raw_estimate / two_32).ln();
        }

        raw_estimate
    }

    pub fn merge(&self, other: &Self) -> Option<Self> {
        if self.precision != other.precision {
            return None;
        }

        let mut result = Self::new(self.precision);
        result.sparse_set = None; // Convert to dense for merge

        // If both are sparse, merge sparse sets
        if let (Some(s1), Some(s2)) = (&self.sparse_set, &other.sparse_set) {
            let mut merged: Vec<u64> = s1.clone();
            for &h in s2 {
                if !merged.contains(&h) {
                    merged.push(h);
                }
            }

            if merged.len() < result.sparse_threshold {
                result.sparse_set = Some(merged);
                return Some(result);
            }

            // Convert to dense
            for hash in merged {
                result.insert_dense(hash);
            }
            return Some(result);
        }

        // At least one is dense, merge registers
        // First, convert sparse to dense if needed
        let mut self_registers = self.registers.clone();
        let mut other_registers = other.registers.clone();

        if let Some(ref sparse) = self.sparse_set {
            for &hash in sparse {
                let register_idx = (hash >> (64 - self.precision)) as usize;
                let remaining_bits = hash << self.precision;
                let leading_zeros = remaining_bits.leading_zeros() as u8 + 1;
                self_registers[register_idx] = self_registers[register_idx].max(leading_zeros);
            }
        }

        if let Some(ref sparse) = other.sparse_set {
            for &hash in sparse {
                let register_idx = (hash >> (64 - other.precision)) as usize;
                let remaining_bits = hash << other.precision;
                let leading_zeros = remaining_bits.leading_zeros() as u8 + 1;
                other_registers[register_idx] = other_registers[register_idx].max(leading_zeros);
            }
        }

        for i in 0..result.num_registers {
            result.registers[i] = self_registers[i].max(other_registers[i]);
        }

        Some(result)
    }

    pub fn precision(&self) -> usize {
        self.precision
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse_set.is_some()
    }

    pub fn clear(&mut self) {
        self.registers.fill(0);
        self.sparse_set = Some(Vec::new());
    }
}

// Estimate intersection cardinality using inclusion-exclusion
pub fn estimate_intersection<T: Hash>(a: &HyperLogLog<T>, b: &HyperLogLog<T>) -> Option<f64> {
    if a.precision != b.precision {
        return None;
    }

    let merged = a.merge(b)?;
    let union_card = merged.cardinality();
    let a_card = a.cardinality();
    let b_card = b.cardinality();

    // |A ∩ B| = |A| + |B| - |A ∪ B|
    let intersection = a_card + b_card - union_card;

    Some(intersection.max(0.0))
}

// Estimate Jaccard similarity
pub fn jaccard_similarity<T: Hash>(a: &HyperLogLog<T>, b: &HyperLogLog<T>) -> Option<f64> {
    if a.precision != b.precision {
        return None;
    }

    let merged = a.merge(b)?;
    let union_card = merged.cardinality();

    if union_card == 0.0 {
        return Some(1.0); // Both empty
    }

    let intersection = estimate_intersection(a, b)?;
    Some(intersection / union_card)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hll_basic() {
        let mut hll: HyperLogLog<&str> = HyperLogLog::new(10);

        hll.insert(&"hello");
        hll.insert(&"world");
        hll.insert(&"hello"); // Duplicate

        let estimate = hll.cardinality();
        // Should be close to 2
        assert!(estimate >= 1.0 && estimate <= 4.0, "Estimate: {}", estimate);
    }

    #[test]
    fn test_hll_accuracy() {
        let mut hll: HyperLogLog<i32> = HyperLogLog::new(14); // ~0.8% error

        let n = 100000;
        for i in 0..n {
            hll.insert(&i);
        }

        let estimate = hll.cardinality();
        let error_rate = (estimate - n as f64).abs() / n as f64;

        // Should be within ~3% (about 3 standard deviations)
        assert!(
            error_rate < 0.03,
            "Error rate {} too high for n={}, estimate={}",
            error_rate,
            n,
            estimate
        );
    }

    #[test]
    fn test_hll_with_duplicates() {
        let mut hll: HyperLogLog<i32> = HyperLogLog::new(12);

        // Insert 1000 unique values, each 10 times
        for i in 0..1000 {
            for _ in 0..10 {
                hll.insert(&i);
            }
        }

        let estimate = hll.cardinality();
        // Should be close to 1000, not 10000
        let error_rate = (estimate - 1000.0).abs() / 1000.0;

        assert!(
            error_rate < 0.1,
            "Error rate {} too high, estimate={}",
            error_rate,
            estimate
        );
    }

    #[test]
    fn test_hll_merge() {
        let mut hll1: HyperLogLog<i32> = HyperLogLog::new(10);
        let mut hll2: HyperLogLog<i32> = HyperLogLog::new(10);

        // Insert 0..500 in hll1
        for i in 0..500 {
            hll1.insert(&i);
        }

        // Insert 250..750 in hll2
        for i in 250..750 {
            hll2.insert(&i);
        }

        let merged = hll1.merge(&hll2).unwrap();
        let estimate = merged.cardinality();

        // Should be close to 750 (union of 0..500 and 250..750)
        let error_rate = (estimate - 750.0).abs() / 750.0;

        assert!(
            error_rate < 0.15,
            "Error rate {} too high, estimate={}",
            error_rate,
            estimate
        );
    }

    #[test]
    fn test_hll_merge_different_precision() {
        let hll1: HyperLogLog<i32> = HyperLogLog::new(10);
        let hll2: HyperLogLog<i32> = HyperLogLog::new(12);

        assert!(hll1.merge(&hll2).is_none());
    }

    #[test]
    fn test_hll_with_error_rate() {
        let hll: HyperLogLog<i32> = HyperLogLog::with_error_rate(0.05);

        // Should have precision that gives ~5% error
        // 1.04 / sqrt(m) ≈ 0.05 → m ≈ 433 → p ≈ 9
        assert!(hll.precision() >= 8 && hll.precision() <= 10);
    }

    #[test]
    fn test_hll_standard_error() {
        let hll: HyperLogLog<i32> = HyperLogLog::new(10);

        let error = hll.standard_error();
        // For p=10, m=1024, error ≈ 1.04/32 ≈ 0.0325
        assert!(error > 0.02 && error < 0.05);
    }

    #[test]
    fn test_hll_empty() {
        let hll: HyperLogLog<i32> = HyperLogLog::new(10);

        assert!(hll.is_empty());
        assert!(hll.cardinality() < 1.0);
    }

    #[test]
    fn test_hll_clear() {
        let mut hll: HyperLogLog<i32> = HyperLogLog::new(10);

        for i in 0..1000 {
            hll.insert(&i);
        }

        hll.clear();
        assert!(hll.is_empty());
        assert!(hll.cardinality() < 1.0);
    }

    #[test]
    fn test_hll_memory_usage() {
        let hll: HyperLogLog<i32> = HyperLogLog::new(14);

        // 2^14 = 16384 bytes
        assert_eq!(hll.memory_usage(), 16384);
    }

    #[test]
    fn test_hll_intersection() {
        let mut hll1: HyperLogLog<i32> = HyperLogLog::new(12);
        let mut hll2: HyperLogLog<i32> = HyperLogLog::new(12);

        // Insert 0..1000 in hll1
        for i in 0..1000 {
            hll1.insert(&i);
        }

        // Insert 500..1500 in hll2
        for i in 500..1500 {
            hll2.insert(&i);
        }

        let intersection = estimate_intersection(&hll1, &hll2).unwrap();

        // Intersection should be ~500 (500..1000)
        // This is less accurate than union estimation
        assert!(
            intersection >= 200.0 && intersection <= 800.0,
            "Intersection estimate: {}",
            intersection
        );
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut hll1: HyperLogLog<i32> = HyperLogLog::new(12);
        let mut hll2: HyperLogLog<i32> = HyperLogLog::new(12);

        // Same set in both
        for i in 0..1000 {
            hll1.insert(&i);
            hll2.insert(&i);
        }

        let jaccard = jaccard_similarity(&hll1, &hll2).unwrap();

        // Should be close to 1.0
        assert!(jaccard > 0.8, "Jaccard: {}", jaccard);
    }

    #[test]
    fn test_hll_plus_basic() {
        let mut hll: HyperLogLogPlus<i32> = HyperLogLogPlus::new(12);

        // Small cardinality - should be in sparse mode
        for i in 0..100 {
            hll.insert(&i);
        }

        assert!(hll.is_sparse());

        // Exact count in sparse mode
        let estimate = hll.cardinality();
        assert_eq!(estimate, 100.0);
    }

    #[test]
    fn test_hll_plus_dense() {
        let mut hll: HyperLogLogPlus<i32> = HyperLogLogPlus::new(14); // Higher precision

        // Large cardinality - should switch to dense
        for i in 0..100000 {
            hll.insert(&i);
        }

        assert!(!hll.is_sparse());

        let estimate = hll.cardinality();
        let error_rate = (estimate - 100000.0).abs() / 100000.0;

        // With p=14 (m=16384), error should be ~0.8%, allow up to 5%
        assert!(
            error_rate < 0.05,
            "Error rate {} too high, estimate={}",
            error_rate,
            estimate
        );
    }

    #[test]
    fn test_hll_strings() {
        let mut hll: HyperLogLog<String> = HyperLogLog::new(12);

        let words = vec!["apple", "banana", "cherry", "date", "elderberry"];

        for word in &words {
            for _ in 0..100 {
                hll.insert(&word.to_string());
            }
        }

        let estimate = hll.cardinality();
        // Should be close to 5
        assert!(estimate >= 4.0 && estimate <= 7.0, "Estimate: {}", estimate);
    }
}
