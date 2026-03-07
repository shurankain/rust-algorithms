// Count-Min Sketch - Frequency estimation in data streams
//
// A Count-Min Sketch is a probabilistic data structure that estimates the frequency
// of elements in a data stream using sub-linear space. Unlike exact counting, it may
// overestimate frequencies but never underestimates them.
//
// How it works:
// 1. Create a 2D array of counters with d rows and w columns
// 2. Use d independent hash functions, one per row
// 3. To increment count for element x:
//    - For each row i, compute hash_i(x) mod w
//    - Increment counter at position [i][hash_i(x) mod w]
// 4. To query frequency of element x:
//    - For each row i, look up counter at [i][hash_i(x) mod w]
//    - Return the minimum of all d counters
//
// Visualization (d=3, w=5):
//
//   Element "apple" hashes to columns [2, 0, 4]
//   Element "banana" hashes to columns [1, 2, 4]
//
//   Row 0:  [0  1  5  0  0]   ← "banana" and "apple" hash here
//              ↑  ↑
//   Row 1:  [3  0  2  0  0]   ← "apple" and "banana" hash here
//           ↑     ↑
//   Row 2:  [0  0  0  0  7]   ← both hash to column 4 (collision!)
//                       ↑
//
//   Query "apple": min(5, 3, 7) = 3  (actual count: 3, no error)
//   Query "banana": min(1, 2, 7) = 1 (actual count: 1, collision caused overcount in row 2)
//
// Error bounds:
// - With probability 1 - δ, the estimate differs from true count by at most εn
//   where n is total count of all items
// - Width w = ⌈e/ε⌉ and depth d = ⌈ln(1/δ)⌉
//
// Space complexity: O(w * d) = O((1/ε) * log(1/δ))
// Time complexity: O(d) for both update and query
//
// Applications:
// - Network traffic monitoring (counting packet types)
// - Database query optimization (frequency estimates)
// - Finding heavy hitters in streams
// - Natural language processing (word frequency)
// - Click stream analysis

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

// Count-Min Sketch implementation
pub struct CountMinSketch<T> {
    counters: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
    total_count: u64,
    _marker: PhantomData<T>,
}

impl<T: Hash> CountMinSketch<T> {
    // Create with specified width and depth
    pub fn new(width: usize, depth: usize) -> Self {
        assert!(width > 0, "Width must be positive");
        assert!(depth > 0, "Depth must be positive");

        Self {
            counters: vec![vec![0u64; width]; depth],
            width,
            depth,
            total_count: 0,
            _marker: PhantomData,
        }
    }

    // Create with desired error rate (epsilon) and confidence (1 - delta)
    // epsilon: maximum relative error
    // delta: failure probability
    pub fn with_error_rate(epsilon: f64, delta: f64) -> Self {
        assert!(
            epsilon > 0.0 && epsilon < 1.0,
            "Epsilon must be between 0 and 1"
        );
        assert!(delta > 0.0 && delta < 1.0, "Delta must be between 0 and 1");

        // w = ceil(e / epsilon)
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        // d = ceil(ln(1 / delta))
        let depth = (1.0 / delta).ln().ceil() as usize;

        Self::new(width.max(1), depth.max(1))
    }

    // Compute hash for row i
    fn hash_index(&self, item: &T, row: usize) -> usize {
        // Use double hashing: h(x,i) = (h1(x) + i * h2(x)) mod w
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(row);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish();

        ((h1.wrapping_add((row as u64).wrapping_mul(h2))) % self.width as u64) as usize
    }

    // Increment count for an item
    pub fn increment(&mut self, item: &T) {
        self.add(item, 1);
    }

    // Add count to an item
    pub fn add(&mut self, item: &T, count: u64) {
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            self.counters[row][col] = self.counters[row][col].saturating_add(count);
        }
        self.total_count = self.total_count.saturating_add(count);
    }

    // Estimate frequency of an item (returns minimum across all rows)
    pub fn estimate(&self, item: &T) -> u64 {
        let mut min_count = u64::MAX;
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            min_count = min_count.min(self.counters[row][col]);
        }
        min_count
    }

    // Get total count of all items
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    // Get width (number of columns)
    pub fn width(&self) -> usize {
        self.width
    }

    // Get depth (number of rows/hash functions)
    pub fn depth(&self) -> usize {
        self.depth
    }

    // Clear all counters
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
        self.total_count = 0;
    }

    // Merge two sketches (must have same dimensions)
    pub fn merge(&self, other: &Self) -> Option<Self> {
        if self.width != other.width || self.depth != other.depth {
            return None;
        }

        let mut result = Self::new(self.width, self.depth);
        for row in 0..self.depth {
            for col in 0..self.width {
                result.counters[row][col] =
                    self.counters[row][col].saturating_add(other.counters[row][col]);
            }
        }
        result.total_count = self.total_count.saturating_add(other.total_count);
        Some(result)
    }

    // Inner product of two sketches (estimates sum of f(x)*g(x) for all x)
    pub fn inner_product(&self, other: &Self) -> Option<u64> {
        if self.width != other.width || self.depth != other.depth {
            return None;
        }

        let mut min_product = u64::MAX;
        for row in 0..self.depth {
            let mut row_product = 0u64;
            for col in 0..self.width {
                row_product =
                    row_product.saturating_add(self.counters[row][col] * other.counters[row][col]);
            }
            min_product = min_product.min(row_product);
        }
        Some(min_product)
    }
}

// Count-Min Sketch with conservative update
// Only increments the minimum counters, reducing overestimation
pub struct ConservativeCountMinSketch<T> {
    counters: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
    total_count: u64,
    _marker: PhantomData<T>,
}

impl<T: Hash> ConservativeCountMinSketch<T> {
    pub fn new(width: usize, depth: usize) -> Self {
        assert!(width > 0, "Width must be positive");
        assert!(depth > 0, "Depth must be positive");

        Self {
            counters: vec![vec![0u64; width]; depth],
            width,
            depth,
            total_count: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_error_rate(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width.max(1), depth.max(1))
    }

    fn hash_index(&self, item: &T, row: usize) -> usize {
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(row);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish();

        ((h1.wrapping_add((row as u64).wrapping_mul(h2))) % self.width as u64) as usize
    }

    // Conservative update: only increment counters that are at the minimum
    pub fn increment(&mut self, item: &T) {
        self.add(item, 1);
    }

    pub fn add(&mut self, item: &T, count: u64) {
        // First, find the current minimum estimate
        let mut indices = Vec::with_capacity(self.depth);
        let mut min_val = u64::MAX;

        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            indices.push(col);
            min_val = min_val.min(self.counters[row][col]);
        }

        // Only update counters to the new minimum + count
        let new_val = min_val.saturating_add(count);
        for (row, &col) in indices.iter().enumerate() {
            if self.counters[row][col] < new_val {
                self.counters[row][col] = new_val;
            }
        }

        self.total_count = self.total_count.saturating_add(count);
    }

    pub fn estimate(&self, item: &T) -> u64 {
        let mut min_count = u64::MAX;
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            min_count = min_count.min(self.counters[row][col]);
        }
        min_count
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
        self.total_count = 0;
    }
}

// Count-Min Sketch with decay for time-windowed counting
pub struct DecayingCountMinSketch<T> {
    counters: Vec<Vec<f64>>,
    width: usize,
    depth: usize,
    decay_rate: f64,
    total_count: f64,
    _marker: PhantomData<T>,
}

impl<T: Hash> DecayingCountMinSketch<T> {
    pub fn new(width: usize, depth: usize, decay_rate: f64) -> Self {
        assert!(width > 0, "Width must be positive");
        assert!(depth > 0, "Depth must be positive");
        assert!(
            decay_rate > 0.0 && decay_rate <= 1.0,
            "Decay rate must be between 0 and 1"
        );

        Self {
            counters: vec![vec![0.0; width]; depth],
            width,
            depth,
            decay_rate,
            total_count: 0.0,
            _marker: PhantomData,
        }
    }

    fn hash_index(&self, item: &T, row: usize) -> usize {
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(row);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish();

        ((h1.wrapping_add((row as u64).wrapping_mul(h2))) % self.width as u64) as usize
    }

    // Apply decay to all counters (call periodically)
    pub fn decay(&mut self) {
        for row in &mut self.counters {
            for counter in row {
                *counter *= self.decay_rate;
            }
        }
        self.total_count *= self.decay_rate;
    }

    pub fn increment(&mut self, item: &T) {
        self.add(item, 1.0);
    }

    pub fn add(&mut self, item: &T, count: f64) {
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            self.counters[row][col] += count;
        }
        self.total_count += count;
    }

    pub fn estimate(&self, item: &T) -> f64 {
        let mut min_count = f64::MAX;
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            min_count = min_count.min(self.counters[row][col]);
        }
        min_count
    }

    pub fn total_count(&self) -> f64 {
        self.total_count
    }

    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0.0);
        }
        self.total_count = 0.0;
    }
}

// Heavy hitters finder using Count-Min Sketch
pub struct HeavyHitters<T: Hash + Clone + Eq> {
    sketch: CountMinSketch<T>,
    candidates: Vec<(T, u64)>,
    threshold: f64, // Report items with frequency > threshold * total_count
}

impl<T: Hash + Clone + Eq> HeavyHitters<T> {
    pub fn new(width: usize, depth: usize, threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold <= 1.0,
            "Threshold must be between 0 and 1"
        );

        Self {
            sketch: CountMinSketch::new(width, depth),
            candidates: Vec::new(),
            threshold,
        }
    }

    pub fn increment(&mut self, item: T) {
        self.sketch.increment(&item);

        // Check if this item might be a heavy hitter
        let estimate = self.sketch.estimate(&item);
        let threshold_count = (self.threshold * self.sketch.total_count() as f64) as u64;

        if estimate >= threshold_count {
            // Update or add to candidates
            if let Some(entry) = self.candidates.iter_mut().find(|(i, _)| *i == item) {
                entry.1 = estimate;
            } else {
                self.candidates.push((item, estimate));
            }
        }
    }

    // Get items that appear to be heavy hitters
    pub fn get_heavy_hitters(&self) -> Vec<(&T, u64)> {
        let threshold_count = (self.threshold * self.sketch.total_count() as f64) as u64;
        self.candidates
            .iter()
            .filter(|(item, _)| self.sketch.estimate(item) >= threshold_count)
            .map(|(item, _)| (item, self.sketch.estimate(item)))
            .collect()
    }

    pub fn total_count(&self) -> u64 {
        self.sketch.total_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_min_basic() {
        let mut sketch: CountMinSketch<&str> = CountMinSketch::new(100, 5);

        sketch.increment(&"hello");
        sketch.increment(&"hello");
        sketch.increment(&"hello");
        sketch.increment(&"world");

        assert!(sketch.estimate(&"hello") >= 3);
        assert!(sketch.estimate(&"world") >= 1);
        assert_eq!(sketch.total_count(), 4);
    }

    #[test]
    fn test_count_min_no_underestimate() {
        let mut sketch: CountMinSketch<i32> = CountMinSketch::with_error_rate(0.01, 0.01);

        // Insert many items
        for i in 0..1000 {
            for _ in 0..i {
                sketch.increment(&i);
            }
        }

        // Estimates should never be less than true count
        for i in 0..1000 {
            assert!(
                sketch.estimate(&i) >= i as u64,
                "Underestimate for {}: {} < {}",
                i,
                sketch.estimate(&i),
                i
            );
        }
    }

    #[test]
    fn test_count_min_error_bound() {
        let epsilon = 0.1;
        let mut sketch: CountMinSketch<i32> = CountMinSketch::with_error_rate(epsilon, 0.01);

        let n = 10000;
        for i in 0..n {
            sketch.increment(&i);
        }

        // Most estimates should be within epsilon * n of true count
        let mut large_errors = 0;
        for i in 0..n {
            let estimate = sketch.estimate(&i);
            let true_count = 1u64;
            let error = estimate - true_count;
            if error as f64 > epsilon * n as f64 {
                large_errors += 1;
            }
        }

        // With delta=0.01, we expect less than 1% to exceed the error bound
        assert!(
            large_errors < n / 10,
            "Too many large errors: {}",
            large_errors
        );
    }

    #[test]
    fn test_count_min_add() {
        let mut sketch: CountMinSketch<&str> = CountMinSketch::new(100, 5);

        sketch.add(&"apple", 10);
        sketch.add(&"banana", 5);

        assert!(sketch.estimate(&"apple") >= 10);
        assert!(sketch.estimate(&"banana") >= 5);
        assert_eq!(sketch.total_count(), 15);
    }

    #[test]
    fn test_count_min_merge() {
        let mut sketch1: CountMinSketch<&str> = CountMinSketch::new(100, 5);
        let mut sketch2: CountMinSketch<&str> = CountMinSketch::new(100, 5);

        sketch1.add(&"apple", 10);
        sketch2.add(&"apple", 5);
        sketch2.add(&"banana", 3);

        let merged = sketch1.merge(&sketch2).unwrap();

        assert!(merged.estimate(&"apple") >= 15);
        assert!(merged.estimate(&"banana") >= 3);
        assert_eq!(merged.total_count(), 18);
    }

    #[test]
    fn test_count_min_merge_different_sizes() {
        let sketch1: CountMinSketch<&str> = CountMinSketch::new(100, 5);
        let sketch2: CountMinSketch<&str> = CountMinSketch::new(50, 5);

        assert!(sketch1.merge(&sketch2).is_none());
    }

    #[test]
    fn test_count_min_clear() {
        let mut sketch: CountMinSketch<&str> = CountMinSketch::new(100, 5);

        sketch.add(&"test", 100);
        assert!(sketch.estimate(&"test") >= 100);

        sketch.clear();
        assert_eq!(sketch.estimate(&"test"), 0);
        assert_eq!(sketch.total_count(), 0);
    }

    #[test]
    fn test_count_min_inner_product() {
        let mut sketch1: CountMinSketch<i32> = CountMinSketch::new(100, 5);
        let mut sketch2: CountMinSketch<i32> = CountMinSketch::new(100, 5);

        // Same element in both
        sketch1.add(&1, 10);
        sketch2.add(&1, 5);

        let product = sketch1.inner_product(&sketch2).unwrap();
        assert!(product >= 50); // At least 10 * 5
    }

    #[test]
    fn test_conservative_update() {
        let mut sketch: ConservativeCountMinSketch<&str> = ConservativeCountMinSketch::new(100, 5);

        // Add same item multiple times
        for _ in 0..100 {
            sketch.increment(&"hello");
        }

        // Conservative update should give more accurate estimates
        assert_eq!(sketch.estimate(&"hello"), 100);
    }

    #[test]
    fn test_conservative_vs_regular() {
        let mut regular: CountMinSketch<i32> = CountMinSketch::new(20, 3);
        let mut conservative: ConservativeCountMinSketch<i32> =
            ConservativeCountMinSketch::new(20, 3);

        // Insert many items to cause collisions
        for i in 0..100 {
            for _ in 0..10 {
                regular.increment(&i);
                conservative.increment(&i);
            }
        }

        // Both should never underestimate
        for i in 0..100 {
            assert!(regular.estimate(&i) >= 10);
            assert!(conservative.estimate(&i) >= 10);
        }

        // Conservative should generally have lower overestimates
        // (but this is probabilistic, so we don't assert it strongly)
    }

    #[test]
    fn test_decaying_sketch() {
        let mut sketch: DecayingCountMinSketch<&str> = DecayingCountMinSketch::new(100, 5, 0.5);

        sketch.add(&"hello", 100.0);
        assert!(sketch.estimate(&"hello") >= 100.0);

        sketch.decay();
        assert!(sketch.estimate(&"hello") >= 49.0 && sketch.estimate(&"hello") <= 51.0);

        sketch.decay();
        assert!(sketch.estimate(&"hello") >= 24.0 && sketch.estimate(&"hello") <= 26.0);
    }

    #[test]
    fn test_heavy_hitters() {
        let mut hh: HeavyHitters<&str> = HeavyHitters::new(100, 5, 0.1);

        // Add a heavy hitter
        for _ in 0..100 {
            hh.increment("frequent");
        }

        // Add some noise
        for i in 0..10 {
            let s = format!("rare_{}", i);
            let s_static: &'static str = Box::leak(s.into_boxed_str());
            hh.increment(s_static);
        }

        let heavy = hh.get_heavy_hitters();
        assert!(
            heavy.iter().any(|(item, _)| **item == "frequent"),
            "Should identify 'frequent' as heavy hitter"
        );
    }

    #[test]
    fn test_count_min_integers() {
        let mut sketch: CountMinSketch<u64> = CountMinSketch::new(1000, 5);

        for i in 0..1000u64 {
            sketch.add(&i, i);
        }

        // Check that estimates are at least the true counts
        for i in 0..1000u64 {
            assert!(sketch.estimate(&i) >= i);
        }
    }

    #[test]
    fn test_with_error_rate() {
        let sketch: CountMinSketch<i32> = CountMinSketch::with_error_rate(0.01, 0.01);

        // Width should be approximately e/0.01 ≈ 272
        assert!(sketch.width() >= 200 && sketch.width() <= 300);
        // Depth should be approximately ln(100) ≈ 5
        assert!(sketch.depth() >= 4 && sketch.depth() <= 6);
    }

    #[test]
    fn test_count_min_strings() {
        let mut sketch: CountMinSketch<String> = CountMinSketch::with_error_rate(0.01, 0.01);

        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        for word in &words {
            for _ in 0..50 {
                sketch.increment(word);
            }
        }

        for word in &words {
            assert!(sketch.estimate(word) >= 50);
        }
    }
}
