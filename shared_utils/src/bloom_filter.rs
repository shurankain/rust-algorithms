// Bloom Filter - Space-efficient probabilistic set membership
//
// A Bloom filter is a space-efficient probabilistic data structure that tests
// whether an element is a member of a set. False positives are possible, but
// false negatives are not. Elements can be added but not removed.
//
// How it works:
// 1. Initialize a bit array of m bits, all set to 0
// 2. Use k independent hash functions, each mapping elements to [0, m-1]
// 3. To add an element: set bits at all k hash positions to 1
// 4. To query: check if ALL k hash positions are 1
//    - If any bit is 0: element is definitely NOT in set
//    - If all bits are 1: element is PROBABLY in set (may be false positive)
//
// Visualization (m=10, k=3):
//
//   Add "hello":  h1("hello")=2, h2("hello")=5, h3("hello")=9
//   Bit array:    [0 0 1 0 0 1 0 0 0 1]
//                      ^     ^       ^
//
//   Add "world":  h1("world")=1, h2("world")=5, h3("world")=7
//   Bit array:    [0 1 1 0 0 1 0 1 0 1]
//                    ^       ^   ^
//
//   Query "hello": positions 2,5,9 all set → probably in set (true positive)
//   Query "foo":   positions 1,3,8 → position 3 is 0 → definitely not in set
//
// Optimal parameters:
// - Given n elements and desired false positive rate p:
//   m = -n * ln(p) / (ln(2))^2  (optimal number of bits)
//   k = (m/n) * ln(2)           (optimal number of hash functions)
//
// False positive probability:
//   p ≈ (1 - e^(-kn/m))^k
//
// Space complexity: O(m) bits
// Time complexity: O(k) for both insert and query
//
// Applications:
// - Cache filtering (avoid expensive lookups for absent items)
// - Spell checkers (quick rejection of valid words)
// - Network routing (packet filtering)
// - Database query optimization
// - Duplicate detection in streams

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

// Bloom filter implementation
pub struct BloomFilter<T> {
    bits: Vec<bool>,
    num_bits: usize,
    num_hashes: usize,
    count: usize, // Approximate number of elements added
    _marker: PhantomData<T>,
}

impl<T: Hash> BloomFilter<T> {
    // Create with specified number of bits and hash functions
    pub fn new(num_bits: usize, num_hashes: usize) -> Self {
        assert!(num_bits > 0, "Number of bits must be positive");
        assert!(num_hashes > 0, "Number of hash functions must be positive");

        Self {
            bits: vec![false; num_bits],
            num_bits,
            num_hashes,
            count: 0,
            _marker: PhantomData,
        }
    }

    // Create with optimal parameters for expected elements and false positive rate
    pub fn with_rate(expected_elements: usize, false_positive_rate: f64) -> Self {
        assert!(expected_elements > 0, "Expected elements must be positive");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "False positive rate must be between 0 and 1"
        );

        let (num_bits, num_hashes) = optimal_parameters(expected_elements, false_positive_rate);

        Self::new(num_bits, num_hashes)
    }

    // Insert an element
    pub fn insert(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let index = self.hash_index(item, i);
            self.bits[index] = true;
        }
        self.count += 1;
    }

    // Check if element might be in the set
    // Returns true if element is PROBABLY in set (may be false positive)
    // Returns false if element is DEFINITELY not in set
    pub fn contains(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let index = self.hash_index(item, i);
            if !self.bits[index] {
                return false; // Definitely not in set
            }
        }
        true // Probably in set
    }

    // Compute hash index for item using hash function i
    fn hash_index(&self, item: &T, i: usize) -> usize {
        // Use double hashing: h(x,i) = (h1(x) + i * h2(x)) mod m
        // This simulates k independent hash functions with just 2
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(i);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish();

        ((h1.wrapping_add((i as u64).wrapping_mul(h2))) % self.num_bits as u64) as usize
    }

    // Get approximate number of elements inserted
    pub fn count(&self) -> usize {
        self.count
    }

    // Get number of bits set to 1
    pub fn bits_set(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    // Get total number of bits
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    // Get number of hash functions
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    // Estimate current false positive rate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let bits_set = self.bits_set() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        // p ≈ (bits_set / m)^k
        (bits_set / m).powf(k)
    }

    // Clear all bits
    pub fn clear(&mut self) {
        self.bits.fill(false);
        self.count = 0;
    }

    // Check if filter is empty (no elements added)
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    // Union of two bloom filters (bitwise OR)
    // Both filters must have same parameters
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return None;
        }

        let mut result = Self::new(self.num_bits, self.num_hashes);
        for i in 0..self.num_bits {
            result.bits[i] = self.bits[i] || other.bits[i];
        }
        result.count = self.count + other.count; // Upper bound estimate
        Some(result)
    }

    // Intersection of two bloom filters (bitwise AND)
    // Both filters must have same parameters
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return None;
        }

        let mut result = Self::new(self.num_bits, self.num_hashes);
        for i in 0..self.num_bits {
            result.bits[i] = self.bits[i] && other.bits[i];
        }
        // Count is not meaningful for intersection
        Some(result)
    }
}

// Calculate optimal parameters for bloom filter
pub fn optimal_parameters(n: usize, p: f64) -> (usize, usize) {
    // m = -n * ln(p) / (ln(2))^2
    let ln2 = std::f64::consts::LN_2;
    let m = (-(n as f64) * p.ln() / (ln2 * ln2)).ceil() as usize;

    // k = (m/n) * ln(2)
    let k = ((m as f64 / n as f64) * ln2).round() as usize;

    (m.max(1), k.max(1))
}

// Estimate false positive rate for given parameters
pub fn false_positive_rate(n: usize, m: usize, k: usize) -> f64 {
    // p ≈ (1 - e^(-kn/m))^k
    let exponent = -(k as f64 * n as f64 / m as f64);
    (1.0 - exponent.exp()).powf(k as f64)
}

// Counting Bloom Filter - supports deletions
// Each slot is a counter instead of a single bit
pub struct CountingBloomFilter<T> {
    counters: Vec<u8>, // Use u8 for reasonable memory, max count 255
    num_slots: usize,
    num_hashes: usize,
    count: usize,
    _marker: PhantomData<T>,
}

impl<T: Hash> CountingBloomFilter<T> {
    pub fn new(num_slots: usize, num_hashes: usize) -> Self {
        assert!(num_slots > 0, "Number of slots must be positive");
        assert!(num_hashes > 0, "Number of hash functions must be positive");

        Self {
            counters: vec![0; num_slots],
            num_slots,
            num_hashes,
            count: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_rate(expected_elements: usize, false_positive_rate: f64) -> Self {
        let (num_slots, num_hashes) = optimal_parameters(expected_elements, false_positive_rate);
        Self::new(num_slots, num_hashes)
    }

    fn hash_index(&self, item: &T, i: usize) -> usize {
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(i);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish();

        ((h1.wrapping_add((i as u64).wrapping_mul(h2))) % self.num_slots as u64) as usize
    }

    pub fn insert(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let index = self.hash_index(item, i);
            self.counters[index] = self.counters[index].saturating_add(1);
        }
        self.count += 1;
    }

    pub fn remove(&mut self, item: &T) -> bool {
        // First check if item might be present
        if !self.contains(item) {
            return false;
        }

        for i in 0..self.num_hashes {
            let index = self.hash_index(item, i);
            self.counters[index] = self.counters[index].saturating_sub(1);
        }
        self.count = self.count.saturating_sub(1);
        true
    }

    pub fn contains(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let index = self.hash_index(item, i);
            if self.counters[index] == 0 {
                return false;
            }
        }
        true
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        self.counters.fill(0);
        self.count = 0;
    }
}

// Scalable Bloom Filter - grows automatically
// Creates new filters when current one gets too full
pub struct ScalableBloomFilter<T> {
    filters: Vec<BloomFilter<T>>,
    initial_capacity: usize,
    false_positive_rate: f64,
    growth_factor: usize,
}

impl<T: Hash> ScalableBloomFilter<T> {
    pub fn new(initial_capacity: usize, false_positive_rate: f64) -> Self {
        let first_filter = BloomFilter::with_rate(initial_capacity, false_positive_rate);

        Self {
            filters: vec![first_filter],
            initial_capacity,
            false_positive_rate,
            growth_factor: 2,
        }
    }

    pub fn insert(&mut self, item: &T) {
        // Check if current filter is getting full
        let current = self.filters.last().unwrap();
        let fill_ratio = current.bits_set() as f64 / current.num_bits() as f64;

        if fill_ratio > 0.5 {
            // Create a new, larger filter
            let new_capacity =
                self.initial_capacity * self.growth_factor.pow(self.filters.len() as u32);
            // Use tighter false positive rate for new filters
            let new_rate = self.false_positive_rate * 0.5_f64.powi(self.filters.len() as i32);
            let new_filter = BloomFilter::with_rate(new_capacity, new_rate.max(0.0001));
            self.filters.push(new_filter);
        }

        self.filters.last_mut().unwrap().insert(item);
    }

    pub fn contains(&self, item: &T) -> bool {
        // Check all filters (item could be in any of them)
        self.filters.iter().any(|f| f.contains(item))
    }

    pub fn count(&self) -> usize {
        self.filters.iter().map(|f| f.count()).sum()
    }

    pub fn num_filters(&self) -> usize {
        self.filters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_basic() {
        let mut filter: BloomFilter<&str> = BloomFilter::new(1000, 5);

        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        // Very unlikely to be a false positive with good parameters
        assert!(!filter.contains(&"foobar"));
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut filter: BloomFilter<i32> = BloomFilter::with_rate(1000, 0.01);

        // Insert many elements
        for i in 0..1000 {
            filter.insert(&i);
        }

        // All inserted elements must be found (no false negatives)
        for i in 0..1000 {
            assert!(filter.contains(&i), "False negative for {}", i);
        }
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let n = 10000;
        let target_fp_rate = 0.01;
        let mut filter: BloomFilter<i32> = BloomFilter::with_rate(n, target_fp_rate);

        // Insert elements
        for i in 0..n as i32 {
            filter.insert(&i);
        }

        // Check false positives on elements NOT in the set
        let test_range = n as i32..(n as i32 * 2);
        let false_positives = test_range.clone().filter(|x| filter.contains(x)).count();

        let actual_fp_rate = false_positives as f64 / test_range.count() as f64;

        // Allow some margin (FP rate should be roughly target or less)
        assert!(
            actual_fp_rate < target_fp_rate * 2.0,
            "False positive rate {} is too high (target {})",
            actual_fp_rate,
            target_fp_rate
        );
    }

    #[test]
    fn test_optimal_parameters() {
        let (m, k) = optimal_parameters(10000, 0.01);

        // For n=10000 and p=0.01:
        // m ≈ 95851 bits, k ≈ 7
        assert!(m > 90000 && m < 100000, "m = {}", m);
        assert!(k >= 6 && k <= 8, "k = {}", k);
    }

    #[test]
    fn test_false_positive_rate_calculation() {
        let fp = false_positive_rate(1000, 10000, 7);
        assert!(fp < 0.01); // Should be very low with these parameters
    }

    #[test]
    fn test_bloom_clear() {
        let mut filter: BloomFilter<&str> = BloomFilter::new(100, 3);

        filter.insert(&"test");
        assert!(filter.contains(&"test"));

        filter.clear();
        assert!(!filter.contains(&"test"));
        assert!(filter.is_empty());
    }

    #[test]
    fn test_bloom_union() {
        let mut filter1: BloomFilter<&str> = BloomFilter::new(100, 3);
        let mut filter2: BloomFilter<&str> = BloomFilter::new(100, 3);

        filter1.insert(&"hello");
        filter2.insert(&"world");

        let union = filter1.union(&filter2).unwrap();
        assert!(union.contains(&"hello"));
        assert!(union.contains(&"world"));
    }

    #[test]
    fn test_bloom_intersection() {
        let mut filter1: BloomFilter<i32> = BloomFilter::new(1000, 5);
        let mut filter2: BloomFilter<i32> = BloomFilter::new(1000, 5);

        // Insert overlapping ranges
        for i in 0..50 {
            filter1.insert(&i);
        }
        for i in 25..75 {
            filter2.insert(&i);
        }

        let intersection = filter1.intersection(&filter2).unwrap();

        // Elements in both should likely be in intersection
        // (with some chance of false positives)
        for i in 25..50 {
            assert!(intersection.contains(&i), "Missing {}", i);
        }
    }

    #[test]
    fn test_counting_bloom_basic() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 5);

        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"foo"));
    }

    #[test]
    fn test_counting_bloom_remove() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 5);

        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));

        filter.remove(&"hello");
        // After removal, item should not be found
        // (assuming no hash collisions with other items)
        assert!(!filter.contains(&"hello"));
    }

    #[test]
    fn test_counting_bloom_multiple_inserts() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 5);

        filter.insert(&"hello");
        filter.insert(&"hello");
        filter.insert(&"hello");

        assert!(filter.contains(&"hello"));

        // Remove twice, should still contain
        filter.remove(&"hello");
        filter.remove(&"hello");
        assert!(filter.contains(&"hello"));

        // Remove third time
        filter.remove(&"hello");
        assert!(!filter.contains(&"hello"));
    }

    #[test]
    fn test_scalable_bloom() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);

        // Insert more than initial capacity
        for i in 0..1000 {
            filter.insert(&i);
        }

        // Should have created multiple internal filters
        assert!(filter.num_filters() > 1);

        // All elements should still be found
        for i in 0..1000 {
            assert!(filter.contains(&i), "Missing {}", i);
        }
    }

    #[test]
    fn test_bloom_strings() {
        let mut filter: BloomFilter<String> = BloomFilter::with_rate(100, 0.01);

        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        for word in &words {
            filter.insert(word);
        }

        for word in &words {
            assert!(filter.contains(word));
        }

        assert!(!filter.contains(&"dragon".to_string()));
    }

    #[test]
    fn test_bloom_integers() {
        let mut filter: BloomFilter<u64> = BloomFilter::new(10000, 7);

        for i in 0..1000u64 {
            filter.insert(&(i * 7));
        }

        for i in 0..1000u64 {
            assert!(filter.contains(&(i * 7)));
        }
    }

    #[test]
    fn test_estimated_fp_rate() {
        let mut filter: BloomFilter<i32> = BloomFilter::with_rate(1000, 0.01);

        // Empty filter should have ~0 FP rate
        assert!(filter.estimated_false_positive_rate() < 0.0001);

        // Insert elements
        for i in 0..500 {
            filter.insert(&i);
        }

        // FP rate should increase but still be reasonable
        let fp_rate = filter.estimated_false_positive_rate();
        assert!(fp_rate > 0.0 && fp_rate < 0.5);
    }

    #[test]
    fn test_bloom_bits_set() {
        let mut filter: BloomFilter<i32> = BloomFilter::new(100, 3);

        assert_eq!(filter.bits_set(), 0);

        filter.insert(&42);
        assert!(filter.bits_set() > 0);
        assert!(filter.bits_set() <= 3); // At most k bits set for one element
    }
}
