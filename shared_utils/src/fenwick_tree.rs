// Fenwick Tree (Binary Indexed Tree) - Efficient prefix sums and point updates
//
// A Fenwick tree is a data structure that efficiently supports:
// - Point updates: add a value to element at index i
// - Prefix queries: compute sum of elements from 0 to i
// - Range queries: compute sum of elements from i to j (using two prefix queries)
//
// Key insight: Uses binary representation of indices to determine which ranges
// each node is responsible for. The lowest set bit (LSB) determines the range size.
//
// Structure visualization for n=8:
//
// Index (1-based):  1    2    3    4    5    6    7    8
// Binary:          001  010  011  100  101  110  111 1000
// LSB:              1    2    1    4    1    2    1    8
// Responsible for: [1] [1,2] [3] [1-4] [5] [5,6] [7] [1-8]
//
//                         tree[8] (sum of 1-8)
//                        /
//              tree[4] (sum of 1-4)
//             /         \
//     tree[2] (1-2)    tree[6] (5-6)
//     /      \         /      \
// tree[1]  tree[3]  tree[5]  tree[7]
//
// Operations use bit manipulation:
// - Parent in update: i += i & (-i)  (add LSB)
// - Parent in query:  i -= i & (-i)  (remove LSB)
//
// Complexity:
// - Build: O(n)
// - Point update: O(log n)
// - Prefix sum: O(log n)
// - Range sum: O(log n)
// - Space: O(n)
//
// Advantages over Segment Tree:
// - Simpler implementation
// - Less memory (n vs 2n or 4n)
// - Better cache performance
// - Faster constant factors
//
// Limitations:
// - Only supports invertible operations (sum, xor) not min/max
// - Range updates require more complex variants

use std::ops::{Add, Sub};

// Basic Fenwick tree for prefix sums
pub struct FenwickTree<T> {
    tree: Vec<T>, // 1-indexed internally
    n: usize,
}

impl<T> FenwickTree<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default,
{
    // Create empty Fenwick tree of given size
    pub fn new(n: usize) -> Self {
        Self {
            tree: vec![T::default(); n + 1],
            n,
        }
    }

    // Build from array in O(n)
    pub fn from_slice(arr: &[T]) -> Self {
        let n = arr.len();
        let mut tree = vec![T::default(); n + 1];

        // Copy array to tree (1-indexed)
        for (i, &val) in arr.iter().enumerate() {
            tree[i + 1] = val;
        }

        // Build tree in O(n) by propagating values up
        for i in 1..=n {
            let parent = i + (i & i.wrapping_neg());
            if parent <= n {
                tree[parent] = tree[parent] + tree[i];
            }
        }

        Self { tree, n }
    }

    // Point update: add delta to element at index idx (0-indexed)
    pub fn add(&mut self, idx: usize, delta: T) {
        assert!(idx < self.n, "Index out of bounds");
        let mut i = idx + 1; // Convert to 1-indexed
        while i <= self.n {
            self.tree[i] = self.tree[i] + delta;
            i += i & i.wrapping_neg(); // Add LSB
        }
    }

    // Prefix sum: compute sum of elements [0, idx] (inclusive)
    pub fn prefix_sum(&self, idx: usize) -> T {
        assert!(idx < self.n, "Index out of bounds");
        let mut sum = T::default();
        let mut i = idx + 1; // Convert to 1-indexed
        while i > 0 {
            sum = sum + self.tree[i];
            i -= i & i.wrapping_neg(); // Remove LSB
        }
        sum
    }

    // Range sum: compute sum of elements [left, right] (inclusive)
    pub fn range_sum(&self, left: usize, right: usize) -> T {
        assert!(left <= right && right < self.n, "Invalid range");
        if left == 0 {
            self.prefix_sum(right)
        } else {
            self.prefix_sum(right) - self.prefix_sum(left - 1)
        }
    }

    // Get single element value at index
    pub fn get(&self, idx: usize) -> T {
        if idx == 0 {
            self.prefix_sum(0)
        } else {
            self.prefix_sum(idx) - self.prefix_sum(idx - 1)
        }
    }

    // Set element at index to specific value
    pub fn set(&mut self, idx: usize, val: T) {
        let current = self.get(idx);
        let delta = val - current;
        self.add(idx, delta);
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

// Fenwick tree with range updates and point queries
// Uses difference array technique
pub struct FenwickTreeRangeUpdate<T> {
    tree: FenwickTree<T>,
}

impl<T> FenwickTreeRangeUpdate<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default,
{
    pub fn new(n: usize) -> Self {
        Self {
            tree: FenwickTree::new(n),
        }
    }

    // Add delta to all elements in range [left, right]
    pub fn range_add(&mut self, left: usize, right: usize, delta: T) {
        assert!(left <= right && right < self.tree.n, "Invalid range");
        self.tree.add(left, delta);
        if right + 1 < self.tree.n {
            self.tree.add(right + 1, T::default() - delta);
        }
    }

    // Get value at index (includes all range updates)
    pub fn get(&self, idx: usize) -> T {
        self.tree.prefix_sum(idx)
    }

    pub fn len(&self) -> usize {
        self.tree.n
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
}

// Fenwick tree with range updates and range queries
// Uses two trees to support both operations
pub struct FenwickTreeRangeUpdateRangeQuery<T> {
    b1: FenwickTree<T>, // Stores coefficients
    b2: FenwickTree<T>, // Stores prefix multipliers
}

impl<T> FenwickTreeRangeUpdateRangeQuery<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default + From<i32> + std::ops::Mul<Output = T>,
{
    pub fn new(n: usize) -> Self {
        Self {
            b1: FenwickTree::new(n),
            b2: FenwickTree::new(n),
        }
    }

    // Add delta to all elements in range [left, right]
    pub fn range_add(&mut self, left: usize, right: usize, delta: T) {
        assert!(left <= right && right < self.b1.n, "Invalid range");

        let left_t = T::from(left as i32);
        let right_plus_one_t = T::from((right + 1) as i32);

        // Update b1
        self.b1.add(left, delta);
        if right + 1 < self.b1.n {
            self.b1.add(right + 1, T::default() - delta);
        }

        // Update b2
        self.b2.add(left, delta * (left_t - T::from(1)));
        if right + 1 < self.b1.n {
            self.b2
                .add(right + 1, T::default() - delta * right_plus_one_t + delta);
        }
    }

    // Prefix sum [0, idx]
    fn prefix_sum_internal(&self, idx: usize) -> T {
        let idx_t = T::from(idx as i32);
        self.b1.prefix_sum(idx) * idx_t - self.b2.prefix_sum(idx)
    }

    // Range sum [left, right]
    pub fn range_sum(&self, left: usize, right: usize) -> T {
        assert!(left <= right && right < self.b1.n, "Invalid range");
        if left == 0 {
            self.prefix_sum_internal(right)
        } else {
            self.prefix_sum_internal(right) - self.prefix_sum_internal(left - 1)
        }
    }

    pub fn len(&self) -> usize {
        self.b1.n
    }

    pub fn is_empty(&self) -> bool {
        self.b1.is_empty()
    }
}

// 2D Fenwick Tree for 2D prefix sums
pub struct FenwickTree2D<T> {
    tree: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T> FenwickTree2D<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            tree: vec![vec![T::default(); cols + 1]; rows + 1],
            rows,
            cols,
        }
    }

    // Point update at (row, col)
    pub fn add(&mut self, row: usize, col: usize, delta: T) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        let mut i = row + 1;
        while i <= self.rows {
            let mut j = col + 1;
            while j <= self.cols {
                self.tree[i][j] = self.tree[i][j] + delta;
                j += j & j.wrapping_neg();
            }
            i += i & i.wrapping_neg();
        }
    }

    // Prefix sum from (0,0) to (row, col) inclusive
    pub fn prefix_sum(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        let mut sum = T::default();
        let mut i = row + 1;
        while i > 0 {
            let mut j = col + 1;
            while j > 0 {
                sum = sum + self.tree[i][j];
                j -= j & j.wrapping_neg();
            }
            i -= i & i.wrapping_neg();
        }
        sum
    }

    // Rectangle sum from (r1, c1) to (r2, c2) inclusive
    pub fn range_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> T {
        assert!(r1 <= r2 && c1 <= c2, "Invalid range");
        assert!(r2 < self.rows && c2 < self.cols, "Index out of bounds");

        let mut sum = self.prefix_sum(r2, c2);

        if r1 > 0 {
            sum = sum - self.prefix_sum(r1 - 1, c2);
        }
        if c1 > 0 {
            sum = sum - self.prefix_sum(r2, c1 - 1);
        }
        if r1 > 0 && c1 > 0 {
            sum = sum + self.prefix_sum(r1 - 1, c1 - 1);
        }

        sum
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

// Find k-th smallest element using Fenwick tree (for frequency counting)
// Useful when elements are from a known range [0, max_val]
pub fn find_kth<T>(tree: &FenwickTree<T>, k: T) -> Option<usize>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default + Ord,
{
    let n = tree.len();
    if n == 0 {
        return None;
    }

    // Binary search for k-th element
    let mut pos = 0;
    let mut sum = T::default();
    let mut step = n.next_power_of_two();

    while step > 0 {
        let next = pos + step;
        if next <= n {
            let next_sum = sum + tree.tree[next];
            if next_sum < k {
                pos = next;
                sum = next_sum;
            }
        }
        step /= 2;
    }

    if pos < n {
        Some(pos) // Return 0-indexed position
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fenwick_basic() {
        let arr = vec![1, 2, 3, 4, 5];
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.prefix_sum(0), 1);
        assert_eq!(tree.prefix_sum(1), 3); // 1 + 2
        assert_eq!(tree.prefix_sum(2), 6); // 1 + 2 + 3
        assert_eq!(tree.prefix_sum(3), 10); // 1 + 2 + 3 + 4
        assert_eq!(tree.prefix_sum(4), 15); // 1 + 2 + 3 + 4 + 5
    }

    #[test]
    fn test_fenwick_range_sum() {
        let arr = vec![1, 2, 3, 4, 5];
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.range_sum(0, 4), 15);
        assert_eq!(tree.range_sum(1, 3), 9); // 2 + 3 + 4
        assert_eq!(tree.range_sum(2, 2), 3); // just 3
        assert_eq!(tree.range_sum(0, 0), 1);
    }

    #[test]
    fn test_fenwick_update() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut tree = FenwickTree::from_slice(&arr);

        tree.add(2, 7); // Add 7 to index 2: [1, 2, 10, 4, 5]
        assert_eq!(tree.prefix_sum(4), 22); // 1 + 2 + 10 + 4 + 5
        assert_eq!(tree.range_sum(2, 2), 10);
    }

    #[test]
    fn test_fenwick_get() {
        let arr = vec![1, 2, 3, 4, 5];
        let tree = FenwickTree::from_slice(&arr);

        for i in 0..arr.len() {
            assert_eq!(tree.get(i), arr[i]);
        }
    }

    #[test]
    fn test_fenwick_set() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut tree = FenwickTree::from_slice(&arr);

        tree.set(2, 100);
        assert_eq!(tree.get(2), 100);
        assert_eq!(tree.prefix_sum(4), 112); // 1 + 2 + 100 + 4 + 5
    }

    #[test]
    fn test_fenwick_empty_build() {
        let tree: FenwickTree<i32> = FenwickTree::new(5);
        assert_eq!(tree.prefix_sum(0), 0);
        assert_eq!(tree.prefix_sum(4), 0);
    }

    #[test]
    fn test_fenwick_single_element() {
        let arr = vec![42];
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.prefix_sum(0), 42);
        assert_eq!(tree.get(0), 42);
    }

    #[test]
    fn test_fenwick_range_update() {
        let mut tree: FenwickTreeRangeUpdate<i32> = FenwickTreeRangeUpdate::new(5);

        tree.range_add(1, 3, 10); // Add 10 to indices 1,2,3
        assert_eq!(tree.get(0), 0);
        assert_eq!(tree.get(1), 10);
        assert_eq!(tree.get(2), 10);
        assert_eq!(tree.get(3), 10);
        assert_eq!(tree.get(4), 0);
    }

    #[test]
    fn test_fenwick_range_update_multiple() {
        let mut tree: FenwickTreeRangeUpdate<i32> = FenwickTreeRangeUpdate::new(5);

        tree.range_add(0, 4, 5); // Add 5 to all
        tree.range_add(1, 2, 10); // Add 10 to indices 1,2

        assert_eq!(tree.get(0), 5);
        assert_eq!(tree.get(1), 15);
        assert_eq!(tree.get(2), 15);
        assert_eq!(tree.get(3), 5);
        assert_eq!(tree.get(4), 5);
    }

    #[test]
    fn test_fenwick_2d_basic() {
        let mut tree: FenwickTree2D<i32> = FenwickTree2D::new(3, 3);

        // Set up a 3x3 matrix:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        tree.add(0, 0, 1);
        tree.add(0, 1, 2);
        tree.add(0, 2, 3);
        tree.add(1, 0, 4);
        tree.add(1, 1, 5);
        tree.add(1, 2, 6);
        tree.add(2, 0, 7);
        tree.add(2, 1, 8);
        tree.add(2, 2, 9);

        // Prefix sum of entire matrix
        assert_eq!(tree.prefix_sum(2, 2), 45); // sum of all

        // Prefix sums
        assert_eq!(tree.prefix_sum(0, 0), 1);
        assert_eq!(tree.prefix_sum(0, 2), 6); // 1 + 2 + 3
        assert_eq!(tree.prefix_sum(1, 1), 12); // 1 + 2 + 4 + 5
    }

    #[test]
    fn test_fenwick_2d_range_sum() {
        let mut tree: FenwickTree2D<i32> = FenwickTree2D::new(3, 3);

        // 1 2 3
        // 4 5 6
        // 7 8 9
        tree.add(0, 0, 1);
        tree.add(0, 1, 2);
        tree.add(0, 2, 3);
        tree.add(1, 0, 4);
        tree.add(1, 1, 5);
        tree.add(1, 2, 6);
        tree.add(2, 0, 7);
        tree.add(2, 1, 8);
        tree.add(2, 2, 9);

        // Center element only
        assert_eq!(tree.range_sum(1, 1, 1, 1), 5);

        // Bottom-right 2x2
        assert_eq!(tree.range_sum(1, 1, 2, 2), 28); // 5 + 6 + 8 + 9

        // Full matrix
        assert_eq!(tree.range_sum(0, 0, 2, 2), 45);
    }

    #[test]
    fn test_fenwick_large_array() {
        let arr: Vec<i64> = (1..=1000).collect();
        let tree = FenwickTree::from_slice(&arr);

        // Sum of 1 to 1000
        assert_eq!(tree.prefix_sum(999), 500500);

        // Range sum of indices 99 to 199 = values 100 to 200 (101 elements)
        // Sum = (100 + 200) * 101 / 2 = 15150
        assert_eq!(tree.range_sum(99, 199), 15150);
    }

    #[test]
    fn test_fenwick_negative_numbers() {
        let arr = vec![-5, 3, -2, 7, -1];
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.prefix_sum(4), 2); // -5 + 3 - 2 + 7 - 1
        assert_eq!(tree.range_sum(1, 3), 8); // 3 - 2 + 7
    }

    #[test]
    fn test_fenwick_floats() {
        let arr: Vec<f64> = vec![1.5, 2.5, 3.5];
        let tree = FenwickTree::from_slice(&arr);

        assert!((tree.prefix_sum(2) - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_kth() {
        // Frequency tree: index i stores count of element i
        let frequencies = vec![2, 0, 3, 1, 0]; // 2 zeros, 0 ones, 3 twos, 1 three
        let tree = FenwickTree::from_slice(&frequencies);

        // Find 1st element (0-indexed k=1)
        assert_eq!(find_kth(&tree, 1), Some(0)); // First element is 0

        // Find 3rd element (k=3)
        assert_eq!(find_kth(&tree, 3), Some(2)); // Third element is 2

        // Find 6th element (k=6)
        assert_eq!(find_kth(&tree, 6), Some(3)); // 2+0+3+1 = 6, last is 3
    }

    #[test]
    fn test_fenwick_power_of_two_size() {
        let arr: Vec<i32> = (1..=8).collect();
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.prefix_sum(7), 36);
        assert_eq!(tree.range_sum(2, 5), 18); // 3 + 4 + 5 + 6
    }

    #[test]
    fn test_fenwick_non_power_of_two_size() {
        let arr: Vec<i32> = (1..=7).collect();
        let tree = FenwickTree::from_slice(&arr);

        assert_eq!(tree.prefix_sum(6), 28);
        assert_eq!(tree.range_sum(1, 5), 20); // 2 + 3 + 4 + 5 + 6
    }

    #[test]
    fn test_compare_with_naive() {
        let arr: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let mut tree = FenwickTree::from_slice(&arr);
        let mut naive = arr.clone();

        // Verify initial state
        for i in 0..arr.len() {
            let naive_sum: i32 = naive[..=i].iter().sum();
            assert_eq!(tree.prefix_sum(i), naive_sum);
        }

        // Update and verify
        tree.add(3, 10);
        naive[3] += 10;

        for i in 0..arr.len() {
            let naive_sum: i32 = naive[..=i].iter().sum();
            assert_eq!(tree.prefix_sum(i), naive_sum);
        }
    }
}
