// Segment Tree - Efficient range queries and point updates
//
// A segment tree is a binary tree where each node represents a segment (range)
// of the underlying array. The root represents the entire array, and each leaf
// represents a single element. Internal nodes store aggregate values over their
// ranges, enabling efficient range queries.
//
// Structure (for array of 8 elements):
//
//                    [0,7]              <- root (full range)
//                   /     \
//              [0,3]       [4,7]
//             /    \       /    \
//          [0,1]  [2,3]  [4,5]  [6,7]
//          /  \   /  \   /  \   /  \
//         0   1  2   3  4   5  6   7   <- leaves (individual elements)
//
// Key properties:
// - Height: O(log n)
// - Space: O(n) - stored as array of size 2n or 4n
// - Build: O(n)
// - Point update: O(log n)
// - Range query: O(log n)
//
// Common operations:
// - Range sum, Range min/max, Range GCD
// - Can extend to range updates with lazy propagation
//
// Array representation:
// - Node i has children at 2i and 2i+1
// - Node i has parent at i/2
// - Leaves start at index n (for 1-indexed tree)

use std::ops::Add;

// Generic segment tree supporting any associative operation
pub struct SegmentTree<T, F> {
    n: usize,     // Size of original array
    tree: Vec<T>, // 1-indexed tree array (size 2n)
    identity: T,  // Identity element for the operation
    op: F,        // Associative operation
}

impl<T, F> SegmentTree<T, F>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    // Build segment tree from array
    // Time: O(n), Space: O(n)
    pub fn new(arr: &[T], identity: T, op: F) -> Self {
        let n = arr.len();
        if n == 0 {
            return Self {
                n: 0,
                tree: vec![identity.clone()],
                identity,
                op,
            };
        }

        // Tree array: indices 0..n are unused/internal, n..2n are leaves
        let mut tree = vec![identity.clone(); 2 * n];

        // Copy array to leaves (indices n to 2n-1)
        for (i, val) in arr.iter().enumerate() {
            tree[n + i] = val.clone();
        }

        // Build internal nodes bottom-up
        for i in (1..n).rev() {
            tree[i] = op(&tree[2 * i], &tree[2 * i + 1]);
        }

        Self {
            n,
            tree,
            identity,
            op,
        }
    }

    // Point update: set arr[idx] = val
    // Time: O(log n)
    pub fn update(&mut self, idx: usize, val: T) {
        assert!(idx < self.n, "Index out of bounds");

        // Update leaf
        let mut i = self.n + idx;
        self.tree[i] = val;

        // Update ancestors
        i /= 2;
        while i >= 1 {
            self.tree[i] = (self.op)(&self.tree[2 * i], &self.tree[2 * i + 1]);
            i /= 2;
        }
    }

    // Range query: compute op over [left, right)
    // Time: O(log n)
    pub fn query(&self, left: usize, right: usize) -> T {
        assert!(left <= right && right <= self.n, "Invalid range");

        if left == right {
            return self.identity.clone();
        }

        let mut result_left = self.identity.clone();
        let mut result_right = self.identity.clone();

        let mut l = self.n + left;
        let mut r = self.n + right;

        while l < r {
            // If l is right child, include it and move to parent's right sibling
            if l % 2 == 1 {
                result_left = (self.op)(&result_left, &self.tree[l]);
                l += 1;
            }
            // If r is right child, include left sibling
            if r % 2 == 1 {
                r -= 1;
                result_right = (self.op)(&self.tree[r], &result_right);
            }
            l /= 2;
            r /= 2;
        }

        (self.op)(&result_left, &result_right)
    }

    // Get value at index
    pub fn get(&self, idx: usize) -> T {
        assert!(idx < self.n, "Index out of bounds");
        self.tree[self.n + idx].clone()
    }

    // Get size of underlying array
    pub fn len(&self) -> usize {
        self.n
    }

    // Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

// Convenience constructors for common use cases

// Sum segment tree
pub struct SumSegmentTree<T> {
    inner: SegmentTree<T, fn(&T, &T) -> T>,
}

impl<T> SumSegmentTree<T>
where
    T: Clone + Add<Output = T> + Default,
{
    pub fn new(arr: &[T]) -> Self {
        fn add<T: Clone + Add<Output = T>>(a: &T, b: &T) -> T {
            a.clone() + b.clone()
        }
        Self {
            inner: SegmentTree::new(arr, T::default(), add),
        }
    }

    pub fn update(&mut self, idx: usize, val: T) {
        self.inner.update(idx, val);
    }

    pub fn query(&self, left: usize, right: usize) -> T {
        self.inner.query(left, right)
    }

    pub fn get(&self, idx: usize) -> T {
        self.inner.get(idx)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// Min segment tree
pub struct MinSegmentTree<T> {
    inner: SegmentTree<T, fn(&T, &T) -> T>,
}

impl<T> MinSegmentTree<T>
where
    T: Clone + Ord + MaxValue,
{
    pub fn new(arr: &[T]) -> Self {
        fn min<T: Clone + Ord>(a: &T, b: &T) -> T {
            if a <= b { a.clone() } else { b.clone() }
        }
        Self {
            inner: SegmentTree::new(arr, T::max_value(), min),
        }
    }

    pub fn update(&mut self, idx: usize, val: T) {
        self.inner.update(idx, val);
    }

    pub fn query(&self, left: usize, right: usize) -> T {
        self.inner.query(left, right)
    }

    pub fn get(&self, idx: usize) -> T {
        self.inner.get(idx)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// Max segment tree
pub struct MaxSegmentTree<T> {
    inner: SegmentTree<T, fn(&T, &T) -> T>,
}

impl<T> MaxSegmentTree<T>
where
    T: Clone + Ord + MinValue,
{
    pub fn new(arr: &[T]) -> Self {
        fn max<T: Clone + Ord>(a: &T, b: &T) -> T {
            if a >= b { a.clone() } else { b.clone() }
        }
        Self {
            inner: SegmentTree::new(arr, T::min_value(), max),
        }
    }

    pub fn update(&mut self, idx: usize, val: T) {
        self.inner.update(idx, val);
    }

    pub fn query(&self, left: usize, right: usize) -> T {
        self.inner.query(left, right)
    }

    pub fn get(&self, idx: usize) -> T {
        self.inner.get(idx)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// Traits for identity elements
pub trait MaxValue {
    fn max_value() -> Self;
}

pub trait MinValue {
    fn min_value() -> Self;
}

// Implement for common types
macro_rules! impl_max_min_value {
    ($($t:ty),*) => {
        $(
            impl MaxValue for $t {
                fn max_value() -> Self {
                    <$t>::MAX
                }
            }
            impl MinValue for $t {
                fn min_value() -> Self {
                    <$t>::MIN
                }
            }
        )*
    };
}

impl_max_min_value!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

// Lazy segment tree with range updates
// Supports both range queries and range updates
pub struct LazySegmentTree<T, F, U> {
    n: usize,
    tree: Vec<T>,
    lazy: Vec<Option<T>>,
    identity: T,
    op: F,
    apply: U, // How to apply a lazy value to a node
}

impl<T, F, U> LazySegmentTree<T, F, U>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
    U: Fn(&T, &T, usize) -> T, // (node_value, lazy_value, range_size) -> new_value
{
    pub fn new(arr: &[T], identity: T, op: F, apply: U) -> Self {
        let n = arr.len();
        if n == 0 {
            return Self {
                n: 0,
                tree: vec![identity.clone()],
                lazy: vec![],
                identity,
                op,
                apply,
            };
        }

        let size = 4 * n; // Need more space for recursive structure
        let mut tree = vec![identity.clone(); size];
        let lazy = vec![None; size];

        Self::build(&mut tree, arr, 1, 0, n - 1, &identity, &op);

        Self {
            n,
            tree,
            lazy,
            identity,
            op,
            apply,
        }
    }

    fn build(
        tree: &mut [T],
        arr: &[T],
        node: usize,
        start: usize,
        end: usize,
        _identity: &T,
        op: &F,
    ) where
        F: Fn(&T, &T) -> T,
        T: Clone,
    {
        if start == end {
            tree[node] = arr[start].clone();
        } else {
            let mid = start + (end - start) / 2;
            Self::build(tree, arr, 2 * node, start, mid, _identity, op);
            Self::build(tree, arr, 2 * node + 1, mid + 1, end, _identity, op);
            tree[node] = op(&tree[2 * node], &tree[2 * node + 1]);
        }
    }

    // Push lazy value down to children
    fn push_down(&mut self, node: usize, start: usize, end: usize) {
        if let Some(lazy_val) = self.lazy[node].take() {
            let mid = start + (end - start) / 2;
            let left_size = mid - start + 1;
            let right_size = end - mid;

            // Apply to left child
            self.tree[2 * node] = (self.apply)(&self.tree[2 * node], &lazy_val, left_size);
            self.lazy[2 * node] = Some(lazy_val.clone());

            // Apply to right child
            self.tree[2 * node + 1] = (self.apply)(&self.tree[2 * node + 1], &lazy_val, right_size);
            self.lazy[2 * node + 1] = Some(lazy_val);
        }
    }

    // Range update: apply value to all elements in [l, r]
    pub fn update_range(&mut self, l: usize, r: usize, val: T) {
        assert!(l <= r && r < self.n, "Invalid range");
        self.update_range_impl(1, 0, self.n - 1, l, r, val);
    }

    fn update_range_impl(
        &mut self,
        node: usize,
        start: usize,
        end: usize,
        l: usize,
        r: usize,
        val: T,
    ) {
        if r < start || end < l {
            return; // Out of range
        }

        if l <= start && end <= r {
            // Fully covered
            let size = end - start + 1;
            self.tree[node] = (self.apply)(&self.tree[node], &val, size);
            self.lazy[node] = Some(val);
            return;
        }

        self.push_down(node, start, end);

        let mid = start + (end - start) / 2;
        self.update_range_impl(2 * node, start, mid, l, r, val.clone());
        self.update_range_impl(2 * node + 1, mid + 1, end, l, r, val);
        self.tree[node] = (self.op)(&self.tree[2 * node], &self.tree[2 * node + 1]);
    }

    // Range query
    pub fn query(&mut self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "Invalid range");
        self.query_impl(1, 0, self.n - 1, l, r)
    }

    fn query_impl(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize) -> T {
        if r < start || end < l {
            return self.identity.clone();
        }

        if l <= start && end <= r {
            return self.tree[node].clone();
        }

        self.push_down(node, start, end);

        let mid = start + (end - start) / 2;
        let left_result = self.query_impl(2 * node, start, mid, l, r);
        let right_result = self.query_impl(2 * node + 1, mid + 1, end, l, r);
        (self.op)(&left_result, &right_result)
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

// Segment tree with index tracking (for range min/max queries that need position)
pub struct SegmentTreeWithIndex<T, F> {
    n: usize,
    tree: Vec<(T, usize)>, // (value, index)
    identity: T,
    op: F,
}

impl<T, F> SegmentTreeWithIndex<T, F>
where
    T: Clone,
    F: Fn(&T, &T) -> bool, // Returns true if first arg is "better"
{
    pub fn new(arr: &[T], identity: T, op: F) -> Self {
        let n = arr.len();
        if n == 0 {
            return Self {
                n: 0,
                tree: vec![(identity.clone(), 0)],
                identity,
                op,
            };
        }

        let mut tree = vec![(identity.clone(), 0); 2 * n];

        // Initialize leaves
        for (i, val) in arr.iter().enumerate() {
            tree[n + i] = (val.clone(), i);
        }

        // Build internal nodes
        for i in (1..n).rev() {
            let left = &tree[2 * i];
            let right = &tree[2 * i + 1];
            tree[i] = if op(&left.0, &right.0) {
                left.clone()
            } else {
                right.clone()
            };
        }

        Self {
            n,
            tree,
            identity,
            op,
        }
    }

    pub fn update(&mut self, idx: usize, val: T) {
        assert!(idx < self.n, "Index out of bounds");

        let mut i = self.n + idx;
        self.tree[i] = (val, idx);

        i /= 2;
        while i >= 1 {
            let left = &self.tree[2 * i];
            let right = &self.tree[2 * i + 1];
            self.tree[i] = if (self.op)(&left.0, &right.0) {
                left.clone()
            } else {
                right.clone()
            };
            i /= 2;
        }
    }

    // Returns (value, index) for the "best" element in range
    pub fn query(&self, left: usize, right: usize) -> (T, usize) {
        assert!(left <= right && right <= self.n, "Invalid range");

        if left == right {
            return (self.identity.clone(), 0);
        }

        let mut result_left: Option<(T, usize)> = None;
        let mut result_right: Option<(T, usize)> = None;

        let mut l = self.n + left;
        let mut r = self.n + right;

        while l < r {
            if l % 2 == 1 {
                result_left = self.combine_opt(result_left, Some(self.tree[l].clone()));
                l += 1;
            }
            if r % 2 == 1 {
                r -= 1;
                result_right = self.combine_opt(Some(self.tree[r].clone()), result_right);
            }
            l /= 2;
            r /= 2;
        }

        self.combine_opt(result_left, result_right)
            .unwrap_or((self.identity.clone(), 0))
    }

    fn combine_opt(&self, a: Option<(T, usize)>, b: Option<(T, usize)>) -> Option<(T, usize)> {
        match (a, b) {
            (None, None) => None,
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (Some(x), Some(y)) => {
                if (self.op)(&x.0, &y.0) {
                    Some(x)
                } else {
                    Some(y)
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_segment_tree_basic() {
        let arr = vec![1, 3, 5, 7, 9, 11];
        let tree = SumSegmentTree::new(&arr);

        assert_eq!(tree.query(0, 6), 36); // sum of all
        assert_eq!(tree.query(0, 3), 9); // 1 + 3 + 5
        assert_eq!(tree.query(2, 5), 21); // 5 + 7 + 9
        assert_eq!(tree.query(1, 2), 3); // just 3
    }

    #[test]
    fn test_sum_segment_tree_update() {
        let arr = vec![1, 3, 5, 7, 9, 11];
        let mut tree = SumSegmentTree::new(&arr);

        tree.update(2, 10); // Change 5 to 10
        assert_eq!(tree.query(0, 6), 41); // 36 - 5 + 10
        assert_eq!(tree.query(0, 3), 14); // 1 + 3 + 10
        assert_eq!(tree.get(2), 10);
    }

    #[test]
    fn test_min_segment_tree() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let tree = MinSegmentTree::new(&arr);

        assert_eq!(tree.query(0, 6), 1);
        assert_eq!(tree.query(0, 3), 2);
        assert_eq!(tree.query(3, 6), 1);
        assert_eq!(tree.query(4, 6), 3);
    }

    #[test]
    fn test_min_segment_tree_update() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let mut tree = MinSegmentTree::new(&arr);

        tree.update(3, 10); // Change min element
        assert_eq!(tree.query(0, 6), 2); // New min is 2
        assert_eq!(tree.query(3, 6), 3); // Min in [3,6) is now 3
    }

    #[test]
    fn test_max_segment_tree() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let tree = MaxSegmentTree::new(&arr);

        assert_eq!(tree.query(0, 6), 9);
        assert_eq!(tree.query(0, 3), 8);
        assert_eq!(tree.query(0, 2), 5);
    }

    #[test]
    fn test_generic_segment_tree_product() {
        let arr = vec![2, 3, 4, 5];
        let tree = SegmentTree::new(&arr, 1, |a, b| a * b);

        assert_eq!(tree.query(0, 4), 120); // 2*3*4*5
        assert_eq!(tree.query(0, 2), 6); // 2*3
        assert_eq!(tree.query(1, 3), 12); // 3*4
    }

    #[test]
    fn test_generic_segment_tree_gcd() {
        fn gcd(a: &i32, b: &i32) -> i32 {
            let (mut a, mut b) = (*a, *b);
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            a.abs()
        }

        let arr = vec![12, 18, 24, 30];
        let tree = SegmentTree::new(&arr, 0, gcd);

        assert_eq!(tree.query(0, 4), 6); // GCD of all
        assert_eq!(tree.query(0, 2), 6); // GCD(12, 18)
        assert_eq!(tree.query(2, 4), 6); // GCD(24, 30)
    }

    #[test]
    fn test_segment_tree_empty() {
        let arr: Vec<i32> = vec![];
        let tree = SumSegmentTree::new(&arr);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_segment_tree_single() {
        let arr = vec![42];
        let tree = SumSegmentTree::new(&arr);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.query(0, 1), 42);
        assert_eq!(tree.get(0), 42);
    }

    #[test]
    fn test_segment_tree_power_of_two() {
        let arr = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tree = SumSegmentTree::new(&arr);

        assert_eq!(tree.query(0, 8), 36);
        assert_eq!(tree.query(0, 4), 10);
        assert_eq!(tree.query(4, 8), 26);
    }

    #[test]
    fn test_lazy_segment_tree_range_add() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut tree = LazySegmentTree::new(
            &arr,
            0,
            |a, b| a + b,                               // Sum operation
            |val, lazy, size| val + lazy * size as i32, // Range add
        );

        assert_eq!(tree.query(0, 4), 15); // 1+2+3+4+5

        tree.update_range(1, 3, 10); // Add 10 to indices 1,2,3
        assert_eq!(tree.query(0, 4), 45); // 1+12+13+14+5
        assert_eq!(tree.query(1, 3), 39); // 12+13+14
    }

    #[test]
    fn test_lazy_segment_tree_range_set() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut tree = LazySegmentTree::new(
            &arr,
            0,
            |a, b| a + b,                          // Sum operation
            |_val, lazy, size| lazy * size as i32, // Range set
        );

        tree.update_range(1, 3, 10); // Set indices 1,2,3 to 10
        assert_eq!(tree.query(1, 3), 30); // 10+10+10
    }

    #[test]
    fn test_segment_tree_with_index_min() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let tree = SegmentTreeWithIndex::new(&arr, i32::MAX, |a, b| a < b);

        let (val, idx) = tree.query(0, 6);
        assert_eq!(val, 1);
        assert_eq!(idx, 3);

        let (val, idx) = tree.query(0, 3);
        assert_eq!(val, 2);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_segment_tree_with_index_max() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let tree = SegmentTreeWithIndex::new(&arr, i32::MIN, |a, b| a > b);

        let (val, idx) = tree.query(0, 6);
        assert_eq!(val, 9);
        assert_eq!(idx, 4);

        let (val, idx) = tree.query(0, 3);
        assert_eq!(val, 8);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_segment_tree_with_index_update() {
        let arr = vec![5, 2, 8, 1, 9, 3];
        let mut tree = SegmentTreeWithIndex::new(&arr, i32::MAX, |a, b| a < b);

        tree.update(3, 100); // Change minimum element to 100

        let (val, idx) = tree.query(0, 6);
        assert_eq!(val, 2);
        assert_eq!(idx, 1); // New minimum is at index 1
    }

    #[test]
    fn test_sum_tree_adjacent_queries() {
        let arr = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tree = SumSegmentTree::new(&arr);

        // Non-overlapping ranges should sum to total
        assert_eq!(tree.query(0, 4) + tree.query(4, 8), tree.query(0, 8));
        assert_eq!(tree.query(0, 2) + tree.query(2, 4), tree.query(0, 4));
    }

    #[test]
    fn test_all_single_element_queries() {
        let arr = vec![1, 2, 3, 4, 5];
        let tree = SumSegmentTree::new(&arr);

        for i in 0..arr.len() {
            assert_eq!(tree.query(i, i + 1), arr[i]);
        }
    }

    #[test]
    fn test_empty_range_query() {
        let arr = vec![1, 2, 3, 4, 5];
        let tree = SumSegmentTree::new(&arr);

        assert_eq!(tree.query(2, 2), 0); // Empty range returns identity
        assert_eq!(tree.query(0, 0), 0);
    }

    #[test]
    fn test_large_array() {
        let arr: Vec<i64> = (1..=1000).collect();
        let tree = SumSegmentTree::new(&arr);

        // Sum of 1 to 1000 = 1000 * 1001 / 2 = 500500
        assert_eq!(tree.query(0, 1000), 500500);

        // Sum of 100 to 200 (exclusive)
        let expected: i64 = (100..200).sum();
        assert_eq!(tree.query(99, 199), expected);
    }

    #[test]
    fn test_multiple_updates() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut tree = SumSegmentTree::new(&arr);

        tree.update(0, 10);
        tree.update(4, 50);
        tree.update(2, 30);

        assert_eq!(tree.query(0, 5), 10 + 2 + 30 + 4 + 50);
    }
}
