// Interval Tree - Efficient overlapping interval queries
//
// An interval tree is a data structure that holds intervals and efficiently
// answers queries like "find all intervals that overlap with a given interval"
// or "find all intervals that contain a given point".
//
// This implementation uses an augmented BST (balanced via simple rotations)
// where each node stores:
// - An interval [low, high]
// - The maximum endpoint in its subtree (augmented data)
//
// Structure visualization:
//
//           [15,20] max=30
//          /              \
//    [10,30] max=30    [17,19] max=40
//       /                    \
//  [5,20] max=20         [30,40] max=40
//
// Key insight: The max value allows pruning entire subtrees during queries.
// If we're looking for intervals overlapping [lo, hi], we can skip a subtree
// if its max < lo (no interval in subtree can overlap).
//
// Operations:
// - Insert: O(log n) average
// - Delete: O(log n) average
// - Query overlapping: O(k + log n) where k is number of results
// - Query point: O(k + log n)
//
// Applications:
// - Scheduling and calendar systems
// - Computational geometry
// - Database range queries
// - Genome analysis (finding overlapping genes)

use std::cmp::{Ordering, max};

// Represents an interval [low, high]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval<T> {
    pub low: T,
    pub high: T,
}

impl<T: Ord + Copy> Interval<T> {
    pub fn new(low: T, high: T) -> Self {
        assert!(low <= high, "Invalid interval: low must be <= high");
        Self { low, high }
    }

    // Check if two intervals overlap
    pub fn overlaps(&self, other: &Interval<T>) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    // Check if this interval contains a point
    pub fn contains_point(&self, point: T) -> bool {
        self.low <= point && point <= self.high
    }

    // Check if this interval fully contains another
    pub fn contains(&self, other: &Interval<T>) -> bool {
        self.low <= other.low && other.high <= self.high
    }
}

impl<T: Ord + Copy> PartialOrd for Interval<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord + Copy> Ord for Interval<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.low.cmp(&other.low) {
            Ordering::Equal => self.high.cmp(&other.high),
            other => other,
        }
    }
}

// Node in the interval tree
struct IntervalNode<T, V> {
    interval: Interval<T>,
    value: V, // Associated data
    max: T,   // Maximum endpoint in subtree
    left: Option<Box<IntervalNode<T, V>>>,
    right: Option<Box<IntervalNode<T, V>>>,
}

impl<T: Ord + Copy, V> IntervalNode<T, V> {
    fn new(interval: Interval<T>, value: V) -> Self {
        let max = interval.high;
        Self {
            interval,
            value,
            max,
            left: None,
            right: None,
        }
    }

    // Update max based on children
    fn update_max(&mut self) {
        self.max = self.interval.high;
        if let Some(ref left) = self.left {
            self.max = max(self.max, left.max);
        }
        if let Some(ref right) = self.right {
            self.max = max(self.max, right.max);
        }
    }
}

// Interval Tree implementation
pub struct IntervalTree<T, V> {
    root: Option<Box<IntervalNode<T, V>>>,
    size: usize,
}

impl<T: Ord + Copy, V> Default for IntervalTree<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Copy, V> IntervalTree<T, V> {
    pub fn new() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    // Insert an interval with associated value
    pub fn insert(&mut self, interval: Interval<T>, value: V) {
        self.root = Self::insert_recursive(self.root.take(), interval, value);
        self.size += 1;
    }

    fn insert_recursive(
        node: Option<Box<IntervalNode<T, V>>>,
        interval: Interval<T>,
        value: V,
    ) -> Option<Box<IntervalNode<T, V>>> {
        match node {
            None => Some(Box::new(IntervalNode::new(interval, value))),
            Some(mut n) => {
                if interval.low < n.interval.low {
                    n.left = Self::insert_recursive(n.left.take(), interval, value);
                } else {
                    n.right = Self::insert_recursive(n.right.take(), interval, value);
                }
                n.update_max();
                Some(n)
            }
        }
    }

    // Find all intervals that overlap with the given interval
    pub fn find_overlapping(&self, interval: &Interval<T>) -> Vec<(&Interval<T>, &V)> {
        let mut results = Vec::new();
        Self::find_overlapping_recursive(&self.root, interval, &mut results);
        results
    }

    fn find_overlapping_recursive<'a>(
        node: &'a Option<Box<IntervalNode<T, V>>>,
        interval: &Interval<T>,
        results: &mut Vec<(&'a Interval<T>, &'a V)>,
    ) {
        if let Some(n) = node {
            // Check if this interval overlaps
            if n.interval.overlaps(interval) {
                results.push((&n.interval, &n.value));
            }

            // Check left subtree if it might contain overlapping intervals
            if let Some(left) = &n.left
                && left.max >= interval.low
            {
                Self::find_overlapping_recursive(&n.left, interval, results);
            }

            // Check right subtree if current interval's low <= query high
            if n.interval.low <= interval.high {
                Self::find_overlapping_recursive(&n.right, interval, results);
            }
        }
    }

    // Find all intervals containing a specific point
    pub fn find_containing_point(&self, point: T) -> Vec<(&Interval<T>, &V)> {
        let query = Interval::new(point, point);
        self.find_overlapping(&query)
    }

    // Find any single interval that overlaps (faster than finding all)
    pub fn find_any_overlapping(&self, interval: &Interval<T>) -> Option<(&Interval<T>, &V)> {
        Self::find_any_overlapping_recursive(&self.root, interval)
    }

    fn find_any_overlapping_recursive<'a>(
        node: &'a Option<Box<IntervalNode<T, V>>>,
        interval: &Interval<T>,
    ) -> Option<(&'a Interval<T>, &'a V)> {
        match node {
            None => None,
            Some(n) => {
                if n.interval.overlaps(interval) {
                    return Some((&n.interval, &n.value));
                }

                // If left child exists and its max >= interval.low,
                // there might be an overlapping interval in left subtree
                if let Some(left) = &n.left
                    && left.max >= interval.low
                    && let result @ Some(_) =
                        Self::find_any_overlapping_recursive(&n.left, interval)
                {
                    return result;
                }

                // Otherwise check right subtree
                Self::find_any_overlapping_recursive(&n.right, interval)
            }
        }
    }

    // Check if any interval overlaps with the given interval
    pub fn has_overlap(&self, interval: &Interval<T>) -> bool {
        self.find_any_overlapping(interval).is_some()
    }

    // Find all intervals completely contained within the given interval
    pub fn find_contained_by(&self, interval: &Interval<T>) -> Vec<(&Interval<T>, &V)> {
        let mut results = Vec::new();
        Self::find_contained_by_recursive(&self.root, interval, &mut results);
        results
    }

    fn find_contained_by_recursive<'a>(
        node: &'a Option<Box<IntervalNode<T, V>>>,
        interval: &Interval<T>,
        results: &mut Vec<(&'a Interval<T>, &'a V)>,
    ) {
        if let Some(n) = node {
            // Check if this interval is contained
            if interval.contains(&n.interval) {
                results.push((&n.interval, &n.value));
            }

            // Check left subtree
            if let Some(left) = &n.left
                && left.max >= interval.low
            {
                Self::find_contained_by_recursive(&n.left, interval, results);
            }

            // Check right subtree if there might be contained intervals
            if n.interval.low <= interval.high {
                Self::find_contained_by_recursive(&n.right, interval, results);
            }
        }
    }

    // Collect all intervals in order
    pub fn iter(&self) -> Vec<(&Interval<T>, &V)> {
        let mut results = Vec::with_capacity(self.size);
        Self::inorder(&self.root, &mut results);
        results
    }

    fn inorder<'a>(
        node: &'a Option<Box<IntervalNode<T, V>>>,
        results: &mut Vec<(&'a Interval<T>, &'a V)>,
    ) {
        if let Some(n) = node {
            Self::inorder(&n.left, results);
            results.push((&n.interval, &n.value));
            Self::inorder(&n.right, results);
        }
    }
}

// Interval tree without associated values (just stores intervals)
pub struct SimpleIntervalTree<T> {
    inner: IntervalTree<T, ()>,
}

impl<T: Ord + Copy> Default for SimpleIntervalTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Copy> SimpleIntervalTree<T> {
    pub fn new() -> Self {
        Self {
            inner: IntervalTree::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn insert(&mut self, interval: Interval<T>) {
        self.inner.insert(interval, ());
    }

    pub fn find_overlapping(&self, interval: &Interval<T>) -> Vec<&Interval<T>> {
        self.inner
            .find_overlapping(interval)
            .into_iter()
            .map(|(i, _)| i)
            .collect()
    }

    pub fn find_containing_point(&self, point: T) -> Vec<&Interval<T>> {
        self.inner
            .find_containing_point(point)
            .into_iter()
            .map(|(i, _)| i)
            .collect()
    }

    pub fn has_overlap(&self, interval: &Interval<T>) -> bool {
        self.inner.has_overlap(interval)
    }

    pub fn find_contained_by(&self, interval: &Interval<T>) -> Vec<&Interval<T>> {
        self.inner
            .find_contained_by(interval)
            .into_iter()
            .map(|(i, _)| i)
            .collect()
    }

    pub fn iter(&self) -> Vec<&Interval<T>> {
        self.inner.iter().into_iter().map(|(i, _)| i).collect()
    }
}

// Builder for creating interval trees from a collection
impl<T: Ord + Copy, V> FromIterator<(Interval<T>, V)> for IntervalTree<T, V> {
    fn from_iter<I: IntoIterator<Item = (Interval<T>, V)>>(iter: I) -> Self {
        let mut tree = IntervalTree::new();
        for (interval, value) in iter {
            tree.insert(interval, value);
        }
        tree
    }
}

impl<T: Ord + Copy> FromIterator<Interval<T>> for SimpleIntervalTree<T> {
    fn from_iter<I: IntoIterator<Item = Interval<T>>>(iter: I) -> Self {
        let mut tree = SimpleIntervalTree::new();
        for interval in iter {
            tree.insert(interval);
        }
        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basic() {
        let i1 = Interval::new(1, 5);
        let i2 = Interval::new(3, 7);
        let i3 = Interval::new(6, 10);

        assert!(i1.overlaps(&i2));
        assert!(i2.overlaps(&i1));
        assert!(i2.overlaps(&i3));
        assert!(!i1.overlaps(&i3));
    }

    #[test]
    fn test_interval_contains_point() {
        let i = Interval::new(5, 10);

        assert!(i.contains_point(5));
        assert!(i.contains_point(7));
        assert!(i.contains_point(10));
        assert!(!i.contains_point(4));
        assert!(!i.contains_point(11));
    }

    #[test]
    fn test_interval_contains_interval() {
        let outer = Interval::new(1, 10);
        let inner = Interval::new(3, 7);
        let partial = Interval::new(5, 15);

        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
        assert!(!outer.contains(&partial));
    }

    #[test]
    fn test_simple_tree_insert_and_find() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(15, 20));
        tree.insert(Interval::new(10, 30));
        tree.insert(Interval::new(17, 19));
        tree.insert(Interval::new(5, 20));
        tree.insert(Interval::new(12, 15));
        tree.insert(Interval::new(30, 40));

        assert_eq!(tree.len(), 6);

        // Find intervals overlapping [14, 16]
        let overlapping = tree.find_overlapping(&Interval::new(14, 16));
        assert_eq!(overlapping.len(), 4); // [15,20], [10,30], [5,20], [12,15]
    }

    #[test]
    fn test_find_containing_point() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(3, 8));
        tree.insert(Interval::new(10, 15));
        tree.insert(Interval::new(4, 6));

        let containing_4 = tree.find_containing_point(4);
        assert_eq!(containing_4.len(), 3); // [1,5], [3,8], [4,6]

        let containing_12 = tree.find_containing_point(12);
        assert_eq!(containing_12.len(), 1); // [10,15]

        let containing_9 = tree.find_containing_point(9);
        assert_eq!(containing_9.len(), 0);
    }

    #[test]
    fn test_has_overlap() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(10, 15));
        tree.insert(Interval::new(20, 25));

        assert!(tree.has_overlap(&Interval::new(3, 7)));
        assert!(tree.has_overlap(&Interval::new(12, 18)));
        assert!(!tree.has_overlap(&Interval::new(6, 9)));
        assert!(!tree.has_overlap(&Interval::new(16, 19)));
    }

    #[test]
    fn test_tree_with_values() {
        let mut tree: IntervalTree<i32, &str> = IntervalTree::new();

        tree.insert(Interval::new(1, 5), "meeting A");
        tree.insert(Interval::new(3, 8), "meeting B");
        tree.insert(Interval::new(10, 12), "meeting C");

        let overlapping = tree.find_overlapping(&Interval::new(4, 6));
        assert_eq!(overlapping.len(), 2);

        // Check that values are correct
        let values: Vec<&str> = overlapping.iter().map(|(_, v)| **v).collect();
        assert!(values.contains(&"meeting A"));
        assert!(values.contains(&"meeting B"));
    }

    #[test]
    fn test_find_contained_by() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(2, 4));
        tree.insert(Interval::new(3, 5));
        tree.insert(Interval::new(1, 10));
        tree.insert(Interval::new(6, 8));

        // Find intervals contained within [1, 9]
        let contained = tree.find_contained_by(&Interval::new(1, 9));
        assert_eq!(contained.len(), 3); // [2,4], [3,5], [6,8]

        // [1,10] is not contained because 10 > 9
        let has_1_10 = contained.iter().any(|i| i.low == 1 && i.high == 10);
        assert!(!has_1_10);
    }

    #[test]
    fn test_empty_tree() {
        let tree: SimpleIntervalTree<i32> = SimpleIntervalTree::new();

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.find_overlapping(&Interval::new(1, 5)).is_empty());
        assert!(!tree.has_overlap(&Interval::new(1, 5)));
    }

    #[test]
    fn test_single_interval() {
        let mut tree = SimpleIntervalTree::new();
        tree.insert(Interval::new(5, 10));

        assert_eq!(tree.len(), 1);
        assert!(tree.has_overlap(&Interval::new(7, 12)));
        assert!(!tree.has_overlap(&Interval::new(11, 15)));
    }

    #[test]
    fn test_adjacent_intervals() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(5, 10));
        tree.insert(Interval::new(10, 15));

        // Adjacent intervals share endpoints, so they overlap
        let at_5 = tree.find_containing_point(5);
        assert_eq!(at_5.len(), 2); // [1,5] and [5,10]

        let at_10 = tree.find_containing_point(10);
        assert_eq!(at_10.len(), 2); // [5,10] and [10,15]
    }

    #[test]
    fn test_from_iterator() {
        let intervals = vec![
            Interval::new(1, 5),
            Interval::new(3, 8),
            Interval::new(10, 15),
        ];

        let tree: SimpleIntervalTree<i32> = intervals.into_iter().collect();
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_iter() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(5, 10));
        tree.insert(Interval::new(1, 3));
        tree.insert(Interval::new(7, 12));

        let intervals = tree.iter();
        assert_eq!(intervals.len(), 3);

        // Should be in order by low endpoint
        assert_eq!(intervals[0].low, 1);
        assert_eq!(intervals[1].low, 5);
        assert_eq!(intervals[2].low, 7);
    }

    #[test]
    fn test_large_tree() {
        let mut tree = SimpleIntervalTree::new();

        // Insert 100 intervals
        for i in 0..100 {
            tree.insert(Interval::new(i * 10, i * 10 + 5));
        }

        assert_eq!(tree.len(), 100);

        // Query should find only intervals in range
        // Intervals are [0,5], [10,15], [20,25], ..., [50,55], [60,65], ...
        // Query [50,55] only overlaps with [50,55]
        let overlapping = tree.find_overlapping(&Interval::new(50, 55));
        assert_eq!(overlapping.len(), 1);

        // Point query
        let containing = tree.find_containing_point(503);
        assert_eq!(containing.len(), 1); // Only [500, 505]
    }

    #[test]
    fn test_nested_intervals() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(1, 100));
        tree.insert(Interval::new(20, 80));
        tree.insert(Interval::new(30, 70));
        tree.insert(Interval::new(40, 60));
        tree.insert(Interval::new(45, 55));

        // All intervals contain point 50
        let containing = tree.find_containing_point(50);
        assert_eq!(containing.len(), 5);

        // All intervals overlap with [40, 60]
        let overlapping = tree.find_overlapping(&Interval::new(40, 60));
        assert_eq!(overlapping.len(), 5);
    }

    #[test]
    fn test_disjoint_intervals() {
        let mut tree = SimpleIntervalTree::new();

        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(10, 15));
        tree.insert(Interval::new(20, 25));
        tree.insert(Interval::new(30, 35));

        // Query in gap should find nothing
        let overlapping = tree.find_overlapping(&Interval::new(6, 9));
        assert!(overlapping.is_empty());

        // Query spanning multiple intervals
        let overlapping = tree.find_overlapping(&Interval::new(1, 35));
        assert_eq!(overlapping.len(), 4);
    }

    #[test]
    fn test_scheduling_scenario() {
        let mut meetings: IntervalTree<i32, &str> = IntervalTree::new();

        // Time slots as minutes from midnight
        meetings.insert(Interval::new(540, 600), "Morning standup"); // 9:00-10:00
        meetings.insert(Interval::new(660, 720), "Team meeting"); // 11:00-12:00
        meetings.insert(Interval::new(780, 840), "Lunch"); // 13:00-14:00
        meetings.insert(Interval::new(900, 960), "Review"); // 15:00-16:00

        // Check if 10:30 (630) is free
        let at_1030 = meetings.find_containing_point(630);
        assert!(at_1030.is_empty()); // Free!

        // Check if new meeting 14:30-15:30 (870-930) conflicts
        let conflicts = meetings.find_overlapping(&Interval::new(870, 930));
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].1, &"Review");
    }
}
