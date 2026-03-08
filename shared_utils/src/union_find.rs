// Union-Find (Disjoint Set Union) - Efficient set operations
//
// A data structure that tracks elements partitioned into disjoint sets.
// Supports near-constant time union and find operations.
//
// Time Complexity (with path compression and union by rank):
// - Find: O(α(n)) amortized, where α is inverse Ackermann function
// - Union: O(α(n)) amortized
// - In practice, α(n) ≤ 4 for any reasonable n
//
// Space Complexity: O(n)
//
// Common use cases:
// - Kruskal's minimum spanning tree algorithm
// - Detecting cycles in undirected graphs
// - Finding connected components
// - Network connectivity queries

use std::cmp::Ordering;

/// Union-Find with path compression and union by rank
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    count: usize, // Number of disjoint sets
}

impl UnionFind {
    /// Create a new Union-Find structure with `size` elements.
    /// Initially, each element is in its own singleton set.
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
            count: size,
        }
    }

    /// Find the representative (root) of the set containing `x`.
    /// Uses path compression for efficiency.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`.
    /// Returns true if a union was performed (they were in different sets).
    /// Returns false if `x` and `y` were already in the same set.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            Ordering::Greater => self.parent[root_y] = root_x,
            Ordering::Less => self.parent[root_x] = root_y,
            Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        self.count -= 1;
        true
    }

    /// Check if `x` and `y` are in the same set.
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the number of disjoint sets.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Get the size of the set containing `x`.
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.parent
            .iter()
            .enumerate()
            .filter(|&(i, _)| {
                // Need to find root without mutating
                let mut current = i;
                while self.parent[current] != current {
                    current = self.parent[current];
                }
                current == root
            })
            .count()
    }
}

/// Union-Find with size tracking (union by size instead of rank)
pub struct WeightedUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    count: usize,
}

impl WeightedUnionFind {
    /// Create a new weighted Union-Find with `n` elements.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            count: n,
        }
    }

    /// Find the root of element `x` with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union two sets, attaching smaller tree to larger.
    /// Returns true if union happened.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by size - attach smaller to larger
        if self.size[root_x] < self.size[root_y] {
            self.parent[root_x] = root_y;
            self.size[root_y] += self.size[root_x];
        } else {
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
        }

        self.count -= 1;
        true
    }

    /// Check if two elements are connected.
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the size of the set containing `x`.
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Get the number of disjoint sets.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get total number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);

        assert_eq!(uf.count(), 5);

        uf.union(0, 1);
        assert_eq!(uf.count(), 4);
        assert!(uf.connected(0, 1));
        assert!(!uf.connected(0, 2));

        uf.union(2, 3);
        assert_eq!(uf.count(), 3);
        assert!(uf.connected(2, 3));

        uf.union(1, 3);
        assert_eq!(uf.count(), 2);
        assert!(uf.connected(0, 2));
        assert!(uf.connected(1, 3));
    }

    #[test]
    fn test_union_find_same_set() {
        let mut uf = UnionFind::new(3);

        assert!(uf.union(0, 1)); // First union succeeds
        assert!(!uf.union(0, 1)); // Same set, returns false
        assert!(!uf.union(1, 0)); // Same set, returns false
    }

    #[test]
    fn test_union_find_all_connected() {
        let mut uf = UnionFind::new(4);

        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);

        assert_eq!(uf.count(), 1);

        // All should be connected
        for i in 0..4 {
            for j in 0..4 {
                assert!(uf.connected(i, j));
            }
        }
    }

    #[test]
    fn test_union_find_find_consistency() {
        let mut uf = UnionFind::new(10);

        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(4, 5);
        uf.union(0, 2);
        uf.union(4, 6);

        // Elements in same set should have same root
        let root_01 = uf.find(0);
        assert_eq!(uf.find(1), root_01);
        assert_eq!(uf.find(2), root_01);
        assert_eq!(uf.find(3), root_01);

        let root_456 = uf.find(4);
        assert_eq!(uf.find(5), root_456);
        assert_eq!(uf.find(6), root_456);
    }

    #[test]
    fn test_union_find_len() {
        let uf = UnionFind::new(100);
        assert_eq!(uf.len(), 100);
        assert!(!uf.is_empty());

        let empty_uf = UnionFind::new(0);
        assert!(empty_uf.is_empty());
    }

    // Weighted Union-Find tests
    #[test]
    fn test_weighted_uf_basic() {
        let mut uf = WeightedUnionFind::new(5);

        assert_eq!(uf.count(), 5);
        assert_eq!(uf.set_size(0), 1);

        uf.union(0, 1);
        assert_eq!(uf.set_size(0), 2);
        assert_eq!(uf.set_size(1), 2);

        uf.union(2, 3);
        uf.union(0, 2);

        assert_eq!(uf.set_size(0), 4);
        assert_eq!(uf.count(), 2);
    }

    #[test]
    fn test_weighted_uf_connected() {
        let mut uf = WeightedUnionFind::new(6);

        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(4, 5);

        assert!(uf.connected(0, 1));
        assert!(!uf.connected(0, 2));

        uf.union(0, 2);
        assert!(uf.connected(0, 3));
        assert!(uf.connected(1, 2));
    }

    #[test]
    fn test_weighted_uf_large() {
        let mut uf = WeightedUnionFind::new(1000);

        // Connect all even numbers
        for i in (0..1000).step_by(2) {
            if i + 2 < 1000 {
                uf.union(i, i + 2);
            }
        }

        // Connect all odd numbers
        for i in (1..1000).step_by(2) {
            if i + 2 < 1000 {
                uf.union(i, i + 2);
            }
        }

        assert_eq!(uf.count(), 2);
        assert_eq!(uf.set_size(0), 500);
        assert_eq!(uf.set_size(1), 500);

        // Connect evens and odds
        uf.union(0, 1);
        assert_eq!(uf.count(), 1);
        assert_eq!(uf.set_size(0), 1000);
    }
}
