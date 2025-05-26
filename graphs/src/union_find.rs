use std::cmp::Ordering;

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    // Creates new structure for `size` elements.
    // Each element in it's component at first.
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(), // each it's own parent
            rank: vec![0; size],
        }
    }

    // Returns representative for `x`.
    // Path compression to speed up future finds.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    // Joins unions, containing `x` and `y`.
    // Returns true, if union happened.
    // If `x` and `y` already in the same component - returns false.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        match self.rank[root_x].cmp(&self.rank[root_y]) {
            Ordering::Greater => self.parent[root_y] = root_x,
            Ordering::Less => self.parent[root_x] = root_y,
            Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        true
    }
}
