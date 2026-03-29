// D* Lite Algorithm for Dynamic Pathfinding
//
// D* Lite is an incremental heuristic search algorithm that efficiently
// re-plans paths when the environment changes (e.g., obstacles appear or
// disappear). Unlike A* which must restart from scratch, D* Lite reuses
// previous search results for efficient replanning.
//
// Key concepts:
// - Incremental Search: Reuses previous computations when map changes
// - Reverse Search: Searches from goal to start, enabling efficient updates
// - Key-based Priority: Uses two-part keys for correct node ordering
// - Lazy Updates: Only updates affected nodes when changes occur
//
// Use cases:
// - Robot navigation with sensor updates
// - Real-time strategy games with fog of war
// - Dynamic routing in changing networks
//
// Complexity:
// - Initial search: O(n log n) similar to A*
// - Replanning: O(k log n) where k is affected nodes (often << n)
//
// References:
// - Koenig & Likhachev "D* Lite" (2002)
// - http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// Re-use grid types from JPS module
#[cfg(test)]
use crate::jps::ArrayGrid;
use crate::jps::{Direction, Grid, Position};

/// Two-part key for priority queue ordering in D* Lite
#[derive(Debug, Clone, Copy, PartialEq)]
struct Key {
    k1: f64,
    k2: f64,
}

impl Key {
    fn new(k1: f64, k2: f64) -> Self {
        Self { k1, k2 }
    }
}

impl Eq for Key {}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap ordering: smaller keys have higher priority
        match other.k1.partial_cmp(&self.k1) {
            Some(Ordering::Equal) | None => {
                other.k2.partial_cmp(&self.k2).unwrap_or(Ordering::Equal)
            }
            Some(ord) => ord,
        }
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// State in the priority queue
#[derive(Debug, Clone)]
struct DStarState {
    key: Key,
    position: Position,
}

impl Eq for DStarState {}

impl PartialEq for DStarState {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.position == other.position
    }
}

impl Ord for DStarState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for DStarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of D* Lite pathfinding
#[derive(Debug, Clone)]
pub struct DStarResult {
    /// The path from start to goal (if found)
    pub path: Option<Vec<Position>>,
    /// Total cost of the path
    pub cost: f64,
    /// Number of nodes expanded
    pub nodes_expanded: usize,
    /// Whether this was an initial search or replan
    pub is_replan: bool,
}

/// D* Lite pathfinder for dynamic environments
///
/// This implementation supports incremental replanning when the grid changes.
/// The algorithm searches from goal to start, which allows efficient updates
/// when obstacles are discovered near the current position.
pub struct DStarLite<G: Grid> {
    grid: G,
    start: Position,
    goal: Position,
    /// Allow diagonal movement
    allow_diagonal: bool,
    /// Cost multiplier for precision
    cost_scale: f64,

    // D* Lite internal state
    /// g-values (cost from goal)
    g: HashMap<Position, f64>,
    /// rhs-values (one-step lookahead)
    rhs: HashMap<Position, f64>,
    /// Priority queue
    open: BinaryHeap<DStarState>,
    /// Set of positions in the open list (for fast lookup)
    open_set: HashSet<Position>,
    /// Key modifier for maintaining consistency
    km: f64,
    /// Previous start position (for km updates)
    last_start: Position,
    /// Whether initial search has been done
    initialized: bool,
    /// Statistics
    nodes_expanded: usize,
}

impl<G: Grid> DStarLite<G> {
    pub fn new(grid: G, start: Position, goal: Position) -> Self {
        Self {
            grid,
            start,
            goal,
            allow_diagonal: true,
            cost_scale: 1.0,
            g: HashMap::new(),
            rhs: HashMap::new(),
            open: BinaryHeap::new(),
            open_set: HashSet::new(),
            km: 0.0,
            last_start: start,
            initialized: false,
            nodes_expanded: 0,
        }
    }

    pub fn with_diagonal(mut self, allow: bool) -> Self {
        self.allow_diagonal = allow;
        self
    }

    /// Get the current grid (immutable)
    pub fn grid(&self) -> &G {
        &self.grid
    }

    /// Get the current start position
    pub fn start(&self) -> Position {
        self.start
    }

    /// Get the goal position
    pub fn goal(&self) -> Position {
        self.goal
    }

    /// Initialize the algorithm (called once before first search)
    fn initialize(&mut self) {
        self.g.clear();
        self.rhs.clear();
        self.open.clear();
        self.open_set.clear();
        self.km = 0.0;

        // rhs(goal) = 0
        self.rhs.insert(self.goal, 0.0);

        // Insert goal into open list
        let key = self.calculate_key(self.goal);
        self.open.push(DStarState {
            key,
            position: self.goal,
        });
        self.open_set.insert(self.goal);

        self.initialized = true;
        self.nodes_expanded = 0;
    }

    /// Calculate the two-part key for a position
    fn calculate_key(&self, pos: Position) -> Key {
        let g_val = self.g.get(&pos).copied().unwrap_or(f64::INFINITY);
        let rhs_val = self.rhs.get(&pos).copied().unwrap_or(f64::INFINITY);
        let min_val = g_val.min(rhs_val);
        let h = self.heuristic(self.start, pos);

        Key::new(min_val + h + self.km, min_val)
    }

    /// Heuristic function (Euclidean distance)
    fn heuristic(&self, from: Position, to: Position) -> f64 {
        let dx = (to.x - from.x) as f64;
        let dy = (to.y - from.y) as f64;
        (dx * dx + dy * dy).sqrt() * self.cost_scale
    }

    /// Cost to move between adjacent positions
    fn cost(&self, from: Position, to: Position) -> f64 {
        if !self.grid.is_passable(from) || !self.grid.is_passable(to) {
            return f64::INFINITY;
        }

        let dx = (to.x - from.x).abs();
        let dy = (to.y - from.y).abs();

        if dx > 0 && dy > 0 {
            // Diagonal movement
            std::f64::consts::SQRT_2 * self.cost_scale
        } else {
            // Cardinal movement
            self.cost_scale
        }
    }

    /// Get predecessors of a position (cells that can reach it)
    fn predecessors(&self, pos: Position) -> Vec<Position> {
        self.neighbors(pos)
    }

    /// Get successors of a position (cells reachable from it)
    fn successors(&self, pos: Position) -> Vec<Position> {
        self.neighbors(pos)
    }

    /// Get valid neighbors
    fn neighbors(&self, pos: Position) -> Vec<Position> {
        let directions = if self.allow_diagonal {
            Direction::all().to_vec()
        } else {
            Direction::cardinals().to_vec()
        };

        directions
            .into_iter()
            .filter_map(|dir| {
                let next = pos.step(dir);
                if self.grid.in_bounds(next) {
                    Some(next)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update a vertex (node)
    fn update_vertex(&mut self, pos: Position) {
        if pos != self.goal {
            // rhs(s) = min over all successors s' of (c(s,s') + g(s'))
            let min_rhs = self
                .successors(pos)
                .into_iter()
                .map(|succ| {
                    let c = self.cost(pos, succ);
                    let g_succ = self.g.get(&succ).copied().unwrap_or(f64::INFINITY);
                    c + g_succ
                })
                .fold(f64::INFINITY, f64::min);

            self.rhs.insert(pos, min_rhs);
        }

        // Remove from open list if present
        if self.open_set.contains(&pos) {
            self.open_set.remove(&pos);
            // Note: We don't remove from heap (lazy deletion)
        }

        let g_val = self.g.get(&pos).copied().unwrap_or(f64::INFINITY);
        let rhs_val = self.rhs.get(&pos).copied().unwrap_or(f64::INFINITY);

        // If inconsistent, add to open list
        if (g_val - rhs_val).abs() > 1e-10 {
            let key = self.calculate_key(pos);
            self.open.push(DStarState { key, position: pos });
            self.open_set.insert(pos);
        }
    }

    /// Main computation loop
    fn compute_shortest_path(&mut self) {
        while let Some(state) = self.open.pop() {
            // Lazy deletion: skip if not in open set
            if !self.open_set.contains(&state.position) {
                continue;
            }

            let pos = state.position;
            let old_key = state.key;

            // Recompute key to check if it changed
            let new_key = self.calculate_key(pos);

            // Get values
            let g_val = self.g.get(&pos).copied().unwrap_or(f64::INFINITY);
            let rhs_val = self.rhs.get(&pos).copied().unwrap_or(f64::INFINITY);

            // Check termination condition
            let start_key = self.calculate_key(self.start);
            let start_g = self.g.get(&self.start).copied().unwrap_or(f64::INFINITY);
            let start_rhs = self.rhs.get(&self.start).copied().unwrap_or(f64::INFINITY);

            if old_key >= start_key
                && (start_rhs - start_g).abs() < 1e-10
                && start_rhs != f64::INFINITY
            {
                break;
            }

            if old_key < new_key {
                // Key has increased, reinsert with new key
                self.open.push(DStarState {
                    key: new_key,
                    position: pos,
                });
                // Keep in open_set
            } else if g_val > rhs_val {
                // Overconsistent: make consistent
                self.g.insert(pos, rhs_val);
                self.open_set.remove(&pos);
                self.nodes_expanded += 1;

                // Update predecessors
                for pred in self.predecessors(pos) {
                    self.update_vertex(pred);
                }
            } else {
                // Underconsistent: reset g-value
                self.g.insert(pos, f64::INFINITY);
                self.open_set.remove(&pos);
                self.nodes_expanded += 1;

                // Update self and predecessors
                self.update_vertex(pos);
                for pred in self.predecessors(pos) {
                    self.update_vertex(pred);
                }
            }
        }
    }

    /// Find the initial path
    pub fn find_path(&mut self) -> DStarResult {
        if !self.initialized {
            self.initialize();
        }

        // Check if start and goal are valid
        if !self.grid.is_passable(self.start) || !self.grid.is_passable(self.goal) {
            return DStarResult {
                path: None,
                cost: f64::INFINITY,
                nodes_expanded: 0,
                is_replan: false,
            };
        }

        if self.start == self.goal {
            return DStarResult {
                path: Some(vec![self.start]),
                cost: 0.0,
                nodes_expanded: 0,
                is_replan: false,
            };
        }

        let prev_expanded = self.nodes_expanded;
        self.compute_shortest_path();

        let path = self.extract_path();
        let cost = self.g.get(&self.start).copied().unwrap_or(f64::INFINITY);

        DStarResult {
            path,
            cost,
            nodes_expanded: self.nodes_expanded - prev_expanded,
            is_replan: false,
        }
    }

    /// Update the start position (robot moved)
    pub fn update_start(&mut self, new_start: Position) {
        // Update km to maintain key consistency
        self.km += self.heuristic(self.last_start, new_start);
        self.last_start = new_start;
        self.start = new_start;
    }

    /// Notify that edge costs have changed (obstacle added/removed)
    ///
    /// Call this when cells become blocked or unblocked, then call replan()
    pub fn update_edge(&mut self, pos: Position) {
        // Update all vertices that could be affected by this cell changing
        self.update_vertex(pos);

        // Also update neighbors since edge costs to/from them changed
        for neighbor in self.neighbors(pos) {
            self.update_vertex(neighbor);
        }
    }

    /// Replan after edge cost changes
    pub fn replan(&mut self) -> DStarResult {
        if !self.initialized {
            return self.find_path();
        }

        // Check if start is still valid
        if !self.grid.is_passable(self.start) {
            return DStarResult {
                path: None,
                cost: f64::INFINITY,
                nodes_expanded: 0,
                is_replan: true,
            };
        }

        let prev_expanded = self.nodes_expanded;
        self.compute_shortest_path();

        let path = self.extract_path();
        let cost = self.g.get(&self.start).copied().unwrap_or(f64::INFINITY);

        DStarResult {
            path,
            cost,
            nodes_expanded: self.nodes_expanded - prev_expanded,
            is_replan: true,
        }
    }

    /// Extract the path from start to goal using g-values
    fn extract_path(&self) -> Option<Vec<Position>> {
        let start_g = self.g.get(&self.start).copied().unwrap_or(f64::INFINITY);
        if start_g == f64::INFINITY {
            return None;
        }

        let mut path = vec![self.start];
        let mut current = self.start;

        while current != self.goal {
            // Find the successor with minimum g + cost
            let mut best_next = None;
            let mut best_value = f64::INFINITY;

            for succ in self.successors(current) {
                if !self.grid.is_passable(succ) {
                    continue;
                }

                let c = self.cost(current, succ);
                let g_succ = self.g.get(&succ).copied().unwrap_or(f64::INFINITY);
                let value = c + g_succ;

                if value < best_value {
                    best_value = value;
                    best_next = Some(succ);
                }
            }

            match best_next {
                Some(next) => {
                    path.push(next);
                    current = next;

                    // Prevent infinite loops
                    if path.len() > 10000 {
                        return None;
                    }
                }
                None => return None,
            }
        }

        Some(path)
    }
}

/// Mutable grid wrapper for D* Lite that supports dynamic changes
#[derive(Debug, Clone)]
pub struct DynamicGrid {
    pub width: usize,
    pub height: usize,
    blocked: HashSet<(i32, i32)>,
}

impl DynamicGrid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            blocked: HashSet::new(),
        }
    }

    /// Set a cell as blocked or unblocked
    pub fn set_blocked(&mut self, x: i32, y: i32, blocked: bool) {
        if blocked {
            self.blocked.insert((x, y));
        } else {
            self.blocked.remove(&(x, y));
        }
    }

    /// Check if a cell is blocked
    pub fn is_blocked(&self, x: i32, y: i32) -> bool {
        self.blocked.contains(&(x, y))
    }
}

impl Grid for DynamicGrid {
    fn is_walkable(&self, pos: Position) -> bool {
        !self.blocked.contains(&(pos.x, pos.y))
    }

    fn in_bounds(&self, pos: Position) -> bool {
        pos.x >= 0 && pos.y >= 0 && pos.x < self.width as i32 && pos.y < self.height as i32
    }
}

/// Convenience function for D* Lite pathfinding (single search)
pub fn d_star_lite_find_path<G: Grid>(
    grid: G,
    start: Position,
    goal: Position,
    allow_diagonal: bool,
) -> DStarResult {
    DStarLite::new(grid, start, goal)
        .with_diagonal(allow_diagonal)
        .find_path()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> DynamicGrid {
        // 10x10 grid with some obstacles
        let mut grid = DynamicGrid::new(10, 10);
        for y in 2..5 {
            for x in 2..5 {
                grid.set_blocked(x, y, true);
            }
        }
        grid
    }

    #[test]
    fn test_d_star_lite_straight_line() {
        let grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 0);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_d_star_lite_diagonal() {
        let grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        // Diagonal path should be 6 steps
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_d_star_lite_around_obstacle() {
        let grid = create_test_grid();
        let start = Position::new(0, 3);
        let goal = Position::new(9, 3);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));

        // Verify path avoids obstacles
        for pos in &path {
            assert!(
                pos.x < 2 || pos.x > 4 || pos.y < 2 || pos.y > 4,
                "Path goes through obstacle at {:?}",
                pos
            );
        }
    }

    #[test]
    fn test_d_star_lite_no_path() {
        let mut grid = DynamicGrid::new(10, 10);
        // Block around goal
        for x in 4..7 {
            for y in 4..7 {
                grid.set_blocked(x, y, true);
            }
        }

        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_d_star_lite_same_position() {
        let grid = DynamicGrid::new(10, 10);
        let pos = Position::new(5, 5);

        let result = d_star_lite_find_path(grid, pos, pos, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], pos);
    }

    #[test]
    fn test_d_star_lite_cardinal_only() {
        let grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(3, 3);

        let result = d_star_lite_find_path(grid, start, goal, false);

        assert!(result.path.is_some());
        let path = result.path.unwrap();

        // Verify no diagonal moves
        for i in 1..path.len() {
            let dx = (path[i].x - path[i - 1].x).abs();
            let dy = (path[i].y - path[i - 1].y).abs();
            assert!(
                (dx == 1 && dy == 0) || (dx == 0 && dy == 1),
                "Non-cardinal move detected"
            );
        }

        // Cardinal path from (0,0) to (3,3) should be 7 steps
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn test_d_star_lite_replan_obstacle_added() {
        let mut grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 0);

        let mut planner = DStarLite::new(grid.clone(), start, goal).with_diagonal(true);

        // Initial search
        let result1 = planner.find_path();
        assert!(result1.path.is_some());
        let path1 = result1.path.unwrap();
        assert_eq!(path1.len(), 6); // Direct path

        // Add obstacle in the middle of the path
        grid.set_blocked(3, 0, true);

        // Update the planner's grid (in real usage, the grid would be shared)
        planner.grid = grid;
        planner.update_edge(Position::new(3, 0));

        // Replan
        let result2 = planner.replan();
        assert!(result2.path.is_some());
        assert!(result2.is_replan);

        let path2 = result2.path.unwrap();
        // New path should avoid (3, 0)
        assert!(!path2.contains(&Position::new(3, 0)));
    }

    #[test]
    fn test_d_star_lite_replan_obstacle_removed() {
        let mut grid = DynamicGrid::new(10, 10);
        // Block the direct path (multiple cells to force longer detour)
        for y in 0..3 {
            grid.set_blocked(3, y, true);
        }

        let start = Position::new(0, 0);
        let goal = Position::new(5, 0);

        // Use cardinal only to make the test clearer
        let mut planner = DStarLite::new(grid.clone(), start, goal).with_diagonal(false);

        // Initial search (must go around)
        let result1 = planner.find_path();
        assert!(result1.path.is_some());
        let path1 = result1.path.unwrap();
        // Without diagonal, must go around the wall
        assert!(path1.len() > 6);

        // Remove obstacle at y=0
        grid.set_blocked(3, 0, false);

        // Update planner
        planner.grid = grid;
        planner.update_edge(Position::new(3, 0));

        // Replan
        let result2 = planner.replan();
        assert!(result2.path.is_some());
        assert!(result2.is_replan);

        let path2 = result2.path.unwrap();
        // New path should be shorter or equal
        assert!(path2.len() <= path1.len());
    }

    #[test]
    fn test_d_star_lite_update_start() {
        let grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(9, 0);

        let mut planner = DStarLite::new(grid, start, goal).with_diagonal(true);

        // Initial search
        let result1 = planner.find_path();
        assert!(result1.path.is_some());

        // Move start (robot moved)
        planner.update_start(Position::new(3, 0));

        // Replan from new position
        let result2 = planner.replan();
        assert!(result2.path.is_some());

        let path2 = result2.path.unwrap();
        assert_eq!(path2.first(), Some(&Position::new(3, 0)));
        assert_eq!(path2.last(), Some(&goal));
    }

    #[test]
    fn test_d_star_lite_efficiency() {
        let grid = DynamicGrid::new(20, 20);
        let start = Position::new(0, 0);
        let goal = Position::new(19, 19);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        // D* Lite should find a path efficiently (total grid is 400 cells)
        assert!(
            result.nodes_expanded <= 400,
            "Expanded {} nodes, expected <= 400",
            result.nodes_expanded
        );
    }

    #[test]
    fn test_dynamic_grid() {
        let mut grid = DynamicGrid::new(5, 5);

        assert!(grid.is_passable(Position::new(2, 2)));
        grid.set_blocked(2, 2, true);
        assert!(!grid.is_passable(Position::new(2, 2)));
        grid.set_blocked(2, 2, false);
        assert!(grid.is_passable(Position::new(2, 2)));
    }

    #[test]
    fn test_d_star_lite_path_continuity() {
        let grid = create_test_grid();
        let start = Position::new(0, 0);
        let goal = Position::new(9, 9);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();

        // Check path is continuous
        for i in 1..path.len() {
            let dx = (path[i].x - path[i - 1].x).abs();
            let dy = (path[i].y - path[i - 1].y).abs();
            assert!(dx <= 1 && dy <= 1, "Path is not continuous at index {}", i);
            assert!(dx + dy > 0, "Duplicate position at index {}", i);
        }
    }

    #[test]
    fn test_d_star_lite_impassable_start() {
        let mut grid = DynamicGrid::new(10, 10);
        grid.set_blocked(0, 0, true);

        let result = d_star_lite_find_path(grid, Position::new(0, 0), Position::new(5, 5), true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_d_star_lite_impassable_goal() {
        let mut grid = DynamicGrid::new(10, 10);
        grid.set_blocked(5, 5, true);

        let result = d_star_lite_find_path(grid, Position::new(0, 0), Position::new(5, 5), true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_d_star_lite_multiple_replans() {
        let mut grid = DynamicGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(9, 0);

        let mut planner = DStarLite::new(grid.clone(), start, goal).with_diagonal(true);

        // Initial search
        let result1 = planner.find_path();
        assert!(result1.path.is_some());

        // Multiple obstacle changes
        for x in [3, 5, 7] {
            grid.set_blocked(x, 0, true);
            planner.grid = grid.clone();
            planner.update_edge(Position::new(x, 0));

            let result = planner.replan();
            assert!(result.path.is_some());
            assert!(result.is_replan);
        }

        // Path should still exist (going around obstacles)
        let final_path = planner.replan().path.unwrap();
        assert_eq!(final_path.first(), Some(&start));
        assert_eq!(final_path.last(), Some(&goal));
    }

    #[test]
    fn test_d_star_lite_with_array_grid() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
    }

    #[test]
    fn test_key_ordering() {
        let k1 = Key::new(1.0, 2.0);
        let k2 = Key::new(1.0, 3.0);
        let k3 = Key::new(2.0, 1.0);

        // k1 < k2 (same k1, smaller k2)
        assert!(k1 > k2); // Reversed for min-heap

        // k1 < k3 (smaller k1)
        assert!(k1 > k3); // Reversed for min-heap
    }

    #[test]
    fn test_large_grid_with_obstacles() {
        let mut grid = DynamicGrid::new(50, 50);

        // Create a maze-like structure
        for y in 0..50 {
            if y % 5 != 0 {
                for x in 10..40 {
                    if (y / 5) % 2 == 0 {
                        if x < 35 {
                            grid.set_blocked(x, y, true);
                        }
                    } else if x > 15 {
                        grid.set_blocked(x, y, true);
                    }
                }
            }
        }

        let start = Position::new(0, 25);
        let goal = Position::new(49, 25);

        let result = d_star_lite_find_path(grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
    }
}
