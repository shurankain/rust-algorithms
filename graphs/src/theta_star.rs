// Theta* Algorithm for Any-Angle Pathfinding
//
// Theta* is an any-angle pathfinding algorithm that produces shorter, smoother
// paths than A* by allowing line-of-sight shortcuts. While A* restricts paths
// to grid edges, Theta* can create paths that cut across grid cells when there's
// a clear line-of-sight.
//
// Key concepts:
// - Line of Sight (LOS): Two points have LOS if a straight line between them
//   doesn't pass through obstacles
// - Parent Visibility: When expanding a node, if there's LOS from the parent's
//   parent to the current node, we can skip the intermediate parent
// - Any-Angle Paths: Resulting paths have turns at any angle, not just 45°/90°
//
// Complexity: Same as A* - O((V+E) log V) with binary heap
// Path Quality: Produces near-optimal paths (within 1% of true shortest path)
//
// References:
// - Nash et al. "Theta*: Any-Angle Path Planning on Grids" (2007)
// - http://aigamedev.com/open/tutorial/theta-star-any-angle-paths/

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// Re-use grid types from JPS module
#[cfg(test)]
use crate::jps::ArrayGrid;
use crate::jps::{Direction, Grid, Position};

/// Result of Theta* pathfinding
#[derive(Debug, Clone)]
pub struct ThetaStarResult {
    /// The path from start to goal (if found)
    pub path: Option<Vec<Position>>,
    /// Total cost of the path (Euclidean distance * 100 for precision)
    pub cost: u32,
    /// Number of nodes expanded
    pub nodes_expanded: usize,
    /// Number of line-of-sight checks performed
    pub los_checks: usize,
}

/// State for the priority queue
#[derive(Clone)]
struct ThetaState {
    f_score: u32, // g + h (scaled by 100 for precision)
    g_score: u32,
    position: Position,
}

impl Eq for ThetaState {}

impl PartialEq for ThetaState {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score && self.position == other.position
    }
}

impl Ord for ThetaState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse comparison
        other
            .f_score
            .cmp(&self.f_score)
            .then_with(|| other.g_score.cmp(&self.g_score))
    }
}

impl PartialOrd for ThetaState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Theta* pathfinder for any-angle pathfinding
pub struct ThetaStar<'a, G: Grid> {
    grid: &'a G,
    /// Allow diagonal movement when no line-of-sight
    allow_diagonal: bool,
    /// Cost multiplier for integer math (default 100)
    cost_scale: u32,
}

impl<'a, G: Grid> ThetaStar<'a, G> {
    pub fn new(grid: &'a G) -> Self {
        Self {
            grid,
            allow_diagonal: true,
            cost_scale: 100,
        }
    }

    pub fn with_diagonal(mut self, allow: bool) -> Self {
        self.allow_diagonal = allow;
        self
    }

    /// Find path from start to goal using Theta*
    pub fn find_path(&self, start: Position, goal: Position) -> ThetaStarResult {
        if !self.grid.is_passable(start) || !self.grid.is_passable(goal) {
            return ThetaStarResult {
                path: None,
                cost: 0,
                nodes_expanded: 0,
                los_checks: 0,
            };
        }

        if start == goal {
            return ThetaStarResult {
                path: Some(vec![start]),
                cost: 0,
                nodes_expanded: 0,
                los_checks: 0,
            };
        }

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<Position, Position> = HashMap::new();
        let mut g_score: HashMap<Position, u32> = HashMap::new();
        let mut closed: HashSet<Position> = HashSet::new();
        let mut nodes_expanded = 0;
        let mut los_checks = 0;

        let h = |pos: Position| -> u32 { self.euclidean_distance(pos, goal) };

        g_score.insert(start, 0);
        came_from.insert(start, start); // Start is its own parent
        open.push(ThetaState {
            f_score: h(start),
            g_score: 0,
            position: start,
        });

        while let Some(current) = open.pop() {
            if current.position == goal {
                let path = self.reconstruct_path(&came_from, goal);
                return ThetaStarResult {
                    path: Some(path),
                    cost: current.g_score,
                    nodes_expanded,
                    los_checks,
                };
            }

            if closed.contains(&current.position) {
                continue;
            }
            closed.insert(current.position);
            nodes_expanded += 1;

            // Get neighbors
            let neighbors = self.get_neighbors(current.position);

            for neighbor in neighbors {
                if closed.contains(&neighbor) || !self.grid.is_passable(neighbor) {
                    continue;
                }

                // Get parent of current node
                let parent = came_from
                    .get(&current.position)
                    .copied()
                    .unwrap_or(current.position);

                // Theta* key insight: check line-of-sight from parent to neighbor
                los_checks += 1;
                let (new_parent, tentative_g) = if self.line_of_sight(parent, neighbor) {
                    // Path 2: Connect directly from parent (skip current)
                    let g = g_score.get(&parent).copied().unwrap_or(u32::MAX);
                    if g == u32::MAX {
                        continue;
                    }
                    (parent, g + self.euclidean_distance(parent, neighbor))
                } else {
                    // Path 1: Traditional A* path through current
                    let g = current.g_score;
                    (
                        current.position,
                        g + self.euclidean_distance(current.position, neighbor),
                    )
                };

                if tentative_g < *g_score.get(&neighbor).unwrap_or(&u32::MAX) {
                    came_from.insert(neighbor, new_parent);
                    g_score.insert(neighbor, tentative_g);

                    open.push(ThetaState {
                        f_score: tentative_g + h(neighbor),
                        g_score: tentative_g,
                        position: neighbor,
                    });
                }
            }
        }

        ThetaStarResult {
            path: None,
            cost: 0,
            nodes_expanded,
            los_checks,
        }
    }

    /// Calculate Euclidean distance (scaled by cost_scale for integer math)
    fn euclidean_distance(&self, from: Position, to: Position) -> u32 {
        let dx = (to.x - from.x) as f64;
        let dy = (to.y - from.y) as f64;
        ((dx * dx + dy * dy).sqrt() * self.cost_scale as f64) as u32
    }

    /// Get neighbors of a position (with corner-cutting prevention for diagonals)
    fn get_neighbors(&self, pos: Position) -> Vec<Position> {
        let directions = if self.allow_diagonal {
            Direction::all().to_vec()
        } else {
            Direction::cardinals().to_vec()
        };

        directions
            .into_iter()
            .filter_map(|dir| {
                let next = pos.step(dir);
                if !self.grid.is_passable(next) {
                    return None;
                }

                // For diagonal moves, prevent corner cutting
                if dir.is_diagonal() {
                    let (v_dir, h_dir) = dir.cardinal_components().unwrap();
                    let v_pos = pos.step(v_dir);
                    let h_pos = pos.step(h_dir);

                    // At least one adjacent cell must be passable to prevent cutting through corner
                    if !self.grid.is_passable(v_pos) && !self.grid.is_passable(h_pos) {
                        return None;
                    }
                }

                Some(next)
            })
            .collect()
    }

    /// Check line of sight between two positions using Bresenham's line algorithm
    ///
    /// Returns true if there's a clear path (no obstacles) between the two points.
    /// Uses a modified Bresenham's algorithm that checks all cells the line passes through.
    pub fn line_of_sight(&self, from: Position, to: Position) -> bool {
        let mut x0 = from.x;
        let mut y0 = from.y;
        let x1 = to.x;
        let y1 = to.y;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();

        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };

        let mut err = dx - dy;

        loop {
            // Check current cell (skip start position since we know it's passable)
            if (x0 != from.x || y0 != from.y) && !self.grid.is_passable(Position::new(x0, y0)) {
                return false;
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;

            // Handle diagonal moves - prevent corner cutting only if BOTH adjacent cells are blocked
            if e2 > -dy && e2 < dx {
                // Moving diagonally - at least one adjacent cell must be passable
                let h_passable = self.grid.is_passable(Position::new(x0 + sx, y0));
                let v_passable = self.grid.is_passable(Position::new(x0, y0 + sy));
                if !h_passable && !v_passable {
                    return false;
                }
            }

            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }

            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }

        true
    }

    /// Reconstruct path from came_from map
    fn reconstruct_path(
        &self,
        came_from: &HashMap<Position, Position>,
        goal: Position,
    ) -> Vec<Position> {
        let mut path = Vec::new();
        let mut current = goal;

        path.push(current);
        while let Some(&parent) = came_from.get(&current) {
            if parent == current {
                break; // Reached start
            }
            path.push(parent);
            current = parent;
        }

        path.reverse();
        path
    }
}

/// Convenience function for Theta* pathfinding
pub fn theta_star_find_path<G: Grid>(
    grid: &G,
    start: Position,
    goal: Position,
    allow_diagonal: bool,
) -> ThetaStarResult {
    ThetaStar::new(grid)
        .with_diagonal(allow_diagonal)
        .find_path(start, goal)
}

/// Basic A* for comparison (to demonstrate Theta* improvements)
pub struct BasicAStar<'a, G: Grid> {
    grid: &'a G,
    allow_diagonal: bool,
    cost_scale: u32,
}

impl<'a, G: Grid> BasicAStar<'a, G> {
    pub fn new(grid: &'a G) -> Self {
        Self {
            grid,
            allow_diagonal: true,
            cost_scale: 100,
        }
    }

    pub fn with_diagonal(mut self, allow: bool) -> Self {
        self.allow_diagonal = allow;
        self
    }

    /// Find path using basic A* (for comparison with Theta*)
    pub fn find_path(&self, start: Position, goal: Position) -> ThetaStarResult {
        if !self.grid.is_passable(start) || !self.grid.is_passable(goal) {
            return ThetaStarResult {
                path: None,
                cost: 0,
                nodes_expanded: 0,
                los_checks: 0,
            };
        }

        if start == goal {
            return ThetaStarResult {
                path: Some(vec![start]),
                cost: 0,
                nodes_expanded: 0,
                los_checks: 0,
            };
        }

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<Position, Position> = HashMap::new();
        let mut g_score: HashMap<Position, u32> = HashMap::new();
        let mut closed: HashSet<Position> = HashSet::new();
        let mut nodes_expanded = 0;

        let h = |pos: Position| -> u32 {
            let dx = (pos.x - goal.x).abs() as f64;
            let dy = (pos.y - goal.y).abs() as f64;
            ((dx * dx + dy * dy).sqrt() * self.cost_scale as f64) as u32
        };

        g_score.insert(start, 0);
        open.push(ThetaState {
            f_score: h(start),
            g_score: 0,
            position: start,
        });

        while let Some(current) = open.pop() {
            if current.position == goal {
                let path = self.reconstruct_path(&came_from, goal);
                return ThetaStarResult {
                    path: Some(path),
                    cost: current.g_score,
                    nodes_expanded,
                    los_checks: 0,
                };
            }

            if closed.contains(&current.position) {
                continue;
            }
            closed.insert(current.position);
            nodes_expanded += 1;

            let directions = if self.allow_diagonal {
                Direction::all().to_vec()
            } else {
                Direction::cardinals().to_vec()
            };

            for dir in directions {
                let neighbor = current.position.step(dir);

                if closed.contains(&neighbor) || !self.grid.is_passable(neighbor) {
                    continue;
                }

                // A* uses grid distance
                let move_cost = if dir.is_diagonal() {
                    141 // sqrt(2) * 100
                } else {
                    100
                };

                let tentative_g = current.g_score + move_cost;

                if tentative_g < *g_score.get(&neighbor).unwrap_or(&u32::MAX) {
                    came_from.insert(neighbor, current.position);
                    g_score.insert(neighbor, tentative_g);

                    open.push(ThetaState {
                        f_score: tentative_g + h(neighbor),
                        g_score: tentative_g,
                        position: neighbor,
                    });
                }
            }
        }

        ThetaStarResult {
            path: None,
            cost: 0,
            nodes_expanded,
            los_checks: 0,
        }
    }

    fn reconstruct_path(
        &self,
        came_from: &HashMap<Position, Position>,
        goal: Position,
    ) -> Vec<Position> {
        let mut path = Vec::new();
        let mut current = goal;

        path.push(current);
        while let Some(&parent) = came_from.get(&current) {
            path.push(parent);
            current = parent;
        }

        path.reverse();
        path
    }
}

/// Compare Theta* path with A* path for the same grid
pub fn compare_algorithms<G: Grid>(
    grid: &G,
    start: Position,
    goal: Position,
) -> (ThetaStarResult, ThetaStarResult) {
    let theta_result = theta_star_find_path(grid, start, goal, true);
    let astar_result = BasicAStar::new(grid).find_path(start, goal);
    (theta_result, astar_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> ArrayGrid {
        // 10x10 grid with some obstacles
        // . = walkable, # = blocked
        // ..........
        // ..........
        // ..###.....
        // ..###.....
        // ..###.....
        // ..........
        // ..........
        // ..........
        // ..........
        // ..........
        let mut grid = ArrayGrid::new(10, 10);
        for y in 2..5 {
            for x in 2..5 {
                grid.set_blocked(x, y, true);
            }
        }
        grid
    }

    #[test]
    fn test_theta_star_straight_line() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 0);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        // Theta* with LOS should produce direct path
        assert_eq!(path.len(), 2); // Just start and goal
    }

    #[test]
    fn test_theta_star_diagonal() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        // Direct line of sight - just 2 points
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_theta_star_around_obstacle() {
        let grid = create_test_grid();
        let start = Position::new(0, 3);
        let goal = Position::new(9, 3);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));

        // All positions in path should be passable
        for pos in &path {
            assert!(grid.is_passable(*pos), "Position {:?} is not passable", pos);
        }
    }

    #[test]
    fn test_theta_star_no_path() {
        let mut grid = ArrayGrid::new(10, 10);
        // Block around goal
        for x in 4..7 {
            for y in 4..7 {
                grid.set_blocked(x, y, true);
            }
        }

        let start = Position::new(0, 0);
        let goal = Position::new(5, 5); // Inside blocked area

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_theta_star_same_position() {
        let grid = ArrayGrid::new(10, 10);
        let pos = Position::new(5, 5);

        let result = theta_star_find_path(&grid, pos, pos, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], pos);
    }

    #[test]
    fn test_line_of_sight_clear() {
        let grid = ArrayGrid::new(10, 10);
        let theta = ThetaStar::new(&grid);

        assert!(theta.line_of_sight(Position::new(0, 0), Position::new(5, 5)));
        assert!(theta.line_of_sight(Position::new(0, 0), Position::new(9, 0)));
        assert!(theta.line_of_sight(Position::new(3, 3), Position::new(7, 8)));
    }

    #[test]
    fn test_line_of_sight_blocked() {
        let mut grid = ArrayGrid::new(10, 10);
        grid.set_blocked(5, 5, true);

        let theta = ThetaStar::new(&grid);

        // Line passes through blocked cell
        assert!(!theta.line_of_sight(Position::new(0, 0), Position::new(9, 9)));
        // Line doesn't pass through blocked cell
        assert!(theta.line_of_sight(Position::new(0, 0), Position::new(3, 3)));
    }

    #[test]
    fn test_line_of_sight_corner_cutting() {
        let mut grid = ArrayGrid::new(10, 10);
        // Create corner obstacle
        grid.set_blocked(5, 4, true);
        grid.set_blocked(4, 5, true);

        let theta = ThetaStar::new(&grid);

        // Should not allow cutting through corner
        let can_cut = theta.line_of_sight(Position::new(4, 4), Position::new(5, 5));
        assert!(!can_cut, "Should not allow corner cutting");
    }

    #[test]
    fn test_theta_vs_astar_path_length() {
        let grid = ArrayGrid::new(20, 20);
        let start = Position::new(0, 0);
        let goal = Position::new(15, 10);

        let (theta_result, astar_result) = compare_algorithms(&grid, start, goal);

        assert!(theta_result.path.is_some());
        assert!(astar_result.path.is_some());

        let theta_path = theta_result.path.unwrap();
        let astar_path = astar_result.path.unwrap();

        // Theta* should produce shorter path (fewer waypoints)
        assert!(
            theta_path.len() <= astar_path.len(),
            "Theta* path length {} should be <= A* path length {}",
            theta_path.len(),
            astar_path.len()
        );

        // Theta* should have lower or equal cost
        assert!(
            theta_result.cost <= astar_result.cost,
            "Theta* cost {} should be <= A* cost {}",
            theta_result.cost,
            astar_result.cost
        );
    }

    #[test]
    fn test_theta_star_cardinal_only() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = theta_star_find_path(&grid, start, goal, false);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
    }

    #[test]
    fn test_theta_star_complex_obstacle() {
        // Create L-shaped obstacle
        // ..........
        // .###......
        // .#........
        // .#........
        // ..........
        let mut grid = ArrayGrid::new(10, 10);
        grid.set_blocked(1, 1, true);
        grid.set_blocked(2, 1, true);
        grid.set_blocked(3, 1, true);
        grid.set_blocked(1, 2, true);
        grid.set_blocked(1, 3, true);

        let start = Position::new(0, 0);
        let goal = Position::new(5, 5);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        // Path exists, verify LOS checks were performed
        assert!(result.los_checks > 0);
    }

    #[test]
    fn test_theta_star_narrow_corridor() {
        // Create narrow corridor
        // #####.####
        // #####.####
        // ..........
        // #####.####
        // #####.####
        let mut grid = ArrayGrid::new(10, 5);
        for y in [0, 1, 3, 4] {
            for x in 0..10 {
                if x != 5 {
                    grid.set_blocked(x, y, true);
                }
            }
        }

        let start = Position::new(0, 2);
        let goal = Position::new(9, 2);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
    }

    #[test]
    fn test_euclidean_distance() {
        let grid = ArrayGrid::new(10, 10);
        let theta = ThetaStar::new(&grid);

        // Distance from (0,0) to (3,4) should be 5 * 100 = 500
        let dist = theta.euclidean_distance(Position::new(0, 0), Position::new(3, 4));
        assert_eq!(dist, 500);

        // Distance from (0,0) to (1,1) should be sqrt(2) * 100 ≈ 141
        let dist = theta.euclidean_distance(Position::new(0, 0), Position::new(1, 1));
        assert!((dist as i32 - 141).abs() <= 1);
    }

    #[test]
    fn test_los_checks_count() {
        let grid = ArrayGrid::new(20, 20);
        let start = Position::new(0, 0);
        let goal = Position::new(10, 10);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        assert!(result.los_checks > 0, "Should have performed LOS checks");
    }

    #[test]
    fn test_large_grid() {
        let grid = ArrayGrid::new(100, 100);
        let start = Position::new(0, 0);
        let goal = Position::new(99, 99);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        // Direct line of sight in open grid
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_path_validity() {
        let grid = create_test_grid();
        let start = Position::new(0, 0);
        let goal = Position::new(9, 9);

        let result = theta_star_find_path(&grid, start, goal, true);

        assert!(result.path.is_some());
        let path = result.path.unwrap();

        // Verify all path points are passable
        for pos in &path {
            assert!(grid.is_passable(*pos));
        }

        // Verify line of sight between consecutive points
        let theta = ThetaStar::new(&grid);
        for i in 1..path.len() {
            assert!(
                theta.line_of_sight(path[i - 1], path[i]),
                "No LOS between {:?} and {:?}",
                path[i - 1],
                path[i]
            );
        }
    }

    #[test]
    fn test_impassable_start() {
        let mut grid = ArrayGrid::new(10, 10);
        grid.set_blocked(0, 0, true);

        let result = theta_star_find_path(&grid, Position::new(0, 0), Position::new(5, 5), true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_impassable_goal() {
        let mut grid = ArrayGrid::new(10, 10);
        grid.set_blocked(5, 5, true);

        let result = theta_star_find_path(&grid, Position::new(0, 0), Position::new(5, 5), true);

        assert!(result.path.is_none());
    }

    #[test]
    fn test_basic_astar_comparison() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(7, 7);

        let astar = BasicAStar::new(&grid);
        let result = astar.find_path(start, goal);

        assert!(result.path.is_some());
        let path = result.path.unwrap();
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        // A* produces step-by-step path
        assert!(path.len() > 2);
    }

    #[test]
    fn test_basic_astar_cardinal_only() {
        let grid = ArrayGrid::new(10, 10);
        let start = Position::new(0, 0);
        let goal = Position::new(3, 0);

        let astar = BasicAStar::new(&grid).with_diagonal(false);
        let result = astar.find_path(start, goal);

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
    }
}
