// Iterative Deepening A* (IDA*)
//
// Memory-efficient pathfinding algorithm combining A*'s heuristic guidance
// with iterative deepening's space efficiency. Instead of maintaining an
// open/closed set like A*, IDA* uses depth-first search with a cost threshold.
//
// Key idea: Run depth-limited DFS, where limit is f(n) = g(n) + h(n).
// If no solution found, increase threshold to minimum f-value that exceeded it.
//
// Space: O(d) where d = solution depth (vs O(b^d) for A*)
// Time: O(b^d) worst case, but often much better with good heuristic
//
// Best for: Memory-constrained environments, very deep search spaces

use std::hash::Hash;

pub trait IDAState: Clone + Eq + Hash {
    type Action: Clone;

    fn successors(&self) -> Vec<(Self::Action, Self, u32)>; // (action, new_state, step_cost)
    fn is_goal(&self) -> bool;
    fn heuristic(&self) -> u32; // Admissible heuristic (never overestimates)
}

#[derive(Debug, Clone, PartialEq)]
pub enum IDAResult<A> {
    Found(Vec<A>, u32), // (path, total_cost)
    NotFound,
    CutoffExceeded,
}

pub fn ida_star<S: IDAState>(start: S) -> IDAResult<S::Action> {
    let mut threshold = start.heuristic();

    loop {
        let (result, new_threshold) = search(&start, 0, threshold, vec![]);

        match result {
            SearchResult::Found(path, cost) => return IDAResult::Found(path, cost),
            SearchResult::NotFound => return IDAResult::NotFound,
            SearchResult::Cutoff => {
                if new_threshold == u32::MAX {
                    return IDAResult::NotFound;
                }
                threshold = new_threshold;
            }
        }
    }
}

enum SearchResult<A> {
    Found(Vec<A>, u32),
    NotFound,
    Cutoff,
}

fn search<S: IDAState>(
    state: &S,
    g: u32,
    threshold: u32,
    path: Vec<S::Action>,
) -> (SearchResult<S::Action>, u32) {
    let f = g.saturating_add(state.heuristic());

    if f > threshold {
        return (SearchResult::Cutoff, f);
    }

    if state.is_goal() {
        return (SearchResult::Found(path, g), f);
    }

    let mut min_threshold = u32::MAX;
    let successors = state.successors();

    if successors.is_empty() {
        return (SearchResult::NotFound, min_threshold);
    }

    for (action, next_state, step_cost) in successors {
        let mut new_path = path.clone();
        new_path.push(action);

        let (result, new_t) = search(
            &next_state,
            g.saturating_add(step_cost),
            threshold,
            new_path,
        );

        match result {
            SearchResult::Found(p, c) => return (SearchResult::Found(p, c), new_t),
            SearchResult::Cutoff => min_threshold = min_threshold.min(new_t),
            SearchResult::NotFound => {}
        }
    }

    if min_threshold == u32::MAX {
        (SearchResult::NotFound, min_threshold)
    } else {
        (SearchResult::Cutoff, min_threshold)
    }
}

// Convenience wrapper with iteration limit
pub fn ida_star_limited<S: IDAState>(start: S, max_iterations: u32) -> IDAResult<S::Action> {
    let mut threshold = start.heuristic();

    for _ in 0..max_iterations {
        let (result, new_threshold) = search(&start, 0, threshold, vec![]);

        match result {
            SearchResult::Found(path, cost) => return IDAResult::Found(path, cost),
            SearchResult::NotFound => return IDAResult::NotFound,
            SearchResult::Cutoff => {
                if new_threshold == u32::MAX {
                    return IDAResult::NotFound;
                }
                threshold = new_threshold;
            }
        }
    }

    IDAResult::CutoffExceeded
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple grid pathfinding for testing
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct GridPos {
        x: i32,
        y: i32,
        goal_x: i32,
        goal_y: i32,
        walls: Vec<(i32, i32)>,
    }

    #[derive(Clone, Debug, PartialEq)]
    enum Direction {
        Up,
        Down,
        Left,
        Right,
    }

    impl IDAState for GridPos {
        type Action = Direction;

        fn successors(&self) -> Vec<(Direction, Self, u32)> {
            let moves = [
                (Direction::Up, 0, -1),
                (Direction::Down, 0, 1),
                (Direction::Left, -1, 0),
                (Direction::Right, 1, 0),
            ];

            moves
                .into_iter()
                .filter_map(|(dir, dx, dy)| {
                    let new_x = self.x + dx;
                    let new_y = self.y + dy;

                    // Check bounds and walls
                    if new_x < 0 || new_y < 0 || new_x > 10 || new_y > 10 {
                        return None;
                    }
                    if self.walls.contains(&(new_x, new_y)) {
                        return None;
                    }

                    Some((
                        dir,
                        GridPos {
                            x: new_x,
                            y: new_y,
                            goal_x: self.goal_x,
                            goal_y: self.goal_y,
                            walls: self.walls.clone(),
                        },
                        1,
                    ))
                })
                .collect()
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }

        fn heuristic(&self) -> u32 {
            // Manhattan distance (admissible for 4-directional movement)
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
        }
    }

    #[test]
    fn test_ida_star_simple_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 0,
            walls: vec![],
        };

        let result = ida_star(start);

        match result {
            IDAResult::Found(path, cost) => {
                assert_eq!(cost, 3);
                assert_eq!(path.len(), 3);
                assert!(path.iter().all(|d| *d == Direction::Right));
            }
            _ => panic!("Expected to find path"),
        }
    }

    #[test]
    fn test_ida_star_with_walls() {
        // Start at (0,0), goal at (2,0)
        // Wall at (1,0), must go around
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 2,
            goal_y: 0,
            walls: vec![(1, 0)],
        };

        let result = ida_star(start);

        match result {
            IDAResult::Found(path, cost) => {
                // Must go down, right, right, up (or similar)
                assert_eq!(cost, 4);
                assert_eq!(path.len(), 4);
            }
            _ => panic!("Expected to find path"),
        }
    }

    #[test]
    fn test_ida_star_already_at_goal() {
        let start = GridPos {
            x: 5,
            y: 5,
            goal_x: 5,
            goal_y: 5,
            walls: vec![],
        };

        let result = ida_star(start);

        match result {
            IDAResult::Found(path, cost) => {
                assert_eq!(cost, 0);
                assert!(path.is_empty());
            }
            _ => panic!("Expected to find empty path"),
        }
    }

    #[test]
    fn test_ida_star_diagonal_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 2,
            goal_y: 2,
            walls: vec![],
        };

        let result = ida_star(start);

        match result {
            IDAResult::Found(path, cost) => {
                assert_eq!(cost, 4); // Manhattan distance
                assert_eq!(path.len(), 4);
            }
            _ => panic!("Expected to find path"),
        }
    }

    #[test]
    fn test_ida_star_limited_iterations() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 3,
            walls: vec![],
        };

        // Should find with enough iterations
        let result = ida_star_limited(start.clone(), 100);
        assert!(matches!(result, IDAResult::Found(_, _)));

        // With very few iterations, might not finish
        let result = ida_star_limited(start, 1);
        // Either finds it or exceeds cutoff
        assert!(matches!(
            result,
            IDAResult::Found(_, _) | IDAResult::CutoffExceeded
        ));
    }

    // 15-puzzle state for more complex testing
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct SlidingTile {
        tiles: [u8; 9], // 3x3 puzzle, 0 = empty
        empty_pos: usize,
    }

    impl SlidingTile {
        fn goal() -> [u8; 9] {
            [1, 2, 3, 4, 5, 6, 7, 8, 0]
        }
    }

    impl IDAState for SlidingTile {
        type Action = usize; // Position to swap with empty

        fn successors(&self) -> Vec<(usize, Self, u32)> {
            let row = self.empty_pos / 3;
            let col = self.empty_pos % 3;

            let mut moves = vec![];

            // Up
            if row > 0 {
                moves.push(self.empty_pos - 3);
            }
            // Down
            if row < 2 {
                moves.push(self.empty_pos + 3);
            }
            // Left
            if col > 0 {
                moves.push(self.empty_pos - 1);
            }
            // Right
            if col < 2 {
                moves.push(self.empty_pos + 1);
            }

            moves
                .into_iter()
                .map(|swap_pos| {
                    let mut new_tiles = self.tiles;
                    new_tiles.swap(self.empty_pos, swap_pos);
                    (
                        swap_pos,
                        SlidingTile {
                            tiles: new_tiles,
                            empty_pos: swap_pos,
                        },
                        1,
                    )
                })
                .collect()
        }

        fn is_goal(&self) -> bool {
            self.tiles == Self::goal()
        }

        fn heuristic(&self) -> u32 {
            // Manhattan distance for each tile
            let mut dist = 0u32;
            for (i, &tile) in self.tiles.iter().enumerate() {
                if tile == 0 {
                    continue;
                }
                let goal_pos = (tile - 1) as usize;
                let curr_row = i / 3;
                let curr_col = i % 3;
                let goal_row = goal_pos / 3;
                let goal_col = goal_pos % 3;
                dist += (curr_row as i32 - goal_row as i32).unsigned_abs()
                    + (curr_col as i32 - goal_col as i32).unsigned_abs();
            }
            dist
        }
    }

    #[test]
    fn test_ida_star_sliding_puzzle() {
        // Easy puzzle: just one move away
        let start = SlidingTile {
            tiles: [1, 2, 3, 4, 5, 6, 7, 0, 8],
            empty_pos: 7,
        };

        let result = ida_star(start);

        match result {
            IDAResult::Found(path, cost) => {
                assert_eq!(cost, 1);
                assert_eq!(path.len(), 1);
            }
            _ => panic!("Expected to find solution"),
        }
    }
}
