// Best-First Search
//
// Greedy heuristic search that always expands the node with the best heuristic value.
// Unlike A*, it only considers h(n), not g(n) + h(n), making it faster but not optimal.
//
// Variants implemented:
// - Greedy Best-First: expands node with lowest h(n) - fast but may find suboptimal paths
// - A* (for comparison): expands node with lowest f(n) = g(n) + h(n) - optimal with admissible h
//
// Time: O(b^m) worst case, but often much better with good heuristic
// Space: O(b^m) - must store all generated nodes
//
// Use when: Speed matters more than optimality, or as foundation for other algorithms

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::Hash;

pub trait SearchState: Clone + Eq + Hash {
    type Action: Clone;

    fn successors(&self) -> Vec<(Self::Action, Self, u32)>; // (action, state, cost)
    fn is_goal(&self) -> bool;
    fn heuristic(&self) -> u32;
}

#[derive(Clone)]
struct Node<S: SearchState> {
    state: S,
    g_cost: u32, // cost from start
    h_cost: u32, // heuristic estimate to goal
    path: Vec<S::Action>,
}

impl<S: SearchState> Node<S> {
    fn f_cost(&self) -> u32 {
        self.g_cost.saturating_add(self.h_cost)
    }
}

// For greedy best-first: order by h only
struct GreedyNode<S: SearchState>(Node<S>);

impl<S: SearchState> PartialEq for GreedyNode<S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.h_cost == other.0.h_cost
    }
}

impl<S: SearchState> Eq for GreedyNode<S> {}

impl<S: SearchState> PartialOrd for GreedyNode<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: SearchState> Ord for GreedyNode<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (lower h = higher priority)
        other.0.h_cost.cmp(&self.0.h_cost)
    }
}

// For A*: order by f = g + h
struct AStarNode<S: SearchState>(Node<S>);

impl<S: SearchState> PartialEq for AStarNode<S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.f_cost() == other.0.f_cost()
    }
}

impl<S: SearchState> Eq for AStarNode<S> {}

impl<S: SearchState> PartialOrd for AStarNode<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: SearchState> Ord for AStarNode<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap, tie-break by h
        match other.0.f_cost().cmp(&self.0.f_cost()) {
            Ordering::Equal => other.0.h_cost.cmp(&self.0.h_cost),
            ord => ord,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult<A> {
    pub path: Vec<A>,
    pub cost: u32,
    pub nodes_expanded: u32,
}

// Greedy Best-First Search - expands by h(n) only
pub fn greedy_best_first<S: SearchState>(start: S) -> Option<SearchResult<S::Action>> {
    let mut open: BinaryHeap<GreedyNode<S>> = BinaryHeap::new();
    let mut closed: HashSet<S> = HashSet::new();
    let mut nodes_expanded = 0;

    let h = start.heuristic();
    open.push(GreedyNode(Node {
        state: start,
        g_cost: 0,
        h_cost: h,
        path: vec![],
    }));

    while let Some(GreedyNode(current)) = open.pop() {
        if current.state.is_goal() {
            return Some(SearchResult {
                path: current.path,
                cost: current.g_cost,
                nodes_expanded,
            });
        }

        if closed.contains(&current.state) {
            continue;
        }
        closed.insert(current.state.clone());
        nodes_expanded += 1;

        for (action, next_state, step_cost) in current.state.successors() {
            if closed.contains(&next_state) {
                continue;
            }

            let mut new_path = current.path.clone();
            new_path.push(action);

            open.push(GreedyNode(Node {
                h_cost: next_state.heuristic(),
                state: next_state,
                g_cost: current.g_cost + step_cost,
                path: new_path,
            }));
        }
    }

    None
}

// A* Search - expands by f(n) = g(n) + h(n)
pub fn a_star<S: SearchState>(start: S) -> Option<SearchResult<S::Action>> {
    let mut open: BinaryHeap<AStarNode<S>> = BinaryHeap::new();
    let mut closed: HashSet<S> = HashSet::new();
    let mut nodes_expanded = 0;

    let h = start.heuristic();
    open.push(AStarNode(Node {
        state: start,
        g_cost: 0,
        h_cost: h,
        path: vec![],
    }));

    while let Some(AStarNode(current)) = open.pop() {
        if current.state.is_goal() {
            return Some(SearchResult {
                path: current.path,
                cost: current.g_cost,
                nodes_expanded,
            });
        }

        if closed.contains(&current.state) {
            continue;
        }
        closed.insert(current.state.clone());
        nodes_expanded += 1;

        for (action, next_state, step_cost) in current.state.successors() {
            if closed.contains(&next_state) {
                continue;
            }

            let mut new_path = current.path.clone();
            new_path.push(action);

            open.push(AStarNode(Node {
                h_cost: next_state.heuristic(),
                state: next_state,
                g_cost: current.g_cost + step_cost,
                path: new_path,
            }));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Eq, Hash)]
    struct GridPos {
        x: i32,
        y: i32,
        goal_x: i32,
        goal_y: i32,
    }

    #[derive(Clone, Debug, PartialEq)]
    enum Dir {
        Up,
        Down,
        Left,
        Right,
    }

    impl SearchState for GridPos {
        type Action = Dir;

        fn successors(&self) -> Vec<(Dir, Self, u32)> {
            let moves = [
                (Dir::Up, 0, -1),
                (Dir::Down, 0, 1),
                (Dir::Left, -1, 0),
                (Dir::Right, 1, 0),
            ];

            moves
                .into_iter()
                .filter_map(|(dir, dx, dy)| {
                    let new_x = self.x + dx;
                    let new_y = self.y + dy;
                    if new_x < 0 || new_y < 0 || new_x > 10 || new_y > 10 {
                        return None;
                    }
                    Some((
                        dir,
                        GridPos {
                            x: new_x,
                            y: new_y,
                            goal_x: self.goal_x,
                            goal_y: self.goal_y,
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
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
        }
    }

    #[test]
    fn test_greedy_simple_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 0,
        };

        let result = greedy_best_first(start).unwrap();
        assert_eq!(result.cost, 3);
        assert_eq!(result.path.len(), 3);
    }

    #[test]
    fn test_a_star_simple_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 0,
        };

        let result = a_star(start).unwrap();
        assert_eq!(result.cost, 3);
        assert_eq!(result.path.len(), 3);
    }

    #[test]
    fn test_already_at_goal() {
        let start = GridPos {
            x: 5,
            y: 5,
            goal_x: 5,
            goal_y: 5,
        };

        let greedy = greedy_best_first(start.clone()).unwrap();
        assert_eq!(greedy.cost, 0);
        assert!(greedy.path.is_empty());

        let astar = a_star(start).unwrap();
        assert_eq!(astar.cost, 0);
        assert!(astar.path.is_empty());
    }

    #[test]
    fn test_diagonal_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 2,
            goal_y: 2,
        };

        let result = a_star(start).unwrap();
        assert_eq!(result.cost, 4); // Manhattan path
    }

    // Grid with weighted edges
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct WeightedGrid {
        x: i32,
        y: i32,
        goal_x: i32,
        goal_y: i32,
    }

    impl SearchState for WeightedGrid {
        type Action = Dir;

        fn successors(&self) -> Vec<(Dir, Self, u32)> {
            let moves = [
                (Dir::Up, 0, -1, 1),
                (Dir::Down, 0, 1, 1),
                (Dir::Left, -1, 0, 10), // Left is expensive
                (Dir::Right, 1, 0, 1),
            ];

            moves
                .into_iter()
                .filter_map(|(dir, dx, dy, cost)| {
                    let new_x = self.x + dx;
                    let new_y = self.y + dy;
                    if new_x < 0 || new_y < 0 || new_x > 10 || new_y > 10 {
                        return None;
                    }
                    Some((
                        dir,
                        WeightedGrid {
                            x: new_x,
                            y: new_y,
                            goal_x: self.goal_x,
                            goal_y: self.goal_y,
                        },
                        cost,
                    ))
                })
                .collect()
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }

        fn heuristic(&self) -> u32 {
            // Optimistic: assume all moves cost 1
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
        }
    }

    #[test]
    fn test_a_star_finds_optimal_with_weights() {
        // Going left is expensive (10), right is cheap (1)
        // From (1,0) to (0,0): greedy might go left (cost 10)
        // A* should find going right then around is cheaper if possible
        let start = WeightedGrid {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 0,
        };

        let result = a_star(start).unwrap();
        // Should go right 3 times, cost 3
        assert_eq!(result.cost, 3);
    }

    #[test]
    fn test_greedy_expands_fewer_nodes_sometimes() {
        // Greedy often expands fewer nodes but may find suboptimal path
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 5,
            goal_y: 5,
        };

        let greedy = greedy_best_first(start.clone()).unwrap();
        let astar = a_star(start).unwrap();

        // Both find a path
        assert!(!greedy.path.is_empty());
        assert!(!astar.path.is_empty());

        // A* guarantees optimal cost
        assert_eq!(astar.cost, 10);
    }
}
