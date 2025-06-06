use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// A node in the priority queue for A*
#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: usize, // f(n) = g(n) + h(n)
    position: usize,
}

// BinaryHeap is max-heap by default; reverse for min-heap
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// A* algorithm on graph represented by adjacency list.
// Each edge is a tuple (neighbor, edge_cost).
// h: heuristic function that maps node index -> estimated cost to goal.
pub fn astar(
    graph: &[Vec<(usize, usize)>],
    start: usize,
    goal: usize,
    h: impl Fn(usize) -> usize,
) -> Option<(usize, Vec<usize>)> {
    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<usize, usize> = HashMap::new();
    let mut g_score: HashMap<usize, usize> = HashMap::new();

    open.push(State {
        cost: h(start),
        position: start,
    });
    g_score.insert(start, 0);

    while let Some(State { cost: _, position }) = open.pop() {
        if position == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut current = goal;
            while let Some(&prev) = came_from.get(&current) {
                path.push(prev);
                current = prev;
            }
            path.reverse();
            return Some((*g_score.get(&goal).unwrap(), path));
        }

        for &(neighbor, edge_cost) in &graph[position] {
            let tentative_g = g_score.get(&position).unwrap() + edge_cost;

            if tentative_g < *g_score.get(&neighbor).unwrap_or(&usize::MAX) {
                came_from.insert(neighbor, position);
                g_score.insert(neighbor, tentative_g);
                let f = tentative_g + h(neighbor);
                open.push(State {
                    cost: f,
                    position: neighbor,
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astar_simple_graph() {
        // Graph:
        // 0 --1--> 1 --1--> 2
        //  \-------4--------> 2
        let graph = vec![
            vec![(1, 1), (2, 4)], // 0
            vec![(2, 1)],         // 1
            vec![],               // 2
        ];

        // Zero heuristic — behaves like Dijkstra
        let h = |_| 0;

        let result = astar(&graph, 0, 2, h);
        assert_eq!(result, Some((2, vec![0, 1, 2])));
    }

    #[test]
    fn test_astar_with_heuristic() {
        // Graph with cost = distance, use heuristic to bias the path
        let graph = vec![
            vec![(1, 1), (2, 5)], // 0
            vec![(2, 1)],         // 1
            vec![],               // 2
        ];

        // Misleading heuristic: encourages taking long path
        let h = |node: usize| match node {
            0 => 2,
            1 => 1,
            2 => 0,
            _ => 0,
        };

        let result = astar(&graph, 0, 2, h);
        assert_eq!(result, Some((2, vec![0, 1, 2])));
    }

    #[test]
    fn test_astar_disconnected() {
        let graph = vec![vec![], vec![], vec![]];
        let h = |_| 0;
        let result = astar(&graph, 0, 2, h);
        assert_eq!(result, None);
    }
}
