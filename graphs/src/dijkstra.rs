use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: usize,
    position: usize,
}

// Min-heap ordering by reversing the comparison
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // reverse for min-heap
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Edge in the graph: (destination, weight)
type Edge = (usize, usize);

// Adjacency list representation
pub fn dijkstra(adj_list: &[Vec<Edge>], start: usize) -> Vec<usize> {
    let n = adj_list.len();
    let mut dist = vec![usize::MAX; n];
    dist[start] = 0;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0,
        position: start,
    });

    while let Some(State { cost, position }) = heap.pop() {
        // Skip if we already found a better path
        if cost > dist[position] {
            continue;
        }

        for &(neighbor, weight) in &adj_list[position] {
            let next = State {
                cost: cost + weight,
                position: neighbor,
            };

            if next.cost < dist[neighbor] {
                dist[neighbor] = next.cost;
                heap.push(next);
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra_basic() {
        let graph = vec![vec![(1, 2), (2, 4)], vec![(2, 3)], vec![]];

        let dist = dijkstra(&graph, 0);
        assert_eq!(dist, vec![0, 2, 4]);
    }

    #[test]
    fn test_unreachable_node() {
        // Node 3 is disconnected
        let graph = vec![vec![(1, 1)], vec![(2, 2)], vec![], vec![]];

        let dist = dijkstra(&graph, 0);
        assert_eq!(dist[3], usize::MAX);
    }
}
