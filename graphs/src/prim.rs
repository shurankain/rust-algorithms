use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: i32,
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight.cmp(&other.weight)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Returns MST.
pub fn prim(n_vertices: usize, mut edges: Vec<Edge>) -> Vec<Edge> {
    edges.sort(); // by weight ascending

    let mut mst = Vec::new();
    let mut nodes_visited: Vec<bool> = vec![false; n_vertices];
    nodes_visited[0] = true;

    let mut heap: BinaryHeap<Reverse<Edge>> = BinaryHeap::with_capacity(edges.len());
    for edge in &edges {
        if edge.from == 0 {
            heap.push(Reverse(*edge));
        }
    }

    while !heap.is_empty() {
        let current_edge = heap.pop().unwrap().0;
        if nodes_visited[current_edge.to] {
            continue;
        }
        mst.push(current_edge);
        nodes_visited[current_edge.to] = true;
        for edge in &edges {
            if edge.from == current_edge.to && !nodes_visited[edge.to] {
                heap.push(Reverse(*edge));
            }
        }
    }

    mst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prim_bidirectional() {
        let edges = vec![
            Edge {
                from: 0,
                to: 1,
                weight: 4,
            },
            Edge {
                from: 1,
                to: 0,
                weight: 4,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 3,
            },
            Edge {
                from: 2,
                to: 0,
                weight: 3,
            },
            Edge {
                from: 1,
                to: 2,
                weight: 1,
            },
            Edge {
                from: 2,
                to: 1,
                weight: 1,
            },
            Edge {
                from: 1,
                to: 3,
                weight: 2,
            },
            Edge {
                from: 3,
                to: 1,
                weight: 2,
            },
            Edge {
                from: 2,
                to: 3,
                weight: 4,
            },
            Edge {
                from: 3,
                to: 2,
                weight: 4,
            },
            Edge {
                from: 3,
                to: 4,
                weight: 2,
            },
            Edge {
                from: 4,
                to: 3,
                weight: 2,
            },
            Edge {
                from: 4,
                to: 5,
                weight: 6,
            },
            Edge {
                from: 5,
                to: 4,
                weight: 6,
            },
        ];

        let mst = prim(6, edges);
        let total_weight: i32 = mst.iter().map(|e| e.weight).sum();

        assert_eq!(mst.len(), 5); // 6 nodes → 5 edges
        assert_eq!(total_weight, 14);
    }
}
