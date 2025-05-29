use std::collections::{HashMap, VecDeque};

// DFS-based Topological Sort
pub fn topo_sort_dfs(graph: &HashMap<usize, Vec<usize>>, n: usize) -> Option<Vec<usize>> {
    #[derive(Clone)]
    enum State {
        Unvisited,
        Visiting,
        Visited,
    }

    fn dfs(
        node: usize,
        graph: &HashMap<usize, Vec<usize>>,
        state: &mut Vec<State>,
        result: &mut Vec<usize>,
    ) -> bool {
        match state[node] {
            State::Visiting => return false, // found a cycle
            State::Visited => return true,
            State::Unvisited => {
                state[node] = State::Visiting;
                if let Some(neighbors) = graph.get(&node) {
                    for &neighbor in neighbors {
                        if !dfs(neighbor, graph, state, result) {
                            return false;
                        }
                    }
                }
                state[node] = State::Visited;
                result.push(node);
            }
        }
        true
    }

    let mut state = vec![State::Unvisited; n];
    let mut result = Vec::with_capacity(n);

    for node in 0..n {
        if !dfs(node, graph, &mut state, &mut result) {
            return None; // cycle detected
        }
    }

    result.reverse();
    Some(result)
}

// Kahn’s Algorithm for Topological Sort
pub fn topo_sort_kahn(graph: &HashMap<usize, Vec<usize>>, n: usize) -> Option<Vec<usize>> {
    let mut in_degree = vec![0; n];
    for neighbors in graph.values() {
        for &v in neighbors {
            in_degree[v] += 1;
        }
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut result = Vec::new();

    while let Some(node) = queue.pop_front() {
        result.push(node);
        if let Some(neighbors) = graph.get(&node) {
            for &v in neighbors {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    if result.len() == n {
        Some(result)
    } else {
        None // cycle detected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_graph() -> HashMap<usize, Vec<usize>> {
        let mut graph = HashMap::new();
        graph.insert(0, vec![1]);
        graph.insert(1, vec![2]);
        graph.insert(2, vec![3]);
        graph.insert(3, vec![]);
        graph
    }

    fn build_graph_with_cycle() -> HashMap<usize, Vec<usize>> {
        let mut graph = HashMap::new();
        graph.insert(0, vec![1]);
        graph.insert(1, vec![2]);
        graph.insert(2, vec![0]); // cycle
        graph
    }

    #[test]
    fn test_topo_sort_dfs_valid() {
        let graph = build_graph();
        let result = topo_sort_dfs(&graph, 4);
        assert_eq!(result, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn test_topo_sort_dfs_cycle() {
        let graph = build_graph_with_cycle();
        let result = topo_sort_dfs(&graph, 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_topo_sort_kahn_valid() {
        let graph = build_graph();
        let result = topo_sort_kahn(&graph, 4);
        assert_eq!(result, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn test_topo_sort_kahn_cycle() {
        let graph = build_graph_with_cycle();
        let result = topo_sort_kahn(&graph, 3);
        assert!(result.is_none());
    }
}
