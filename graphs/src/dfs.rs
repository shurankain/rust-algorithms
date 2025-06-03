// Depth-First Search (DFS) implementation for an undirected graph.
// The graph is represented as an adjacency list: `Vec<Vec<usize>>`
pub fn dfs(graph: &Vec<Vec<usize>>, start: usize, visited: &mut Vec<bool>) {
    visited[start] = true;
    for &neighbor in &graph[start] {
        if !visited[neighbor] {
            dfs(graph, neighbor, visited);
        }
    }
}

// Returns the connected component starting from a given node
pub fn dfs_component(graph: &Vec<Vec<usize>>, start: usize) -> Vec<usize> {
    let mut visited = vec![false; graph.len()];
    let mut component = Vec::new();

    fn explore(v: usize, graph: &Vec<Vec<usize>>, visited: &mut Vec<bool>, comp: &mut Vec<usize>) {
        visited[v] = true;
        comp.push(v);
        for &u in &graph[v] {
            if !visited[u] {
                explore(u, graph, visited, comp);
            }
        }
    }

    explore(start, graph, &mut visited, &mut component);
    component
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfs_visits_all_connected_nodes() {
        // Graph: 0 - 1 - 2   3
        let graph = vec![
            vec![1],    // 0
            vec![0, 2], // 1
            vec![1],    // 2
            vec![],     // 3 (isolated node)
        ];

        let mut visited = vec![false; 4];
        dfs(&graph, 0, &mut visited);

        assert!(visited[0]);
        assert!(visited[1]);
        assert!(visited[2]);
        assert!(!visited[3]);
    }

    #[test]
    fn test_dfs_component_returns_correct_nodes() {
        // Graph: 0 - 1   2 - 3
        let graph = vec![
            vec![1], // 0
            vec![0], // 1
            vec![3], // 2
            vec![2], // 3
        ];

        let mut component = dfs_component(&graph, 2);
        component.sort();
        assert_eq!(component, vec![2, 3]);

        let mut component2 = dfs_component(&graph, 0);
        component2.sort();
        assert_eq!(component2, vec![0, 1]);
    }
}
