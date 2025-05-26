use std::collections::VecDeque;

// Returns vector of distances from `start` to all other nodes (or None, if non-reachable).
pub fn bfs_shortest_distances(graph: &[Vec<usize>], start: usize) -> Vec<Option<usize>> {
    let n = graph.len();
    let mut distances = vec![None; n];
    let mut queue = VecDeque::new();

    distances[start] = Some(0);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        for &neighbor in &graph[node] {
            if distances[neighbor].is_none() {
                distances[neighbor] = Some(distances[node].unwrap() + 1);
                queue.push_back(neighbor);
            }
        }
    }

    distances
}

/// Returns shortest path from `start` to `end` (if exists).
pub fn bfs_shortest_path(graph: &[Vec<usize>], start: usize, end: usize) -> Option<Vec<usize>> {
    let n = graph.len();
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == end {
            break;
        }

        for &neighbor in &graph[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }

    if !visited[end] {
        return None;
    }

    // Restoration of the path from the end to beginning
    let mut path = Vec::new();
    let mut current = end;
    while let Some(p) = parent[current] {
        path.push(current);
        current = p;
    }
    path.push(start);
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_distances() {
        let graph = vec![vec![1, 2], vec![0, 3, 4], vec![0], vec![1], vec![1]];
        let dist = bfs_shortest_distances(&graph, 0);

        assert_eq!(dist[0], Some(0));
        assert_eq!(dist[1], Some(1));
        assert_eq!(dist[2], Some(1));
        assert_eq!(dist[3], Some(2));
        assert_eq!(dist[4], Some(2));
    }

    #[test]
    fn test_bfs_shortest_path_unreachable() {
        let graph = vec![
            vec![1], // 0
            vec![0], // 1
            vec![],  // 2 (isolated)
            vec![],  // 3 (isolated)
        ];

        let unreachable = bfs_shortest_path(&graph, 0, 3);
        assert!(unreachable.is_none());
    }
}
