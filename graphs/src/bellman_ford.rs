const INF: i32 = i32::MAX / 2; // prevent overflow

// Bellman-Ford: returns distance vector or None if negative cycle detected
pub fn bellman_ford(n: usize, edges: &[(usize, usize, i32)], start: usize) -> Option<Vec<i32>> {
    let mut dist = vec![INF; n];
    dist[start] = 0;

    for _ in 0..n - 1 {
        for &(u, v, w) in edges {
            if dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
            }
        }
    }

    // Check for negative cycles
    for &(u, v, w) in edges {
        if dist[u] + w < dist[v] {
            return None; // negative cycle detected
        }
    }

    Some(dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_graph() {
        // 0 --5--> 1, 1 ---3--> 2, 0 --4--> 2
        let edges = vec![(0, 1, 5), (1, 2, -3), (0, 2, 4)];
        let result = bellman_ford(3, &edges, 0);
        assert_eq!(result, Some(vec![0, 5, 2]));
    }

    #[test]
    fn test_negative_cycle() {
        // 0 --1--> 1, 1 --1--> 2, 2 ---3--> 0 → total cycle weight = -1
        let edges = vec![(0, 1, 1), (1, 2, 1), (2, 0, -3)];
        let result = bellman_ford(3, &edges, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_disconnected() {
        let edges = vec![(0, 1, 2)];
        let result = bellman_ford(3, &edges, 0);
        assert_eq!(result, Some(vec![0, 2, INF]));
    }
}
