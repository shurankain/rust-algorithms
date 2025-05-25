const INF: i32 = i32::MAX / 2; // preventing overflow in adding

pub fn floyd_warshall(weights: &[Vec<i32>]) -> (Vec<Vec<i32>>, Vec<Vec<Option<usize>>>) {
    let n = weights.len();
    let mut dist = weights.to_vec();
    let mut next = vec![vec![None; n]; n];

    // "next" table init
    for i in 0..n {
        for j in 0..n {
            if i != j && weights[i][j] < INF {
                next[i][j] = Some(j);
            }
        }
    }

    // Main implementation
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] + dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    (dist, next)
}

/// Restoration of path from "next" table
pub fn reconstruct_path(i: usize, j: usize, next: &[Vec<Option<usize>>]) -> Option<Vec<usize>> {
    next[i][j]?;

    let mut path = vec![i];
    let mut current = i;

    while current != j {
        current = next[current][j]?;
        path.push(current);
    }

    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floyd_warshall_basic() {
        let inf = INF;
        let graph = vec![vec![0, 3, inf], vec![inf, 0, 1], vec![2, inf, 0]];

        let (dist, next) = floyd_warshall(&graph);

        assert_eq!(dist[0][2], 4); // 0 → 1 → 2
        assert_eq!(dist[1][0], 3); // 1 → 2 → 0

        let path = reconstruct_path(0, 2, &next).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_negative_cycle_detection() {
        let graph = vec![vec![0, 1, INF], vec![INF, 0, -1], vec![-1, INF, 0]];

        let (dist, _) = floyd_warshall(&graph);

        // if и dist[i][i] < 0, than cycle is negative
        for (i, val) in dist.iter().enumerate().take(3) {
            assert!(val[i] < 0);
        }
    }

    #[test]
    fn test_unreachable() {
        let graph = vec![vec![0, INF], vec![INF, 0]];

        let (dist, next) = floyd_warshall(&graph);

        assert_eq!(dist[0][1], INF);
        assert!(reconstruct_path(0, 1, &next).is_none());
    }
}
