// Данная реализация - алгоритм Крускала для числовых вершин (см. тесты). Логика та же
use crate::union_find::UnionFind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: i32,
}

// Для сортировки рёбер по весу
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

/// Принимает количество вершин и вектор рёбер.
/// Возвращает вектор рёбер, образующих минимальное остовное дерево (MST).
pub fn kruskal(n_vertices: usize, mut edges: Vec<Edge>) -> Vec<Edge> {
    edges.sort(); // по возрастанию веса

    let mut uf = UnionFind::new(n_vertices);
    let mut mst = Vec::new();

    for edge in edges {
        if uf.union(edge.from, edge.to) {
            mst.push(edge);
        }
        if mst.len() == n_vertices - 1 {
            break; // достаточно рёбер
        }
    }

    mst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kruskal() {
        let edges = vec![
            Edge {
                from: 0,
                to: 1,
                weight: 4,
            },
            Edge {
                from: 0,
                to: 2,
                weight: 3,
            },
            Edge {
                from: 1,
                to: 2,
                weight: 1,
            },
            Edge {
                from: 1,
                to: 3,
                weight: 2,
            },
            Edge {
                from: 2,
                to: 3,
                weight: 4,
            },
            Edge {
                from: 3,
                to: 4,
                weight: 2,
            },
            Edge {
                from: 4,
                to: 5,
                weight: 6,
            },
        ];

        let mst = kruskal(6, edges);
        let total_weight: i32 = mst.iter().map(|e| e.weight).sum();

        assert_eq!(mst.len(), 5); // для 6 вершин
        assert_eq!(total_weight, 14);
    }
}
