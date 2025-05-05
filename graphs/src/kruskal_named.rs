// Данная реализация - алгоритм Крускала для не числовых вершин (см. тесты). Логика та же
use crate::union_find::UnionFind;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedEdge {
    pub from: String,
    pub to: String,
    pub weight: i32,
}

// Для сортировки по весу
impl Ord for NamedEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight.cmp(&other.weight)
    }
}

impl PartialOrd for NamedEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Kruskal для нечисловых (именованных) вершин.
/// Возвращает MST в виде кортежей (from, to, weight).
pub fn kruskal_named(edges: Vec<NamedEdge>) -> Vec<(String, String, i32)> {
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut id_counter = 0;

    // Назначаем индекс каждой уникальной вершине
    for edge in &edges {
        for node in [&edge.from, &edge.to] {
            if !map.contains_key(node) {
                map.insert(node.clone(), id_counter);
                id_counter += 1;
            }
        }
    }

    let mut dsu = UnionFind::new(map.len());
    let mut edges_sorted = edges;
    edges_sorted.sort();

    let mut result = Vec::new();

    for edge in edges_sorted {
        let u = map[&edge.from];
        let v = map[&edge.to];
        if dsu.union(u, v) {
            result.push((edge.from, edge.to, edge.weight));
        }
        if result.len() == map.len() - 1 {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_kruskal() {
        let edges = vec![
            NamedEdge {
                from: "A".into(),
                to: "B".into(),
                weight: 4,
            },
            NamedEdge {
                from: "A".into(),
                to: "C".into(),
                weight: 3,
            },
            NamedEdge {
                from: "B".into(),
                to: "C".into(),
                weight: 1,
            },
            NamedEdge {
                from: "B".into(),
                to: "D".into(),
                weight: 2,
            },
            NamedEdge {
                from: "C".into(),
                to: "D".into(),
                weight: 4,
            },
            NamedEdge {
                from: "D".into(),
                to: "E".into(),
                weight: 2,
            },
        ];

        let mst = kruskal_named(edges);
        let total_weight: i32 = mst.iter().map(|(_, _, w)| *w).sum();

        assert_eq!(mst.len(), 4); // 5 вершин → 4 ребра
        assert_eq!(total_weight, 8);
    }
}
