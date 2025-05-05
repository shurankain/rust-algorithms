use std::cmp::Ordering;

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Создаёт новую структуру для `size` элементов.
    /// Каждый элемент сначала в своей компоненте.
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(), // каждый сам себе родитель
            rank: vec![0; size],
        }
    }

    /// Возвращает представителя множества для `x`.
    /// Сжатие пути: ускоряет будущие find'ы.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Объединяет множества, содержащие `x` и `y`.
    /// Возвращает true, если произошло объединение.
    /// Если `x` и `y` уже в одной компоненте — возвращает false.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        match self.rank[root_x].cmp(&self.rank[root_y]) {
            Ordering::Greater => self.parent[root_y] = root_x,
            Ordering::Less => self.parent[root_x] = root_y,
            Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        true
    }
}
