// Random Walk Algorithms for Graphs
//
// Random walks are fundamental for graph-based machine learning:
// - Node2Vec: Learns node embeddings via biased random walks
// - DeepWalk: Uses uniform random walks for node embeddings
// - Graph kernels: Random walk kernels for graph similarity
// - PageRank: Based on random walk with teleportation
//
// Types of random walks:
// 1. Uniform: Equal probability to all neighbors
// 2. Biased (Node2Vec): Control BFS vs DFS behavior with p, q parameters
// 3. Weighted: Based on edge weights
// 4. Temporal: Respecting time constraints
//
// References:
// - "node2vec: Scalable Feature Learning for Networks" (Grover & Leskovec, KDD 2016)
// - "DeepWalk: Online Learning of Social Representations" (Perozzi et al., KDD 2014)
// - "LINE: Large-scale Information Network Embedding" (Tang et al., WWW 2015)

use crate::message_passing::{Graph, NodeId};
use std::collections::HashMap;

/// Configuration for random walks
#[derive(Debug, Clone)]
pub struct RandomWalkConfig {
    /// Length of each walk
    pub walk_length: usize,
    /// Number of walks per node
    pub walks_per_node: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Walk strategy
    pub strategy: WalkStrategy,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        Self {
            walk_length: 80,
            walks_per_node: 10,
            seed: 42,
            strategy: WalkStrategy::Uniform,
        }
    }
}

impl RandomWalkConfig {
    /// Create config with specified walk length
    pub fn new(walk_length: usize) -> Self {
        Self {
            walk_length,
            ..Default::default()
        }
    }

    /// Set walks per node
    pub fn with_walks_per_node(mut self, walks: usize) -> Self {
        self.walks_per_node = walks;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set walk strategy
    pub fn with_strategy(mut self, strategy: WalkStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Create Node2Vec config with p, q parameters
    pub fn node2vec(walk_length: usize, p: f64, q: f64) -> Self {
        Self {
            walk_length,
            walks_per_node: 10,
            seed: 42,
            strategy: WalkStrategy::Node2Vec { p, q },
        }
    }
}

/// Walk strategy for random walks
#[derive(Debug, Clone, Copy)]
pub enum WalkStrategy {
    /// Uniform random walk (DeepWalk style)
    Uniform,
    /// Node2Vec biased walk with return parameter p and in-out parameter q
    /// - p: Return parameter (high p = less likely to return)
    /// - q: In-out parameter (q > 1 = BFS-like, q < 1 = DFS-like)
    Node2Vec { p: f64, q: f64 },
    /// Weighted by edge weights (if available)
    Weighted,
}

/// Simple deterministic pseudo-random number generator
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    /// Generate next random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate next random usize in [0, max)
    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

/// A single random walk (sequence of nodes)
pub type Walk = Vec<NodeId>;

/// Random walk generator
#[derive(Debug, Clone)]
pub struct RandomWalker {
    /// Configuration
    config: RandomWalkConfig,
    /// Random number generator
    rng: SimpleRng,
}

impl RandomWalker {
    /// Create a new random walker
    pub fn new(config: RandomWalkConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self { config, rng }
    }

    /// Create with default config
    pub fn default_walker() -> Self {
        Self::new(RandomWalkConfig::default())
    }

    /// Create a DeepWalk-style walker
    pub fn deepwalk(walk_length: usize, walks_per_node: usize) -> Self {
        let config = RandomWalkConfig::new(walk_length)
            .with_walks_per_node(walks_per_node)
            .with_strategy(WalkStrategy::Uniform);
        Self::new(config)
    }

    /// Create a Node2Vec-style walker
    pub fn node2vec(walk_length: usize, walks_per_node: usize, p: f64, q: f64) -> Self {
        let config = RandomWalkConfig::new(walk_length)
            .with_walks_per_node(walks_per_node)
            .with_strategy(WalkStrategy::Node2Vec { p, q });
        Self::new(config)
    }

    /// Reset random seed
    pub fn reset_seed(&mut self, seed: u64) {
        self.rng = SimpleRng::new(seed);
    }

    /// Perform a single random walk starting from a node
    pub fn walk(&mut self, graph: &Graph, start: NodeId) -> Walk {
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(start);

        if graph.neighbors(start).is_empty() {
            return walk;
        }

        // First step: uniform random
        let first_neighbors = graph.neighbors(start);
        if first_neighbors.is_empty() {
            return walk;
        }
        let first_idx = self.rng.next_usize(first_neighbors.len());
        let (first_next, _) = first_neighbors[first_idx];
        walk.push(first_next);

        // Subsequent steps
        for _ in 2..self.config.walk_length {
            let current = *walk.last().unwrap();
            let prev = walk[walk.len() - 2];

            let next = match self.config.strategy {
                WalkStrategy::Uniform => self.uniform_step(graph, current),
                WalkStrategy::Node2Vec { p, q } => self.node2vec_step(graph, current, prev, p, q),
                WalkStrategy::Weighted => self.weighted_step(graph, current),
            };

            match next {
                Some(node) => walk.push(node),
                None => break, // Dead end
            }
        }

        walk
    }

    /// Uniform random step
    fn uniform_step(&mut self, graph: &Graph, current: NodeId) -> Option<NodeId> {
        let neighbors = graph.neighbors(current);
        if neighbors.is_empty() {
            return None;
        }
        let idx = self.rng.next_usize(neighbors.len());
        Some(neighbors[idx].0)
    }

    /// Node2Vec biased step
    fn node2vec_step(
        &mut self,
        graph: &Graph,
        current: NodeId,
        prev: NodeId,
        p: f64,
        q: f64,
    ) -> Option<NodeId> {
        let neighbors = graph.neighbors(current);
        if neighbors.is_empty() {
            return None;
        }

        // Get neighbors of previous node for distance calculation
        let prev_neighbors: std::collections::HashSet<NodeId> =
            graph.neighbors(prev).iter().map(|&(n, _)| n).collect();

        // Compute unnormalized probabilities
        let mut weights: Vec<f64> = Vec::with_capacity(neighbors.len());

        for &(neighbor, _) in neighbors {
            let weight = if neighbor == prev {
                // Return to previous node
                1.0 / p
            } else if prev_neighbors.contains(&neighbor) {
                // Distance 1 from prev (BFS-like)
                1.0
            } else {
                // Distance 2 from prev (DFS-like)
                1.0 / q
            };
            weights.push(weight);
        }

        // Sample from distribution
        let total: f64 = weights.iter().sum();
        let r = self.rng.next_f64() * total;

        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                return Some(neighbors[i].0);
            }
        }

        // Fallback (shouldn't happen)
        Some(neighbors.last()?.0)
    }

    /// Weighted step (uses node degree as proxy for edge weight)
    fn weighted_step(&mut self, graph: &Graph, current: NodeId) -> Option<NodeId> {
        let neighbors = graph.neighbors(current);
        if neighbors.is_empty() {
            return None;
        }

        // Weight by neighbor's degree
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&(n, _)| (graph.degree(n) as f64).sqrt())
            .collect();

        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            return self.uniform_step(graph, current);
        }

        let r = self.rng.next_f64() * total;
        let mut cumulative = 0.0;

        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                return Some(neighbors[i].0);
            }
        }

        Some(neighbors.last()?.0)
    }

    /// Generate all walks for a single node
    pub fn walks_from_node(&mut self, graph: &Graph, node: NodeId) -> Vec<Walk> {
        (0..self.config.walks_per_node)
            .map(|_| self.walk(graph, node))
            .collect()
    }

    /// Generate walks for all nodes in the graph
    pub fn walks_for_all_nodes(&mut self, graph: &Graph) -> Vec<Walk> {
        let mut all_walks = Vec::new();

        for _ in 0..self.config.walks_per_node {
            for node in graph.nodes() {
                let walk = self.walk(graph, node);
                all_walks.push(walk);
            }
        }

        all_walks
    }

    /// Generate walks for a subset of nodes
    pub fn walks_for_nodes(&mut self, graph: &Graph, nodes: &[NodeId]) -> Vec<Walk> {
        let mut all_walks = Vec::new();

        for _ in 0..self.config.walks_per_node {
            for &node in nodes {
                let walk = self.walk(graph, node);
                all_walks.push(walk);
            }
        }

        all_walks
    }
}

/// Statistics about random walks
#[derive(Debug, Clone, Default)]
pub struct WalkStats {
    /// Total number of walks
    pub num_walks: usize,
    /// Average walk length
    pub avg_length: f64,
    /// Minimum walk length
    pub min_length: usize,
    /// Maximum walk length
    pub max_length: usize,
    /// Node visit counts
    pub visit_counts: HashMap<NodeId, usize>,
}

/// Analyze a set of walks
pub fn analyze_walks(walks: &[Walk]) -> WalkStats {
    if walks.is_empty() {
        return WalkStats::default();
    }

    let mut visit_counts: HashMap<NodeId, usize> = HashMap::new();
    let mut total_length = 0;
    let mut min_length = usize::MAX;
    let mut max_length = 0;

    for walk in walks {
        let len = walk.len();
        total_length += len;
        min_length = min_length.min(len);
        max_length = max_length.max(len);

        for &node in walk {
            *visit_counts.entry(node).or_insert(0) += 1;
        }
    }

    WalkStats {
        num_walks: walks.len(),
        avg_length: total_length as f64 / walks.len() as f64,
        min_length,
        max_length,
        visit_counts,
    }
}

/// Compute node visit frequency from walks
pub fn compute_visit_frequency(walks: &[Walk], num_nodes: usize) -> Vec<f64> {
    let mut counts = vec![0usize; num_nodes];
    let mut total = 0usize;

    for walk in walks {
        for &node in walk {
            if node < num_nodes {
                counts[node] += 1;
                total += 1;
            }
        }
    }

    if total == 0 {
        return vec![0.0; num_nodes];
    }

    counts.iter().map(|&c| c as f64 / total as f64).collect()
}

/// Compute co-occurrence matrix from walks (within a window)
pub fn compute_cooccurrence(
    walks: &[Walk],
    num_nodes: usize,
    window_size: usize,
) -> Vec<Vec<usize>> {
    let mut cooccur = vec![vec![0usize; num_nodes]; num_nodes];

    for walk in walks {
        for (i, &center) in walk.iter().enumerate() {
            if center >= num_nodes {
                continue;
            }

            // Look at context window
            let start = i.saturating_sub(window_size);
            let end = (i + window_size + 1).min(walk.len());

            for (j, &context) in walk.iter().enumerate().take(end).skip(start) {
                if i != j && context < num_nodes {
                    cooccur[center][context] += 1;
                }
            }
        }
    }

    cooccur
}

/// Compute transition probabilities from walks
pub fn compute_transition_matrix(walks: &[Walk], num_nodes: usize) -> Vec<Vec<f64>> {
    let mut transitions = vec![vec![0usize; num_nodes]; num_nodes];

    for walk in walks {
        for window in walk.windows(2) {
            let from = window[0];
            let to = window[1];
            if from < num_nodes && to < num_nodes {
                transitions[from][to] += 1;
            }
        }
    }

    // Normalize to probabilities
    transitions
        .into_iter()
        .map(|row| {
            let sum: usize = row.iter().sum();
            if sum == 0 {
                vec![0.0; num_nodes]
            } else {
                row.iter().map(|&c| c as f64 / sum as f64).collect()
            }
        })
        .collect()
}

/// Random walk with restart (for Personalized PageRank)
#[derive(Debug, Clone)]
pub struct RandomWalkWithRestart {
    /// Restart probability (alpha in PPR)
    restart_prob: f64,
    /// Random number generator
    rng: SimpleRng,
}

impl RandomWalkWithRestart {
    /// Create a new RWR walker
    pub fn new(restart_prob: f64, seed: u64) -> Self {
        Self {
            restart_prob,
            rng: SimpleRng::new(seed),
        }
    }

    /// Perform a random walk with restart
    pub fn walk(&mut self, graph: &Graph, start: NodeId, max_steps: usize) -> Walk {
        let mut walk = vec![start];
        let mut current = start;

        for _ in 1..max_steps {
            // Check for restart
            if self.rng.next_f64() < self.restart_prob {
                current = start;
                walk.push(current);
                continue;
            }

            // Take a step
            let neighbors = graph.neighbors(current);
            if neighbors.is_empty() {
                current = start; // Restart from dead end
                walk.push(current);
                continue;
            }

            let idx = self.rng.next_usize(neighbors.len());
            current = neighbors[idx].0;
            walk.push(current);
        }

        walk
    }

    /// Estimate PPR scores via random walks
    pub fn estimate_ppr(
        &mut self,
        graph: &Graph,
        start: NodeId,
        num_walks: usize,
        walk_length: usize,
    ) -> Vec<f64> {
        let num_nodes = graph.num_nodes();
        let mut visit_counts = vec![0usize; num_nodes];

        for _ in 0..num_walks {
            let walk = self.walk(graph, start, walk_length);
            for &node in &walk {
                if node < num_nodes {
                    visit_counts[node] += 1;
                }
            }
        }

        let total: usize = visit_counts.iter().sum();
        if total == 0 {
            return vec![0.0; num_nodes];
        }

        visit_counts
            .iter()
            .map(|&c| c as f64 / total as f64)
            .collect()
    }
}

/// Generate skip-gram training pairs from walks (for Node2Vec)
pub fn generate_skipgram_pairs(walks: &[Walk], window_size: usize) -> Vec<(NodeId, NodeId)> {
    let mut pairs = Vec::new();

    for walk in walks {
        for (i, &center) in walk.iter().enumerate() {
            let start = i.saturating_sub(window_size);
            let end = (i + window_size + 1).min(walk.len());

            for (j, &context) in walk.iter().enumerate().take(end).skip(start) {
                if i != j {
                    pairs.push((center, context));
                }
            }
        }
    }

    pairs
}

/// Negative sampling for skip-gram (based on node frequency)
pub fn negative_sample(
    frequency: &[f64],
    num_samples: usize,
    exclude: NodeId,
    rng: &mut SimpleRng,
) -> Vec<NodeId> {
    let mut samples = Vec::with_capacity(num_samples);
    let num_nodes = frequency.len();

    if num_nodes == 0 {
        return samples;
    }

    // Use frequency^0.75 for negative sampling (as in Word2Vec)
    let adjusted: Vec<f64> = frequency.iter().map(|&f| f.powf(0.75)).collect();
    let total: f64 = adjusted.iter().sum();

    if total == 0.0 {
        return samples;
    }

    let mut attempts = 0;
    while samples.len() < num_samples && attempts < num_samples * 10 {
        let r = rng.next_f64() * total;
        let mut cumulative = 0.0;

        for (node, &prob) in adjusted.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                if node != exclude && !samples.contains(&node) {
                    samples.push(node);
                }
                break;
            }
        }
        attempts += 1;
    }

    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_passing::Graph;

    fn create_line_graph() -> Graph {
        // 0 -- 1 -- 2 -- 3 -- 4
        let mut graph = Graph::undirected(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph
    }

    fn create_star_graph() -> Graph {
        // 0 is center, connected to 1, 2, 3, 4
        let mut graph = Graph::undirected(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(0, 4);
        graph
    }

    fn create_ring_graph() -> Graph {
        // 0 -- 1 -- 2 -- 3 -- 4 -- 0
        let mut graph = Graph::undirected(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 0);
        graph
    }

    #[test]
    fn test_random_walk_config() {
        let config = RandomWalkConfig::new(100)
            .with_walks_per_node(5)
            .with_seed(123);

        assert_eq!(config.walk_length, 100);
        assert_eq!(config.walks_per_node, 5);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_node2vec_config() {
        let config = RandomWalkConfig::node2vec(80, 2.0, 0.5);

        assert_eq!(config.walk_length, 80);
        match config.strategy {
            WalkStrategy::Node2Vec { p, q } => {
                assert!((p - 2.0).abs() < 1e-6);
                assert!((q - 0.5).abs() < 1e-6);
            }
            _ => panic!("Wrong strategy"),
        }
    }

    #[test]
    fn test_uniform_walk() {
        let graph = create_line_graph();
        let config = RandomWalkConfig::new(10).with_strategy(WalkStrategy::Uniform);
        let mut walker = RandomWalker::new(config);

        let walk = walker.walk(&graph, 2);

        assert!(!walk.is_empty());
        assert_eq!(walk[0], 2); // Starts at node 2
        assert!(walk.len() <= 10);

        // All nodes should be valid
        for &node in &walk {
            assert!(node < 5);
        }
    }

    #[test]
    fn test_walk_from_isolated_node() {
        let graph = Graph::undirected(3); // No edges
        let mut walker = RandomWalker::default_walker();

        let walk = walker.walk(&graph, 0);

        // Should just contain the start node
        assert_eq!(walk.len(), 1);
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn test_walk_respects_length() {
        let graph = create_ring_graph();
        let config = RandomWalkConfig::new(20);
        let mut walker = RandomWalker::new(config);

        let walk = walker.walk(&graph, 0);

        assert_eq!(walk.len(), 20);
    }

    #[test]
    fn test_walks_from_node() {
        let graph = create_ring_graph();
        let config = RandomWalkConfig::new(5).with_walks_per_node(3);
        let mut walker = RandomWalker::new(config);

        let walks = walker.walks_from_node(&graph, 0);

        assert_eq!(walks.len(), 3);
        for walk in &walks {
            assert_eq!(walk[0], 0);
        }
    }

    #[test]
    fn test_walks_for_all_nodes() {
        let graph = create_ring_graph();
        let config = RandomWalkConfig::new(5).with_walks_per_node(2);
        let mut walker = RandomWalker::new(config);

        let walks = walker.walks_for_all_nodes(&graph);

        // 5 nodes * 2 walks = 10 walks
        assert_eq!(walks.len(), 10);
    }

    #[test]
    fn test_node2vec_walk() {
        let graph = create_ring_graph();
        let mut walker = RandomWalker::node2vec(10, 2, 1.0, 1.0);

        let walk = walker.walk(&graph, 0);

        assert!(!walk.is_empty());
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn test_node2vec_bfs_like() {
        // q > 1 should be BFS-like (explore locally)
        let graph = create_star_graph();
        let mut walker = RandomWalker::node2vec(20, 5, 1.0, 4.0);

        let walks = walker.walks_from_node(&graph, 0);

        // With BFS-like behavior, should frequently return to center
        let center_visits: usize = walks
            .iter()
            .map(|w| w.iter().filter(|&&n| n == 0).count())
            .sum();

        assert!(center_visits > 0);
    }

    #[test]
    fn test_weighted_walk() {
        let graph = create_ring_graph();
        let config = RandomWalkConfig::new(10).with_strategy(WalkStrategy::Weighted);
        let mut walker = RandomWalker::new(config);

        let walk = walker.walk(&graph, 0);

        assert!(!walk.is_empty());
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn test_reproducible_walks() {
        let graph = create_ring_graph();

        let config1 = RandomWalkConfig::new(10).with_seed(42);
        let mut walker1 = RandomWalker::new(config1);

        let config2 = RandomWalkConfig::new(10).with_seed(42);
        let mut walker2 = RandomWalker::new(config2);

        let walk1 = walker1.walk(&graph, 0);
        let walk2 = walker2.walk(&graph, 0);

        assert_eq!(walk1, walk2);
    }

    #[test]
    fn test_analyze_walks() {
        let graph = create_ring_graph();
        let mut walker = RandomWalker::deepwalk(10, 5);

        let walks = walker.walks_for_all_nodes(&graph);
        let stats = analyze_walks(&walks);

        assert_eq!(stats.num_walks, 25); // 5 nodes * 5 walks
        assert!(stats.avg_length > 0.0);
        assert!(stats.min_length > 0);
        assert!(stats.max_length <= 10);
    }

    #[test]
    fn test_visit_frequency() {
        let graph = create_star_graph();
        let mut walker = RandomWalker::deepwalk(20, 10);

        let walks = walker.walks_for_all_nodes(&graph);
        let freq = compute_visit_frequency(&walks, 5);

        // Center node (0) should have high frequency
        assert!(freq[0] > 0.0);

        // Sum should be 1
        let sum: f64 = freq.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cooccurrence() {
        let graph = create_line_graph();
        let mut walker = RandomWalker::deepwalk(5, 10);

        let walks = walker.walks_for_all_nodes(&graph);
        let cooccur = compute_cooccurrence(&walks, 5, 2);

        // Adjacent nodes should co-occur
        assert!(cooccur[1][0] > 0 || cooccur[0][1] > 0);
        assert!(cooccur[1][2] > 0 || cooccur[2][1] > 0);
    }

    #[test]
    fn test_transition_matrix() {
        let graph = create_line_graph();
        let mut walker = RandomWalker::deepwalk(10, 20);

        let walks = walker.walks_for_all_nodes(&graph);
        let trans = compute_transition_matrix(&walks, 5);

        // From node 2, can go to 1 or 3
        // Rows should sum to 1 (or 0 if no transitions)
        for row in &trans {
            let sum: f64 = row.iter().sum();
            assert!(sum < 1e-6 || (sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rwr() {
        let graph = create_ring_graph();
        let mut rwr = RandomWalkWithRestart::new(0.15, 42);

        let walk = rwr.walk(&graph, 0, 50);

        assert!(!walk.is_empty());
        assert_eq!(walk[0], 0);

        // Should contain restarts (returns to start)
        let start_visits = walk.iter().filter(|&&n| n == 0).count();
        assert!(start_visits >= 1);
    }

    #[test]
    fn test_estimate_ppr() {
        let graph = create_star_graph();
        let mut rwr = RandomWalkWithRestart::new(0.15, 42);

        let ppr = rwr.estimate_ppr(&graph, 0, 100, 50);

        // Start node should have high PPR score
        assert!(ppr[0] > 0.0);

        // Sum should be ~1
        let sum: f64 = ppr.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_skipgram_pairs() {
        let walks = vec![vec![0, 1, 2, 3, 4]];

        let pairs = generate_skipgram_pairs(&walks, 2);

        // Window size 2: each node sees 2 neighbors on each side
        assert!(!pairs.is_empty());

        // Should include (1, 0), (1, 2), (1, 3) etc.
        assert!(pairs.contains(&(1, 0)));
        assert!(pairs.contains(&(1, 2)));
    }

    #[test]
    fn test_negative_sample() {
        let frequency = vec![0.1, 0.2, 0.3, 0.2, 0.2];
        let mut rng = SimpleRng::new(42);

        let samples = negative_sample(&frequency, 3, 0, &mut rng);

        assert_eq!(samples.len(), 3);
        assert!(!samples.contains(&0)); // Excluded node
    }

    #[test]
    fn test_deepwalk_factory() {
        let walker = RandomWalker::deepwalk(80, 10);

        assert_eq!(walker.config.walk_length, 80);
        assert_eq!(walker.config.walks_per_node, 10);
        matches!(walker.config.strategy, WalkStrategy::Uniform);
    }

    #[test]
    fn test_node2vec_factory() {
        let walker = RandomWalker::node2vec(40, 5, 0.5, 2.0);

        assert_eq!(walker.config.walk_length, 40);
        assert_eq!(walker.config.walks_per_node, 5);
    }

    #[test]
    fn test_walks_for_subset() {
        let graph = create_ring_graph();
        let mut walker = RandomWalker::deepwalk(5, 2);

        let walks = walker.walks_for_nodes(&graph, &[0, 1]);

        // 2 nodes * 2 walks = 4 walks
        assert_eq!(walks.len(), 4);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::undirected(0);
        let mut walker = RandomWalker::default_walker();

        let walks = walker.walks_for_all_nodes(&graph);

        assert!(walks.is_empty());
    }

    #[test]
    fn test_reset_seed() {
        let graph = create_ring_graph();
        let mut walker = RandomWalker::deepwalk(10, 1);

        let walk1 = walker.walk(&graph, 0);

        walker.reset_seed(walker.config.seed);
        let walk2 = walker.walk(&graph, 0);

        assert_eq!(walk1, walk2);
    }
}
