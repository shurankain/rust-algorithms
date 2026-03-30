// Personalized PageRank (PPR) Algorithms
//
// PageRank and its personalized variant are fundamental for:
// - Node importance ranking in graphs
// - Recommendation systems (personalized recommendations)
// - Graph-based semi-supervised learning
// - Feature propagation in GNNs (APPNP, etc.)
// - Community detection and graph clustering
//
// Key variants:
// 1. Standard PageRank: Global importance measure
// 2. Personalized PageRank (PPR): Importance relative to seed nodes
// 3. Topic-Sensitive PageRank: Multiple topic-specific rankings
// 4. Approximate PPR: Efficient local computation
//
// References:
// - "The PageRank Citation Ranking" (Page et al., 1999)
// - "Topic-Sensitive PageRank" (Haveliwala, 2002)
// - "Local Graph Partitioning using PageRank Vectors" (Andersen et al., 2006)
// - "Predict then Propagate: Graph Neural Networks meet Personalized PageRank" (APPNP, ICLR 2019)

use crate::message_passing::{Graph, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for PageRank computation
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (probability of following an edge vs teleporting)
    /// Typical value: 0.85
    pub damping: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to handle dangling nodes (nodes with no outgoing edges)
    pub handle_dangling: bool,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            handle_dangling: true,
        }
    }
}

impl PageRankConfig {
    /// Create config with custom damping factor
    pub fn new(damping: f64) -> Self {
        Self {
            damping,
            ..Default::default()
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set dangling node handling
    pub fn with_handle_dangling(mut self, handle: bool) -> Self {
        self.handle_dangling = handle;
        self
    }
}

/// Result of PageRank computation
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// PageRank scores for each node
    pub scores: Vec<f64>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final convergence delta
    pub convergence: f64,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl PageRankResult {
    /// Get the score for a specific node
    pub fn score(&self, node: NodeId) -> f64 {
        if node < self.scores.len() {
            self.scores[node]
        } else {
            0.0
        }
    }

    /// Get top-k nodes by PageRank score
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f64)> {
        let mut indexed: Vec<(NodeId, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Get nodes above a score threshold
    pub fn above_threshold(&self, threshold: f64) -> Vec<(NodeId, f64)> {
        self.scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s >= threshold)
            .map(|(i, &s)| (i, s))
            .collect()
    }

    /// Normalize scores to sum to 1
    pub fn normalized(&self) -> Vec<f64> {
        let sum: f64 = self.scores.iter().sum();
        if sum > 0.0 {
            self.scores.iter().map(|&s| s / sum).collect()
        } else {
            self.scores.clone()
        }
    }
}

/// Standard PageRank computation using power iteration
pub struct PageRank {
    config: PageRankConfig,
}

impl PageRank {
    /// Create a new PageRank instance
    pub fn new(config: PageRankConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PageRankConfig::default())
    }

    /// Compute PageRank scores for all nodes
    pub fn compute(&self, graph: &Graph) -> PageRankResult {
        let n = graph.num_nodes();
        if n == 0 {
            return PageRankResult {
                scores: Vec::new(),
                iterations: 0,
                convergence: 0.0,
                converged: true,
            };
        }

        // Initialize scores uniformly
        let mut scores = vec![1.0 / n as f64; n];
        let mut new_scores = vec![0.0; n];

        // Precompute out-degrees
        let out_degrees: Vec<usize> = (0..n).map(|i| graph.degree(i)).collect();

        // Find dangling nodes (no outgoing edges)
        let dangling: Vec<NodeId> = out_degrees
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let teleport = (1.0 - self.config.damping) / n as f64;
        let mut convergence = f64::MAX;
        let mut iterations = 0;

        while iterations < self.config.max_iterations && convergence > self.config.tolerance {
            // Handle dangling nodes: redistribute their rank equally
            let dangling_sum: f64 = if self.config.handle_dangling {
                dangling.iter().map(|&d| scores[d]).sum()
            } else {
                0.0
            };
            let dangling_contrib = self.config.damping * dangling_sum / n as f64;

            // Compute new scores
            for (i, new_score) in new_scores.iter_mut().enumerate() {
                let mut rank_sum = 0.0;

                // Sum contributions from incoming edges
                // For undirected graphs, neighbors are both in and out
                for &(neighbor, _) in graph.neighbors(i) {
                    if out_degrees[neighbor] > 0 {
                        rank_sum += scores[neighbor] / out_degrees[neighbor] as f64;
                    }
                }

                *new_score = teleport + self.config.damping * rank_sum + dangling_contrib;
            }

            // Compute convergence (L1 norm of change)
            convergence = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(&old, &new)| (old - new).abs())
                .sum();

            std::mem::swap(&mut scores, &mut new_scores);
            iterations += 1;
        }

        PageRankResult {
            scores,
            iterations,
            convergence,
            converged: convergence <= self.config.tolerance,
        }
    }
}

/// Personalized PageRank (PPR) computation
///
/// PPR computes importance scores relative to a set of seed/source nodes.
/// This is useful for:
/// - Finding nodes relevant to a query node
/// - Local community detection
/// - Recommendation systems
pub struct PersonalizedPageRank {
    config: PageRankConfig,
}

impl PersonalizedPageRank {
    /// Create a new PPR instance
    pub fn new(config: PageRankConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PageRankConfig::default())
    }

    /// Compute PPR scores from a single source node
    pub fn compute_from_node(&self, graph: &Graph, source: NodeId) -> PageRankResult {
        self.compute_from_nodes(graph, &[source])
    }

    /// Compute PPR scores from multiple source nodes (uniform distribution)
    pub fn compute_from_nodes(&self, graph: &Graph, sources: &[NodeId]) -> PageRankResult {
        let n = graph.num_nodes();
        if n == 0 || sources.is_empty() {
            return PageRankResult {
                scores: vec![0.0; n],
                iterations: 0,
                convergence: 0.0,
                converged: true,
            };
        }

        // Create personalization vector
        let mut personalization = vec![0.0; n];
        let source_weight = 1.0 / sources.len() as f64;
        for &src in sources {
            if src < n {
                personalization[src] = source_weight;
            }
        }

        self.compute_with_personalization(graph, &personalization)
    }

    /// Compute PPR with custom personalization vector
    pub fn compute_with_personalization(
        &self,
        graph: &Graph,
        personalization: &[f64],
    ) -> PageRankResult {
        let n = graph.num_nodes();
        if n == 0 {
            return PageRankResult {
                scores: Vec::new(),
                iterations: 0,
                convergence: 0.0,
                converged: true,
            };
        }

        // Normalize personalization vector
        let pers_sum: f64 = personalization.iter().sum();
        let pers: Vec<f64> = if pers_sum > 0.0 {
            personalization.iter().map(|&p| p / pers_sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        // Initialize scores to personalization
        let mut scores = pers.clone();
        let mut new_scores = vec![0.0; n];

        // Precompute out-degrees
        let out_degrees: Vec<usize> = (0..n).map(|i| graph.degree(i)).collect();

        // Find dangling nodes
        let dangling: Vec<NodeId> = out_degrees
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut convergence = f64::MAX;
        let mut iterations = 0;

        while iterations < self.config.max_iterations && convergence > self.config.tolerance {
            // Handle dangling nodes
            let dangling_sum: f64 = if self.config.handle_dangling {
                dangling.iter().map(|&d| scores[d]).sum()
            } else {
                0.0
            };

            // Compute new scores
            for i in 0..n {
                let mut rank_sum = 0.0;

                // Sum contributions from neighbors
                for &(neighbor, _) in graph.neighbors(i) {
                    if out_degrees[neighbor] > 0 {
                        rank_sum += scores[neighbor] / out_degrees[neighbor] as f64;
                    }
                }

                // PPR formula: (1-d) * p_i + d * (sum of incoming + dangling)
                let dangling_contrib = dangling_sum * pers[i];
                new_scores[i] = (1.0 - self.config.damping) * pers[i]
                    + self.config.damping * (rank_sum + dangling_contrib);
            }

            // Compute convergence
            convergence = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(&old, &new)| (old - new).abs())
                .sum();

            std::mem::swap(&mut scores, &mut new_scores);
            iterations += 1;
        }

        PageRankResult {
            scores,
            iterations,
            convergence,
            converged: convergence <= self.config.tolerance,
        }
    }
}

/// Approximate Personalized PageRank using forward push
///
/// More efficient for local queries, computes PPR only for
/// nodes that have significant scores.
pub struct ApproximatePPR {
    config: PageRankConfig,
    /// Approximation parameter (higher = more accurate but slower)
    epsilon: f64,
}

impl ApproximatePPR {
    /// Create a new approximate PPR instance
    pub fn new(config: PageRankConfig, epsilon: f64) -> Self {
        Self { config, epsilon }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PageRankConfig::default(), 1e-4)
    }

    /// Set epsilon parameter
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Compute approximate PPR from a source node using forward push
    ///
    /// This algorithm pushes probability mass forward through the graph,
    /// only processing nodes with significant residual.
    pub fn compute_from_node(&self, graph: &Graph, source: NodeId) -> PageRankResult {
        let n = graph.num_nodes();
        if n == 0 || source >= n {
            return PageRankResult {
                scores: vec![0.0; n],
                iterations: 0,
                convergence: 0.0,
                converged: true,
            };
        }

        // PPR estimates and residuals
        let mut estimates: HashMap<NodeId, f64> = HashMap::new();
        let mut residuals: HashMap<NodeId, f64> = HashMap::new();

        // Initialize: all mass starts at source
        residuals.insert(source, 1.0);

        // Queue for nodes to process
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(source);
        let mut in_queue: HashSet<NodeId> = HashSet::new();
        in_queue.insert(source);

        let alpha = 1.0 - self.config.damping;
        let mut iterations = 0;

        while let Some(node) = queue.pop_front() {
            in_queue.remove(&node);

            let residual = *residuals.get(&node).unwrap_or(&0.0);
            if residual < self.epsilon {
                continue;
            }

            // Push alpha fraction to estimate
            *estimates.entry(node).or_insert(0.0) += alpha * residual;

            // Push (1-alpha) fraction to neighbors
            let neighbors = graph.neighbors(node);
            let out_degree = neighbors.len();

            if out_degree > 0 {
                let push_amount = (1.0 - alpha) * residual / out_degree as f64;

                for &(neighbor, _) in neighbors {
                    let new_residual = residuals.entry(neighbor).or_insert(0.0);
                    *new_residual += push_amount;

                    // Add to queue if residual is significant
                    if *new_residual >= self.epsilon && !in_queue.contains(&neighbor) {
                        queue.push_back(neighbor);
                        in_queue.insert(neighbor);
                    }
                }
            } else {
                // Dangling node: teleport back to source
                let new_residual = residuals.entry(source).or_insert(0.0);
                *new_residual += (1.0 - alpha) * residual;
                if *new_residual >= self.epsilon && !in_queue.contains(&source) {
                    queue.push_back(source);
                    in_queue.insert(source);
                }
            }

            // Clear processed residual
            residuals.insert(node, 0.0);
            iterations += 1;

            // Safety limit
            if iterations > n * 100 {
                break;
            }
        }

        // Convert to dense vector
        let mut scores = vec![0.0; n];
        for (&node, &score) in &estimates {
            if node < n {
                scores[node] = score;
            }
        }

        // Compute remaining residual as convergence measure
        let residual_sum: f64 = residuals.values().sum();

        PageRankResult {
            scores,
            iterations,
            convergence: residual_sum,
            converged: residual_sum < self.epsilon * n as f64,
        }
    }

    /// Compute approximate PPR for multiple source nodes
    pub fn compute_from_nodes(&self, graph: &Graph, sources: &[NodeId]) -> PageRankResult {
        let n = graph.num_nodes();
        if n == 0 || sources.is_empty() {
            return PageRankResult {
                scores: vec![0.0; n],
                iterations: 0,
                convergence: 0.0,
                converged: true,
            };
        }

        // Compute PPR from each source and average
        let mut combined_scores = vec![0.0; n];
        let mut total_iterations = 0;
        let mut max_convergence: f64 = 0.0;

        for &source in sources {
            let result = self.compute_from_node(graph, source);
            for (i, &score) in result.scores.iter().enumerate() {
                combined_scores[i] += score;
            }
            total_iterations += result.iterations;
            max_convergence = max_convergence.max(result.convergence);
        }

        // Normalize
        let weight = 1.0 / sources.len() as f64;
        for score in &mut combined_scores {
            *score *= weight;
        }

        PageRankResult {
            scores: combined_scores,
            iterations: total_iterations,
            convergence: max_convergence,
            converged: max_convergence < self.epsilon * n as f64,
        }
    }
}

/// Topic-Sensitive PageRank
///
/// Precomputes multiple PageRank vectors for different "topics" (seed sets).
/// Useful for query-dependent ranking.
pub struct TopicSensitivePageRank {
    config: PageRankConfig,
    /// Precomputed topic vectors
    topic_vectors: Vec<PageRankResult>,
    /// Topic names/identifiers
    topic_names: Vec<String>,
}

impl TopicSensitivePageRank {
    /// Create a new topic-sensitive PageRank instance
    pub fn new(config: PageRankConfig) -> Self {
        Self {
            config,
            topic_vectors: Vec::new(),
            topic_names: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PageRankConfig::default())
    }

    /// Add a topic with its seed nodes
    pub fn add_topic(&mut self, graph: &Graph, name: &str, seed_nodes: &[NodeId]) {
        let ppr = PersonalizedPageRank::new(self.config.clone());
        let result = ppr.compute_from_nodes(graph, seed_nodes);

        self.topic_vectors.push(result);
        self.topic_names.push(name.to_string());
    }

    /// Get number of topics
    pub fn num_topics(&self) -> usize {
        self.topic_vectors.len()
    }

    /// Get topic names
    pub fn topic_names(&self) -> &[String] {
        &self.topic_names
    }

    /// Get PageRank vector for a specific topic
    pub fn topic_vector(&self, topic_idx: usize) -> Option<&PageRankResult> {
        self.topic_vectors.get(topic_idx)
    }

    /// Compute combined ranking given topic weights
    pub fn rank_with_weights(&self, topic_weights: &[f64]) -> Vec<f64> {
        if self.topic_vectors.is_empty() {
            return Vec::new();
        }

        let n = self.topic_vectors[0].scores.len();
        let mut combined = vec![0.0; n];

        let weight_sum: f64 = topic_weights.iter().sum();
        let normalized_weights: Vec<f64> = if weight_sum > 0.0 {
            topic_weights.iter().map(|&w| w / weight_sum).collect()
        } else {
            vec![1.0 / self.topic_vectors.len() as f64; self.topic_vectors.len()]
        };

        for (topic_idx, &weight) in normalized_weights.iter().enumerate() {
            if let Some(topic_result) = self.topic_vectors.get(topic_idx) {
                for (i, &score) in topic_result.scores.iter().enumerate() {
                    combined[i] += weight * score;
                }
            }
        }

        combined
    }

    /// Get top-k nodes for a topic
    pub fn top_k_for_topic(&self, topic_idx: usize, k: usize) -> Vec<(NodeId, f64)> {
        self.topic_vectors
            .get(topic_idx)
            .map(|r| r.top_k(k))
            .unwrap_or_default()
    }
}

/// Statistics about PageRank computation
#[derive(Debug, Clone, Default)]
pub struct PageRankStats {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Number of dangling nodes
    pub num_dangling: usize,
    /// Average PageRank score
    pub avg_score: f64,
    /// Maximum PageRank score
    pub max_score: f64,
    /// Minimum PageRank score
    pub min_score: f64,
    /// Standard deviation of scores
    pub std_dev: f64,
}

/// Analyze PageRank results
pub fn analyze_pagerank(graph: &Graph, result: &PageRankResult) -> PageRankStats {
    let n = graph.num_nodes();
    if n == 0 {
        return PageRankStats::default();
    }

    let num_dangling = (0..n).filter(|&i| graph.degree(i) == 0).count();

    let sum: f64 = result.scores.iter().sum();
    let avg_score = sum / n as f64;

    let max_score = result
        .scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let min_score = result.scores.iter().cloned().fold(f64::INFINITY, f64::min);

    let variance: f64 = result
        .scores
        .iter()
        .map(|&s| (s - avg_score).powi(2))
        .sum::<f64>()
        / n as f64;

    PageRankStats {
        num_nodes: n,
        num_edges: graph.num_edges(),
        num_dangling,
        avg_score,
        max_score,
        min_score,
        std_dev: variance.sqrt(),
    }
}

/// Compute PageRank-based node similarity
///
/// Two nodes are similar if they have similar PPR vectors.
pub fn pagerank_similarity(graph: &Graph, node_a: NodeId, node_b: NodeId, damping: f64) -> f64 {
    let config = PageRankConfig::new(damping);
    let ppr = PersonalizedPageRank::new(config);

    let ppr_a = ppr.compute_from_node(graph, node_a);
    let ppr_b = ppr.compute_from_node(graph, node_b);

    // Cosine similarity between PPR vectors
    let dot: f64 = ppr_a
        .scores
        .iter()
        .zip(ppr_b.scores.iter())
        .map(|(&a, &b)| a * b)
        .sum();

    let norm_a: f64 = ppr_a.scores.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = ppr_b.scores.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// SimRank-style similarity using PPR
///
/// Computes how similar two nodes are based on their structural roles.
pub fn structural_similarity(graph: &Graph, node_a: NodeId, node_b: NodeId, damping: f64) -> f64 {
    // Symmetrized PPR similarity
    let sim_ab = pagerank_similarity(graph, node_a, node_b, damping);
    let sim_ba = pagerank_similarity(graph, node_b, node_a, damping);
    (sim_ab + sim_ba) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn create_directed_graph() -> Graph {
        // 0 -> 1 -> 2 -> 3 -> 0 (cycle)
        let mut graph = Graph::directed(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 0);
        graph
    }

    #[test]
    fn test_pagerank_config() {
        let config = PageRankConfig::new(0.9)
            .with_max_iterations(50)
            .with_tolerance(1e-8);

        assert!((config.damping - 0.9).abs() < 1e-6);
        assert_eq!(config.max_iterations, 50);
        assert!((config.tolerance - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = Graph::undirected(0);
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert!(result.scores.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_pagerank_single_node() {
        let graph = Graph::undirected(1);
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert_eq!(result.scores.len(), 1);
        assert!((result.scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_ring() {
        let graph = create_ring_graph();
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert!(result.converged);

        // All nodes should have equal PageRank in a ring
        let avg = result.scores.iter().sum::<f64>() / result.scores.len() as f64;
        for &score in &result.scores {
            assert!((score - avg).abs() < 1e-4);
        }
    }

    #[test]
    fn test_pagerank_star() {
        let graph = create_star_graph();
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert!(result.converged);

        // Center node should have highest PageRank
        let (max_node, _) = result.top_k(1)[0];
        assert_eq!(max_node, 0);
    }

    #[test]
    fn test_pagerank_line() {
        let graph = create_line_graph();
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert!(result.converged);

        // Middle nodes should have higher PageRank than endpoints
        assert!(result.scores[2] > result.scores[0]);
        assert!(result.scores[2] > result.scores[4]);
    }

    #[test]
    fn test_pagerank_directed() {
        let graph = create_directed_graph();
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        assert!(result.converged);

        // Scores should sum to approximately 1
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ppr_from_node() {
        let graph = create_ring_graph();
        let ppr = PersonalizedPageRank::default_config();
        let result = ppr.compute_from_node(&graph, 0);

        assert!(result.converged);

        // Source node should have highest PPR score
        let (max_node, _) = result.top_k(1)[0];
        assert_eq!(max_node, 0);
    }

    #[test]
    fn test_ppr_from_multiple_nodes() {
        let graph = create_ring_graph();
        let ppr = PersonalizedPageRank::default_config();
        let result = ppr.compute_from_nodes(&graph, &[0, 2]);

        assert!(result.converged);

        // Both source nodes should have high scores
        assert!(result.scores[0] > result.scores[1]);
        assert!(result.scores[2] > result.scores[1]);
    }

    #[test]
    fn test_ppr_star_from_leaf() {
        let graph = create_star_graph();
        let ppr = PersonalizedPageRank::default_config();
        let result = ppr.compute_from_node(&graph, 1);

        // Source and center should have high scores (center is hub)
        // Source leaf should have higher score than other leaves
        assert!(result.scores[1] > result.scores[2]);
        assert!(result.scores[1] > result.scores[3]);
        assert!(result.scores[1] > result.scores[4]);
    }

    #[test]
    fn test_approximate_ppr() {
        let graph = create_ring_graph();
        let appr = ApproximatePPR::default_config();
        let result = appr.compute_from_node(&graph, 0);

        // Source should have high score
        assert!(result.scores[0] > 0.0);

        // Scores should be non-negative
        for &score in &result.scores {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_approximate_ppr_accuracy() {
        let graph = create_star_graph();

        // Compare exact and approximate PPR
        let ppr = PersonalizedPageRank::new(PageRankConfig::new(0.85));
        let appr = ApproximatePPR::new(PageRankConfig::new(0.85), 1e-6);

        let exact = ppr.compute_from_node(&graph, 0);
        let approx = appr.compute_from_node(&graph, 0);

        // Top nodes should match
        let exact_top = exact.top_k(3);
        let approx_top = approx.top_k(3);

        // At least the top node should match
        assert_eq!(exact_top[0].0, approx_top[0].0);
    }

    #[test]
    fn test_topic_sensitive_pagerank() {
        let graph = create_star_graph();
        let mut tspr = TopicSensitivePageRank::default_config();

        tspr.add_topic(&graph, "center", &[0]);
        tspr.add_topic(&graph, "leaves", &[1, 2, 3, 4]);

        assert_eq!(tspr.num_topics(), 2);

        // Center topic should rank center highest
        let center_top = tspr.top_k_for_topic(0, 1);
        assert_eq!(center_top[0].0, 0);

        // Weighted ranking
        let combined = tspr.rank_with_weights(&[0.7, 0.3]);
        assert_eq!(combined.len(), 5);
    }

    #[test]
    fn test_pagerank_result_methods() {
        let result = PageRankResult {
            scores: vec![0.1, 0.4, 0.2, 0.15, 0.15],
            iterations: 10,
            convergence: 1e-7,
            converged: true,
        };

        // Test score lookup
        assert!((result.score(1) - 0.4).abs() < 1e-6);
        assert_eq!(result.score(100), 0.0);

        // Test top_k
        let top2 = result.top_k(2);
        assert_eq!(top2[0].0, 1);
        assert_eq!(top2[1].0, 2);

        // Test above_threshold
        let above = result.above_threshold(0.15);
        assert_eq!(above.len(), 4);

        // Test normalized
        let norm = result.normalized();
        let sum: f64 = norm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_pagerank() {
        let graph = create_star_graph();
        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        let stats = analyze_pagerank(&graph, &result);

        assert_eq!(stats.num_nodes, 5);
        assert!(stats.max_score >= stats.min_score);
        assert!(stats.avg_score > 0.0);
    }

    #[test]
    fn test_pagerank_similarity() {
        let graph = create_ring_graph();

        // Adjacent nodes should be more similar
        let sim_01 = pagerank_similarity(&graph, 0, 1, 0.85);
        let sim_02 = pagerank_similarity(&graph, 0, 2, 0.85);

        // In a ring, adjacent nodes should have higher similarity
        assert!(sim_01 > 0.0);
        assert!(sim_02 > 0.0);
    }

    #[test]
    fn test_structural_similarity() {
        let graph = create_ring_graph();

        // Adjacent nodes should have high similarity
        let sim_adjacent = structural_similarity(&graph, 0, 1, 0.85);
        let sim_opposite = structural_similarity(&graph, 0, 2, 0.85);

        // Both should be positive
        assert!(sim_adjacent > 0.0);
        assert!(sim_opposite > 0.0);

        // Adjacent should be more similar (in a ring, distance matters)
        assert!(sim_adjacent >= sim_opposite);
    }

    #[test]
    fn test_ppr_empty_sources() {
        let graph = create_ring_graph();
        let ppr = PersonalizedPageRank::default_config();
        let result = ppr.compute_from_nodes(&graph, &[]);

        // Should return zeros
        assert!(result.scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_pagerank_convergence() {
        let graph = create_ring_graph();
        let config = PageRankConfig::new(0.85)
            .with_max_iterations(1000)
            .with_tolerance(1e-10);
        let pr = PageRank::new(config);
        let result = pr.compute(&graph);

        assert!(result.converged);
        assert!(result.convergence < 1e-10);
    }

    #[test]
    fn test_dangling_nodes() {
        // Create graph with dangling node (no outgoing edges)
        let mut graph = Graph::directed(3);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        // Node 1 and 2 are dangling (no outgoing edges)

        let pr = PageRank::default_config();
        let result = pr.compute(&graph);

        // All scores should be valid
        for &score in &result.scores {
            assert!(score >= 0.0);
            assert!(!score.is_nan());
        }
    }

    #[test]
    fn test_approximate_ppr_from_multiple() {
        let graph = create_ring_graph();
        let appr = ApproximatePPR::default_config();
        let result = appr.compute_from_nodes(&graph, &[0, 2]);

        // Both sources should have high scores
        assert!(result.scores[0] > 0.0);
        assert!(result.scores[2] > 0.0);
    }

    #[test]
    fn test_topic_pagerank_weights() {
        let graph = create_star_graph();
        let mut tspr = TopicSensitivePageRank::default_config();

        tspr.add_topic(&graph, "a", &[0]);
        tspr.add_topic(&graph, "b", &[1]);

        // Full weight on topic 0
        let w1 = tspr.rank_with_weights(&[1.0, 0.0]);
        // Full weight on topic 1
        let w2 = tspr.rank_with_weights(&[0.0, 1.0]);

        // Results should differ
        assert!((w1[0] - w2[0]).abs() > 0.01);
    }

    #[test]
    fn test_ppr_with_personalization() {
        let graph = create_ring_graph();
        let ppr = PersonalizedPageRank::default_config();

        // Custom personalization: higher weight on node 0
        let pers = vec![0.5, 0.2, 0.1, 0.1, 0.1];
        let result = ppr.compute_with_personalization(&graph, &pers);

        assert!(result.converged);
        // Node 0 should have highest score
        assert!(result.scores[0] > result.scores[2]);
    }
}
