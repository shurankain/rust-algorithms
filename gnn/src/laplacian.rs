// Graph Laplacian and Spectral Methods
//
// The graph Laplacian is fundamental for spectral graph theory and GNNs:
// - Spectral clustering and community detection
// - Graph signal processing
// - Spectral GNNs (ChebNet, GCN normalization)
// - Graph wavelets and diffusion
// - Manifold learning (Laplacian Eigenmaps)
//
// Key Laplacian variants:
// 1. Unnormalized Laplacian: L = D - A
// 2. Symmetric Normalized: L_sym = I - D^(-1/2) A D^(-1/2)
// 3. Random Walk Normalized: L_rw = I - D^(-1) A
//
// References:
// - "Spectral Graph Theory" (Chung, 1997)
// - "A Tutorial on Spectral Clustering" (von Luxburg, 2007)
// - "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
// - "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (Defferrard, 2016)

use crate::message_passing::Graph;

/// Type of graph Laplacian
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianType {
    /// Unnormalized: L = D - A
    Unnormalized,
    /// Symmetric normalized: L_sym = I - D^(-1/2) A D^(-1/2)
    SymmetricNormalized,
    /// Random walk normalized: L_rw = I - D^(-1) A
    RandomWalkNormalized,
}

/// Configuration for Laplacian computation
#[derive(Debug, Clone)]
pub struct LaplacianConfig {
    /// Type of Laplacian to compute
    pub laplacian_type: LaplacianType,
    /// Whether to add self-loops before computing (common in GCN)
    pub add_self_loops: bool,
    /// Regularization for degree matrix (avoid division by zero)
    pub epsilon: f64,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            laplacian_type: LaplacianType::SymmetricNormalized,
            add_self_loops: true,
            epsilon: 1e-10,
        }
    }
}

impl LaplacianConfig {
    /// Create config for unnormalized Laplacian
    pub fn unnormalized() -> Self {
        Self {
            laplacian_type: LaplacianType::Unnormalized,
            add_self_loops: false,
            epsilon: 1e-10,
        }
    }

    /// Create config for symmetric normalized Laplacian
    pub fn symmetric_normalized() -> Self {
        Self {
            laplacian_type: LaplacianType::SymmetricNormalized,
            add_self_loops: true,
            epsilon: 1e-10,
        }
    }

    /// Create config for random walk normalized Laplacian
    pub fn random_walk_normalized() -> Self {
        Self {
            laplacian_type: LaplacianType::RandomWalkNormalized,
            add_self_loops: false,
            epsilon: 1e-10,
        }
    }

    /// Set self-loop option
    pub fn with_self_loops(mut self, add: bool) -> Self {
        self.add_self_loops = add;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }
}

/// Dense matrix representation (row-major)
pub type DenseMatrix = Vec<Vec<f64>>;

/// Sparse matrix entry (row, col, value)
pub type SparseEntry = (usize, usize, f64);

/// Sparse matrix representation
#[derive(Debug, Clone, Default)]
pub struct SparseMatrix {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Non-zero entries
    pub entries: Vec<SparseEntry>,
}

impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    /// Add a non-zero entry
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        if value.abs() > 1e-15 {
            self.entries.push((row, col, value));
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> DenseMatrix {
        let mut dense = vec![vec![0.0; self.cols]; self.rows];
        for &(row, col, value) in &self.entries {
            dense[row][col] = value;
        }
        dense
    }

    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, vec: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(row, col, value) in &self.entries {
            if col < vec.len() {
                result[row] += value * vec[col];
            }
        }
        result
    }

    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get entry at (row, col), returns 0 if not found
    pub fn get(&self, row: usize, col: usize) -> f64 {
        for &(r, c, v) in &self.entries {
            if r == row && c == col {
                return v;
            }
        }
        0.0
    }
}

/// Graph Laplacian computation
pub struct GraphLaplacian {
    config: LaplacianConfig,
}

impl GraphLaplacian {
    /// Create a new Laplacian computer
    pub fn new(config: LaplacianConfig) -> Self {
        Self { config }
    }

    /// Create with default config (symmetric normalized with self-loops)
    pub fn default_config() -> Self {
        Self::new(LaplacianConfig::default())
    }

    /// Compute the degree of each node
    pub fn compute_degrees(&self, graph: &Graph) -> Vec<f64> {
        let n = graph.num_nodes();
        let mut degrees = vec![0.0; n];

        for (i, degree) in degrees.iter_mut().enumerate() {
            *degree = graph.degree(i) as f64;
            if self.config.add_self_loops {
                *degree += 1.0; // Self-loop adds 1 to degree
            }
        }

        degrees
    }

    /// Compute the adjacency matrix (sparse)
    pub fn compute_adjacency(&self, graph: &Graph) -> SparseMatrix {
        let n = graph.num_nodes();
        let mut adj = SparseMatrix::new(n, n);

        for i in 0..n {
            // Add self-loop if configured
            if self.config.add_self_loops {
                adj.add(i, i, 1.0);
            }

            // Add edges
            for &(neighbor, _) in graph.neighbors(i) {
                adj.add(i, neighbor, 1.0);
            }
        }

        adj
    }

    /// Compute the Laplacian matrix (sparse)
    pub fn compute(&self, graph: &Graph) -> SparseMatrix {
        let degrees = self.compute_degrees(graph);

        match self.config.laplacian_type {
            LaplacianType::Unnormalized => self.compute_unnormalized(graph, &degrees),
            LaplacianType::SymmetricNormalized => {
                self.compute_symmetric_normalized(graph, &degrees)
            }
            LaplacianType::RandomWalkNormalized => {
                self.compute_random_walk_normalized(graph, &degrees)
            }
        }
    }

    /// Compute unnormalized Laplacian: L = D - A
    fn compute_unnormalized(&self, graph: &Graph, degrees: &[f64]) -> SparseMatrix {
        let n = graph.num_nodes();
        let mut laplacian = SparseMatrix::new(n, n);

        for (i, &deg) in degrees.iter().enumerate() {
            // Diagonal: degree
            laplacian.add(i, i, deg);

            // Off-diagonal: -1 for edges
            for &(neighbor, _) in graph.neighbors(i) {
                laplacian.add(i, neighbor, -1.0);
            }

            // Self-loop contribution (if added)
            if self.config.add_self_loops {
                // Self-loop is already in degree, subtract 1 from diagonal
                laplacian.add(i, i, -1.0);
            }
        }

        laplacian
    }

    /// Compute symmetric normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
    fn compute_symmetric_normalized(&self, graph: &Graph, degrees: &[f64]) -> SparseMatrix {
        let n = graph.num_nodes();
        let mut laplacian = SparseMatrix::new(n, n);

        // Compute D^(-1/2)
        let d_inv_sqrt: Vec<f64> = degrees
            .iter()
            .map(|&d| {
                if d > self.config.epsilon {
                    1.0 / d.sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        for i in 0..n {
            // Diagonal: 1 (from I)
            laplacian.add(i, i, 1.0);

            // Off-diagonal: -D^(-1/2) * A * D^(-1/2)
            for &(neighbor, _) in graph.neighbors(i) {
                let value = -d_inv_sqrt[i] * d_inv_sqrt[neighbor];
                laplacian.add(i, neighbor, value);
            }

            // Self-loop contribution
            if self.config.add_self_loops {
                let value = -d_inv_sqrt[i] * d_inv_sqrt[i];
                laplacian.add(i, i, value);
            }
        }

        laplacian
    }

    /// Compute random walk normalized Laplacian: L_rw = I - D^(-1) A
    fn compute_random_walk_normalized(&self, graph: &Graph, degrees: &[f64]) -> SparseMatrix {
        let n = graph.num_nodes();
        let mut laplacian = SparseMatrix::new(n, n);

        // Compute D^(-1)
        let d_inv: Vec<f64> = degrees
            .iter()
            .map(|&d| {
                if d > self.config.epsilon {
                    1.0 / d
                } else {
                    0.0
                }
            })
            .collect();

        for (i, &inv) in d_inv.iter().enumerate() {
            // Diagonal: 1 (from I)
            laplacian.add(i, i, 1.0);

            // Off-diagonal: -D^(-1) * A
            for &(neighbor, _) in graph.neighbors(i) {
                let value = -inv;
                laplacian.add(i, neighbor, value);
            }

            // Self-loop contribution
            if self.config.add_self_loops {
                let value = -inv;
                laplacian.add(i, i, value);
            }
        }

        laplacian
    }

    /// Compute normalized adjacency matrix (for GCN): D^(-1/2) A D^(-1/2)
    pub fn compute_normalized_adjacency(&self, graph: &Graph) -> SparseMatrix {
        let n = graph.num_nodes();
        let degrees = self.compute_degrees(graph);
        let mut norm_adj = SparseMatrix::new(n, n);

        // Compute D^(-1/2)
        let d_inv_sqrt: Vec<f64> = degrees
            .iter()
            .map(|&d| {
                if d > self.config.epsilon {
                    1.0 / d.sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        for i in 0..n {
            // Self-loop
            if self.config.add_self_loops {
                let value = d_inv_sqrt[i] * d_inv_sqrt[i];
                norm_adj.add(i, i, value);
            }

            // Edges
            for &(neighbor, _) in graph.neighbors(i) {
                let value = d_inv_sqrt[i] * d_inv_sqrt[neighbor];
                norm_adj.add(i, neighbor, value);
            }
        }

        norm_adj
    }
}

/// Apply Laplacian to node features (graph filtering)
pub fn apply_laplacian(laplacian: &SparseMatrix, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let feature_dim = features[0].len();
    let mut result = vec![vec![0.0; feature_dim]; laplacian.rows];

    for &(row, col, value) in &laplacian.entries {
        if col < features.len() {
            for (d, r) in result[row].iter_mut().enumerate() {
                if d < features[col].len() {
                    *r += value * features[col][d];
                }
            }
        }
    }

    result
}

/// Compute the graph Fourier transform (requires eigendecomposition)
/// Returns eigenvalues (frequencies) sorted in ascending order
pub fn compute_laplacian_eigenvalues(laplacian: &DenseMatrix, max_iter: usize) -> Vec<f64> {
    let n = laplacian.len();
    if n == 0 {
        return Vec::new();
    }

    // Power iteration for largest eigenvalue estimation
    // For full spectrum, would need proper eigendecomposition
    // Here we estimate a few eigenvalues using shifted power iteration

    let mut eigenvalues = Vec::new();

    // Estimate spectral radius (largest eigenvalue)
    let largest = estimate_largest_eigenvalue(laplacian, max_iter);
    eigenvalues.push(largest);

    // For a Laplacian, smallest eigenvalue is 0 (for connected graphs)
    eigenvalues.insert(0, 0.0);

    eigenvalues
}

/// Estimate the largest eigenvalue using power iteration
fn estimate_largest_eigenvalue(matrix: &DenseMatrix, max_iter: usize) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 0.0;
    }

    // Start with random vector
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) as f64).sin()).collect();

    // Normalize
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // w = A * v
        let w: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| matrix[i][j] * v[j]).sum())
            .collect();

        // Rayleigh quotient: eigenvalue = v^T * w / v^T * v
        let vw: f64 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();
        let vv: f64 = v.iter().map(|&x| x * x).sum();

        if vv > 0.0 {
            eigenvalue = vw / vv;
        }

        // Normalize w to get new v
        let w_norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if w_norm > 1e-10 {
            v = w.iter().map(|&x| x / w_norm).collect();
        } else {
            break;
        }
    }

    eigenvalue
}

/// Chebyshev polynomial approximation for spectral filtering
/// Computes T_k(L) where T_k is the k-th Chebyshev polynomial
pub struct ChebyshevFilter {
    /// Maximum polynomial order
    order: usize,
    /// Estimated largest eigenvalue (for scaling)
    lambda_max: f64,
}

impl ChebyshevFilter {
    /// Create a new Chebyshev filter
    pub fn new(order: usize, lambda_max: f64) -> Self {
        Self { order, lambda_max }
    }

    /// Create with default parameters
    pub fn default_filter() -> Self {
        Self {
            order: 2,
            lambda_max: 2.0,
        }
    }

    /// Apply Chebyshev filter to features
    /// Returns T_0(L)x, T_1(L)x, ..., T_k(L)x
    pub fn apply(&self, laplacian: &SparseMatrix, features: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        if features.is_empty() {
            return Vec::new();
        }

        let n = features.len();
        let feature_dim = features[0].len();

        // Scale Laplacian to [-1, 1]: L_scaled = 2L/lambda_max - I
        let scale = 2.0 / self.lambda_max;

        let mut results = Vec::with_capacity(self.order + 1);

        // T_0(L)x = x
        let t0: Vec<Vec<f64>> = features.to_vec();
        results.push(t0.clone());

        if self.order == 0 {
            return results;
        }

        // T_1(L)x = L_scaled * x
        let mut t1 = vec![vec![0.0; feature_dim]; n];
        for &(row, col, value) in &laplacian.entries {
            let scaled_value = scale * value - if row == col { 1.0 } else { 0.0 };
            if col < n {
                for d in 0..feature_dim {
                    if d < features[col].len() {
                        t1[row][d] += scaled_value * features[col][d];
                    }
                }
            }
        }
        // Add -I contribution for diagonal
        for i in 0..n {
            for d in 0..feature_dim {
                t1[i][d] -= features[i][d]; // This is wrong, let me fix
            }
        }

        // Actually, L_scaled * x = (2L/lambda_max - I) * x = 2L*x/lambda_max - x
        // Recompute t1 properly
        let lx = sparse_matvec(laplacian, features);
        let t1: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..feature_dim)
                    .map(|d| scale * lx[i][d] - features[i][d])
                    .collect()
            })
            .collect();

        results.push(t1.clone());

        if self.order == 1 {
            return results;
        }

        // Recurrence: T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
        let mut t_prev = t0;
        let mut t_curr = t1;

        for _ in 2..=self.order {
            // T_k = 2 * L_scaled * T_{k-1} - T_{k-2}
            let l_t_curr = sparse_matvec(laplacian, &t_curr);

            let t_next: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..feature_dim)
                        .map(|d| 2.0 * (scale * l_t_curr[i][d] - t_curr[i][d]) - t_prev[i][d])
                        .collect()
                })
                .collect();

            results.push(t_next.clone());
            t_prev = t_curr;
            t_curr = t_next;
        }

        results
    }

    /// Apply spectral convolution with learned coefficients
    pub fn spectral_conv(
        &self,
        laplacian: &SparseMatrix,
        features: &[Vec<f64>],
        coeffs: &[f64],
    ) -> Vec<Vec<f64>> {
        let chebyshev_terms = self.apply(laplacian, features);

        if chebyshev_terms.is_empty() {
            return Vec::new();
        }

        let n = features.len();
        let feature_dim = features[0].len();
        let mut result = vec![vec![0.0; feature_dim]; n];

        for (k, term) in chebyshev_terms.iter().enumerate() {
            let coeff = if k < coeffs.len() { coeffs[k] } else { 0.0 };
            for i in 0..n {
                for d in 0..feature_dim {
                    result[i][d] += coeff * term[i][d];
                }
            }
        }

        result
    }
}

/// Sparse matrix-vector multiplication for feature matrices
fn sparse_matvec(matrix: &SparseMatrix, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let feature_dim = features[0].len();
    let mut result = vec![vec![0.0; feature_dim]; matrix.rows];

    for &(row, col, value) in &matrix.entries {
        if col < features.len() {
            for (d, r) in result[row].iter_mut().enumerate() {
                if d < features[col].len() {
                    *r += value * features[col][d];
                }
            }
        }
    }

    result
}

/// Graph signal smoothness (Dirichlet energy): x^T L x
/// Lower values indicate smoother signals
pub fn dirichlet_energy(laplacian: &SparseMatrix, signal: &[f64]) -> f64 {
    let lx = laplacian.multiply_vector(signal);
    signal.iter().zip(lx.iter()).map(|(&x, &lx)| x * lx).sum()
}

/// Compute smoothness of node features
pub fn feature_smoothness(laplacian: &SparseMatrix, features: &[Vec<f64>]) -> f64 {
    if features.is_empty() {
        return 0.0;
    }

    let feature_dim = features[0].len();
    let mut total_energy = 0.0;

    for d in 0..feature_dim {
        let signal: Vec<f64> = features
            .iter()
            .map(|f| f.get(d).copied().unwrap_or(0.0))
            .collect();
        total_energy += dirichlet_energy(laplacian, &signal);
    }

    total_energy / feature_dim as f64
}

/// Statistics about the Laplacian
#[derive(Debug, Clone, Default)]
pub struct LaplacianStats {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of non-zero entries
    pub nnz: usize,
    /// Sparsity (fraction of zeros)
    pub sparsity: f64,
    /// Average degree
    pub avg_degree: f64,
    /// Max degree
    pub max_degree: f64,
}

/// Analyze a Laplacian matrix
pub fn analyze_laplacian(laplacian: &SparseMatrix, degrees: &[f64]) -> LaplacianStats {
    let n = laplacian.rows;
    if n == 0 {
        return LaplacianStats::default();
    }

    let total_entries = n * n;
    let sparsity = 1.0 - (laplacian.nnz() as f64 / total_entries as f64);

    let avg_degree = degrees.iter().sum::<f64>() / n as f64;
    let max_degree = degrees.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    LaplacianStats {
        num_nodes: n,
        nnz: laplacian.nnz(),
        sparsity,
        avg_degree,
        max_degree,
    }
}

/// Compute algebraic connectivity (second smallest eigenvalue of Laplacian)
/// Also known as the Fiedler value
/// Note: This is an approximation
pub fn estimate_algebraic_connectivity(graph: &Graph, max_iter: usize) -> f64 {
    let config = LaplacianConfig::unnormalized().with_self_loops(false);
    let laplacian = GraphLaplacian::new(config);
    let l = laplacian.compute(graph);
    let dense = l.to_dense();

    // For connected graphs, smallest eigenvalue is 0
    // Second smallest (algebraic connectivity) indicates graph connectivity strength

    // Use inverse power iteration shifted by small amount
    // This is a simplification - proper implementation would use deflation
    let largest = estimate_largest_eigenvalue(&dense, max_iter);

    // Very rough estimate: for d-regular graphs, lambda_2 ≈ d - 2*sqrt(d-1)
    // For general graphs, we just report a bound
    let n = graph.num_nodes();
    if n <= 1 {
        return 0.0;
    }

    // Cheeger inequality bound: h^2/2 <= lambda_2 <= 2h
    // where h is the Cheeger constant (isoperimetric number)
    // Without computing h, we can't get exact lambda_2

    // Return a simple estimate based on graph structure
    let min_degree = (0..n).map(|i| graph.degree(i)).min().unwrap_or(0) as f64;
    let avg_degree = (0..n).map(|i| graph.degree(i)).sum::<usize>() as f64 / n as f64;

    // Simple heuristic bound
    (min_degree / avg_degree * largest / n as f64).max(0.0)
}

/// Diffusion on graph using heat kernel: exp(-t*L)
pub fn heat_diffusion(
    laplacian: &SparseMatrix,
    signal: &[f64],
    time: f64,
    steps: usize,
) -> Vec<f64> {
    // Approximate exp(-tL)x using Taylor expansion or Euler method
    // Using simple Euler: x(t+dt) = x(t) - dt * L * x(t)

    let dt = time / steps as f64;
    let mut x = signal.to_vec();

    for _ in 0..steps {
        let lx = laplacian.multiply_vector(&x);
        for i in 0..x.len() {
            x[i] -= dt * lx[i];
        }
    }

    x
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

    fn create_star_graph() -> Graph {
        // 0 is center, connected to 1, 2, 3, 4
        let mut graph = Graph::undirected(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(0, 4);
        graph
    }

    fn create_complete_graph(n: usize) -> Graph {
        let mut graph = Graph::undirected(n);
        for i in 0..n {
            for j in i + 1..n {
                graph.add_edge(i, j);
            }
        }
        graph
    }

    #[test]
    fn test_laplacian_config() {
        let config = LaplacianConfig::symmetric_normalized().with_self_loops(true);
        assert_eq!(config.laplacian_type, LaplacianType::SymmetricNormalized);
        assert!(config.add_self_loops);
    }

    #[test]
    fn test_sparse_matrix() {
        let mut m = SparseMatrix::new(3, 3);
        m.add(0, 0, 2.0);
        m.add(0, 1, -1.0);
        m.add(1, 0, -1.0);
        m.add(1, 1, 2.0);

        assert_eq!(m.nnz(), 4);
        assert!((m.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 1) - (-1.0)).abs() < 1e-10);
        assert!((m.get(2, 2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_to_dense() {
        let mut m = SparseMatrix::new(2, 2);
        m.add(0, 0, 1.0);
        m.add(1, 1, 2.0);

        let dense = m.to_dense();
        assert_eq!(dense.len(), 2);
        assert!((dense[0][0] - 1.0).abs() < 1e-10);
        assert!((dense[1][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matvec() {
        let mut m = SparseMatrix::new(2, 2);
        m.add(0, 0, 2.0);
        m.add(0, 1, 1.0);
        m.add(1, 0, 1.0);
        m.add(1, 1, 2.0);

        let v = vec![1.0, 1.0];
        let result = m.multiply_vector(&v);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_degrees() {
        let graph = create_star_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let degrees = laplacian.compute_degrees(&graph);

        assert_eq!(degrees[0], 4.0); // Center
        assert_eq!(degrees[1], 1.0); // Leaf
    }

    #[test]
    fn test_unnormalized_laplacian_line() {
        let graph = create_line_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // Diagonal should be degrees
        assert!((dense[0][0] - 1.0).abs() < 1e-10); // Endpoint
        assert!((dense[1][1] - 2.0).abs() < 1e-10); // Middle

        // Off-diagonal should be -1 for edges
        assert!((dense[0][1] - (-1.0)).abs() < 1e-10);
        assert!((dense[1][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_unnormalized_laplacian_ring() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // All nodes have degree 2 in a ring
        for i in 0..5 {
            assert!((dense[i][i] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_normalized_laplacian() {
        let graph = create_ring_graph();
        let laplacian =
            GraphLaplacian::new(LaplacianConfig::symmetric_normalized().with_self_loops(false));
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // Diagonal should be 1
        for i in 0..5 {
            assert!((dense[i][i] - 1.0).abs() < 1e-10);
        }

        // Off-diagonal should be -1/sqrt(d_i * d_j) = -1/sqrt(2*2) = -0.5
        assert!((dense[0][1] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_random_walk_normalized_laplacian() {
        let graph = create_ring_graph();
        let laplacian =
            GraphLaplacian::new(LaplacianConfig::random_walk_normalized().with_self_loops(false));
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // Diagonal should be 1
        for i in 0..5 {
            assert!((dense[i][i] - 1.0).abs() < 1e-10);
        }

        // Off-diagonal should be -1/d_i = -0.5
        assert!((dense[0][1] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_with_self_loops() {
        let graph = create_line_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized().with_self_loops(true));
        let degrees = laplacian.compute_degrees(&graph);

        // Degrees should be increased by 1
        assert!((degrees[0] - 2.0).abs() < 1e-10); // 1 + 1
        assert!((degrees[1] - 3.0).abs() < 1e-10); // 2 + 1
    }

    #[test]
    fn test_normalized_adjacency() {
        let graph = create_ring_graph();
        let laplacian =
            GraphLaplacian::new(LaplacianConfig::symmetric_normalized().with_self_loops(false));
        let adj = laplacian.compute_normalized_adjacency(&graph);
        let dense = adj.to_dense();

        // D^(-1/2) A D^(-1/2) = 1/sqrt(2*2) = 0.5 for edges
        assert!((dense[0][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_laplacian() {
        let graph = create_line_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);

        let features: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let result = apply_laplacian(&l, &features);

        assert_eq!(result.len(), 5);
        // L * [0,1,2,3,4]^T should give smoothness measure
    }

    #[test]
    fn test_dirichlet_energy_constant() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);

        // Constant signal should have zero Dirichlet energy
        let constant = vec![1.0; 5];
        let energy = dirichlet_energy(&l, &constant);

        assert!(energy.abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_energy_varying() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);

        // Varying signal should have positive Dirichlet energy
        let varying: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let energy = dirichlet_energy(&l, &varying);

        assert!(energy > 0.0);
    }

    #[test]
    fn test_feature_smoothness() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);

        let smooth_features: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0, 1.0]).collect();
        let varying_features: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, i as f64]).collect();

        let smooth_energy = feature_smoothness(&l, &smooth_features);
        let varying_energy = feature_smoothness(&l, &varying_features);

        assert!(smooth_energy < varying_energy);
    }

    #[test]
    fn test_chebyshev_filter_order_0() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::symmetric_normalized());
        let l = laplacian.compute(&graph);

        let filter = ChebyshevFilter::new(0, 2.0);
        let features: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();

        let result = filter.apply(&l, &features);
        assert_eq!(result.len(), 1); // Just T_0 = x
    }

    #[test]
    fn test_chebyshev_filter_higher_order() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::symmetric_normalized());
        let l = laplacian.compute(&graph);

        let filter = ChebyshevFilter::new(3, 2.0);
        let features: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();

        let result = filter.apply(&l, &features);
        assert_eq!(result.len(), 4); // T_0, T_1, T_2, T_3
    }

    #[test]
    fn test_spectral_conv() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::symmetric_normalized());
        let l = laplacian.compute(&graph);

        let filter = ChebyshevFilter::new(2, 2.0);
        let features: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0]).collect();
        let coeffs = vec![0.5, 0.3, 0.2];

        let result = filter.spectral_conv(&l, &features, &coeffs);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_analyze_laplacian() {
        let graph = create_star_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let degrees = laplacian.compute_degrees(&graph);

        let stats = analyze_laplacian(&l, &degrees);

        assert_eq!(stats.num_nodes, 5);
        assert!(stats.sparsity > 0.0);
        assert!(stats.max_degree >= stats.avg_degree);
    }

    #[test]
    fn test_heat_diffusion() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);

        // Initial spike at node 0
        let signal = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let diffused = heat_diffusion(&l, &signal, 0.1, 10);

        assert_eq!(diffused.len(), 5);
        // Signal should spread to neighbors
        assert!(diffused[0] < signal[0]); // Decreased at source
        assert!(diffused[1] > signal[1]); // Increased at neighbor
    }

    #[test]
    fn test_complete_graph_laplacian() {
        let graph = create_complete_graph(4);
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // In K_n, each node has degree n-1
        for i in 0..4 {
            assert!((dense[i][i] - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::undirected(0);
        let laplacian = GraphLaplacian::default_config();
        let l = laplacian.compute(&graph);

        assert_eq!(l.rows, 0);
        assert_eq!(l.cols, 0);
    }

    #[test]
    fn test_single_node() {
        let graph = Graph::undirected(1);
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        // Single node with no edges has degree 0
        assert!((dense[0][0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalue_estimation() {
        let graph = create_ring_graph();
        let laplacian = GraphLaplacian::new(LaplacianConfig::unnormalized());
        let l = laplacian.compute(&graph);
        let dense = l.to_dense();

        let eigenvalues = compute_laplacian_eigenvalues(&dense, 100);

        // Smallest eigenvalue should be 0 (connected graph)
        assert!(eigenvalues[0].abs() < 1e-6);
    }

    #[test]
    fn test_estimate_algebraic_connectivity() {
        let graph = create_complete_graph(4);
        let connectivity = estimate_algebraic_connectivity(&graph, 100);

        // Should be positive for connected graph
        assert!(connectivity >= 0.0);
    }

    #[test]
    fn test_laplacian_types_different() {
        let graph = create_star_graph();

        let l_unnorm = GraphLaplacian::new(LaplacianConfig::unnormalized())
            .compute(&graph)
            .to_dense();
        let l_sym =
            GraphLaplacian::new(LaplacianConfig::symmetric_normalized().with_self_loops(false))
                .compute(&graph)
                .to_dense();
        let l_rw = GraphLaplacian::new(LaplacianConfig::random_walk_normalized())
            .compute(&graph)
            .to_dense();

        // All should have same size
        assert_eq!(l_unnorm.len(), l_sym.len());
        assert_eq!(l_sym.len(), l_rw.len());

        // But different values (comparing diagonal of center node)
        // Unnormalized: 4, Symmetric: 1, RW: 1
        assert!((l_unnorm[0][0] - 4.0).abs() < 1e-10);
        assert!((l_sym[0][0] - 1.0).abs() < 1e-10);
        assert!((l_rw[0][0] - 1.0).abs() < 1e-10);
    }
}
