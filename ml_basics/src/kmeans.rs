// K-Means Clustering - Unsupervised learning algorithm for partitioning data into k clusters

use std::collections::HashMap;

/// K-Means clustering algorithm
/// Partitions n observations into k clusters where each observation belongs
/// to the cluster with the nearest centroid.
#[derive(Debug, Clone)]
pub struct KMeans {
    k: usize,
    max_iterations: usize,
    tolerance: f64,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
    n_iterations: usize,
}

impl KMeans {
    /// Create a new K-Means instance
    /// - k: number of clusters
    /// - max_iterations: maximum number of iterations
    /// - tolerance: convergence threshold for centroid movement
    pub fn new(k: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            k,
            max_iterations,
            tolerance,
            centroids: Vec::new(),
            labels: Vec::new(),
            inertia: 0.0,
            n_iterations: 0,
        }
    }

    /// Fit the model to the data using random initialization
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() || self.k == 0 {
            return;
        }

        let n_features = data[0].len();

        // Initialize centroids randomly by selecting k random points
        self.centroids = self.random_init(data);

        for iteration in 0..self.max_iterations {
            // Assign each point to nearest centroid
            self.labels = self.assign_clusters(data);

            // Calculate new centroids
            let new_centroids = self.calculate_centroids(data, n_features);

            // Check for convergence
            let max_shift = self.max_centroid_shift(&new_centroids);
            self.centroids = new_centroids;
            self.n_iterations = iteration + 1;

            if max_shift < self.tolerance {
                break;
            }
        }

        // Calculate final inertia (sum of squared distances to centroids)
        self.inertia = self.calculate_inertia(data);
    }

    /// Fit using k-means++ initialization for better convergence
    pub fn fit_kmeanspp(&mut self, data: &[Vec<f64>], seed: u64) {
        if data.is_empty() || self.k == 0 {
            return;
        }

        let n_features = data[0].len();

        // Initialize centroids using k-means++
        self.centroids = self.kmeanspp_init(data, seed);

        for iteration in 0..self.max_iterations {
            self.labels = self.assign_clusters(data);
            let new_centroids = self.calculate_centroids(data, n_features);
            let max_shift = self.max_centroid_shift(&new_centroids);
            self.centroids = new_centroids;
            self.n_iterations = iteration + 1;

            if max_shift < self.tolerance {
                break;
            }
        }

        self.inertia = self.calculate_inertia(data);
    }

    /// Random initialization: select k random data points as initial centroids
    fn random_init(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Simple deterministic selection for reproducibility
        let step = data.len() / self.k;
        (0..self.k).map(|i| data[i * step].clone()).collect()
    }

    /// K-means++ initialization: select initial centroids with probability
    /// proportional to squared distance from nearest existing centroid
    fn kmeanspp_init(&self, data: &[Vec<f64>], seed: u64) -> Vec<Vec<f64>> {
        let mut centroids = Vec::with_capacity(self.k);
        let mut rng = SimpleRng::new(seed);

        // Choose first centroid randomly
        let first_idx = rng.next_usize() % data.len();
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids
        for _ in 1..self.k {
            // Calculate squared distances to nearest centroid
            let distances: Vec<f64> = data
                .iter()
                .map(|point| {
                    centroids
                        .iter()
                        .map(|c| Self::squared_euclidean_distance(point, c))
                        .fold(f64::MAX, f64::min)
                })
                .collect();

            // Calculate cumulative probabilities
            let total: f64 = distances.iter().sum();
            if total == 0.0 {
                break;
            }

            let threshold = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut selected_idx = 0;

            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    selected_idx = i;
                    break;
                }
            }

            centroids.push(data[selected_idx].clone());
        }

        centroids
    }

    /// Assign each data point to the nearest centroid
    fn assign_clusters(&self, data: &[Vec<f64>]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                self.centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, Self::squared_euclidean_distance(point, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Calculate new centroids as the mean of assigned points
    fn calculate_centroids(&self, data: &[Vec<f64>], n_features: usize) -> Vec<Vec<f64>> {
        let mut new_centroids = vec![vec![0.0; n_features]; self.k];
        let mut counts = vec![0usize; self.k];

        for (point, &label) in data.iter().zip(self.labels.iter()) {
            counts[label] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[label][j] += val;
            }
        }

        // Calculate mean for each centroid
        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for val in centroid.iter_mut() {
                    *val /= counts[i] as f64;
                }
            } else {
                // If cluster is empty, keep the old centroid
                *centroid = self.centroids[i].clone();
            }
        }

        new_centroids
    }

    /// Calculate maximum shift of any centroid
    fn max_centroid_shift(&self, new_centroids: &[Vec<f64>]) -> f64 {
        self.centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| Self::squared_euclidean_distance(old, new).sqrt())
            .fold(0.0, f64::max)
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, data: &[Vec<f64>]) -> f64 {
        data.iter()
            .zip(self.labels.iter())
            .map(|(point, &label)| Self::squared_euclidean_distance(point, &self.centroids[label]))
            .sum()
    }

    /// Squared Euclidean distance between two points
    fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<usize> {
        self.assign_clusters(data)
    }

    /// Get the cluster centroids
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Get the labels for the training data
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get the inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Get the number of iterations run
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }
}

/// K-Means with silhouette score calculation
#[derive(Debug, Clone)]
pub struct KMeansWithMetrics {
    kmeans: KMeans,
}

impl KMeansWithMetrics {
    pub fn new(k: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            kmeans: KMeans::new(k, max_iterations, tolerance),
        }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) {
        self.kmeans.fit(data);
    }

    pub fn fit_kmeanspp(&mut self, data: &[Vec<f64>], seed: u64) {
        self.kmeans.fit_kmeanspp(data, seed);
    }

    /// Calculate silhouette score for clustering quality
    /// Range: -1 to 1, higher is better
    pub fn silhouette_score(&self, data: &[Vec<f64>]) -> f64 {
        if data.len() < 2 || self.kmeans.k < 2 {
            return 0.0;
        }

        let labels = &self.kmeans.labels;
        let mut silhouettes = Vec::with_capacity(data.len());

        for (i, point) in data.iter().enumerate() {
            let cluster = labels[i];

            // Calculate a(i) - average distance to same cluster
            let same_cluster: Vec<&Vec<f64>> = data
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && labels[*j] == cluster)
                .map(|(_, p)| p)
                .collect();

            let a = if same_cluster.is_empty() {
                0.0
            } else {
                same_cluster
                    .iter()
                    .map(|other| KMeans::squared_euclidean_distance(point, other).sqrt())
                    .sum::<f64>()
                    / same_cluster.len() as f64
            };

            // Calculate b(i) - minimum average distance to other clusters
            let mut cluster_distances: HashMap<usize, (f64, usize)> = HashMap::new();
            for (j, other) in data.iter().enumerate() {
                if labels[j] != cluster {
                    let entry = cluster_distances.entry(labels[j]).or_insert((0.0, 0));
                    entry.0 += KMeans::squared_euclidean_distance(point, other).sqrt();
                    entry.1 += 1;
                }
            }

            let b = cluster_distances
                .values()
                .map(|(sum, count)| sum / *count as f64)
                .fold(f64::MAX, f64::min);

            let b = if b == f64::MAX { 0.0 } else { b };

            // Silhouette coefficient for point i
            let s = if a.max(b) == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            };
            silhouettes.push(s);
        }

        silhouettes.iter().sum::<f64>() / silhouettes.len() as f64
    }

    pub fn kmeans(&self) -> &KMeans {
        &self.kmeans
    }
}

/// Simple PRNG for reproducible results without external dependencies
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Elbow method helper to find optimal k
pub fn elbow_method(data: &[Vec<f64>], max_k: usize, max_iterations: usize) -> Vec<(usize, f64)> {
    (1..=max_k)
        .map(|k| {
            let mut kmeans = KMeans::new(k, max_iterations, 1e-4);
            kmeans.fit(data);
            (k, kmeans.inertia())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_clustered_data() -> Vec<Vec<f64>> {
        // Three well-separated clusters
        let cluster1 = vec![
            vec![1.0, 1.0],
            vec![1.5, 1.5],
            vec![1.2, 0.8],
            vec![0.8, 1.2],
        ];
        let cluster2 = vec![
            vec![5.0, 5.0],
            vec![5.5, 5.5],
            vec![5.2, 4.8],
            vec![4.8, 5.2],
        ];
        let cluster3 = vec![
            vec![9.0, 1.0],
            vec![9.5, 1.5],
            vec![9.2, 0.8],
            vec![8.8, 1.2],
        ];

        [cluster1, cluster2, cluster3].concat()
    }

    #[test]
    fn test_kmeans_basic() {
        let data = create_clustered_data();
        let mut kmeans = KMeans::new(3, 100, 1e-4);
        kmeans.fit(&data);

        assert_eq!(kmeans.centroids().len(), 3);
        assert_eq!(kmeans.labels().len(), 12);
        assert!(kmeans.inertia() >= 0.0);
    }

    #[test]
    fn test_kmeans_finds_clusters() {
        let data = create_clustered_data();
        let mut kmeans = KMeans::new(3, 100, 1e-4);
        kmeans.fit(&data);

        let labels = kmeans.labels();

        // Points 0-3 should have same label (cluster 1)
        assert!(labels[0] == labels[1] && labels[1] == labels[2] && labels[2] == labels[3]);

        // Points 4-7 should have same label (cluster 2)
        assert!(labels[4] == labels[5] && labels[5] == labels[6] && labels[6] == labels[7]);

        // Points 8-11 should have same label (cluster 3)
        assert!(labels[8] == labels[9] && labels[9] == labels[10] && labels[10] == labels[11]);

        // Different clusters should have different labels
        assert!(labels[0] != labels[4]);
        assert!(labels[4] != labels[8]);
        assert!(labels[0] != labels[8]);
    }

    #[test]
    fn test_kmeans_predict() {
        let data = create_clustered_data();
        let mut kmeans = KMeans::new(3, 100, 1e-4);
        kmeans.fit(&data);

        // New points near each cluster
        let new_points = vec![
            vec![1.1, 1.1], // Near cluster 1
            vec![5.1, 5.1], // Near cluster 2
            vec![9.1, 1.1], // Near cluster 3
        ];

        let predictions = kmeans.predict(&new_points);

        // Should assign to same clusters as training data
        assert_eq!(predictions[0], kmeans.labels()[0]);
        assert_eq!(predictions[1], kmeans.labels()[4]);
        assert_eq!(predictions[2], kmeans.labels()[8]);
    }

    #[test]
    fn test_kmeans_convergence() {
        let data = create_clustered_data();
        let mut kmeans = KMeans::new(3, 1000, 1e-6);
        kmeans.fit(&data);

        // Should converge before max iterations for well-separated clusters
        assert!(kmeans.n_iterations() < 1000);
    }

    #[test]
    fn test_kmeanspp_initialization() {
        let data = create_clustered_data();
        let mut kmeans = KMeans::new(3, 100, 1e-4);
        kmeans.fit_kmeanspp(&data, 42);

        assert_eq!(kmeans.centroids().len(), 3);
        assert_eq!(kmeans.labels().len(), 12);

        // k-means++ should also find the clusters
        let labels = kmeans.labels();
        assert!(labels[0] == labels[1] && labels[1] == labels[2] && labels[2] == labels[3]);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![vec![1.0, 1.0], vec![1.1, 1.1], vec![0.9, 0.9]];
        let mut kmeans = KMeans::new(1, 100, 1e-4);
        kmeans.fit(&data);

        assert_eq!(kmeans.centroids().len(), 1);
        assert!(kmeans.labels().iter().all(|&l| l == 0));
    }

    #[test]
    fn test_kmeans_empty_data() {
        let data: Vec<Vec<f64>> = Vec::new();
        let mut kmeans = KMeans::new(3, 100, 1e-4);
        kmeans.fit(&data);

        assert!(kmeans.centroids().is_empty());
        assert!(kmeans.labels().is_empty());
    }

    #[test]
    fn test_kmeans_with_metrics() {
        let data = create_clustered_data();
        let mut kmeans = KMeansWithMetrics::new(3, 100, 1e-4);
        kmeans.fit(&data);

        let silhouette = kmeans.silhouette_score(&data);

        // Well-separated clusters should have high silhouette score
        assert!(silhouette > 0.5);
        assert!(silhouette <= 1.0);
    }

    #[test]
    fn test_silhouette_score_range() {
        let data = create_clustered_data();
        let mut kmeans = KMeansWithMetrics::new(3, 100, 1e-4);
        kmeans.fit(&data);

        let score = kmeans.silhouette_score(&data);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_elbow_method() {
        let data = create_clustered_data();
        let results = elbow_method(&data, 5, 100);

        assert_eq!(results.len(), 5);

        // Inertia should decrease as k increases
        for i in 1..results.len() {
            assert!(results[i].1 <= results[i - 1].1);
        }

        // k=3 should have significantly lower inertia than k=1
        assert!(results[2].1 < results[0].1 * 0.5);
    }

    #[test]
    fn test_kmeans_reproducibility() {
        let data = create_clustered_data();

        let mut kmeans1 = KMeans::new(3, 100, 1e-4);
        kmeans1.fit_kmeanspp(&data, 12345);

        let mut kmeans2 = KMeans::new(3, 100, 1e-4);
        kmeans2.fit_kmeanspp(&data, 12345);

        // Same seed should give same results
        assert_eq!(kmeans1.labels(), kmeans2.labels());
        assert_eq!(kmeans1.inertia(), kmeans2.inertia());
    }

    #[test]
    fn test_kmeans_higher_dimensions() {
        // 3D data
        let data = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.1, 1.1, 1.1],
            vec![5.0, 5.0, 5.0],
            vec![5.1, 5.1, 5.1],
        ];

        let mut kmeans = KMeans::new(2, 100, 1e-4);
        kmeans.fit(&data);

        let labels = kmeans.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = KMeans::squared_euclidean_distance(&a, &b);
        assert!((dist - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_calculation() {
        let data = vec![vec![0.0, 0.0], vec![2.0, 2.0], vec![4.0, 4.0]];

        let mut kmeans = KMeans::new(1, 100, 1e-4);
        kmeans.fit(&data);

        let centroid = &kmeans.centroids()[0];
        assert!((centroid[0] - 2.0).abs() < 1e-10);
        assert!((centroid[1] - 2.0).abs() < 1e-10);
    }
}
