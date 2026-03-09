// K-Nearest Neighbors (KNN) - Instance-based learning for classification and regression

use std::collections::HashMap;

/// K-Nearest Neighbors classifier
/// Classifies new points based on majority vote of k nearest training points
#[derive(Debug, Clone)]
pub struct KNNClassifier {
    k: usize,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<usize>,
    distance_metric: DistanceMetric,
}

/// K-Nearest Neighbors regressor
/// Predicts values based on average of k nearest training points
#[derive(Debug, Clone)]
pub struct KNNRegressor {
    k: usize,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    distance_metric: DistanceMetric,
    weighted: bool,
}

/// Distance metrics for KNN
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64), // p-norm
    Chebyshev,      // L-infinity
}

impl KNNClassifier {
    /// Create a new KNN classifier
    pub fn new(k: usize) -> Self {
        Self {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
            distance_metric: DistanceMetric::Euclidean,
        }
    }

    /// Create KNN classifier with custom distance metric
    pub fn with_metric(k: usize, metric: DistanceMetric) -> Self {
        Self {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
            distance_metric: metric,
        }
    }

    /// Fit the model (store training data)
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();
    }

    /// Predict class labels for new data
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|point| self.predict_one(point)).collect()
    }

    /// Predict class label for a single point
    fn predict_one(&self, point: &[f64]) -> usize {
        let neighbors = self.find_k_nearest(point);

        // Majority vote
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for &(_, label) in &neighbors {
            *votes.entry(label).or_insert(0) += 1;
        }

        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0)
    }

    /// Predict class probabilities for new data
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<HashMap<usize, f64>> {
        x.iter()
            .map(|point| self.predict_proba_one(point))
            .collect()
    }

    /// Predict class probabilities for a single point
    fn predict_proba_one(&self, point: &[f64]) -> HashMap<usize, f64> {
        let neighbors = self.find_k_nearest(point);

        let mut votes: HashMap<usize, usize> = HashMap::new();
        for &(_, label) in &neighbors {
            *votes.entry(label).or_insert(0) += 1;
        }

        let total = neighbors.len() as f64;
        votes
            .into_iter()
            .map(|(label, count)| (label, count as f64 / total))
            .collect()
    }

    /// Find k nearest neighbors, returns (distance, label) pairs
    fn find_k_nearest(&self, point: &[f64]) -> Vec<(f64, usize)> {
        let mut distances: Vec<(f64, usize)> = self
            .x_train
            .iter()
            .zip(self.y_train.iter())
            .map(|(x, &y)| (self.distance(point, x), y))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);
        distances
    }

    /// Calculate distance between two points using the configured metric
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric)
    }

    /// Calculate accuracy score on test data
    pub fn score(&self, x: &[Vec<f64>], y: &[usize]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| **pred == **actual)
            .count();
        correct as f64 / y.len() as f64
    }
}

impl KNNRegressor {
    /// Create a new KNN regressor
    pub fn new(k: usize) -> Self {
        Self {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
            distance_metric: DistanceMetric::Euclidean,
            weighted: false,
        }
    }

    /// Create KNN regressor with distance-weighted averaging
    pub fn weighted(k: usize) -> Self {
        Self {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
            distance_metric: DistanceMetric::Euclidean,
            weighted: true,
        }
    }

    /// Create KNN regressor with custom distance metric
    pub fn with_metric(k: usize, metric: DistanceMetric, weighted: bool) -> Self {
        Self {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
            distance_metric: metric,
            weighted,
        }
    }

    /// Fit the model (store training data)
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();
    }

    /// Predict values for new data
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|point| self.predict_one(point)).collect()
    }

    /// Predict value for a single point
    fn predict_one(&self, point: &[f64]) -> f64 {
        let neighbors = self.find_k_nearest(point);

        if self.weighted {
            // Distance-weighted average (inverse distance weighting)
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (dist, value) in neighbors {
                let weight = if dist < 1e-10 {
                    // If distance is essentially zero, this point dominates
                    return value;
                } else {
                    1.0 / dist
                };
                weighted_sum += weight * value;
                weight_total += weight;
            }

            if weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                0.0
            }
        } else {
            // Simple average
            let sum: f64 = neighbors.iter().map(|(_, v)| v).sum();
            sum / neighbors.len() as f64
        }
    }

    /// Find k nearest neighbors, returns (distance, value) pairs
    fn find_k_nearest(&self, point: &[f64]) -> Vec<(f64, f64)> {
        let mut distances: Vec<(f64, f64)> = self
            .x_train
            .iter()
            .zip(self.y_train.iter())
            .map(|(x, &y)| (self.distance(point, x), y))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);
        distances
    }

    /// Calculate distance between two points
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric)
    }

    /// Calculate R² score on test data
    pub fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, actual)| (actual - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

/// Compute distance between two points using the specified metric
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),

        DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum(),

        DistanceMetric::Minkowski(p) => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p),

        DistanceMetric::Chebyshev => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max),
    }
}

/// Normalize features to [0, 1] range (min-max scaling)
pub fn normalize_features(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }

    let n_features = data[0].len();
    let mut mins = vec![f64::MAX; n_features];
    let mut maxs = vec![f64::MIN; n_features];

    for point in data {
        for (i, &val) in point.iter().enumerate() {
            mins[i] = mins[i].min(val);
            maxs[i] = maxs[i].max(val);
        }
    }

    data.iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let range = maxs[i] - mins[i];
                    if range == 0.0 {
                        0.0
                    } else {
                        (val - mins[i]) / range
                    }
                })
                .collect()
        })
        .collect()
}

/// Standardize features (z-score normalization)
pub fn standardize_features(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }

    let n_features = data[0].len();
    let n_samples = data.len() as f64;

    // Calculate means
    let mut means = vec![0.0; n_features];
    for point in data {
        for (i, &val) in point.iter().enumerate() {
            means[i] += val;
        }
    }
    for mean in &mut means {
        *mean /= n_samples;
    }

    // Calculate standard deviations
    let mut stds = vec![0.0; n_features];
    for point in data {
        for (i, &val) in point.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n_samples).sqrt();
    }

    // Standardize
    data.iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    if stds[i] == 0.0 {
                        0.0
                    } else {
                        (val - means[i]) / stds[i]
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_classification_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        // Two well-separated clusters
        let x = vec![
            // Class 0 (around origin)
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.3, 0.2],
            vec![0.1, 0.4],
            // Class 1 (around (5, 5))
            vec![5.0, 5.0],
            vec![5.5, 5.5],
            vec![5.3, 5.2],
            vec![5.1, 5.4],
        ];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn create_regression_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        // y = x1 + x2
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        (x, y)
    }

    #[test]
    fn test_knn_classifier_basic() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::new(3);
        knn.fit(&x, &y);

        // Test points clearly in each class
        let test = vec![vec![0.2, 0.3], vec![5.2, 5.3]];
        let predictions = knn.predict(&test);

        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_knn_classifier_accuracy() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::new(3);
        knn.fit(&x, &y);

        let score = knn.score(&x, &y);
        assert_eq!(score, 1.0); // Should perfectly classify training data
    }

    #[test]
    fn test_knn_classifier_proba() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::new(3);
        knn.fit(&x, &y);

        let test = vec![vec![0.2, 0.3]];
        let proba = knn.predict_proba(&test);

        assert!(proba[0].get(&0).unwrap() > &0.5);
    }

    #[test]
    fn test_knn_regressor_basic() {
        let (x, y) = create_regression_data();
        let mut knn = KNNRegressor::new(2);
        knn.fit(&x, &y);

        // Predict at a point between training points
        let test = vec![vec![2.5, 2.5]];
        let predictions = knn.predict(&test);

        // Should be average of two nearest: (4.0 + 6.0) / 2 = 5.0
        assert!((predictions[0] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_knn_regressor_weighted() {
        let (x, y) = create_regression_data();
        let mut knn = KNNRegressor::weighted(3);
        knn.fit(&x, &y);

        let score = knn.score(&x, &y);
        assert!(score > 0.9);
    }

    #[test]
    fn test_knn_regressor_score() {
        let (x, y) = create_regression_data();
        let mut knn = KNNRegressor::new(1);
        knn.fit(&x, &y);

        // With k=1, training score should be perfect
        let score = knn.score(&x, &y);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Euclidean);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Manhattan);
        assert!((dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        // p=2 should equal Euclidean
        let dist = compute_distance(&a, &b, DistanceMetric::Minkowski(2.0));
        assert!((dist - 5.0).abs() < 1e-10);

        // p=1 should equal Manhattan
        let dist = compute_distance(&a, &b, DistanceMetric::Minkowski(1.0));
        assert!((dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Chebyshev);
        assert!((dist - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_with_manhattan() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::with_metric(3, DistanceMetric::Manhattan);
        knn.fit(&x, &y);

        let test = vec![vec![0.2, 0.3]];
        let predictions = knn.predict(&test);
        assert_eq!(predictions[0], 0);
    }

    #[test]
    fn test_normalize_features() {
        let data = vec![vec![0.0, 10.0], vec![5.0, 20.0], vec![10.0, 30.0]];

        let normalized = normalize_features(&data);

        // All values should be in [0, 1]
        for point in &normalized {
            for &val in point {
                assert!(val >= 0.0 && val <= 1.0);
            }
        }

        // First feature: 0->0, 5->0.5, 10->1
        assert!((normalized[0][0] - 0.0).abs() < 1e-10);
        assert!((normalized[1][0] - 0.5).abs() < 1e-10);
        assert!((normalized[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardize_features() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];

        let standardized = standardize_features(&data);

        // Mean should be approximately 0
        let mean: f64 = standardized.iter().map(|p| p[0]).sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-10);

        // Std should be approximately 1
        let variance: f64 = standardized.iter().map(|p| p[0].powi(2)).sum::<f64>() / 5.0;
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_knn_empty_data() {
        let data: Vec<Vec<f64>> = Vec::new();
        let normalized = normalize_features(&data);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_knn_higher_k() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::new(5);
        knn.fit(&x, &y);

        // Should still work with higher k
        let score = knn.score(&x, &y);
        assert!(score > 0.5);
    }

    #[test]
    fn test_knn_3d_data() {
        let x = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![5.0, 5.0, 5.0],
            vec![5.1, 5.1, 5.1],
        ];
        let y = vec![0, 0, 1, 1];

        let mut knn = KNNClassifier::new(1);
        knn.fit(&x, &y);

        let test = vec![vec![0.05, 0.05, 0.05], vec![5.05, 5.05, 5.05]];
        let predictions = knn.predict(&test);

        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }
}
