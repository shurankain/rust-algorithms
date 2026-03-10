// Perceptron - The foundational neural network unit for binary classification

/// Single-layer Perceptron for binary classification
/// Learns a linear decision boundary: y = sign(w·x + b)
#[derive(Debug, Clone)]
pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    n_iterations: usize,
}

impl Perceptron {
    /// Create a new Perceptron
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            n_iterations: 0,
        }
    }

    /// Fit the perceptron using the perceptron learning algorithm
    /// Returns the number of misclassifications per epoch
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[i32], max_epochs: usize) -> Vec<usize> {
        if x.is_empty() {
            return Vec::new();
        }

        let n_features = x[0].len();
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let mut errors_per_epoch = Vec::with_capacity(max_epochs);

        for epoch in 0..max_epochs {
            let mut errors = 0;

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let prediction = self.predict_one(xi);
                let update = self.learning_rate * (yi - prediction) as f64;

                if update != 0.0 {
                    errors += 1;
                    for (w, &xij) in self.weights.iter_mut().zip(xi.iter()) {
                        *w += update * xij;
                    }
                    self.bias += update;
                }
            }

            errors_per_epoch.push(errors);
            self.n_iterations = epoch + 1;

            // Early stopping if no errors
            if errors == 0 {
                break;
            }
        }

        errors_per_epoch
    }

    /// Predict class labels for multiple samples
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i32> {
        x.iter().map(|xi| self.predict_one(xi)).collect()
    }

    /// Predict class label for a single sample
    fn predict_one(&self, x: &[f64]) -> i32 {
        let activation = self.net_input(x);
        if activation >= 0.0 { 1 } else { -1 }
    }

    /// Calculate net input (weighted sum)
    fn net_input(&self, x: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>()
            + self.bias
    }

    /// Calculate accuracy on test data
    pub fn score(&self, x: &[Vec<f64>], y: &[i32]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| *pred == *actual)
            .count();
        correct as f64 / y.len() as f64
    }

    /// Get the learned weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the learned bias
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Get the number of iterations run
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }
}

/// Multi-class Perceptron using one-vs-all strategy
#[derive(Debug, Clone)]
pub struct MultiClassPerceptron {
    classifiers: Vec<Perceptron>,
    classes: Vec<usize>,
    learning_rate: f64,
}

impl MultiClassPerceptron {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            learning_rate,
        }
    }

    /// Fit using one-vs-all strategy
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize], max_epochs: usize) {
        // Find unique classes
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort();
        classes.dedup();
        self.classes = classes;

        // Train one perceptron per class
        self.classifiers = self
            .classes
            .iter()
            .map(|&class| {
                let binary_y: Vec<i32> = y
                    .iter()
                    .map(|&yi| if yi == class { 1 } else { -1 })
                    .collect();

                let mut perceptron = Perceptron::new(self.learning_rate);
                perceptron.fit(x, &binary_y, max_epochs);
                perceptron
            })
            .collect();
    }

    /// Predict class labels
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|xi| self.predict_one(xi)).collect()
    }

    /// Predict class label for a single sample
    fn predict_one(&self, x: &[f64]) -> usize {
        // Return class with highest net input
        self.classifiers
            .iter()
            .zip(self.classes.iter())
            .map(|(clf, &class)| (class, clf.net_input(x)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
            .unwrap_or(0)
    }

    /// Calculate accuracy
    pub fn score(&self, x: &[Vec<f64>], y: &[usize]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| *pred == *actual)
            .count();
        correct as f64 / y.len() as f64
    }
}

/// Averaged Perceptron - more robust version that averages weights
#[derive(Debug, Clone)]
pub struct AveragedPerceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl AveragedPerceptron {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
        }
    }

    /// Fit using averaged perceptron algorithm
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[i32], max_epochs: usize) {
        if x.is_empty() {
            return;
        }

        let n_features = x[0].len();
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        // Cached weights for averaging
        let mut cached_weights = vec![0.0; n_features];
        let mut cached_bias = 0.0;
        let mut counter = 1.0;

        for _ in 0..max_epochs {
            let mut converged = true;

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let activation: f64 = weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                let prediction = if activation >= 0.0 { 1 } else { -1 };

                if prediction != yi {
                    converged = false;
                    let update = self.learning_rate * yi as f64;

                    for (j, &xij) in xi.iter().enumerate() {
                        weights[j] += update * xij;
                        cached_weights[j] += counter * update * xij;
                    }
                    bias += update;
                    cached_bias += counter * update;
                }
                counter += 1.0;
            }

            if converged {
                break;
            }
        }

        // Average the weights
        self.weights = weights
            .iter()
            .zip(cached_weights.iter())
            .map(|(w, cw)| w - cw / counter)
            .collect();
        self.bias = bias - cached_bias / counter;
    }

    /// Predict class labels
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i32> {
        x.iter()
            .map(|xi| {
                let activation: f64 = self
                    .weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.bias;
                if activation >= 0.0 { 1 } else { -1 }
            })
            .collect()
    }

    /// Calculate accuracy
    pub fn score(&self, x: &[Vec<f64>], y: &[i32]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| *pred == *actual)
            .count();
        correct as f64 / y.len() as f64
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

/// Pocket Perceptron - keeps best weights seen during training
#[derive(Debug, Clone)]
pub struct PocketPerceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl PocketPerceptron {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
        }
    }

    /// Fit using pocket algorithm (keeps best weights)
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[i32], max_epochs: usize) {
        if x.is_empty() {
            return;
        }

        let n_features = x[0].len();
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        // Best weights seen so far
        let mut best_weights = weights.clone();
        let mut best_bias = bias;
        let mut best_correct = 0;

        for _ in 0..max_epochs {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let activation: f64 = weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                let prediction = if activation >= 0.0 { 1 } else { -1 };

                if prediction != yi {
                    let update = self.learning_rate * yi as f64;
                    for (j, &xij) in xi.iter().enumerate() {
                        weights[j] += update * xij;
                    }
                    bias += update;

                    // Check if this is the best so far
                    let correct = self.count_correct(&weights, bias, x, y);
                    if correct > best_correct {
                        best_correct = correct;
                        best_weights = weights.clone();
                        best_bias = bias;
                    }
                }
            }
        }

        self.weights = best_weights;
        self.bias = best_bias;
    }

    fn count_correct(&self, weights: &[f64], bias: f64, x: &[Vec<f64>], y: &[i32]) -> usize {
        x.iter()
            .zip(y.iter())
            .filter(|(xi, yi)| {
                let activation: f64 = weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                let prediction = if activation >= 0.0 { 1 } else { -1 };
                prediction == **yi
            })
            .count()
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i32> {
        x.iter()
            .map(|xi| {
                let activation: f64 = self
                    .weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.bias;
                if activation >= 0.0 { 1 } else { -1 }
            })
            .collect()
    }

    pub fn score(&self, x: &[Vec<f64>], y: &[i32]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| *pred == *actual)
            .count();
        correct as f64 / y.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linearly_separable_data() -> (Vec<Vec<f64>>, Vec<i32>) {
        // Two linearly separable clusters
        let x = vec![
            // Class -1 (lower left)
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            // Class 1 (upper right)
            vec![3.0, 3.0],
            vec![3.5, 3.5],
            vec![4.0, 3.0],
            vec![3.0, 4.0],
        ];
        let y = vec![-1, -1, -1, -1, 1, 1, 1, 1];
        (x, y)
    }

    fn create_xor_data() -> (Vec<Vec<f64>>, Vec<i32>) {
        // XOR - not linearly separable
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let y = vec![-1, 1, 1, -1];
        (x, y)
    }

    #[test]
    fn test_perceptron_linearly_separable() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = Perceptron::new(0.1);
        let errors = perceptron.fit(&x, &y, 100);

        // Should converge to zero errors
        assert_eq!(*errors.last().unwrap(), 0);

        // Should achieve perfect accuracy
        let score = perceptron.score(&x, &y);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_perceptron_weights_learned() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = Perceptron::new(0.1);
        perceptron.fit(&x, &y, 100);

        // Weights should be non-zero
        assert!(perceptron.weights().iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_perceptron_early_stopping() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = Perceptron::new(0.1);
        perceptron.fit(&x, &y, 1000);

        // Should stop early since data is linearly separable
        assert!(perceptron.n_iterations() < 1000);
    }

    #[test]
    fn test_perceptron_predict() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = Perceptron::new(0.1);
        perceptron.fit(&x, &y, 100);

        // Test new points
        let test = vec![vec![0.5, 0.5], vec![3.5, 3.5]];
        let predictions = perceptron.predict(&test);

        assert_eq!(predictions[0], -1);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_perceptron_xor_not_separable() {
        let (x, y) = create_xor_data();
        let mut perceptron = Perceptron::new(0.1);
        let errors = perceptron.fit(&x, &y, 100);

        // Should NOT converge to zero errors (XOR is not linearly separable)
        assert!(*errors.last().unwrap() > 0);
    }

    #[test]
    fn test_multiclass_perceptron() {
        // Three classes
        let x = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 0.0],
            vec![5.5, 0.5],
            vec![2.5, 5.0],
            vec![3.0, 5.5],
        ];
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut classifier = MultiClassPerceptron::new(0.1);
        classifier.fit(&x, &y, 100);

        let score = classifier.score(&x, &y);
        assert!(score > 0.5);
    }

    #[test]
    fn test_averaged_perceptron() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = AveragedPerceptron::new(0.1);
        perceptron.fit(&x, &y, 200);

        let score = perceptron.score(&x, &y);
        // Averaged perceptron may not achieve perfect accuracy due to weight averaging
        assert!(score > 0.8, "Expected score > 0.8, got {}", score);
    }

    #[test]
    fn test_pocket_perceptron() {
        let (x, y) = create_linearly_separable_data();
        let mut perceptron = PocketPerceptron::new(0.1);
        perceptron.fit(&x, &y, 100);

        let score = perceptron.score(&x, &y);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_pocket_perceptron_non_separable() {
        let (x, y) = create_xor_data();
        let mut perceptron = PocketPerceptron::new(0.1);
        perceptron.fit(&x, &y, 100);

        // Pocket should keep the best weights it found
        let score = perceptron.score(&x, &y);
        // At best can get 50% on XOR
        assert!(score >= 0.5);
    }

    #[test]
    fn test_perceptron_empty_data() {
        let x: Vec<Vec<f64>> = Vec::new();
        let y: Vec<i32> = Vec::new();
        let mut perceptron = Perceptron::new(0.1);
        let errors = perceptron.fit(&x, &y, 100);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_perceptron_3d_data() {
        let x = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.5],
            vec![5.0, 5.0, 5.0],
            vec![5.5, 5.5, 5.5],
        ];
        let y = vec![-1, -1, 1, 1];

        let mut perceptron = Perceptron::new(0.1);
        perceptron.fit(&x, &y, 100);

        assert_eq!(perceptron.weights().len(), 3);
        assert_eq!(perceptron.score(&x, &y), 1.0);
    }

    #[test]
    fn test_perceptron_learning_rate_effect() {
        let (x, y) = create_linearly_separable_data();

        let mut slow = Perceptron::new(0.01);
        let mut fast = Perceptron::new(1.0);

        slow.fit(&x, &y, 50);
        fast.fit(&x, &y, 50);

        // Both should converge, but potentially at different rates
        assert!(slow.score(&x, &y) > 0.5);
        assert!(fast.score(&x, &y) > 0.5);
    }
}
