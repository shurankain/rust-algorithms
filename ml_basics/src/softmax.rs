// Softmax and Cross-Entropy - Classification output layer and loss functions

/// Compute softmax probabilities for a single sample
/// Converts logits to probabilities that sum to 1
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Subtract max for numerical stability
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum).collect()
}

/// Compute softmax for multiple samples (batch)
pub fn softmax_batch(logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
    logits.iter().map(|row| softmax(row)).collect()
}

/// Cross-entropy loss for a single sample
/// y_true is one-hot encoded, y_pred is probabilities
pub fn cross_entropy_loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let epsilon = 1e-15; // Prevent log(0)
    -y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t * (p.max(epsilon)).ln())
        .sum::<f64>()
}

/// Cross-entropy loss for a batch
pub fn cross_entropy_loss_batch(y_true: &[Vec<f64>], y_pred: &[Vec<f64>]) -> f64 {
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| cross_entropy_loss(t, p))
        .sum::<f64>()
        / n
}

/// Sparse cross-entropy loss (labels as class indices instead of one-hot)
pub fn sparse_cross_entropy_loss(y_true: &[usize], y_pred: &[Vec<f64>]) -> f64 {
    let epsilon = 1e-15;
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&label, probs)| -(probs[label].max(epsilon)).ln())
        .sum::<f64>()
        / n
}

/// Convert class index to one-hot encoding
pub fn to_one_hot(label: usize, num_classes: usize) -> Vec<f64> {
    let mut one_hot = vec![0.0; num_classes];
    if label < num_classes {
        one_hot[label] = 1.0;
    }
    one_hot
}

/// Convert batch of class indices to one-hot encodings
pub fn to_one_hot_batch(labels: &[usize], num_classes: usize) -> Vec<Vec<f64>> {
    labels.iter().map(|&l| to_one_hot(l, num_classes)).collect()
}

/// Convert probabilities to predicted class (argmax)
pub fn argmax(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Softmax Regression (Multinomial Logistic Regression)
/// Multi-class classifier using softmax activation and cross-entropy loss
#[derive(Debug, Clone)]
pub struct SoftmaxRegression {
    weights: Vec<Vec<f64>>, // [num_classes][num_features]
    biases: Vec<f64>,       // [num_classes]
    learning_rate: f64,
    num_classes: usize,
}

impl SoftmaxRegression {
    /// Create a new softmax regression classifier
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
            learning_rate,
            num_classes: 0,
        }
    }

    /// Fit the model using gradient descent
    /// Returns the loss per epoch
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[usize],
        num_classes: usize,
        epochs: usize,
    ) -> Vec<f64> {
        if x.is_empty() {
            return Vec::new();
        }

        let n_features = x[0].len();
        self.num_classes = num_classes;

        // Initialize weights and biases
        self.weights = vec![vec![0.0; n_features]; num_classes];
        self.biases = vec![0.0; num_classes];

        let y_one_hot = to_one_hot_batch(y, num_classes);
        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            // Forward pass
            let logits = self.compute_logits(x);
            let probs = softmax_batch(&logits);

            // Compute loss
            let loss = cross_entropy_loss_batch(&y_one_hot, &probs);
            losses.push(loss);

            // Backward pass (gradient descent)
            let n = x.len() as f64;

            for i in 0..num_classes {
                for j in 0..n_features {
                    let mut grad = 0.0;
                    for (k, xk) in x.iter().enumerate() {
                        grad += (probs[k][i] - y_one_hot[k][i]) * xk[j];
                    }
                    self.weights[i][j] -= self.learning_rate * grad / n;
                }

                // Bias gradient
                let mut bias_grad = 0.0;
                for k in 0..x.len() {
                    bias_grad += probs[k][i] - y_one_hot[k][i];
                }
                self.biases[i] -= self.learning_rate * bias_grad / n;
            }
        }

        losses
    }

    /// Compute logits for input samples
    fn compute_logits(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|xi| {
                self.weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(w, &b)| w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                    .collect()
            })
            .collect()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        softmax_batch(&self.compute_logits(x))
    }

    /// Predict class labels
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        self.predict_proba(x).iter().map(|p| argmax(p)).collect()
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

    /// Get learned weights
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }

    /// Get learned biases
    pub fn biases(&self) -> &[f64] {
        &self.biases
    }
}

/// Temperature-scaled softmax for controlling prediction confidence
pub fn softmax_with_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() || temperature <= 0.0 {
        return Vec::new();
    }

    let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();
    softmax(&scaled)
}

/// Log-softmax for numerical stability in loss computation
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f64>()
        .ln()
        + max_logit;

    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Negative log-likelihood loss using log-softmax
pub fn nll_loss(log_probs: &[f64], target: usize) -> f64 {
    if target < log_probs.len() {
        -log_probs[target]
    } else {
        0.0
    }
}

/// Gumbel-softmax for differentiable sampling
pub fn gumbel_softmax(logits: &[f64], temperature: f64, seed: u64) -> Vec<f64> {
    let mut rng = SimpleRng::new(seed);
    let gumbel_noise: Vec<f64> = logits
        .iter()
        .map(|_| {
            let u = rng.next_f64().max(1e-10);
            -((-u.ln()).ln())
        })
        .collect();

    let perturbed: Vec<f64> = logits
        .iter()
        .zip(gumbel_noise.iter())
        .map(|(&l, &g)| (l + g) / temperature)
        .collect();

    softmax(&perturbed)
}

/// Simple PRNG for reproducible results
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
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // All probabilities should be positive
        assert!(probs.iter().all(|&p| p > 0.0));

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could overflow without max subtraction
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-1.0, -2.0, -3.0];
        let probs = softmax(&logits);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f64> = Vec::new();
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_single_element() {
        let logits = vec![5.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let y_true = vec![0.0, 1.0, 0.0];
        let y_pred = vec![0.1, 0.8, 0.1];

        let loss = cross_entropy_loss(&y_true, &y_pred);
        // -log(0.8) ≈ 0.223
        assert!((loss - 0.223).abs() < 0.01);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let y_true = vec![0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 0.0];

        let loss = cross_entropy_loss(&y_true, &y_pred);
        assert!(loss < 1e-10);
    }

    #[test]
    fn test_sparse_cross_entropy() {
        let y_true = vec![1, 0, 2];
        let y_pred = vec![
            vec![0.1, 0.8, 0.1],
            vec![0.9, 0.05, 0.05],
            vec![0.1, 0.1, 0.8],
        ];

        let loss = sparse_cross_entropy_loss(&y_true, &y_pred);
        // All predictions are fairly good, loss should be low
        assert!(loss < 0.5);
    }

    #[test]
    fn test_to_one_hot() {
        let one_hot = to_one_hot(2, 4);
        assert_eq!(one_hot, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_to_one_hot_batch() {
        let labels = vec![0, 1, 2];
        let one_hot = to_one_hot_batch(&labels, 3);

        assert_eq!(one_hot[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(one_hot[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_argmax() {
        let probs = vec![0.1, 0.7, 0.2];
        assert_eq!(argmax(&probs), 1);
    }

    #[test]
    fn test_softmax_regression_basic() {
        // Simple 3-class classification
        let x = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 0.0],
            vec![5.5, 0.5],
            vec![2.5, 5.0],
            vec![3.0, 5.5],
        ];
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut model = SoftmaxRegression::new(0.1);
        let losses = model.fit(&x, &y, 3, 100);

        // Loss should decrease
        assert!(losses.last().unwrap() < losses.first().unwrap());

        // Should achieve reasonable accuracy
        let score = model.score(&x, &y);
        assert!(score > 0.5);
    }

    #[test]
    fn test_softmax_regression_predict_proba() {
        let x = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        let y = vec![0, 1];

        let mut model = SoftmaxRegression::new(0.5);
        model.fit(&x, &y, 2, 100);

        let probs = model.predict_proba(&x);

        // Each row should sum to 1
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0];

        // Low temperature -> more confident (sharper distribution)
        let low_temp = softmax_with_temperature(&logits, 0.5);
        // High temperature -> less confident (flatter distribution)
        let high_temp = softmax_with_temperature(&logits, 2.0);

        // Lower temperature should have higher max probability
        let max_low: f64 = low_temp.iter().fold(0.0, |a, &b| a.max(b));
        let max_high: f64 = high_temp.iter().fold(0.0, |a, &b| a.max(b));
        assert!(max_low > max_high);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);
        let probs = softmax(&logits);

        // log_softmax should equal log(softmax)
        for (lp, p) in log_probs.iter().zip(probs.iter()) {
            assert!((lp - p.ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nll_loss() {
        let log_probs = vec![-2.0, -0.5, -1.5]; // log probabilities
        let target = 1;

        let loss = nll_loss(&log_probs, target);
        assert!((loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let result = gumbel_softmax(&logits, 1.0, 42);

        // Should still sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // All values should be valid probabilities
        assert!(result.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_gumbel_softmax_reproducibility() {
        let logits = vec![1.0, 2.0, 3.0];
        let result1 = gumbel_softmax(&logits, 1.0, 123);
        let result2 = gumbel_softmax(&logits, 1.0, 123);

        // Same seed should give same result
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_regression_empty() {
        let x: Vec<Vec<f64>> = Vec::new();
        let y: Vec<usize> = Vec::new();

        let mut model = SoftmaxRegression::new(0.1);
        let losses = model.fit(&x, &y, 3, 100);

        assert!(losses.is_empty());
    }
}
