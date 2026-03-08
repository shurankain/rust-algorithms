// Linear Regression with Gradient Descent
//
// Linear regression finds the best linear relationship between features and target.
// Uses gradient descent to minimize the Mean Squared Error (MSE) loss function.
//
// Model: y = X * w + b (or y = X * W where X includes bias column)
//
// Gradient Descent Update:
// - w = w - learning_rate * gradient
// - gradient = (2/n) * X^T * (X*w - y)
//
// Time Complexity per iteration: O(n * d) where n = samples, d = features
// Space Complexity: O(d) for weights

/// Simple linear regression (single feature)
/// y = w * x + b
pub struct SimpleLinearRegression {
    pub weight: f64,
    pub bias: f64,
    learning_rate: f64,
}

impl SimpleLinearRegression {
    /// Create a new simple linear regression model
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weight: 0.0,
            bias: 0.0,
            learning_rate,
        }
    }

    /// Fit the model using gradient descent
    pub fn fit(&mut self, x: &[f64], y: &[f64], epochs: usize) -> Vec<f64> {
        assert_eq!(x.len(), y.len(), "x and y must have same length");
        let n = x.len() as f64;
        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            // Compute predictions
            let predictions: Vec<f64> = x.iter().map(|&xi| self.predict_single(xi)).collect();

            // Compute loss (MSE)
            let loss: f64 = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f64>()
                / n;
            losses.push(loss);

            // Compute gradients
            let dw: f64 = (2.0 / n)
                * x.iter()
                    .zip(predictions.iter())
                    .zip(y.iter())
                    .map(|((xi, pred), yi)| xi * (pred - yi))
                    .sum::<f64>();

            let db: f64 = (2.0 / n)
                * predictions
                    .iter()
                    .zip(y.iter())
                    .map(|(pred, yi)| pred - yi)
                    .sum::<f64>();

            // Update parameters
            self.weight -= self.learning_rate * dw;
            self.bias -= self.learning_rate * db;
        }

        losses
    }

    /// Predict for a single value
    pub fn predict_single(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    /// Predict for multiple values
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.predict_single(xi)).collect()
    }

    /// Compute R² score (coefficient of determination)
    pub fn score(&self, x: &[f64], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, yi)| (yi - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            1.0 // Perfect fit if all y values are the same
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

impl Default for SimpleLinearRegression {
    fn default() -> Self {
        Self::new(0.01)
    }
}

/// Multiple linear regression (multiple features)
/// y = X * w + b (or y = X_aug * W where X_aug includes bias column)
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    learning_rate: f64,
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
        }
    }

    /// Fit the model using gradient descent
    /// x: 2D array as flat vec, shape [n_samples, n_features]
    /// y: target values, shape [n_samples]
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) -> Vec<f64> {
        assert!(!x.is_empty(), "x cannot be empty");
        assert_eq!(x.len(), y.len(), "x and y must have same number of samples");

        let n_samples = x.len();
        let n_features = x[0].len();
        let n = n_samples as f64;

        // Initialize weights to zero
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            // Compute predictions
            let predictions: Vec<f64> = x.iter().map(|xi| self.predict_single(xi)).collect();

            // Compute loss (MSE)
            let loss: f64 = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f64>()
                / n;
            losses.push(loss);

            // Compute errors
            let errors: Vec<f64> = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, yi)| pred - yi)
                .collect();

            // Compute gradients for weights
            let mut dw = vec![0.0; n_features];
            for (xi, error) in x.iter().zip(errors.iter()) {
                for (j, &xij) in xi.iter().enumerate() {
                    dw[j] += xij * error;
                }
            }
            for dwj in &mut dw {
                *dwj *= 2.0 / n;
            }

            // Compute gradient for bias
            let db: f64 = (2.0 / n) * errors.iter().sum::<f64>();

            // Update parameters
            for (wj, dwj) in self.weights.iter_mut().zip(dw.iter()) {
                *wj -= self.learning_rate * dwj;
            }
            self.bias -= self.learning_rate * db;
        }

        losses
    }

    /// Predict for a single sample
    pub fn predict_single(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + self.bias
    }

    /// Predict for multiple samples
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_single(xi)).collect()
    }

    /// Compute R² score
    pub fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, yi)| (yi - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new(0.01)
    }
}

/// Linear Regression with L2 Regularization (Ridge Regression)
/// Loss = MSE + lambda * ||w||²
pub struct RidgeRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    learning_rate: f64,
    lambda: f64, // Regularization strength
}

impl RidgeRegression {
    pub fn new(learning_rate: f64, lambda: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            lambda,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) -> Vec<f64> {
        assert!(!x.is_empty(), "x cannot be empty");
        assert_eq!(x.len(), y.len(), "x and y must have same number of samples");

        let n_samples = x.len();
        let n_features = x[0].len();
        let n = n_samples as f64;

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            let predictions: Vec<f64> = x.iter().map(|xi| self.predict_single(xi)).collect();

            // Loss = MSE + regularization
            let mse: f64 = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f64>()
                / n;
            let reg_term: f64 = self.lambda * self.weights.iter().map(|w| w.powi(2)).sum::<f64>();
            losses.push(mse + reg_term);

            let errors: Vec<f64> = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, yi)| pred - yi)
                .collect();

            // Gradients with regularization
            let mut dw = vec![0.0; n_features];
            for (xi, error) in x.iter().zip(errors.iter()) {
                for (j, &xij) in xi.iter().enumerate() {
                    dw[j] += xij * error;
                }
            }
            for (j, dwj) in dw.iter_mut().enumerate() {
                *dwj = (2.0 / n) * *dwj + 2.0 * self.lambda * self.weights[j];
            }

            let db: f64 = (2.0 / n) * errors.iter().sum::<f64>();

            for (wj, dwj) in self.weights.iter_mut().zip(dw.iter()) {
                *wj -= self.learning_rate * dwj;
            }
            self.bias -= self.learning_rate * db;
        }

        losses
    }

    pub fn predict_single(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + self.bias
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_single(xi)).collect()
    }

    pub fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, yi)| (yi - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

/// Stochastic Gradient Descent Linear Regression
/// Updates weights using mini-batches for efficiency on large datasets
pub struct SGDRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    learning_rate: f64,
    batch_size: usize,
}

impl SGDRegression {
    pub fn new(learning_rate: f64, batch_size: usize) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            batch_size,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) -> Vec<f64> {
        assert!(!x.is_empty(), "x cannot be empty");
        assert_eq!(x.len(), y.len(), "x and y must have same number of samples");

        let n_samples = x.len();
        let n_features = x[0].len();

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            // Compute epoch loss
            let predictions: Vec<f64> = x.iter().map(|xi| self.predict_single(xi)).collect();
            let loss: f64 = predictions
                .iter()
                .zip(y.iter())
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            losses.push(loss);

            // Process in batches
            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_x = &x[batch_start..batch_end];
                let batch_y = &y[batch_start..batch_end];
                let batch_n = batch_x.len() as f64;

                let batch_preds: Vec<f64> =
                    batch_x.iter().map(|xi| self.predict_single(xi)).collect();

                let errors: Vec<f64> = batch_preds
                    .iter()
                    .zip(batch_y.iter())
                    .map(|(pred, yi)| pred - yi)
                    .collect();

                let mut dw = vec![0.0; n_features];
                for (xi, error) in batch_x.iter().zip(errors.iter()) {
                    for (j, &xij) in xi.iter().enumerate() {
                        dw[j] += xij * error;
                    }
                }
                for dwj in &mut dw {
                    *dwj *= 2.0 / batch_n;
                }

                let db: f64 = (2.0 / batch_n) * errors.iter().sum::<f64>();

                for (wj, dwj) in self.weights.iter_mut().zip(dw.iter()) {
                    *wj -= self.learning_rate * dwj;
                }
                self.bias -= self.learning_rate * db;
            }
        }

        losses
    }

    pub fn predict_single(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + self.bias
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_single(xi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_linear_regression_basic() {
        // y = 2x + 1
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let mut model = SimpleLinearRegression::new(0.0001);
        let losses = model.fit(&x, &y, 1000);

        // Loss should decrease
        assert!(losses.last().unwrap() < &losses[0]);

        // Weights should be close to true values
        assert!((model.weight - 2.0).abs() < 0.1);
        assert!((model.bias - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_simple_linear_regression_score() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();

        let mut model = SimpleLinearRegression::new(0.1);
        model.fit(&x, &y, 500);

        let score = model.score(&x, &y);
        assert!(score > 0.99, "R² score should be > 0.99, got {}", score);
    }

    #[test]
    fn test_multiple_linear_regression() {
        // y = 2*x1 + 3*x2 + 1
        let x: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64 * 0.1, (100 - i) as f64 * 0.1])
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] + 3.0 * xi[1] + 1.0).collect();

        let mut model = LinearRegression::new(0.01);
        let losses = model.fit(&x, &y, 500);

        assert!(losses.last().unwrap() < &losses[0]);

        let score = model.score(&x, &y);
        assert!(score > 0.95, "R² score should be > 0.95, got {}", score);
    }

    #[test]
    fn test_linear_regression_predict() {
        // Use normalized features to prevent gradient explosion
        let x = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let y = vec![0.5, 1.1, 1.7]; // y = x1 + 2*x2

        let mut model = LinearRegression::new(0.5);
        model.fit(&x, &y, 1000);

        let predictions = model.predict(&x);
        for (pred, &actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 0.5,
                "Prediction {} too far from {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_ridge_regression() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] + 1.0).collect();

        let mut model = RidgeRegression::new(0.01, 0.1);
        let losses = model.fit(&x, &y, 500);

        assert!(losses.last().unwrap() < &losses[0]);

        let score = model.score(&x, &y);
        assert!(score > 0.9, "R² score should be > 0.9, got {}", score);
    }

    #[test]
    fn test_ridge_regularization_effect() {
        // With high lambda, weights should be smaller
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<f64> = x.iter().map(|xi| 5.0 * xi[0] + 1.0).collect();

        let mut model_low_reg = RidgeRegression::new(0.01, 0.001);
        let mut model_high_reg = RidgeRegression::new(0.01, 1.0);

        model_low_reg.fit(&x, &y, 500);
        model_high_reg.fit(&x, &y, 500);

        // Higher regularization should result in smaller weights
        let low_reg_weight_norm: f64 = model_low_reg.weights.iter().map(|w| w.powi(2)).sum::<f64>();
        let high_reg_weight_norm: f64 = model_high_reg
            .weights
            .iter()
            .map(|w| w.powi(2))
            .sum::<f64>();

        assert!(
            high_reg_weight_norm < low_reg_weight_norm,
            "High regularization should produce smaller weights"
        );
    }

    #[test]
    fn test_sgd_regression() {
        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] + 1.0).collect();

        let mut model = SGDRegression::new(0.01, 16);
        let losses = model.fit(&x, &y, 100);

        // SGD should also converge
        assert!(losses.last().unwrap() < &losses[0]);
    }

    #[test]
    fn test_loss_decreases() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.5 * xi + 0.5).collect();

        let mut model = SimpleLinearRegression::new(0.001);
        let losses = model.fit(&x, &y, 100);

        // Check that loss generally decreases
        let first_10_avg: f64 = losses[..10].iter().sum::<f64>() / 10.0;
        let last_10_avg: f64 = losses[losses.len() - 10..].iter().sum::<f64>() / 10.0;

        assert!(
            last_10_avg < first_10_avg,
            "Loss should decrease over training"
        );
    }
}
