// Similarity and Distance Functions for Vector Search
// Foundation for ANN algorithms like HNSW, LSH, IVF

/// Euclidean distance (L2) between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Squared Euclidean distance (avoids sqrt for comparison)
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Cosine similarity between two vectors
/// Returns value in [-1, 1], where 1 is identical direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Cosine distance (1 - cosine_similarity)
/// Returns value in [0, 2], where 0 is identical
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Dot product (inner product) between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Negative dot product as distance (for normalized vectors)
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

/// Manhattan distance (L1)
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// Normalize a vector to unit length
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Distance metric types for HNSW
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    EuclideanSquared,
    Cosine,
    DotProduct,
    Manhattan,
}

impl DistanceMetric {
    /// Compute distance between two vectors
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::EuclideanSquared => euclidean_distance_squared(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
            DistanceMetric::Manhattan => manhattan_distance(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![3.0, 4.0];
        let d = vec![0.0, 0.0];
        assert!((euclidean_distance(&c, &d) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_squared() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance_squared(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let b = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];
        assert!(cosine_similarity(&b, &c).abs() < 1e-6);

        // Opposite vectors
        let d = vec![1.0, 0.0];
        let e = vec![-1.0, 0.0];
        assert!((cosine_similarity(&d, &e) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(cosine_distance(&a, &b).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        let d = vec![-1.0, 0.0];
        assert!((cosine_distance(&c, &d) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((manhattan_distance(&a, &b) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);

        // Magnitude should be 1
        let mag: f32 = n.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, v);
    }

    #[test]
    fn test_distance_metric_enum() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        assert!((DistanceMetric::Euclidean.distance(&a, &b) - 5.0).abs() < 1e-6);
        assert!((DistanceMetric::EuclideanSquared.distance(&a, &b) - 25.0).abs() < 1e-6);
        assert!((DistanceMetric::Manhattan.distance(&a, &b) - 7.0).abs() < 1e-6);
    }
}
