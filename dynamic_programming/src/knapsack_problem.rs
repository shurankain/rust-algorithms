// 0/1 Knapsack Problem using Dynamic Programming
pub fn knapsack(weights: &[usize], values: &[usize], capacity: usize) -> usize {
    let n = weights.len();
    let mut dp = vec![vec![0; capacity + 1]; n + 1];

    for i in 1..=n {
        for w in 0..=capacity {
            if weights[i - 1] > w {
                dp[i][w] = dp[i - 1][w];
            } else {
                dp[i][w] = dp[i - 1][w].max(dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            }
        }
    }

    dp[n][capacity]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knapsack_basic() {
        let weights = vec![3, 2, 4, 1];
        let values = vec![4, 3, 5, 2];
        let capacity = 5;

        let max_value = knapsack(&weights, &values, capacity);
        assert_eq!(max_value, 7); // 2 + 5 from items with weights 1 and 4
    }

    #[test]
    fn test_knapsack_empty() {
        let weights = vec![];
        let values = vec![];
        let capacity = 10;

        let max_value = knapsack(&weights, &values, capacity);
        assert_eq!(max_value, 0);
    }

    #[test]
    fn test_knapsack_zero_capacity() {
        let weights = vec![1, 2, 3];
        let values = vec![10, 20, 30];
        let capacity = 0;

        let max_value = knapsack(&weights, &values, capacity);
        assert_eq!(max_value, 0);
    }
}
