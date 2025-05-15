pub fn matrix_chain_order(dimensions: &[usize]) -> usize {
    let n = dimensions.len() - 1; // num of matrixes
    let mut dp = vec![vec![0; n]; n];

    for length in 2..=n {
        for i in 0..=n - length {
            let j = i + length - 1;
            dp[i][j] = usize::MAX;

            for k in i..j {
                let cost =
                    dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                dp[i][j] = dp[i][j].min(cost);
            }
        }
    }

    dp[0][n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_chain_order() {
        let dims = vec![10, 100, 5, 50]; // 3 matrix: 10×100, 100×5, 5×50
        let result = matrix_chain_order(&dims);
        assert_eq!(result, 7500); // minimal cost
    }
}
