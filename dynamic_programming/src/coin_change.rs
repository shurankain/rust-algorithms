// Coin Change - Classic dynamic programming problems

/// Find the minimum number of coins needed to make the given amount
/// Returns None if it's impossible to make the amount
/// Time: O(amount * coins.len()), Space: O(amount)
pub fn coin_change_min(coins: &[u32], amount: u32) -> Option<u32> {
    if amount == 0 {
        return Some(0);
    }

    let amount = amount as usize;
    // dp[i] = minimum coins needed to make amount i
    let mut dp = vec![u32::MAX; amount + 1];
    dp[0] = 0;

    for i in 1..=amount {
        for &coin in coins {
            let coin = coin as usize;
            if coin <= i && dp[i - coin] != u32::MAX {
                dp[i] = dp[i].min(dp[i - coin] + 1);
            }
        }
    }

    if dp[amount] == u32::MAX {
        None
    } else {
        Some(dp[amount])
    }
}

/// Find the minimum coins and return the actual coins used
pub fn coin_change_min_with_coins(coins: &[u32], amount: u32) -> Option<Vec<u32>> {
    if amount == 0 {
        return Some(Vec::new());
    }

    let amount = amount as usize;
    let mut dp = vec![u32::MAX; amount + 1];
    let mut parent = vec![None; amount + 1]; // Track which coin was used
    dp[0] = 0;

    for i in 1..=amount {
        for &coin in coins {
            let coin_usize = coin as usize;
            if coin_usize <= i && dp[i - coin_usize] != u32::MAX && dp[i - coin_usize] + 1 < dp[i] {
                dp[i] = dp[i - coin_usize] + 1;
                parent[i] = Some(coin);
            }
        }
    }

    if dp[amount] == u32::MAX {
        return None;
    }

    // Reconstruct the coins used
    let mut result = Vec::new();
    let mut remaining = amount;
    while let Some(coin) = parent[remaining] {
        result.push(coin);
        remaining -= coin as usize;
    }

    Some(result)
}

/// Count the number of ways to make the given amount
/// Each combination is counted once (not permutations)
/// Time: O(amount * coins.len()), Space: O(amount)
pub fn coin_change_ways(coins: &[u32], amount: u32) -> u64 {
    if amount == 0 {
        return 1;
    }

    let amount = amount as usize;
    // dp[i] = number of ways to make amount i
    let mut dp = vec![0u64; amount + 1];
    dp[0] = 1;

    // Process coins one by one to avoid counting permutations
    for &coin in coins {
        let coin = coin as usize;
        for i in coin..=amount {
            dp[i] += dp[i - coin];
        }
    }

    dp[amount]
}

/// Count permutations (order matters) to make the given amount
/// Time: O(amount * coins.len()), Space: O(amount)
pub fn coin_change_permutations(coins: &[u32], amount: u32) -> u64 {
    if amount == 0 {
        return 1;
    }

    let amount = amount as usize;
    let mut dp = vec![0u64; amount + 1];
    dp[0] = 1;

    // Process amounts first (counting all orderings)
    for i in 1..=amount {
        for &coin in coins {
            let coin = coin as usize;
            if coin <= i {
                dp[i] += dp[i - coin];
            }
        }
    }

    dp[amount]
}

/// Find all unique combinations of coins that sum to amount
pub fn coin_change_all_combinations(coins: &[u32], amount: u32) -> Vec<Vec<u32>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    backtrack_combinations(coins, amount, 0, &mut current, &mut result);
    result
}

fn backtrack_combinations(
    coins: &[u32],
    remaining: u32,
    start: usize,
    current: &mut Vec<u32>,
    result: &mut Vec<Vec<u32>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }

    for i in start..coins.len() {
        if coins[i] <= remaining {
            current.push(coins[i]);
            backtrack_combinations(coins, remaining - coins[i], i, current, result);
            current.pop();
        }
    }
}

/// Greedy coin change (not always optimal, but fast)
/// Works correctly for canonical coin systems (like US coins)
pub fn coin_change_greedy(coins: &[u32], amount: u32) -> Option<Vec<u32>> {
    let mut sorted_coins = coins.to_vec();
    sorted_coins.sort_by(|a, b| b.cmp(a)); // Sort descending

    let mut result = Vec::new();
    let mut remaining = amount;

    for &coin in &sorted_coins {
        while coin <= remaining {
            result.push(coin);
            remaining -= coin;
        }
    }

    if remaining == 0 { Some(result) } else { None }
}

/// Check if greedy gives optimal solution for a coin system
pub fn is_canonical_coin_system(coins: &[u32]) -> bool {
    if coins.is_empty() {
        return true;
    }

    let max_coin = *coins.iter().max().unwrap();

    // Check for small amounts up to 3 * max_coin
    for amount in 1..=(3 * max_coin) {
        let greedy = coin_change_greedy(coins, amount).map(|v| v.len());
        let optimal = coin_change_min(coins, amount).map(|v| v as usize);

        if greedy != optimal {
            return false;
        }
    }

    true
}

/// Unbounded knapsack variant: maximize value with unlimited items
pub fn unbounded_knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32 {
    let capacity = capacity as usize;
    let mut dp = vec![0u32; capacity + 1];

    for i in 1..=capacity {
        for (j, &weight) in weights.iter().enumerate() {
            let weight = weight as usize;
            if weight <= i {
                dp[i] = dp[i].max(dp[i - weight] + values[j]);
            }
        }
    }

    dp[capacity]
}

/// Rod cutting problem: maximize revenue from cutting a rod
pub fn rod_cutting(prices: &[u32], length: usize) -> u32 {
    if length == 0 || prices.is_empty() {
        return 0;
    }

    let mut dp = vec![0u32; length + 1];

    for i in 1..=length {
        for j in 1..=i.min(prices.len()) {
            dp[i] = dp[i].max(prices[j - 1] + dp[i - j]);
        }
    }

    dp[length]
}

/// Rod cutting with cut positions reconstruction
pub fn rod_cutting_with_cuts(prices: &[u32], length: usize) -> (u32, Vec<usize>) {
    if length == 0 || prices.is_empty() {
        return (0, Vec::new());
    }

    let mut dp = vec![0u32; length + 1];
    let mut cuts = vec![0usize; length + 1];

    for i in 1..=length {
        for j in 1..=i.min(prices.len()) {
            let revenue = prices[j - 1] + dp[i - j];
            if revenue > dp[i] {
                dp[i] = revenue;
                cuts[i] = j;
            }
        }
    }

    // Reconstruct cuts
    let mut cut_list = Vec::new();
    let mut remaining = length;
    while remaining > 0 {
        cut_list.push(cuts[remaining]);
        remaining -= cuts[remaining];
    }

    (dp[length], cut_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coin_change_min_basic() {
        assert_eq!(coin_change_min(&[1, 2, 5], 11), Some(3)); // 5+5+1
        assert_eq!(coin_change_min(&[2], 3), None);
        assert_eq!(coin_change_min(&[1], 0), Some(0));
        assert_eq!(coin_change_min(&[1], 2), Some(2));
    }

    #[test]
    fn test_coin_change_min_impossible() {
        assert_eq!(coin_change_min(&[2, 4], 7), None);
        assert_eq!(coin_change_min(&[], 5), None);
    }

    #[test]
    fn test_coin_change_min_with_coins() {
        let result = coin_change_min_with_coins(&[1, 2, 5], 11);
        assert!(result.is_some());
        let coins = result.unwrap();
        assert_eq!(coins.len(), 3);
        assert_eq!(coins.iter().sum::<u32>(), 11);
    }

    #[test]
    fn test_coin_change_ways() {
        // 5 = 5, 2+2+1, 2+1+1+1, 1+1+1+1+1
        assert_eq!(coin_change_ways(&[1, 2, 5], 5), 4);
        assert_eq!(coin_change_ways(&[2], 3), 0);
        assert_eq!(coin_change_ways(&[1, 2], 4), 3); // 1111, 112, 22
    }

    #[test]
    fn test_coin_change_ways_zero() {
        assert_eq!(coin_change_ways(&[1, 2, 5], 0), 1);
    }

    #[test]
    fn test_coin_change_permutations() {
        // Different from ways - order matters
        // 3 = 1+1+1, 1+2, 2+1
        assert_eq!(coin_change_permutations(&[1, 2], 3), 3);
        // Compare with ways: 3 = 1+1+1, 1+2 (2 ways, not 3)
        assert_eq!(coin_change_ways(&[1, 2], 3), 2);
    }

    #[test]
    fn test_coin_change_all_combinations() {
        let combos = coin_change_all_combinations(&[1, 2], 3);
        assert_eq!(combos.len(), 2);
        // Should be [1,1,1] and [1,2]
        assert!(combos.iter().all(|c| c.iter().sum::<u32>() == 3));
    }

    #[test]
    fn test_coin_change_greedy() {
        let result = coin_change_greedy(&[1, 5, 10, 25], 41);
        assert!(result.is_some());
        let coins = result.unwrap();
        assert_eq!(coins.iter().sum::<u32>(), 41);
    }

    #[test]
    fn test_coin_change_greedy_not_optimal() {
        // For coins [1, 3, 4] and amount 6:
        // Greedy: 4+1+1 = 3 coins
        // Optimal: 3+3 = 2 coins
        let greedy = coin_change_greedy(&[1, 3, 4], 6).map(|v| v.len());
        let optimal = coin_change_min(&[1, 3, 4], 6);

        assert_eq!(greedy, Some(3));
        assert_eq!(optimal, Some(2));
    }

    #[test]
    fn test_is_canonical() {
        // US coins are canonical
        assert!(is_canonical_coin_system(&[1, 5, 10, 25]));
        // This system is not canonical
        assert!(!is_canonical_coin_system(&[1, 3, 4]));
    }

    #[test]
    fn test_unbounded_knapsack() {
        // Weights: [2, 3, 4], Values: [3, 4, 5], Capacity: 7
        // Best: 2+2+3 = weight 7, value 3+3+4 = 10
        let result = unbounded_knapsack(&[2, 3, 4], &[3, 4, 5], 7);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_rod_cutting() {
        // Prices for lengths 1,2,3,4,5,6,7,8
        let prices = [1, 5, 8, 9, 10, 17, 17, 20];
        assert_eq!(rod_cutting(&prices, 8), 22); // Cut into 2+6 = 5+17 = 22
    }

    #[test]
    fn test_rod_cutting_with_cuts() {
        let prices = [1, 5, 8, 9, 10, 17, 17, 20];
        let (revenue, cuts) = rod_cutting_with_cuts(&prices, 8);
        assert_eq!(revenue, 22);
        assert_eq!(cuts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn test_rod_cutting_small() {
        // For prices [1, 5, 8, 10] with length 4:
        // length 4 whole = 10
        // length 2+2 = 5+5 = 10 (same value)
        // Algorithm may choose either
        let prices = [1, 5, 8, 10];
        let (revenue, cuts) = rod_cutting_with_cuts(&prices, 4);
        assert_eq!(revenue, 10);
        assert_eq!(cuts.iter().sum::<usize>(), 4);
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(coin_change_min(&[], 0), Some(0));
        assert_eq!(coin_change_ways(&[], 0), 1);
        assert_eq!(rod_cutting(&[], 5), 0);
    }
}
