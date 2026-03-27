// Expert Choice Routing for Mixture of Experts
//
// Unlike Top-K routing where tokens select experts, Expert Choice routing
// lets each expert select its top-k tokens. This naturally balances load
// and prevents routing collapse.
//
// Key advantages:
// - Perfect load balancing by design
// - No dropped tokens
// - Better expert utilization
// - Simpler auxiliary losses
//
// References:
// - Zhou et al. "Mixture-of-Experts with Expert Choice Routing" (2022)
// - https://arxiv.org/abs/2202.09368

use std::collections::HashSet;

/// Configuration for Expert Choice routing
#[derive(Debug, Clone)]
pub struct ExpertChoiceConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Capacity factor: tokens_per_expert = capacity_factor * num_tokens / num_experts
    pub capacity_factor: f64,
    /// Whether to use soft assignments (weighted) or hard (binary)
    pub soft_assignment: bool,
    /// Temperature for score computation
    pub temperature: f64,
    /// Minimum capacity per expert
    pub min_capacity: usize,
}

impl Default for ExpertChoiceConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            capacity_factor: 1.0,
            soft_assignment: true,
            temperature: 1.0,
            min_capacity: 1,
        }
    }
}

impl ExpertChoiceConfig {
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            ..Default::default()
        }
    }

    pub fn with_capacity_factor(mut self, factor: f64) -> Self {
        self.capacity_factor = factor;
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_soft_assignment(mut self, soft: bool) -> Self {
        self.soft_assignment = soft;
        self
    }
}

/// Assignment of a token to an expert in Expert Choice routing
#[derive(Debug, Clone, PartialEq)]
pub struct TokenAssignment {
    /// Token index
    pub token_idx: usize,
    /// Expert index
    pub expert_idx: usize,
    /// Assignment weight (1.0 for hard assignment)
    pub weight: f64,
    /// Rank of this token for the expert (0 = top choice)
    pub rank: usize,
    /// Original score before normalization
    pub score: f64,
}

/// Result of Expert Choice routing
#[derive(Debug, Clone)]
pub struct ExpertChoiceResult {
    /// Assignments grouped by expert
    pub expert_assignments: Vec<Vec<TokenAssignment>>,
    /// Assignments grouped by token
    pub token_assignments: Vec<Vec<TokenAssignment>>,
    /// Number of times each token is processed (can be 0 to num_experts)
    pub token_counts: Vec<usize>,
    /// Capacity per expert
    pub expert_capacity: usize,
    /// Tokens that were not selected by any expert
    pub orphan_tokens: Vec<usize>,
    /// Score matrix after softmax [num_experts, num_tokens]
    pub score_matrix: Vec<Vec<f64>>,
    /// Average score of selected tokens
    pub avg_selection_score: f64,
}

/// Expert Choice Router
///
/// Each expert selects its top-k preferred tokens, ensuring perfect load balance.
#[derive(Debug, Clone)]
pub struct ExpertChoiceRouter {
    pub config: ExpertChoiceConfig,
}

impl ExpertChoiceRouter {
    pub fn new(config: ExpertChoiceConfig) -> Self {
        Self { config }
    }

    /// Route tokens using expert choice
    ///
    /// # Arguments
    /// * `router_logits` - Router logits [num_tokens, num_experts]
    ///
    /// # Returns
    /// Expert Choice routing result
    pub fn route(&self, router_logits: &[Vec<f64>]) -> ExpertChoiceResult {
        let num_tokens = router_logits.len();
        if num_tokens == 0 {
            return ExpertChoiceResult {
                expert_assignments: vec![vec![]; self.config.num_experts],
                token_assignments: vec![],
                token_counts: vec![],
                expert_capacity: 0,
                orphan_tokens: vec![],
                score_matrix: vec![],
                avg_selection_score: 0.0,
            };
        }

        // Compute expert capacity
        let expert_capacity = expert_capacity(
            num_tokens,
            self.config.num_experts,
            self.config.capacity_factor,
        )
        .max(self.config.min_capacity);

        // Transpose and compute scores: [num_experts, num_tokens]
        // Each row represents an expert's affinity for all tokens
        let score_matrix = compute_expert_scores(
            router_logits,
            self.config.num_experts,
            self.config.temperature,
        );

        // Each expert selects its top-k tokens
        let mut expert_assignments: Vec<Vec<TokenAssignment>> =
            vec![vec![]; self.config.num_experts];
        let mut token_assignments: Vec<Vec<TokenAssignment>> = vec![vec![]; num_tokens];
        let mut token_counts = vec![0usize; num_tokens];
        let mut total_score = 0.0;
        let mut total_selections = 0;

        for (expert_idx, expert_scores) in score_matrix.iter().enumerate() {
            // Get top-k tokens for this expert
            let mut indexed: Vec<(usize, f64)> =
                expert_scores.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, (token_idx, score)) in indexed.into_iter().take(expert_capacity).enumerate()
            {
                let weight = if self.config.soft_assignment {
                    score
                } else {
                    1.0
                };

                let assignment = TokenAssignment {
                    token_idx,
                    expert_idx,
                    weight,
                    rank,
                    score,
                };

                expert_assignments[expert_idx].push(assignment.clone());
                token_assignments[token_idx].push(assignment);
                token_counts[token_idx] += 1;

                total_score += score;
                total_selections += 1;
            }
        }

        // Find orphan tokens (not selected by any expert)
        let orphan_tokens: Vec<usize> = token_counts
            .iter()
            .enumerate()
            .filter(|&(_, count)| *count == 0)
            .map(|(idx, _)| idx)
            .collect();

        let avg_selection_score = if total_selections > 0 {
            total_score / total_selections as f64
        } else {
            0.0
        };

        ExpertChoiceResult {
            expert_assignments,
            token_assignments,
            token_counts,
            expert_capacity,
            orphan_tokens,
            score_matrix,
            avg_selection_score,
        }
    }

    /// Route with coverage guarantee (ensure all tokens are processed)
    ///
    /// If a token is not selected by any expert, assign it to its preferred expert
    pub fn route_with_coverage(&self, router_logits: &[Vec<f64>]) -> ExpertChoiceResult {
        let mut result = self.route(router_logits);

        if result.orphan_tokens.is_empty() {
            return result;
        }

        // Assign orphan tokens to their preferred experts
        for &token_idx in &result.orphan_tokens.clone() {
            let probs = &router_logits[token_idx];
            let preferred_expert = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let score = result.score_matrix[preferred_expert][token_idx];
            let weight = if self.config.soft_assignment {
                score
            } else {
                1.0
            };

            let assignment = TokenAssignment {
                token_idx,
                expert_idx: preferred_expert,
                weight,
                rank: result.expert_assignments[preferred_expert].len(),
                score,
            };

            result.expert_assignments[preferred_expert].push(assignment.clone());
            result.token_assignments[token_idx].push(assignment);
            result.token_counts[token_idx] = 1;
        }

        result.orphan_tokens.clear();
        result
    }

    /// Compute coverage statistics
    pub fn coverage_stats(&self, result: &ExpertChoiceResult) -> CoverageStats {
        let num_tokens = result.token_counts.len();
        if num_tokens == 0 {
            return CoverageStats::default();
        }

        let covered = result.token_counts.iter().filter(|&&c| c > 0).count();
        let multi_expert = result.token_counts.iter().filter(|&&c| c > 1).count();
        let avg_experts_per_token =
            result.token_counts.iter().sum::<usize>() as f64 / num_tokens as f64;
        let max_experts_per_token = *result.token_counts.iter().max().unwrap_or(&0);

        CoverageStats {
            coverage_rate: covered as f64 / num_tokens as f64,
            multi_expert_rate: multi_expert as f64 / num_tokens as f64,
            avg_experts_per_token,
            max_experts_per_token,
            num_orphans: result.orphan_tokens.len(),
        }
    }
}

/// Coverage statistics for Expert Choice routing
#[derive(Debug, Clone, Default)]
pub struct CoverageStats {
    /// Fraction of tokens processed by at least one expert
    pub coverage_rate: f64,
    /// Fraction of tokens processed by multiple experts
    pub multi_expert_rate: f64,
    /// Average number of experts per token
    pub avg_experts_per_token: f64,
    /// Maximum number of experts for any token
    pub max_experts_per_token: usize,
    /// Number of orphan tokens
    pub num_orphans: usize,
}

/// Calculate expert capacity
pub fn expert_capacity(num_tokens: usize, num_experts: usize, capacity_factor: f64) -> usize {
    ((num_tokens as f64 * capacity_factor) / num_experts as f64).ceil() as usize
}

/// Compute expert scores matrix
///
/// Transposes token scores and applies softmax per expert (row-wise softmax on transposed matrix)
pub fn compute_expert_scores(
    router_logits: &[Vec<f64>],
    num_experts: usize,
    temperature: f64,
) -> Vec<Vec<f64>> {
    let num_tokens = router_logits.len();
    if num_tokens == 0 {
        return vec![vec![]; num_experts];
    }

    // Transpose: [num_experts, num_tokens]
    let mut expert_scores: Vec<Vec<f64>> = vec![vec![0.0; num_tokens]; num_experts];

    for (token_idx, logits) in router_logits.iter().enumerate() {
        for (expert_idx, &logit) in logits.iter().enumerate() {
            if expert_idx < num_experts {
                expert_scores[expert_idx][token_idx] = logit / temperature;
            }
        }
    }

    // Apply softmax per expert (across tokens)
    for expert_row in &mut expert_scores {
        let max_val = expert_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = expert_row.iter().map(|&x| (x - max_val).exp()).sum();
        for val in expert_row.iter_mut() {
            *val = (*val - max_val).exp() / exp_sum;
        }
    }

    expert_scores
}

/// Analyze expert overlap in assignments
pub fn analyze_expert_overlap(result: &ExpertChoiceResult) -> ExpertOverlapStats {
    let num_experts = result.expert_assignments.len();
    let num_tokens = result.token_counts.len();

    if num_experts == 0 || num_tokens == 0 {
        return ExpertOverlapStats::default();
    }

    // Compute pairwise overlap between experts
    let mut overlaps = vec![vec![0usize; num_experts]; num_experts];

    for (e1, assignments1) in result.expert_assignments.iter().enumerate() {
        let tokens1: HashSet<usize> = assignments1.iter().map(|a| a.token_idx).collect();

        for (e2, assignments2) in result.expert_assignments.iter().enumerate() {
            if e2 <= e1 {
                continue;
            }

            let tokens2: HashSet<usize> = assignments2.iter().map(|a| a.token_idx).collect();
            let overlap = tokens1.intersection(&tokens2).count();
            overlaps[e1][e2] = overlap;
            overlaps[e2][e1] = overlap;
        }
    }

    // Calculate statistics
    let mut total_overlap = 0;
    let mut pairs = 0;
    let mut max_overlap = 0;

    for (i, overlap_row) in overlaps.iter().enumerate() {
        for &overlap in overlap_row.iter().skip(i + 1) {
            total_overlap += overlap;
            pairs += 1;
            max_overlap = max_overlap.max(overlap);
        }
    }

    let avg_overlap = if pairs > 0 {
        total_overlap as f64 / pairs as f64
    } else {
        0.0
    };

    ExpertOverlapStats {
        pairwise_overlaps: overlaps,
        avg_overlap,
        max_overlap,
    }
}

/// Statistics about expert token overlap
#[derive(Debug, Clone, Default)]
pub struct ExpertOverlapStats {
    /// Pairwise overlap counts [num_experts, num_experts]
    pub pairwise_overlaps: Vec<Vec<usize>>,
    /// Average overlap between any two experts
    pub avg_overlap: f64,
    /// Maximum overlap between any two experts
    pub max_overlap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_capacity() {
        // 100 tokens, 4 experts, capacity factor 1.0
        assert_eq!(expert_capacity(100, 4, 1.0), 25);

        // 100 tokens, 4 experts, capacity factor 2.0
        assert_eq!(expert_capacity(100, 4, 2.0), 50);

        // 10 tokens, 8 experts, capacity factor 1.0 -> ceil(1.25) = 2
        assert_eq!(expert_capacity(10, 8, 1.0), 2);
    }

    #[test]
    fn test_compute_expert_scores() {
        // 2 tokens, 3 experts
        let logits = vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];

        let scores = compute_expert_scores(&logits, 3, 1.0);

        // Should be [num_experts, num_tokens] = [3, 2]
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].len(), 2);

        // Each expert row should sum to 1 (softmax)
        for expert_scores in &scores {
            let sum: f64 = expert_scores.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }

        // Expert 0 prefers token 1 (higher logit 3.0 vs 1.0)
        assert!(scores[0][1] > scores[0][0]);

        // Expert 2 prefers token 0 (higher logit 3.0 vs 1.0)
        assert!(scores[2][0] > scores[2][1]);
    }

    #[test]
    fn test_basic_expert_choice_routing() {
        let config = ExpertChoiceConfig {
            num_experts: 4,
            capacity_factor: 1.0,
            soft_assignment: true,
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        // 8 tokens, 4 experts, capacity=2 each
        let logits = vec![
            vec![3.0, 1.0, 1.0, 1.0], // Token 0: expert 0 likes
            vec![3.0, 1.0, 1.0, 1.0], // Token 1: expert 0 likes
            vec![1.0, 3.0, 1.0, 1.0], // Token 2: expert 1 likes
            vec![1.0, 3.0, 1.0, 1.0], // Token 3: expert 1 likes
            vec![1.0, 1.0, 3.0, 1.0], // Token 4: expert 2 likes
            vec![1.0, 1.0, 3.0, 1.0], // Token 5: expert 2 likes
            vec![1.0, 1.0, 1.0, 3.0], // Token 6: expert 3 likes
            vec![1.0, 1.0, 1.0, 3.0], // Token 7: expert 3 likes
        ];

        let result = router.route(&logits);

        // Each expert should have exactly 2 tokens
        assert_eq!(result.expert_capacity, 2);
        for (e, assignments) in result.expert_assignments.iter().enumerate() {
            assert_eq!(
                assignments.len(),
                2,
                "Expert {} should have 2 assignments",
                e
            );
        }

        // Perfect balance should mean no orphans with this setup
        // (capacity_factor=1.0 means total selections = num_tokens)
    }

    #[test]
    fn test_soft_vs_hard_assignment() {
        let soft_config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            soft_assignment: true,
            ..Default::default()
        };

        let hard_config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            soft_assignment: false,
            ..Default::default()
        };

        let logits = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let soft_result = ExpertChoiceRouter::new(soft_config).route(&logits);
        let hard_result = ExpertChoiceRouter::new(hard_config).route(&logits);

        // Soft assignment: weights vary based on score
        let soft_weights: Vec<f64> = soft_result.expert_assignments[0]
            .iter()
            .map(|a| a.weight)
            .collect();
        assert!(soft_weights.iter().any(|&w| w != 1.0));

        // Hard assignment: all weights are 1.0
        let hard_weights: Vec<f64> = hard_result.expert_assignments[0]
            .iter()
            .map(|a| a.weight)
            .collect();
        assert!(hard_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_orphan_tokens() {
        let config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 0.5, // Low capacity
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        // 4 tokens but only capacity for 1 each (2 total)
        let logits = vec![
            vec![3.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 3.0],
            vec![1.0, 2.0],
        ];

        let result = router.route(&logits);

        // Should have 2 orphan tokens
        assert_eq!(result.orphan_tokens.len(), 2);

        // Coverage rate should be 0.5
        let stats = router.coverage_stats(&result);
        assert!((stats.coverage_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_route_with_coverage() {
        let config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 0.5,
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        let logits = vec![
            vec![3.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 3.0],
            vec![1.0, 2.0],
        ];

        let result = router.route_with_coverage(&logits);

        // No orphans after coverage guarantee
        assert!(result.orphan_tokens.is_empty());

        // All tokens should be processed
        assert!(result.token_counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_coverage_stats() {
        let config = ExpertChoiceConfig {
            num_experts: 4,
            capacity_factor: 2.0, // High capacity allows multi-expert processing
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        let logits = vec![
            vec![3.0, 2.5, 1.0, 1.0], // Token 0: liked by experts 0, 1
            vec![1.0, 1.0, 3.0, 2.5], // Token 1: liked by experts 2, 3
        ];

        let result = router.route(&logits);
        let stats = router.coverage_stats(&result);

        // Both tokens should be covered
        assert!((stats.coverage_rate - 1.0).abs() < 1e-10);

        // Average experts per token should be > 1 with high capacity
        assert!(stats.avg_experts_per_token > 1.0);
    }

    #[test]
    fn test_token_assignments() {
        let config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        let logits = vec![vec![3.0, 1.0], vec![1.0, 3.0]];

        let result = router.route(&logits);

        // Token 0 should be assigned to expert 0 (higher score)
        assert_eq!(result.token_assignments[0].len(), 1);
        assert_eq!(result.token_assignments[0][0].expert_idx, 0);

        // Token 1 should be assigned to expert 1
        assert_eq!(result.token_assignments[1].len(), 1);
        assert_eq!(result.token_assignments[1][0].expert_idx, 1);
    }

    #[test]
    fn test_empty_batch() {
        let config = ExpertChoiceConfig::new(4);
        let router = ExpertChoiceRouter::new(config);

        let result = router.route(&[]);

        assert!(result.expert_assignments.iter().all(|a| a.is_empty()));
        assert!(result.token_counts.is_empty());
        assert_eq!(result.expert_capacity, 0);
    }

    #[test]
    fn test_temperature_effect() {
        let low_temp_config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            temperature: 0.5,
            ..Default::default()
        };

        let high_temp_config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 1.0,
            temperature: 2.0,
            ..Default::default()
        };

        let logits = vec![vec![2.0, 1.0], vec![1.5, 1.5]];

        let low_temp_result = ExpertChoiceRouter::new(low_temp_config).route(&logits);
        let high_temp_result = ExpertChoiceRouter::new(high_temp_config).route(&logits);

        // Low temperature should have more peaked scores
        let low_temp_score_diff =
            (low_temp_result.score_matrix[0][0] - low_temp_result.score_matrix[0][1]).abs();
        let high_temp_score_diff =
            (high_temp_result.score_matrix[0][0] - high_temp_result.score_matrix[0][1]).abs();

        // Lower temperature -> sharper distribution -> bigger difference
        assert!(low_temp_score_diff >= high_temp_score_diff);
    }

    #[test]
    fn test_analyze_expert_overlap() {
        let config = ExpertChoiceConfig {
            num_experts: 2,
            capacity_factor: 2.0, // Allow both experts to pick same tokens
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        // Both tokens are equally preferred by both experts
        let logits = vec![vec![2.0, 2.0], vec![2.0, 2.0]];

        let result = router.route(&logits);
        let overlap_stats = analyze_expert_overlap(&result);

        // Both experts should pick both tokens (capacity = 2 each)
        // So overlap should be 2
        assert_eq!(overlap_stats.max_overlap, 2);
        assert!((overlap_stats.avg_overlap - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_assignment_rank() {
        let config = ExpertChoiceConfig {
            num_experts: 1,
            capacity_factor: 1.0,
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        // 4 tokens, single expert selects all
        let logits = vec![vec![4.0], vec![3.0], vec![2.0], vec![1.0]];

        let result = router.route(&logits);

        // Check ranks are assigned correctly
        let assignments = &result.expert_assignments[0];
        for (rank, assignment) in assignments.iter().enumerate() {
            assert_eq!(assignment.rank, rank);
        }

        // Token 0 should have rank 0 (highest score)
        assert_eq!(assignments[0].token_idx, 0);
        assert_eq!(assignments[0].rank, 0);
    }

    #[test]
    fn test_large_scale() {
        let config = ExpertChoiceConfig {
            num_experts: 8,
            capacity_factor: 1.0,
            ..Default::default()
        };
        let router = ExpertChoiceRouter::new(config);

        // 64 tokens
        let logits: Vec<Vec<f64>> = (0..64)
            .map(|i| {
                let mut l = vec![0.0; 8];
                l[i % 8] = 3.0; // Prefer expert i%8
                l
            })
            .collect();

        let result = router.route(&logits);

        // Each expert should have exactly 8 tokens (64/8)
        for (e, assignments) in result.expert_assignments.iter().enumerate() {
            assert_eq!(
                assignments.len(),
                8,
                "Expert {} should have 8 assignments",
                e
            );
        }

        // Perfect assignment means all tokens covered once
        assert!(result.orphan_tokens.is_empty());
        let stats = router.coverage_stats(&result);
        assert!((stats.coverage_rate - 1.0).abs() < 1e-10);
    }
}
