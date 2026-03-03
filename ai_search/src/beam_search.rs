// Beam Search
//
// A heuristic search algorithm used in sequence generation (LLM text generation, speech recognition).
// It's a breadth-first search that keeps only the top-k (beam width) most promising candidates
// at each step, trading completeness for efficiency.
//
// Key concept: Instead of exploring all possibilities (exponential) or just the best one (greedy),
// beam search maintains k parallel hypotheses, balancing exploration and efficiency.
//
// Time: O(b * k * n) where b = beam width, k = branching factor, n = sequence length
// Space: O(b * n) for storing beam candidates

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// A candidate in the beam with its score and state
#[derive(Clone)]
struct Candidate<S> {
    state: S,
    score: f64,
    sequence: Vec<usize>, // indices of choices made
}

impl<S> PartialEq for Candidate<S> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<S> Eq for Candidate<S> {}

impl<S> PartialOrd for Candidate<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S> Ord for Candidate<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = better, so reverse for max-heap behavior
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

// Trait for states that beam search can work with
pub trait BeamState: Clone {
    // Get possible next states with their scores
    // Returns: Vec<(next_state, transition_score, choice_index)>
    fn expand(&self) -> Vec<(Self, f64, usize)>;

    // Check if this is a terminal state
    fn is_terminal(&self) -> bool;
}

pub struct BeamSearch {
    beam_width: usize,
    max_steps: usize,
}

impl BeamSearch {
    pub fn new(beam_width: usize, max_steps: usize) -> Self {
        Self {
            beam_width,
            max_steps,
        }
    }

    // Run beam search and return the best sequence found
    pub fn search<S: BeamState>(&self, initial: S, initial_score: f64) -> Option<(S, Vec<usize>)> {
        let mut beam: Vec<Candidate<S>> = vec![Candidate {
            state: initial,
            score: initial_score,
            sequence: vec![],
        }];

        let mut best_terminal: Option<Candidate<S>> = None;

        for _ in 0..self.max_steps {
            let mut next_candidates: BinaryHeap<Candidate<S>> = BinaryHeap::new();

            for candidate in &beam {
                if candidate.state.is_terminal() {
                    // Track best terminal state
                    if best_terminal
                        .as_ref()
                        .is_none_or(|b| candidate.score > b.score)
                    {
                        best_terminal = Some(candidate.clone());
                    }
                    continue;
                }

                // Expand this candidate
                for (next_state, transition_score, choice_idx) in candidate.state.expand() {
                    let mut new_sequence = candidate.sequence.clone();
                    new_sequence.push(choice_idx);

                    next_candidates.push(Candidate {
                        state: next_state,
                        score: candidate.score + transition_score,
                        sequence: new_sequence,
                    });
                }
            }

            if next_candidates.is_empty() {
                break;
            }

            // Keep only top beam_width candidates
            beam = (0..self.beam_width)
                .filter_map(|_| next_candidates.pop())
                .collect();
        }

        // Return best terminal or best current candidate
        best_terminal
            .or_else(|| {
                beam.into_iter()
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            })
            .map(|c| (c.state, c.sequence))
    }
}

// Simplified beam search for token generation (common LLM use case)
// Takes a scoring function and vocabulary size
pub fn beam_search_tokens<F>(
    beam_width: usize,
    max_length: usize,
    vocab_size: usize,
    end_token: usize,
    score_fn: F,
) -> Vec<usize>
where
    F: Fn(&[usize], usize) -> f64, // (current_sequence, next_token) -> score
{
    let mut beam: Vec<(Vec<usize>, f64)> = vec![(vec![], 0.0)];

    for _ in 0..max_length {
        let mut candidates: Vec<(Vec<usize>, f64)> = Vec::new();

        for (tokens, score) in &beam {
            // Check if sequence ended
            if !tokens.is_empty() && tokens.last() == Some(&end_token) {
                candidates.push((tokens.clone(), *score));
                continue;
            }

            // Expand with all possible next tokens
            for next_token in 0..vocab_size {
                let transition_score = score_fn(tokens, next_token);
                let mut new_tokens = tokens.clone();
                new_tokens.push(next_token);
                candidates.push((new_tokens, score + transition_score));
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Sort by score (descending) and keep top beam_width
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(beam_width);
        beam = candidates;

        // Early termination if all beams ended
        if beam.iter().all(|(t, _)| t.last() == Some(&end_token)) {
            break;
        }
    }

    // Return best sequence
    beam.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(tokens, _)| tokens)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_tokens_simple() {
        // Simple scoring: prefer token 1, then 2, then 0
        let result = beam_search_tokens(
            3, // beam width
            5, // max length
            3, // vocab size (0, 1, 2)
            2, // end token
            |_seq, token| match token {
                1 => 1.0,
                2 => 0.5, // end token
                _ => 0.0,
            },
        );

        // Should generate sequence of 1s until end token becomes attractive
        assert!(!result.is_empty());
        // With this scoring, it should prefer 1s
        assert!(result.iter().filter(|&&t| t == 1).count() > 0);
    }

    #[test]
    fn test_beam_search_immediate_end() {
        // End token has highest score - use beam width 1 for greedy behavior
        let result = beam_search_tokens(
            1,
            5,
            3,
            2,
            |_seq, token| if token == 2 { 10.0 } else { 0.0 },
        );

        assert_eq!(result, vec![2]); // Should immediately end
    }

    #[test]
    fn test_beam_search_length_limit() {
        // Never prefer end token
        let result =
            beam_search_tokens(2, 3, 3, 2, |_seq, token| if token == 0 { 1.0 } else { 0.0 });

        assert_eq!(result.len(), 3); // Should hit max length
        assert!(result.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_beam_search_with_state() {
        // Test using BeamState trait with a simple path-finding example
        #[derive(Clone)]
        struct PathState {
            position: i32,
            target: i32,
        }

        impl BeamState for PathState {
            fn expand(&self) -> Vec<(Self, f64, usize)> {
                // Can move left (-1) or right (+1)
                vec![
                    (
                        PathState {
                            position: self.position - 1,
                            target: self.target,
                        },
                        -((self.position - 1 - self.target).abs() as f64), // closer = better
                        0,
                    ),
                    (
                        PathState {
                            position: self.position + 1,
                            target: self.target,
                        },
                        -((self.position + 1 - self.target).abs() as f64),
                        1,
                    ),
                ]
            }

            fn is_terminal(&self) -> bool {
                self.position == self.target
            }
        }

        let initial = PathState {
            position: 0,
            target: 3,
        };
        let beam = BeamSearch::new(2, 10);
        let result = beam.search(initial, 0.0);

        assert!(result.is_some());
        let (final_state, sequence) = result.unwrap();
        assert_eq!(final_state.position, 3);
        // Should have moved right 3 times (choice index 1)
        assert_eq!(sequence.len(), 3);
        assert!(sequence.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_beam_width_affects_result() {
        // With very narrow beam, might miss optimal path
        // With wider beam, should find better solutions
        let score_fn = |seq: &[usize], token: usize| {
            // Reward alternating pattern
            if seq.is_empty() {
                if token == 0 { 1.0 } else { 0.5 }
            } else {
                let last = seq.last().unwrap();
                if token != *last { 1.0 } else { 0.1 }
            }
        };

        let narrow = beam_search_tokens(1, 4, 2, 99, score_fn);
        let wide = beam_search_tokens(4, 4, 2, 99, score_fn);

        // Both should produce valid sequences
        assert_eq!(narrow.len(), 4);
        assert_eq!(wide.len(), 4);

        // Wide beam should find alternating pattern
        let wide_alternates = wide.windows(2).all(|w| w[0] != w[1]);
        assert!(wide_alternates);
    }
}
