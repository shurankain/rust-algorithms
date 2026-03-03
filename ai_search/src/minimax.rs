// Minimax with Alpha-Beta Pruning
//
// Classic adversarial search algorithm for two-player zero-sum games.
// Minimax assumes both players play optimally - maximizer tries to maximize score,
// minimizer tries to minimize it.
//
// Alpha-Beta pruning optimization:
// - Alpha: best score maximizer can guarantee (starts at -infinity)
// - Beta: best score minimizer can guarantee (starts at +infinity)
// - If beta <= alpha, we can prune the remaining branches (no need to explore)
//
// This can reduce time complexity from O(b^d) to O(b^(d/2)) in best case,
// where b = branching factor, d = depth.

// Trait for game states that minimax can work with
pub trait MinimaxState: Clone {
    type Move: Clone;

    fn get_moves(&self) -> Vec<Self::Move>;
    fn apply_move(&self, mv: &Self::Move) -> Self;
    fn is_terminal(&self) -> bool;
    // Heuristic evaluation from maximizer's perspective
    // Positive = good for maximizer, negative = good for minimizer
    fn evaluate(&self) -> i32;
    // True if it's maximizer's turn
    fn is_maximizing(&self) -> bool;
}

// Result of minimax search
#[derive(Debug, Clone)]
pub struct MinimaxResult<M> {
    pub best_move: Option<M>,
    pub score: i32,
    pub nodes_explored: u64,
}

pub struct Minimax {
    max_depth: usize,
}

impl Minimax {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    // Run minimax with alpha-beta pruning
    pub fn search<S: MinimaxState>(&self, state: &S) -> MinimaxResult<S::Move> {
        let mut nodes_explored = 0;
        let (score, best_move) = self.alphabeta(
            state,
            self.max_depth,
            i32::MIN,
            i32::MAX,
            state.is_maximizing(),
            &mut nodes_explored,
        );

        MinimaxResult {
            best_move,
            score,
            nodes_explored,
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn alphabeta<S: MinimaxState>(
        &self,
        state: &S,
        depth: usize,
        mut alpha: i32,
        mut beta: i32,
        maximizing: bool,
        nodes_explored: &mut u64,
    ) -> (i32, Option<S::Move>) {
        *nodes_explored += 1;

        // Terminal or depth limit reached
        if depth == 0 || state.is_terminal() {
            return (state.evaluate(), None);
        }

        let moves = state.get_moves();
        if moves.is_empty() {
            return (state.evaluate(), None);
        }

        let mut best_move = None;

        if maximizing {
            let mut max_eval = i32::MIN;

            for mv in moves {
                let new_state = state.apply_move(&mv);
                let (eval, _) =
                    self.alphabeta(&new_state, depth - 1, alpha, beta, false, nodes_explored);

                if eval > max_eval {
                    max_eval = eval;
                    best_move = Some(mv);
                }

                alpha = alpha.max(eval);
                if beta <= alpha {
                    break; // Beta cutoff
                }
            }

            (max_eval, best_move)
        } else {
            let mut min_eval = i32::MAX;

            for mv in moves {
                let new_state = state.apply_move(&mv);
                let (eval, _) =
                    self.alphabeta(&new_state, depth - 1, alpha, beta, true, nodes_explored);

                if eval < min_eval {
                    min_eval = eval;
                    best_move = Some(mv);
                }

                beta = beta.min(eval);
                if beta <= alpha {
                    break; // Alpha cutoff
                }
            }

            (min_eval, best_move)
        }
    }
}

// Standalone function for simple use cases
pub fn minimax<S: MinimaxState>(state: &S, depth: usize) -> MinimaxResult<S::Move> {
    Minimax::new(depth).search(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tic-Tac-Toe for testing (X = maximizer, O = minimizer)
    #[derive(Clone)]
    struct TicTacToe {
        board: [i32; 9], // 0 = empty, 1 = X (max), -1 = O (min)
        x_turn: bool,
    }

    impl TicTacToe {
        fn new() -> Self {
            Self {
                board: [0; 9],
                x_turn: true,
            }
        }

        fn winner(&self) -> Option<i32> {
            const LINES: [[usize; 3]; 8] = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8], // rows
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8], // cols
                [0, 4, 8],
                [2, 4, 6], // diagonals
            ];

            for [a, b, c] in LINES {
                if self.board[a] != 0
                    && self.board[a] == self.board[b]
                    && self.board[b] == self.board[c]
                {
                    return Some(self.board[a]);
                }
            }
            None
        }
    }

    impl MinimaxState for TicTacToe {
        type Move = usize;

        fn get_moves(&self) -> Vec<usize> {
            if self.winner().is_some() {
                return vec![];
            }
            (0..9).filter(|&i| self.board[i] == 0).collect()
        }

        fn apply_move(&self, &mv: &usize) -> Self {
            let mut new = self.clone();
            new.board[mv] = if self.x_turn { 1 } else { -1 };
            new.x_turn = !self.x_turn;
            new
        }

        fn is_terminal(&self) -> bool {
            self.winner().is_some() || self.board.iter().all(|&c| c != 0)
        }

        fn evaluate(&self) -> i32 {
            match self.winner() {
                Some(1) => 100,   // X wins
                Some(-1) => -100, // O wins
                _ => 0,           // draw or ongoing
            }
        }

        fn is_maximizing(&self) -> bool {
            self.x_turn
        }
    }

    #[test]
    fn test_minimax_finds_winning_move() {
        // X to play, can win at position 2
        // X | X | _
        // O | O | _
        // _ | _ | _
        let mut game = TicTacToe::new();
        game.board = [1, 1, 0, -1, -1, 0, 0, 0, 0];
        game.x_turn = true;

        let result = minimax(&game, 9);

        assert_eq!(result.best_move, Some(2));
        assert_eq!(result.score, 100); // X wins
    }

    #[test]
    fn test_minimax_blocks_opponent() {
        // X to play, must block O at position 2
        // O | O | _
        // X | _ | _
        // _ | _ | X
        let mut game = TicTacToe::new();
        game.board = [-1, -1, 0, 1, 0, 0, 0, 0, 1];
        game.x_turn = true;

        let result = minimax(&game, 9);

        assert_eq!(result.best_move, Some(2)); // Must block
    }

    #[test]
    fn test_minimax_perfect_play_draw() {
        // From empty board, perfect play leads to draw
        let game = TicTacToe::new();
        let result = minimax(&game, 9);

        assert!(result.best_move.is_some());
        assert_eq!(result.score, 0); // Draw with perfect play
    }

    #[test]
    fn test_minimax_terminal_state() {
        // X already won
        let mut game = TicTacToe::new();
        game.board = [1, 1, 1, -1, -1, 0, 0, 0, 0];
        game.x_turn = false;

        let result = minimax(&game, 9);

        assert!(result.best_move.is_none());
        assert_eq!(result.score, 100);
    }

    #[test]
    fn test_alpha_beta_pruning_efficiency() {
        // Compare nodes explored with and without effective pruning
        // More moves explored = same board should need fewer nodes with good move ordering
        let game = TicTacToe::new();
        let result = minimax(&game, 9);

        // Full tree would be much larger; alpha-beta should prune significantly
        // 9! = 362880 maximum nodes, but with pruning it's much less
        assert!(result.nodes_explored < 100000);
    }

    #[test]
    fn test_minimax_o_wins() {
        // O to play, can win at position 8
        // X | X | O
        // X | O | _
        // _ | _ | _
        let mut game = TicTacToe::new();
        game.board = [1, 1, -1, 1, -1, 0, 0, 0, 0];
        game.x_turn = false;

        let result = minimax(&game, 9);

        // O should play to win (diagonal)
        assert_eq!(result.best_move, Some(6));
        assert_eq!(result.score, -100); // O wins
    }

    #[test]
    fn test_depth_limited_search() {
        let game = TicTacToe::new();

        // Shallow search should explore fewer nodes
        let shallow = minimax(&game, 2);
        let deep = minimax(&game, 5);

        assert!(shallow.nodes_explored < deep.nodes_explored);
    }
}
