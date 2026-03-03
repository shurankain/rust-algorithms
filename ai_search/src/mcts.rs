// Monte Carlo Tree Search (MCTS)
//
// MCTS is a heuristic search algorithm used in decision processes, most notably in game AI.
// It builds a search tree incrementally using random simulations to evaluate positions.
//
// Four phases per iteration:
// 1. Selection: Start from root, use UCB1 to pick child nodes until reaching a leaf
// 2. Expansion: Add one or more child nodes to the tree
// 3. Simulation: Play out randomly from new node to terminal state
// 4. Backpropagation: Update win/visit counts back up the tree
//
// UCB1 formula: win_rate + C * sqrt(ln(parent_visits) / node_visits)
// - Balances exploitation (high win rate) vs exploration (less visited nodes)
// - C is exploration constant, typically sqrt(2)

use std::collections::HashMap;

// Generic trait for game states that MCTS can work with
pub trait GameState: Clone {
    type Move: Clone + Eq + std::hash::Hash;

    fn get_moves(&self) -> Vec<Self::Move>;
    fn apply_move(&self, mv: &Self::Move) -> Self;
    fn is_terminal(&self) -> bool;
    // Returns score from perspective of player who just moved
    // 1.0 = win, 0.5 = draw, 0.0 = loss
    fn evaluate(&self) -> f64;
    fn current_player(&self) -> i32;
}

struct Node<M> {
    visits: u32,
    wins: f64,
    children: HashMap<M, usize>, // move -> node index
    parent: Option<usize>,
    unexplored_moves: Vec<M>,
}

impl<M> Node<M> {
    fn new(moves: Vec<M>, parent: Option<usize>) -> Self {
        Self {
            visits: 0,
            wins: 0.0,
            children: HashMap::new(),
            parent,
            unexplored_moves: moves,
        }
    }

    fn ucb1(&self, parent_visits: u32, exploration: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.wins / self.visits as f64;
        let exploration_term =
            exploration * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration_term
    }
}

pub struct Mcts<S: GameState> {
    nodes: Vec<Node<S::Move>>,
    root_state: S,
    exploration: f64,
}

impl<S: GameState> Mcts<S> {
    pub fn new(state: S) -> Self {
        let moves = state.get_moves();
        Self {
            nodes: vec![Node::new(moves, None)],
            root_state: state,
            exploration: std::f64::consts::SQRT_2,
        }
    }

    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration = c;
        self
    }

    // Run MCTS for given number of iterations
    pub fn search(&mut self, iterations: u32) -> Option<S::Move> {
        for _ in 0..iterations {
            let (leaf_idx, state) = self.select();
            let (expanded_idx, state) = self.expand(leaf_idx, state);
            let result = self.simulate(state);
            self.backpropagate(expanded_idx, result);
        }
        self.best_move()
    }

    // Phase 1: Selection - traverse tree using UCB1 until reaching expandable node
    fn select(&self) -> (usize, S) {
        let mut node_idx = 0;
        let mut state = self.root_state.clone();

        loop {
            let node = &self.nodes[node_idx];

            // If there are unexplored moves, this is our selection target
            if !node.unexplored_moves.is_empty() {
                return (node_idx, state);
            }

            // If no children (terminal node), return
            if node.children.is_empty() {
                return (node_idx, state);
            }

            // Select child with highest UCB1
            let parent_visits = node.visits;
            let (best_move, best_child) = node
                .children
                .iter()
                .max_by(|&(_, a), &(_, b)| {
                    let ucb_a = self.nodes[*a].ucb1(parent_visits, self.exploration);
                    let ucb_b = self.nodes[*b].ucb1(parent_visits, self.exploration);
                    ucb_a.partial_cmp(&ucb_b).unwrap()
                })
                .unwrap();

            state = state.apply_move(best_move);
            node_idx = *best_child;
        }
    }

    // Phase 2: Expansion - add a new child node
    fn expand(&mut self, node_idx: usize, state: S) -> (usize, S) {
        // Check if terminal
        if state.is_terminal() {
            return (node_idx, state);
        }

        // Pop an unexplored move
        let mv = match self.nodes[node_idx].unexplored_moves.pop() {
            Some(m) => m,
            None => return (node_idx, state),
        };

        let new_state = state.apply_move(&mv);
        let new_moves = new_state.get_moves();
        let new_idx = self.nodes.len();

        self.nodes.push(Node::new(new_moves, Some(node_idx)));
        self.nodes[node_idx].children.insert(mv, new_idx);

        (new_idx, new_state)
    }

    // Phase 3: Simulation - random playout until terminal
    fn simulate(&self, mut state: S) -> f64 {
        let original_player = state.current_player();

        while !state.is_terminal() {
            let moves = state.get_moves();
            if moves.is_empty() {
                break;
            }
            // Random move selection (simple strategy)
            let mv = &moves[self.random_index(moves.len())];
            state = state.apply_move(mv);
        }

        let score = state.evaluate();
        // Flip score if needed based on who we're evaluating for
        if state.current_player() != original_player {
            1.0 - score
        } else {
            score
        }
    }

    // Phase 4: Backpropagation - update statistics up the tree
    fn backpropagate(&mut self, mut node_idx: usize, mut result: f64) {
        loop {
            let node = &mut self.nodes[node_idx];
            node.visits += 1;
            node.wins += result;

            // Flip result for parent (opponent's perspective)
            result = 1.0 - result;

            match node.parent {
                Some(parent) => node_idx = parent,
                None => break,
            }
        }
    }

    // Select move with most visits (most robust choice)
    fn best_move(&self) -> Option<S::Move> {
        let root = &self.nodes[0];
        root.children
            .iter()
            .max_by_key(|&(_, idx)| self.nodes[*idx].visits)
            .map(|(mv, _)| mv.clone())
    }

    // Simple deterministic "random" for reproducibility in tests
    fn random_index(&self, len: usize) -> usize {
        // Use node count as simple seed for determinism
        self.nodes.len() % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple Tic-Tac-Toe implementation for testing
    #[derive(Clone)]
    struct TicTacToe {
        board: [i32; 9], // 0 = empty, 1 = X, 2 = O
        current: i32,
    }

    impl TicTacToe {
        fn new() -> Self {
            Self {
                board: [0; 9],
                current: 1,
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

            for line in LINES {
                let [a, b, c] = line;
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

    impl GameState for TicTacToe {
        type Move = usize;

        fn get_moves(&self) -> Vec<usize> {
            (0..9).filter(|&i| self.board[i] == 0).collect()
        }

        fn apply_move(&self, &mv: &usize) -> Self {
            let mut new = self.clone();
            new.board[mv] = self.current;
            new.current = 3 - self.current; // toggle 1 <-> 2
            new
        }

        fn is_terminal(&self) -> bool {
            self.winner().is_some() || self.get_moves().is_empty()
        }

        fn evaluate(&self) -> f64 {
            match self.winner() {
                Some(p) if p == 3 - self.current => 1.0, // previous player won
                Some(_) => 0.0,
                None => 0.5, // draw
            }
        }

        fn current_player(&self) -> i32 {
            self.current
        }
    }

    #[test]
    fn test_mcts_finds_winning_move() {
        // X to play, can win at position 2
        // X | X | _
        // O | O | _
        // _ | _ | _
        let mut game = TicTacToe::new();
        game.board = [1, 1, 0, 2, 2, 0, 0, 0, 0];
        game.current = 1;

        let mut mcts = Mcts::new(game);
        let best = mcts.search(1000);

        assert_eq!(best, Some(2)); // position 2 wins
    }

    #[test]
    fn test_mcts_blocks_opponent() {
        // X to play, must block O at position 2
        // O | O | _
        // X | _ | _
        // _ | _ | X
        let mut game = TicTacToe::new();
        game.board = [2, 2, 0, 1, 0, 0, 0, 0, 1];
        game.current = 1;

        let mut mcts = Mcts::new(game);
        let best = mcts.search(1000);

        assert_eq!(best, Some(2)); // must block at position 2
    }

    #[test]
    fn test_mcts_empty_board() {
        let game = TicTacToe::new();
        let mut mcts = Mcts::new(game);
        let best = mcts.search(500);

        // Should return some valid move
        assert!(best.is_some());
        assert!(best.unwrap() < 9);
    }

    #[test]
    fn test_mcts_terminal_state() {
        // X already won
        let mut game = TicTacToe::new();
        game.board = [1, 1, 1, 2, 2, 0, 0, 0, 0];
        game.current = 2;

        assert!(game.is_terminal());
        assert_eq!(game.winner(), Some(1));
    }

    #[test]
    fn test_mcts_draw_detection() {
        // Full board, no winner
        let mut game = TicTacToe::new();
        game.board = [1, 2, 1, 1, 2, 2, 2, 1, 1];
        game.current = 1;

        assert!(game.is_terminal());
        assert_eq!(game.winner(), None);
        assert_eq!(game.evaluate(), 0.5);
    }
}
