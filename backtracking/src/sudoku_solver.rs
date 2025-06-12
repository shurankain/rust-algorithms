// Sudoku solver using backtracking
pub struct Sudoku {
    board: [[char; 9]; 9],
}

impl Sudoku {
    pub fn new(board: [[char; 9]; 9]) -> Self {
        Self { board }
    }

    // Solve the Sudoku puzzle
    pub fn solve(&mut self) -> bool {
        self.solve_internal()
    }

    // Internal recursive solver
    fn solve_internal(&mut self) -> bool {
        for row in 0..9 {
            for col in 0..9 {
                if self.board[row][col] == '.' {
                    for c in '1'..='9' {
                        if self.is_valid(row, col, c) {
                            self.board[row][col] = c;
                            if self.solve_internal() {
                                return true;
                            }
                            self.board[row][col] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        true
    }

    // Check if placing a digit is valid
    fn is_valid(&self, row: usize, col: usize, ch: char) -> bool {
        for i in 0..9 {
            if self.board[row][i] == ch || self.board[i][col] == ch {
                return false;
            }
            let block_row = 3 * (row / 3) + i / 3;
            let block_col = 3 * (col / 3) + i % 3;
            if self.board[block_row][block_col] == ch {
                return false;
            }
        }
        true
    }

    pub fn board(&self) -> [[char; 9]; 9] {
        self.board
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sudoku_solver() {
        let board = [
            ['5', '3', '.', '.', '7', '.', '.', '.', '.'],
            ['6', '.', '.', '1', '9', '5', '.', '.', '.'],
            ['.', '9', '8', '.', '.', '.', '.', '6', '.'],
            ['8', '.', '.', '.', '6', '.', '.', '.', '3'],
            ['4', '.', '.', '8', '.', '3', '.', '.', '1'],
            ['7', '.', '.', '.', '2', '.', '.', '.', '6'],
            ['.', '6', '.', '.', '.', '.', '2', '8', '.'],
            ['.', '.', '.', '4', '1', '9', '.', '.', '5'],
            ['.', '.', '.', '.', '8', '.', '.', '7', '9'],
        ];

        let mut sudoku = Sudoku::new(board);
        assert!(sudoku.solve());

        let solved = sudoku.board();
        for line in solved {
            for c in line {
                assert!(c != '.');
            }
        }
    }
}
