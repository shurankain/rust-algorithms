// Returns all solutions to the N-Queens problem as vectors of column positions.
pub fn n_queens(n: usize) -> Vec<Vec<usize>> {
    let mut board = vec![0; n];
    let mut solutions = Vec::new();
    backtrack(0, n, &mut board, &mut solutions);
    solutions
}

fn backtrack(row: usize, n: usize, board: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {
    if row == n {
        solutions.push(board.clone());
        return;
    }

    for col in 0..n {
        if is_safe(row, col, board) {
            board[row] = col;
            backtrack(row + 1, n, board, solutions);
        }
    }
}

fn is_safe(row: usize, col: usize, board: &[usize]) -> bool {
    for (prev_row, &prev_col) in board.iter().enumerate().take(row) {
        if prev_col == col
            || prev_row + prev_col == row + col
            || prev_row as isize - prev_col as isize == row as isize - col as isize
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_queens_4() {
        let solutions = n_queens(4);
        assert_eq!(solutions.len(), 2);

        // Each solution correctness check
        for solution in solutions {
            assert!(is_valid_solution(&solution));
        }
    }

    fn is_valid_solution(solution: &[usize]) -> bool {
        let n = solution.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let same_col = solution[i] == solution[j];
                let same_diag = (i as isize - j as isize).abs()
                    == (solution[i] as isize - solution[j] as isize).abs();
                if same_col || same_diag {
                    return false;
                }
            }
        }
        true
    }
}
