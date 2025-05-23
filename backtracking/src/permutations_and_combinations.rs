pub fn generate_permutations<T: Clone>(input: &[T]) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let mut current = input.to_vec();
    backtrack_permute(0, &mut current, &mut result);
    result
}

fn backtrack_permute<T: Clone>(start: usize, arr: &mut Vec<T>, result: &mut Vec<Vec<T>>) {
    if start == arr.len() {
        result.push(arr.clone());
        return;
    }

    for i in start..arr.len() {
        arr.swap(start, i);
        backtrack_permute(start + 1, arr, result);
        arr.swap(start, i); // backtrack
    }
}

pub fn generate_combinations<T: Clone>(input: &[T], r: usize) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(r);
    backtrack_combine(0, input, r, &mut current, &mut result);
    result
}

fn backtrack_combine<T: Clone>(
    start: usize,
    input: &[T],
    r: usize,
    current: &mut Vec<T>,
    result: &mut Vec<Vec<T>>,
) {
    if current.len() == r {
        result.push(current.clone());
        return;
    }

    for i in start..input.len() {
        current.push(input[i].clone());
        backtrack_combine(i + 1, input, r, current, result);
        current.pop(); // backtrack
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutations() {
        let input = vec![1, 2, 3];
        let mut result = generate_permutations(&input);
        let mut expected = vec![
            vec![1, 2, 3],
            vec![1, 3, 2],
            vec![2, 1, 3],
            vec![2, 3, 1],
            vec![3, 2, 1],
            vec![3, 1, 2],
        ];

        result.sort();
        expected.sort();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_combinations() {
        let input = vec![1, 2, 3];
        let mut result = generate_combinations(&input, 2);
        result.sort();
        assert_eq!(result, vec![vec![1, 2], vec![1, 3], vec![2, 3]]);
    }
}
