pub fn quick_sort_iterative(arr: &mut [i32]) {
    if arr.len() < 2 {
        return;
    }

    let mut stack = vec![(0, arr.len() - 1)];

    while let Some((low, high)) = stack.pop() {
        if low < high {
            let p = partition(arr, low, high);

            // Push right then left (LIFO)
            if p > 0 {
                stack.push((low, p - 1)); // left side
            }
            stack.push((p + 1, high)); // right side
        }
    }
}

fn partition(arr: &mut [i32], low: usize, high: usize) -> usize {
    let pivot = arr[high];
    let mut i = low;

    for j in low..high {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, high);
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted() {
        let mut arr = [3, 6, 8, 10, 1, 2, 1];
        quick_sort_iterative(&mut arr);
        assert_eq!(arr, [1, 1, 2, 3, 6, 8, 10]);
    }

    #[test]
    fn test_empty() {
        let mut arr: [i32; 0] = [];
        quick_sort_iterative(&mut arr);
        assert_eq!(arr, []);
    }

    #[test]
    fn test_single() {
        let mut arr = [42];
        quick_sort_iterative(&mut arr);
        assert_eq!(arr, [42]);
    }

    #[test]
    fn test_duplicates() {
        let mut arr = [5, 3, 8, 3, 9, 1, 3];
        quick_sort_iterative(&mut arr);
        assert_eq!(arr, [1, 3, 3, 3, 5, 8, 9]);
    }

    #[test]
    fn test_sorted_reverse() {
        let mut arr = [9, 8, 7, 6, 5];
        quick_sort_iterative(&mut arr);
        assert_eq!(arr, [5, 6, 7, 8, 9]);
    }
}
