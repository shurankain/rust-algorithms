pub fn counting_sort(arr: &mut [usize], max_value: usize) {
    let mut count = vec![0; max_value + 1];

    // Calculating amount of entries
    for &num in arr.iter() {
        count[num] += 1;
    }

    // Restoring sorted array
    let mut index = 0;
    for (num, &times) in count.iter().enumerate() {
        for _ in 0..times {
            arr[index] = num;
            index += 1;
        }
    }
}

pub fn counting_sort_with_negatives(arr: &mut [i32]) {
    if arr.is_empty() {
        return;
    }

    let min = *arr.iter().min().unwrap();
    let max = *arr.iter().max().unwrap();
    let range = (max - min + 1) as usize;

    let mut count = vec![0; range];

    // Shifitng values to min to make index work
    for &num in arr.iter() {
        count[(num - min) as usize] += 1;
    }

    let mut index = 0;
    for (i, &times) in count.iter().enumerate() {
        for _ in 0..times {
            arr[index] = i as i32 + min;
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_sort_basic() {
        let mut arr = [4, 2, 2, 8, 3, 3, 1];
        counting_sort(&mut arr, 8);
        assert_eq!(arr, [1, 2, 2, 3, 3, 4, 8]);
    }

    #[test]
    fn test_with_duplicates_and_zero() {
        let mut arr = [0, 5, 2, 0, 1, 5];
        counting_sort(&mut arr, 5);
        assert_eq!(arr, [0, 0, 1, 2, 5, 5]);
    }

    #[test]
    fn test_counting_sort_with_negatives() {
        let mut arr = [-5, -10, 0, -3, 8, 5, -1, 10];
        counting_sort_with_negatives(&mut arr);
        assert_eq!(arr, [-10, -5, -3, -1, 0, 5, 8, 10]);
    }

    #[test]
    fn test_empty_array() {
        let mut arr: [usize; 0] = [];
        counting_sort(&mut arr, 0);
        assert_eq!(arr, []);
    }
}
