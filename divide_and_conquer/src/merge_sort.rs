pub fn merge_sort(arr: &mut [i32]) -> Vec<i32> {
    if arr.len() < 2 {
        return arr.to_vec();
    }

    let mid = arr.len() / 2;

    // divide
    let left = merge_sort(&mut arr[..mid]);
    let right = merge_sort(&mut arr[mid..]);

    let mut result: Vec<i32> = Vec::new();

    // merge
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if left[i] < right[j] {
            result.push(left[i]);
            i += 1;
        } else {
            result.push(right[j]);
            j += 1;
        }
    }
    result.extend_from_slice(&left[i..]);
    result.extend_from_slice(&right[j..]);

    result
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let mut input = [12, 1, 34, 23, 6, 52, 74, 9];

        let output = merge_sort(&mut input);
        assert_eq!(output, [1, 6, 9, 12, 23, 34, 52, 74]);
    }
}
