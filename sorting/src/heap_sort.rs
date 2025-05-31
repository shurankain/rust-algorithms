pub fn heap_sort(arr: &mut [i32]) {
    let len = arr.len();

    // Build max heap
    for start in (0..len / 2).rev() {
        heapify(arr, len, start);
    }

    // Extract elements from heap one by one
    for end in (1..len).rev() {
        arr.swap(0, end); // Move max to the end
        heapify(arr, end, 0); // Restore heap for reduced array
    }
}

fn heapify(arr: &mut [i32], heap_size: usize, root_index: usize) {
    let mut largest = root_index;
    let left = 2 * root_index + 1;
    let right = 2 * root_index + 2;

    // Check if left child is larger
    if left < heap_size && arr[left] > arr[largest] {
        largest = left;
    }

    // Check if right child is larger
    if right < heap_size && arr[right] > arr[largest] {
        largest = right;
    }

    // If root is not largest, swap and continue heapifying
    if largest != root_index {
        arr.swap(root_index, largest);
        heapify(arr, heap_size, largest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_sort_basic() {
        let mut data = [4, 10, 3, 5, 1];
        heap_sort(&mut data);
        assert_eq!(data, [1, 3, 4, 5, 10]);
    }

    #[test]
    fn test_heap_sort_sorted() {
        let mut data = [1, 2, 3, 4, 5];
        heap_sort(&mut data);
        assert_eq!(data, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_heap_sort_reversed() {
        let mut data = [5, 4, 3, 2, 1];
        heap_sort(&mut data);
        assert_eq!(data, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_heap_sort_duplicates() {
        let mut data = [3, 1, 2, 3, 1];
        heap_sort(&mut data);
        assert_eq!(data, [1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_heap_sort_empty() {
        let mut data: [i32; 0] = [];
        heap_sort(&mut data);
        assert_eq!(data, []);
    }

    #[test]
    fn test_heap_sort_one_element() {
        let mut data = [42];
        heap_sort(&mut data);
        assert_eq!(data, [42]);
    }
}
