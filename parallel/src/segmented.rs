// Segmented Scan
//
// Performs independent scans within segments of a single array.
// Essential for processing variable-length sequences packed into one buffer.
//
// GPU motivation:
// - GPUs work best with regular, fixed-size data structures
// - Variable-length sequences (sentences, transactions) must be packed together
// - Segmented operations let us process all sequences in parallel
//
// Segment representation:
// - Head flags: flags[i] = true marks start of new segment
// - Segment IDs: ids[i] = segment number for element i
// - Offsets: start positions of each segment
//
// Example with head flags:
//   data:  [1, 2, 3, 1, 2, 1, 2, 3, 4]
//   flags: [T, F, F, T, F, T, F, F, F]
//   segments: [1,2,3], [1,2], [1,2,3,4]
//   result:   [1,3,6], [1,3], [1,3,6,10]
//
// Parallel segmented scan key insight:
// - Combine (value, flag) pairs during scan
// - When flag is set, reset accumulator to current value
// - Operation: (v1,f1) ⊕ (v2,f2) = (if f2 then v2 else v1+v2, f1 or f2)
//
// Use cases:
// - NLP: scan within sentences in a batch
// - Finance: running totals per account
// - Graphics: operations within mesh segments
// - Database: group-by aggregations

use std::ops::Add;

// Segmented inclusive scan with head flags
// flags[i] = true indicates start of new segment
pub fn segmented_inclusive_scan<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(
        data.len(),
        flags.len(),
        "Data and flags must have same length"
    );

    let mut result = Vec::with_capacity(data.len());
    let mut acc = T::default();

    for i in 0..data.len() {
        if flags[i] {
            acc = data[i]; // Start new segment
        } else {
            acc = acc + data[i];
        }
        result.push(acc);
    }

    result
}

// Segmented exclusive scan with head flags
pub fn segmented_exclusive_scan<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(
        data.len(),
        flags.len(),
        "Data and flags must have same length"
    );

    let mut result = Vec::with_capacity(data.len());
    let mut acc = T::default();

    for i in 0..data.len() {
        if flags[i] {
            result.push(T::default()); // New segment starts with identity
            acc = data[i];
        } else {
            result.push(acc);
            acc = acc + data[i];
        }
    }

    result
}

// Parallel-friendly segmented scan using (value, flag) pairs
// This formulation allows parallel prefix computation
#[derive(Clone, Copy, Debug)]
pub struct SegmentedValue<T> {
    pub value: T,
    pub flag: bool,
}

impl<T: Copy + Add<Output = T>> SegmentedValue<T> {
    pub fn new(value: T, flag: bool) -> Self {
        Self { value, flag }
    }

    // Associative combining operation for segmented scan
    // (v1, f1) ⊕ (v2, f2) = (if f2 then v2 else v1+v2, f1 or f2)
    pub fn combine(self, other: Self) -> Self {
        if other.flag {
            // Right operand starts new segment, ignore left
            other
        } else {
            // Same segment, combine values
            Self {
                value: self.value + other.value,
                flag: self.flag,
            }
        }
    }
}

// Parallel segmented scan using the SegmentedValue formulation
pub fn parallel_segmented_scan<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), flags.len());

    let n = data.len();

    // Create (value, flag) pairs
    let mut pairs: Vec<SegmentedValue<T>> = data
        .iter()
        .zip(flags.iter())
        .map(|(&v, &f)| SegmentedValue::new(v, f))
        .collect();

    // Hillis-Steele style parallel scan with segmented combine
    let mut stride = 1;
    while stride < n {
        let mut next = pairs.clone();

        // This loop can be parallelized
        for i in stride..n {
            next[i] = pairs[i - stride].combine(pairs[i]);
        }

        pairs = next;
        stride *= 2;
    }

    pairs.iter().map(|p| p.value).collect()
}

// Segmented scan using segment IDs instead of flags
// More efficient when segments are known by ID
pub fn segmented_scan_by_id<T>(data: &[T], segment_ids: &[usize]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), segment_ids.len());

    let mut result = Vec::with_capacity(data.len());
    let mut acc = T::default();
    let mut current_segment = segment_ids[0];

    for i in 0..data.len() {
        if segment_ids[i] != current_segment {
            // New segment
            current_segment = segment_ids[i];
            acc = data[i];
        } else if i == 0 {
            acc = data[i];
        } else {
            acc = acc + data[i];
        }
        result.push(acc);
    }

    result
}

// Segmented scan using offset array (CSR-style)
// offsets[i] is the start index of segment i
// More memory efficient for many small segments
pub fn segmented_scan_by_offsets<T>(data: &[T], offsets: &[usize]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() || offsets.is_empty() {
        return vec![];
    }

    let mut result = vec![T::default(); data.len()];

    // Process each segment (can be parallel)
    for seg in 0..offsets.len() {
        let start = offsets[seg];
        let end = if seg + 1 < offsets.len() {
            offsets[seg + 1]
        } else {
            data.len()
        };

        if start >= data.len() {
            break;
        }

        let mut acc = T::default();
        for i in start..end.min(data.len()) {
            acc = acc + data[i];
            result[i] = acc;
        }
    }

    result
}

// Segmented reduction (one output per segment)
pub fn segmented_reduce<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), flags.len());

    let mut results = Vec::new();
    let mut acc = T::default();

    for i in 0..data.len() {
        if flags[i] && i > 0 {
            // Save previous segment's result
            results.push(acc);
            acc = data[i];
        } else if flags[i] {
            acc = data[i];
        } else {
            acc = acc + data[i];
        }
    }

    // Don't forget the last segment
    results.push(acc);

    results
}

// Segmented reduction by offsets
pub fn segmented_reduce_by_offsets<T>(data: &[T], offsets: &[usize]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Default,
{
    if data.is_empty() || offsets.is_empty() {
        return vec![];
    }

    let mut results = Vec::with_capacity(offsets.len());

    for seg in 0..offsets.len() {
        let start = offsets[seg];
        let end = if seg + 1 < offsets.len() {
            offsets[seg + 1]
        } else {
            data.len()
        };

        if start >= data.len() {
            break;
        }

        let mut acc = T::default();
        for item in data.iter().take(end.min(data.len())).skip(start) {
            acc = acc + *item;
        }
        results.push(acc);
    }

    results
}

// Convert between segment representations
pub fn flags_to_offsets(flags: &[bool]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| if f { Some(i) } else { None })
        .collect()
}

pub fn offsets_to_flags(offsets: &[usize], len: usize) -> Vec<bool> {
    let mut flags = vec![false; len];
    for &offset in offsets {
        if offset < len {
            flags[offset] = true;
        }
    }
    flags
}

pub fn ids_to_flags(segment_ids: &[usize]) -> Vec<bool> {
    if segment_ids.is_empty() {
        return vec![];
    }

    let mut flags = vec![false; segment_ids.len()];
    flags[0] = true;

    for i in 1..segment_ids.len() {
        if segment_ids[i] != segment_ids[i - 1] {
            flags[i] = true;
        }
    }

    flags
}

// Segmented scan with custom operation
pub fn segmented_scan_with_op<T, F>(data: &[T], flags: &[bool], identity: T, op: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), flags.len());

    let mut result = Vec::with_capacity(data.len());
    let mut acc = identity;

    for i in 0..data.len() {
        if flags[i] {
            acc = data[i];
        } else {
            acc = op(acc, data[i]);
        }
        result.push(acc);
    }

    result
}

// Segmented min scan
pub fn segmented_min_scan<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Ord,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), flags.len());

    let mut result = Vec::with_capacity(data.len());
    let mut min_val = data[0];

    for i in 0..data.len() {
        if flags[i] {
            min_val = data[i];
        } else {
            min_val = min_val.min(data[i]);
        }
        result.push(min_val);
    }

    result
}

// Segmented max scan
pub fn segmented_max_scan<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Ord,
{
    if data.is_empty() {
        return vec![];
    }

    assert_eq!(data.len(), flags.len());

    let mut result = Vec::with_capacity(data.len());
    let mut max_val = data[0];

    for i in 0..data.len() {
        if flags[i] {
            max_val = data[i];
        } else {
            max_val = max_val.max(data[i]);
        }
        result.push(max_val);
    }

    result
}

// Compute segment lengths from flags
pub fn segment_lengths(flags: &[bool]) -> Vec<usize> {
    if flags.is_empty() {
        return vec![];
    }

    let offsets = flags_to_offsets(flags);
    let mut lengths = Vec::with_capacity(offsets.len());

    for i in 0..offsets.len() {
        let start = offsets[i];
        let end = if i + 1 < offsets.len() {
            offsets[i + 1]
        } else {
            flags.len()
        };
        lengths.push(end - start);
    }

    lengths
}

// Broadcast segment values to all elements in segment
// Useful for normalizing by segment sum
pub fn segment_broadcast<T>(segment_values: &[T], flags: &[bool]) -> Vec<T>
where
    T: Copy + Default,
{
    if flags.is_empty() || segment_values.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(flags.len());
    let mut seg_idx = 0;
    let mut current_val = segment_values.first().copied().unwrap_or_default();

    for &flag in flags {
        if flag {
            current_val = segment_values.get(seg_idx).copied().unwrap_or_default();
            seg_idx += 1;
        }
        result.push(current_val);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmented_inclusive_scan() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = segmented_inclusive_scan(&data, &flags);
        assert_eq!(result, vec![1, 3, 6, 1, 3, 1, 3, 6, 10]);
    }

    #[test]
    fn test_segmented_exclusive_scan() {
        let data = vec![1, 2, 3, 1, 2];
        let flags = vec![true, false, false, true, false];

        let result = segmented_exclusive_scan(&data, &flags);
        assert_eq!(result, vec![0, 1, 3, 0, 1]);
    }

    #[test]
    fn test_segmented_scan_empty() {
        let data: Vec<i32> = vec![];
        let flags: Vec<bool> = vec![];

        assert!(segmented_inclusive_scan(&data, &flags).is_empty());
        assert!(segmented_exclusive_scan(&data, &flags).is_empty());
    }

    #[test]
    fn test_segmented_scan_single_segment() {
        let data = vec![1, 2, 3, 4, 5];
        let flags = vec![true, false, false, false, false];

        let result = segmented_inclusive_scan(&data, &flags);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_segmented_scan_all_segments() {
        let data = vec![1, 2, 3, 4, 5];
        let flags = vec![true, true, true, true, true];

        let result = segmented_inclusive_scan(&data, &flags);
        assert_eq!(result, vec![1, 2, 3, 4, 5]); // Each element is its own segment
    }

    #[test]
    fn test_parallel_segmented_scan() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = parallel_segmented_scan(&data, &flags);
        let expected = segmented_inclusive_scan(&data, &flags);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_segmented_scan_by_id() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let ids = vec![0, 0, 0, 1, 1, 2, 2, 2, 2];

        let result = segmented_scan_by_id(&data, &ids);
        assert_eq!(result, vec![1, 3, 6, 1, 3, 1, 3, 6, 10]);
    }

    #[test]
    fn test_segmented_scan_by_offsets() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let offsets = vec![0, 3, 5];

        let result = segmented_scan_by_offsets(&data, &offsets);
        assert_eq!(result, vec![1, 3, 6, 1, 3, 1, 3, 6, 10]);
    }

    #[test]
    fn test_segmented_reduce() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = segmented_reduce(&data, &flags);
        assert_eq!(result, vec![6, 3, 10]); // Sum of each segment
    }

    #[test]
    fn test_segmented_reduce_by_offsets() {
        let data = vec![1, 2, 3, 1, 2, 1, 2, 3, 4];
        let offsets = vec![0, 3, 5];

        let result = segmented_reduce_by_offsets(&data, &offsets);
        assert_eq!(result, vec![6, 3, 10]);
    }

    #[test]
    fn test_flags_to_offsets() {
        let flags = vec![true, false, false, true, false, true, false, false, false];
        let offsets = flags_to_offsets(&flags);
        assert_eq!(offsets, vec![0, 3, 5]);
    }

    #[test]
    fn test_offsets_to_flags() {
        let offsets = vec![0, 3, 5];
        let flags = offsets_to_flags(&offsets, 9);
        assert_eq!(
            flags,
            vec![true, false, false, true, false, true, false, false, false]
        );
    }

    #[test]
    fn test_ids_to_flags() {
        let ids = vec![0, 0, 0, 1, 1, 2, 2, 2, 2];
        let flags = ids_to_flags(&ids);
        assert_eq!(
            flags,
            vec![true, false, false, true, false, true, false, false, false]
        );
    }

    #[test]
    fn test_segmented_scan_with_op() {
        // Segmented product
        let data = vec![1, 2, 3, 1, 2, 3];
        let flags = vec![true, false, false, true, false, false];

        let result = segmented_scan_with_op(&data, &flags, 1, |a, b| a * b);
        assert_eq!(result, vec![1, 2, 6, 1, 2, 6]);
    }

    #[test]
    fn test_segmented_min_scan() {
        let data = vec![3, 1, 4, 5, 2, 8, 6, 7];
        let flags = vec![true, false, false, false, true, false, false, false];

        let result = segmented_min_scan(&data, &flags);
        assert_eq!(result, vec![3, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_segmented_max_scan() {
        let data = vec![3, 1, 4, 5, 2, 8, 6, 7];
        let flags = vec![true, false, false, false, true, false, false, false];

        let result = segmented_max_scan(&data, &flags);
        assert_eq!(result, vec![3, 3, 4, 5, 2, 8, 8, 8]);
    }

    #[test]
    fn test_segment_lengths() {
        let flags = vec![true, false, false, true, false, true, false, false, false];
        let lengths = segment_lengths(&flags);
        assert_eq!(lengths, vec![3, 2, 4]);
    }

    #[test]
    fn test_segment_broadcast() {
        let segment_values = vec![10, 20, 30];
        let flags = vec![true, false, false, true, false, true, false, false, false];

        let result = segment_broadcast(&segment_values, &flags);
        assert_eq!(result, vec![10, 10, 10, 20, 20, 30, 30, 30, 30]);
    }

    #[test]
    fn test_segmented_value_combine() {
        // Within same segment
        let v1 = SegmentedValue::new(5, false);
        let v2 = SegmentedValue::new(3, false);
        let combined = v1.combine(v2);
        assert_eq!(combined.value, 8);
        assert!(!combined.flag);

        // New segment starts
        let v3 = SegmentedValue::new(5, false);
        let v4 = SegmentedValue::new(7, true);
        let combined2 = v3.combine(v4);
        assert_eq!(combined2.value, 7); // Resets to v4
        assert!(combined2.flag);
    }

    #[test]
    fn test_parallel_matches_sequential_large() {
        let data: Vec<i32> = (1..=100).collect();
        let mut flags = vec![false; 100];
        flags[0] = true;
        flags[25] = true;
        flags[50] = true;
        flags[75] = true;

        let sequential = segmented_inclusive_scan(&data, &flags);
        let parallel = parallel_segmented_scan(&data, &flags);

        assert_eq!(sequential, parallel);
    }

    #[test]
    fn test_floating_point_segmented_scan() {
        let data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.1, 0.2];
        let flags = vec![true, false, false, true, false];

        let result = segmented_inclusive_scan(&data, &flags);

        assert!((result[0] - 0.1).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.6).abs() < 1e-10);
        assert!((result[3] - 0.1).abs() < 1e-10);
        assert!((result[4] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_representations_equivalent() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        // All three representations should give same result
        let flags = vec![true, false, false, true, false, true, false, false, false];
        let ids = vec![0, 0, 0, 1, 1, 2, 2, 2, 2];
        let offsets = vec![0, 3, 5];

        let result_flags = segmented_inclusive_scan(&data, &flags);
        let result_ids = segmented_scan_by_id(&data, &ids);
        let result_offsets = segmented_scan_by_offsets(&data, &offsets);

        assert_eq!(result_flags, result_ids);
        assert_eq!(result_ids, result_offsets);
    }
}
