// Edit Distance (Levenshtein Distance) - Minimum operations to transform one string into another

/// Calculate the Levenshtein distance between two strings
/// Operations: insert, delete, substitute (each costs 1)
/// Time: O(m*n), Space: O(m*n)
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    // dp[i][j] = edit distance between s1[0..i] and s2[0..j]
    let mut dp = vec![vec![0; n + 1]; m + 1];

    // Base cases: transforming empty string
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i; // Delete all characters from s1
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j; // Insert all characters from s2
    }

    // Fill the DP table
    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1]; // No operation needed
            } else {
                dp[i][j] = 1 + dp[i - 1][j - 1] // Substitute
                    .min(dp[i - 1][j]) // Delete from s1
                    .min(dp[i][j - 1]); // Insert into s1
            }
        }
    }

    dp[m][n]
}

/// Space-optimized Levenshtein distance
/// Time: O(m*n), Space: O(min(m,n))
pub fn levenshtein_distance_optimized(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    // Ensure s2 is shorter for space optimization
    let (chars1, chars2) = if chars1.len() < chars2.len() {
        (chars2, chars1)
    } else {
        (chars1, chars2)
    };

    let m = chars1.len();
    let n = chars2.len();

    let mut prev = vec![0; n + 1];
    let mut curr = vec![0; n + 1];

    // Initialize first row
    for (j, val) in prev.iter_mut().enumerate().take(n + 1) {
        *val = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = 1 + prev[j - 1].min(prev[j]).min(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Weighted edit distance with custom costs
pub fn weighted_edit_distance(
    s1: &str,
    s2: &str,
    insert_cost: usize,
    delete_cost: usize,
    substitute_cost: usize,
) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i * delete_cost;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j * insert_cost;
    }

    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = (dp[i - 1][j - 1] + substitute_cost)
                    .min(dp[i - 1][j] + delete_cost)
                    .min(dp[i][j - 1] + insert_cost);
            }
        }
    }

    dp[m][n]
}

/// Damerau-Levenshtein distance (allows transposition of adjacent characters)
pub fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Need one extra row/col for transposition lookback
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

            dp[i][j] = (dp[i - 1][j] + 1) // Delete
                .min(dp[i][j - 1] + 1) // Insert
                .min(dp[i - 1][j - 1] + cost); // Substitute

            // Transposition
            if i > 1 && j > 1 && chars1[i - 1] == chars2[j - 2] && chars1[i - 2] == chars2[j - 1] {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + 1);
            }
        }
    }

    dp[m][n]
}

/// Get the edit operations to transform s1 into s2
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    Insert(usize, char),     // Position in s1, character to insert
    Delete(usize),           // Position in s1 to delete
    Substitute(usize, char), // Position in s1, new character
    Keep(usize),             // Position in s1 (no change)
}

/// Compute edit operations to transform s1 into s2
pub fn edit_operations(s1: &str, s2: &str) -> Vec<EditOp> {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    // Build DP table
    let mut dp = vec![vec![0; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find operations
    let mut ops = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && chars1[i - 1] == chars2[j - 1] {
            ops.push(EditOp::Keep(i - 1));
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            ops.push(EditOp::Substitute(i - 1, chars2[j - 1]));
            i -= 1;
            j -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            ops.push(EditOp::Insert(i, chars2[j - 1]));
            j -= 1;
        } else {
            ops.push(EditOp::Delete(i - 1));
            i -= 1;
        }
    }

    ops.reverse();
    ops
}

/// Similarity ratio between two strings (0.0 to 1.0)
pub fn similarity_ratio(s1: &str, s2: &str) -> f64 {
    let max_len = s1.chars().count().max(s2.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let distance = levenshtein_distance(s1, s2);
    1.0 - (distance as f64 / max_len as f64)
}

/// Hamming distance (only for equal-length strings)
/// Counts positions where characters differ
pub fn hamming_distance(s1: &str, s2: &str) -> Option<usize> {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    if chars1.len() != chars2.len() {
        return None;
    }

    Some(
        chars1
            .iter()
            .zip(chars2.iter())
            .filter(|(a, b)| a != b)
            .count(),
    )
}

/// Longest Common Subsequence length (related to edit distance)
pub fn lcs_length(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}

/// Jaro similarity (0.0 to 1.0)
pub fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let len1 = chars1.len();
    let len2 = chars2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];

    let mut matches = 0;
    let mut transpositions = 0;

    // Find matches
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] || chars1[i] != chars2[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if chars1[i] != chars2[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let matches = matches as f64;
    let transpositions = (transpositions / 2) as f64;

    (matches / len1 as f64 + matches / len2 as f64 + (matches - transpositions) / matches) / 3.0
}

/// Jaro-Winkler similarity (gives more weight to common prefixes)
pub fn jaro_winkler_similarity(s1: &str, s2: &str, prefix_scale: f64) -> f64 {
    let jaro = jaro_similarity(s1, s2);

    // Find common prefix (up to 4 characters)
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let prefix_len = chars1
        .iter()
        .zip(chars2.iter())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    jaro + (prefix_len as f64 * prefix_scale * (1.0 - jaro))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("test", "test"), 0);
    }

    #[test]
    fn test_levenshtein_single_operations() {
        // Single insert
        assert_eq!(levenshtein_distance("cat", "cats"), 1);
        // Single delete
        assert_eq!(levenshtein_distance("cats", "cat"), 1);
        // Single substitute
        assert_eq!(levenshtein_distance("cat", "bat"), 1);
    }

    #[test]
    fn test_levenshtein_optimized() {
        assert_eq!(levenshtein_distance_optimized("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance_optimized("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance_optimized("", "abc"), 3);
    }

    #[test]
    fn test_levenshtein_unicode() {
        assert_eq!(levenshtein_distance("café", "cafe"), 1);
        assert_eq!(levenshtein_distance("日本語", "日本人"), 1);
    }

    #[test]
    fn test_weighted_edit_distance() {
        // With equal weights (1,1,1), should equal standard Levenshtein
        assert_eq!(weighted_edit_distance("kitten", "sitting", 1, 1, 1), 3);

        // With different weights
        assert_eq!(weighted_edit_distance("ab", "ba", 1, 1, 2), 2); // Two substitutes cost 4, delete+insert costs 2
    }

    #[test]
    fn test_damerau_levenshtein() {
        // Transposition should cost 1
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein_distance("abc", "bac"), 1);

        // Regular Levenshtein would give 2 for "ab" -> "ba"
        assert_eq!(levenshtein_distance("ab", "ba"), 2);
    }

    #[test]
    fn test_edit_operations() {
        let ops = edit_operations("cat", "cats");
        assert!(ops.iter().any(|op| matches!(op, EditOp::Insert(3, 's'))));

        let ops = edit_operations("cats", "cat");
        assert!(ops.iter().any(|op| matches!(op, EditOp::Delete(3))));
    }

    #[test]
    fn test_edit_operations_count() {
        let ops = edit_operations("kitten", "sitting");
        let edit_count = ops
            .iter()
            .filter(|op| !matches!(op, EditOp::Keep(_)))
            .count();
        assert_eq!(edit_count, 3);
    }

    #[test]
    fn test_similarity_ratio() {
        assert!((similarity_ratio("hello", "hello") - 1.0).abs() < 0.001);
        assert!((similarity_ratio("", "") - 1.0).abs() < 0.001);

        let ratio = similarity_ratio("kitten", "sitting");
        assert!(ratio > 0.5 && ratio < 1.0);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance("karolin", "kathrin"), Some(3));
        assert_eq!(hamming_distance("1011101", "1001001"), Some(2));
        assert_eq!(hamming_distance("abc", "abc"), Some(0));
        assert_eq!(hamming_distance("abc", "ab"), None); // Different lengths
    }

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("ABCDGH", "AEDFHR"), 3); // ADH
        assert_eq!(lcs_length("AGGTAB", "GXTXAYB"), 4); // GTAB
    }

    #[test]
    fn test_jaro_similarity() {
        let sim = jaro_similarity("martha", "marhta");
        assert!(sim > 0.9);

        assert!((jaro_similarity("", "") - 1.0).abs() < 0.001);
        assert!(jaro_similarity("abc", "").abs() < 0.001);
    }

    #[test]
    fn test_jaro_winkler() {
        let jw = jaro_winkler_similarity("martha", "marhta", 0.1);
        let jaro = jaro_similarity("martha", "marhta");

        // Jaro-Winkler should be >= Jaro for strings with common prefix
        assert!(jw >= jaro);
    }

    #[test]
    fn test_jaro_winkler_common_prefix() {
        // Strings with longer common prefix should have higher Jaro-Winkler
        let jw1 = jaro_winkler_similarity("prefix_abc", "prefix_xyz", 0.1);
        let jw2 = jaro_winkler_similarity("abc_suffix", "xyz_suffix", 0.1);

        assert!(jw1 > jw2);
    }

    #[test]
    fn test_empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance_optimized("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(lcs_length("", ""), 0);
    }
}
