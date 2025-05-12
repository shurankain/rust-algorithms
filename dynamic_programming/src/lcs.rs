pub fn lcs(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let n = s1_chars.len();
    let m = s2_chars.len();

    let mut dp = vec![vec![0; m + 1]; n + 1];

    for i in 1..=n {
        for j in 1..=m {
            if s1_chars[i - 1] == s2_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[n][m]
}

// Same but restores the String from sequence
pub fn lcs_string(s1: &str, s2: &str) -> String {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let n = s1_chars.len();
    let m = s2_chars.len();

    let mut dp = vec![vec![0; m + 1]; n + 1];

    for i in 1..=n {
        for j in 1..=m {
            if s1_chars[i - 1] == s2_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Sequence restoration
    let mut result = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 && j > 0 {
        if s1_chars[i - 1] == s2_chars[j - 1] {
            result.push(s1_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    result.iter().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs() {
        assert_eq!(lcs("abcde", "ace"), 3);
        assert_eq!(lcs("abc", "abc"), 3);
        assert_eq!(lcs("abc", "def"), 0);
    }

    #[test]
    fn test_lcs_string() {
        assert_eq!(lcs_string("abcde", "ace"), "ace");
        assert_eq!(lcs_string("abc", "abc"), "abc");
        assert_eq!(lcs_string("abc", "def"), "");
    }
}
