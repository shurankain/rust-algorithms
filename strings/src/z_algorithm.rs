// Computes the Z-function for a string `s`.
// Z[i] = length of the longest prefix starting at `i` that is also a prefix of `s`.
pub fn z_function(s: &str) -> Vec<usize> {
    let s: Vec<char> = s.chars().collect();
    let n = s.len();
    let mut z = vec![0; n];
    let (mut l, mut r) = (0, 0);

    for i in 1..n {
        if i <= r {
            let k = i - l;
            if z[k] < r - i + 1 {
                z[i] = z[k]; // reuse previous Z-box
                continue;
            }
            l = i; // start manual extension
        } else {
            l = i; // reset l and r
        }

        r = l;
        while r < n && s[r - l] == s[r] {
            r += 1;
        }

        z[i] = r - l;
        r -= 1; // step back since we overshoot in last match
    }

    z
}

// Finds all starting indices of `pattern` in `text` using the Z-algorithm.
pub fn find_pattern_with_z(pattern: &str, text: &str) -> Vec<usize> {
    let combined = format!("{}${}", pattern, text);
    let z = z_function(&combined);
    let p_len = pattern.len();
    let mut result = Vec::new();

    for (i, &z_val) in z.iter().enumerate().skip(p_len + 1) {
        if z_val == p_len {
            result.push(i - p_len - 1); // shift index back to original text
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_basic() {
        let input = "aabxaabx";
        let output = z_function(input);
        assert_eq!(vec![0, 1, 0, 0, 4, 1, 0, 0], output);
    }

    #[test]
    fn test_z_longer() {
        let input = "ababcababd";
        let output = z_function(input);
        assert_eq!(vec![0, 0, 2, 0, 0, 4, 0, 2, 0, 0], output);
    }

    #[test]
    fn test_find_pattern_with_z() {
        let pattern = "abc";
        let text = "abxabcabcaby";
        let result = find_pattern_with_z(pattern, text);
        assert_eq!(vec![3, 6], result);
    }
}
