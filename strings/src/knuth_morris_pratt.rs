pub fn knuth_morris_pratt(pattern: &str, text: &str) -> Option<usize> {
    let pref = prefix(pattern);

    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    let mut i: usize = 0; // text index
    let mut j: usize = 0; // pattern index
    while i < text_chars.len() {
        if text_chars[i] == pattern_chars[j] {
            i += 1;
            j += 1;

            if j == pattern_chars.len() {
                return Some(i - j);
            }
        } else if j != 0 {
            j = pref[j - 1];
        } else {
            i += 1;
        }
    }
    None
}

fn prefix(input: &str) -> Vec<usize> {
    let mut p = vec![0; input.len()];
    let input_chars: Vec<char> = input.chars().collect();

    for i in 1..p.len() {
        let mut k = p[i - 1];
        while k > 0 && input_chars[k] != input_chars[i] {
            k = p[k - 1];
        }
        if input_chars[k] == input_chars[i] {
            k += 1;
        }
        p[i] = k;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let pattern = "abc";
        let text = "asersvabcgrwgrsabcdfs";

        let result = knuth_morris_pratt(pattern, text);

        assert_eq!(6, result.unwrap());
    }
}
