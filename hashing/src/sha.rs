use sha2::{Digest, Sha256};

pub fn sha(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let input = "test";

        let test_output = sha(input);

        let output = sha(input);
        assert!(!output.is_empty());
        assert_eq!(64, output.len());
        assert_eq!(test_output, output);
    }
}
