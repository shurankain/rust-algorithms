use sha2::{Digest, Sha256};
use sha3::Sha3_256;

pub fn sha2(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

pub fn sha3(data: &str) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha2() {
        let input = "test";

        let test_output = sha2(input);

        let output = sha2(input);
        assert!(!output.is_empty());
        assert_eq!(64, output.len());
        assert_eq!(test_output, output);
    }

    #[test]
    fn test_sha3() {
        let input = "test";

        let test_output = sha3(input);

        let output = sha3(input);
        assert!(!output.is_empty());
        assert_eq!(64, output.len());
        assert_eq!(test_output, output);
    }

    #[test]
    fn test_sha2_vs_sha3() {
        let sha2_input = "test";
        let test_sha2_output = sha2(sha2_input);

        let sha3_input = "test";
        let test_sha3_output = sha3(sha3_input);

        let sha2_output = sha2(sha2_input);
        let sha3_output = sha3(sha3_input);
        assert!(!sha2_output.is_empty());
        assert!(!sha3_output.is_empty());
        assert_eq!(64, sha2_output.len());
        assert_eq!(64, sha3_output.len());
        assert_eq!(test_sha2_output, sha2_output);
        assert_eq!(test_sha3_output, sha3_output);
        assert_ne!(sha2_output, sha3_output);
    }
}
