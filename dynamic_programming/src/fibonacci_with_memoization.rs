// Fibonacci with memoization using HashMap
use std::collections::HashMap;

pub struct Fibonacci {
    memo: HashMap<u64, u64>,
}

impl Fibonacci {
    pub fn fib(&mut self, n: u64) -> u64 {
        if let Some(&value) = self.memo.get(&n) {
            return value;
        }
        let value = self.fib(n - 1) + self.fib(n - 2);
        self.memo.insert(n, value);
        value
    }
}

impl Default for Fibonacci {
    fn default() -> Self {
        let mut memo = HashMap::new();
        memo.insert(0, 0);
        memo.insert(1, 1);
        Self { memo }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib_small() {
        let mut fib = Fibonacci::default();
        assert_eq!(fib.fib(0), 0);
        assert_eq!(fib.fib(1), 1);
        assert_eq!(fib.fib(2), 1);
        assert_eq!(fib.fib(3), 2);
        assert_eq!(fib.fib(10), 55);
    }

    #[test]
    fn test_fib_larger() {
        let mut fib = Fibonacci::default();
        assert_eq!(fib.fib(50), 12586269025);
    }
}
