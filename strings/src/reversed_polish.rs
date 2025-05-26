pub fn reversed_polish(input: &str) -> i32 {
    let mut stack = Vec::new();

    for token in input.split_whitespace() {
        match token {
            "+" => {
                let (a, b) = pop_operands(&mut stack);
                stack.push(a + b);
            }
            "-" => {
                let (a, b) = pop_operands(&mut stack);
                stack.push(a - b);
            }
            "*" => {
                let (a, b) = pop_operands(&mut stack);
                stack.push(a * b);
            }
            "/" => {
                let (a, b) = pop_operands(&mut stack);
                stack.push(a / b);
            }
            _ => {
                stack.push(token.parse::<i32>().unwrap());
            }
        }
    }
    stack.pop().unwrap()
}

fn pop_operands(stack: &mut Vec<i32>) -> (i32, i32) {
    let a = stack.pop().expect("expected wto operands, but zero found!");
    let b = stack
        .pop()
        .expect("expected wto operands, but only one found!");

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let input = "1 2 +"; // 2 + 1

        let result = reversed_polish(input);

        assert_eq!(3, result);
    }

    #[test]
    fn test_sub() {
        let input = "1 2 -"; // 2 - 1

        let result = reversed_polish(input);

        assert_eq!(1, result);
    }

    #[test]
    fn test_mod() {
        let input = "1 2 *"; // 2 * 1

        let result = reversed_polish(input);

        assert_eq!(2, result);
    }

    #[test]
    fn test_div() {
        let input = "1 2 /"; // 2 / 1

        let result = reversed_polish(input);

        assert_eq!(2, result);
    }

    #[test]
    fn test_complex() {
        let input = "1 2 + 3 * 5 - 8 /"; // 8 / (10 - (3 * (2 + 1)))

        let result = reversed_polish(input);

        assert_eq!(-2, result);
    }
}
