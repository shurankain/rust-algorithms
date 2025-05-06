pub fn z_function(s: &str) -> Vec<usize> {
    let s: Vec<char> = s.chars().collect();
    let n = s.len();
    let mut z = vec![0; n];
    let mut l = 0;
    let mut r = 0;

    for i in 1..n {
        if i <= r {
            // если мы еще внутри повторяющегося отрезка
            let k = i - l;
            // если ранее посчитанный блок не выходит за правую границу r
            if z[k] < r - i + 1 {
                z[i] = z[k]; // можем переиспользовать
                continue;
            } else {
                l = i; // переходим к ручному расширению от i
            }
        } else {
            l = i; // когда мы заканчиваем проход до глубины куда сходил R
        }

        r = l; // если мы вышли из отрезка то cдвигаем r до уровня нового l
               // R начинает новый проход пока символы отрезка не перестанут совпадать
               // Или пока не достигнет конца строки
        while r < n && s[r - l] == s[r] {
            // порядок важен иначе выход за пределы массива
            r += 1; // R растет пока идут повторения
        }
        z[i] = r - l; // записываем новую длину отрезка
        r -= 1; // откатываем r потому что он на всегда на 1 впереди был
    }

    z
}

pub fn find_pattern_with_z(pattern: &str, text: &str) -> Vec<usize> {
    let combined = format!("{}${}", pattern, text); // adding search string as a prefix

    // alternative better option, avoids allocation on string recreation as before
    // let mut combined = String::with_capacity(pattern.len() + 1 + text.len());
    // combined.push_str(pattern);
    // combined.push('$');
    // combined.push_str(text);
    let z = z_function(&combined); // sending it to a search function
    let p_len = pattern.len();
    let mut result = Vec::new();

    for (i, z) in z.iter().enumerate().skip(p_len + 1) {
        if *z == p_len {
            result.push(i - p_len - 1); // позиция в оригинальном тексте
        }
    }

    result
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let input = "aabxaabx";

        let output = z_function(input);

        assert_eq!(vec![0, 1, 0, 0, 4, 1, 0, 0], output);
    }

    #[test]
    fn test_longer() {
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
