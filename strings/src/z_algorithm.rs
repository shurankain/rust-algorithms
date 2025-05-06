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
}
