// Recursive implementation of Cooley-Tukey FFT (radix-2)
use num_complex::Complex;
use std::f64::consts::PI;

pub fn fft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    if n == 1 {
        return vec![input[0]];
    }
    assert!(n.is_power_of_two(), "Input length must be a power of 2");

    let even = fft(&input.iter().step_by(2).cloned().collect::<Vec<_>>());
    let odd = fft(&input.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>());

    let mut result = vec![Complex::new(0.0, 0.0); n];
    for k in 0..n / 2 {
        let t = Complex::from_polar(1.0, -2.0 * PI * k as f64 / n as f64) * odd[k];
        result[k] = even[k] + t;
        result[k + n / 2] = even[k] - t;
    }
    result
}

// Inverse FFT using conjugation and scaling
pub fn ifft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let conjugated: Vec<_> = input.iter().map(|x| x.conj()).collect();
    let mut transformed = fft(&conjugated);
    for x in &mut transformed {
        *x = x.conj() / n as f64;
    }
    transformed
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_fft_ifft_identity() {
        let input: Vec<Complex<f64>> = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect();

        let spectrum = fft(&input);
        let restored = ifft(&spectrum[..]);

        for (orig, back) in input.iter().zip(restored.iter()) {
            assert!((orig - back).norm() < 1e-10);
        }
    }
}
