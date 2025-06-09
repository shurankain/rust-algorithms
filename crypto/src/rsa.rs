use num_bigint::{BigInt, BigUint, ToBigInt, ToBigUint};
use num_traits::{One, Signed, Zero};
use rand::Rng;

// Generates a pair of RSA keys (public and private).
pub fn generate_rsa_keys(bits: usize) -> (RSAPublicKey, RSAPrivateKey) {
    let mut rng = rand::rng();

    // Generate two distinct large primes p and q
    let p = rand_prime(bits / 2, &mut rng);
    let mut q = rand_prime(bits / 2, &mut rng);
    while p == q {
        q = rand_prime(bits / 2, &mut rng);
    }

    let n = &p * &q;
    let phi = (&p - BigUint::one()) * (&q - BigUint::one());

    // Common public exponent
    let e = 65537u32.to_biguint().unwrap();

    // Compute d, the modular inverse of e mod phi
    let d = modinv(&e, &phi).expect("No modular inverse for e");

    (
        RSAPublicKey {
            n: n.clone(),
            e: e.clone(),
        },
        RSAPrivateKey { n, d },
    )
}

// Encrypts a message using the RSA public key.
pub fn rsa_encrypt(pubkey: &RSAPublicKey, plaintext: &[u8]) -> Vec<u8> {
    let m = BigUint::from_bytes_be(plaintext);
    assert!(m < pubkey.n, "Message too large for the modulus");
    let c = m.modpow(&pubkey.e, &pubkey.n);
    c.to_bytes_be()
}

/// Decrypts a message using the RSA private key.
pub fn rsa_decrypt(privkey: &RSAPrivateKey, ciphertext: &[u8]) -> Vec<u8> {
    let c = BigUint::from_bytes_be(ciphertext);
    let m = c.modpow(&privkey.d, &privkey.n);
    m.to_bytes_be()
}

#[derive(Debug, Clone)]
pub struct RSAPublicKey {
    pub n: BigUint,
    pub e: BigUint,
}

#[derive(Debug, Clone)]
pub struct RSAPrivateKey {
    pub n: BigUint,
    pub d: BigUint,
}

// Generate a random prime number of specified bit size
fn rand_prime(bits: usize, rng: &mut impl rand::RngCore) -> BigUint {
    use primal::is_prime;

    loop {
        // Generate a random odd BigUint of the given bit size
        let mut bytes = vec![0u8; bits.div_ceil(8)];
        rng.fill(&mut bytes[..]);
        // Ensure the highest bit is set (to guarantee bit size)
        if !bytes.is_empty() {
            bytes[0] |= 0b1000_0000;
        }
        // Ensure the number is odd
        if let Some(last) = bytes.last_mut() {
            *last |= 1;
        }
        let candidate = BigUint::from_bytes_be(&bytes);
        // primal::is_prime works for u64, so check only if candidate fits
        if let Some(candidate_u64) = candidate.to_u64_digits().first().copied() {
            if is_prime(candidate_u64) {
                return candidate_u64.to_biguint().unwrap();
            }
        }
    }
}

// Modular inverse using extended Euclidean algorithm (works for BigUint)
fn modinv(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    let zero = BigInt::zero();
    let one = BigInt::one();

    let (mut t, mut new_t) = (zero.clone(), one.clone());
    let (mut r, mut new_r) = (m.to_bigint().unwrap(), a.to_bigint().unwrap());

    while new_r != zero {
        let quotient = &r / &new_r;
        t -= &quotient * &new_t;
        r -= &quotient * &new_r;

        // swap
        std::mem::swap(&mut t, &mut new_t);
        std::mem::swap(&mut r, &mut new_r);
    }

    if r > one {
        return None; // not invertible
    }
    if t.is_negative() {
        t += m.to_bigint().unwrap();
    }
    t.to_biguint()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsa_encrypt_decrypt() {
        let (pubkey, privkey) = generate_rsa_keys(32); // Small for test, use >=2048 in real
        let message = b"hi";
        let ciphertext = rsa_encrypt(&pubkey, message);
        let decrypted = rsa_decrypt(&privkey, &ciphertext);
        assert_eq!(message.to_vec(), decrypted);
    }
}
