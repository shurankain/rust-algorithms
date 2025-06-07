// Basic demonstration of the Diffie–Hellman key exchange algorithm
// using modular exponentiation in Rust with u128 integers.
// WARNING: This is for educational purposes only. Real cryptographic use
// should rely on vetted libraries with constant-time primitives.

fn mod_exp(base: u128, exponent: u128, modulus: u128) -> u128 {
    // Computes (base^exponent) % modulus efficiently
    let mut result = 1;
    let mut base = base % modulus;
    let mut exp = exponent;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

pub fn diffie_hellman_demo() -> (u128, u128, u128) {
    // Publicly shared parameters (should be large primes in practice)
    let p: u128 = 23; // prime modulus
    let g: u128 = 5; // primitive root modulo p

    // Private secrets (normally random and large)
    let a: u128 = 6; // Alice's private key
    let b: u128 = 15; // Bob's private key

    // Public values to be exchanged
    let a_pub = mod_exp(g, a, p); // A = g^a mod p
    let b_pub = mod_exp(g, b, p); // B = g^b mod p

    // Shared secrets computed independently
    let alice_shared = mod_exp(b_pub, a, p); // s = B^a mod p
    let bob_shared = mod_exp(a_pub, b, p); // s = A^b mod p

    // For verification: alice_shared == bob_shared
    (alice_shared, bob_shared, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_exp_correctness() {
        // 5^6 mod 23 = 15625 mod 23 = 8
        assert_eq!(mod_exp(5, 6, 23), 8);
    }

    #[test]
    fn test_diffie_hellman_shared_secret() {
        let (alice, bob, p) = diffie_hellman_demo();
        assert_eq!(alice, bob);
        assert!(alice < p);
    }
}
