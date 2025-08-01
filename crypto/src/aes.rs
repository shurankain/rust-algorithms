use aes::Aes128;
use cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};

type Aes128CbcEnc = cbc::Encryptor<Aes128>;
type Aes128CbcDec = cbc::Decryptor<Aes128>;

pub fn encrypt_aes128_cbc(key: &[u8; 16], iv: &[u8; 16], plaintext: &[u8]) -> Vec<u8> {
    let mut buf = plaintext.to_vec();
    // Pad buffer to a multiple of block size
    let pos = buf.len();
    buf.resize(pos + 16, 0u8);

    let enc = Aes128CbcEnc::new(key.into(), iv.into());
    let ciphertext = enc.encrypt_padded_mut::<Pkcs7>(&mut buf, pos).unwrap();
    ciphertext.to_vec()
}

pub fn decrypt_aes128_cbc(key: &[u8; 16], iv: &[u8; 16], ciphertext: &[u8]) -> Vec<u8> {
    let mut buf = ciphertext.to_vec();
    let dec = Aes128CbcDec::new(key.into(), iv.into());
    let decrypted = dec.decrypt_padded_mut::<Pkcs7>(&mut buf).unwrap();
    decrypted.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_cbc_encrypt_decrypt() {
        let key = b"verysecretkey123";
        let iv = b"uniqueinitvector";
        let plaintext = b"Rust AES CBC mode test!";

        let ciphertext = encrypt_aes128_cbc(key, iv, plaintext);
        let decrypted = decrypt_aes128_cbc(key, iv, &ciphertext);

        assert_eq!(plaintext.to_vec(), decrypted);
    }
}
