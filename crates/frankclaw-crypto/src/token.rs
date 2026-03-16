use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;

/// Generate a cryptographically secure random token (256 bits, base64url encoded).
///
/// Suitable for: gateway auth tokens, webhook secrets, pairing codes.
pub fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Constant-time comparison of two token strings.
///
/// Prevents timing attacks on token validation. Both tokens are compared
/// byte-by-byte in fixed time regardless of where they differ.
pub fn verify_token_eq(provided: &str, expected: &str) -> bool {
    // Length check is not constant-time, but token length is not secret
    // (all generated tokens are the same length).
    if provided.len() != expected.len() {
        return false;
    }
    constant_time_eq(provided.as_bytes(), expected.as_bytes())
}

/// Constant-time byte slice comparison.
/// XORs all bytes and checks if the result is zero.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_are_unique() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_ne!(t1, t2);
    }

    #[test]
    fn generated_token_length() {
        let t = generate_token();
        // 32 bytes → 43 base64url chars (no padding)
        assert_eq!(t.len(), 43);
    }

    #[test]
    fn verify_same_token() {
        let t = generate_token();
        assert!(verify_token_eq(&t, &t));
    }

    #[test]
    fn verify_different_tokens() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert!(!verify_token_eq(&t1, &t2));
    }

    #[test]
    fn verify_different_lengths() {
        assert!(!verify_token_eq("short", "longer-string"));
    }

    #[test]
    fn token_has_256_bits_entropy() {
        let token = generate_token();
        let decoded = URL_SAFE_NO_PAD
            .decode(&token)
            .expect("token should be valid base64url");
        assert_eq!(decoded.len(), 32, "token should be 32 bytes (256 bits)");
    }

    #[test]
    fn tokens_have_no_obvious_patterns() {
        // Generate 100 tokens and verify byte distribution isn't degenerate.
        let tokens: Vec<Vec<u8>> = std::iter::repeat_with(|| {
            URL_SAFE_NO_PAD
                .decode(generate_token())
                .expect("generated token should decode as base64url")
        })
        .take(100)
        .collect();
        // Each byte position should have some variance.
        for pos in 0..32 {
            let values: std::collections::HashSet<u8> = tokens.iter().map(|t| t[pos]).collect();
            assert!(
                values.len() > 5,
                "byte position {pos} should have varied values, got {}",
                values.len()
            );
        }
    }

    #[test]
    fn constant_time_eq_catches_single_bit_diff() {
        let a = [0u8; 32];
        let mut b = [0u8; 32];
        b[31] = 1; // Single bit difference in last byte.
        assert!(!constant_time_eq(&a, &b));
    }
}
