// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::bindings::DemBindingError;
use sha2::{Digest, Sha256};
use std::fmt::Write;

/// SHA-256 of HUGR bytes, one zero byte, then UTF-8 layout bytes, in supplied
/// order: `[["tag",slot],...]`. Brackets and commas are written explicitly with
/// no whitespace; each tag uses `serde_json::to_string`, and each `usize` slot
/// uses decimal `Display` formatting without leading zeros. Python checks
/// layout shapes and non-boolean, non-negative integer slots before conversion.
/// The result is 64 lowercase hexadecimal digits.
///
/// String escaping is independent of `arbitrary_precision`: in the installed
/// `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/serde_json-1.0.150/`,
/// `src/ser.rs:190` unconditionally calls `format_escaped_str` (`src/ser.rs:2069`);
/// the feature's number-token handling is separate (`src/ser.rs:374`). No JSON
/// numbers or maps are serialized by this encoding.
#[must_use]
pub fn generator_certificate_digest(hugr: &[u8], layout: &[(String, usize)]) -> String {
    let mut digest = Sha256::new();
    digest.update(hugr);
    digest.update([0]);
    digest.update(layout_bytes(layout));
    hex_digest(&digest.finalize())
}

/// Check a supplied certificate against the shared digest calculation.
/// # Errors
/// Rejects a certificate that does not bind the current program and layout.
pub fn check_generator_certificate_digest(
    actual: &str,
    expected: &str,
) -> Result<(), DemBindingError> {
    if actual != expected {
        return Err(DemBindingError::Value(
            "Guppy generator measurement-layout certificate does not match the program and layout"
                .into(),
        ));
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn layout_bytes(layout: &[(String, usize)]) -> Vec<u8> {
    let mut encoded = String::from("[");
    for (index, (tag, occurrence)) in layout.iter().enumerate() {
        if index != 0 {
            encoded.push(',');
        }
        let tag = serde_json::to_string(tag).expect("serializing a string cannot fail");
        write!(encoded, "[{tag},{occurrence}]").expect("writing to a String cannot fail");
    }
    encoded.push(']');
    encoded.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_pins_non_ascii_bytes() {
        let layout = vec![("é水😀\n\"\\\0".into(), 12), ("z".into(), 1)];
        let bytes = r#"[["é水😀\n\"\\\u0000",12],["z",1]]"#.as_bytes();
        assert_eq!(layout_bytes(&layout), bytes);
        assert_eq!(
            generator_certificate_digest(b"HUGR", &layout),
            "fcfc36f422a5c722e22f22edab7b8f9c4fcd4935e4571fe1016ad3d7b0f7dd87"
        );
        let mut input = b"HUGR\0".to_vec();
        input.extend_from_slice(bytes);
        assert_eq!(
            generator_certificate_digest(b"HUGR", &layout),
            hex_digest(&Sha256::digest(&input))
        );
    }

    #[test]
    fn certificate_is_program_bound_and_rejects_permutation() {
        let layout = vec![("sx1:init:meas:0".into(), 0), ("sx1:init:meas:1".into(), 0)];
        let digest = generator_certificate_digest(b"program", &layout);
        check_generator_certificate_digest(&digest, &digest).unwrap();
        assert_eq!(
            check_generator_certificate_digest(
                &generator_certificate_digest(b"foreign", &layout),
                &digest
            )
            .unwrap_err()
            .to_string(),
            "Guppy generator measurement-layout certificate does not match the program and layout"
        );
        assert_ne!(digest, generator_certificate_digest(b"foreign", &layout));
        let reversed = layout.into_iter().rev().collect::<Vec<_>>();
        assert_ne!(digest, generator_certificate_digest(b"program", &reversed));
    }
}
