// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::DemBuildError;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write;

/// SHA-256, in lowercase hex, of the version-2 schema payload.
/// The UTF-8 byte rule sorts object keys, uses compact separators and serializes
/// strings with `serde_json::to_string(&str)`. Numbers use decimal `as_u64()`,
/// else decimal `as_i64()`, else finite `as_f64()` formatted with Rust Display;
/// either signed zero is written as `0`. No finite f64 means an error.
/// `float_roundtrip` parsing makes this identical with and without
/// `serde_json/arbitrary_precision`. These bytes intentionally differ from Python v1.
///
/// # Errors
/// Rejects numbers without a finite f64 representation after integer handling.
pub fn schema_fingerprint(
    detectors: &Value,
    observables: &Value,
    runtime_measurement_order: &[usize],
    named_result_measurements: &BTreeMap<String, Vec<Option<usize>>>,
) -> Result<String, DemBuildError> {
    let payload = serde_json::json!({
        "version": 2,
        "detectors": detectors,
        "observables": observables,
        "runtime_measurement_order": runtime_measurement_order,
        "named_result_measurements": named_result_measurements,
    });
    Ok(super::certificate::hex_digest(&Sha256::digest(
        canonical_bytes(&payload)?,
    )))
}

fn canonical_bytes(value: &Value) -> Result<Vec<u8>, DemBuildError> {
    fn write(value: &Value, output: &mut String) -> Result<(), DemBuildError> {
        match value {
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::String(value) => output.push_str(
                &serde_json::to_string(value.as_str()).expect("serializing a string cannot fail"),
            ),
            Value::Number(value) => {
                if let Some(value) = value.as_u64() {
                    write!(output, "{value}").expect("writing to a String cannot fail");
                } else if let Some(value) = value.as_i64() {
                    write!(output, "{value}").expect("writing to a String cannot fail");
                } else {
                    let value = value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            DemBuildError::Configuration(
                                "schema fingerprint number has no finite f64 representation".into(),
                            )
                        })?;
                    if value == 0.0 {
                        output.push('0');
                    } else {
                        write!(output, "{value}").expect("writing to a String cannot fail");
                    }
                }
            }
            Value::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(',');
                    }
                    write(value, output)?;
                }
                output.push(']');
            }
            Value::Object(values) => {
                output.push('{');
                for (index, (key, value)) in values
                    .iter()
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .enumerate()
                {
                    if index != 0 {
                        output.push(',');
                    }
                    output.push_str(
                        &serde_json::to_string(key.as_str())
                            .expect("serializing a string cannot fail"),
                    );
                    output.push(':');
                    write(value, output)?;
                }
                output.push('}');
            }
        }
        Ok(())
    }
    let mut output = String::new();
    write(value, &mut output)?;
    Ok(output.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_two_payload_golden_digest() {
        let detectors = serde_json::json!([{"id":0,"meas_ids":[7],"reference":1}]);
        let observables = serde_json::json!([{"id":70,"meas_ids":[7],"reference":1}]);
        let bindings = BTreeMap::from([("raw".into(), vec![Some(7), None])]);
        assert_eq!(
            schema_fingerprint(&detectors, &observables, &[7, 3], &bindings).unwrap(),
            "0ecc7fa949c5196c5e4a3b3d98d6c2e241b95e1d7bdc42e3d7efe0afe8a59b7c"
        );
    }

    #[test]
    fn typed_float_parsing_is_correctly_rounded() {
        // Without float_roundtrip, serde_json's fast path yields 0x3fbeb851eb851eb8.
        let parsed: f64 = serde_json::from_str("0.12000000000000001").unwrap();
        assert_eq!(parsed.to_bits(), 0x3fbe_b851_eb85_1eb9);
    }

    #[test]
    fn canonical_number_golden_bytes() {
        for (input, expected) in [
            ("1.0", "1".into()),
            ("1e2", "100".into()),
            ("-0", "0".into()),
            ("-0.0", "0".into()),
            ("0.1", "0.1".into()),
            ("1e300", format!("1{}", "0".repeat(300))),
            ("0.12345678901234567", "0.12345678901234566".into()),
            ("18446744073709551615", "18446744073709551615".into()),
            ("18446744073709551616", "18446744073709552000".into()),
            ("-9223372036854775808", "-9223372036854775808".into()),
        ] {
            let value: Value = serde_json::from_str(input).unwrap();
            assert_eq!(
                canonical_bytes(&value).unwrap(),
                expected.as_bytes(),
                "{input}"
            );
        }
    }

    #[test]
    fn canonical_rejects_nonfinite_number() {
        let result = serde_json::from_str::<Value>("1e400")
            .map_err(|error| error.to_string())
            .and_then(|value| canonical_bytes(&value).map_err(|error| error.to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn canonical_objects_and_strings_golden_bytes() {
        let value =
            serde_json::json!({"z": [null, true, false], "é": "水😀\n\"\\\0", "a": {"b": 1.0}});
        assert_eq!(
            canonical_bytes(&value).unwrap(),
            r#"{"a":{"b":1},"z":[null,true,false],"é":"水😀\n\"\\\u0000"}"#.as_bytes()
        );
    }
}
