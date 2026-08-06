// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Versioned, self-describing shot-corpus serialization.

use pecos_decoder_core::dem::SparseDem;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const MAGIC: &[u8; 12] = b"PECOSCORPUS\0";
pub(super) const FORMAT_VERSION: u32 = 1;
const PREFIX_LEN: usize = MAGIC.len() + size_of::<u32>();

#[derive(Debug)]
pub(super) enum CorpusError {
    Io(std::io::Error),
    Invalid(String),
}

impl From<std::io::Error> for CorpusError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub(super) struct CorpusToSave<'a> {
    pub det_columns: &'a [Vec<u64>],
    pub obs_columns: &'a [Vec<u64>],
    pub num_shots: usize,
    pub seed: Option<u64>,
    pub dem: &'a str,
    pub metadata_json: Option<&'a str>,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct LoadedCorpus {
    pub det_columns: Vec<Vec<u64>>,
    pub obs_columns: Vec<Vec<u64>>,
    pub num_shots: usize,
    pub seed: Option<u64>,
    pub dem: String,
    pub metadata_json: Option<String>,
    pub format_version: u32,
}

fn invalid(message: impl Into<String>) -> CorpusError {
    CorpusError::Invalid(message.into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn checked_payload_len(
    num_detectors: usize,
    num_observables: usize,
    words_per_column: usize,
) -> Result<usize, CorpusError> {
    num_detectors
        .checked_add(num_observables)
        .and_then(|columns| columns.checked_mul(words_per_column))
        .and_then(|words| words.checked_mul(size_of::<u64>()))
        .ok_or_else(|| invalid("corpus dimensions overflow the supported payload size"))
}

fn validate_columns(columns: &[Vec<u64>], words_per_column: usize) -> Result<(), CorpusError> {
    if let Some((index, column)) = columns
        .iter()
        .enumerate()
        .find(|(_, column)| column.len() != words_per_column)
    {
        return Err(invalid(format!(
            "sample column {index} has {} word(s), expected {words_per_column}",
            column.len()
        )));
    }
    Ok(())
}

pub(super) fn save(path: &Path, corpus: CorpusToSave<'_>) -> Result<(), CorpusError> {
    let parsed_dem = SparseDem::from_dem_str(corpus.dem)
        .map_err(|error| invalid(format!("invalid DEM supplied to SampleBatch.save: {error}")))?;
    if parsed_dem.num_detectors != corpus.det_columns.len()
        || parsed_dem.num_observables != corpus.obs_columns.len()
    {
        return Err(invalid(format!(
            "DEM dimensions do not match SampleBatch: DEM has {} detector(s) and {} observable(s), batch has {} detector(s) and {} observable(s)",
            parsed_dem.num_detectors,
            parsed_dem.num_observables,
            corpus.det_columns.len(),
            corpus.obs_columns.len()
        )));
    }

    if let Some(metadata) = corpus.metadata_json {
        serde_json::from_str::<Value>(metadata)
            .map_err(|error| invalid(format!("metadata_json is not valid JSON: {error}")))?;
    }

    let words_per_column = corpus.num_shots.div_ceil(64);
    validate_columns(corpus.det_columns, words_per_column)?;
    validate_columns(corpus.obs_columns, words_per_column)?;
    let payload_len = checked_payload_len(
        corpus.det_columns.len(),
        corpus.obs_columns.len(),
        words_per_column,
    )?;
    let mut payload = Vec::with_capacity(payload_len);
    for column in corpus.det_columns.iter().chain(corpus.obs_columns) {
        for word in column {
            payload.extend_from_slice(&word.to_le_bytes());
        }
    }

    let header = serde_json::json!({
        "format_version": FORMAT_VERSION,
        "num_shots": corpus.num_shots,
        "num_detectors": corpus.det_columns.len(),
        "num_observables": corpus.obs_columns.len(),
        "words_per_column": words_per_column,
        "seed": corpus.seed,
        "dem": corpus.dem,
        "dem_sha256": sha256_hex(corpus.dem.as_bytes()),
        "payload_sha256": sha256_hex(&payload),
        "metadata_json": corpus.metadata_json,
        "generator": concat!("pecos-rslib ", env!("CARGO_PKG_VERSION")),
    });
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|error| invalid(format!("could not serialize corpus header: {error}")))?;
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| invalid("corpus JSON header is too large to encode"))?;
    let file_len = PREFIX_LEN
        .checked_add(header_bytes.len())
        .and_then(|len| len.checked_add(payload.len()))
        .ok_or_else(|| invalid("corpus file size overflows this platform"))?;
    let mut bytes = Vec::with_capacity(file_len);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    std::fs::write(path, bytes)?;
    Ok(())
}

fn required_u64(header: &Map<String, Value>, field: &str) -> Result<u64, CorpusError> {
    header.get(field).and_then(Value::as_u64).ok_or_else(|| {
        invalid(format!(
            "corpus header field {field:?} must be an unsigned integer"
        ))
    })
}

fn required_usize(header: &Map<String, Value>, field: &str) -> Result<usize, CorpusError> {
    usize::try_from(required_u64(header, field)?).map_err(|_| {
        invalid(format!(
            "corpus header field {field:?} is too large for this platform"
        ))
    })
}

fn required_string<'a>(
    header: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, CorpusError> {
    header
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("corpus header field {field:?} must be a string")))
}

fn nullable_u64(header: &Map<String, Value>, field: &str) -> Result<Option<u64>, CorpusError> {
    match header.get(field) {
        Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            invalid(format!(
                "corpus header field {field:?} must be an unsigned integer or null"
            ))
        }),
        None => Err(invalid(format!(
            "corpus header is missing required field {field:?}"
        ))),
    }
}

fn nullable_string(
    header: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, CorpusError> {
    match header.get(field) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid(format!(
            "corpus header field {field:?} must be a string or null"
        ))),
        None => Err(invalid(format!(
            "corpus header is missing required field {field:?}"
        ))),
    }
}

pub(super) fn load(path: &Path) -> Result<LoadedCorpus, CorpusError> {
    let bytes = std::fs::read(path)?;
    if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
        return Err(invalid(
            "bad shot-corpus magic: expected PECOSCORPUS followed by a NUL byte",
        ));
    }
    if bytes.len() < PREFIX_LEN {
        return Err(invalid("shot corpus is missing its 4-byte header length"));
    }
    let mut header_len_bytes = [0_u8; size_of::<u32>()];
    header_len_bytes.copy_from_slice(&bytes[MAGIC.len()..PREFIX_LEN]);
    let header_len = usize::try_from(u32::from_le_bytes(header_len_bytes))
        .map_err(|_| invalid("corpus header length is too large for this platform"))?;
    let header_end = PREFIX_LEN
        .checked_add(header_len)
        .ok_or_else(|| invalid("corpus header length overflows this platform"))?;
    if header_end > bytes.len() {
        return Err(invalid(format!(
            "corpus header length declares {header_len} byte(s), but the file contains only {} after the prefix",
            bytes.len() - PREFIX_LEN
        )));
    }
    let header_value: Value = serde_json::from_slice(&bytes[PREFIX_LEN..header_end])
        .map_err(|error| invalid(format!("invalid corpus header JSON: {error}")))?;
    let header = header_value
        .as_object()
        .ok_or_else(|| invalid("invalid corpus header JSON: top-level value must be an object"))?;

    let version = required_u64(header, "format_version")?;
    if version != u64::from(FORMAT_VERSION) {
        return Err(invalid(format!(
            "unsupported corpus format_version {version}; this PECOS build supports version {FORMAT_VERSION}"
        )));
    }

    let num_shots = required_usize(header, "num_shots")?;
    let num_detectors = required_usize(header, "num_detectors")?;
    let num_observables = required_usize(header, "num_observables")?;
    let words_per_column = required_usize(header, "words_per_column")?;
    let expected_words = num_shots.div_ceil(64);
    if words_per_column != expected_words {
        return Err(invalid(format!(
            "corpus words_per_column is {words_per_column}, but num_shots={num_shots} requires {expected_words}"
        )));
    }
    let seed = nullable_u64(header, "seed")?;
    let dem = required_string(header, "dem")?.to_owned();
    let expected_dem_sha = required_string(header, "dem_sha256")?;
    let expected_payload_sha = required_string(header, "payload_sha256")?;
    let metadata_json = nullable_string(header, "metadata_json")?;
    required_string(header, "generator")?;

    let payload = &bytes[header_end..];
    let expected_payload_len =
        checked_payload_len(num_detectors, num_observables, words_per_column)?;
    if payload.len() != expected_payload_len {
        return Err(invalid(format!(
            "corpus payload length is {} byte(s), but declared dimensions require {expected_payload_len} byte(s)",
            payload.len()
        )));
    }
    let actual_payload_sha = sha256_hex(payload);
    if expected_payload_sha != actual_payload_sha {
        return Err(invalid(format!(
            "corpus payload SHA-256 mismatch: expected {expected_payload_sha}, computed {actual_payload_sha}"
        )));
    }
    let actual_dem_sha = sha256_hex(dem.as_bytes());
    if expected_dem_sha != actual_dem_sha {
        return Err(invalid(format!(
            "corpus DEM SHA-256 mismatch: expected {expected_dem_sha}, computed {actual_dem_sha}"
        )));
    }
    if let Some(metadata) = &metadata_json {
        serde_json::from_str::<Value>(metadata)
            .map_err(|error| invalid(format!("corpus metadata_json is not valid JSON: {error}")))?;
    }
    let parsed_dem = SparseDem::from_dem_str(&dem)
        .map_err(|error| invalid(format!("corpus contains an invalid DEM: {error}")))?;
    if parsed_dem.num_detectors != num_detectors || parsed_dem.num_observables != num_observables {
        return Err(invalid(format!(
            "corpus DEM dimensions disagree with its header: DEM has {} detector(s) and {} observable(s), header declares {num_detectors} detector(s) and {num_observables} observable(s)",
            parsed_dem.num_detectors, parsed_dem.num_observables
        )));
    }

    let mut words = Vec::with_capacity(payload.len() / size_of::<u64>());
    for chunk in payload.chunks_exact(size_of::<u64>()) {
        let mut bytes = [0_u8; size_of::<u64>()];
        bytes.copy_from_slice(chunk);
        words.push(u64::from_le_bytes(bytes));
    }
    let mut offset = 0;
    let mut read_columns = |count: usize| {
        let mut columns = Vec::with_capacity(count);
        for _ in 0..count {
            let end = offset + words_per_column;
            columns.push(words[offset..end].to_vec());
            offset = end;
        }
        columns
    };
    let det_columns = read_columns(num_detectors);
    let obs_columns = read_columns(num_observables);

    Ok(LoadedCorpus {
        det_columns,
        obs_columns,
        num_shots,
        seed,
        dem,
        metadata_json,
        format_version: FORMAT_VERSION,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const DEM: &str = "error(0.125) D0 L0\n";

    fn corpus_path() -> (TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shots.pecos");
        (directory, path)
    }

    fn save_test_corpus(path: &Path) {
        save(
            path,
            CorpusToSave {
                det_columns: &[vec![0b10]],
                obs_columns: &[vec![0b10]],
                num_shots: 2,
                seed: Some(42),
                dem: DEM,
                metadata_json: Some(r#"{ "decoder": "pymatching" }"#),
            },
        )
        .unwrap();
    }

    fn invalid_message(result: Result<LoadedCorpus, CorpusError>) -> String {
        match result.unwrap_err() {
            CorpusError::Invalid(message) => message,
            CorpusError::Io(error) => panic!("unexpected I/O error: {error}"),
        }
    }

    fn header_end(bytes: &[u8]) -> usize {
        let mut length = [0_u8; 4];
        length.copy_from_slice(&bytes[MAGIC.len()..PREFIX_LEN]);
        PREFIX_LEN + usize::try_from(u32::from_le_bytes(length)).unwrap()
    }

    fn replace_header(bytes: &[u8], update: impl FnOnce(&mut Map<String, Value>)) -> Vec<u8> {
        let old_header_end = header_end(bytes);
        let mut header: Value = serde_json::from_slice(&bytes[PREFIX_LEN..old_header_end]).unwrap();
        update(header.as_object_mut().unwrap());
        let new_header = serde_json::to_vec(&header).unwrap();
        let new_header_len = u32::try_from(new_header.len()).unwrap();
        let mut updated = Vec::new();
        updated.extend_from_slice(MAGIC);
        updated.extend_from_slice(&new_header_len.to_le_bytes());
        updated.extend_from_slice(&new_header);
        updated.extend_from_slice(&bytes[old_header_end..]);
        updated
    }

    #[test]
    fn round_trip_preserves_columns_dimensions_and_provenance() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.det_columns, vec![vec![0b10]]);
        assert_eq!(loaded.obs_columns, vec![vec![0b10]]);
        assert_eq!(loaded.num_shots, 2);
        assert_eq!(loaded.seed, Some(42));
        assert_eq!(loaded.dem, DEM);
        assert_eq!(
            loaded.metadata_json.as_deref(),
            Some(r#"{ "decoder": "pymatching" }"#)
        );
        assert_eq!(loaded.format_version, FORMAT_VERSION);
    }

    #[test]
    fn wide_observable_column_round_trips_without_narrowing() {
        let (_directory, path) = corpus_path();
        let mut observables = vec![vec![0]; 65];
        observables[64][0] = 1;
        save(
            &path,
            CorpusToSave {
                det_columns: &[vec![1]],
                obs_columns: &observables,
                num_shots: 1,
                seed: None,
                dem: "error(0.125) D0 L64\n",
                metadata_json: None,
            },
        )
        .unwrap();

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.obs_columns.len(), 65);
        assert_eq!(loaded.obs_columns[64], vec![1]);
        assert_eq!(loaded.seed, None);
    }

    #[test]
    fn corrupted_payload_fails_payload_checksum() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x80;
        std::fs::write(&path, bytes).unwrap();

        let message = invalid_message(load(&path));
        assert!(message.contains("payload SHA-256 mismatch"), "{message}");
    }

    #[test]
    fn truncated_payload_fails_length_check_before_checksum() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();

        let message = invalid_message(load(&path));
        assert!(message.contains("payload length"), "{message}");
    }

    #[test]
    fn bad_magic_is_rejected_first() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&path, bytes).unwrap();

        let message = invalid_message(load(&path));
        assert!(message.contains("bad shot-corpus magic"), "{message}");
    }

    #[test]
    fn future_format_version_is_actionable() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let bytes = std::fs::read(&path).unwrap();
        let updated = replace_header(&bytes, |header| {
            header.insert("format_version".to_owned(), Value::from(999));
        });
        std::fs::write(&path, updated).unwrap();

        let message = invalid_message(load(&path));
        assert!(
            message.contains("unsupported corpus format_version 999"),
            "{message}"
        );
    }

    #[test]
    fn invalid_header_json_is_rejected_without_panicking() {
        let (_directory, path) = corpus_path();
        let mut bytes = Vec::from(MAGIC.as_slice());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.push(b'{');
        std::fs::write(&path, bytes).unwrap();

        let message = invalid_message(load(&path));
        assert!(message.contains("invalid corpus header JSON"), "{message}");
    }

    #[test]
    fn dem_checksum_is_verified_after_payload_checksum() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let bytes = std::fs::read(&path).unwrap();
        let updated = replace_header(&bytes, |header| {
            header.insert("dem".to_owned(), Value::from("error(0.25) D0 L0\n"));
        });
        std::fs::write(&path, updated).unwrap();

        let message = invalid_message(load(&path));
        assert!(message.contains("DEM SHA-256 mismatch"), "{message}");
    }

    #[test]
    fn loaded_dem_dimensions_must_match_header() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let bytes = std::fs::read(&path).unwrap();
        let replacement_dem = "error(0.25) D1 L0\n";
        let updated = replace_header(&bytes, |header| {
            header.insert("dem".to_owned(), Value::from(replacement_dem));
            header.insert(
                "dem_sha256".to_owned(),
                Value::from(sha256_hex(replacement_dem.as_bytes())),
            );
        });
        std::fs::write(&path, updated).unwrap();

        let message = invalid_message(load(&path));
        assert!(
            message.contains("corpus DEM dimensions disagree with its header"),
            "{message}"
        );
    }

    #[test]
    fn loaded_metadata_json_must_be_valid() {
        let (_directory, path) = corpus_path();
        save_test_corpus(&path);
        let bytes = std::fs::read(&path).unwrap();
        let updated = replace_header(&bytes, |header| {
            header.insert("metadata_json".to_owned(), Value::from("{"));
        });
        std::fs::write(&path, updated).unwrap();

        let message = invalid_message(load(&path));
        assert!(
            message.contains("corpus metadata_json is not valid JSON"),
            "{message}"
        );
    }

    #[test]
    fn mismatched_dem_dimensions_are_rejected_before_writing() {
        let (_directory, path) = corpus_path();
        let result = save(
            &path,
            CorpusToSave {
                det_columns: &[vec![0]],
                obs_columns: &[vec![0]],
                num_shots: 1,
                seed: None,
                dem: "error(0.125) D1 L0\n",
                metadata_json: None,
            },
        );

        let CorpusError::Invalid(message) = result.unwrap_err() else {
            panic!("expected malformed-input error");
        };
        assert!(message.contains("DEM dimensions do not match SampleBatch"));
        assert!(!path.exists());
    }

    #[test]
    fn invalid_metadata_json_is_rejected_before_writing() {
        let (_directory, path) = corpus_path();
        let result = save(
            &path,
            CorpusToSave {
                det_columns: &[vec![0]],
                obs_columns: &[vec![0]],
                num_shots: 1,
                seed: None,
                dem: DEM,
                metadata_json: Some("{"),
            },
        );

        let CorpusError::Invalid(message) = result.unwrap_err() else {
            panic!("expected malformed-input error");
        };
        assert!(message.contains("metadata_json is not valid JSON"));
        assert!(!path.exists());
    }
}
