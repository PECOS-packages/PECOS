// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HUGR loading utilities.

use anyhow::{Result, anyhow};
use std::path::Path;
use tket::hugr::Hugr;
use tket::hugr::package::Package;

/// Load a HUGR from bytes (binary envelope or JSON format).
///
/// # Errors
///
/// Returns an error if:
/// - The input is empty
/// - The HUGR format is invalid
/// - Package validation fails
pub fn load_hugr_from_bytes(bytes: &[u8]) -> Result<Hugr> {
    if bytes.is_empty() {
        return Err(anyhow!("Empty HUGR input"));
    }

    // Check if input is JSON format (starts with '{') vs binary envelope format
    let (bytes_to_load, is_json) = if bytes[0] == b'{' {
        // JSON format - wrap it in a binary envelope so HUGR can load it
        let json_str =
            std::str::from_utf8(bytes).map_err(|e| anyhow!("Invalid UTF-8 in JSON HUGR: {e}"))?;

        // Create a binary envelope with JSON content
        let mut envelope = Vec::new();

        // Magic header for HUGR envelope
        envelope.extend_from_slice(b"HUGRiHJv");

        // Format byte: 0x3F (63) for JSON format
        envelope.push(0x3F);

        // Compression byte: 0x40 (64)
        envelope.push(0x40);

        // Append the JSON content
        envelope.extend_from_slice(json_str.as_bytes());

        (envelope, true)
    } else {
        (bytes.to_vec(), false)
    };

    // Try to load as a Package first
    let mut cursor = std::io::Cursor::new(&bytes_to_load);
    match Package::load(&mut cursor, None) {
        Ok(package) => {
            if package.modules.is_empty() {
                return Err(anyhow!("Package contains no modules"));
            }

            // Validate the package
            package
                .validate()
                .map_err(|e| anyhow!("HUGR package validation failed: {e}"))?;

            // Return the first module
            Ok(package.modules[0].clone())
        }
        Err(_) if is_json => {
            // Try loading as a direct HUGR
            let mut cursor = std::io::Cursor::new(&bytes_to_load);
            Hugr::load(&mut cursor, None).map_err(|e| anyhow!("Failed to load HUGR: {e}"))
        }
        Err(e) => {
            // For binary format, try direct HUGR loading
            log::debug!("Package::load failed: {e:?}");
            let mut cursor = std::io::Cursor::new(&bytes_to_load);
            Hugr::load(&mut cursor, None).map_err(|e| anyhow!("Failed to load HUGR: {e}"))
        }
    }
}

/// Load a HUGR from a file path.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_hugr_from_file(path: impl AsRef<Path>) -> Result<Hugr> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| anyhow!("Failed to read HUGR file {}: {e}", path.as_ref().display()))?;
    load_hugr_from_bytes(&bytes)
}
