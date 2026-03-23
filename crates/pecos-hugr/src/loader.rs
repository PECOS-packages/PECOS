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

use anyhow::{Error, Result, anyhow};
use std::path::Path;
use tket::extension::rotation::ROTATION_EXTENSION;
use tket::extension::{TKET_EXTENSION, TKET1_EXTENSION};
use tket::hugr::Hugr;
use tket::hugr::envelope::read_envelope;
use tket::hugr::extension::{ExtensionRegistry, prelude};
use tket::hugr::std_extensions::arithmetic::{
    conversions, float_ops, float_types, int_ops, int_types,
};
use tket::hugr::std_extensions::{collections, logic, ptr};
use tket_qsystem::extension::{futures as qsystem_futures, qsystem, result as qsystem_result};

/// Extension registry matching the one used by pecos-hugr-qis and selene.
static REGISTRY: std::sync::LazyLock<ExtensionRegistry> = std::sync::LazyLock::new(|| {
    ExtensionRegistry::new([
        prelude::PRELUDE.to_owned(),
        int_types::EXTENSION.to_owned(),
        int_ops::EXTENSION.to_owned(),
        float_types::EXTENSION.to_owned(),
        float_ops::EXTENSION.to_owned(),
        conversions::EXTENSION.to_owned(),
        logic::EXTENSION.to_owned(),
        ptr::EXTENSION.to_owned(),
        collections::list::EXTENSION.to_owned(),
        collections::array::EXTENSION.to_owned(),
        collections::static_array::EXTENSION.to_owned(),
        collections::borrow_array::EXTENSION.to_owned(),
        qsystem_futures::EXTENSION.to_owned(),
        qsystem_result::EXTENSION.to_owned(),
        qsystem::EXTENSION.to_owned(),
        ROTATION_EXTENSION.to_owned(),
        TKET_EXTENSION.to_owned(),
        TKET1_EXTENSION.to_owned(),
        tket::extension::bool::BOOL_EXTENSION.to_owned(),
        tket::extension::debug::DEBUG_EXTENSION.to_owned(),
        tket_qsystem::extension::gpu::EXTENSION.to_owned(),
        tket_qsystem::extension::wasm::EXTENSION.to_owned(),
    ])
});

/// Load a HUGR from bytes (binary envelope format).
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

    let (_desc, package) = read_envelope(bytes, &REGISTRY)
        .map_err(|e| Error::new(e).context("Failed to load HUGR"))?;

    if package.modules.is_empty() {
        return Err(anyhow!("Package contains no modules"));
    }

    package
        .validate()
        .map_err(|e| Error::new(e).context("HUGR package validation failed"))?;

    Ok(package.modules[0].clone())
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
