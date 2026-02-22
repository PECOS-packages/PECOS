//! Utilities for HUGR processing and validation

use anyhow::{Error, Result, anyhow};
use tket::extension::rotation::ROTATION_EXTENSION;
use tket::extension::{TKET_EXTENSION, TKET1_EXTENSION, TKET1_EXTENSION_ID, TKET1_OP_NAME};
use tket::hugr::envelope::read_envelope;
use tket::hugr::extension::{ExtensionRegistry, prelude};
use tket::hugr::ops::OpType;
use tket::hugr::std_extensions::arithmetic::{
    conversions, float_ops, float_types, int_ops, int_types,
};
use tket::hugr::std_extensions::{collections, logic, ptr};
use tket::hugr::types::Term;
use tket::hugr::{Hugr, HugrView};
use tket_qsystem::extension::{futures as qsystem_futures, qsystem, result as qsystem_result};

/// Extension registry matching the one used by selene-hugr-qis-compiler.
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

/// Loads a HUGR package from a binary [Envelope][tket::hugr::envelope::Envelope].
///
/// Interprets the bytes as a hugr package, verifies there is exactly one module in the
/// package, validates it, checks for unsupported operations, then extracts and returns that module.
///
/// # Errors
/// Returns an error if:
/// - The input is empty
/// - The HUGR format is invalid
/// - The package doesn't contain exactly one module
/// - The package contains unsupported operations
/// - Package validation fails
pub fn read_hugr_envelope(bytes: &[u8]) -> Result<Hugr> {
    if bytes.is_empty() {
        return Err(anyhow!("Empty HUGR input"));
    }

    let (_desc, package) = read_envelope(bytes, &REGISTRY)
        .map_err(|e| Error::new(e).context("Failed to read HUGR"))?;

    if package.modules.len() != 1 {
        return Err(anyhow!(
            "Expected exactly one module in the package, found {}",
            package.modules.len()
        ));
    }

    package
        .validate()
        .map_err(|e| Error::new(e).context("HUGR package validation failed"))?;

    // Check that no opaque tket1 operations are present
    for node in package.modules[0].nodes() {
        let op = package.modules[0].get_optype(node);
        if let Some(name) = is_opaque_tket1_op(op) {
            return Err(anyhow!(
                "Pytket op '{name}' is not currently supported by the PECOS HUGR-QIS compiler"
            ));
        }
    }

    Ok(package.modules[0].clone())
}

/// Check if the optype is an opaque tket1 operation,
/// and return its name if so.
///
// TODO: Interpreting the operation payload to get the name is a bit hacky atm,
// since `tket` does not make the `OpaqueTk1Op` payload definition public.
fn is_opaque_tket1_op(op: &OpType) -> Option<String> {
    fn get_pytket_op_name(payload: Option<&Term>) -> Option<String> {
        let Some(Term::String(payload)) = payload else {
            return None;
        };
        let json_payload: serde_json::Value = serde_json::from_str(payload).ok()?;
        let name = json_payload
            .as_object()?
            .get("op")?
            .as_object()?
            .get("type")?
            .as_str()?;
        Some(name.to_string())
    }

    let ext_op = op.as_extension_op()?;

    if ext_op.extension_id() != &TKET1_EXTENSION_ID || ext_op.unqualified_id() != TKET1_OP_NAME {
        return None;
    }

    Some(get_pytket_op_name(ext_op.args().first()).unwrap_or_else(|| "unknown".to_string()))
}
