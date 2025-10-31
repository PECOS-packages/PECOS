//! Utilities for HUGR processing and validation

use anyhow::{Error, Result, anyhow};
use tket::extension::{TKET1_EXTENSION_ID, TKET1_OP_NAME};
use tket::hugr::envelope::get_generator;
use tket::hugr::ops::OpType;
use tket::hugr::package::Package;
use tket::hugr::types::Term;
use tket::hugr::{Hugr, HugrView};

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
    // Check if input is JSON format (starts with '{') vs binary envelope format
    if bytes.is_empty() {
        return Err(anyhow!("Empty HUGR input"));
    }

    // Handle JSON format by wrapping in envelope
    let (bytes_to_load, is_json) = if bytes[0] == b'{' {
        // JSON format - wrap it in a binary envelope so HUGR can load it
        let json_str =
            std::str::from_utf8(bytes).map_err(|e| anyhow!("Invalid UTF-8 in JSON HUGR: {e}"))?;

        // Create a binary envelope with JSON content
        // The envelope format is: MAGIC_HEADER + JSON_CONTENT
        // HUGR expects: "HUGRiHJv" (8 bytes) + format byte + compression byte + JSON
        let mut envelope = Vec::new();

        // Magic header for HUGR envelope
        envelope.extend_from_slice(b"HUGRiHJv");

        // Format byte: 0x3F (63) for JSON format (EnvelopeFormat::JSON)
        envelope.push(0x3F);

        // Compression byte: 0x40 (64) - this is what HUGR expects
        envelope.push(0x40);

        // Append the JSON content
        envelope.extend_from_slice(json_str.as_bytes());

        (envelope, true)
    } else {
        (bytes.to_vec(), false)
    };

    // Try to load as a Package first
    // Use None for the registry to allow loading HUGRs with unknown/newer extensions
    // This is more permissive and matches how Selene loads HUGRs
    let mut cursor = std::io::Cursor::new(&bytes_to_load);
    match Package::load(&mut cursor, None) {
        Ok(package) => {
            // Validate package module count
            if package.modules.len() != 1 {
                return Err(anyhow!(
                    "Expected exactly one module in the package, found {}",
                    package.modules.len()
                ));
            }

            // Validate the package
            package.validate().map_err(|e| {
                let generator = get_generator(&package.modules);
                let any = Error::new(e);
                if let Some(generator) = generator {
                    any.context(format!("in package with generator {generator}"))
                } else {
                    any
                }
            })?;

            // Check that no opaque tket1 operations are present
            for node in package.modules[0].nodes() {
                let op = package.modules[0].get_optype(node);
                if let Some(name) = is_opaque_tket1_op(op) {
                    return Err(anyhow!(
                        "Pytket op '{name}' is not currently supported by the PECOS HUGR-QIS compiler"
                    ));
                }
            }

            // Return the single module
            Ok(package.modules[0].clone())
        }
        Err(_) if is_json => {
            // If Package loading failed for JSON, it might be a direct HUGR
            // Try loading as a direct HUGR with None for more permissive loading
            let mut cursor = std::io::Cursor::new(&bytes_to_load);
            match Hugr::load(&mut cursor, None) {
                Ok(hugr) => {
                    // Still check for unsupported operations
                    for node in hugr.nodes() {
                        let op = hugr.get_optype(node);
                        if let Some(name) = is_opaque_tket1_op(op) {
                            return Err(anyhow!(
                                "Pytket op '{name}' is not currently supported by the PECOS HUGR-QIS compiler"
                            ));
                        }
                    }
                    Ok(hugr)
                }
                Err(e) => Err(anyhow!("Failed to load HUGR: {e}")),
            }
        }
        Err(e) => {
            // For binary format, if Package loading failed, try direct HUGR loading
            // Use None for the registry to be more permissive
            log::debug!("Package::load failed with: {e:?}");
            let mut cursor = std::io::Cursor::new(&bytes_to_load);
            match Hugr::load(&mut cursor, None) {
                Ok(hugr) => {
                    log::debug!("Successfully loaded as direct HUGR (not package)");
                    // Still check for unsupported operations
                    for node in hugr.nodes() {
                        let op = hugr.get_optype(node);
                        if let Some(name) = is_opaque_tket1_op(op) {
                            return Err(anyhow!(
                                "Pytket op '{name}' is not currently supported by the PECOS HUGR-QIS compiler"
                            ));
                        }
                    }
                    Ok(hugr)
                }
                Err(hugr_err) => {
                    log::error!("Both Package::load and Hugr::load failed");
                    log::error!("Package error: {e:?}");
                    log::error!("Hugr error: {hugr_err:?}");
                    Err(Error::new(e).context(format!("Error loading HUGR package (also tried direct HUGR load which failed with: {hugr_err})")))
                }
            }
        }
    }
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
