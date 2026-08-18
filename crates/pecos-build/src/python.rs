//! Build-script helpers for Python extension modules.

use std::path::Path;

/// Emit the Python distribution version for a `pyo3` module's `__version__`.
///
/// Reads `[project].version` from the `pyproject.toml` beside the calling crate's
/// `Cargo.toml` and exposes it to the crate as `PECOS_PYTHON_VERSION`, for
/// `env!("PECOS_PYTHON_VERSION")`.
///
/// Crate versions ride the Rust workspace train (`[workspace.package].version`) and are
/// deliberately a different number from the wheel's `[project].version`, so
/// `CARGO_PKG_VERSION` is the wrong source for a version users see from Python.
///
/// # Panics
///
/// Panics if the `pyproject.toml` is missing, unparsable, or has no `[project].version` --
/// a wheel that cannot report its own version is a broken build, not a recoverable state.
pub fn emit_python_version() {
    println!("cargo:rerun-if-changed=pyproject.toml");

    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let pyproject_path = Path::new(&manifest_dir).join("pyproject.toml");
    let pyproject = std::fs::read_to_string(&pyproject_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", pyproject_path.display()));
    let parsed = pyproject
        .parse::<toml::Table>()
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", pyproject_path.display()));
    let version = parsed
        .get("project")
        .and_then(|project| project.get("version"))
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("{}: missing [project].version", pyproject_path.display()));

    println!("cargo:rustc-env=PECOS_PYTHON_VERSION={version}");
}
