//! Utility functions for Selene runtime plugins
//!
//! Convenient access to Selene runtime implementations.
//! Runtimes come from `PECOS_SELENE_DIR` or a registered host bundle. Prepare
//! development libraries with `pecos qis-runtime`, or `just build-debug` for Python.

use crate::{RuntimeNativeGateSet, SeleneRuntime};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static BUNDLED_LIBRARY_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Register a host package's bundled QIS libraries before creating an engine.
///
/// Explicit runtime environment overrides retain precedence. The directory is
/// process-wide, like the QIS runtime singleton, and cannot be changed after
/// registration. Rust applications can use `PECOS_SELENE_DIR` with `pecos qis-runtime`.
///
/// # Errors
/// Returns an error if a different bundle was already registered.
pub fn set_bundled_library_dir(path: impl AsRef<Path>) -> Result<(), &'static str> {
    let path = path.as_ref();
    let registered = BUNDLED_LIBRARY_DIR.get_or_init(|| path.to_path_buf());
    if registered == path {
        Ok(())
    } else {
        Err("A different QIS library bundle is already registered")
    }
}

pub(crate) fn bundled_library(name: &str) -> Option<PathBuf> {
    find_library_in_dir(BUNDLED_LIBRARY_DIR.get()?, name)
}

const SIMPLE_RUNTIME_DEFAULT_INIT_ARGS: &[&str] = &[
    "--duration-ns-rxy=0",
    "--duration-ns-rzz=0",
    "--duration-ns-rz=0",
    "--duration-ns-rpp=0",
    "--duration-ns-measure=0",
    "--duration-ns-reset=0",
    "--duration-ns-measure-leaked=0",
];

const SOFT_RZ_RUNTIME_DEFAULT_INIT_ARGS: &[&str] = &[
    "--duration-ns-rxy=0",
    "--duration-ns-rzz=0",
    "--duration-ns-measure=0",
    "--duration-ns-reset=0",
    "--duration-ns-measure-leaked=0",
    "--max-batch-size=1",
];

fn platform_library_parts() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("", "dll")
    } else if cfg!(target_os = "macos") {
        ("lib", "dylib")
    } else {
        ("lib", "so")
    }
}

pub(crate) fn platform_library_name(lib_name: &str) -> String {
    let (lib_prefix, lib_ext) = platform_library_parts();
    format!("{lib_prefix}{lib_name}.{lib_ext}")
}

/// Find only the exact platform library filename directly in this directory.
pub(crate) fn find_library_in_dir(dir: &Path, lib_name: &str) -> Option<PathBuf> {
    let path = dir.join(platform_library_name(lib_name));
    path.is_file().then_some(path)
}

/// Error type for runtime fetching
#[derive(Debug)]
pub enum RuntimeFetchError {
    IoError(std::io::Error),
    DownloadError(String),
    InvalidPath(String),
}

impl std::fmt::Display for RuntimeFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::DownloadError(msg) => write!(f, "Download error: {msg}"),
            Self::InvalidPath(msg) => write!(f, "Invalid path: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeFetchError {}

impl From<std::io::Error> for RuntimeFetchError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Create a Selene Simple Runtime
///
/// Load from `PECOS_SELENE_DIR`, then the registered host bundle.
/// Prepare development libraries with `pecos qis-runtime` or `just build-debug`.
///
/// # Example
/// ```rust
/// use pecos_qis::selene_simple_runtime;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the simple runtime (prepared with pecos qis-runtime)
/// match selene_simple_runtime() {
///     Ok(runtime) => {
///         println!("Runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
///     }
///     Err(e) => {
///         // Prepare libraries with pecos qis-runtime or just build-debug
///         eprintln!("Simple runtime not available: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the Selene simple runtime library cannot be found.
pub fn selene_simple_runtime() -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime("selene_simple_runtime")?;
    log::debug!(
        "selene_simple_runtime: Found runtime at: {}",
        runtime_path.display()
    );
    let runtime = SeleneRuntime::with_plugin_config(
        runtime_path,
        SIMPLE_RUNTIME_DEFAULT_INIT_ARGS
            .iter()
            .map(ToString::to_string)
            .collect(),
        Vec::new(),
    )
    .with_native_gate_set(RuntimeNativeGateSet::ALL);
    Ok(runtime)
}

/// Create a Selene Soft RZ Runtime
///
/// This runtime implements soft RZ gates for more accurate gate modeling.
/// Load from `PECOS_SELENE_DIR`, then the registered host bundle.
/// Prepare development libraries with `pecos qis-runtime` or `just build-debug`.
///
/// # Example
/// ```rust
/// use pecos_qis::selene_soft_rz_runtime;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the soft RZ runtime (prepared with pecos qis-runtime)
/// match selene_soft_rz_runtime() {
///     Ok(runtime) => {
///         println!("Soft RZ runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
///     }
///     Err(e) => {
///         // Prepare libraries with pecos qis-runtime or just build-debug
///         eprintln!("Soft RZ runtime not available: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the Selene soft RZ runtime library cannot be found.
pub fn selene_soft_rz_runtime() -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime("selene_soft_rz_runtime")?;
    Ok(SeleneRuntime::with_plugin_config(
        runtime_path,
        SOFT_RZ_RUNTIME_DEFAULT_INIT_ARGS
            .iter()
            .map(ToString::to_string)
            .collect(),
        Vec::new(),
    )
    .with_native_gate_set(RuntimeNativeGateSet::RXY_RZ_RZZ))
}

// Note: We only expose convenience functions for actual Selene runtime plugins.
// Other Selene plugins (error models, simulators, compilers) can still be loaded
// using find_selene_runtime() or selene_runtime() with an explicit path.

/// Find a runtime in `PECOS_SELENE_DIR`, then the registered host bundle.
/// Cargo build directories are internal; `pecos qis-runtime` stages JSON-reported artifacts.
fn find_built_selene_runtime(lib_name: &str) -> Result<PathBuf, RuntimeFetchError> {
    if let Some(directory) = std::env::var_os("PECOS_SELENE_DIR") {
        let directory = Path::new(&directory);
        return find_library_in_dir(directory, lib_name).ok_or_else(|| {
            RuntimeFetchError::InvalidPath(format!(
                "PECOS_SELENE_DIR is set to {}, but {} was not found directly in that directory. Prepare it with `pecos qis-runtime --profile <profile>` (add --tests for Rust tests), using the profile you built with",
                directory.display(), platform_library_name(lib_name)
            ))
        });
    }
    bundled_library(lib_name).ok_or_else(|| RuntimeFetchError::InvalidPath(format!(
        "Selene runtime {} not found. Set PECOS_SELENE_DIR and PECOS_QIS_FFI_PATH to the directory printed by `pecos qis-runtime --profile <profile>` (add --tests for Rust tests), using the profile you built with, or use `pecos rust test --profile <profile>`. For Python, run `pecos python build --profile <profile>` with the same profile (for example `just build-debug` for debug).",
        platform_library_name(lib_name)
    )))
}

/// Try to find a Selene runtime in common locations
///
/// Searches in order:
/// 1. `PECOS_SELENE_DIR` environment variable
/// 2. Registered host bundle
/// 3. System library paths (/usr/local/lib, /usr/lib, /opt/pecos/lib)
///
/// Only the exact platform filename in each directory is considered.
/// An invalid `PECOS_SELENE_DIR` override warns and stops the search.
#[must_use]
pub fn find_selene_runtime(name: &str) -> Option<PathBuf> {
    let lib_name = if name.starts_with("selene_") {
        name.to_string()
    } else {
        format!("selene_{name}")
    };
    let filename = platform_library_name(&lib_name);

    // Check environment variable
    if let Some(selene_dir) = std::env::var_os("PECOS_SELENE_DIR") {
        let selene_dir = PathBuf::from(selene_dir);
        if let Some(path) = find_library_in_dir(&selene_dir, &lib_name) {
            return Some(path);
        }
        log::warn!(
            "PECOS_SELENE_DIR is set to {}, but {filename} was not found directly in that directory. Prepare it with `pecos qis-runtime --profile <profile>` using the profile you built with",
            selene_dir.display()
        );
        return None;
    }

    if let Some(path) = bundled_library(&lib_name) {
        return Some(path);
    }

    // Check system paths
    for sys_path in &["/usr/local/lib", "/usr/lib", "/opt/pecos/lib"] {
        let path = PathBuf::from(sys_path).join(&filename);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

/// Create a Selene runtime automatically
///
/// This loads from `PECOS_SELENE_DIR` or the host bundle. The name should be
/// the library name (e.g., "`selene_simple_runtime`", "`selene_soft_rz_runtime`").
///
/// # Example
/// ```rust
/// use pecos_qis::selene_runtime_auto;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load a runtime by name (prepared with pecos qis-runtime)
/// match selene_runtime_auto("selene_simple_runtime") {
///     Ok(runtime) => {
///         println!("Runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
///     }
///     Err(e) => {
///         // Prepare libraries with pecos qis-runtime or just build-debug
///         eprintln!("Could not load runtime: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the specified Selene runtime library cannot be found.
pub fn selene_runtime_auto(lib_name: &str) -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime(lib_name)?;
    let init_args = match lib_name {
        "selene_simple_runtime" => SIMPLE_RUNTIME_DEFAULT_INIT_ARGS,
        "selene_soft_rz_runtime" => SOFT_RZ_RUNTIME_DEFAULT_INIT_ARGS,
        _ => &[],
    };
    Ok(SeleneRuntime::with_plugin_config(
        runtime_path,
        init_args.iter().map(ToString::to_string).collect(),
        Vec::new(),
    )
    .with_native_gate_set(match lib_name {
        "selene_soft_rz_runtime" => RuntimeNativeGateSet::RXY_RZ_RZZ,
        _ => RuntimeNativeGateSet::ALL,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::{ENV_MUTEX, EnvVarGuard, run_test_in_child};
    use std::fs::File;

    #[test]
    fn no_bundle_or_environment_does_not_search_cargo_layout() {
        const CHILD: &str = "PECOS_TEST_NO_RUNTIME_SOURCES";
        if std::env::var_os(CHILD).is_none() {
            crate::test_env::join_test_child(crate::test_env::spawn_test_child_removing(
                "selene_runtimes::tests::no_bundle_or_environment_does_not_search_cargo_layout",
                &[(CHILD, "1".as_ref())],
                &["PECOS_SELENE_DIR", "PECOS_QIS_FFI_PATH"],
            ));
            return;
        }
        assert!(BUNDLED_LIBRARY_DIR.get().is_none());
        assert!(std::env::var_os("PECOS_SELENE_DIR").is_none());
        assert!(std::env::var_os("PECOS_QIS_FFI_PATH").is_none());
        let temp = tempfile::tempdir().expect("fake cargo target");
        let deps = temp.path().join("debug/deps");
        std::fs::create_dir_all(&deps).expect("fake cargo deps");
        File::create(deps.join(platform_library_name("selene_simple_runtime")))
            .expect("ignored library");
        let _lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _target = EnvVarGuard::set("CARGO_TARGET_DIR", temp.path());
        let error = find_built_selene_runtime("selene_simple_runtime")
            .expect_err("no implicit Cargo lookup")
            .to_string();
        for text in [
            "selene_simple_runtime",
            "PECOS_SELENE_DIR",
            "PECOS_QIS_FFI_PATH",
            "pecos qis-runtime",
            "pecos python build --profile <profile>",
            "just build-debug",
        ] {
            assert!(error.contains(text), "{error}");
        }
    }

    #[test]
    fn registered_bundle_resolves_both_runtime_kinds() {
        const CHILD: &str = "PECOS_TEST_RUNTIME_BUNDLE";
        if std::env::var_os(CHILD).is_none() {
            crate::test_env::join_test_child(crate::test_env::spawn_test_child_removing(
                "selene_runtimes::tests::registered_bundle_resolves_both_runtime_kinds",
                &[(CHILD, "1".as_ref())],
                &["PECOS_SELENE_DIR", "PECOS_QIS_FFI_PATH"],
            ));
            return;
        }
        let temp = tempfile::tempdir().expect("bundle directory");
        set_bundled_library_dir(temp.path()).expect("register bundle");
        for name in ["selene_simple_runtime", "selene_soft_rz_runtime"] {
            let library = temp.path().join(platform_library_name(name));
            File::create(&library).expect("bundle library");
            assert_eq!(
                find_built_selene_runtime(name).expect("bundled runtime"),
                library
            );
            assert_eq!(find_selene_runtime(name), Some(library));
        }
    }

    #[test]
    fn invalid_override_does_not_fall_back_to_registered_bundle() {
        const CHILD: &str = "PECOS_TEST_SELENE_INVALID_OVERRIDE_BUNDLE";
        if std::env::var_os(CHILD).is_none() {
            crate::test_env::join_test_child(crate::test_env::spawn_test_child_removing(
                "selene_runtimes::tests::invalid_override_does_not_fall_back_to_registered_bundle",
                &[(CHILD, "1".as_ref())],
                &["PECOS_SELENE_DIR", "PECOS_QIS_FFI_PATH"],
            ));
            return;
        }
        let _lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let name = "selene_simple_runtime";
        let bundle = tempfile::tempdir().expect("bundle directory");
        let filename = platform_library_name(name);
        let library = bundle.path().join(&filename);
        File::create(&library).expect("bundled library");
        set_bundled_library_dir(bundle.path()).expect("register bundle");
        assert_eq!(
            find_built_selene_runtime(name).expect("bundle lookup"),
            library
        );
        assert_eq!(find_selene_runtime(name), Some(library));

        let empty = tempfile::tempdir().expect("invalid override directory");
        let _invalid = EnvVarGuard::set("PECOS_SELENE_DIR", empty.path());
        let error = find_built_selene_runtime(name)
            .expect_err("invalid override must not fall back to bundle")
            .to_string();
        for text in [
            "PECOS_SELENE_DIR",
            &empty.path().display().to_string(),
            &filename,
        ] {
            assert!(error.contains(text), "{error}");
        }
        assert_eq!(find_selene_runtime(name), None);
    }

    #[test]
    fn bundled_runtime_defaults_satisfy_selene_0_3_required_arguments() {
        assert_eq!(SIMPLE_RUNTIME_DEFAULT_INIT_ARGS.len(), 7);
        assert!(
            SIMPLE_RUNTIME_DEFAULT_INIT_ARGS
                .iter()
                .all(|arg| arg.starts_with("--duration-ns-") && arg.ends_with("=0"))
        );
        assert_eq!(SOFT_RZ_RUNTIME_DEFAULT_INIT_ARGS.len(), 6);
        assert_eq!(
            SOFT_RZ_RUNTIME_DEFAULT_INIT_ARGS.last(),
            Some(&"--max-batch-size=1")
        );
    }

    #[test]
    fn test_find_selene_runtime() {
        let _env_lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // This might not find anything in test environment
        let result = find_selene_runtime("simple");
        // Just verify it doesn't panic
        if let Some(path) = result {
            assert!(path.to_string_lossy().contains("selene_simple"));
        }
    }

    #[test]
    fn test_find_selene_runtime_rejects_hashed_env_deps_dir() {
        // Every runtime lookup reads PECOS_SELENE_DIR, including lookups in tests
        // that do not hold ENV_MUTEX, so change it only in a child process.
        const CHILD_ENV: &str = "PECOS_TEST_HASHED_ENV_DEPS_DIR";
        if std::env::var_os(CHILD_ENV).is_none() {
            run_test_in_child(
                "selene_runtimes::tests::test_find_selene_runtime_rejects_hashed_env_deps_dir",
                &[(CHILD_ENV, "1".as_ref())],
            );
            return;
        }
        let _env_lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let temp_dir = tempfile::tempdir().expect("create temp directory");
        let deps_dir = temp_dir.path().join("deps");
        std::fs::create_dir(&deps_dir).expect("create deps directory");
        let runtime_path = deps_dir.join({
            let (prefix, ext) = platform_library_parts();
            format!("{prefix}selene_simple_runtime-abc123.{ext}")
        });
        File::create(&runtime_path).expect("create hashed runtime library");
        let _env = EnvVarGuard::set("PECOS_SELENE_DIR", temp_dir.path());

        assert_eq!(find_selene_runtime("selene_simple_runtime"), None);
        let error = find_built_selene_runtime("selene_simple_runtime")
            .expect_err("invalid override")
            .to_string();
        for text in [
            "PECOS_SELENE_DIR",
            &temp_dir.path().display().to_string(),
            &platform_library_name("selene_simple_runtime"),
            "pecos qis-runtime --profile <profile>",
        ] {
            assert!(error.contains(text), "{error}");
        }
    }

    #[test]
    fn library_scan_only_accepts_exact_file_in_requested_directory() {
        let directory = tempfile::tempdir().expect("test directory");
        let name = "selene_simple_runtime";
        let exact = directory.path().join(platform_library_name(name));
        let deps = directory.path().join("deps");
        std::fs::create_dir(&deps).expect("deps");
        File::create(deps.join(platform_library_name(name))).expect("deps exact library");
        let (prefix, extension) = platform_library_parts();
        for root in [directory.path(), deps.as_path()] {
            File::create(root.join(format!("{prefix}{name}-abc123.{extension}")))
                .expect("hashed library");
        }
        assert_eq!(find_library_in_dir(directory.path(), name), None);
        File::create(&exact).expect("exact library");
        assert_eq!(find_library_in_dir(directory.path(), name), Some(exact));
    }
}
