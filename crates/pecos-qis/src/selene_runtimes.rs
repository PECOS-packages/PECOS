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

pub(crate) fn platform_hashed_library_pattern(lib_name: &str) -> String {
    let (lib_prefix, lib_ext) = platform_library_parts();
    format!("{lib_prefix}{lib_name}-*.{lib_ext}")
}

/// Find an exact or Cargo hash-suffixed dynamic library in a directory or its `deps/`.
pub(crate) fn find_library_in_dir(dir: &Path, lib_name: &str) -> Option<PathBuf> {
    let (lib_prefix, lib_ext) = platform_library_parts();
    let exact_name = platform_library_name(lib_name);
    let hashed_stem_prefix = format!("{lib_prefix}{lib_name}-");

    // Cargo dependency artifacts are fresher than copied top-level artifacts.
    for search_dir in [dir.join("deps"), dir.to_path_buf()] {
        let exact_path = search_dir.join(&exact_name);
        if exact_path.is_file() {
            return Some(exact_path);
        }

        let mut hashed_paths = std::fs::read_dir(&search_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(lib_ext))
                    && path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .and_then(|stem| stem.strip_prefix(&hashed_stem_prefix))
                        .is_some_and(|hash| !hash.is_empty())
            })
            .collect::<Vec<_>>();

        let modified_times = hashed_paths
            .iter()
            .map(|path| path.metadata().and_then(|metadata| metadata.modified()))
            .collect::<Result<Vec<_>, _>>();
        if let Ok(modified_times) = modified_times {
            let mut paths_with_times = hashed_paths
                .into_iter()
                .zip(modified_times)
                .collect::<Vec<_>>();
            paths_with_times.sort_unstable_by(
                |(left_path, left_time), (right_path, right_time)| {
                    right_time
                        .cmp(left_time)
                        .then_with(|| left_path.cmp(right_path))
                },
            );
            if let Some((path, _)) = paths_with_times.into_iter().next() {
                return Some(path);
            }
            continue;
        }

        // Some filesystems do not expose modification times.
        hashed_paths.sort_unstable();
        if let Some(path) = hashed_paths.into_iter().next() {
            return Some(path);
        }
    }

    None
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
    if let Some(path) = std::env::var_os("PECOS_SELENE_DIR")
        .and_then(|dir| find_library_in_dir(Path::new(&dir), lib_name))
        .or_else(|| bundled_library(lib_name))
    {
        return Ok(path);
    }
    Err(RuntimeFetchError::InvalidPath(format!(
        "Selene runtime {lib_name} not found. Set PECOS_SELENE_DIR and PECOS_QIS_FFI_PATH to the directory printed by `pecos qis-runtime --profile debug` (add --tests for Rust tests), use `pecos rust test` / `just rstest debug`, or run `just build-debug` for Python."
    )))
}

/// Try to find a Selene runtime in common locations
///
/// Searches in order:
/// 1. `PECOS_SELENE_DIR` environment variable
/// 2. Registered host bundle
/// 3. System library paths (/usr/local/lib, /usr/lib, /opt/pecos/lib)
///
/// Hash-suffixed matching is anchored on the complete requested library name
/// followed by `-`. A partial name such as `"simple"` therefore does not match
/// `selene_simple_runtime-*` artifacts.
///
/// `PECOS_SELENE_DIR` is one source in this fallthrough search, so a bad value
/// warns and continues. In contrast, `PECOS_QIS_FFI_PATH` is an authoritative
/// per-purpose override and fails loudly when set incorrectly.
#[must_use]
pub fn find_selene_runtime(name: &str) -> Option<PathBuf> {
    let lib_name = if name.starts_with("selene_") {
        name.to_string()
    } else {
        format!("selene_{name}")
    };
    let filename = platform_library_name(&lib_name);
    let hashed_pattern = platform_hashed_library_pattern(&lib_name);

    // Check environment variable
    if let Some(selene_dir) = std::env::var_os("PECOS_SELENE_DIR") {
        let selene_dir = PathBuf::from(selene_dir);
        if let Some(path) = find_library_in_dir(&selene_dir, &lib_name) {
            return Some(path);
        }
        log::warn!(
            "PECOS_SELENE_DIR is set to {}, but neither {filename} nor {hashed_pattern} was found there or in its deps directory",
            selene_dir.display()
        );
    }

    if let Some(path) = bundled_library(&lib_name) {
        return Some(path);
    }

    // Check system paths
    for sys_path in &["/usr/local/lib", "/usr/lib", "/opt/pecos/lib"] {
        let path = PathBuf::from(sys_path).join(&filename);
        if path.exists() {
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
    use std::time::{Duration, SystemTime};

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
            "just rstest debug",
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
    fn test_find_selene_runtime_in_hashed_env_deps_dir() {
        // Every runtime lookup reads PECOS_SELENE_DIR, including lookups in tests
        // that do not hold ENV_MUTEX, so change it only in a child process.
        const CHILD_ENV: &str = "PECOS_TEST_HASHED_ENV_DEPS_DIR";
        if std::env::var_os(CHILD_ENV).is_none() {
            run_test_in_child(
                "selene_runtimes::tests::test_find_selene_runtime_in_hashed_env_deps_dir",
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

        assert_eq!(
            find_selene_runtime("selene_simple_runtime"),
            Some(runtime_path)
        );
    }

    #[test]
    fn library_scan_prefers_newest_hashed_dep_over_top_level_exact() {
        let temp_dir = tempfile::tempdir().expect("create temp directory");
        let exact_path = temp_dir
            .path()
            .join(platform_library_name("selene_simple_runtime"));
        File::create(exact_path).expect("create top-level exact library");

        let deps_dir = temp_dir.path().join("deps");
        std::fs::create_dir(&deps_dir).expect("create deps directory");
        let (prefix, ext) = platform_library_parts();
        let older_path = deps_dir.join(format!("{prefix}selene_simple_runtime-aaa-old.{ext}"));
        let older_file = File::create(older_path).expect("create older hashed library");
        older_file
            .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1))
            .expect("set older modification time");
        let newer_path = deps_dir.join(format!("{prefix}selene_simple_runtime-zzz-new.{ext}"));
        let newer_file = File::create(&newer_path).expect("create newer hashed library");
        newer_file
            .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(2))
            .expect("set newer modification time");

        assert_eq!(
            find_library_in_dir(temp_dir.path(), "selene_simple_runtime"),
            Some(newer_path)
        );
    }
}
