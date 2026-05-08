//! Runtime plugin discovery.
//!
//! Scans a plugin directory for shared libraries, loads each one, and calls
//! its `pecos_plugin_init` entry point to get versioned vtables.
//!
//! # Plugin contract
//!
//! A plugin shared library must export a C function:
//!
//! ```c
//! int pecos_plugin_init(PecosPluginDescriptor *desc);
//! ```
//!
//! The plugin fills in `desc` with its name, version, and any vtables it provides.
//! Returns 0 on success, non-zero on error.
//!
//! # Plugin directory
//!
//! Default: `~/.pecos/plugins/`
//!
//! PECOS scans this directory for `.so` (Linux), `.dylib` (macOS), or `.dll` (Windows)
//! files and attempts to load each one.

use crate::decoder::{ForeignDecoder, ForeignDecoderVTable};
use crate::simulator::{ForeignSimulator, ForeignSimulatorVTable};
use log::{info, warn};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

/// Descriptor filled by a plugin's `pecos_plugin_init` function.
#[repr(C)]
pub struct PluginDescriptor {
    /// Plugin name (null-terminated, static -- do not free).
    pub name: *const c_char,

    /// Plugin ABI version. Must match what PECOS expects.
    /// This is the plugin protocol version, separate from decoder/simulator vtable versions.
    pub plugin_api_version: u32,

    /// Opaque handle to the decoder, or null if the plugin does not provide one.
    pub decoder_handle: *mut (),
    /// Decoder vtable, or null if the plugin does not provide a decoder.
    pub decoder_vtable: *const ForeignDecoderVTable,

    /// Opaque handle to the simulator, or null if the plugin does not provide one.
    pub simulator_handle: *mut (),
    /// Number of qubits the simulator was created with.
    pub simulator_num_qubits: usize,
    /// Simulator vtable, or null if the plugin does not provide a simulator.
    pub simulator_vtable: *const ForeignSimulatorVTable,
}

/// Current plugin API version.
pub const PLUGIN_API_VERSION: u32 = 1;

/// Type signature of the plugin init function.
type PluginInitFn = unsafe extern "C" fn(desc: *mut PluginDescriptor) -> i32;

/// A loaded plugin with its decoder and/or simulator.
pub struct LoadedPlugin {
    /// Plugin name.
    pub name: String,
    /// Source path.
    pub path: PathBuf,
    /// Loaded decoder, if the plugin provides one.
    pub decoder: Option<ForeignDecoder>,
    /// Loaded simulator, if the plugin provides one.
    pub simulator: Option<ForeignSimulator>,
    /// Keep the library alive so symbols remain valid.
    _library: libloading::Library,
}

/// Errors from plugin discovery.
#[derive(Debug)]
pub enum PluginError {
    /// Failed to load the shared library.
    LoadFailed(String, libloading::Error),
    /// The library does not export `pecos_plugin_init`.
    MissingInitFn(String),
    /// The plugin's `pecos_plugin_init` returned an error.
    InitFailed(String, i32),
    /// Plugin API version mismatch.
    VersionMismatch(String, u32, u32),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadFailed(path, e) => write!(f, "failed to load {path}: {e}"),
            Self::MissingInitFn(path) => {
                write!(f, "{path}: no pecos_plugin_init symbol")
            }
            Self::InitFailed(path, code) => {
                write!(f, "{path}: pecos_plugin_init returned {code}")
            }
            Self::VersionMismatch(name, got, expected) => {
                write!(
                    f,
                    "{name}: plugin API version mismatch (plugin v{got}, PECOS expects v{expected})"
                )
            }
        }
    }
}

impl std::error::Error for PluginError {}

/// Load a single plugin from a shared library path.
///
/// # Errors
///
/// Returns a `PluginError` if the library cannot be loaded, is missing the
/// init function, or fails version checks.
pub fn load_plugin(path: &Path) -> Result<LoadedPlugin, PluginError> {
    let path_str = path.display().to_string();

    // SAFETY: Loading a shared library is inherently unsafe. The caller
    // is responsible for ensuring the library is trustworthy.
    let library = unsafe { libloading::Library::new(path) }
        .map_err(|e| PluginError::LoadFailed(path_str.clone(), e))?;

    // Look up the init function.
    let init_fn: libloading::Symbol<'_, PluginInitFn> = unsafe {
        library
            .get(b"pecos_plugin_init\0")
            .map_err(|_| PluginError::MissingInitFn(path_str.clone()))?
    };

    // Call init to get the descriptor.
    let mut desc = PluginDescriptor {
        name: std::ptr::null(),
        plugin_api_version: 0,
        decoder_handle: std::ptr::null_mut(),
        decoder_vtable: std::ptr::null(),
        simulator_handle: std::ptr::null_mut(),
        simulator_num_qubits: 0,
        simulator_vtable: std::ptr::null(),
    };

    let rc = unsafe { init_fn(&raw mut desc) };
    if rc != 0 {
        return Err(PluginError::InitFailed(path_str, rc));
    }

    // Check plugin API version.
    if desc.plugin_api_version != PLUGIN_API_VERSION {
        let name = plugin_name(&desc);
        return Err(PluginError::VersionMismatch(
            name,
            desc.plugin_api_version,
            PLUGIN_API_VERSION,
        ));
    }

    let name = plugin_name(&desc);

    // Wrap decoder if provided.
    let decoder = if !desc.decoder_handle.is_null() && !desc.decoder_vtable.is_null() {
        let vtable_copy = unsafe { *desc.decoder_vtable };
        unsafe { ForeignDecoder::new(desc.decoder_handle, vtable_copy) }
    } else {
        None
    };

    // Wrap simulator if provided.
    let simulator = if !desc.simulator_handle.is_null() && !desc.simulator_vtable.is_null() {
        let vtable_copy = unsafe { *desc.simulator_vtable };
        unsafe {
            ForeignSimulator::new(
                desc.simulator_handle,
                vtable_copy,
                desc.simulator_num_qubits,
            )
        }
    } else {
        None
    };

    info!(
        "Loaded plugin '{}' from {} (decoder: {}, simulator: {})",
        name,
        path_str,
        decoder.is_some(),
        simulator.is_some()
    );

    Ok(LoadedPlugin {
        name,
        path: path.to_path_buf(),
        decoder,
        simulator,
        _library: library,
    })
}

/// Discover and load all plugins from the default plugin directory (`~/.pecos/plugins/`).
///
/// Returns successfully loaded plugins. Errors for individual plugins are logged
/// as warnings but do not prevent other plugins from loading.
#[must_use]
pub fn discover_plugins() -> Vec<LoadedPlugin> {
    let plugin_dir = default_plugin_dir();
    discover_plugins_in(&plugin_dir)
}

/// Discover and load all plugins from a specific directory.
///
/// Returns successfully loaded plugins. Individual plugin errors are logged as warnings.
#[must_use]
pub fn discover_plugins_in(dir: &Path) -> Vec<LoadedPlugin> {
    if !dir.is_dir() {
        return vec![];
    }

    let extension = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    let mut plugins = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("Cannot read plugin directory {}: {}", dir.display(), e);
            return plugins;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some(extension) {
            match load_plugin(&path) {
                Ok(plugin) => plugins.push(plugin),
                Err(e) => warn!("Skipping plugin {}: {}", path.display(), e),
            }
        }
    }

    info!(
        "Discovered {} plugin(s) in {}",
        plugins.len(),
        dir.display()
    );
    plugins
}

/// Default plugin directory: `~/.pecos/plugins/`
fn default_plugin_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pecos")
        .join("plugins")
}

fn plugin_name(desc: &PluginDescriptor) -> String {
    if desc.name.is_null() {
        "unnamed".to_string()
    } else {
        unsafe { CStr::from_ptr(desc.name) }
            .to_str()
            .unwrap_or("unnamed")
            .to_string()
    }
}
