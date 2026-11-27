// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Wasmtime-based WebAssembly foreign object implementation.
//!
//! Uses the Wasmtime runtime with timeout handling and type conversions.
//!
//! Fork is supported when no other thread is performing a Wasmtime host-side
//! operation: compiling, instantiating, creating the Engine, dropping stores or
//! modules, or handling a trap. A worker parked in guest code is fine. Forks
//! that skip `pthread_atfork` handlers (raw `clone`/`clone3`, `vfork`, or glibc
//! `_Fork`) are unsupported.

use crate::foreign_object::ForeignObject;
use log::{debug, warn};
use parking_lot::RwLock;
use pecos_core::errors::PecosError;
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;
use wasmtime::{
    Config, Engine, Func, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Trap, Val,
    ValRaw,
};

/// Scalar types supported by the strict Guppy WASM interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmScalarType {
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 double precision, transported without changing its bits.
    F64,
    /// Signed 32-bit integer in a Guppy 64-bit transport slot. Arguments wrap to their
    /// low 32 bits and results are sign-extended into the 64-bit transport slot.
    I32,
    /// Select i32 or i64 from the actual export signature for each integer lane.
    IntegerAuto,
}

impl WasmScalarType {
    fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::IntegerAuto => "i32|i64",
        }
    }
}

/// Length of each tick in milliseconds (10ms per tick)
const WASM_EXECUTION_TICK_LENGTH_MS: u64 = 10;
// One engine per statically linked copy of this crate. Initialization failures
// are retained so all callers receive the same error rather than a panic.
static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
static FORK_GENERATION: AtomicU64 = AtomicU64::new(0);
// Low bit: STARTING(g) = g << 1, RUNNING(g) = (g << 1) | 1.
// NONE is u64::MAX; the encoding provides 63 bits of fork generations.
static TICKER: AtomicU64 = AtomicU64::new(u64::MAX);
#[cfg(test)]
static TICKER_SPAWNS: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
extern "C" fn after_fork() {
    FORK_GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Current fork generation, without initializing the Engine or fork handler.
/// Meaningful only after WASM compilation has initialized the shared Engine.
///
/// Fork is supported only when no other thread is performing a Wasmtime
/// host-side operation (compiling, instantiating, creating the Engine, dropping
/// stores or modules, or handling a trap). A worker parked in guest code is fine.
/// Forks that skip `pthread_atfork` handlers (raw `clone`/`clone3`, `vfork`, or
/// glibc `_Fork`) are unsupported. On non-Unix platforms this is always zero.
#[must_use]
pub fn fork_generation() -> u64 {
    FORK_GENERATION.load(Ordering::Relaxed)
}

fn shared_engine() -> Result<&'static Engine, PecosError> {
    ENGINE
        .get_or_init(|| {
            crate::runtime_library::pin()?;
            #[cfg(unix)]
            {
                // SAFETY: The child handler only increments a lock-free atomic;
                // it captures nothing and remains valid for the process lifetime.
                let code = unsafe { libc::pthread_atfork(None, None, Some(after_fork)) };
                if code != 0 {
                    return Err(format!(
                        "Failed to register WebAssembly fork handler: error {code}"
                    ));
                }
            }
            let mut config = Config::new();
            config.epoch_interruption(true);
            config.macos_use_mach_ports(false);
            Engine::new(&config).map_err(|e| format!("Failed to create WebAssembly engine: {e}"))
        })
        .as_ref()
        .map_err(|error| PecosError::Processing(error.clone()))
}

fn ensure_ticker() -> Result<(), PecosError> {
    let engine = shared_engine()?;
    loop {
        let generation = fork_generation();
        let starting = generation << 1;
        let running = starting | 1;
        let observed = TICKER.load(Ordering::Acquire);
        if observed == running {
            return Ok(());
        }
        if observed == starting {
            thread::yield_now();
            continue;
        }
        if TICKER
            .compare_exchange(observed, starting, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        // This detached ticker lives for the process lifetime. The containing library
        // is pinned at Engine creation on Linux, macOS and Windows.
        match thread::Builder::new()
            .name("pecos-wasm-epoch".into())
            .spawn(move || {
                loop {
                    engine.increment_epoch();
                    thread::sleep(Duration::from_millis(WASM_EXECUTION_TICK_LENGTH_MS));
                }
            }) {
            Ok(_) => {
                #[cfg(test)]
                TICKER_SPAWNS.fetch_add(1, Ordering::Relaxed);
                TICKER.store(running, Ordering::Release);
                return Ok(());
            }
            Err(error) => {
                TICKER.store(u64::MAX, Ordering::Release);
                return Err(PecosError::Processing(format!(
                    "Failed to start WebAssembly epoch ticker: {error}"
                )));
            }
        }
    }
}

/// Default timeout in seconds (1 second to match Python implementation)
const DEFAULT_TIMEOUT_SECONDS: f64 = 1.0;

// Owned primitive signatures never root types in the shared Engine registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    I32,
    I64,
    F32,
    F64,
    Other,
}

fn format_lanes<T: Copy>(lanes: &[T], name: fn(T) -> &'static str) -> String {
    lanes
        .iter()
        .map(|lane| name(*lane))
        .collect::<Vec<_>>()
        .join(",")
}

impl Lane {
    fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::Other => "other",
        }
    }
}

impl From<wasmtime::ValType> for Lane {
    fn from(value: wasmtime::ValType) -> Self {
        match value {
            wasmtime::ValType::I32 => Self::I32,
            wasmtime::ValType::I64 => Self::I64,
            wasmtime::ValType::F32 => Self::F32,
            wasmtime::ValType::F64 => Self::F64,
            _ => Self::Other,
        }
    }
}

#[derive(Debug)]
struct Signature {
    params: Box<[Lane]>,
    results: Box<[Lane]>,
}

impl Signature {
    fn new(ty: &wasmtime::FuncType) -> Self {
        Self {
            params: ty.params().map(Lane::from).collect(),
            results: ty.results().map(Lane::from).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CachedFunction {
    func: Func,
    export: usize,
}

impl CachedFunction {
    fn signature<'a>(&self, exports: &'a [Export]) -> &'a Signature {
        exports[self.export]
            .signature
            .as_ref()
            .expect("cached function index identifies a function export")
    }
}

#[derive(Debug)]
struct Export {
    name: String,
    signature: Option<Signature>,
}

/// Store context holding resource limits
#[derive(Debug)]
struct StoreContext {
    limits: StoreLimits,
}

impl StoreContext {
    fn new(memory_size: Option<usize>) -> Self {
        let mut builder = StoreLimitsBuilder::new();
        if let Some(size) = memory_size {
            builder = builder.memory_size(size);
        }
        Self {
            limits: builder.build(),
        }
    }
}

/// Immutable compiled WebAssembly code that can create independent instances.
///
/// Clones share compiled code and the process-wide Engine, never stores or mutable
/// WASM state. Retaining this value avoids recompilation between shots.
/// Fork support requires the host-side quiescence and `pthread_atfork` preconditions
/// documented in [`fork_generation`].
#[derive(Clone, Debug)]
pub struct CompiledWasmModule {
    bytes: Arc<[u8]>,
    module: Module,
    exports: Arc<[Export]>,
}

impl CompiledWasmModule {
    /// Compile bytes without executing the module's start function or init export.
    ///
    /// # Errors
    /// Returns an error if engine creation or compilation fails.
    pub fn new(bytes: &[u8]) -> Result<Self, PecosError> {
        let module = Module::new(shared_engine()?, bytes).map_err(|e| {
            PecosError::Processing(format!("Failed to compile WebAssembly module: {e}"))
        })?;
        // Type materialization touches the Engine registry. Do it once per
        // compilation; instances share this ordered, owned primitive metadata.
        let exports = module
            .exports()
            .map(|export| Export {
                name: export.name().into(),
                signature: export.ty().func().map(Signature::new),
            })
            .collect();
        Ok(Self {
            bytes: Arc::from(bytes),
            module,
            exports,
        })
    }

    /// Original input bytes, used for exact cache identity.
    #[must_use]
    pub fn wasm_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Create fresh mutable state with the default execution limits.
    ///
    /// # Errors
    /// Returns an error if instantiation or the start function fails.
    pub fn instantiate(&self) -> Result<WasmForeignObject, PecosError> {
        self.instantiate_with_limits(DEFAULT_TIMEOUT_SECONDS, None)
    }

    /// Create a fresh store and instance with independent resource limits.
    ///
    /// # Errors
    /// Returns an error if instantiation or the start function fails.
    pub fn instantiate_with_limits(
        &self,
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<WasmForeignObject, PecosError> {
        let mut store = Store::new(self.module.engine(), StoreContext::new(memory_size));
        store.limiter(|ctx| &mut ctx.limits);
        let mut object = WasmForeignObject {
            wasm_bytes: self.bytes.clone(),
            module: self.module.clone(),
            store: RwLock::new(store),
            instance: RwLock::new(None),
            functions: HashMap::new(),
            exports: Arc::clone(&self.exports),
            #[cfg(test)]
            checked_fallback_calls: 0,
            raw: Vec::new(),
            timeout_seconds,
            memory_size,
        };
        object.new_instance()?;
        Ok(object)
    }
}

/// WebAssembly foreign object implementation using Wasmtime
///
/// This implementation provides:
/// - Thread-safe execution with RwLock/Mutex synchronization
/// - Configurable timeout via epoch interruption (default: 1 second)
/// - Configurable memory limits (default: unlimited)
/// - Type conversion between i32/i64 with bounds checking and warnings
/// - Function discovery and caching
/// - Proper resource cleanup via Drop trait
///
/// Fork support requires the host-side quiescence and `pthread_atfork` preconditions
/// documented in [`fork_generation`].
///
/// # Example
///
/// ```no_run
/// # use pecos_wasm::{WasmForeignObject, ForeignObject};
/// // Create with defaults (1-second timeout, unlimited memory)
/// let mut wasm = WasmForeignObject::new("math.wasm").unwrap();
/// wasm.init().unwrap();
///
/// // Or create with custom timeout (5 seconds)
/// let mut wasm = WasmForeignObject::with_timeout("math.wasm", 5.0).unwrap();
///
/// // Or create with custom memory limit (10 MB)
/// let mut wasm = WasmForeignObject::with_limits("math.wasm", 5.0, Some(10 * 1024 * 1024)).unwrap();
///
/// // Execute a function
/// let result = wasm.exec("add", &[5, 3]).unwrap();
/// assert_eq!(result, vec![8]);
/// ```
#[derive(Debug)]
pub struct WasmForeignObject {
    /// WebAssembly binary
    wasm_bytes: Arc<[u8]>,
    /// Wasmtime module
    module: Module,
    /// Wasmtime store (thread-safe)
    store: RwLock<Store<StoreContext>>,
    /// Wasmtime instance (thread-safe)
    instance: RwLock<Option<Instance>>,
    /// Handles belong to the current instance; cleared only after replacement succeeds.
    functions: HashMap<String, CachedFunction>,
    /// Immutable module metadata also supports the public shared-reference queries.
    exports: Arc<[Export]>,
    #[cfg(test)]
    checked_fallback_calls: usize,
    /// Reused numeric argument/result storage, never containing references.
    raw: Vec<ValRaw>,
    /// Timeout in seconds for WASM execution (default: 1.0 second)
    timeout_seconds: f64,
    /// Maximum memory size in bytes per linear memory (default: None = unlimited)
    memory_size: Option<usize>,
}

impl WasmForeignObject {
    fn arm_deadline(
        store: &mut Store<StoreContext>,
        timeout_seconds: f64,
    ) -> Result<(), PecosError> {
        ensure_ticker()?;
        store.set_epoch_deadline(Self::calculate_max_ticks(timeout_seconds));
        Ok(())
    }

    /// Calculate maximum ticks from timeout in seconds
    ///
    /// # Parameters
    ///
    /// * `timeout_seconds` - Timeout in seconds
    ///
    /// # Returns
    ///
    /// Number of ticks before timeout
    fn calculate_max_ticks(timeout_seconds: f64) -> u64 {
        // Ensure non-negative timeout (clamp to 0 for negative values)
        // Round positive limits up so a sub-tick timeout never becomes an
        // already-expired deadline. Leave headroom for Wasmtime's epoch + delta.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let ticks = (timeout_seconds.max(0.0) * 1000.0 / WASM_EXECUTION_TICK_LENGTH_MS as f64)
            .ceil()
            .min((u64::MAX / 10) as f64) as u64;
        ticks
    }

    /// Create a new WebAssembly foreign object from a file with default timeout
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object with 1-second timeout
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        Self::with_timeout(path, DEFAULT_TIMEOUT_SECONDS)
    }

    /// Create a new WebAssembly foreign object from a file with custom timeout
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn with_timeout<P: AsRef<Path>>(path: P, timeout_seconds: f64) -> Result<Self, PecosError> {
        Self::with_limits(path, timeout_seconds, None)
    }

    /// Create a new WebAssembly foreign object from a file with custom limits
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    /// * `memory_size` - Optional maximum memory size in bytes per linear memory (None = unlimited)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn with_limits<P: AsRef<Path>>(
        path: P,
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<Self, PecosError> {
        // Read the WebAssembly file
        let wasm_bytes = std::fs::read(path)
            .map_err(|e| PecosError::Input(format!("Failed to read WebAssembly file: {e}")))?;

        Self::from_bytes_with_limits(&wasm_bytes, timeout_seconds, memory_size)
    }

    /// Create a new WebAssembly foreign object from bytes with default timeout
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object with 1-second timeout
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes(wasm_bytes: &[u8]) -> Result<Self, PecosError> {
        Self::from_bytes_with_limits(wasm_bytes, DEFAULT_TIMEOUT_SECONDS, None)
    }

    /// Create a new WebAssembly foreign object from bytes with custom timeout
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes_with_timeout(
        wasm_bytes: &[u8],
        timeout_seconds: f64,
    ) -> Result<Self, PecosError> {
        Self::from_bytes_with_limits(wasm_bytes, timeout_seconds, None)
    }

    /// Create a new WebAssembly foreign object from bytes with custom limits
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    /// * `memory_size` - Optional maximum memory size in bytes per linear memory (None = unlimited)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes_with_limits(
        wasm_bytes: &[u8],
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<Self, PecosError> {
        CompiledWasmModule::new(wasm_bytes)?.instantiate_with_limits(timeout_seconds, memory_size)
    }

    fn legacy_arguments(raw: &mut Vec<ValRaw>, signature: &Signature, args: &[i64]) {
        raw.resize(
            signature.params.len().max(signature.results.len()),
            ValRaw::i64(0),
        );
        // Preserve legacy conversion warnings, including for extra arguments,
        // before arming the deadline and reporting arity/type errors.
        for (i, a) in args.iter().enumerate() {
            let value = match signature.params.get(i) {
                Some(Lane::I64) => ValRaw::i64(*a),
                Some(Lane::I32) | None => {
                    let value = if *a > i64::from(i32::MAX) {
                        warn!("Argument value {a} exceeds i32::MAX, clamping to i32::MAX");
                        i32::MAX
                    } else if *a < i64::from(i32::MIN) {
                        warn!("Argument value {a} is less than i32::MIN, clamping to i32::MIN");
                        i32::MIN
                    } else {
                        i32::try_from(*a).expect("argument is within the checked i32 bounds")
                    };
                    ValRaw::i32(value)
                }
                _ => {
                    warn!("Unexpected parameter type for argument {i}, defaulting to i32");
                    ValRaw::i32(i32::try_from(*a).unwrap_or(i32::MAX))
                }
            };
            if let Some(slot) = raw.get_mut(i) {
                *slot = value;
            }
        }
    }

    fn cache_function(&mut self, name: &str) -> Result<CachedFunction, PecosError> {
        // Func and the export index are Copy: returning them lets the hot path
        // use one borrowed-key lookup without allocating an owned entry key.
        if let Some(function) = self.functions.get(name) {
            return Ok(*function);
        }
        let instance =
            self.instance.get_mut().as_ref().ok_or_else(|| {
                PecosError::Resource("WebAssembly instance not created".to_string())
            })?;
        let missing = || PecosError::Resource(format!("WebAssembly function '{name}' not found"));
        let export = self
            .exports
            .iter()
            .position(|export| export.name == name && export.signature.is_some())
            .ok_or_else(missing)?;
        let func = instance
            .get_func(self.store.get_mut(), name)
            .ok_or_else(missing)?;
        let function = CachedFunction { func, export };
        self.functions.insert(name.into(), function);
        Ok(function)
    }

    fn function_names(&self) -> impl Iterator<Item = &str> {
        self.exports
            .iter()
            .filter(|export| export.signature.is_some())
            .map(|export| export.name.as_str())
    }

    #[cfg(test)]
    pub(crate) fn shares_export_metadata_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.exports, &other.exports)
    }

    #[cfg(test)]
    pub(crate) fn checked_fallback_calls(&self) -> usize {
        self.checked_fallback_calls
    }

    /// Validate the scalar i64 ABI used by public Guppy WASM calls.
    ///
    /// # Errors
    /// Returns an error for a missing export, differing arity, or non-i64 types.
    pub fn validate_i64_signature(
        &self,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<(), PecosError> {
        // The runtime validates on every call: try the cached export position first.
        let export = match self.functions.get(name) {
            Some(function) => self.exports.get(function.export),
            None => self.exports.iter().find(|export| export.name == name),
        };
        let signature = export
            .and_then(|export| export.signature.as_ref())
            .ok_or_else(|| PecosError::Input(format!("Missing WASM function export: {name}")))?;
        if signature.params.len() != inputs
            || signature.results.len() != outputs
            || !signature
                .params
                .iter()
                .chain(signature.results.iter())
                .all(|ty| *ty == Lane::I64)
        {
            return Err(PecosError::Input(format!(
                "WASM i64 signature mismatch for {name}: export ({}) -> ({}); \
                 expected {inputs} i64 inputs, {outputs} i64 outputs",
                format_lanes(&signature.params, Lane::name),
                format_lanes(&signature.results, Lane::name)
            )));
        }
        Ok(())
    }

    /// Execute a strictly typed scalar call using one raw 64-bit slot per value.
    ///
    /// Float slots contain IEEE 754 bits, not integer conversions. This keeps
    /// the Guppy interface separate from the legacy integer/clamping API.
    ///
    /// # Errors
    /// Rejects missing exports, argument/signature mismatches, traps and timeouts.
    pub fn exec_scalar_bits(
        &mut self,
        name: &str,
        inputs: &[WasmScalarType],
        outputs: &[WasmScalarType],
        arguments: &[i64],
    ) -> Result<Vec<i64>, PecosError> {
        let function = self.cache_function(name)?;
        let signature = function.signature(&self.exports);
        let matches = |expected: &WasmScalarType, actual: &Lane| {
            matches!(
                (expected, actual),
                (WasmScalarType::I64, Lane::I64)
                    | (WasmScalarType::I32, Lane::I32)
                    | (WasmScalarType::F64, Lane::F64)
                    | (WasmScalarType::IntegerAuto, Lane::I32 | Lane::I64)
            )
        };
        if arguments.len() != inputs.len()
            || signature.params.len() != inputs.len()
            || signature.results.len() != outputs.len()
            || !inputs
                .iter()
                .zip(signature.params.iter())
                .all(|(a, b)| matches(a, b))
            || !outputs
                .iter()
                .zip(signature.results.iter())
                .all(|(a, b)| matches(a, b))
        {
            return Err(PecosError::Input(format!(
                "WASM scalar signature mismatch for {name}: {} argument{}; export ({}) -> ({}); \
                 expected ({}) -> ({})",
                arguments.len(),
                if arguments.len() == 1 { "" } else { "s" },
                format_lanes(&signature.params, Lane::name),
                format_lanes(&signature.results, Lane::name),
                format_lanes(inputs, WasmScalarType::name),
                format_lanes(outputs, WasmScalarType::name)
            )));
        }
        self.raw.resize(
            signature.params.len().max(signature.results.len()),
            ValRaw::i64(0),
        );
        for ((slot, ty), bits) in self
            .raw
            .iter_mut()
            .zip(signature.params.iter())
            .zip(arguments)
        {
            *slot = match ty {
                Lane::I64 => ValRaw::i64(*bits),
                Lane::I32 => {
                    let [a, b, c, d, ..] = bits.to_le_bytes();
                    ValRaw::i32(i32::from_le_bytes([a, b, c, d]))
                }
                Lane::F64 => ValRaw::f64(u64::from_ne_bytes(bits.to_ne_bytes())),
                _ => unreachable!("scalar signature validation accepts only i32, i64 and f64"),
            };
        }
        let store = self.store.get_mut();
        Self::arm_deadline(store, self.timeout_seconds)?;
        // SAFETY: raw is a valid slice sized for max(params, results). Every
        // parameter was written with its exact declared type; validation excludes
        // references (and other non-scalar types) from parameters and results.
        // This crate never invokes Wasmtime call_async or fibers, and instances
        // have no imports (Instance::new uses &[]), so there is no reentrancy.
        // The disabled Wasmtime async feature additionally enforces that usage.
        let result = unsafe { function.func.call_unchecked(store, self.raw.as_mut_slice()) };
        result.map_err(|error| self.execution_error(name, &error))?;
        Ok(signature
            .results
            .iter()
            .zip(&self.raw)
            .map(|(ty, raw)| match ty {
                Lane::I32 => i64::from(raw.get_i32()),
                Lane::I64 => raw.get_i64(),
                Lane::F64 => i64::from_ne_bytes(raw.get_f64().to_ne_bytes()),
                _ => unreachable!("scalar signature validation accepts only i32, i64 and f64"),
            })
            .collect())
    }

    fn execution_error(&self, name: &str, error: &wasmtime::Error) -> PecosError {
        if let Some(trap) = error.downcast_ref::<Trap>() {
            let trap_str = trap.to_string();
            if trap_str.contains("interrupt") {
                return PecosError::Processing(format!(
                    "WebAssembly execution timed out after {}s (possible infinite loop in '{name}')",
                    self.timeout_seconds
                ));
            }
            if trap_str.contains("integer divide by zero") {
                return PecosError::RuntimeDivisionByZero;
            }
            return PecosError::Generic(format!("WebAssembly function '{name}' trapped: {trap}"));
        }
        PecosError::Generic(format!("WebAssembly function '{name}' failed: {error}"))
    }

    /// Call before each shot to reset variables (if function exists)
    ///
    /// This is a convenience method that calls the `shot_reinit` function in the
    /// WebAssembly module if it exists. If the function doesn't exist, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the `shot_reinit` function exists but execution fails
    pub fn shot_reinit(&mut self) -> Result<(), PecosError> {
        if self.has_func("shot_reinit") {
            self.exec("shot_reinit", &[])?;
        }
        Ok(())
    }

    /// Check if a function exists in the WebAssembly module (zero-allocation)
    ///
    /// This is more efficient than `get_funcs().contains()` for existence checks
    /// as it doesn't allocate or clone the function list.
    ///
    /// # Parameters
    ///
    /// * `func_name` - Name of the function to check
    ///
    /// # Returns
    ///
    /// `true` if the function exists, `false` otherwise
    pub fn has_func(&self, func_name: &str) -> bool {
        self.function_names().any(|name| name == func_name)
    }

    /// Resolve a zero-based export-list position to a function name.
    ///
    /// All exports occupy positions, including memories, tables and globals.
    /// This is the order used by Guppy's `@wasm(index)`, not the WebAssembly
    /// internal function-index space.
    ///
    /// # Errors
    /// Rejects out-of-range positions and exports that are not functions.
    pub fn exported_function_name(&self, index: u64) -> Result<String, PecosError> {
        let export = usize::try_from(index)
            .ok()
            .and_then(|index| self.exports.get(index))
            .ok_or_else(|| {
                PecosError::Input(format!("WASM export index {index} is out of range"))
            })?;
        if export.signature.is_none() {
            return Err(PecosError::Input(format!(
                "WASM export index {index} is not a function ({})",
                export.name
            )));
        }
        Ok(export.name.clone())
    }

    /// Get the WebAssembly binary bytes
    ///
    /// This is useful for serialization and cloning.
    ///
    /// # Returns
    ///
    /// A reference to the WebAssembly binary bytes
    #[must_use]
    pub fn wasm_bytes(&self) -> &[u8] {
        &self.wasm_bytes
    }

    /// Get the configured timeout in seconds
    ///
    /// # Returns
    ///
    /// The timeout in seconds for WASM execution
    #[must_use]
    pub fn timeout_seconds(&self) -> f64 {
        self.timeout_seconds
    }

    /// Get the configured memory size limit
    ///
    /// # Returns
    ///
    /// The memory size limit in bytes per linear memory (None = unlimited)
    #[must_use]
    pub fn memory_size(&self) -> Option<usize> {
        self.memory_size
    }
}

impl ForeignObject for WasmForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        // Create a new instance from the same bytes with the same timeout and memory limit
        let mut result =
            Self::from_bytes_with_limits(&self.wasm_bytes, self.timeout_seconds, self.memory_size)
                .expect("Failed to clone WasmForeignObject");

        // Initialize it the same way
        if self.instance.read().is_some() {
            let _ = result.new_instance();
        }

        Box::new(result)
    }

    fn init(&mut self) -> Result<(), PecosError> {
        // Create a new instance
        self.new_instance()?;

        // Check if the init function exists
        if !self.has_func("init") {
            return Err(PecosError::Input(
                "WebAssembly module must contain an 'init' function".to_string(),
            ));
        }

        // Call the init function
        self.exec("init", &[])?;

        Ok(())
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        let mut store = self.store.write();

        // Instantiation can execute enough Wasm work to observe an epoch
        // interrupt. Arm the deadline before Instance::new, just as exec does
        // before calling an exported function.
        Self::arm_deadline(&mut store, self.timeout_seconds)?;

        // Create a new instance
        let instance = Instance::new(&mut *store, &self.module, &[]).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly instance: {e}"))
        })?;

        // Store the instance
        *self.instance.write() = Some(instance);
        self.functions.clear();

        Ok(())
    }

    fn get_funcs(&self) -> Vec<String> {
        self.function_names().map(str::to_owned).collect()
    }

    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
        debug!("Executing WebAssembly function '{func_name}' with args {args:?}");
        let function = self.cache_function(func_name)?;
        let signature = function.signature(&self.exports);
        Self::legacy_arguments(&mut self.raw, signature, args);
        let store = self.store.get_mut();
        Self::arm_deadline(store, self.timeout_seconds)?;
        if args.len() != signature.params.len() {
            return Err(PecosError::Generic(format!(
                "WebAssembly function '{func_name}' failed: expected {} arguments, got {}",
                signature.params.len(),
                args.len()
            )));
        }
        if !signature
            .params
            .iter()
            .all(|ty| matches!(ty, Lane::I32 | Lane::I64))
        {
            return Err(PecosError::Generic(format!(
                "WebAssembly function '{func_name}' failed: argument type mismatch"
            )));
        }
        if signature.results.contains(&Lane::Other) {
            // References and vectors retain Wasmtime's checked marshalling/GC path.
            let values: Vec<_> = signature
                .params
                .iter()
                .zip(&self.raw)
                .map(|(ty, raw)| match ty {
                    Lane::I32 => Val::I32(raw.get_i32()),
                    Lane::I64 => Val::I64(raw.get_i64()),
                    _ => unreachable!("legacy parameter validation accepts only i32 and i64"),
                })
                .collect();
            let mut results = vec![Val::I32(0); signature.results.len()];
            #[cfg(test)]
            {
                self.checked_fallback_calls += 1;
            }
            function
                .func
                .call(store, &values, &mut results)
                .map_err(|error| self.execution_error(func_name, &error))?;
            return Ok(results
                .iter()
                .map(|value| match value {
                    Val::I32(value) => i64::from(*value),
                    Val::I64(value) => *value,
                    _ => {
                        warn!("Unexpected result type from WebAssembly function");
                        0
                    }
                })
                .collect());
        }
        // SAFETY: raw is a valid slice sized for max(params, results). Arity and
        // types were checked, and each parameter uses its exact i32/i64 constructor.
        // Results are all numeric, so no references reach this path. This crate
        // never invokes Wasmtime call_async or fibers, and instances have no
        // imports (Instance::new uses &[]), so there is no reentrancy. The disabled
        // Wasmtime async feature additionally enforces that usage.
        let result = unsafe { function.func.call_unchecked(store, self.raw.as_mut_slice()) };
        result.map_err(|error| self.execution_error(func_name, &error))?;
        if signature.results.is_empty() {
            return Ok(vec![0]);
        }
        Ok(signature
            .results
            .iter()
            .zip(&self.raw)
            .map(|(ty, raw)| match ty {
                Lane::I32 => i64::from(raw.get_i32()),
                Lane::I64 => raw.get_i64(),
                _ => {
                    warn!("Unexpected result type from WebAssembly function");
                    0
                }
            })
            .collect())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::WasmForeignObject;

    #[test]
    fn ticker_spawns_once() {
        use crate::ForeignObject;
        let modules: Vec<_> = (0..4)
            .map(|value| {
                let source =
                    format!("(module (func (export \"value\") (result i64) i64.const {value}))");
                super::CompiledWasmModule::new(source.as_bytes()).unwrap()
            })
            .collect();
        for (value, module) in modules.iter().enumerate() {
            assert!(wasmtime::Engine::same(
                modules[0].module.engine(),
                module.module.engine()
            ));
            let mut object = module.instantiate().unwrap();
            assert_eq!(
                object.exec("value", &[]).unwrap(),
                [i64::try_from(value).unwrap()]
            );
        }
        assert_eq!(super::TICKER_SPAWNS.load(super::Ordering::Acquire), 1);
    }

    #[test]
    fn managed_roots_survive_collection_and_stores_are_independent() {
        use crate::ForeignObject;
        let module = super::CompiledWasmModule::new(
            br#"(module
            (type $node (struct (field (mut i64)) (field (mut (ref null $node)))))
            (global $live (mut (ref null $node)) (ref.null $node))
            (func (export "init")
                i64.const 42 ref.null $node struct.new $node global.set $live)
            (func (export "cycle") (local $node (ref null $node))
                i64.const 9 ref.null $node struct.new $node local.tee $node
                local.get $node struct.set $node 1)
            (func (export "get") (result i64) global.get $live struct.get $node 0)
            (func (export "increment")
                global.get $live global.get $live struct.get $node 0
                i64.const 1 i64.add struct.set $node 0)
            (func (export "reference") (result (ref null $node)) global.get $live)
            (func (export "consume") (param (ref null $node))))"#,
        )
        .unwrap();
        let mut first = module.instantiate().unwrap();
        let mut second = module.instantiate().unwrap();
        first.exec("init", &[]).unwrap();
        second.exec("init", &[]).unwrap();
        for expected in 43..51 {
            first.exec("increment", &[]).unwrap();
            for _ in 0..1000 {
                first.exec("cycle", &[]).unwrap();
            }
            // Explicitly invoke collection: allocation alone does not prove
            // that a moving collector preserves live module roots.
            first.store.write().gc(None).unwrap();
            assert_eq!(first.exec("get", &[]).unwrap(), [expected]);
            assert_eq!(second.exec("get", &[]).unwrap(), [42]);
        }
        assert!(first.validate_i64_signature("reference", 0, 1).is_err());
        assert!(first.validate_i64_signature("consume", 1, 0).is_err());
        first.teardown();
        second.store.write().gc(None).unwrap();
        assert_eq!(second.exec("get", &[]).unwrap(), [42]);
    }

    #[test]
    fn positive_deadlines_round_up_and_leave_epoch_headroom() {
        for (seconds, expected) in [
            (f64::from_bits(1), 1),
            (0.001, 1),
            (0.01, 1),
            (0.011, 2),
            (1.0, 100),
        ] {
            assert_eq!(WasmForeignObject::calculate_max_ticks(seconds), expected);
        }
        let largest = WasmForeignObject::calculate_max_ticks(f64::MAX);
        assert!(largest > 0 && largest < u64::MAX / 2);
        assert_eq!(WasmForeignObject::calculate_max_ticks(0.0), 0);
    }
}
