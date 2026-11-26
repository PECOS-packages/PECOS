//! Public WASM ABI v1. Implemented from the public tket.wasm operations.
//!
//! Mutable state is local to the calling worker thread. Shot boundaries reset it
//! by default; explicit worker scope retains clean instances across shots.
//! A bounded per-thread cache retains immutable compiled code between shots.
//! C entry points return a status; they never unwind or transfer control while
//! Rust frames own resources. The generated caller handles errors afterwards.
//!
//! Fork is supported through the C ABI. A child discards all of the forking
//! thread's per-thread runtime state: registrations, function handles, contexts,
//! retained worker-scope instances and pending results. It must start a new shot
//! and register again; inherited pending results fail as unknown contexts.
//! Compiled code is kept, and timeouts and traps keep working. Fork requires
//! that no other thread is performing a Wasmtime host-side operation (compiling,
//! instantiating, creating the Engine, dropping stores or modules, or handling a
//! trap). A worker parked in guest code is fine. Forks that skip `pthread_atfork`
//! handlers (raw `clone`/`clone3`, `vfork`, or glibc `_Fork`) are unsupported.
//! On Apple targets the runtime's own waits survive fork, but std thread parking
//! does not: a forking thread that parked before the fork (for example in a
//! contended std lock or an `mpsc` receive) can crash if it parks again in the child.

use pecos_wasm::{CompiledWasmModule, ForeignObject, WasmForeignObject, WasmScalarType};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};

mod async_pool;
#[cfg(test)]
mod test_support;
mod tls;

type Result<T> = std::result::Result<T, String>;

type ModuleInstances = BTreeMap<u64, WasmForeignObject>;
type AsyncReply = (ModuleInstances, Result<Vec<i64>>);

enum Pending {
    Ready(Vec<i64>),
    Async(async_pool::ReplyReceiver<AsyncReply>),
}

#[derive(Default)]
struct Context {
    leased: bool,
    modules: ModuleInstances,
    pending: Option<Pending>,
}

#[derive(Clone, PartialEq, Eq)]
enum FunctionTarget {
    Name(String),
    ExportIndex(u64),
}

#[derive(Clone, PartialEq, Eq)]
struct Function {
    module: u64,
    target: FunctionTarget,
    inputs: usize,
    outputs: usize,
    scalar_types: Option<(Vec<WasmScalarType>, Vec<WasmScalarType>)>,
}

struct ModuleConfiguration {
    bytes: Vec<u8>,
    timeout_seconds: f64,
    auto_init: bool,
}

#[derive(Default)]
struct State {
    modules: BTreeMap<u64, ModuleConfiguration>,
    functions: Vec<Function>,
    contexts: BTreeMap<u64, Context>,
    worker_key: Option<[u8; 32]>,
    shot_active: bool,
    async_workers: Option<usize>,
}

// These bounds limit cache retention, not total process memory or compiled code
// size. Active instances can outlive eviction. Large inputs still run uncached.
const CACHE_ENTRIES: usize = 8;
const CACHE_SOURCE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct CompilationCache {
    entries: VecDeque<CompiledWasmModule>,
    source_bytes: usize,
    #[cfg(test)]
    compilations: usize,
}

impl CompilationCache {
    fn get(&mut self, bytes: &[u8]) -> Result<CompiledWasmModule> {
        self.get_with_limits(bytes, CACHE_ENTRIES, CACHE_SOURCE_BYTES)
    }

    fn get_with_limits(
        &mut self,
        bytes: &[u8],
        max_entries: usize,
        max_bytes: usize,
    ) -> Result<CompiledWasmModule> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.wasm_bytes() == bytes)
        {
            let entry = self.entries.remove(index).expect("existing cache index");
            self.entries.push_back(entry.clone());
            return Ok(entry);
        }
        let compiled = CompiledWasmModule::new(bytes).map_err(|e| e.to_string())?;
        #[cfg(test)]
        {
            self.compilations += 1;
        }
        if max_entries > 0 && bytes.len() <= max_bytes {
            while self.entries.len() >= max_entries || self.source_bytes + bytes.len() > max_bytes {
                let entry = self.entries.pop_front().expect("cache over capacity");
                self.source_bytes -= entry.wasm_bytes().len();
            }
            self.source_bytes += bytes.len();
            self.entries.push_back(compiled.clone());
        }
        Ok(compiled)
    }
}

thread_local! {
    // No destructor: remains readable while STATE, COMPILED and POOL are destroyed.
    #[cfg(unix)]
    static GENERATION: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
    static STATE: tls::Value<RefCell<State>> = tls::Value::new(RefCell::default());
    // Immutable code survives shot reset; stores and instances never do.
    static COMPILED: tls::Value<RefCell<CompilationCache>> = tls::Value::new(RefCell::default());
    // Idle workers intentionally survive shot resets until this execution thread
    // exits. The Windows process-exit guard also covers their synchronization.
    static POOL: tls::Value<RefCell<Option<async_pool::Pool>>> = const { tls::Value::new(RefCell::new(None)) };
    // QIS panic uses a length-prefixed byte string (at most 255 bytes).
    static ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]);
}

#[cfg(unix)]
fn inherited() -> bool {
    GENERATION.with(|generation| {
        let recorded = generation.get();
        recorded != u64::MAX && recorded != pecos_wasm::fork_generation()
    })
}

// Access only the calling thread's executor; workers and destructors never use TLS.
fn with_pool<T>(operation: impl FnOnce(Option<&async_pool::Pool>) -> T) -> T {
    POOL.with(|pool| operation(pool.borrow().as_ref()))
}

fn set_error(message: &str) {
    ERROR.with(|error| {
        let bytes = message.as_bytes();
        let length = message.floor_char_boundary(bytes.len().min(255));
        let mut error = error.borrow_mut();
        error.clear();
        error.push(u8::try_from(length).expect("bounded error message"));
        error.extend_from_slice(&bytes[..length]);
    });
}

/// Execute an operation without allowing a Rust panic to cross the ABI.
pub fn boundary<T: Copy>(failure: T, operation: impl FnOnce() -> Result<T>) -> T {
    #[cfg(unix)]
    {
        if inherited() {
            // Sibling threads vanished at fork. Even dropping their channels or
            // instances may take locks they held, so deliberately leak this state.
            POOL.with(|pool| std::mem::forget(pool.borrow_mut().take()));
            STATE.with(|state| std::mem::forget(std::mem::take(&mut *state.borrow_mut())));
            // COMPILED remains valid on the shared Engine and is reused normally.
        }
        GENERATION.with(|generation| generation.set(pecos_wasm::fork_generation()));
    }
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            let _ = reset();
            set_error(&message);
            failure
        }
        Err(_) => {
            let _ = reset();
            set_error("WASM runtime internal panic");
            failure
        }
    }
}

/// Start or finish a shot, releasing all module instances and pending results.
/// Waits for async calls already running; WASM phase deadlines bound that wait.
///
/// # Errors
/// This operation always succeeds; the result matches the status-returning ABI.
pub fn reset() -> Result<i32> {
    STATE.with(|state| *state.borrow_mut() = State::default());
    with_pool(|pool| {
        if let Some(pool) = pool {
            pool.cancel();
        }
    });
    Ok(0)
}

/// Begin a worker-scoped shot, retaining instances only after a clean prior shot
/// with identical program and module configuration. The key is not a batch ID;
/// callers control the worker lifetime and may explicitly reset it.
///
/// # Errors
/// This operation always succeeds for a fixed-size key.
pub fn begin_worker_shot(key: [u8; 32]) -> Result<i32> {
    let replaced = STATE.with(|state| {
        let mut state = state.borrow_mut();
        let replaced = state.worker_key != Some(key) || state.shot_active;
        if replaced {
            *state = State::default();
        }
        state.worker_key = Some(key);
        state.shot_active = true;
        state.modules.clear();
        state.functions.clear();
        replaced
    });
    if replaced {
        with_pool(|pool| {
            if let Some(pool) = pool {
                pool.cancel();
            }
        });
    }
    Ok(0)
}

/// Finish a worker-scoped shot while retaining its mutable WASM instances.
///
/// # Errors
/// Rejects inactive shots, leased contexts, and unread results. ABI callers reset
/// all state on failure; an unfinished shot is also discarded by the next begin.
pub fn finish_worker_shot() -> Result<i32> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.worker_key.is_none() || !state.shot_active {
            return Err("No active WASM worker shot".into());
        }
        if state
            .contexts
            .values()
            .any(|context| context.leased || context.pending.is_some())
        {
            return Err("WASM worker shot has leased contexts or unread results".into());
        }
        state.shot_active = false;
        Ok(0)
    })
}

/// Register caller-supplied module bytes for this shot.
///
/// # Errors
/// Returns an error if the module identifier is already registered.
pub fn register(module: u64, bytes: &[u8]) -> Result<i32> {
    register_with_timeout(module, bytes, 1.0)
}

/// Register module bytes with a per-start/init/call execution timeout in seconds.
/// Compilation time is excluded. The epoch clock has approximately 10 ms resolution.
///
/// # Errors
/// Rejects nonpositive/nonfinite timeouts and duplicate module identifiers.
pub fn register_with_timeout(module: u64, bytes: &[u8], timeout_seconds: f64) -> Result<i32> {
    register_with_options(module, bytes, timeout_seconds, 0)
}

/// Suppress the optional exported `init` call; WASM start still runs.
pub const SKIP_AUTO_INIT: u32 = 1;

/// Register module bytes with execution limits and initialization flags.
///
/// # Errors
/// Rejects unknown flags, invalid timeouts and duplicate module identifiers.
pub fn register_with_options(
    module: u64,
    bytes: &[u8],
    timeout_seconds: f64,
    flags: u32,
) -> Result<i32> {
    if flags & !SKIP_AUTO_INIT != 0 {
        return Err(format!(
            "Unknown WASM registration flags {flags:#x}; this runtime supports only \
             SKIP_AUTO_INIT ({SKIP_AUTO_INIT:#x}): upgrade the native WASM runtime to \
             match the compiler that generated the program"
        ));
    }
    if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 {
        return Err(format!(
            "WASM timeout must be a positive finite number of seconds; got {timeout_seconds}"
        ));
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.modules.contains_key(&module) {
            return Err(format!("Duplicate WASM module {module}"));
        }
        state.modules.insert(
            module,
            ModuleConfiguration {
                bytes: bytes.to_vec(),
                timeout_seconds,
                auto_init: flags & SKIP_AUTO_INIT == 0,
            },
        );
        Ok(0)
    })
}

/// Acquire a context. Zero represents None; positive handles represent Some.
///
/// # Errors
/// Returns an error if the index cannot be represented as a nonzero handle.
pub fn acquire(index: u64) -> Result<u64> {
    let handle = index.checked_add(1).ok_or("WASM context index overflow")?;
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let context = state.contexts.entry(handle).or_default();
        if context.leased {
            return Ok(0);
        }
        context.leased = true;
        Ok(handle)
    })
}

/// Release a context lease, retaining module state until the end of the shot.
///
/// # Errors
/// Rejects unknown or unleased contexts and contexts with unread results.
pub fn release(handle: u64) -> Result<i32> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let context = state
            .contexts
            .get_mut(&handle)
            .ok_or("Unknown WASM context")?;
        if !context.leased || context.pending.is_some() {
            return Err("WASM context is not leased or has an unread result".into());
        }
        context.leased = false;
        Ok(0)
    })
}

/// Bind an export and its expected scalar i64 signature.
///
/// # Errors
/// Rejects unknown modules, multiple results, and unrepresentable handles.
pub fn lookup(module: u64, name: &str, inputs: usize, outputs: usize) -> Result<u64> {
    if outputs > 1 {
        return Err("WASM ABI v1 supports at most one result".into());
    }
    lookup_function(Function {
        module,
        target: FunctionTarget::Name(name.into()),
        inputs,
        outputs,
        scalar_types: None,
    })
}

/// Bind an export with an explicit scalar signature.
///
/// # Errors
/// Rejects unknown modules, multiple results, and unrepresentable handles.
pub fn lookup_scalar(
    module: u64,
    name: &str,
    inputs: &[WasmScalarType],
    outputs: &[WasmScalarType],
) -> Result<u64> {
    lookup_typed(module, FunctionTarget::Name(name.into()), inputs, outputs)
}

/// Bind a function at a zero-based position in the module's export list.
///
/// This matches Guppy's `@wasm(index)`: non-function exports occupy positions
/// too. The index and exact signature are checked against the supplied module
/// on the first call, when its instance is lazily created.
///
/// # Errors
/// Rejects unknown modules, multiple results, and unrepresentable handles.
pub fn lookup_by_id(
    module: u64,
    index: u64,
    inputs: &[WasmScalarType],
    outputs: &[WasmScalarType],
) -> Result<u64> {
    lookup_typed(module, FunctionTarget::ExportIndex(index), inputs, outputs)
}

fn lookup_typed(
    module: u64,
    target: FunctionTarget,
    inputs: &[WasmScalarType],
    outputs: &[WasmScalarType],
) -> Result<u64> {
    if outputs.len() > 1 {
        return Err("WASM scalar ABI supports at most one result".into());
    }
    let scalar_types = inputs
        .iter()
        .chain(outputs)
        .any(|ty| *ty != WasmScalarType::I64)
        .then(|| (inputs.to_vec(), outputs.to_vec()));
    lookup_function(Function {
        module,
        target,
        inputs: inputs.len(),
        outputs: outputs.len(),
        scalar_types,
    })
}

fn lookup_function(function: Function) -> Result<u64> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if !state.modules.contains_key(&function.module) {
            return Err(format!("Unknown WASM module {}", function.module));
        }
        if let Some(index) = state
            .functions
            .iter()
            .position(|existing| existing == &function)
        {
            return Ok(u64::try_from(index).map_err(|e| e.to_string())? + 1);
        }
        let handle = u64::try_from(state.functions.len()).map_err(|e| e.to_string())? + 1;
        state.functions.push(function);
        Ok(handle)
    })
}

/// Decode the stable scalar type tags used by the typed lookup ABI.
///
/// # Errors
/// Rejects unknown type tags rather than interpreting them as integers.
pub fn scalar_types(tags: &[u8]) -> Result<Vec<WasmScalarType>> {
    tags.iter().map(|tag| scalar_type(*tag, false)).collect()
}

/// Decode v3 tags, including tag 3 for per-lane native integer width selection.
///
/// # Errors
/// Rejects unknown type tags. Existing v2 decoding still rejects tag 3.
pub fn scalar_types_auto(tags: &[u8]) -> Result<Vec<WasmScalarType>> {
    tags.iter().map(|tag| scalar_type(*tag, true)).collect()
}

fn scalar_type(tag: u8, allow_auto: bool) -> Result<WasmScalarType> {
    match tag {
        0 => Ok(WasmScalarType::I64),
        1 => Ok(WasmScalarType::F64),
        2 => Ok(WasmScalarType::I32),
        3 if allow_auto => Ok(WasmScalarType::IntegerAuto),
        _ => Err(format!("Unknown WASM scalar type tag: {tag}")),
    }
}

/// Execute synchronously and retain the result until `read_result` consumes it.
///
/// # Errors
/// Rejects invalid handles, leases, arguments, signatures, module instantiation,
/// and execution failures including traps and timeouts.
pub fn call(context: u64, function: u64, arguments: &[i64]) -> Result<i32> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let State {
            modules,
            functions,
            contexts,
            ..
        } = &mut *state;
        let index = usize::try_from(function.checked_sub(1).ok_or("Invalid WASM function")?)
            .map_err(|e| e.to_string())?;
        let function = functions.get(index).ok_or("Unknown WASM function")?;
        if arguments.len() != function.inputs {
            return Err("WASM argument count mismatch".into());
        }
        let context = contexts.get_mut(&context).ok_or("Unknown WASM context")?;
        if !context.leased || context.pending.is_some() {
            return Err("WASM context is not leased or has an unread result".into());
        }
        if let std::collections::btree_map::Entry::Vacant(entry) =
            context.modules.entry(function.module)
        {
            let configuration = modules.get(&function.module).ok_or("Unknown WASM module")?;
            let compiled = COMPILED.with(|cache| cache.borrow_mut().get(&configuration.bytes))?;
            let mut object = compiled
                .instantiate_with_limits(configuration.timeout_seconds, None)
                .map_err(|e| e.to_string())?;
            // init is optional for Guppy modules (unlike the legacy QASM contract).
            if configuration.auto_init && object.has_func("init") {
                object
                    .validate_i64_signature("init", 0, 0)
                    .map_err(|e| e.to_string())?;
                object.exec("init", &[]).map_err(|e| e.to_string())?;
            }
            entry.insert(object);
        }
        let object = context
            .modules
            .get_mut(&function.module)
            .ok_or("Unknown WASM instance")?;
        let values = execute(object, function, arguments)?;
        context.pending = Some(Pending::Ready(values));
        Ok(0)
    })
}

fn execute(
    object: &mut WasmForeignObject,
    function: &Function,
    arguments: &[i64],
) -> Result<Vec<i64>> {
    let name = match &function.target {
        FunctionTarget::Name(name) => Cow::Borrowed(name.as_str()),
        FunctionTarget::ExportIndex(index) => Cow::Owned(
            object
                .exported_function_name(*index)
                .map_err(|e| e.to_string())?,
        ),
    };
    let values = if let Some((inputs, outputs)) = &function.scalar_types {
        object
            .exec_scalar_bits(&name, inputs, outputs, arguments)
            .map_err(|e| e.to_string())?
    } else {
        object
            .validate_i64_signature(&name, function.inputs, function.outputs)
            .map_err(|e| e.to_string())?;
        let mut values = object.exec(&name, arguments).map_err(|e| e.to_string())?;
        // The legacy API returns a sentinel zero for void exports.
        if function.outputs == 0 {
            values.clear();
        }
        values
    };
    Ok(values)
}

/// Launch a call on a shared executor without waiting for its result.
/// Compilation happens on the caller thread. Instantiation, initialization and
/// execution errors are delivered by `read`. Queue time is not execution time.
///
/// # Errors
/// Rejects invalid handles, leases, argument counts, worker counts, module
/// compilation failures and executor startup/submission failures.
pub fn call_async(context: u64, function: u64, arguments: &[i64], workers: usize) -> Result<i32> {
    if !(1..=async_pool::MAX_WORKERS).contains(&workers) {
        return Err(format!(
            "WASM async worker count must be between 1 and {}",
            async_pool::MAX_WORKERS
        ));
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let State {
            modules,
            functions,
            contexts,
            async_workers,
            ..
        } = &mut *state;
        let index = usize::try_from(function.checked_sub(1).ok_or("Invalid WASM function")?)
            .map_err(|e| e.to_string())?;
        let function = functions.get(index).ok_or("Unknown WASM function")?.clone();
        if arguments.len() != function.inputs {
            return Err("WASM argument count mismatch".into());
        }
        let context = contexts.get_mut(&context).ok_or("Unknown WASM context")?;
        if !context.leased || context.pending.is_some() {
            return Err("WASM context is not leased or has an unread result".into());
        }
        if async_workers.is_some_and(|count| count != workers) {
            return Err("WASM async worker count changed without reset".into());
        }
        let prepared = if context.modules.contains_key(&function.module) {
            None
        } else {
            let configuration = modules.get(&function.module).ok_or("Unknown WASM module")?;
            let compiled = COMPILED.with(|cache| cache.borrow_mut().get(&configuration.bytes))?;
            Some((
                compiled,
                configuration.timeout_seconds,
                configuration.auto_init,
            ))
        };
        POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.as_ref().is_some_and(|pool| pool.size() != workers) {
                // State replacement already cancelled the old pool's work, and
                // nothing was submitted since: this State has not set a count.
                debug_assert!(
                    async_workers.is_none(),
                    "WASM async pool size differs from this State's worker count"
                );
                drop(pool.take());
            }
            if pool.is_none() {
                *pool = Some(async_pool::Pool::new(workers)?);
            }
            Ok::<_, String>(())
        })?;
        *async_workers = Some(workers);
        let arguments = arguments.to_vec();
        let mut instances = std::mem::take(&mut context.modules);
        let (sender, receiver) = async_pool::reply();
        let job = Box::new(move || {
            let result = (|| {
                if let Some((compiled, timeout, auto_init)) = prepared {
                    let mut object = compiled
                        .instantiate_with_limits(timeout, None)
                        .map_err(|e| e.to_string())?;
                    if auto_init && object.has_func("init") {
                        object
                            .validate_i64_signature("init", 0, 0)
                            .map_err(|e| e.to_string())?;
                        object.exec("init", &[]).map_err(|e| e.to_string())?;
                    }
                    instances.insert(function.module, object);
                }
                let object = instances
                    .get_mut(&function.module)
                    .ok_or("Unknown WASM instance")?;
                execute(object, &function, &arguments)
            })();
            // An abandoned receiver must never block executor shutdown.
            sender.send((instances, result));
        });
        with_pool(|pool| pool.ok_or("WASM async executor unavailable")?.submit(job))?;
        context.pending = Some(Pending::Async(receiver));
        Ok(0)
    })
}

/// Consume exactly the expected number of result values.
///
/// # Errors
/// Rejects unknown or unleased contexts, missing results, and incorrect counts.
pub fn read(context: u64, count: usize) -> Result<Vec<i64>> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let context = state
            .contexts
            .get_mut(&context)
            .ok_or("Unknown WASM context")?;
        if !context.leased {
            return Err("WASM context is not leased".into());
        }
        let values = match context.pending.take().ok_or("No pending WASM result")? {
            Pending::Ready(values) => values,
            Pending::Async(receiver) => {
                let (instances, result) = receiver.recv().ok_or("WASM async job disconnected")?;
                context.modules = instances;
                result?
            }
        };
        if values.len() != count {
            return Err("WASM result count mismatch".into());
        }
        Ok(values)
    })
}

/// The pointer remains valid until another error is recorded on this thread.
#[must_use]
pub fn error_message() -> *const u8 {
    ERROR.with(|error| error.borrow().as_ptr())
}

/// Export the same versioned C ABI from the PECOS QIS FFI and Selene utility.
///
/// Pointer arguments must refer to live buffers of the indicated lengths.
/// Counts are bounded by the buffers emitted by the public lowering pass.
#[macro_export]
macro_rules! export_wasm_runtime {
    () => {
        #[unsafe(no_mangle)]
        pub extern "C" fn pecos_wasm_v1_reset() -> i32 {
            $crate::boundary(-1, $crate::reset)
        }
        /// # Safety
        /// key must point to 32 readable bytes identifying the complete program/configuration.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v2_begin_worker_shot(key: *const u8) -> i32 {
            $crate::boundary(-1, || {
                if key.is_null() {
                    return Err("Null WASM worker state key".into());
                }
                let mut bytes = [0_u8; 32];
                bytes.copy_from_slice(unsafe { std::slice::from_raw_parts(key, 32) });
                $crate::begin_worker_shot(bytes)
            })
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn pecos_wasm_v2_finish_worker_shot() -> i32 {
            $crate::boundary(-1, $crate::finish_worker_shot)
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn pecos_wasm_v1_error() -> *const u8 {
            $crate::error_message()
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn pecos_wasm_v1_acquire(index: u64) -> u64 {
            $crate::boundary(0, || $crate::acquire(index))
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn pecos_wasm_v1_release(handle: u64) -> i32 {
            $crate::boundary(-1, || $crate::release(handle))
        }
        /// # Safety
        /// bytes must point to length readable bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v1_register(
            module: u64,
            bytes: *const u8,
            length: usize,
        ) -> i32 {
            $crate::boundary(-1, || {
                if bytes.is_null() {
                    return Err("Null WASM module".into());
                }
                $crate::register(module, unsafe { std::slice::from_raw_parts(bytes, length) })
            })
        }
        /// Register bytes with a positive finite per-execution timeout in seconds.
        ///
        /// # Safety
        /// bytes must point to length readable bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v2_register(
            module: u64,
            bytes: *const u8,
            length: usize,
            timeout_seconds: f64,
        ) -> i32 {
            $crate::boundary(-1, || {
                if bytes.is_null() {
                    return Err("Null WASM module".into());
                }
                $crate::register_with_timeout(
                    module,
                    unsafe { std::slice::from_raw_parts(bytes, length) },
                    timeout_seconds,
                )
            })
        }
        /// Register with execution timeout and flags (bit 0 skips optional init).
        /// Unknown flag bits are rejected; WASM start is never skipped.
        ///
        /// # Safety
        /// bytes must point to length readable bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v3_register(
            module: u64,
            bytes: *const u8,
            length: usize,
            timeout_seconds: f64,
            flags: u32,
        ) -> i32 {
            $crate::boundary(-1, || {
                if bytes.is_null() {
                    return Err("Null WASM module".into());
                }
                $crate::register_with_options(
                    module,
                    unsafe { std::slice::from_raw_parts(bytes, length) },
                    timeout_seconds,
                    flags,
                )
            })
        }
        /// # Safety
        /// name must point to length readable UTF-8 bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v1_lookup(
            module: u64,
            name: *const u8,
            length: usize,
            inputs: usize,
            outputs: usize,
        ) -> u64 {
            $crate::boundary(0, || {
                if name.is_null() {
                    return Err("Null WASM export name".into());
                }
                let bytes = unsafe { std::slice::from_raw_parts(name, length) };
                let name = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
                $crate::lookup(module, name, inputs, outputs)
            })
        }
        /// Typed extension to v1 lookup. Tags: 0 = i64, 1 = f64, 2 = i32 in a 64-bit slot.
        /// The returned handle uses v1 call/read's 64-bit slots; float slots
        /// carry IEEE 754 bits. Existing integer lookup/call clients are unchanged.
        ///
        /// # Safety
        /// name points to length UTF-8 bytes. Type arrays point to their counts
        /// of readable bytes, or may be null when the corresponding count is zero.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v2_lookup(
            module: u64,
            name: *const u8,
            length: usize,
            input_types: *const u8,
            inputs: usize,
            output_types: *const u8,
            outputs: usize,
        ) -> u64 {
            $crate::boundary(0, || {
                if name.is_null() {
                    return Err("Null WASM export name".into());
                }
                let bytes = unsafe { std::slice::from_raw_parts(name, length) };
                let name = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
                let types = |pointer: *const u8, count: usize| {
                    if count == 0 {
                        return $crate::scalar_types(&[]);
                    }
                    if pointer.is_null() {
                        return Err("Null WASM scalar types".into());
                    }
                    $crate::scalar_types(unsafe { std::slice::from_raw_parts(pointer, count) })
                };
                $crate::lookup_scalar(
                    module,
                    name,
                    &types(input_types, inputs)?,
                    &types(output_types, outputs)?,
                )
            })
        }
        /// Lookup by zero-based export-list position, with the same scalar tags
        /// as v2 named lookup. Non-function exports occupy positions as well.
        ///
        /// # Safety
        /// Type arrays point to their counts of readable bytes, or may be null
        /// when the corresponding count is zero.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v2_lookup_by_id(
            module: u64,
            index: u64,
            input_types: *const u8,
            inputs: usize,
            output_types: *const u8,
            outputs: usize,
        ) -> u64 {
            $crate::boundary(0, || {
                let types = |pointer: *const u8, count: usize| {
                    if count == 0 {
                        return $crate::scalar_types(&[]);
                    }
                    if pointer.is_null() {
                        return Err("Null WASM scalar types".into());
                    }
                    $crate::scalar_types(unsafe { std::slice::from_raw_parts(pointer, count) })
                };
                $crate::lookup_by_id(
                    module,
                    index,
                    &types(input_types, inputs)?,
                    &types(output_types, outputs)?,
                )
            })
        }
        /// Typed v3 lookup: v2 tags plus tag 3 for native i32/i64 selection per integer lane.
        /// The returned handle uses v1 call/read's 64-bit slots; float slots
        /// carry IEEE 754 bits. Existing integer lookup/call clients are unchanged.
        ///
        /// # Safety
        /// name points to length UTF-8 bytes. Type arrays point to their counts
        /// of readable bytes, or may be null when the corresponding count is zero.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v3_lookup(
            module: u64,
            name: *const u8,
            length: usize,
            input_types: *const u8,
            inputs: usize,
            output_types: *const u8,
            outputs: usize,
        ) -> u64 {
            $crate::boundary(0, || {
                if name.is_null() {
                    return Err("Null WASM export name".into());
                }
                let bytes = unsafe { std::slice::from_raw_parts(name, length) };
                let name = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
                let types = |pointer: *const u8, count: usize| {
                    if count == 0 {
                        return $crate::scalar_types_auto(&[]);
                    }
                    if pointer.is_null() {
                        return Err("Null WASM scalar types".into());
                    }
                    $crate::scalar_types_auto(unsafe { std::slice::from_raw_parts(pointer, count) })
                };
                $crate::lookup_scalar(
                    module,
                    name,
                    &types(input_types, inputs)?,
                    &types(output_types, outputs)?,
                )
            })
        }
        /// Lookup by zero-based export-list position, with the same scalar tags
        /// as v3 named lookup. Non-function exports occupy positions as well.
        ///
        /// # Safety
        /// Type arrays point to their counts of readable bytes, or may be null
        /// when the corresponding count is zero.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v3_lookup_by_id(
            module: u64,
            index: u64,
            input_types: *const u8,
            inputs: usize,
            output_types: *const u8,
            outputs: usize,
        ) -> u64 {
            $crate::boundary(0, || {
                let types = |pointer: *const u8, count: usize| {
                    if count == 0 {
                        return $crate::scalar_types_auto(&[]);
                    }
                    if pointer.is_null() {
                        return Err("Null WASM scalar types".into());
                    }
                    $crate::scalar_types_auto(unsafe { std::slice::from_raw_parts(pointer, count) })
                };
                $crate::lookup_by_id(
                    module,
                    index,
                    &types(input_types, inputs)?,
                    &types(output_types, outputs)?,
                )
            })
        }
        /// Launch an asynchronous call; execution errors surface at read.
        ///
        /// # Safety
        /// arguments must point to count readable i64 values, or be null for count=0.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v2_call(
            context: u64,
            function: u64,
            arguments: *const i64,
            count: usize,
            workers: usize,
        ) -> i32 {
            $crate::boundary(-1, || {
                let arguments = if count == 0 {
                    &[]
                } else {
                    if arguments.is_null() {
                        return Err("Null WASM arguments".into());
                    }
                    unsafe { std::slice::from_raw_parts(arguments, count) }
                };
                $crate::call_async(context, function, arguments, workers)
            })
        }
        /// # Safety
        /// arguments must point to count readable i64 values, or be null for count=0.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v1_call(
            context: u64,
            function: u64,
            arguments: *const i64,
            count: usize,
        ) -> i32 {
            $crate::boundary(-1, || {
                let arguments = if count == 0 {
                    &[]
                } else {
                    if arguments.is_null() {
                        return Err("Null WASM arguments".into());
                    }
                    unsafe { std::slice::from_raw_parts(arguments, count) }
                };
                $crate::call(context, function, arguments)
            })
        }
        /// # Safety
        /// output must point to count writable i64 values, or be null for count=0.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn pecos_wasm_v1_read(
            context: u64,
            output: *mut i64,
            count: usize,
        ) -> i32 {
            $crate::boundary(-1, || {
                if count > 0 && output.is_null() {
                    return Err("Null WASM result".into());
                }
                let values = $crate::read(context, count)?;
                if count > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(values.as_ptr(), output, count);
                    }
                }
                Ok(0)
            })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    mod async_tests;
    #[cfg(unix)]
    mod fork_tests;

    #[test]
    fn compilation_cache_reuses_exact_bytes_and_evicts_lru() {
        let mut cache = CompilationCache::default();
        let a = b"(module)";
        let b = b"(module (func))";
        let c = b"(module (func) (func))";
        for bytes in [a.as_slice(), b, a, c] {
            cache.get_with_limits(bytes, 2, 1024).unwrap();
        }
        assert_eq!(cache.compilations, 3);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries[0].wasm_bytes(), a);
        assert_eq!(cache.entries[1].wasm_bytes(), c);
        cache.get_with_limits(b, 2, 1024).unwrap();
        assert_eq!(cache.compilations, 4);
        assert_eq!(cache.source_bytes, b.len() + c.len());
        let before = cache.source_bytes;
        assert!(cache.get(b"not wasm").is_err());
        assert_eq!(cache.source_bytes, before);
        assert_eq!(cache.compilations, 4);
    }

    #[test]
    fn compilation_cache_byte_budget_and_oversized_bypass() {
        let mut cache = CompilationCache::default();
        let a = b"(module)";
        let b = b"(module (func))";
        cache.get_with_limits(a, 8, b.len()).unwrap();
        cache.get_with_limits(b, 8, b.len()).unwrap();
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.source_bytes, b.len());
        let large = b"(module (func) (func))";
        cache.get_with_limits(large, 8, b.len()).unwrap();
        cache.get_with_limits(large, 8, b.len()).unwrap();
        assert_eq!(cache.compilations, 4);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.source_bytes, b.len());
    }

    #[test]
    fn cached_code_survives_shots_but_memory_and_globals_do_not() {
        let bytes = br#"(module
            (memory 1)
            (global $n (mut i64) (i64.const 0))
            (func $start i32.const 0 i64.const 40 i64.store)
            (start $start)
            (func (export "init") global.get $n i64.const 1 i64.add global.set $n)
            (func (export "next") (result i64)
                i32.const 0 i32.const 0 i64.load i64.const 1 i64.add i64.store
                i32.const 0 i64.load global.get $n i64.add))"#;
        COMPILED.with(|cache| *cache.borrow_mut() = CompilationCache::default());
        for module in [3, 7, 3] {
            reset().unwrap();
            register(module, bytes).unwrap();
            let function = lookup(module, "next", 0, 1).unwrap();
            for index in [0, 1] {
                let context = acquire(index).unwrap();
                call(context, function, &[]).unwrap();
                assert_eq!(read(context, 1).unwrap(), [42]);
                call(context, function, &[]).unwrap();
                assert_eq!(read(context, 1).unwrap(), [43]);
            }
        }
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 1));
        reset().unwrap();
        register(
            3,
            b"(module (func (export \"next\") (result i64) i64.const 9))",
        )
        .unwrap();
        let context = acquire(0).unwrap();
        let function = lookup(3, "next", 0, 1).unwrap();
        call(context, function, &[]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [9]);
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 2));
    }

    #[test]
    fn timeout_configuration_is_per_instance_not_per_compilation() {
        reset().unwrap();
        COMPILED.with(|cache| *cache.borrow_mut() = CompilationCache::default());
        register_with_timeout(0, MODULE, 0.05).unwrap();
        register_with_timeout(1, MODULE, 7.5).unwrap();
        let context = acquire(0).unwrap();
        assert_eq!(invoke(context, 0, "add", &[2, 3], 1), [5]);
        assert_eq!(invoke(context, 1, "add", &[4, 5], 1), [9]);
        STATE.with(|state| {
            let state = state.borrow();
            let modules = &state.contexts[&context].modules;
            assert!((modules[&0].timeout_seconds() - 0.05).abs() < f64::EPSILON);
            assert!((modules[&1].timeout_seconds() - 7.5).abs() < f64::EPSILON);
        });
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 1));
        let context = start();
        invoke(context, 0, "add", &[1, 2], 1);
        STATE.with(|state| {
            assert!(
                (state.borrow().contexts[&context].modules[&0].timeout_seconds() - 1.0).abs()
                    < f64::EPSILON
            );
        });
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 1));
    }

    #[test]
    fn invalid_timeout_registration_does_not_mutate_state() {
        reset().unwrap();
        for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                register_with_timeout(0, MODULE, timeout).unwrap_err(),
                format!("WASM timeout must be a positive finite number of seconds; got {timeout}")
            );
            assert!(STATE.with(|state| state.borrow().modules.is_empty()));
        }
        register(0, MODULE).unwrap();
    }

    #[test]
    fn initialization_policy_is_per_instance_and_preserves_start() {
        let bytes = br#"(module
            (global $n (mut i64) (i64.const 0))
            (func $start i64.const 11 global.set $n)
            (start $start)
            (func (export "init") global.get $n i64.const 31 i64.add global.set $n)
            (func (export "get") (result i64) global.get $n)
            (func (export "increment") (result i64)
                global.get $n i64.const 1 i64.add global.set $n global.get $n))"#;
        COMPILED.with(|cache| *cache.borrow_mut() = CompilationCache::default());
        for _ in 0..2 {
            reset().unwrap();
            register(0, bytes).unwrap();
            register_with_options(1, bytes, 2.5, SKIP_AUTO_INIT).unwrap();
            for index in [0, 1] {
                let context = acquire(index).unwrap();
                assert_eq!(invoke(context, 0, "get", &[], 1), [42]);
                assert_eq!(invoke(context, 1, "get", &[], 1), [11]);
                assert_eq!(invoke(context, 1, "init", &[], 0), Vec::<i64>::new());
                assert_eq!(invoke(context, 1, "get", &[], 1), [42]);
                invoke(context, 0, "increment", &[], 1);
                release(context).unwrap();
                let context = acquire(index).unwrap();
                assert_eq!(invoke(context, 0, "get", &[], 1), [43]);
                assert_eq!(invoke(context, 1, "get", &[], 1), [42]);
                release(context).unwrap();
            }
        }
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 1));
    }

    #[test]
    fn skip_init_does_not_validate_or_execute_optional_export() {
        for init in [
            "(func (export \"init\") unreachable)",
            "(func (export \"init\") (param i64))",
        ] {
            let bytes = format!("(module {init} (func (export \"get\") (result i64) i64.const 7))");
            for flags in [0, SKIP_AUTO_INIT] {
                reset().unwrap();
                register_with_options(0, bytes.as_bytes(), 1.0, flags).unwrap();
                let context = acquire(0).unwrap();
                let function = lookup(0, "get", 0, 1).unwrap();
                let outcome = call(context, function, &[]);
                if flags == 0 {
                    assert!(outcome.is_err());
                } else {
                    outcome.unwrap();
                    assert_eq!(read(context, 1).unwrap(), [7]);
                }
            }
        }
    }

    #[test]
    fn initialization_opt_out_never_skips_wasm_start() {
        let bytes = b"(module (func $start unreachable) (start $start)
            (func (export \"get\") (result i64) i64.const 7))";
        for flags in [0, SKIP_AUTO_INIT] {
            reset().unwrap();
            register_with_options(0, bytes, 1.0, flags).unwrap();
            let context = acquire(0).unwrap();
            let function = lookup(0, "get", 0, 1).unwrap();
            assert!(call(context, function, &[]).is_err());
        }
    }

    #[test]
    fn unknown_registration_flags_do_not_mutate_state() {
        reset().unwrap();
        for flags in [2, 3, u32::MAX] {
            assert_eq!(
                register_with_options(0, MODULE, 1.0, flags).unwrap_err(),
                format!(
                    "Unknown WASM registration flags {flags:#x}; this runtime supports only \
                     SKIP_AUTO_INIT (0x1): upgrade the native WASM runtime to match the \
                     compiler that generated the program"
                )
            );
            assert!(STATE.with(|state| state.borrow().modules.is_empty()));
        }
        register_with_options(0, MODULE, 1.0, SKIP_AUTO_INIT).unwrap();
    }

    const MODULE: &[u8] = br#"(module
        (global $counter (mut i64) (i64.const 0))
        (func (export "init") (global.set $counter (i64.const 10)))
        (func (export "add") (param i64 i64) (result i64)
            local.get 0 local.get 1 i64.add)
        (func (export "increment") (result i64)
            global.get $counter i64.const 1 i64.add global.set $counter
            global.get $counter)
        (func (export "clear") (global.set $counter (i64.const 0)))
        (func (export "trap") unreachable)
        (func (export "wrong_type") (param i32) (result i32) local.get 0)
    )"#;

    fn start() -> u64 {
        reset().unwrap();
        register(0, MODULE).unwrap();
        acquire(0).unwrap()
    }

    fn invoke(context: u64, module: u64, name: &str, args: &[i64], outputs: usize) -> Vec<i64> {
        let function = lookup(module, name, args.len(), outputs).unwrap();
        call(context, function, args).unwrap();
        read(context, outputs).unwrap()
    }

    #[test]
    fn signed_arithmetic_and_wrapping() {
        let context = start();
        assert_eq!(invoke(context, 0, "add", &[-8, 3], 1), [-5]);
        assert_eq!(invoke(context, 0, "add", &[i64::MAX, 1], 1), [i64::MIN]);
    }

    #[test]
    fn leases_state_and_shot_reset() {
        let context = start();
        assert_eq!(acquire(0).unwrap(), 0);
        assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
        release(context).unwrap();
        assert_eq!(acquire(0).unwrap(), context);
        assert_eq!(invoke(context, 0, "increment", &[], 1), [12]);
        let other = acquire(1).unwrap();
        assert_eq!(invoke(other, 0, "increment", &[], 1), [11]);
        register(1, MODULE).unwrap();
        assert_eq!(invoke(context, 1, "increment", &[], 1), [11]);
        let context = start();
        assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
    }

    #[test]
    fn zero_results_still_order_effects() {
        let context = start();
        assert_eq!(invoke(context, 0, "clear", &[], 0), [] as [i64; 0]);
        assert_eq!(invoke(context, 0, "increment", &[], 1), [1]);
    }

    #[test]
    fn pending_results_and_leases_are_checked() {
        let context = start();
        let function = lookup(0, "increment", 0, 1).unwrap();
        call(context, function, &[]).unwrap();
        assert!(release(context).is_err());
        assert!(call(context, function, &[]).is_err());
        assert_eq!(read(context, 1).unwrap(), [11]);
        assert!(read(context, 1).is_err());
        release(context).unwrap();
        assert!(call(context, function, &[]).is_err());
    }

    #[test]
    fn validates_export_signatures_and_handles() {
        let context = start();
        for (name, inputs, outputs) in [("missing", 0, 0), ("add", 0, 1), ("wrong_type", 1, 1)] {
            let function = lookup(0, name, inputs, outputs).unwrap();
            assert!(call(context, function, &vec![0; inputs]).is_err());
        }
        assert!(call(context, 0, &[]).is_err());
        assert!(lookup(2, "add", 2, 1).is_err());
        assert!(lookup(0, "add", 2, 2).is_err());
        assert!(register(0, MODULE).is_err());
        assert!(acquire(u64::MAX).is_err());
    }

    #[test]
    fn traps_cleanup_at_abi_boundary_and_allow_recovery() {
        COMPILED.with(|cache| *cache.borrow_mut() = CompilationCache::default());
        let context = start();
        let trap = lookup(0, "trap", 0, 0).unwrap();
        assert_eq!(boundary(-1, || call(context, trap, &[])), -1);
        assert!(STATE.with(|state| state.borrow().contexts.is_empty()));
        assert!(ERROR.with(|error| error.borrow()[0] > 0));
        let context = start();
        assert_eq!(invoke(context, 0, "add", &[2, 3], 1), [5]);
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 1));
    }

    #[test]
    fn threads_have_independent_contexts() {
        let context = start();
        assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
        std::thread::spawn(|| {
            let context = start();
            assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
            reset().unwrap();
        })
        .join()
        .unwrap();
        assert_eq!(invoke(context, 0, "increment", &[], 1), [12]);
    }

    #[test]
    fn float_bits_mixed_signatures_and_void_results() {
        reset().unwrap();
        register(
            0,
            br#"(module
            (global $x (mut f64) (f64.const 0))
            (func (export "echo") (param f64) (result f64) local.get 0)
            (func (export "set") (param f64) local.get 0 global.set $x)
            (func (export "get") (result f64) global.get $x)
            (func (export "mixed") (param f64 i64) (result f64)
                local.get 0 local.get 1 f64.convert_i64_s f64.add))"#,
        )
        .unwrap();
        let context = acquire(0).unwrap();
        let echo =
            lookup_scalar(0, "echo", &[WasmScalarType::F64], &[WasmScalarType::F64]).unwrap();
        // Include negative zero, infinities, a subnormal and a quiet NaN payload.
        for bits in [
            0_u64,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            1,
            0x7ff8_0000_0000_1234,
        ] {
            let bits = i64::from_ne_bytes(bits.to_ne_bytes());
            call(context, echo, &[bits]).unwrap();
            assert_eq!(read(context, 1).unwrap(), [bits]);
        }
        let set = lookup_scalar(0, "set", &[WasmScalarType::F64], &[]).unwrap();
        let get = lookup_scalar(0, "get", &[], &[WasmScalarType::F64]).unwrap();
        let bits = i64::from_ne_bytes((-2.5_f64).to_bits().to_ne_bytes());
        call(context, set, &[bits]).unwrap();
        assert_eq!(read(context, 0).unwrap(), [] as [i64; 0]);
        release(context).unwrap();
        assert_eq!(acquire(0).unwrap(), context);
        call(context, get, &[]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [bits]);
        let mixed = lookup_scalar(
            0,
            "mixed",
            &[WasmScalarType::F64, WasmScalarType::I64],
            &[WasmScalarType::F64],
        )
        .unwrap();
        call(context, mixed, &[bits, 4]).unwrap();
        let value = read(context, 1).unwrap()[0];
        assert_eq!(f64::from_bits(u64::from_ne_bytes(value.to_ne_bytes())), 1.5);
        // Same arity does not make differently typed exports compatible.
        let wrong = lookup(0, "echo", 1, 1).unwrap();
        assert_ne!(wrong, echo);
        assert!(call(context, wrong, &[bits]).is_err());
        assert!(scalar_types(&[3]).is_err());
    }

    #[test]
    fn i32_slots_wrap_arguments_sign_extend_results_and_keep_widths_strict() {
        reset().unwrap();
        register(
            0,
            br#"(module
            (memory (export "memory") 1)
            (global $value (mut i32) (i32.const 0))
            (func (export "add") (param i32 i32) (result i32)
                local.get 0 local.get 1 i32.add)
            (func (export "set") (param i32) local.get 0 global.set $value)
            (func (export "get") (result i32) global.get $value)
            (func (export "wide") (param i64) (result i64) local.get 0)
            (func (export "float") (param f64) (result f64) local.get 0))"#,
        )
        .unwrap();
        let context = acquire(0).unwrap();
        let int32 = scalar_types(&[2]).unwrap();
        assert_eq!(int32, [WasmScalarType::I32]);
        let add = lookup_scalar(0, "add", &[WasmScalarType::I32; 2], &int32).unwrap();
        let indexed = lookup_by_id(0, 1, &[WasmScalarType::I32; 2], &int32).unwrap();
        for function in [add, indexed] {
            for (arguments, expected) in [
                ([2, 3], 5),
                ([2_147_483_647, 1], -2_147_483_648),
                ([-2_147_483_648, -1], 2_147_483_647),
                ([4_294_967_301, 0], 5),
                ([-2_147_483_649, 0], 2_147_483_647),
                ([i64::MAX, 0], -1),
                ([i64::MIN, 0], 0),
            ] {
                call(context, function, &arguments).unwrap();
                assert_eq!(read(context, 1).unwrap(), [expected]);
            }
        }
        let set = lookup_scalar(0, "set", &int32, &[]).unwrap();
        let get = lookup_scalar(0, "get", &[], &int32).unwrap();
        call(context, set, &[-7]).unwrap();
        assert_eq!(read(context, 0).unwrap(), [] as [i64; 0]);
        release(context).unwrap();
        assert_eq!(acquire(0).unwrap(), context);
        call(context, get, &[]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [-7]);
        // Older entry points retain their exact native-width contract.
        let strict = lookup(0, "add", 2, 1).unwrap();
        assert_ne!(strict, add);
        assert!(
            call(context, strict, &[2, 3])
                .unwrap_err()
                .contains("signature mismatch")
        );
        for name in ["wide", "float"] {
            let wrong = lookup_scalar(0, name, &int32, &int32).unwrap();
            assert!(
                call(context, wrong, &[2])
                    .unwrap_err()
                    .contains("signature mismatch")
            );
        }
        let wrong_return = lookup_scalar(0, "get", &[], &[WasmScalarType::I64]).unwrap();
        assert!(call(context, wrong_return, &[]).is_err());
    }

    #[test]
    fn automatic_integer_lanes_preserve_widths_and_validate_before_execution() {
        use WasmScalarType::{F64, I32, I64, IntegerAuto};
        assert!(scalar_types(&[3]).is_err());
        assert_eq!(
            scalar_types_auto(&[0, 1, 2, 3]).unwrap(),
            [I64, F64, I32, IntegerAuto]
        );
        assert!(scalar_types_auto(&[4]).is_err());
        reset().unwrap();
        register(
            0,
            br#"(module
            (global $calls (mut i64) (i64.const 0))
            (func (export "mixed") (param i32 i64 f64) (result i64)
                global.get $calls i64.const 1 i64.add global.set $calls
                local.get 0 i64.extend_i32_s local.get 1 i64.add
                local.get 2 i64.trunc_f64_s i64.add)
            (func (export "narrow") (param i64) (result i32)
                local.get 0 i32.wrap_i64)
            (func (export "calls") (result i64) global.get $calls))"#,
        )
        .unwrap();
        let context = acquire(0).unwrap();
        let inputs = [IntegerAuto, IntegerAuto, F64];
        let outputs = [IntegerAuto];
        let named = lookup_scalar(0, "mixed", &inputs, &outputs).unwrap();
        let indexed = lookup_by_id(0, 0, &inputs, &outputs).unwrap();
        let float_bits = i64::from_ne_bytes(2.5_f64.to_bits().to_ne_bytes());
        for function in [named, indexed] {
            call(
                context,
                function,
                &[4_294_967_301, 4_294_967_301, float_bits],
            )
            .unwrap();
            assert_eq!(read(context, 1).unwrap(), [4_294_967_308]);
        }
        let narrow = lookup_scalar(0, "narrow", &outputs, &outputs).unwrap();
        call(context, narrow, &[2_147_483_648]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [-2_147_483_648]);
        for (ins, outs) in [
            ([I64, I64, F64], [I64]),
            ([I32, I32, F64], [I32]),
            ([IntegerAuto; 3], [IntegerAuto]),
            (inputs, [F64]),
        ] {
            let wrong = lookup_scalar(0, "mixed", &ins, &outs).unwrap();
            assert_ne!(wrong, named);
            assert!(call(context, wrong, &[1, 2, float_bits]).is_err());
        }
        assert_eq!(invoke(context, 0, "calls", &[], 1), [2]);
        release(context).unwrap();
        assert_eq!(acquire(0).unwrap(), context);
        assert_eq!(invoke(context, 0, "calls", &[], 1), [2]);
        let fresh = acquire(1).unwrap();
        assert_eq!(invoke(fresh, 0, "calls", &[], 1), [0]);
    }

    #[test]
    fn worker_shots_retain_instances_and_reset_on_incomplete_or_changed_sessions() {
        reset().unwrap();
        for expected in [11, 12, 13] {
            begin_worker_shot([1; 32]).unwrap();
            register(0, MODULE).unwrap();
            let context = acquire(0).unwrap();
            assert_eq!(invoke(context, 0, "increment", &[], 1), [expected]);
            release(context).unwrap();
            finish_worker_shot().unwrap();
        }
        // A non-WASM abort can bypass the normal finish, even after releasing
        // every context. The active marker prevents retaining that mutation.
        assert!(
            std::panic::catch_unwind(|| {
                begin_worker_shot([1; 32]).unwrap();
                register(0, MODULE).unwrap();
                let context = acquire(0).unwrap();
                assert_eq!(invoke(context, 0, "increment", &[], 1), [14]);
                release(context).unwrap();
                panic!("External quantum/runtime error after WASM mutation");
            })
            .is_err()
        );
        begin_worker_shot([1; 32]).unwrap();
        register(0, MODULE).unwrap();
        let context = acquire(0).unwrap();
        assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
        release(context).unwrap();
        finish_worker_shot().unwrap();
        begin_worker_shot([2; 32]).unwrap();
        register(0, MODULE).unwrap();
        let context = acquire(0).unwrap();
        assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
        release(context).unwrap();
        finish_worker_shot().unwrap();
        reset().unwrap();
        assert!(finish_worker_shot().is_err());
    }

    #[test]
    fn worker_finish_errors_and_traps_clear_all_mutable_state() {
        for pending in [false, true] {
            reset().unwrap();
            begin_worker_shot([1; 32]).unwrap();
            register(0, MODULE).unwrap();
            let context = acquire(0).unwrap();
            if pending {
                let function = lookup(0, "increment", 0, 1).unwrap();
                call(context, function, &[]).unwrap();
            }
            assert_eq!(boundary(-1, finish_worker_shot), -1);
            STATE.with(|state| {
                let state = state.borrow();
                assert!(state.worker_key.is_none());
                assert!(!state.shot_active);
                assert!(state.contexts.is_empty());
            });
            begin_worker_shot([1; 32]).unwrap();
            register(0, MODULE).unwrap();
            let context = acquire(0).unwrap();
            assert_eq!(invoke(context, 0, "increment", &[], 1), [11]);
            let trap = lookup(0, "trap", 0, 0).unwrap();
            assert_eq!(boundary(-1, || call(context, trap, &[])), -1);
            assert!(finish_worker_shot().is_err());
        }
    }

    #[test]
    fn asynchronous_results_are_routed_by_context_and_preserve_state() {
        if !crate::test_support::watchdog(
            "tests::asynchronous_results_are_routed_by_context_and_preserve_state",
        ) {
            return;
        }
        reset().unwrap();
        for expected in [11, 13] {
            begin_worker_shot([7; 32]).unwrap();
            register(0, MODULE).unwrap();
            let first = acquire(0).unwrap();
            let second = acquire(1).unwrap();
            let increment = lookup(0, "increment", 0, 1).unwrap();
            let add = lookup(0, "add", 2, 1).unwrap();
            call_async(first, increment, &[], 2).unwrap();
            call_async(second, add, &[4, 5], 2).unwrap();
            assert!(release(first).is_err());
            assert!(call_async(first, increment, &[], 2).is_err());
            assert!(finish_worker_shot().is_err());
            assert_eq!(read(second, 1).unwrap(), [9]);
            assert_eq!(read(first, 1).unwrap(), [expected]);
            // Return ownership to inline calls without recreating the store.
            call(first, increment, &[]).unwrap();
            assert_eq!(read(first, 1).unwrap(), [expected + 1]);
            assert!(call_async(first, increment, &[], 1).is_err());
            release(first).unwrap();
            release(second).unwrap();
            finish_worker_shot().unwrap();
        }
        reset().unwrap();
    }

    #[test]
    fn single_async_worker_queues_many_contexts() {
        if !crate::test_support::watchdog("tests::single_async_worker_queues_many_contexts") {
            return;
        }
        let _ = start();
        let add = lookup(0, "add", 2, 1).unwrap();
        let mut contexts = Vec::new();
        for index in 1..9 {
            let context = acquire(index).unwrap();
            call_async(context, add, &[i64::try_from(index).unwrap(), 10], 1).unwrap();
            contexts.push((context, i64::try_from(index).unwrap() + 10));
        }
        for (context, expected) in contexts.into_iter().rev() {
            assert_eq!(read(context, 1).unwrap(), [expected]);
        }
        assert!(call_async(0, add, &[], 0).is_err());
        assert!(call_async(0, add, &[], 65).is_err());
        reset().unwrap();
    }

    #[test]
    fn asynchronous_initialization_and_signature_failures_surface_at_read() {
        if !crate::test_support::watchdog(
            "tests::asynchronous_initialization_and_signature_failures_surface_at_read",
        ) {
            return;
        }
        for module in [
            br#"(module (func (export "init") unreachable) (func (export "get") (result i64) i64.const 1))"#.as_slice(),
            br#"(module (func (export "get") (result f64) f64.const 1))"#.as_slice(),
        ] {
            reset().unwrap();
            register(0, module).unwrap();
            let context = acquire(0).unwrap();
            let function = lookup(0, "get", 0, 1).unwrap();
            assert_eq!(call_async(context, function, &[], 1).unwrap(), 0);
            assert_eq!(boundary(-1, || read(context, 1).map(|_| 0)), -1);
            let context = start();
            assert_eq!(invoke(context, 0, "add", &[2, 3], 1), [5]);
        }
    }

    #[test]
    fn legacy_lookup_keeps_large_arities_compact() {
        let context = start();
        assert!(lookup(u64::MAX, "add", usize::MAX, 1).is_err());
        let function = lookup(0, "add", usize::MAX, 1).unwrap();
        assert_eq!(
            call(context, function, &[]).unwrap_err(),
            "WASM argument count mismatch"
        );
        let integer = lookup(0, "add", 2, 1).unwrap();
        assert_eq!(
            integer,
            lookup_scalar(0, "add", &[WasmScalarType::I64; 2], &[WasmScalarType::I64]).unwrap()
        );
    }

    #[test]
    fn indexed_exports_use_full_export_order_and_exact_signatures() {
        reset().unwrap();
        register(
            0,
            br#"(module
            (memory 1)
            (func $hidden (result i64) i64.const 999)
            (func $second (param i64) (result i64) local.get 0 i64.const 2 i64.mul)
            (func $first (param i64) (result i64) local.get 0 i64.const 1 i64.add)
            (func $float (param f64) (result f64) local.get 0)
            (export "z_first" (func $first))
            (export "memory" (memory 0))
            (export "a_second" (func $second))
            (export "float" (func $float)))"#,
        )
        .unwrap();
        let context = acquire(0).unwrap();
        let integer = [WasmScalarType::I64];
        let first = lookup_by_id(0, 0, &integer, &integer).unwrap();
        let second = lookup_by_id(0, 2, &integer, &integer).unwrap();
        assert_eq!(first, lookup_by_id(0, 0, &integer, &integer).unwrap());
        assert_ne!(first, second);
        call(context, first, &[7]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [8]);
        call(context, second, &[7]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [14]);
        for (index, message) in [
            (1, "not a function"),
            (4, "out of range"),
            (u64::MAX, "out of range"),
        ] {
            let function = lookup_by_id(0, index, &integer, &integer).unwrap();
            assert!(call(context, function, &[7]).unwrap_err().contains(message));
        }
        let wrong = lookup_by_id(0, 3, &integer, &integer).unwrap();
        assert!(
            call(context, wrong, &[7])
                .unwrap_err()
                .contains("signature mismatch")
        );
        let float = [WasmScalarType::F64];
        let echo = lookup_by_id(0, 3, &float, &float).unwrap();
        assert_ne!(echo, wrong);
        // Exact transport of a negative-zero float through indexed dispatch.
        call(context, echo, &[i64::MIN]).unwrap();
        assert_eq!(read(context, 1).unwrap(), [i64::MIN]);
        assert!(lookup_by_id(99, 0, &integer, &integer).is_err());
        assert!(lookup_by_id(0, 0, &integer, &[WasmScalarType::I64; 2]).is_err());
    }

    #[test]
    fn repeated_lookup_reuses_handles() {
        start();
        assert_eq!(
            lookup(0, "add", 2, 1).unwrap(),
            lookup(0, "add", 2, 1).unwrap()
        );
    }
}
