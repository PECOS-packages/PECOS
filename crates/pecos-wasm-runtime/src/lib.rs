//! Public WASM ABI v1. Implemented from the public tket.wasm operations.
//!
//! All state is local to the calling worker thread and reset at shot boundaries.
//! C entry points return a status; they never unwind or transfer control while
//! Rust frames own resources. The generated caller handles errors afterwards.

use pecos_wasm::{ForeignObject, WasmForeignObject};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

type Result<T> = std::result::Result<T, String>;

#[derive(Default)]
struct Context {
    leased: bool,
    modules: BTreeMap<u64, WasmForeignObject>,
    pending: Option<Vec<i64>>,
}

#[derive(PartialEq, Eq)]
struct Function {
    module: u64,
    name: String,
    inputs: usize,
    outputs: usize,
}

#[derive(Default)]
struct State {
    modules: BTreeMap<u64, Vec<u8>>,
    functions: Vec<Function>,
    contexts: BTreeMap<u64, Context>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
    // QIS panic uses a length-prefixed byte string (at most 255 bytes).
    static ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]);
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
///
/// # Errors
/// This operation always succeeds; the result matches the status-returning ABI.
pub fn reset() -> Result<i32> {
    STATE.with(|state| *state.borrow_mut() = State::default());
    Ok(0)
}

/// Register caller-supplied module bytes for this shot.
///
/// # Errors
/// Returns an error if the module identifier is already registered.
pub fn register(module: u64, bytes: &[u8]) -> Result<i32> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.modules.contains_key(&module) {
            return Err(format!("Duplicate WASM module {module}"));
        }
        state.modules.insert(module, bytes.to_vec());
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
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if !state.modules.contains_key(&module) {
            return Err(format!("Unknown WASM module {module}"));
        }
        let function = Function {
            module,
            name: name.into(),
            inputs,
            outputs,
        };
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
            let bytes = modules.get(&function.module).ok_or("Unknown WASM module")?;
            let mut object = WasmForeignObject::from_bytes(bytes).map_err(|e| e.to_string())?;
            // init is optional for Guppy modules (unlike the legacy QASM contract).
            if object.has_func("init") {
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
        object
            .validate_i64_signature(&function.name, function.inputs, function.outputs)
            .map_err(|e| e.to_string())?;
        let mut values = object
            .exec(&function.name, arguments)
            .map_err(|e| e.to_string())?;
        // ForeignObject's legacy API returns a sentinel zero for void exports.
        // The validated public WASM signature has a genuinely empty result row.
        if function.outputs == 0 {
            values.clear();
        }
        context.pending = Some(values);
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
        let values = context.pending.take().ok_or("No pending WASM result")?;
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
        let context = start();
        let trap = lookup(0, "trap", 0, 0).unwrap();
        assert_eq!(boundary(-1, || call(context, trap, &[])), -1);
        assert!(STATE.with(|state| state.borrow().contexts.is_empty()));
        assert!(ERROR.with(|error| error.borrow()[0] > 0));
        let context = start();
        assert_eq!(invoke(context, 0, "add", &[2, 3], 1), [5]);
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
    fn repeated_lookup_reuses_handles() {
        start();
        assert_eq!(
            lookup(0, "add", 2, 1).unwrap(),
            lookup(0, "add", 2, 1).unwrap()
        );
    }
}
