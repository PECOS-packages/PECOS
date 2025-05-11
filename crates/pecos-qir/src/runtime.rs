use pecos_core::QubitId;
use pecos_engines::byte_message::QuantumCmd;
use pecos_engines::core::result_id::ResultId;
use std::collections::HashMap;
use std::env;
use std::ffi::{CStr, c_char};
use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// QIR Runtime Implementation
///
/// This file contains the implementation of the QIR runtime functions that are used
/// when executing QIR programs. It defines the C-compatible functions that are called
/// by the QIR program to perform quantum operations.
///
/// # QIR Runtime Library
///
/// This file is a key component of the QIR runtime library, which is built by the
/// `build.rs` script in the pecos-qir crate. The library is pre-built and placed
/// in the target directory to speed up QIR compilation.
///
/// When the QIR compiler runs, it first checks for a pre-built library. If found,
/// it uses that library directly. If not, it falls back to building the runtime
/// on-demand using this file and related files.
///
/// # Implementation Details
///
/// The runtime provides functions for:
/// - Quantum gate operations (H, X, Y, Z, etc.)
/// - Qubit and result allocation/release
/// - Measurement operations
/// - Classical control operations
/// - Logging and message output
///
/// Helper function to get the current thread ID as a string
fn get_thread_id() -> String {
    format!("{:?}", thread::current().id())
}

// Global counters for qubit and result allocation
static NEXT_QUBIT_ID: AtomicUsize = AtomicUsize::new(0);
static NEXT_RESULT_ID: AtomicUsize = AtomicUsize::new(0);

// Global storage for measurement results
static MEASUREMENT_RESULTS: std::sync::LazyLock<Mutex<HashMap<String, u32>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

// Global storage for commands in structured format
static COMMANDS: std::sync::LazyLock<Mutex<Vec<QuantumCmd>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

/// Helper function to check if we should print commands
///
/// This function checks the `QIR_RUNTIME_QUIET` environment variable
/// to determine if commands should be printed to stdout.
///
/// # Returns
///
/// * `true` - If commands should be printed
/// * `false` - If commands should not be printed
fn should_print_commands() -> bool {
    match env::var("QIR_RUNTIME_QUIET") {
        Ok(val) => val != "1",
        Err(_) => true,
    }
}

/// Helper function to store and optionally print commands
///
/// This function stores the command in the global command collection
/// and optionally prints it to stdout for debugging.
///
/// # Arguments
///
/// * `cmd` - The quantum command to store
fn store_command(cmd: &QuantumCmd) {
    let thread_id = get_thread_id();

    // Always store the command in our collection
    if let Ok(mut commands) = COMMANDS.lock() {
        commands.push(cmd.clone());
    } else {
        eprintln!("QIR Runtime: [Thread {thread_id}] Failed to lock commands mutex");
    }

    // Print the command if not in quiet mode
    if should_print_commands() {
        println!("QIR Runtime: [Thread {thread_id}] {cmd}");
    }
}

// Quantum gate operations

/// Applies a rotation around the Z-axis to the specified qubit.
///
/// # Arguments
///
/// * `theta` - The rotation angle in radians
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__rz__body(theta: f64, qubit: usize) {
    store_command(&QuantumCmd::RZ(theta, QubitId(qubit)));
}

/// Applies a rotation around an axis in the ZY plane to the specified qubit.
///
/// # Arguments
///
/// * `theta` - The rotation angle in radians
/// * `phi` - The phase angle in radians
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__r1xy__body(theta: f64, phi: f64, qubit: usize) {
    store_command(&QuantumCmd::R1XY(theta, phi, QubitId(qubit)));
}

/// Applies a Hadamard gate to the specified qubit.
///
/// # Arguments
///
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__h__body(qubit: usize) {
    store_command(&QuantumCmd::H(QubitId(qubit)));
}

/// Applies an X gate to the specified qubit.
///
/// # Arguments
///
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__x__body(qubit: usize) {
    store_command(&QuantumCmd::X(QubitId(qubit)));
}

/// Applies a Y gate to the specified qubit.
///
/// # Arguments
///
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__y__body(qubit: usize) {
    store_command(&QuantumCmd::Y(QubitId(qubit)));
}

/// Applies a Z gate to the specified qubit.
///
/// # Arguments
///
/// * `qubit` - The qubit index to apply the gate to
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__z__body(qubit: usize) {
    store_command(&QuantumCmd::Z(QubitId(qubit)));
}

/// Applies a controlled-X gate to the specified qubits.
///
/// # Arguments
///
/// * `control` - The control qubit index
/// * `target` - The target qubit index
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit IDs are valid
/// and have been properly allocated. Calling with invalid qubit IDs may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cx__body(control: usize, target: usize) {
    store_command(&QuantumCmd::CX(QubitId(control), QubitId(target)));
}

/// Applies a controlled-Z gate to the specified qubits.
///
/// This is implemented as a sequence of H, CX, H gates.
///
/// # Arguments
///
/// * `control` - The control qubit index
/// * `target` - The target qubit index
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit IDs are valid
/// and have been properly allocated. Calling with invalid qubit IDs may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cz__body(control: usize, target: usize) {
    // Implement CZ as a sequence of H, CX, H
    store_command(&QuantumCmd::H(QubitId(target)));
    store_command(&QuantumCmd::CX(QubitId(control), QubitId(target)));
    store_command(&QuantumCmd::H(QubitId(target)));
}

/// Applies a SZZ gate to the specified qubits.
///
/// # Arguments
///
/// * `qubit1` - The first qubit index
/// * `qubit2` - The second qubit index
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit IDs are valid
/// and have been properly allocated. Calling with invalid qubit IDs may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__szz__body(qubit1: usize, qubit2: usize) {
    store_command(&QuantumCmd::SZZ(QubitId(qubit1), QubitId(qubit2)));
}

/// Applies a RZZ gate to the specified qubits.
///
/// # Arguments
///
/// * `theta` - The rotation angle in radians
/// * `qubit1` - The first qubit index
/// * `qubit2` - The second qubit index
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit IDs are valid
/// and have been properly allocated. Calling with invalid qubit IDs may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__rzz__body(theta: f64, qubit1: usize, qubit2: usize) {
    store_command(&QuantumCmd::RZZ(theta, QubitId(qubit1), QubitId(qubit2)));
}

/// Measures a qubit and stores the result.
///
/// # Arguments
///
/// * `qubit` - The qubit index to measure
/// * `result` - The result ID to store the measurement result
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID and result ID
/// are valid and have been properly allocated. Calling with invalid IDs may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__m__body(qubit: usize, result: usize) -> u32 {
    store_command(&QuantumCmd::Measure(QubitId(qubit), ResultId(result)));
    // In the real QIR runtime, this would return the actual measurement result
    // For this implementation, we just return 0
    0
}

/// Prepares a qubit in the |0⟩ state.
///
/// # Arguments
///
/// * `qubit` - The qubit index to prepare
///
/// # Safety
///
/// This function is called from C/C++ code and assumes that the qubit ID is valid
/// and has been properly allocated. Calling with an invalid qubit ID may lead to
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__reset__body(qubit: usize) {
    store_command(&QuantumCmd::Prep(QubitId(qubit)));
}

/// Allocates a new qubit.
///
/// # Returns
///
/// The ID of the newly allocated qubit
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_allocate() -> usize {
    let qubit_id = NEXT_QUBIT_ID.fetch_add(1, Ordering::SeqCst);
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Allocated qubit {qubit_id}");
    }

    qubit_id
}

/// Allocates a new result.
///
/// # Returns
///
/// The ID of the newly allocated result
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_allocate() -> usize {
    let result_id = NEXT_RESULT_ID.fetch_add(1, Ordering::SeqCst);
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Allocated result {result_id}");
    }

    result_id
}

/// Releases a qubit.
///
/// # Arguments
///
/// * `qubit` - The qubit ID to release
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_release(qubit: usize) {
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Released qubit {qubit}");
    }

    // We don't actually do anything with the qubit ID
    // In a real implementation, we would recycle the ID
}

/// Releases a result.
///
/// # Arguments
///
/// * `result` - The result ID to release
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_release(result: usize) {
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Released result {result}");
    }

    // We don't actually do anything with the result ID
    // In a real implementation, we would recycle the ID
}

/// Records a message.
///
/// # Arguments
///
/// * `msg` - The message to record
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__message(msg: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(msg) };
    let msg_str = c_str.to_string_lossy().into_owned();

    store_command(&QuantumCmd::Message(msg_str));
}

/// Records data.
///
/// # Arguments
///
/// * `data` - The data to record
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__record(data: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(data) };
    let data_str = c_str.to_string_lossy().into_owned();

    store_command(&QuantumCmd::Record(data_str));
}

/// Resets the QIR runtime.
///
/// This function clears all commands and measurement results.
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qir_runtime_reset() {
    let thread_id = get_thread_id();

    // Clear commands
    if let Ok(mut commands) = COMMANDS.lock() {
        commands.clear();

        if should_print_commands() {
            println!("[Thread {thread_id}] Reset QIR runtime (cleared commands)");
        }
    } else {
        // If we can't lock the mutex, print an error
        if should_print_commands() {
            eprintln!("[Thread {thread_id}] ERROR: Failed to lock command mutex during reset");
            io::stderr().flush().unwrap_or_default();
        }
    }

    // Clear measurement results
    if let Ok(mut results) = MEASUREMENT_RESULTS.lock() {
        results.clear();

        if should_print_commands() {
            println!("[Thread {thread_id}] Reset QIR runtime (cleared measurement results)");
        }
    } else {
        // If we can't lock the mutex, print an error
        if should_print_commands() {
            eprintln!(
                "[Thread {thread_id}] ERROR: Failed to lock measurement results mutex during reset"
            );
            io::stderr().flush().unwrap_or_default();
        }
    }

    // Reset qubit and result counters
    NEXT_QUBIT_ID.store(0, Ordering::SeqCst);
    NEXT_RESULT_ID.store(0, Ordering::SeqCst);

    if should_print_commands() {
        println!("[Thread {thread_id}] Reset QIR runtime (reset counters)");
    }
}

/// Gets the binary commands generated by the QIR runtime.
///
/// # Returns
///
/// A pointer to a Vec<QuantumCmd> containing the commands.
/// The caller is responsible for freeing the Vec using `qir_runtime_free_binary_commands`.
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qir_runtime_get_binary_commands() -> *mut Vec<QuantumCmd> {
    let thread_id = get_thread_id();

    // Get the commands from the global collection
    let commands = if let Ok(commands) = COMMANDS.lock() {
        // Clone the commands
        commands.clone()
    } else {
        // If we can't lock the mutex, return an empty vector
        if should_print_commands() {
            eprintln!(
                "[Thread {thread_id}] ERROR: Failed to lock command mutex during get_binary_commands"
            );
            io::stderr().flush().unwrap_or_default();
        }
        Vec::new()
    };

    // Allocate a new Vec on the heap
    let boxed_commands = Box::new(commands);

    // Convert to raw pointer and forget the Box to avoid deallocation
    // The caller is responsible for freeing the Vec using qir_runtime_free_binary_commands
    let ptr: *mut Vec<QuantumCmd> = Box::into_raw(boxed_commands);

    if ptr.is_null() {
        // Handle allocation failure
        if should_print_commands() {
            eprintln!("[Thread {thread_id}] ERROR: Failed to allocate memory for binary commands");
            io::stderr().flush().unwrap_or_default();
        }
    } else if should_print_commands() {
        println!("[Thread {thread_id}] Got binary commands");
    }

    ptr
}

/// Frees a Vec<QuantumCmd> allocated by `qir_runtime_get_binary_commands`.
///
/// # Arguments
///
/// * `ptr` - The pointer to the Vec to free
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qir_runtime_free_binary_commands(ptr: *mut Vec<QuantumCmd>) {
    let thread_id = get_thread_id();

    // Convert the raw pointer back to a Box and drop it
    if ptr.is_null() {
        if should_print_commands() {
            eprintln!("[Thread {thread_id}] ERROR: Attempted to free null binary commands pointer");
            io::stderr().flush().unwrap_or_default();
        }
    } else {
        let _ = unsafe { Box::from_raw(ptr) };

        if should_print_commands() {
            println!("[Thread {thread_id}] Freed binary commands");
        }
    }
}

/// Records a result output.
///
/// # Arguments
///
/// * `result` - The result ID to record
/// * `name` - The name to record the result as, or null for default naming
///
/// # Safety
///
/// This function is called from C/C++ code. It is safe to call but marked as unsafe
/// due to the FFI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_record_output(result: usize, name: *const c_char) {
    let thread_id = get_thread_id();

    // Generate a name for the result
    let name_str = if name.is_null() {
        // If name is null, use a default name based on the result ID
        format!("result_{result}")
    } else {
        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(name) };
        c_str.to_string_lossy().into_owned()
    };

    if should_print_commands() {
        println!("[Thread {thread_id}] Recording result {result} as '{name_str}'");
    }

    store_command(&QuantumCmd::RecordResult(ResultId(result), name_str));
}
