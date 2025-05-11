use crate::common::{get_thread_id, should_print_commands};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// Global counters for qubit and result allocation
static NEXT_QUBIT_ID: AtomicUsize = AtomicUsize::new(0);
static NEXT_RESULT_ID: AtomicUsize = AtomicUsize::new(0);

// Global storage for measurement results
static MEASUREMENT_RESULTS: std::sync::LazyLock<Mutex<HashMap<String, u32>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Allocates a new qubit and returns its ID
///
/// # Returns
///
/// The ID of the newly allocated qubit
pub fn allocate_qubit() -> usize {
    let qubit_id = NEXT_QUBIT_ID.fetch_add(1, Ordering::SeqCst);
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Allocated qubit {qubit_id}");
    }

    qubit_id
}

/// Allocates a new result and returns its ID
///
/// # Returns
///
/// The ID of the newly allocated result
pub fn allocate_result() -> usize {
    let result_id = NEXT_RESULT_ID.fetch_add(1, Ordering::SeqCst);
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Allocated result {result_id}");
    }

    result_id
}

/// Releases a qubit
///
/// # Arguments
///
/// * `qubit` - The qubit ID to release
pub fn release_qubit(qubit: usize) {
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Released qubit {qubit}");
    }

    // We don't actually do anything with the qubit ID
    // In a real implementation, we would recycle the ID
}

/// Releases a result
///
/// # Arguments
///
/// * `result` - The result ID to release
pub fn release_result(result: usize) {
    let thread_id = get_thread_id();

    if should_print_commands() {
        println!("[Thread {thread_id}] Released result {result}");
    }

    // We don't actually do anything with the result ID
    // In a real implementation, we would recycle the ID
}

/// Gets the value of a measurement result
///
/// # Arguments
///
/// * `name` - The name of the result to get
///
/// # Returns
///
/// The value of the measurement result (0 or 1)
pub fn get_measurement_result(name: &str) -> u32 {
    let thread_id = get_thread_id();

    // Get the measurement result from the global map
    if let Ok(results) = MEASUREMENT_RESULTS.lock() {
        if let Some(value) = results.get(name) {
            if should_print_commands() {
                println!("[Thread {thread_id}] Got measurement result {name} = {value}");
            }
            *value
        } else {
            if should_print_commands() {
                println!("[Thread {thread_id}] Measurement result {name} not found, returning 0");
            }
            0
        }
    } else {
        // If we can't lock the mutex, return 0
        if should_print_commands() {
            eprintln!("[Thread {thread_id}] ERROR: Failed to lock measurement results mutex");
            io::stderr().flush().unwrap_or_default();
        }
        0
    }
}

/// Sets the value of a measurement result
///
/// # Arguments
///
/// * `name` - The name of the result to set
/// * `value` - The value to set (0 or 1)
pub fn set_measurement_result(name: &str, value: u32) {
    let thread_id = get_thread_id();

    // Set the measurement result in the global map
    if let Ok(mut results) = MEASUREMENT_RESULTS.lock() {
        results.insert(name.to_string(), value);

        if should_print_commands() {
            println!("[Thread {thread_id}] Set measurement result {name} = {value}");
        }
    } else {
        // If we can't lock the mutex, print an error
        if should_print_commands() {
            eprintln!("[Thread {thread_id}] ERROR: Failed to lock measurement results mutex");
            io::stderr().flush().unwrap_or_default();
        }
    }
}

/// Clears all measurement results
pub fn clear_measurement_results() {
    let thread_id = get_thread_id();

    // Clear measurement results
    if let Ok(mut results) = MEASUREMENT_RESULTS.lock() {
        results.clear();

        if should_print_commands() {
            println!("[Thread {thread_id}] Cleared measurement results");
        }
    } else {
        // If we can't lock the mutex, print an error
        if should_print_commands() {
            eprintln!(
                "[Thread {thread_id}] ERROR: Failed to lock measurement results mutex during clear"
            );
            io::stderr().flush().unwrap_or_default();
        }
    }
}

/// Resets the qubit and result counters
pub fn reset_counters() {
    let thread_id = get_thread_id();

    // Reset qubit and result counters
    NEXT_QUBIT_ID.store(0, Ordering::SeqCst);
    NEXT_RESULT_ID.store(0, Ordering::SeqCst);

    if should_print_commands() {
        println!("[Thread {thread_id}] Reset qubit and result counters");
    }
}
