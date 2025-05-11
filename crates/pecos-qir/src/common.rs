use std::thread;

/// Helper function to get the current thread ID as a string
///
/// This function returns the current thread ID formatted as a string.
/// It's used for logging and debugging purposes.
///
/// # Returns
///
/// A string representation of the current thread ID
#[must_use]
pub fn get_thread_id() -> String {
    format!("{:?}", thread::current().id())
}

/// Helper function to check if we should print commands
///
/// This function checks the `QIR_RUNTIME_QUIET` environment variable
/// to determine if commands should be printed to stdout.
///
/// # Returns
///
/// * `true` - If commands should be printed
/// * `false` - If commands should not be printed
#[must_use]
pub fn should_print_commands() -> bool {
    match std::env::var("QIR_RUNTIME_QUIET") {
        Ok(val) => val != "1",
        Err(_) => true,
    }
}
