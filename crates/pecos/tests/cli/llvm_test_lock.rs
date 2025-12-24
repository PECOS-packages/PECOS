/// File-based lock for tests that modify shared build directories
///
/// This lock is only needed for tests that:
/// - Modify the build directory (e.g., removing cached libraries)
/// - Compile LLVM programs (which may use shared runtime build cache)
///
/// Most LLVM execution tests don't need this lock because:
/// - Each test execution uses thread-local runtime contexts
/// - The runtime library is built once and cached safely
/// - Multiple tests can execute quantum programs in parallel
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::Duration;

const MAX_RETRIES: u32 = 600; // 60 seconds total to handle test load
const RETRY_DELAY_MS: u64 = 100;

pub struct LlvmTestLock {
    _file: File,
    path: PathBuf,
}

impl LlvmTestLock {
    /// Acquire the LLVM test lock
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Failed to create the lock file due to an unexpected error
    /// - Failed to acquire the lock after maximum retries
    #[must_use]
    pub fn acquire() -> Self {
        // Use target directory for lock file to avoid /tmp issues
        let lock_dir = std::env::var("CARGO_TARGET_DIR").map_or_else(
            |_| {
                // Find the workspace root by looking for Cargo.lock
                let mut current = std::env::current_dir().unwrap();
                loop {
                    if current.join("Cargo.lock").exists() {
                        break current.join("target");
                    }
                    if !current.pop() {
                        // Fallback to current directory
                        break PathBuf::from("target");
                    }
                }
            },
            PathBuf::from,
        );

        // Ensure directory exists
        let _ = std::fs::create_dir_all(&lock_dir);
        let lock_path = lock_dir.join("pecos_llvm_test.lock");

        // Try to acquire lock with retries

        for attempt in 0..MAX_RETRIES {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(file) => {
                    eprintln!("Acquired LLVM test lock");
                    return Self {
                        _file: file,
                        path: lock_path,
                    };
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if attempt == 0 {
                        eprintln!("Waiting for LLVM test lock...");
                    }
                    std::thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
                }
                Err(e) => {
                    panic!("Failed to create LLVM test lock file: {e}");
                }
            }
        }

        panic!(
            "Failed to acquire LLVM test lock after {} seconds",
            u64::from(MAX_RETRIES) * RETRY_DELAY_MS / 1000
        );
    }
}

impl Drop for LlvmTestLock {
    fn drop(&mut self) {
        eprintln!("Releasing LLVM test lock");
        let _ = std::fs::remove_file(&self.path);
    }
}
