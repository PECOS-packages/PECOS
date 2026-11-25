//! The compatibility gate must run before any runtime-owned data is touched.

use super::*;
use crate::test_env::run_test_in_child;

fn probe(mode: &str, test: &str) {
    const CHILD_ENV: &str = "PECOS_TEST_QIS_ABI_PROBE";
    if std::env::var_os(CHILD_ENV).is_none() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("probe.rs");
        let library = temp.path().join(platform_library_name("pecos_qis_ffi"));
        let mut words = pecos_qis_ffi_types::QIS_FFI_ABI_FINGERPRINT;
        if mode == "mismatch" {
            // A later word must be compared too, not just a version prefix.
            words[3] ^= 1;
        }
        let handshake = if mode == "missing" {
            String::new()
        } else {
            format!(
                "#[unsafe(no_mangle)] pub extern \"C\" fn pecos_qis_ffi_abi_fingerprint_v1(i: u32) -> u64 {{ \
                 let words: [u64; 4] = {words:?}; words.get(i as usize).copied().unwrap_or(0) }}"
            )
        };
        let handshake = format!(
            "{handshake}\n#[unsafe(no_mangle)] pub extern \"C\" fn pecos_qis_ffi_abi_version() -> u64 {{ {} }}",
            pecos_qis_ffi_types::QIS_FFI_ABI_VERSION
        );
        // Reaching the old ownership-transferring export would kill the probe.
        std::fs::write(&source, format!(
            "{handshake}\n#[unsafe(no_mangle)] pub extern \"C\" fn pecos_qis_get_operations() -> *mut u8 {{ std::process::abort() }}"
        )).unwrap();
        let output = Command::new("rustc")
            .args(["--edition=2024", "--crate-type=cdylib"])
            .arg(&source)
            .arg("-o")
            .arg(&library)
            .output()
            .expect("compile primitive ABI fixture");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        run_test_in_child(
            test,
            &[
                (CHILD_ENV, mode.as_ref()),
                ("PECOS_QIS_FFI_PATH", library.as_os_str()),
            ],
        );
        return;
    }
    let selected = std::env::var("PECOS_QIS_FFI_PATH").unwrap();
    // The selected fixture is authoritative: never silently fall back to the
    // real runtime when it is incompatible. Failure is pinned for this process.
    for _ in 0..2 {
        let result = QisHeliosInterface::get_qis_ffi_lib_singleton();
        if mode == "matching" {
            assert!(result.is_ok(), "matching fingerprint must be accepted");
        } else {
            let error = match result {
                Ok(_) => panic!("incompatible runtime was accepted"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(&selected), "{error}");
            assert!(error.contains("build fingerprint"), "{error}");
            assert!(error.contains("Rebuild pecos-qis-ffi"), "{error}");
            assert!(error.contains("restart the process"), "{error}");
            if mode == "missing" {
                assert!(
                    error.contains("missing pecos_qis_ffi_abi_fingerprint_v1"),
                    "{error}"
                );
            } else {
                assert!(
                    error.contains("expected ") && error.contains("found "),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn abi_guard_rejects_missing_handshake() {
    probe(
        "missing",
        "executor::abi_tests::abi_guard_rejects_missing_handshake",
    );
}

#[test]
fn abi_guard_rejects_mismatched_build() {
    probe(
        "mismatch",
        "executor::abi_tests::abi_guard_rejects_mismatched_build",
    );
}

#[test]
fn abi_guard_accepts_matching_build() {
    probe(
        "matching",
        "executor::abi_tests::abi_guard_accepts_matching_build",
    );
}

#[test]
fn abi_guard_accepts_current_runtime() {
    QisHeliosInterface::get_qis_ffi_lib_singleton().expect("matching built runtime");
}
