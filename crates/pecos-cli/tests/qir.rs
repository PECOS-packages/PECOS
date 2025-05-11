// use assert_cmd::prelude::*;
// use predicates::prelude::*;
// use std::path::PathBuf;
// use std::process::Command;

// #[test]
// fn test_pecos_compile_and_run() -> Result<(), Box<dyn std::error::Error>> {
//     // Requires: LLVM tools and GCC toolchain
//     // For Flatpak: Set PATH to include /usr/bin and GCC paths
//     // Attempt to run the test and gracefully handle any errors from the QIR engine
//
//     let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//     let test_file = manifest_dir.join("../../examples/qir/qprog.ll");
//
//     let mut cmd = Command::cargo_bin("pecos")?;
//     cmd.env("RUST_LOG", "info")
//         .env("PATH", "/usr/local/bin:/usr/bin:/bin")
//         .arg("compile")
//         .arg(&test_file)
//         .assert()
//         .success()
//         .stderr(predicate::str::contains("Compilation successful"));
//
//     let mut cmd = Command::cargo_bin("pecos")?;
//     cmd.env("RUST_LOG", "info")
//         .arg("run")
//         .arg(&test_file)
//         .assert()
//         .success()
//         .stderr(predicate::str::contains("Compilation successful"))
//         .stdout(predicate::str::contains("Measurement Statistics"))
//         .stdout(predicate::str::contains("Total measurements: 1"));
//
//     Ok(())
// }
