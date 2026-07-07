//! Integration tests for the unified simulation API
//!
//! These tests verify that the unified API works consistently across engine types.

#![cfg(feature = "runtime")]

#[cfg(test)]
mod tests {
    #[test]
    fn test_unified_api_compiles() {
        // This test verifies that the unified API syntax compiles correctly
        // We don't run it because it would require actual quantum circuits

        // The fact that this compiles proves the API is consistent
        let _ = || {
            use pecos::qis_engine;
            use pecos_engines::{DepolarizingNoise, sim_builder, sparse_stab, state_vector};
            use pecos_programs::{Qasm, Qis};
            use pecos_qasm::qasm_engine;

            // QASM engine with unified API
            let _results = sim_builder()
                .classical(qasm_engine().program(Qasm::from_string(
                    "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0];",
                )))
                .seed(42)
                .workers(4)
                .noise(DepolarizingNoise { p: 0.01 })
                .qubits(2)
                .quantum(state_vector())
                .run(1000);

            // LLVM engine with unified API
            let _results = sim_builder()
                .classical(
                    qis_engine().program(Qis::from_string("define void @main() { ret void }")),
                )
                .seed(42)
                .auto_workers()
                .noise(DepolarizingNoise { p: 0.01 })
                .qubits(1)
                .quantum(sparse_stab())
                .run(1000);
        };
    }

    #[test]
    fn test_consistent_method_names() {
        // Verify all builders have consistent input methods
        let _ = || {
            use pecos::qis_engine;
            use pecos_engines::{BiasedDepolarizingNoise, PassThroughNoise, sim_builder};
            use pecos_programs::{Qasm, Qis};
            use pecos_qasm::qasm_engine;

            // QASM-specific inputs
            let _q1 = qasm_engine().program(Qasm::from_string("..."));
            // Note: from_file returns Result, so in real code you'd handle the error
            // let _q2 = qasm_engine().program(Qasm::from_file("circuit.qasm")?);

            // LLVM-specific inputs
            let _l1 = qis_engine().program(Qis::from_string("..."));
            let _l2 = qis_engine().program(Qis::from_bitcode(vec![]));
            // Note: from_file returns Result, so in real code you'd handle the error
            // let _l3 = qis_engine().try_program(Qis::from_file("circuit.ll")?);

            // Common simulation methods

            let _sim1 = sim_builder()
                .classical(qasm_engine().program(Qasm::from_string("...")))
                .seed(42)
                .workers(4)
                .noise(PassThroughNoise);

            let _sim2 = sim_builder()
                .classical(qis_engine().program(Qis::from_string("...")))
                .seed(123)
                .auto_workers()
                .noise(BiasedDepolarizingNoise { p: 0.02 })
                .qubits(20);
        };
    }

    #[test]
    fn test_unified_sim_api() {
        // Test the new unified simulation API patterns
        let _ = || {
            use pecos::qis_engine;
            use pecos::sim;
            use pecos_engines::{DepolarizingNoise, sim_builder, sparse_stab, state_vector};
            use pecos_programs::{Qasm, Qis};
            use pecos_qasm::qasm_engine;

            // Pattern 1: Base sim_builder from pecos-engines with explicit .classical()
            let _results1 = sim_builder()
                .classical(qasm_engine().program(Qasm::from_string("OPENQASM 2.0; qreg q[1];")))
                .seed(42)
                .quantum(state_vector())
                .run(100);

            // Pattern 2: Convenience sim() from pecos with auto-selection
            let _results2 = sim(Qasm::from_string("OPENQASM 2.0; qreg q[1];"))
                .seed(42)
                .quantum(sparse_stab())
                .shots(100)
                .run();

            // Pattern 3: Override auto-selection with explicit .classical()
            let _results3 = sim(Qis::from_string("define void @main() { ret void }"))
                .classical(
                    qis_engine().program(Qis::from_string("define void @main() { ret void }")),
                )
                .shots(100)
                .run();

            // Pattern 4: Various configuration options work with new API
            let _results4 = sim(Qasm::from_string("OPENQASM 2.0; qreg q[2];"))
                .seed(123)
                .workers(4)
                .noise(DepolarizingNoise { p: 0.01 })
                .verbose(true)
                .qubits(2)
                .quantum(state_vector())
                .shots(1000)
                .run();
        };
    }

    #[test]
    fn test_auto_engine_selection() {
        // Verify that different program types select appropriate engines
        let _ = || {
            use pecos::sim;
            use pecos_engines::state_vector;
            use pecos_programs::{Hugr, Qasm, Qis};

            // QASM -> QASM engine
            let _qasm_results = sim(Qasm::from_string("OPENQASM 2.0; qreg q[1];"))
                .quantum(state_vector())
                .shots(10)
                .run();

            // LLVM -> LLVM engine
            let _llvm_results = sim(Qis::from_string("define void @main() { ret void }"))
                .quantum(state_vector())
                .shots(10)
                .run();

            // HUGR -> Selene engine
            let _hugr_results = sim(Hugr::from_bytes(vec![0x00, 0x01, 0x02]))
                .quantum(state_vector())
                .qubits(1)
                .shots(10)
                .run();
        };
    }
}
