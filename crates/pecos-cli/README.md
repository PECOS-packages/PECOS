# pecos-cli

PECOS CLI for dependency management, CUDA-aware builds, and system inspection.

Install with `cargo install --path crates/pecos-cli` or `just install-cli`.

The binary is named `pecos`. Run `pecos --help` for usage.

With the `runtime` feature enabled, `pecos run program.ll --qubits 2` runs a QIS
program with a fixed two-qubit simulator. QIS programs require `--qubits N` for
both `--sim statevector` and `--sim stabilizer`; choose a capacity covering all
runtime allocations. QASM and PHIR-JSON programs can infer their static count.
Addressing beyond the configured capacity returns an error and never resizes or
resets the simulator.
