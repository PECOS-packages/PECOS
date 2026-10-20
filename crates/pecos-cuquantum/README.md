# pecos-cuquantum

Safe Rust wrappers for NVIDIA cuQuantum, integrated with PECOS simulator traits.
The crate exposes `CuStateVec`, `CuDensityMat`, `CuTensorNet`, and
`CuFrameSimulator` for the corresponding cuQuantum backends. `CuFrameSimulator`
provides raw Pauli-frame simulation.

Running these backends requires a compatible NVIDIA GPU and the CUDA and cuQuantum
runtime libraries. Libraries are loaded at runtime through `pecos-cuquantum-sys`.
Use `is_cuquantum_available()` to probe library availability, or the backend-specific
`is_custatevec_usable()` and related helpers to check whether a backend can
initialize on the current machine.

See the [crate documentation](src/lib.rs) for a state-vector example and the
[FFI crate](../pecos-cuquantum-sys/README.md) for library configuration.
