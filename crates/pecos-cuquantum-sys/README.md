# pecos-cuquantum-sys

Low-level Rust bindings and runtime library loading for NVIDIA cuQuantum. The crate
provides types and dynamically loaded functions for cuStateVec, cuStabilizer,
cuTensorNet, cuDensityMat, and the CUDA runtime operations used by those backends.
For safe simulator APIs, use [pecos-cuquantum](../pecos-cuquantum/README.md).

The build script generates bindings from installed CUDA and cuQuantum headers when
available, and otherwise generates stub type definitions. Shared libraries are
loaded at runtime; using a backend requires the corresponding libraries and a
compatible NVIDIA GPU.

Set `CUQUANTUM_ROOT` to the cuQuantum installation directory. The runtime loader
also searches PECOS-managed installations and uses `CUDA_PATH` or `CUDA_HOME` when
locating CUDA libraries. `try_load()` reports library-loading errors, while
`is_available()` probes availability without creating a GPU context.

See the [runtime loader](src/loader.rs) for library discovery and the
[build script](build.rs) for binding generation.
