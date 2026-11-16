# pecos-foreign

Foreign-language plugin interfaces for PECOS decoders and simulators. Plugins
expose opaque handles and C-compatible tables of function pointers; Rust wrappers
implement the PECOS traits so callers can use plugins alongside native backends.

`ForeignDecoder` wraps the decoder interface, and `ForeignSimulator` wraps the
simulator interface. The crate also provides plugin discovery, ABI versioning,
and conformance checks. The optional `neo` feature enables integration with
`pecos-neo`.

See the [crate documentation](src/lib.rs), [decoder interface](src/decoder.rs), and
[simulator interface](src/simulator.rs) for the ABI and ownership requirements.
