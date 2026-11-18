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

## Experimental scheduled batch adapters

`scheduled_adapter::NativeScheduledAdapter` loads a trusted library implementing
[`pecos_scheduled_adapter.h`](include/pecos_scheduled_adapter.h). Its `with_noise`
method returns the existing `ScheduledEventNoise` capability for `QuantumSystem`.
The adapter normalizes batches; the host retains timing, noise/quantum feedback,
measurement admission and shot lifecycle. It has no simulator or RNG access.

This Rust/C interface is experimental. Python exposure and performance validation
are separate follow-ups. See the [contract](../../design/native-scheduled-adapter.md)
for pointer lifetimes, thread requirements, failure handling and remaining coverage.
Loading is unsafe: native plugins must obey that contract and contain their own
panics/exceptions. Checked records do not make arbitrary native code safe.
