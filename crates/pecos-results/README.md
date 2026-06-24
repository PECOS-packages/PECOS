# pecos-results

Result types for PECOS quantum program execution: `Shot`, `ShotVec`,
`ShotMap`, `Data`, and `DataVec`.

These types are the shared result contract between PECOS simulation stacks
(`pecos-engines`, `pecos-neo`) and their language bindings. They carry named
registers with flexible values (integers, floats, bit vectors, JSON) in
row-based (`ShotVec`) or columnar (`ShotMap`) form, with conversions between
the two and display/export utilities.

This crate is deliberately free of any execution-protocol or simulator
dependencies so that any producer can emit results in this format.
