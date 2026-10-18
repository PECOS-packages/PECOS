# Graphlike projection fixtures

Produced by an independent reference implementation of the graphlike projection contract with opaque-label semantics, written in Python and kept outside this repository. Do not regenerate or edit them in an implementation change.

Each case has four files:

- `<name>.dem`: a raw detector error model built by PECOS from a surface-code circuit. It is flat, with one component per error.
- `<name>.sectors`: one opaque token per detector, in detector-id order; `-` means unset.
- `<name>.expected`: one row per error line that has more than two detectors, under the `drop` policy. A row is `<error index>\t<components>` or `<error index>\tdrop`, where the error index counts `error(...)` lines from 0 and components are written in canonical order, joined by ` ^ `. Error lines with at most two detectors pass through unchanged and have no row.
- `<name>.refuse.txt`: `ok`, or the error kind expected under the `refuse` policy (`irreducible`).

Six cases (p = 0.004 uniform depolarizing unless noted; three rounds):

- `builder_d3_z`: the abstract surface-memory builder path, Z basis.
- `builder_szz_d3_x`: the same builder with the SZZ interaction basis, X basis.
- `memory_d3_z`: a single-patch memory from the logical-circuit builder, Z basis.
- `h_d3_z`: memory, transversal H, memory. The code's own family is kept across the H.
- `cx_d3_z`: two patches with a transversal CX between memory blocks, Z basis. It has 14 irreducible hyperedges.
- `memory_d3_z_three_sectors`: the memory case labelled with opaque `XX`/`YY`/`ZZ` sectors and some unset detectors.
