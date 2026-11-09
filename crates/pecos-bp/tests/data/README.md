# Test fixtures

## gross_144_12_12

Circuit-level detector error model of the [[144,12,12]] bivariate bicycle
("gross") code and 100 sampled shots, converted from the test data shipped
with the `relay-bp` 0.2.2 crate (IBM, Apache-2.0,
https://github.com/trmue/relay). The conversion is exact: one `error(p)`
line per column of the decoding matrix, detector targets from the check
matrix, observable targets from the logical action matrix, priors copied
as stored.

- `gross_144_12_12.dem`: 1008 detectors, 8785 error mechanisms, 12
  observables. One column has prior 0. Detectors D936 to D1007 are touched
  by no mechanism in the source data (all-zero rows of the check matrix);
  they are kept as `detector(0, 0, 0) Dk` declarations so the matrix keeps
  its 1008 rows and the shot strings stay aligned.
- `gross_144_12_12_shots.txt`: one shot per line, `detectors observables`,
  each a string of `0`/`1` characters in index order. The observable
  string is the true logical flip `A @ e mod 2` for the sampled error `e`.

These files are the oracle for the native Relay-BP parity tests. Do not
regenerate or edit them in an implementation change; regenerate only from
the source data with a note here.

## relay_oracle

Frozen outputs of the external `relay-bp` 0.2.2 decoder (through the former
`pecos-relay-bp` wrapper) on the 100 `gross_144_12_12` shots, one file per
configuration. They are the oracle for `tests/relay_parity.rs` now that the
external crate is no longer a dependency.

Generated on base commit 100c8df25 by the ignored test
`tests/relay_oracle_dump.rs` (removed with the dependency; a copy is kept in
the pecos-docs vault at `design/assets/native-relay-bp-swap/relay_oracle_dump.rs`):

    RELAY_ORACLE_OUT=$PWD/crates/pecos-bp/tests/data/relay_oracle \
      cargo test --release -p pecos-bp --test relay_oracle_dump -- --ignored --nocapture

Format, per file:

- `# key=value` header lines give the full decoder configuration.
  `shared_gamma_vector(n, seed)` is the LCG helper in `tests/relay_parity.rs`.
  Each external decoder was built once per file and decoded shots 0..99 in
  order, so its gamma RNG advanced across shots.
- Then exactly 100 lines, one per shot in order:
  `<converged 0|1> <iterations> <correction>`, where `<correction>` is the
  ascending comma-separated list of mechanism indices set to 1, or `-`.

`c9_preiter_pre8_relay` and `c10_dynamic_alpha_relay` were added after review
from the same base commit with a variant of that test that generates only those
two configurations and sets `alpha_iteration_scaling_factor` per case (vault
copy: `design/assets/native-relay-bp-swap/relay_oracle_dump_c9_c10.rs`).

Do not edit or regenerate these files in an implementation change.
