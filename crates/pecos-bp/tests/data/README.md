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
