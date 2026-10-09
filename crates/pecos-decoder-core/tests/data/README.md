# Test fixtures

Produced by the reference Python implementation of the telescoping decoder
of arXiv:2607.28795 (github.com/a7b/yarn, MIT): its `circuit_to_dem`,
`circuit_xz_detector_mask`, `gari_transform` (arXiv:2510.14060), and
`derive_init_det_system`. Do not regenerate or edit them in an
implementation change.

Priors are stored as f64 bit patterns in decimal strings (`priors_bits`,
`relevant_priors_bits`, `p_bits`): a JSON float round trip is not
guaranteed exact.

## toy_xz_d3 (X memory)

`examples/toy_xz_surface_code_memory.stim` from that package: a distance-3,
three-round rotated surface-code X-memory experiment, flattened. 24
detectors (16 X-type), 1 observable, 53 error lines, no duplicates. The
circuit is Stim-generated, so its detector order is not the canonical
XZ-memory layout.

- `toy_xz_d3.dem`, `toy_xz_d3_mask.txt` (one `X`/`Z` per detector)
- `toy_xz_d3_gari_expected.json`: X-init GARI; `h_rows` / `l_rows` are the
  column indices of each row, columns `[eZ | eX | eY | ebarZ | ebarX]`,
  rows `[detectors | U | V]`; block bounds, answer block, relevant rows and
  priors, `u_map` / `v_map`.
- `toy_xz_d3_init_dets_expected.json`: X-family restriction, columns as
  `(dets, obs, p_bits)` in first-occurrence order.
- `toy_xz_d3_zfamily_init_dets_expected.json`: the Z-family restriction of
  the same model.
- `toy_xz_d3_dup.dem`, `toy_xz_d3_dup_gari_expected.json`,
  `toy_xz_d3_dup_init_dets_expected.json`: every error line repeated
  (every third one three times), 124 lines, to exercise the merge path.

## toy_xz_d3_zinit (Z memory)

`stim.Circuit.generated("surface_code:rotated_memory_z", distance=3,
rounds=3, ...)` with 0.001 depolarizing, data, measurement, and reset
noise, through the same functions with Z initialization. 24 detectors (8
X-type), 1 observable, 219 error lines.

- `toy_xz_d3_zinit.dem`, `toy_xz_d3_zinit_mask.txt`
- `toy_xz_d3_zinit_gari_expected.json` (answer block `ebarX`)
- `toy_xz_d3_zinit_init_dets_expected.json` (Z-family restriction)

## xz_memory_masks_expected.json

The canonical XZ-memory detector mask for several `(n_x, n_z, rounds,
init_basis)` layouts, from the reference `xz_detector_type_mask`.
