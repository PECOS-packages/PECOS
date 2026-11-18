# Attribution

Selected unchanged files from Noah Shutty, *Denser Planar Color Codes*,
arXiv:2609.21376v3, https://arxiv.org/abs/2609.21376v3.
Source release: https://doi.org/10.5281/zenodo.22820614,
`denser_planar_color_codes_source_v1.zip`.

The release authors selected Apache-2.0; the full LICENSE is included.
`geometry.py` is from `src/vendor/_triangle488`; `cx_rule.py`,
`binary.py`, `ideal_css.py`, and `templates.json` are from its `reference`
subdirectory, relocated together without content changes. Original paths
and hashes are in provenance.json. The original release NOTICE.md follows.

---

# Licensing and attribution

The release authors have selected the Apache License, Version 2.0 for this
source release; see `LICENSE`. This license applies to the released material,
not to the rest of the development repository, which is not distributed here.

`src/vendor/gen/` and the four nonempty implementation files in
`src/vendor/clorco/` are selected Python circuit-construction and compilation
sources from Google Quantum AI's Chromobius, revision
`cfcec5333856f2e7a07f0a4fced01dd9d2d418a3` (v1.0.0). Their original
Copyright 2023 Google LLC notices and Apache-2.0 license headers are retained.
The release namespace initializer files are newly written. Upstream source:
<https://github.com/quantumlib/chromobius/tree/cfcec5333856f2e7a07f0a4fced01dd9d2d418a3>.
No Chromobius C++ decoder implementation is included.

The `_triangle488`, `_torus488`, and `_torus488_cnot` packages are the selected
construction, ideal-instrument algebra, and native-scheduling sources used by
this work, adapted to package-relative imports. Standalone research drivers,
large distance searches, cluster controls, and raw samples are not part of the
release. `distance/` contains a release-authored minimal checker and selected
physical-fault witnesses, with their source locations recorded in
`distance/witnesses.json`. `_coordinates.py` and `_native_lowering.py` contain
only the shared coordinate convention and physical RX/MX lowering functions.
The release-authored `_baseline_torus` package adapts the primitive periodic
tilings, CNOT schedules, and ideal-flow construction for the toric and 6.6.6
torus families. Its `SOURCES.json` identifies the development sources; no
general-purpose geometry framework or experiment controller is included.
Original file hashes and explicit adaptation descriptions are in
`source_provenance.json`; scientific source revisions and circuit identities
are recorded separately in `data/` and `circuits/reconstruction_manifest.json`.

Stim and Sinter are external Apache-2.0 dependencies; NumPy is an external
BSD-3-Clause dependency and SciPy is an external BSD-3-Clause dependency.
They are installed separately, not vendored in this release. Their complete
license notices accompany the installed distributions. Tesseract is also not
vendored; its version and configuration are recorded with the statistics.
