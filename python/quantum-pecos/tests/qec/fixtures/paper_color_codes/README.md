# Independent reference fixtures

The 488 circuits were generated with the unmodified Apache-2.0 Zenodo release
10.5281/zenodo.22820614 (`denser_planar_color_codes_source_v1.zip`), using
`denser_codes.build_memory("planar_488", d, basis, rounds=3, start_phase=phase,
gate_set=gate_set)`. Cases cover d=3,7, X/Z, both phases and CNOT/native-CZ.
Generation used Stim 1.15.0; these are reconstruction oracles, not claims of
bytewise equality with the paper's Stim 1.16.0 serialization. Cite Noah Shutty,
*Denser Planar Color Codes*, arXiv:2609.21376v3. See the production
`pecos/qec/color/_shutty488/NOTICE.md` and LICENSE for upstream attribution.

The 666 geometry was captured from the pre-existing local
`output/color-code-alpha-comparison/src/color_code/qeccColorCode.py`, separately
from the new production implementation. This is a migration oracle, not an
author-distributed circuit bundle. Circuit semantics are independently tested
against projective Pauli measurements on Choi states. The conventions follow
Lavasani et al., arXiv:2610.24077v1, Fig. 1 and Supplement B.

SHA-256 hashes of the circuit fixtures are recorded in sha256.json.

`lavasani_memory_statistics.json` transcribes the numerical hardware counts,
rates, and uncertainties in Table I of Lavasani et al.,
[*Quantum error correction well below the color code threshold*,
arXiv:2610.24077v1](https://arxiv.org/abs/2610.24077v1). It retains the source LaTeX
SHA-256 and displayed precision. These are published counts, not new simulations.
The supplementary analysis specifies pooling counts, z=1 Wilson intervals, and
division by r+1/2. One displayed central value differs from that recipe at its
shown precision: d5 Frontier reports 4.0e-5, whereas 18/(21600*20.5) is
4.0650406504e-5. Both values are retained; the estimator follows the counts and
stated normalization. All other central values and all reported uncertainty
increments agree within displayed rounding precision.
