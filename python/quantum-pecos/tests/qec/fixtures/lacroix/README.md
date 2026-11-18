# Lacroix reference circuits

These 16 files are unchanged ideal Stim circuits from Google Quantum AI,
*Data for "Scaling and logic in the color code on a superconducting quantum
processor"*, version 1.0.0, DOI [10.5281/zenodo.14238944](https://doi.org/10.5281/zenodo.14238944).
They cover `dataset_0`, all four patches, both X/Z bases, and three/five rounds.
`provenance.json` records exact archive paths and SHA-256 hashes. The ZIP entries
were fetched with HTTP Range and checked against their ZIP CRC32 and size;
the full 738.5 MB archive was not downloaded. These data are attributed to the
original authors; no additional license grant for the source dataset is asserted.

Cite N. Lacroix et al., [arXiv:2412.14256v1](https://arxiv.org/abs/2412.14256v1),
Supplement C.3/D.1. These fixtures are source data, not regenerated outputs of
PECOS. Tests compare all per-wire gate order and canonical measurement ordinals,
every detector, and the accumulated logical observable against them.

`extract_templates.py` derives `src/pecos/qec/color/lacroix_templates.json` from
only the three-round files. The five-round references are independent checks
of repeating the recurring round. Run the script with the downloaded
`reference-data/dataset_0` directory to reproduce the tables. Production loads
the JSON with the standard library; Stim is only used by this extraction tool
and the reference tests.

Normalization sets every sweep bit to zero, explicitly prepares initially-zero
active qubits, removes spectator-only resets, collapses consecutive fresh-zero
resets, removes identity and empty timing slots, and separates final readout
into its own gadget. These preserve the ideal instrument but alter physical
noise locations and timing. The tables are therefore not substitutes for the
original circuits when reproducing SI1000 or device experiments. X-memory
boundary Hadamards and all detector/observable parities are retained.
