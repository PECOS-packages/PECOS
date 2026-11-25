# Released SI1000 circuits and physical witnesses

These 18 circuit files are copied **unchanged** from `circuits/simulated/` in
Noah Shutty's Apache-2.0 source release for
[*Denser Planar Color Codes*, arXiv:2609.21376v3](https://arxiv.org/abs/2609.21376v3),
[DOI 10.5281/zenodo.22820614](https://doi.org/10.5281/zenodo.22820614).
`witnesses.json` retains each original circuit path, SHA-256, physical locator,
and witness provenance from the release's `distance/witnesses.json`, adding only
the local fixture filename. The source release's license and notices are also
retained with the existing vendored Shutty modules in this repository.

Coverage: planar d=3, r=3, both bases, both starting phases, p=.0005/.001;
periodic d=4/8, r=d, both bases and both rates; periodic d=12, r=12, both bases,
p=.0005. The files are independent of the PECOS noise emitter.

Tests compare every gate/noise channel in per-wire order within each moment,
and all undecomposed detector/logical error supports and probabilities. Annotation
placement, commuting instruction batching, and PECOS's synthetic constant record
do not alter these comparisons. The planar adapter now exports the public generator's detector coordinates.
`planar_coordinates.json` extracts them from the unchanged released p=.001
circuits for d=3,7,11,15, both bases and phases, retaining each source SHA-256.
`benchmark_statistics.json` contains the released 116 per-instance count rows
(with unused provenance fields omitted), all 50 plot-point rows, and the decoder
configuration, with hashes of the original data files. Their Apache-2.0 release
license is retained with the vendored Shutty modules.
`benchmark_circuits.json` retains the original inventory hash and 42 supported
planar/periodic 488 circuit filenames, parameters, and hashes. The larger circuit
files remain in the source release and can be selected with `--reference-dir` in
the benchmark example; this manifest does not regenerate reference circuits.

All 18 witnesses are replayed against constructed PECOS schedules. Twelve small
cases additionally exclude faults through weight 2 (planar d3) or 3 (periodic d4),
matching physical upper and relaxed-support lower bounds. The d8/d12 results are
upper bounds only. General distance preservation and decoder error-rate curves
are not established by these tests. Tests run with the installed Stim version;
the publication pins Stim 1.16.0.
