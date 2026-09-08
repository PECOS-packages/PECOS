# pecos-hugr

Static HUGR loading and result-tag analysis for PECOS.

The loader validates HUGR envelopes. The result-tag helpers recover structural
measurement provenance and detect nontrivial control flow for the Guppy detector
error model bindings. HUGR execution belongs to Selene and Guppy.

## Public helpers

- `load_hugr_from_bytes` and `load_hugr_from_file`
- `extract_result_tag_measurements`
- `measurement_op_count`
- `has_nontrivial_control_flow`

## Acknowledgements

This crate uses [HUGR](https://github.com/Quantinuum/hugr) (Hierarchical Unified Graph Representation), developed by Quantinuum.

**Paper:**
- Koch, M., Borgna, A., Sivarajah, S., Lawrence, A., Edgington, A., Wilson, D., Roy, C., Mondada, L., Heidemann, L., & Duncan, R. (2025). "HUGR: A Quantum-Classical Intermediate Representation." [arXiv:2510.11420](https://arxiv.org/abs/2510.11420)
