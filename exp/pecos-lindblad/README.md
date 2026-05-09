# pecos-lindblad

Lindblad-to-Pauli-Lindblad noise synthesis for PECOS.

Given a per-gate Lindbladian `{H_ideal, H_err, c_ops, tau_g}`, compute the
effective Pauli-Lindblad rates `lambda_k` that feed into
`pecos-qec::DemStabSim` or any Pauli-level noise channel.

**Status:** experimental, Phase 1 (numerical baseline + 1Q identity test).

**Design docs:**
- `design/lindblad_sim_skeleton.md` -- crate layout, API surface, test plan
- `design/lindblad_magnus_algorithm.md` -- math spec, closed forms, references

**Primary reference:** Malekakhlagh et al., *Efficient Lindblad synthesis for
noise model construction*, npj QI 2025, arXiv:2502.03462.
