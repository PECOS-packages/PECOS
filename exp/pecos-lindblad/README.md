# pecos-lindblad

Lindblad-to-Pauli-Lindblad noise synthesis for PECOS.

Given a per-gate Lindbladian `{H_ideal, H_err, c_ops, tau_g}`, compute the
effective Pauli-Lindblad rates `lambda_k` that feed into
`pecos-qec::DemStabSim` or any Pauli-level noise channel.

**Status:** experimental, Phase 1 (numerical baseline + 1Q identity test).

**Primary reference:** Malekakhlagh et al., *Efficient Lindblad synthesis for
noise model construction*, npj QI 2025, arXiv:2502.03462.
