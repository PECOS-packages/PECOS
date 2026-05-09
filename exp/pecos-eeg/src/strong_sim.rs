// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Approximate strong simulation using EEG generators.
//!
//! Computes approximate outcome probabilities p̃_x for arbitrary bit strings x,
//! using the first-order Taylor expansion from Miller et al. (Eq. 17):
//!
//!   p̃_x = p_x + (1/2^ζ) Σ_G α(ψ,G,x) ε_G + O(ε²)
//!
//! where α(ψ,G,x) = 2^ζ Tr(|x⟩⟨x| G[|ψ⟩⟨ψ|]) encodes how each generator
//! affects the probability of outcome x.
//!
//! At first order (l=1), only S-type generators contribute (H-type contributes
//! at second order). The S-type α is:
//!   α(x, S_P, ψ) = [x⊕a ∈ support(ψ)] - [x ∈ support(ψ)]
//! where a is the X-component of P.

use crate::Bm;
use crate::circuit::PropagatedEeg;
use crate::eeg::EegType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;

/// Result of approximate strong simulation for a specific outcome.
#[derive(Clone, Debug)]
pub struct OutcomeProbability {
    /// Noiseless probability p_x = |⟨x|ψ⟩|².
    pub noiseless: f64,
    /// First-order S-type correction.
    pub s_correction: f64,
    /// Second-order H·H correction (via C-type α).
    pub h_correction: f64,
    /// Total approximate probability: noiseless + corrections.
    pub total: f64,
}

/// Compute the approximate probability of outcome x at first order.
///
/// The outcome is a bit string (true = |1⟩, false = |0⟩) for each measured qubit.
/// The generators should be propagated to the end of the expanded circuit.
///
/// At first order, only S-type generators contribute:
///   α(x, S_P, ψ) = [x⊕a ∈ support] - [x ∈ support]
/// where a is the X-component of P and "support" is the set of computational
/// basis states with nonzero amplitude in |ψ⟩.
///
/// For a stabilizer state |ψ⟩ on n qubits: x is in the support iff
/// all stabilizer generators have eigenvalue +1 on |x⟩.
///
/// # Arguments
/// * `generators` - Propagated EEG generators at end of circuit
/// * `outcome` - Bit string x (one bool per qubit)
/// * `stabilizers` - Stabilizer generators of |ψ⟩ as Bm
///
/// # Limitations
/// Currently computes first-order (S-type) corrections only. H-type
/// corrections require second-order computation with phase tracking.
#[must_use]
pub fn outcome_probability(
    generators: &[PropagatedEeg],
    outcome: &[bool],
    stabilizers: &[Bm],
) -> OutcomeProbability {
    let n = outcome.len();

    // Check if x is in the support of |ψ⟩.
    // x ∈ support iff ⟨x|S|x⟩ = +1 for all stabilizer generators S.
    let x_in_support = is_in_support(outcome, stabilizers);

    // Noiseless probability: 1/2^ζ if in support, 0 otherwise.
    // ζ = n - rank(stabilizer group restricted to Z-diagonal).
    // For a pure state: ζ = 0 (deterministic), p_x = 0 or 1.
    // For a projected state: ζ > 0, p_x = 1/2^ζ.
    let zeta = compute_zeta(n, stabilizers);
    let noiseless = if x_in_support {
        1.0 / (1u64 << zeta) as f64
    } else {
        0.0
    };

    // First-order S-type corrections: α(x, S_P, ψ) = [x⊕a ∈ support] - [x ∈ support]
    let mut s_correction = 0.0;
    let scale = if zeta > 0 {
        1.0 / (1u64 << zeta) as f64
    } else {
        1.0
    };

    for g in generators {
        if g.eeg_type != EegType::S {
            continue;
        }

        let x_flipped = flip_outcome(outcome, &g.label);
        let flipped_in_support = is_in_support(&x_flipped, stabilizers);

        let alpha =
            (if flipped_in_support { 1.0 } else { 0.0 }) - (if x_in_support { 1.0 } else { 0.0 });

        s_correction += scale * alpha * g.coeff;
    }

    // Second-order H·H corrections using α(x, C_{P,P'}, ψ).
    // (1/2) Σ_{P,P'} h_P h_{P'} α(x, C_{P,P'}, ψ)
    //
    // For commuting P,P': α(C) = 2 Re(Φ(P,P')) - 2 Re(Φ(PP',I))
    // For anticommuting P,P': α(C) = 2 Re(Φ(P,P')) (since {P,P'}=0 → Φ(PP',I) cancels)
    //
    // Extract stabilizer phases for Φ computation.
    let stab_phases: Vec<bool> = stabilizers
        .iter()
        .map(|_| false) // Default: all +1 stabilizers (sign info not available from Bm)
        .collect();

    let h_gens: Vec<_> = generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H)
        .collect();

    let mut h_correction = 0.0;
    for j in 0..h_gens.len() {
        for k in 0..h_gens.len() {
            let h_j = h_gens[j].coeff;
            let h_k = h_gens[k].coeff;
            let p = &h_gens[j].label;
            let q = &h_gens[k].label;

            // α(x, C_{P,Q}, ψ) for commuting P,Q:
            // = 2 Re(Φ(P,Q)) - 2 Re(Φ(PQ,I))
            let phi_pq = compute_phi(p, q, outcome, stabilizers, &stab_phases);
            let pq = p.multiply(q);
            let identity = Bm::default();
            let phi_pq_i = compute_phi(&pq, &identity, outcome, stabilizers, &stab_phases);

            let alpha = if p.commutes_with(q) {
                2.0 * phi_pq.0 - 2.0 * phi_pq_i.0
            } else {
                // Anticommuting: α(C) = 2 Re(Φ(P,Q))
                2.0 * phi_pq.0
            };

            // (1/2) h_j h_k α
            h_correction += scale * 0.5 * h_j * h_k * alpha;
        }
    }

    // First-order C and A type corrections (if any exist directly in generators).
    let mut ca_correction = 0.0;
    for g in generators {
        match g.eeg_type {
            EegType::C => {
                if let Some(ref q_label) = g.label2 {
                    // α(x, C_{P,Q}) = 2 Re(Φ(P,Q)) - Re(Φ(PQ,I) + Φ(QP,I))
                    let phi_pq = compute_phi(&g.label, q_label, outcome, stabilizers, &stab_phases);
                    let pq = g.label.multiply(q_label);
                    let qp = q_label.multiply(&g.label);
                    let phi_pq_i =
                        compute_phi(&pq, &Bm::default(), outcome, stabilizers, &stab_phases);
                    let phi_qp_i =
                        compute_phi(&qp, &Bm::default(), outcome, stabilizers, &stab_phases);
                    let alpha = 2.0 * phi_pq.0 - (phi_pq_i.0 + phi_qp_i.0);
                    ca_correction += scale * g.coeff * alpha;
                }
            }
            EegType::A => {
                if let Some(ref q_label) = g.label2 {
                    // α(x, A_{P,Q}) = 2 Im(Φ(Q,P)) + Im(Φ(QP,I) - Φ(PQ,I))
                    let phi_qp = compute_phi(q_label, &g.label, outcome, stabilizers, &stab_phases);
                    let qp = q_label.multiply(&g.label);
                    let pq = g.label.multiply(q_label);
                    let phi_qp_i =
                        compute_phi(&qp, &Bm::default(), outcome, stabilizers, &stab_phases);
                    let phi_pq_i =
                        compute_phi(&pq, &Bm::default(), outcome, stabilizers, &stab_phases);
                    let alpha = 2.0 * phi_qp.1 + (phi_qp_i.1 - phi_pq_i.1);
                    ca_correction += scale * g.coeff * alpha;
                }
            }
            _ => {}
        }
    }

    let total = (noiseless + s_correction + h_correction + ca_correction).clamp(0.0, 1.0);

    OutcomeProbability {
        noiseless,
        s_correction,
        h_correction,
        total,
    }
}

/// Compute Φ_{ψ,x}(P,Q) = 2^ζ ⟨x|P|ψ⟩⟨ψ|Q|x⟩ for stabilizer states.
///
/// Returns a complex number as (real, imag). Result is in {0, ±1, ±i}.
///
/// Uses: Φ(P,Q) = phase(P·S_0·Q) · (-1)^{z_{PS_0Q}·x} when conditions met.
/// S_0 is any stabilizer with x_part = x_P ⊕ x_Q.
fn compute_phi(
    p: &Bm,
    q: &Bm,
    outcome: &[bool],
    stabilizers: &[Bm],
    stabilizer_phases: &[bool], // true = -1 sign (minus stabilizer)
) -> (f64, f64) {
    // ⟨x|P|ψ⟩ is nonzero iff x⊕a_P is in the support of |ψ⟩.
    // ⟨ψ|Q|x⟩ is nonzero iff x⊕a_Q is in the support.
    // Both must hold for Φ to be nonzero.
    let x_flip_p = flip_outcome(outcome, p);
    if !is_in_support(&x_flip_p, stabilizers) {
        return (0.0, 0.0);
    }
    let x_flip_q = flip_outcome(outcome, q);
    if !is_in_support(&x_flip_q, stabilizers) {
        return (0.0, 0.0);
    }

    // Target X-pattern for S_0: x_P ⊕ x_Q
    let target_x = p.multiply(q); // product has x_bits = p.x XOR q.x (and z, but we only use x)

    // Find a subset of generators whose X-parts XOR to target_x.x_bits
    let n = stabilizers.len();

    // Work with full Bm for GF2 ops (only X-part matters)
    let mut row_x: Vec<Bm> = stabilizers
        .iter()
        .map(|s| Bm {
            x_bits: s.x_bits.clone(),
            z_bits: smallvec::SmallVec::default(),
        })
        .collect();

    let mut selected = vec![false; n];
    let mut target = Bm {
        x_bits: target_x.x_bits.clone(),
        z_bits: smallvec::SmallVec::default(),
    };

    // GF(2) greedy elimination
    for bit in 0..outcome.len() {
        if !target.x_bits.get_bit(bit) {
            continue;
        }
        let found = row_x
            .iter()
            .enumerate()
            .find(|(_, r)| r.x_bits.get_bit(bit));
        if let Some((row_idx, _)) = found {
            // Find the original stabilizer index for this row
            // (rows may have been XOR-modified but indices track the original)
            selected[row_idx] = true;
            let pivot = row_x[row_idx].clone();
            target = target.multiply(&pivot);
            for (r, row) in row_x.iter_mut().enumerate().take(n) {
                if r != row_idx && row.x_bits.get_bit(bit) {
                    let p_clone = pivot.clone();
                    *row = row.multiply(&p_clone);
                }
            }
        } else {
            return (0.0, 0.0);
        }
    }

    if !target.is_identity() {
        return (0.0, 0.0);
    }

    // Build S_0 = product of selected generators
    let mut s0 = Bm::default();
    let mut s0_phase: u8 = 0; // i^{s0_phase}
    let mut s0_sign_minus = false;

    for i in 0..n {
        if selected[i] {
            let (prod, phase) = s0.multiply_with_phase(&stabilizers[i]);
            s0 = prod;
            s0_phase = (s0_phase + phase) % 4;
            if stabilizer_phases[i] {
                s0_sign_minus = !s0_sign_minus;
            }
        }
    }

    // S_0 has sign (-1)^{s0_sign_minus} · i^{s0_phase}
    // Compute PSQ = P · S_0 · Q
    let (ps, phase_ps) = p.multiply_with_phase(&s0);
    let (psq, phase_psq_part) = ps.multiply_with_phase(q);
    let total_phase = (phase_ps + phase_psq_part + s0_phase) % 4;

    // PSQ should be diagonal (x_part = 0)
    if !psq.x_bits.is_zero() {
        return (0.0, 0.0); // Shouldn't happen if solution found correctly
    }

    // Compute (-1)^{z_{PSQ} · x}
    let mut dot = 0u32;
    for (i, &bit) in outcome.iter().enumerate() {
        if bit && psq.z_bits.get_bit(i) {
            dot += 1;
        }
    }
    let z_sign: f64 = if dot.is_multiple_of(2) { 1.0 } else { -1.0 };

    // Total sign from S_0 being a (-1)^{sign} stabilizer
    let stab_sign: f64 = if s0_sign_minus { -1.0 } else { 1.0 };

    // Φ = i^{total_phase} · stab_sign · z_sign
    let (re, im) = match total_phase {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        3 => (0.0, -1.0),
        _ => unreachable!(),
    };

    (re * stab_sign * z_sign, im * stab_sign * z_sign)
}

/// Check if outcome x is in the support of the stabilizer state.
///
/// x ∈ support iff for every stabilizer generator S, ⟨x|S|x⟩ = +1.
/// For S = phase · X^a Z^b: ⟨x|S|x⟩ = phase · δ_{a,0} · (-1)^{b·x}
/// (since X flips bits, ⟨x|X^a|x⟩ = 0 unless a=0).
///
/// Wait: that's only for diagonal stabilizers. For non-diagonal (X or Y
/// components), ⟨x|S|x⟩ = 0 ≠ +1, so x is not in the support if any
/// stabilizer has X component. But stabilizer states CAN have X-type
/// stabilizers and still have x in the support.
///
/// The correct check: x is in the support iff for all Z-type stabilizers
/// (those with no X component), the Z eigenvalue matches.
fn is_in_support(outcome: &[bool], stabilizers: &[Bm]) -> bool {
    for stab in stabilizers {
        // Only Z-type stabilizers constrain the support
        if !stab.x_bits.is_zero() {
            continue; // Has X component — doesn't constrain Z-basis support
        }

        // Z-type stabilizer: eigenvalue = (-1)^{popcount(z_bits & x)}
        let mut parity = 0u32;
        for (i, &bit) in outcome.iter().enumerate() {
            if bit && stab.z_bits.get_bit(i) {
                parity += 1;
            }
        }
        // Stabilizer eigenvalue should be +1 on support states
        if !parity.is_multiple_of(2) {
            return false; // eigenvalue = -1, not in support
        }
    }
    true
}

/// Flip outcome bits according to the X-component of a Pauli.
fn flip_outcome(outcome: &[bool], pauli: &Bm) -> Vec<bool> {
    outcome
        .iter()
        .enumerate()
        .map(|(i, &bit)| if pauli.has_x(i) { !bit } else { bit })
        .collect()
}

/// Compute ζ = number of qubits whose Z-basis outcome is non-deterministic.
///
/// ζ = n - (number of independent Z-type stabilizer generators).
fn compute_zeta(n: usize, stabilizers: &[Bm]) -> usize {
    // Count independent Z-type stabilizers (no X component).
    // Extract Z-parts as Bm (store z in x_bits for GF2 rank).
    let z_stabs: Vec<Bm> = stabilizers
        .iter()
        .filter(|s| s.x_bits.is_zero())
        .map(|s| Bm {
            x_bits: s.z_bits.clone(),
            z_bits: smallvec::SmallVec::default(),
        })
        .collect();

    let rank = gf2_rank_bitmask(&z_stabs, n);
    n.saturating_sub(rank)
}

/// GF(2) rank of binary vectors stored as Bm x_bits.
fn gf2_rank_bitmask(vectors: &[Bm], max_bits: usize) -> usize {
    let mut rows: Vec<Bm> = vectors.to_vec();
    let mut rank = 0;

    for bit in 0..max_bits {
        if rank >= rows.len() {
            break;
        }
        if rows[rank..]
            .iter()
            .all(pecos_core::PauliBitmaskGeneric::is_identity)
        {
            break;
        }
        if let Some(pivot) = rows[rank..].iter().position(|r| r.x_bits.get_bit(bit)) {
            rows.swap(rank, rank + pivot);
            let pivot_val = rows[rank].clone();
            for (r, row) in rows.iter_mut().enumerate() {
                if r != rank && row.x_bits.get_bit(bit) {
                    *row = row.multiply(&pivot_val);
                }
            }
            rank += 1;
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xx() -> Bm {
        Bm::x(0).multiply(&Bm::x(1))
    }
    fn zz() -> Bm {
        Bm::z(0).multiply(&Bm::z(1))
    }

    #[test]
    fn test_single_qubit_z_basis() {
        // |0⟩ state: stabilizer = +Z. Outcome 0 is deterministic.
        let stabs = vec![Bm::z(0)];
        let outcome_0 = vec![false]; // |0⟩
        let outcome_1 = vec![true]; // |1⟩

        assert!(is_in_support(&outcome_0, &stabs));
        assert!(!is_in_support(&outcome_1, &stabs));

        // With S_X noise: flips |0⟩ to |1⟩
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::S,
            label: Bm::x(0),
            label2: None,
            coeff: -0.01,
            source: None,
        }];

        let p0 = outcome_probability(&gens, &outcome_0, &stabs);
        let p1 = outcome_probability(&gens, &outcome_1, &stabs);

        // p(0) ≈ 1 - 0.01 = 0.99 (S_X moves probability from 0 to 1)
        // p(1) ≈ 0 + 0.01 = 0.01
        assert!((p0.noiseless - 1.0).abs() < 1e-10);
        assert!((p0.s_correction - 0.01).abs() < 1e-10);
        assert!((p0.h_correction).abs() < 1e-10); // no H generators
        assert!((p0.total - 0.99).abs() < 0.02);

        assert!((p1.noiseless - 0.0).abs() < 1e-10);
        assert!((p1.s_correction + 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_h_type_diagonal_correction() {
        // |+⟩ state: stabilizer = +X. Z-basis outcomes 0,1 each with prob 1/2.
        // With H_Z noise (coherent Z rotation): shifts probability from 0 to 1.
        let stabs = vec![Bm::x(0)]; // |+⟩
        let outcome_0 = vec![false];
        let outcome_1 = vec![true];

        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::z(0), // H_Z: Z rotation
            label2: None,
            coeff: 0.1,
            source: None,
        }];

        let p0 = outcome_probability(&gens, &outcome_0, &stabs);
        let p1 = outcome_probability(&gens, &outcome_1, &stabs);

        // |+⟩ support: {0, 1} with ζ=1. Both outcomes in support.
        assert!((p0.noiseless - 0.5).abs() < 1e-10);
        assert!((p1.noiseless - 0.5).abs() < 1e-10);

        // H_Z diagonal α: Z flips no bits (a_Z = 0), so x⊕a = x.
        // α(S_Z) = [x ∈ supp] - [x ∈ supp] = 0 for Z-type Paulis.
        // So the H·H diagonal correction via S_P analogy should be 0
        // (Z has no X component, doesn't flip outcome bits).
        assert!((p0.h_correction).abs() < 1e-10);

        // H_X noise would flip bits:
        let gens_x = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0), // H_X
            label2: None,
            coeff: 0.1,
            source: None,
        }];
        let p0x = outcome_probability(&gens_x, &outcome_0, &stabs);
        // X flips bit: α = [1∈supp] - [0∈supp] = 1-1 = 0 (both in support)
        // So h_correction = 0 for |+⟩ with H_X too (both outcomes in support)
        assert!((p0x.h_correction).abs() < 1e-10);
    }

    #[test]
    fn test_h_type_shifts_probability() {
        // |0⟩ state with H_X noise: X flips from |0⟩ to |1⟩
        let stabs = vec![Bm::z(0)]; // |0⟩
        let outcome_0 = vec![false];
        let outcome_1 = vec![true];

        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0), // H_X
            label2: None,
            coeff: 0.1, // small angle
            source: None,
        }];

        let p0 = outcome_probability(&gens, &outcome_0, &stabs);
        let p1 = outcome_probability(&gens, &outcome_1, &stabs);

        // |0⟩: only outcome 0 is in support.
        // α(S_X) for outcome 0: [1∈supp] - [0∈supp] = 0-1 = -1
        // Diagonal H·H: h² · α = 0.01 · (-1) = -0.01
        assert!((p0.h_correction + 0.01).abs() < 1e-10);
        assert!((p0.total - 0.99).abs() < 0.02);

        // α(S_X) for outcome 1: [0∈supp] - [1∈supp] = 1-0 = +1
        // h² · α = 0.01 · 1 = +0.01
        assert!((p1.h_correction - 0.01).abs() < 1e-10);
        assert!((p1.total - 0.01).abs() < 0.02);
    }

    #[test]
    fn test_bell_state_support() {
        // |Φ+⟩ = (|00⟩+|11⟩)/√2. Stabilizers: +XX, +ZZ
        let stabs = vec![xx(), zz()];

        // Support: {00, 11} (Z-type stabilizer ZZ constrains parity)
        assert!(is_in_support(&[false, false], &stabs)); // 00: ZZ eigenvalue = (-1)^0 = +1
        assert!(is_in_support(&[true, true], &stabs)); // 11: ZZ eigenvalue = (-1)^2 = +1
        assert!(!is_in_support(&[false, true], &stabs)); // 01: ZZ eigenvalue = (-1)^1 = -1
        assert!(!is_in_support(&[true, false], &stabs)); // 10: ZZ eigenvalue = (-1)^1 = -1
    }

    #[test]
    fn test_zeta_computation() {
        // Single qubit |0⟩: 1 Z-type stabilizer, ζ = 1-1 = 0 (deterministic)
        assert_eq!(compute_zeta(1, &[Bm::z(0)]), 0);

        // Bell state: 1 Z-type stabilizer (ZZ), ζ = 2-1 = 1
        let bell_stabs = vec![xx(), zz()];
        assert_eq!(compute_zeta(2, &bell_stabs), 1);

        // |+⟩: 0 Z-type stabilizers, ζ = 1-0 = 1
        assert_eq!(compute_zeta(1, &[Bm::x(0)]), 1);
    }

    #[test]
    fn test_phi_single_qubit() {
        // |0⟩: stabilizer +Z
        let stabs = vec![Bm::z(0)];
        let phases = vec![false]; // +1 stabilizer

        // Φ(I,I) for outcome 0 (in support): should be 1
        let phi = compute_phi(&Bm::default(), &Bm::default(), &[false], &stabs, &phases);
        assert!((phi.0 - 1.0).abs() < 1e-10);
        assert!(phi.1.abs() < 1e-10);

        // Φ(I,I) for outcome 1 (not in support): should be 0
        let phi = compute_phi(&Bm::default(), &Bm::default(), &[true], &stabs, &phases);
        assert!(phi.0.abs() < 1e-10);

        // Φ(X,X) for outcome 0: ⟨0|X|0⟩² = 0 (X flips to |1⟩ which is not in support...
        // wait, x⊕a_X = 1, is |1⟩ in support? No. So Φ = 0.
        let phi = compute_phi(&Bm::x(0), &Bm::x(0), &[false], &stabs, &phases);
        assert!(phi.0.abs() < 1e-10);

        // Φ(X,X) for outcome 1: ⟨1|X|0⟩·⟨0|X|1⟩ = ⟨1|1⟩·⟨0|0⟩ = 1
        // x⊕a_X = 0, which IS in support. So Φ should be 1.
        let phi = compute_phi(&Bm::x(0), &Bm::x(0), &[true], &stabs, &phases);
        assert!(
            (phi.0 - 1.0).abs() < 1e-10,
            "Phi(X,X) at |1> for |0> state: got {phi:?}"
        );

        // Φ(Z,I) for outcome 0: ⟨0|Z|0⟩·⟨0|0⟩ = 1·1 = 1
        let phi = compute_phi(&Bm::z(0), &Bm::default(), &[false], &stabs, &phases);
        assert!((phi.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phi_bell_state() {
        // |Φ+⟩ = (|00⟩+|11⟩)/√2. Stabilizers: +XX, +ZZ. ζ = 1.
        let stabs = vec![xx(), zz()];
        let phases = vec![false, false];

        // Φ(I,I) for outcome 00 (in support): 2^ζ · |⟨00|Φ+⟩|² = 2 · 1/2 = 1
        let phi = compute_phi(
            &Bm::default(),
            &Bm::default(),
            &[false, false],
            &stabs,
            &phases,
        );
        assert!((phi.0 - 1.0).abs() < 1e-10, "Phi(I,I) at 00: {phi:?}");

        // Φ(I,I) for outcome 01 (not in support): 0
        let phi = compute_phi(
            &Bm::default(),
            &Bm::default(),
            &[false, true],
            &stabs,
            &phases,
        );
        assert!(phi.0.abs() < 1e-10);

        // Φ(Z0,I) for outcome 00
        let phi = compute_phi(&Bm::z(0), &Bm::default(), &[false, false], &stabs, &phases);
        assert!((phi.0 - 1.0).abs() < 1e-10, "Phi(Z0,I) at 00: {phi:?}");

        // Φ(Z0,I) for outcome 11
        let phi = compute_phi(&Bm::z(0), &Bm::default(), &[true, true], &stabs, &phases);
        assert!((phi.0 + 1.0).abs() < 1e-10, "Phi(Z0,I) at 11: {phi:?}");
    }

    #[test]
    fn test_bell_state_strong_sim() {
        // |Φ+⟩ with S_{Z₀} noise (Z error on qubit 0)
        let stabs = vec![xx(), zz()];

        let gens = vec![PropagatedEeg {
            eeg_type: EegType::S,
            label: Bm::z(0), // S_Z on qubit 0
            label2: None,
            coeff: -0.01,
            source: None,
        }];

        // Noiseless: p(00) = p(11) = 1/2, p(01) = p(10) = 0
        let p00 = outcome_probability(&gens, &[false, false], &stabs);
        let p11 = outcome_probability(&gens, &[true, true], &stabs);
        let p01 = outcome_probability(&gens, &[false, true], &stabs);
        let p10 = outcome_probability(&gens, &[true, false], &stabs);

        assert!((p00.noiseless - 0.5).abs() < 1e-10);
        assert!((p11.noiseless - 0.5).abs() < 1e-10);
        assert!(p01.noiseless.abs() < 1e-10);
        assert!(p10.noiseless.abs() < 1e-10);

        // S_{Z₀} on |Φ+⟩: Z₀ maps |Φ+⟩ to |Φ-⟩. No Z-basis effect.
        assert!(
            p00.s_correction.abs() < 1e-10,
            "Z error on Bell state: no Z-basis effect"
        );
        assert!(p11.s_correction.abs() < 1e-10);
    }
}
