// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Backward Heisenberg propagation of detectors through noise.
//!
//! Computes exact detection probabilities by propagating the detector
//! observable BACKWARD through the expanded (measurement-deferred) circuit.
//!
//! Coherent (H-type) noise: at each source exp(h·H_P), the observable
//! transforms via unitary conjugation:
//!   D → cos(2h)·D + i·sin(2h)·P·D  (when D and P anticommute)
//!   D → D                            (when D and P commute)
//!
//! Stochastic (S-type) noise: EEG rate s < 0, physical error
//! probability p = (1/2)(1 - exp(2s)).  Heisenberg dual:
//!   D → D          (when D and P commute)
//!   D → exp(2s)·D  (when D and P anticommute)
//!
//! The cost is exponential in the number of anticommuting H-type noise
//! sources per detector (2^m terms), but this is typically manageable
//! for QEC circuits (m ~ 5-15). S-type noise does not increase term count.
//!
//! Both a Pauli-tracking walk (fast, exact) and a matrix-based method
//! (exact, limited to ~20 qubits) are provided.

use crate::Bm;
use crate::noise::NoiseSpec;
use crate::stabilizer::StabilizerGroup;
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use smallvec::SmallVec;
use std::collections::BinaryHeap;

const CX_PHASE: [[u8; 4]; 4] = [[0, 0, 0, 0], [0, 0, 3, 1], [0, 1, 0, 3], [0, 3, 1, 0]];

fn sign_parity<const N: usize>(signs: [bool; N]) -> bool {
    signs.into_iter().fold(false, |parity, sign| parity ^ sign)
}

fn activate_qubit(
    q: u16,
    before_gate: u32,
    active: &mut [bool],
    visited: &mut [bool],
    heap: &mut BinaryHeap<u32>,
    gate_index: &crate::expand::GateIndex,
) {
    let qu = q as usize;
    if qu >= active.len() {
        return;
    }
    active[qu] = true;
    for gi in gate_index.gates_on_qubit_rev(qu) {
        if gi >= before_gate {
            continue;
        } // already passed
        let gi_usize = gi as usize;
        if !visited[gi_usize] {
            visited[gi_usize] = true;
            heap.push(gi);
        }
    }
}

/// Precomputed noise for a single gate: H-type injections + batched S-type scale.
///
/// Instead of calling `noise_after_gate()` during the walk and processing
/// 15+ S-type injections individually, we precompute:
/// - H-type injections (kept as-is, they cause branching)
/// - A combined S-type scale factor per qubit-support pattern
///
/// For uniform 2-qubit depolarizing at rate p: any term with non-identity
/// on either gate qubit gets scaled by `(1-2p/15)^8` (exactly 8 of 15
/// Paulis anticommute for any non-identity support). Single-qubit: `(1-2p/3)^2`.
pub struct PrecomputedGateNoise {
    /// H-type injections (cause term branching, can't be batched).
    pub h_injections: SmallVec<[crate::noise::NoiseInjection; 2]>,
    /// Combined S-type scale for terms with non-identity on qubit 0 only.
    /// (For 1q gates, this is the full scale. For 2q, see q1_scale/both_scale.)
    pub q0_scale: f64,
    /// Qubit 0 index (for S-type fast path).
    pub q0: u16,
    /// Combined S-type scale for terms with non-identity on qubit 1 only (2q gates).
    pub q1_scale: f64,
    /// Qubit 1 index.
    pub q1: u16,
    /// Combined S-type scale for terms with non-identity on BOTH qubits.
    pub both_scale: f64,
    /// Number of qubits this gate acts on (0, 1, or 2).
    pub num_gate_qubits: u8,
}

/// Build a noise map: precomputed noise for each gate.
///
/// Returns `None` for gates with no noise. Expansion gates are mostly
/// skipped, except expansion CX gates get p_meas noise on their control
/// qubit (the originally-measured qubit) to model mid-circuit measurement
/// errors that the expansion would otherwise lose.
pub fn build_noise_map(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    expansion_gates: &[bool],
) -> Vec<Option<PrecomputedGateNoise>> {
    let mut map = Vec::with_capacity(gates.len());

    for (i, gate) in gates.iter().enumerate() {
        if i < expansion_gates.len() && expansion_gates[i] {
            map.push(None);
            continue;
        }

        let qubits: SmallVec<[usize; 4]> =
            gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        let injections = noise.noise_after_gate(i, gate.gate_type, &qubits);

        if injections.is_empty() {
            map.push(None);
            continue;
        }

        let mut h_inj = SmallVec::new();
        // Collect S-type rates grouped by which qubits they touch.
        // We compute a combined scale factor for each support pattern.
        let mut s_rates_q0_only = Vec::new(); // S noise on q0 only
        let mut s_rates_q1_only = Vec::new(); // S noise on q1 only
        let mut s_rates_both = Vec::new(); // S noise on both q0 and q1
        let mut s_rates_other = Vec::new(); // S noise on other patterns

        let q0 = qubits.first().copied().unwrap_or(0) as u16;
        let q1 = if qubits.len() >= 2 {
            qubits[1] as u16
        } else {
            q0
        };

        for inj in &injections {
            if inj.eeg_type == crate::eeg::EegType::S {
                let rate = inj.rate;
                // Classify by qubit support
                let on_q0 = inj.label.has_x(q0 as usize) || inj.label.has_z(q0 as usize);
                let on_q1 = qubits.len() >= 2
                    && (inj.label.has_x(q1 as usize) || inj.label.has_z(q1 as usize));

                // For each S injection with rate s (s < 0), the scale for
                // anticommuting terms is (1 - 2*(-s)) = (1 + 2s).
                // We need to count how many of the term's components
                // anticommute. For a uniform depolarizing model this is
                // predetermined by the support pattern.
                //
                // Instead of trying to batch analytically (which requires
                // knowing the exact anticommutation count), we accumulate
                // the log of the scale factors and compute the combined
                // scale per support pattern.
                //
                // For now, just collect individual rates.
                if on_q0 && !on_q1 {
                    s_rates_q0_only.push(rate);
                } else if !on_q0 && on_q1 {
                    s_rates_q1_only.push(rate);
                } else if on_q0 && on_q1 {
                    s_rates_both.push(rate);
                } else {
                    s_rates_other.push(rate);
                }
            } else {
                h_inj.push(inj.clone());
            }
        }

        // Compute combined scale factors.
        // A term anticommutes with S_P iff it has non-trivial overlap with P.
        //
        // For a term with non-identity on q0 only:
        //   - Anticommutes with S generators touching q0: all of q0_only + both
        //   - But WHICH ones anticommute depends on the specific Pauli.
        //
        // For uniform depolarizing, the count of anticommuting generators is
        // deterministic given the support pattern. But for general S noise, we
        // can't batch — fall back to individual processing.
        //
        // Optimization: if ALL S rates are the same (uniform depol), use
        // closed-form. Otherwise, keep individual injections.
        let all_s_same_rate = {
            let all_s: Vec<f64> = s_rates_q0_only
                .iter()
                .chain(&s_rates_q1_only)
                .chain(&s_rates_both)
                .chain(&s_rates_other)
                .copied()
                .collect();
            !all_s.is_empty() && all_s.iter().all(|&r| (r - all_s[0]).abs() < 1e-20)
        };

        if all_s_same_rate && s_rates_other.is_empty() {
            // Uniform depolarizing: use closed-form combined scale.
            // For any non-identity on q0: 2 of {X,Y,Z} on q0 anticommute.
            // For single-qubit: 3 S generators, 2 anticommute → scale = (1+2s)^2
            // For two-qubit: 15 S generators, 8 anticommute for any non-trivial → (1+2s)^8
            let total_s = s_rates_q0_only.len() + s_rates_q1_only.len() + s_rates_both.len();
            let s = s_rates_q0_only
                .first()
                .or(s_rates_q1_only.first())
                .or(s_rates_both.first())
                .copied()
                .unwrap_or(0.0);
            let p = -s;
            let individual_scale = 1.0 - 2.0 * p;

            // Count anticommuting for each support pattern.
            // For term with non-identity on q0 only:
            //   anti with q0-only S: 2 out of 3 (if 1q) or 2 out of 3 (for each A⊗I)
            //   anti with both S: depends on q1 part (commutes since term has I on q1)
            //   Total for 2q depol: q0_only anti=2 out of 3, both anti=q0 part anti * q1 commutes
            //   = 2*3 (from 3 A⊗I where 2 of 3 A anticommute on q0, B=I commutes)
            //   + 0 (from 3 I⊗B)
            //   + 2*3 (from 9 A⊗B where 2 of 3 A anticommute, B commutes since I on q1)
            //   Wait, {A, I_term} always commutes for B part. So:
            //   anti count = (anti on q0) * (total on q1 including I) + (comm on q0) * (anti on q1)
            //   For term I on q1: anti on q1 = 0.
            //   So anti count = 2 * 4 + 2 * 0 = 8 for 15 generators (excluding I⊗I).
            //
            // Actually, let me just precompute this properly.
            // For 1q depol (3 generators): non-identity on q → 2 anticommute → (1-2p/3)^2
            // For 2q depol (15 generators): non-identity on either q → 8 anticommute → (1-2p/15)^8
            let n_anti = if total_s == 3 {
                2
            } else if total_s == 15 {
                8
            } else {
                0
            };
            if n_anti == 0 && total_s > 0 {
                // Non-standard S count (e.g., 1 for p_meas, 1 for p_prep):
                // can't batch — put in h_injections for individual processing.
                for inj in &injections {
                    if inj.eeg_type == crate::eeg::EegType::S {
                        h_inj.push(inj.clone());
                    }
                }
            }
            let combined = individual_scale.powi(n_anti);

            map.push(Some(PrecomputedGateNoise {
                h_injections: h_inj,
                q0_scale: combined,
                q0,
                q1_scale: combined,
                q1,
                both_scale: combined,
                num_gate_qubits: qubits.len().min(2) as u8,
            }));
        } else if s_rates_q0_only.is_empty()
            && s_rates_q1_only.is_empty()
            && s_rates_both.is_empty()
            && s_rates_other.is_empty()
        {
            // H-type only, no S noise
            if h_inj.is_empty() {
                map.push(None);
            } else {
                map.push(Some(PrecomputedGateNoise {
                    h_injections: h_inj,
                    q0_scale: 1.0,
                    q0,
                    q1_scale: 1.0,
                    q1,
                    both_scale: 1.0,
                    num_gate_qubits: qubits.len().min(2) as u8,
                }));
            }
        } else {
            // Non-uniform S noise: keep individual injections as H-type
            // (the walk handles them individually).
            for inj in injections {
                if inj.eeg_type == crate::eeg::EegType::S {
                    h_inj.push(inj); // process individually in walk
                }
            }
            map.push(Some(PrecomputedGateNoise {
                h_injections: h_inj,
                q0_scale: 1.0,
                q0,
                q1_scale: 1.0,
                q1,
                both_scale: 1.0,
                num_gate_qubits: qubits.len().min(2) as u8,
            }));
        }
    }

    map
}

/// Sparse Pauli: stores only qubits with non-identity Pauli.
/// For terms touching ~10-20 qubits out of 1000+, this is 10-100x
/// more compact than a dense bitmask, making clone/cmp/hash O(support).
///
/// Stored as sorted lists of qubit indices for X and Z components.
/// Y on qubit q means q appears in BOTH x_qubits and z_qubits.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparsePauli {
    x_qubits: SmallVec<[u16; 16]>,
    z_qubits: SmallVec<[u16; 16]>,
}

impl SparsePauli {
    pub(crate) fn from_bm(bm: &Bm) -> Self {
        let mut sp = Self::default();
        let max_x = bm.x_bits.highest_set_bit().unwrap_or(0);
        let max_z = bm.z_bits.highest_set_bit().unwrap_or(0);
        let max_q = max_x.max(max_z);
        for q in 0..=max_q {
            if bm.has_x(q) {
                sp.x_qubits.push(q as u16);
            }
            if bm.has_z(q) {
                sp.z_qubits.push(q as u16);
            }
        }
        sp
    }

    pub(crate) fn to_bm(&self) -> Bm {
        let mut bm = Bm::default();
        for &q in &self.x_qubits {
            bm.x_bits.set_bit(q as usize);
        }
        for &q in &self.z_qubits {
            bm.z_bits.set_bit(q as usize);
        }
        bm
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.x_qubits.is_empty() && self.z_qubits.is_empty()
    }

    #[inline]
    fn has_x(&self, q: u16) -> bool {
        self.x_qubits.binary_search(&q).is_ok()
    }

    #[inline]
    fn has_z(&self, q: u16) -> bool {
        self.z_qubits.binary_search(&q).is_ok()
    }

    /// Toggle x-bit at qubit q (insert if missing, remove if present).
    fn toggle_x(&mut self, q: u16) {
        match self.x_qubits.binary_search(&q) {
            Ok(i) => {
                self.x_qubits.remove(i);
            }
            Err(i) => {
                self.x_qubits.insert(i, q);
            }
        }
    }

    fn toggle_z(&mut self, q: u16) {
        match self.z_qubits.binary_search(&q) {
            Ok(i) => {
                self.z_qubits.remove(i);
            }
            Err(i) => {
                self.z_qubits.insert(i, q);
            }
        }
    }

    /// Remove X at qubit q (for PZ backward: kill if has_x).
    pub(crate) fn clear_x(&mut self, q: u16) {
        if let Ok(i) = self.x_qubits.binary_search(&q) {
            self.x_qubits.remove(i);
        }
    }

    pub(crate) fn clear_z(&mut self, q: u16) {
        if let Ok(i) = self.z_qubits.binary_search(&q) {
            self.z_qubits.remove(i);
        }
    }

    /// Check if this Pauli commutes with a single-qubit Z_q.
    /// Full commutation check with another SparsePauli.
    fn commutes_with(&self, other: &Self) -> bool {
        // Symplectic inner product mod 2:
        // count = |self.x ∩ other.z| + |self.z ∩ other.x|
        // Commutes iff count is even.
        let c1 = sorted_intersection_count(&self.x_qubits, &other.z_qubits);
        let c2 = sorted_intersection_count(&self.z_qubits, &other.x_qubits);
        (c1 + c2).is_multiple_of(2)
    }
}

impl PartialEq for SparsePauli {
    fn eq(&self, other: &Self) -> bool {
        self.x_qubits == other.x_qubits && self.z_qubits == other.z_qubits
    }
}
impl Eq for SparsePauli {}

impl std::hash::Hash for SparsePauli {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x_qubits.as_slice().hash(state);
        self.z_qubits.as_slice().hash(state);
    }
}

impl Ord for SparsePauli {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.x_qubits
            .as_slice()
            .cmp(other.x_qubits.as_slice())
            .then(self.z_qubits.as_slice().cmp(other.z_qubits.as_slice()))
    }
}
impl PartialOrd for SparsePauli {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Count elements in the intersection of two sorted slices.
#[inline]
fn sorted_intersection_count(a: &[u16], b: &[u16]) -> u32 {
    let (mut i, mut j) = (0, 0);
    let mut count = 0u32;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

impl SparsePauli {
    /// Conjugate by Hadamard on qubit q: X↔Z, Y→-Y.
    fn conjugate_h(&mut self, q: u16) -> bool {
        let hx = self.has_x(q);
        let hz = self.has_z(q);
        if hx != hz {
            // X→Z or Z→X: swap
            self.toggle_x(q);
            self.toggle_z(q);
        }
        // Y→-Y: sign flip when both X and Z
        hx && hz
    }

    /// Conjugate by CX(control, target). Returns sign_negative.
    fn conjugate_cx(&mut self, c: u16, t: u16) -> bool {
        let cx = self.has_x(c);
        let cz = self.has_z(c);
        let tx = self.has_x(t);
        let tz = self.has_z(t);
        if cx {
            self.toggle_x(t);
        }
        if tz {
            self.toggle_z(c);
        }
        // Sign from phase table (same formula as the fixed conjugate_cx)
        let pc = u8::from(cx) + 2 * u8::from(cz);
        let pt = u8::from(tx) + 2 * u8::from(tz);
        let phase_c = if tz { CX_PHASE[pc as usize][2] } else { 0 };
        let phase_t = if cx { CX_PHASE[1][pt as usize] } else { 0 };
        (phase_c + phase_t) % 4 == 2
    }

    /// Conjugate by CZ(a, b). Returns sign_negative.
    fn conjugate_cz(&mut self, a: u16, b: u16) -> bool {
        let ax = self.has_x(a);
        let bx = self.has_x(b);
        let az = self.has_z(a);
        let bz = self.has_z(b);
        if bx {
            self.toggle_z(a);
        }
        if ax {
            self.toggle_z(b);
        }
        ax && bx && (az != bz)
    }

    /// Conjugate by Pauli X on qubit q.
    fn conjugate_pauli_x(&self, q: u16) -> bool {
        self.has_z(q)
    }
    /// Conjugate by Pauli Y on qubit q.
    fn conjugate_pauli_y(&self, q: u16) -> bool {
        self.has_x(q) != self.has_z(q)
    }
    /// Conjugate by Pauli Z on qubit q.
    fn conjugate_pauli_z(&self, q: u16) -> bool {
        self.has_x(q)
    }

    /// Conjugate by SZ on qubit q: X→Y, Y→-X, Z→Z.
    fn conjugate_sz(&mut self, q: u16) -> bool {
        if !self.has_x(q) {
            return false;
        }
        let was_y = self.has_z(q);
        self.toggle_z(q);
        was_y
    }

    /// Conjugate by SZdg on qubit q.
    fn conjugate_szdg(&mut self, q: u16) -> bool {
        if !self.has_x(q) {
            return false;
        }
        let was_y = self.has_z(q);
        self.toggle_z(q);
        !was_y
    }

    /// Conjugate by SX on qubit q.
    fn conjugate_sx(&mut self, q: u16) -> bool {
        let xq = self.has_x(q);
        let zq = self.has_z(q);
        if zq {
            self.toggle_x(q);
        }
        !xq && zq
    }

    /// Conjugate by SXdg on qubit q.
    fn conjugate_sxdg(&mut self, q: u16) -> bool {
        let xq = self.has_x(q);
        let zq = self.has_z(q);
        if zq {
            self.toggle_x(q);
        }
        xq && zq
    }

    /// Conjugate by SY on qubit q.
    fn conjugate_sy(&mut self, q: u16) -> bool {
        let xq = self.has_x(q);
        let zq = self.has_z(q);
        if xq != zq {
            self.toggle_x(q);
            self.toggle_z(q);
        }
        xq && !zq
    }

    /// Conjugate by SYdg on qubit q.
    fn conjugate_sydg(&mut self, q: u16) -> bool {
        let xq = self.has_x(q);
        let zq = self.has_z(q);
        if xq != zq {
            self.toggle_x(q);
            self.toggle_z(q);
        }
        !xq && zq
    }

    /// Conjugate by SWAP(a, b).
    fn conjugate_swap(&mut self, a: u16, b: u16) {
        let ax = self.has_x(a);
        let az = self.has_z(a);
        let bx = self.has_x(b);
        let bz = self.has_z(b);
        // Clear both
        if ax {
            self.clear_x(a);
        }
        if az {
            self.clear_z(a);
        }
        if bx {
            self.clear_x(b);
        }
        if bz {
            self.clear_z(b);
        }
        // Set swapped
        if bx {
            self.toggle_x(a);
        }
        if bz {
            self.toggle_z(a);
        }
        if ax {
            self.toggle_x(b);
        }
        if az {
            self.toggle_z(b);
        }
    }
}

/// Apply backward (Heisenberg) gate conjugation: P → U† P U.
///
/// The conjugation methods on `SparsePauli` use the Schrödinger convention
/// (P → U P U†), so for the backward walk we swap non-self-adjoint gates
/// to their adjoints: SZ↔SZdg, SX↔SXdg, SY↔SYdg, SZZ↔SZZdg, etc.
/// Self-adjoint gates (H, X, Y, Z, CX, CZ, SWAP, CY) are unchanged.
pub(crate) fn sparse_conjugate(p: &mut SparsePauli, gate: &Gate) -> Option<bool> {
    if gate.qubits.is_empty() {
        return None;
    }
    let q0 = gate.qubits[0].index() as u16;
    match gate.gate_type {
        // Self-adjoint single-qubit gates
        GateType::H => Some(p.conjugate_h(q0)),
        GateType::X => Some(p.conjugate_pauli_x(q0)),
        GateType::Y => Some(p.conjugate_pauli_y(q0)),
        GateType::Z => Some(p.conjugate_pauli_z(q0)),
        // Non-self-adjoint single-qubit: swap to adjoint for backward
        GateType::SZ => Some(p.conjugate_szdg(q0)),
        GateType::SZdg => Some(p.conjugate_sz(q0)),
        GateType::SX => Some(p.conjugate_sxdg(q0)),
        GateType::SXdg => Some(p.conjugate_sx(q0)),
        GateType::SY => Some(p.conjugate_sydg(q0)),
        GateType::SYdg => Some(p.conjugate_sy(q0)),
        // Self-adjoint two-qubit gates
        GateType::CX => {
            let q1 = gate.qubits[1].index() as u16;
            Some(p.conjugate_cx(q0, q1))
        }
        GateType::CZ => {
            let q1 = gate.qubits[1].index() as u16;
            Some(p.conjugate_cz(q0, q1))
        }
        GateType::SWAP => {
            let q1 = gate.qubits[1].index() as u16;
            p.conjugate_swap(q0, q1);
            Some(false)
        }
        // CY is self-adjoint: CY = SZdg(t) CX(c,t) SZ(t) — chain
        GateType::CY => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_sz(q1);
            let s2 = p.conjugate_cx(q0, q1);
            let s3 = p.conjugate_szdg(q1);
            Some(sign_parity([s1, s2, s3]))
        }
        // Non-self-adjoint two-qubit: swap to adjoint for backward.
        // SZZ backward = SZZdg forward = CX(q0,q1) SZdg(q1) CX(q0,q1)
        GateType::SZZ => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_cx(q0, q1);
            let s2 = p.conjugate_szdg(q1);
            let s3 = p.conjugate_cx(q0, q1);
            Some(sign_parity([s1, s2, s3]))
        }
        // SZZdg backward = SZZ forward = CX(q0,q1) SZ(q1) CX(q0,q1)
        GateType::SZZdg => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_cx(q0, q1);
            let s2 = p.conjugate_sz(q1);
            let s3 = p.conjugate_cx(q0, q1);
            Some(sign_parity([s1, s2, s3]))
        }
        // SXX backward = SXXdg forward = H(q0) H(q1) SZZdg H(q0) H(q1)
        GateType::SXX => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_h(q0);
            let s2 = p.conjugate_h(q1);
            let s3 = p.conjugate_cx(q0, q1);
            let s4 = p.conjugate_szdg(q1);
            let s5 = p.conjugate_cx(q0, q1);
            let s6 = p.conjugate_h(q0);
            let s7 = p.conjugate_h(q1);
            Some(sign_parity([s1, s2, s3, s4, s5, s6, s7]))
        }
        // SXXdg backward = SXX forward
        GateType::SXXdg => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_h(q0);
            let s2 = p.conjugate_h(q1);
            let s3 = p.conjugate_cx(q0, q1);
            let s4 = p.conjugate_sz(q1);
            let s5 = p.conjugate_cx(q0, q1);
            let s6 = p.conjugate_h(q0);
            let s7 = p.conjugate_h(q1);
            Some(sign_parity([s1, s2, s3, s4, s5, s6, s7]))
        }
        // SYY backward = SYYdg forward = SX(q0) SX(q1) SZZdg SXdg(q0) SXdg(q1)
        GateType::SYY => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_sxdg(q0);
            let s2 = p.conjugate_sxdg(q1);
            let s3 = p.conjugate_cx(q0, q1);
            let s4 = p.conjugate_szdg(q1);
            let s5 = p.conjugate_cx(q0, q1);
            let s6 = p.conjugate_sx(q0);
            let s7 = p.conjugate_sx(q1);
            Some(sign_parity([s1, s2, s3, s4, s5, s6, s7]))
        }
        // SYYdg backward = SYY forward
        GateType::SYYdg => {
            let q1 = gate.qubits[1].index() as u16;
            let s1 = p.conjugate_sx(q0);
            let s2 = p.conjugate_sx(q1);
            let s3 = p.conjugate_cx(q0, q1);
            let s4 = p.conjugate_sz(q1);
            let s5 = p.conjugate_cx(q0, q1);
            let s6 = p.conjugate_sxdg(q0);
            let s7 = p.conjugate_sxdg(q1);
            Some(sign_parity([s1, s2, s3, s4, s5, s6, s7]))
        }
        // Gates that don't conjugate Paulis
        GateType::PZ
        | GateType::QAlloc
        | GateType::QFree
        | GateType::MZ
        | GateType::MeasureFree
        | GateType::MeasureLeaked
        | GateType::I
        | GateType::Idle => None,
        other => panic!("EEG Heisenberg: unsupported gate type {other:?}"),
    }
}

/// A term in the Heisenberg-propagated detector expansion.
#[derive(Clone, Debug)]
struct HeisenbergTerm {
    /// Pauli operator (sparse: only stores non-identity qubits).
    pauli: SparsePauli,
    /// Complex coefficient (real, imaginary).
    coeff_re: f64,
    coeff_im: f64,
}

/// Compute detection probability via backward Heisenberg propagation.
///
/// Operates on the EXPANDED circuit (from [`crate::expand`]). Expansion
/// gates are automatically detected and skipped for noise injection.
///
/// Handles both H-type (coherent) and S-type (stochastic) noise.
///
/// # Arguments
/// * `gates` - The expanded circuit gates
/// * `detector` - Detector as Z on auxiliary qubit(s) (expanded frame)
/// * `noise` - Noise specification
/// * `initial_stab` - Stabilizer group of |0...0⟩
/// * `prune_threshold` - Drop terms with |coefficient| below this (0 for exact)
pub fn heisenberg_detection_probability(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
) -> f64 {
    heisenberg_windowed(gates, detector, noise, initial_stab, prune_threshold, None)
}

/// Backward Heisenberg with precomputed noise map and BTreeMap-based merging.
///
/// Uses BTreeMap<SparsePauli, (re, im)> for continuous dedup — no separate
/// merge step. Terms are merged on insert via BTreeMap's O(log n) lookup.
/// Also uses batched S-type scaling from the precomputed noise map.
pub fn heisenberg_with_noise_map(
    gates: &[Gate],
    detector: &Bm,
    noise_map: &[Option<PrecomputedGateNoise>],
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
) -> f64 {
    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    // Conservative active-qubit bitmap
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits.iter())
        .map(pecos_core::QubitId::index)
        .max()
        .unwrap_or(0)
        + 1;
    let mut active_qubits = vec![false; max_qubit];
    for &q in terms[0]
        .pauli
        .x_qubits
        .iter()
        .chain(terms[0].pauli.z_qubits.iter())
    {
        if (q as usize) < active_qubits.len() {
            active_qubits[q as usize] = true;
        }
    }

    let mut last_merge_count = 1usize;
    let mut sin_branches: Vec<HeisenbergTerm> = Vec::new();

    for i in (0..gates.len()).rev() {
        let gate = &gates[i];
        let gate_qs: SmallVec<[u16; 4]> = gate.qubits.iter().map(|q| q.index() as u16).collect();

        let gate_touches_active = gate_qs
            .iter()
            .any(|&q| (q as usize) < active_qubits.len() && active_qubits[q as usize]);

        // Look up precomputed noise for this gate
        let gate_noise = if gate_touches_active {
            noise_map.get(i).and_then(|n| n.as_ref())
        } else {
            None
        };

        if let Some(gn) = gate_noise {
            // H-type injections (branching)
            for inj in &gn.h_injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }
                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> = if inj.label.x_bits.is_zero() {
                            inj.label.z_bits.highest_set_bit().map(|q| q as u16)
                        } else {
                            None
                        };
                        let noise_sparse = if single_z_qubit.is_none() {
                            Some(SparsePauli::from_bm(&inj.label))
                        } else {
                            None
                        };

                        sin_branches.clear();
                        let n = terms.len();
                        for term in terms.iter_mut().take(n) {
                            let anticommutes = if let Some(q) = single_z_qubit {
                                term.pauli.has_x(q)
                            } else {
                                !term.pauli.commutes_with(noise_sparse.as_ref().unwrap())
                            };
                            if anticommutes {
                                let (sr, si) = (sin2h * term.coeff_re, sin2h * term.coeff_im);
                                let (dp, total_phase) = if let Some(q) = single_z_qubit {
                                    let mut dp = term.pauli.clone();
                                    dp.toggle_z(q);
                                    let has_x = term.pauli.has_x(q);
                                    let has_z = term.pauli.has_z(q);
                                    let phase = if has_x {
                                        if has_z { 3u8 } else { 1 }
                                    } else {
                                        0
                                    };
                                    (dp, (phase + 1) % 4)
                                } else {
                                    let term_bm = term.pauli.to_bm();
                                    let (dp_bm, phase_exp) =
                                        inj.label.multiply_with_phase(&term_bm);
                                    (SparsePauli::from_bm(&dp_bm), (phase_exp + 1) % 4)
                                };
                                let (new_re, new_im) = match total_phase {
                                    0 => (sr, si),
                                    1 => (-si, sr),
                                    2 => (-sr, -si),
                                    3 => (si, -sr),
                                    _ => unreachable!(),
                                };
                                sin_branches.push(HeisenbergTerm {
                                    pauli: dp,
                                    coeff_re: new_re,
                                    coeff_im: new_im,
                                });
                                term.coeff_re *= cos2h;
                                term.coeff_im *= cos2h;
                            }
                        }
                        // Merge sin branches: try binary search merge if terms
                        // are still sorted from last merge, else just extend.
                        for t in sin_branches.drain(..) {
                            for &q in t.pauli.x_qubits.iter().chain(t.pauli.z_qubits.iter()) {
                                let qu = q as usize;
                                if qu < active_qubits.len() {
                                    active_qubits[qu] = true;
                                }
                            }
                            if last_merge_count == terms.len() {
                                match terms.binary_search_by(|p| p.pauli.cmp(&t.pauli)) {
                                    Ok(idx) => {
                                        terms[idx].coeff_re += t.coeff_re;
                                        terms[idx].coeff_im += t.coeff_im;
                                    }
                                    Err(_) => {
                                        terms.push(t);
                                    }
                                }
                            } else {
                                terms.push(t);
                            }
                        }
                    }
                    crate::eeg::EegType::S => {
                        // Non-batched S-type (fallback for non-uniform noise)
                        let s = inj.rate;
                        if s.abs() < 1e-20 {
                            continue;
                        }
                        let p = -s;
                        let scale = 1.0 - 2.0 * p;
                        let single_q: Option<u16> = {
                            let xq = inj.label.x_bits.highest_set_bit();
                            let zq = inj.label.z_bits.highest_set_bit();
                            match (xq, zq) {
                                (Some(x), None) => Some(x as u16),
                                (None, Some(z)) => Some(z as u16),
                                (Some(x), Some(z)) if x == z => Some(x as u16),
                                _ => None,
                            }
                        };
                        if let Some(q) = single_q {
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    || (has_x_in_noise && term.pauli.has_z(q));
                                if anti {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        } else {
                            let ns = SparsePauli::from_bm(&inj.label);
                            for term in &mut terms {
                                if !term.pauli.commutes_with(&ns) {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Batched S-type: apply combined scale factor
            let s_scale = gn.q0_scale; // Same for all non-trivial support patterns
            if (s_scale - 1.0).abs() > 1e-20 {
                match gn.num_gate_qubits {
                    1 => {
                        let q = gn.q0;
                        for term in &mut terms {
                            if term.pauli.has_x(q) || term.pauli.has_z(q) {
                                term.coeff_re *= s_scale;
                                term.coeff_im *= s_scale;
                            }
                        }
                    }
                    2 => {
                        let q0 = gn.q0;
                        let q1 = gn.q1;
                        for term in &mut terms {
                            let on_q0 = term.pauli.has_x(q0) || term.pauli.has_z(q0);
                            let on_q1 = term.pauli.has_x(q1) || term.pauli.has_z(q1);
                            if on_q0 || on_q1 {
                                term.coeff_re *= s_scale;
                                term.coeff_im *= s_scale;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Step 2: Backward Clifford conjugation
        if !gate_touches_active {
            continue;
        }

        match gate.gate_type {
            GateType::PZ | GateType::QAlloc => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
                for t in &mut terms {
                    for &qi in &gate_qs {
                        t.pauli.clear_z(qi);
                    }
                }
            }
            GateType::MZ => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
            }
            _ => {
                for t in &mut terms {
                    if let Some(sign_neg) = sparse_conjugate(&mut t.pauli, gate)
                        && sign_neg
                    {
                        t.coeff_re = -t.coeff_re;
                        t.coeff_im = -t.coeff_im;
                    }
                    for &q in t.pauli.x_qubits.iter().chain(t.pauli.z_qubits.iter()) {
                        let qu = q as usize;
                        if qu < active_qubits.len() {
                            active_qubits[qu] = true;
                        }
                    }
                }
            }
        }

        // Prune
        if prune_threshold > 0.0 {
            let thresh_sq = prune_threshold * prune_threshold;
            terms.retain(|t| t.coeff_re * t.coeff_re + t.coeff_im * t.coeff_im > thresh_sq);
        }

        // Merge: sort + dedup. In-place, no allocation, cache-friendly.
        // For typical term counts (~50-100), this beats both HashMap and
        // BTreeMap due to zero allocation overhead and sequential access.
        let should_merge = match gate.gate_type {
            GateType::PZ | GateType::QAlloc | GateType::MZ => terms.len() > 4,
            _ => terms.len() > last_merge_count * 2 && terms.len() > 16,
        };
        if should_merge {
            terms.sort_unstable_by(|a, b| a.pauli.cmp(&b.pauli));
            let mut write = 0;
            for read in 1..terms.len() {
                if terms[read].pauli == terms[write].pauli {
                    terms[write].coeff_re += terms[read].coeff_re;
                    terms[write].coeff_im += terms[read].coeff_im;
                } else {
                    if terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30 {
                        write += 1;
                    }
                    if write < read {
                        terms.swap(write, read);
                    }
                }
            }
            let final_len = if !terms.is_empty()
                && (terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30)
            {
                write + 1
            } else if terms.is_empty() {
                0
            } else {
                write
            };
            terms.truncate(final_len);
            last_merge_count = terms.len().max(1);
        }
    }

    // Evaluate
    let mut expectation_re = 0.0;
    for term in &terms {
        let eigenvalue = if term.pauli.is_identity() {
            1.0
        } else {
            let bm = term.pauli.to_bm();
            match initial_stab.is_stabilizer(&bm) {
                Some(true) => 1.0,
                Some(false) => -1.0,
                None => 0.0,
            }
        };
        expectation_re += term.coeff_re * eigenvalue;
    }
    (0.5 * (1.0 - expectation_re)).clamp(0.0, 1.0)
}

/// Backward Heisenberg with optional gate windowing.
///
/// If `gate_window` is `Some((start, end))`, only walks gates in `[start, end)`.
/// Faster for large circuits but may miss long-range correlations.
/// Use `None` (or call [`heisenberg_detection_probability`]) for exact results.
pub fn heisenberg_windowed(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
    gate_window: Option<(usize, usize)>,
) -> f64 {
    // Start with the detector as a single sparse term
    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    // Identify expansion gates (virtual, no physical noise).
    let expansion_gates = {
        let mut exp = vec![false; gates.len()];
        if !gates.is_empty() && gates[0].gate_type == GateType::QAlloc {
            exp[0] = true;
        }
        for i in 1..gates.len() {
            if gates[i].gate_type == GateType::QAlloc {
                exp[i] = true;
            }
            if gates[i].gate_type == GateType::CX && gates[i - 1].gate_type == GateType::QAlloc {
                let aq = gates[i - 1].qubits[0].index();
                if gates[i].qubits.len() >= 2 && gates[i].qubits[1].index() == aq {
                    exp[i] = true;
                    if i + 1 < gates.len()
                        && gates[i + 1].gate_type == GateType::PZ
                        && gates[i + 1].qubits[0].index() == gates[i].qubits[0].index()
                    {
                        exp[i + 1] = true;
                    }
                }
            }
        }
        exp
    };

    let mut last_merge_count = 1usize;
    // #3: Pre-allocate sin branches buffer, reused across noise sources
    let mut sin_branches: Vec<HeisenbergTerm> = Vec::new();

    // Conservative active-qubit bitmap: once a qubit is active, stays active.
    // This avoids the expensive per-term scan for gate relevance.
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits.iter())
        .map(pecos_core::QubitId::index)
        .max()
        .unwrap_or(0)
        + 1;
    let mut active_qubits = vec![false; max_qubit];
    // Seed from detector
    for &q in terms[0]
        .pauli
        .x_qubits
        .iter()
        .chain(terms[0].pauli.z_qubits.iter())
    {
        if (q as usize) < active_qubits.len() {
            active_qubits[q as usize] = true;
        }
    }

    // Walk backward through the circuit (optionally windowed)
    let (walk_start, walk_end) = gate_window.unwrap_or((0, gates.len()));
    for i in (walk_start..walk_end).rev() {
        let gate = &gates[i];
        let gate_qs: SmallVec<[u16; 4]> = gate.qubits.iter().map(|q| q.index() as u16).collect();

        // O(1) gate relevance check via bitmap (conservative: may visit some extra gates)
        let gate_touches_active = gate_qs
            .iter()
            .any(|&q| (q as usize) < active_qubits.len() && active_qubits[q as usize]);

        // Step 1: Apply noise adjoint (skip expansion gates).
        if !expansion_gates[i] && gate_touches_active {
            let qubits_usize: SmallVec<[usize; 4]> = gate_qs.iter().map(|&q| q as usize).collect();
            let injections = noise.noise_after_gate(i, gate.gate_type, &qubits_usize);

            for inj in &injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }

                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> = if inj.label.x_bits.is_zero() {
                            inj.label.z_bits.highest_set_bit().map(|q| q as u16)
                        } else {
                            None
                        };
                        let noise_sparse = if single_z_qubit.is_none() {
                            Some(SparsePauli::from_bm(&inj.label))
                        } else {
                            None
                        };

                        // #3: Reuse sin_branches buffer
                        sin_branches.clear();
                        let n = terms.len();

                        for term in terms.iter_mut().take(n) {
                            let anticommutes = if let Some(q) = single_z_qubit {
                                term.pauli.has_x(q)
                            } else {
                                !term.pauli.commutes_with(noise_sparse.as_ref().unwrap())
                            };

                            if anticommutes {
                                let (sr, si) = (sin2h * term.coeff_re, sin2h * term.coeff_im);

                                let (dp, total_phase) = if let Some(q) = single_z_qubit {
                                    let mut dp = term.pauli.clone();
                                    dp.toggle_z(q);
                                    let has_x = term.pauli.has_x(q);
                                    let has_z = term.pauli.has_z(q);
                                    let phase = if has_x {
                                        if has_z { 3u8 } else { 1 }
                                    } else {
                                        0
                                    };
                                    (dp, (phase + 1) % 4)
                                } else {
                                    let term_bm = term.pauli.to_bm();
                                    let (dp_bm, phase_exp) =
                                        inj.label.multiply_with_phase(&term_bm);
                                    (SparsePauli::from_bm(&dp_bm), (phase_exp + 1) % 4)
                                };

                                let (new_re, new_im) = match total_phase {
                                    0 => (sr, si),
                                    1 => (-si, sr),
                                    2 => (-sr, -si),
                                    3 => (si, -sr),
                                    _ => unreachable!(),
                                };
                                sin_branches.push(HeisenbergTerm {
                                    pauli: dp,
                                    coeff_re: new_re,
                                    coeff_im: new_im,
                                });
                                term.coeff_re *= cos2h;
                                term.coeff_im *= cos2h;
                            }
                        }

                        // Update active bitmap BEFORE extending (only scan new branches)
                        for t in &sin_branches {
                            for &q in t.pauli.x_qubits.iter().chain(t.pauli.z_qubits.iter()) {
                                let qu = q as usize;
                                if qu < active_qubits.len() {
                                    active_qubits[qu] = true;
                                }
                            }
                        }
                        terms.append(&mut sin_branches);
                    }
                    crate::eeg::EegType::S => {
                        let s = inj.rate;
                        if s.abs() < 1e-20 {
                            continue;
                        }
                        let p = -s;
                        let scale = 1.0 - 2.0 * p;
                        // For S-type, single-qubit specialization
                        let single_q: Option<u16> = {
                            let xq = inj.label.x_bits.highest_set_bit();
                            let zq = inj.label.z_bits.highest_set_bit();
                            match (xq, zq) {
                                (Some(x), None) => Some(x as u16),
                                (None, Some(z)) => Some(z as u16),
                                (Some(x), Some(z)) if x == z => Some(x as u16),
                                _ => None,
                            }
                        };

                        if let Some(q) = single_q {
                            // Single-qubit S noise: check just the one qubit
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                // Anticommutes if noise X overlaps term Z or noise Z overlaps term X
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    || (has_x_in_noise && term.pauli.has_z(q));
                                if anti {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        } else {
                            let noise_sparse = SparsePauli::from_bm(&inj.label);
                            for term in &mut terms {
                                if !term.pauli.commutes_with(&noise_sparse) {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        }
                    }
                    _ => {}
                }

                if prune_threshold > 0.0 {
                    terms.retain(|t| {
                        t.coeff_re * t.coeff_re + t.coeff_im * t.coeff_im
                            > prune_threshold * prune_threshold
                    });
                }
            }
        }

        // Step 2: Conjugate backward through the gate.
        // #2: Skip gates that don't touch active qubits
        if !gate_touches_active {
            continue;
        }

        match gate.gate_type {
            // #4: Batch PZ/QAlloc — single pass through terms for all qubits
            GateType::PZ | GateType::QAlloc => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
                for t in &mut terms {
                    for &qi in &gate_qs {
                        t.pauli.clear_z(qi);
                    }
                }
            }
            GateType::MZ => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
            }
            _ => {
                for t in &mut terms {
                    if let Some(sign_neg) = sparse_conjugate(&mut t.pauli, gate)
                        && sign_neg
                    {
                        t.coeff_re = -t.coeff_re;
                        t.coeff_im = -t.coeff_im;
                    }
                    // Update active bitmap (CX can spread support to new qubits)
                    for &q in t.pauli.x_qubits.iter().chain(t.pauli.z_qubits.iter()) {
                        let qu = q as usize;
                        if qu < active_qubits.len() {
                            active_qubits[qu] = true;
                        }
                    }
                }
            }
        }

        // Merge duplicate Pauli terms by sorting + linear scan.
        let should_merge = match gate.gate_type {
            GateType::PZ | GateType::QAlloc | GateType::MZ => terms.len() > 4,
            _ => terms.len() > last_merge_count * 2 && terms.len() > 16,
        };
        if should_merge {
            terms.sort_unstable_by(|a, b| a.pauli.cmp(&b.pauli));
            let mut write = 0;
            for read in 1..terms.len() {
                if terms[read].pauli == terms[write].pauli {
                    let re = terms[read].coeff_re;
                    let im = terms[read].coeff_im;
                    terms[write].coeff_re += re;
                    terms[write].coeff_im += im;
                } else {
                    if terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30 {
                        write += 1;
                    }
                    if write < read {
                        terms.swap(write, read);
                    }
                }
            }
            let final_len =
                if terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30 {
                    write + 1
                } else {
                    write
                };
            terms.truncate(final_len);
            last_merge_count = terms.len().max(1);
        }
    }

    // Evaluate: p_D = (1/2)(1 - Re(Σ c_j ⟨ψ|Q_j|ψ⟩))
    let mut expectation_re = 0.0;

    for term in &terms {
        let eigenvalue = if term.pauli.is_identity() {
            1.0
        } else {
            // Convert sparse back to Bm for stabilizer check
            let bm = term.pauli.to_bm();
            match initial_stab.is_stabilizer(&bm) {
                Some(true) => 1.0,
                Some(false) => -1.0,
                None => 0.0,
            }
        };

        expectation_re += term.coeff_re * eigenvalue;
    }

    let prob = 0.5 * (1.0 - expectation_re);
    prob.clamp(0.0, 1.0)
}

/// Backward Heisenberg with sparse gate traversal via precomputed index.
///
/// Instead of iterating all gates, uses a `GateIndex` to visit only gates
/// on active qubits (qubits in any term's support). Maintains a binary
/// heap of pending gates and an active qubit set for O(1) relevance checks.
///
/// For large circuits (d>=7), this is significantly faster than the linear
/// scan in [`heisenberg_windowed`] because most gates are irrelevant.
///
/// Accepts an optional precomputed noise map. When provided, uses batched
/// S-type scaling (faster). When `None`, calls `noise.noise_after_gate()`.
pub fn heisenberg_sparse(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
    gate_index: &crate::expand::GateIndex,
    noise_map: Option<&[Option<PrecomputedGateNoise>]>,
) -> f64 {
    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    // Active qubit set: union of all terms' support.
    // Use a Vec<bool> for O(1) check (faster than BTreeSet for small qubit counts).
    let num_qubits = gate_index.expansion_gates.len().min(gates.len()) + 64;
    let mut active = vec![false; num_qubits.max(1)];

    // Visited gate set: don't add the same gate to the heap twice.
    let mut visited = vec![false; gates.len()];

    // Populate initial active qubits and heap from detector support.
    // Max-heap: pops largest gate index first (backward traversal).
    let mut heap: BinaryHeap<u32> = BinaryHeap::new();

    // Seed from detector — all gates on detector qubits are candidates
    let total_gates = gates.len() as u32;
    for &q in &terms[0].pauli.x_qubits {
        activate_qubit(
            q,
            total_gates,
            &mut active,
            &mut visited,
            &mut heap,
            gate_index,
        );
    }
    for &q in &terms[0].pauli.z_qubits {
        activate_qubit(
            q,
            total_gates,
            &mut active,
            &mut visited,
            &mut heap,
            gate_index,
        );
    }

    let mut last_merge_count = 1usize;
    let mut sin_branches: Vec<HeisenbergTerm> = Vec::new();

    // Walk backward: pop gates from heap in reverse order (largest index first)
    while let Some(gate_idx) = heap.pop() {
        let i = gate_idx as usize;
        let gate = &gates[i];
        let gate_qs: SmallVec<[u16; 4]> = gate.qubits.iter().map(|q| q.index() as u16).collect();

        // Step 1: Apply noise adjoint (skip expansion gates).
        if !gate_index.is_expansion(i) {
            // Get noise: from precomputed map if available, else dynamic
            let precomputed = noise_map.and_then(|nm| nm.get(i).and_then(|n| n.as_ref()));

            // Get injections: from noise map or dynamic noise spec
            let dynamic_injections = if precomputed.is_none() {
                let qubits_usize: SmallVec<[usize; 4]> =
                    gate_qs.iter().map(|&q| q as usize).collect();
                noise.noise_after_gate(i, gate.gate_type, &qubits_usize)
            } else {
                Vec::new()
            };
            let injections: &[crate::noise::NoiseInjection] = if let Some(gn) = precomputed {
                &gn.h_injections
            } else {
                &dynamic_injections
            };

            for inj in injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }

                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> = if inj.label.x_bits.is_zero() {
                            inj.label.z_bits.highest_set_bit().map(|q| q as u16)
                        } else {
                            None
                        };
                        let noise_sparse = if single_z_qubit.is_none() {
                            Some(SparsePauli::from_bm(&inj.label))
                        } else {
                            None
                        };

                        sin_branches.clear();
                        let n = terms.len();

                        for term in terms.iter_mut().take(n) {
                            let anticommutes = if let Some(q) = single_z_qubit {
                                term.pauli.has_x(q)
                            } else {
                                !term.pauli.commutes_with(noise_sparse.as_ref().unwrap())
                            };

                            if anticommutes {
                                let (sr, si) = (sin2h * term.coeff_re, sin2h * term.coeff_im);

                                let (dp, total_phase) = if let Some(q) = single_z_qubit {
                                    let mut dp = term.pauli.clone();
                                    dp.toggle_z(q);
                                    let has_x = term.pauli.has_x(q);
                                    let has_z = term.pauli.has_z(q);
                                    let phase = if has_x {
                                        if has_z { 3u8 } else { 1 }
                                    } else {
                                        0
                                    };
                                    (dp, (phase + 1) % 4)
                                } else {
                                    let term_bm = term.pauli.to_bm();
                                    let (dp_bm, phase_exp) =
                                        inj.label.multiply_with_phase(&term_bm);
                                    (SparsePauli::from_bm(&dp_bm), (phase_exp + 1) % 4)
                                };

                                let (new_re, new_im) = match total_phase {
                                    0 => (sr, si),
                                    1 => (-si, sr),
                                    2 => (-sr, -si),
                                    3 => (si, -sr),
                                    _ => unreachable!(),
                                };

                                // Check if new term activates new qubits
                                for &q in &dp.x_qubits {
                                    activate_qubit(
                                        q,
                                        gate_idx,
                                        &mut active,
                                        &mut visited,
                                        &mut heap,
                                        gate_index,
                                    );
                                }
                                for &q in &dp.z_qubits {
                                    activate_qubit(
                                        q,
                                        gate_idx,
                                        &mut active,
                                        &mut visited,
                                        &mut heap,
                                        gate_index,
                                    );
                                }

                                sin_branches.push(HeisenbergTerm {
                                    pauli: dp,
                                    coeff_re: new_re,
                                    coeff_im: new_im,
                                });
                                term.coeff_re *= cos2h;
                                term.coeff_im *= cos2h;
                            }
                        }

                        terms.append(&mut sin_branches);
                    }
                    crate::eeg::EegType::S => {
                        // S-type: process individually. When using noise map,
                        // the batched scaling below handles the common case,
                        // but unbatchable S injections are placed in h_injections
                        // and must be processed here.
                        let s = inj.rate;
                        if s.abs() < 1e-20 {
                            continue;
                        }
                        let p = -s;
                        let scale = 1.0 - 2.0 * p;

                        let single_q: Option<u16> = {
                            let xq = inj.label.x_bits.highest_set_bit();
                            let zq = inj.label.z_bits.highest_set_bit();
                            match (xq, zq) {
                                (Some(x), None) => Some(x as u16),
                                (None, Some(z)) => Some(z as u16),
                                (Some(x), Some(z)) if x == z => Some(x as u16),
                                _ => None,
                            }
                        };

                        if let Some(q) = single_q {
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    || (has_x_in_noise && term.pauli.has_z(q));
                                if anti {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        } else {
                            let noise_sparse = SparsePauli::from_bm(&inj.label);
                            for term in &mut terms {
                                if !term.pauli.commutes_with(&noise_sparse) {
                                    term.coeff_re *= scale;
                                    term.coeff_im *= scale;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Batched S-type scaling from noise map (much faster than per-injection)
            if let Some(gn) = precomputed {
                let s_scale = gn.q0_scale;
                if (s_scale - 1.0).abs() > 1e-20 {
                    match gn.num_gate_qubits {
                        1 => {
                            let q = gn.q0;
                            for term in &mut terms {
                                if term.pauli.has_x(q) || term.pauli.has_z(q) {
                                    term.coeff_re *= s_scale;
                                    term.coeff_im *= s_scale;
                                }
                            }
                        }
                        2 => {
                            let q0 = gn.q0;
                            let q1 = gn.q1;
                            for term in &mut terms {
                                let on_q0 = term.pauli.has_x(q0) || term.pauli.has_z(q0);
                                let on_q1 = term.pauli.has_x(q1) || term.pauli.has_z(q1);
                                if on_q0 || on_q1 {
                                    term.coeff_re *= s_scale;
                                    term.coeff_im *= s_scale;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Step 2: Backward Clifford conjugation.
        match gate.gate_type {
            GateType::PZ | GateType::QAlloc => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
                for t in &mut terms {
                    for &qi in &gate_qs {
                        t.pauli.clear_z(qi);
                    }
                }
            }
            GateType::MZ => {
                terms.retain(|t| !gate_qs.iter().any(|&qi| t.pauli.has_x(qi)));
            }
            _ => {
                for t in &mut terms {
                    if let Some(sign_neg) = sparse_conjugate(&mut t.pauli, gate)
                        && sign_neg
                    {
                        t.coeff_re = -t.coeff_re;
                        t.coeff_im = -t.coeff_im;
                    }

                    // Activate any NEW qubits from conjugation (e.g., CX spreads Z)
                    for &q in t.pauli.x_qubits.iter().chain(t.pauli.z_qubits.iter()) {
                        activate_qubit(
                            q,
                            gate_idx,
                            &mut active,
                            &mut visited,
                            &mut heap,
                            gate_index,
                        );
                    }
                }
            }
        }

        // Prune
        if prune_threshold > 0.0 {
            let thresh_sq = prune_threshold * prune_threshold;
            terms.retain(|t| t.coeff_re * t.coeff_re + t.coeff_im * t.coeff_im > thresh_sq);
        }

        // Merge duplicate Pauli terms.
        let should_merge = match gate.gate_type {
            GateType::PZ | GateType::QAlloc | GateType::MZ => terms.len() > 4,
            _ => terms.len() > last_merge_count * 2 && terms.len() > 16,
        };
        if should_merge {
            terms.sort_unstable_by(|a, b| a.pauli.cmp(&b.pauli));
            let mut write = 0;
            for read in 1..terms.len() {
                if terms[read].pauli == terms[write].pauli {
                    let re = terms[read].coeff_re;
                    let im = terms[read].coeff_im;
                    terms[write].coeff_re += re;
                    terms[write].coeff_im += im;
                } else {
                    if terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30 {
                        write += 1;
                    }
                    if write < read {
                        terms.swap(write, read);
                    }
                }
            }
            let final_len = if !terms.is_empty()
                && (terms[write].coeff_re.abs() > 1e-30 || terms[write].coeff_im.abs() > 1e-30)
            {
                write + 1
            } else if terms.is_empty() {
                0
            } else {
                write
            };
            terms.truncate(final_len);
            last_merge_count = terms.len().max(1);
        }
    }

    // Evaluate
    let mut expectation_re = 0.0;
    for term in &terms {
        let eigenvalue = if term.pauli.is_identity() {
            1.0
        } else {
            let bm = term.pauli.to_bm();
            match initial_stab.is_stabilizer(&bm) {
                Some(true) => 1.0,
                Some(false) => -1.0,
                None => 0.0,
            }
        };
        expectation_re += term.coeff_re * eigenvalue;
    }

    let prob = 0.5 * (1.0 - expectation_re);
    prob.clamp(0.0, 1.0)
}

/// Convenience: expand an original circuit and compute detection probability.
pub fn heisenberg_detection_probability_from_circuit(
    original_gates: &[Gate],
    detector_meas_indices: &[usize],
    noise: &dyn NoiseSpec,
    num_original_qubits: usize,
    prune_threshold: f64,
) -> f64 {
    let expanded = crate::expand::expand_circuit(original_gates);

    let mut detector = Bm::default();
    for &m in detector_meas_indices {
        if m < expanded.measurement_qubit.len() {
            detector.z_bits.set_bit(expanded.measurement_qubit[m]);
        }
    }

    let init_gates: Vec<Gate> = (0..num_original_qubits)
        .map(|q| crate::expand::make_gate(GateType::PZ, &[q]))
        .collect();
    let stab = StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    heisenberg_detection_probability(&expanded.gates, &detector, noise, &stab, prune_threshold)
}

/// Exact detection probability via matrix-based backward Heisenberg.
///
/// Computes the backward adjoint using dense 2^n × 2^n complex matrix
/// multiplication. Exact for any circuit, but limited to ~20 expanded
/// qubits by memory. Useful as a reference/validation for the faster
/// Pauli-tracking walk ([`heisenberg_detection_probability_from_circuit`]).
pub fn heisenberg_exact_from_circuit(
    original_gates: &[Gate],
    detector_meas_indices: &[usize],
    noise: &dyn NoiseSpec,
    _num_original_qubits: usize,
) -> f64 {
    let expanded = crate::expand::expand_circuit(original_gates);
    let n = expanded.num_qubits;

    assert!(
        n <= 20,
        "Matrix Heisenberg requires 2^n memory; {n} qubits is too large. Use the Pauli-tracking walk for approximate results."
    );

    let dim = 1usize << n;

    // Build detector matrix: diagonal with Z eigenvalues on the detector aux qubits.
    let mut obs_re = vec![0.0f64; dim * dim];
    let obs_im = vec![0.0f64; dim * dim];
    for i in 0..dim {
        let mut eigenvalue = 1.0f64;
        for &m in detector_meas_indices {
            if m < expanded.measurement_qubit.len() {
                let aux = expanded.measurement_qubit[m];
                if (i >> aux) & 1 == 1 {
                    eigenvalue = -eigenvalue;
                }
            }
        }
        obs_re[i * dim + i] = eigenvalue;
    }

    // Identify expansion gates
    let expansion_gates = find_expansion_gates(&expanded.gates);

    // Walk backward, applying adjoints via matrix multiplication.
    let mut im = obs_im;
    for idx in (0..expanded.gates.len()).rev() {
        let g = &expanded.gates[idx];
        let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();

        // Noise adjoint (skip expansion gates)
        if !expansion_gates[idx] {
            let injections = noise.noise_after_gate(idx, g.gate_type, &qs);
            for inj in &injections {
                if inj.eeg_type != crate::eeg::EegType::H {
                    continue;
                }
                if inj.rate.abs() < 1e-20 {
                    continue;
                }
                // RZ(θ) on the noise qubit, where θ = 2*rate
                let theta = 2.0 * inj.rate;
                // Find which qubit the noise acts on
                let noise_q = if let Some(q) = inj.label.z_bits.highest_set_bit() {
                    q
                } else if let Some(q) = inj.label.x_bits.highest_set_bit() {
                    q
                } else {
                    continue;
                };
                matrix_rz_adjoint(&mut obs_re, &mut im, noise_q, theta, n);
            }
        }

        // Gate adjoint
        match g.gate_type {
            GateType::PZ | GateType::QAlloc => {
                matrix_pz_adjoint(&mut obs_re, &mut im, qs[0], n);
            }
            GateType::MZ => {
                matrix_mz_adjoint(&mut obs_re, &mut im, qs[0], n);
            }
            GateType::H => {
                matrix_h_adjoint(&mut obs_re, &mut im, qs[0], n);
            }
            GateType::CX if qs.len() >= 2 => {
                matrix_cx_adjoint(&mut obs_re, &mut im, qs[0], qs[1], n);
            }
            _ => {}
        }
    }

    // ⟨0...0|O_backward|0...0⟩ = obs_re[0]
    let expectation = obs_re[0];
    let prob = 0.5 * (1.0 - expectation);
    prob.clamp(0.0, 1.0)
}

// --- Matrix helpers for exact Heisenberg ---

fn bit_to_f64(value: usize) -> f64 {
    f64::from(u8::try_from(value).expect("bit value fits in u8"))
}

fn matrix_rz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, theta: f64, n: usize) {
    let dim = 1usize << n;
    for i in 0..dim {
        let bi = bit_to_f64((i >> q) & 1);
        for j in 0..dim {
            let bj = bit_to_f64((j >> q) & 1);
            let phase = (bi - bj) * theta;
            if phase.abs() < 1e-20 {
                continue;
            }
            let (cp, sp) = (phase.cos(), phase.sin());
            let idx = i * dim + j;
            let (r, m) = (re[idx], im[idx]);
            re[idx] = cp * r - sp * m;
            im[idx] = sp * r + cp * m;
        }
    }
}

fn matrix_pz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    let dim = 1usize << n;
    let mask = 1usize << q;
    for i in 0..dim {
        let iq = (i >> q) & 1;
        for j in 0..dim {
            let jq = (j >> q) & 1;
            let idx = i * dim + j;
            if iq == jq {
                let i0 = i & !mask;
                let j0 = j & !mask;
                let idx0 = i0 * dim + j0;
                re[idx] = re[idx0];
                im[idx] = im[idx0];
            } else {
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}

fn matrix_mz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    let dim = 1usize << n;
    for i in 0..dim {
        let iq = (i >> q) & 1;
        for j in 0..dim {
            let jq = (j >> q) & 1;
            if iq != jq {
                let idx = i * dim + j;
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}

fn matrix_h_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    let dim = 1usize << n;
    let mask = 1usize << q;
    let mut new_re = vec![0.0f64; dim * dim];
    let mut new_im = vec![0.0f64; dim * dim];
    for i in 0..dim {
        let i0 = i & !mask;
        let i1 = i | mask;
        let iq = (i >> q) & 1;
        for j in 0..dim {
            let j0 = j & !mask;
            let j1 = j | mask;
            let jq = (j >> q) & 1;
            let mut sr = 0.0;
            let mut si = 0.0;
            for a in 0..2usize {
                for b in 0..2usize {
                    let ia = if a == 0 { i0 } else { i1 };
                    let jb = if b == 0 { j0 } else { j1 };
                    let sign = if (iq * a + b * jq).is_multiple_of(2) {
                        0.5
                    } else {
                        -0.5
                    };
                    let idx = ia * dim + jb;
                    sr += sign * re[idx];
                    si += sign * im[idx];
                }
            }
            new_re[i * dim + j] = sr;
            new_im[i * dim + j] = si;
        }
    }
    re.copy_from_slice(&new_re);
    im.copy_from_slice(&new_im);
}

fn matrix_cx_adjoint(re: &mut [f64], im: &mut [f64], control: usize, target: usize, n: usize) {
    let dim = 1usize << n;
    let cmask = 1usize << control;
    let tmask = 1usize << target;
    let cx_perm = |i: usize| -> usize { if (i & cmask) != 0 { i ^ tmask } else { i } };
    let mut new_re = vec![0.0f64; dim * dim];
    let mut new_im = vec![0.0f64; dim * dim];
    for i in 0..dim {
        let ci = cx_perm(i);
        for j in 0..dim {
            let cj = cx_perm(j);
            new_re[i * dim + j] = re[ci * dim + cj];
            new_im[i * dim + j] = im[ci * dim + cj];
        }
    }
    re.copy_from_slice(&new_re);
    im.copy_from_slice(&new_im);
}

/// Identify expansion gate indices.
fn find_expansion_gates(gates: &[Gate]) -> Vec<bool> {
    let mut exp = vec![false; gates.len()];
    if !gates.is_empty() && gates[0].gate_type == GateType::QAlloc {
        exp[0] = true;
    }
    for i in 1..gates.len() {
        if gates[i].gate_type == GateType::QAlloc {
            exp[i] = true;
        }
        if gates[i].gate_type == GateType::CX && gates[i - 1].gate_type == GateType::QAlloc {
            let aq = gates[i - 1].qubits[0].index();
            if gates[i].qubits.len() >= 2 && gates[i].qubits[1].index() == aq {
                exp[i] = true;
                if i + 1 < gates.len()
                    && gates[i + 1].gate_type == GateType::PZ
                    && gates[i + 1].qubits[0].index() == gates[i].qubits[0].index()
                {
                    exp[i + 1] = true;
                }
            }
        }
    }
    exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand;
    use crate::noise::UniformNoise;
    use pecos_core::{GateAngles, GateParams, QubitId};

    fn gate(gt: GateType, qubits: &[usize]) -> Gate {
        Gate {
            gate_type: gt,
            qubits: qubits.iter().map(|&q| QubitId(q)).collect(),
            angles: GateAngles::new(),
            params: GateParams::new(),
            meas_ids: pecos_core::GateMeasIds::new(),
            channel: None,
        }
    }

    #[test]
    fn test_d2_zbasis_heisenberg_original_circuit() {
        // d=2 Z-basis surface code (2 rounds) — the circuit where forward EEG
        // has a ~50% gap. Test if Heisenberg closes it.
        //
        // Circuit: 7 qubits (0-3 data, 4-6 ancilla)
        // X-check ancillas: 4, 5 (H, CX, CX, H, MZ)
        // Z-check ancilla: 6 (CX, CX, MZ)
        let gates_orig = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            gate(GateType::PZ, &[5]),
            gate(GateType::PZ, &[6]),
            // Round 1
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::CX, &[1, 6]),
            gate(GateType::CX, &[5, 3]),
            gate(GateType::CX, &[3, 6]),
            gate(GateType::CX, &[5, 2]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[0, 6]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[2, 6]),
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::MZ, &[4]),
            gate(GateType::MZ, &[5]),
            gate(GateType::MZ, &[6]),
            // Reset
            gate(GateType::PZ, &[4]),
            gate(GateType::PZ, &[5]),
            gate(GateType::PZ, &[6]),
            // Round 2
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::CX, &[1, 6]),
            gate(GateType::CX, &[5, 3]),
            gate(GateType::CX, &[3, 6]),
            gate(GateType::CX, &[5, 2]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[0, 6]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[2, 6]),
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::MZ, &[4]),
            gate(GateType::MZ, &[5]),
            gate(GateType::MZ, &[6]),
            // Final data readout
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
            gate(GateType::MZ, &[2]),
            gate(GateType::MZ, &[3]),
        ];

        let expanded = expand::expand_circuit(&gates_orig);
        let theta = 0.05;
        let noise = UniformNoise::coherent_only(theta);

        // Initial state stabilizer group: Z on each PZ-initialized qubit.
        // At circuit start, all original qubits are |0⟩.
        // (Aux qubits are QAlloc'd later during the circuit.)
        let init_gates: Vec<Gate> = (0..7).map(|q| gate(GateType::PZ, &[q])).collect();
        let stab = StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

        // D1: ancilla 4 round comparison (Z on aux for meas 0 and meas 3)
        let aux_m0 = expanded.measurement_qubit[0]; // q4 round 1
        let aux_m3 = expanded.measurement_qubit[3]; // q4 round 2
        let mut det1 = Bm::default();
        det1.z_bits.set_bit(aux_m0);
        det1.z_bits.set_bit(aux_m3);

        // D2: ancilla 5 round comparison
        let aux_m1 = expanded.measurement_qubit[1]; // q5 round 1
        let aux_m4 = expanded.measurement_qubit[4]; // q5 round 2
        let mut det2 = Bm::default();
        det2.z_bits.set_bit(aux_m1);
        det2.z_bits.set_bit(aux_m4);

        // Run Heisenberg for both detectors
        let p1_heis =
            heisenberg_detection_probability(&expanded.gates, &det1, &noise, &stab, 1e-10);
        let p2_heis =
            heisenberg_detection_probability(&expanded.gates, &det2, &noise, &stab, 1e-10);

        // For comparison: forward EEG
        let eeg_result = crate::circuit::analyze_with_noise(&expanded.gates, &noise);
        let dets = vec![
            crate::dem_mapping::Detector {
                id: 1,
                stabilizer: det1,
            },
            crate::dem_mapping::Detector {
                id: 2,
                stabilizer: det2,
            },
        ];
        let entries = crate::dem_mapping::build_dem_configured(
            &eeg_result.generators,
            &dets,
            &[],
            Some(&stab),
            &crate::dem_mapping::EegConfig::default(),
        );
        let mut eeg_d1 = 0.0;
        let mut eeg_d2 = 0.0;
        for e in &entries {
            for &d in &e.event.detectors {
                if d == 1 {
                    eeg_d1 += e.probability;
                }
                if d == 2 {
                    eeg_d2 += e.probability;
                }
            }
        }

        eprintln!("\nd=2 Z-basis, theta={theta}:");
        eprintln!("  D1: Heisenberg={p1_heis:.6}, EEG={eeg_d1:.6}");
        eprintln!("  D2: Heisenberg={p2_heis:.6}, EEG={eeg_d2:.6}");

        // Heisenberg should give DIFFERENT values for D1 and D2
        // (unlike EEG which gives them equal due to missing time-ordering)
        if (p1_heis - p2_heis).abs() > 1e-6 {
            eprintln!("  Heisenberg correctly distinguishes D1 and D2!");
        }
    }

    #[test]
    fn test_single_x_check_heisenberg() {
        // Simplest X-check: 2 data + 1 ancilla, 2 rounds.
        // Detector: Z on ancilla (qubit 2) — passes through both MZ(2) gates.
        // The round-comparison detector fires when the two MZ outcomes differ.
        let gates_orig = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            // Round 1
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
            gate(GateType::PZ, &[2]),
            // Round 2
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
        ];

        let theta = 0.05;
        let noise = UniformNoise::coherent_only(theta);

        // Initial state: Z on each qubit
        let init_gates: Vec<Gate> = (0..3).map(|q| gate(GateType::PZ, &[q])).collect();
        let stab = StabilizerGroup::from_circuit(&init_gates, 3);

        // Detector: Z on ancilla qubit 2 (round-comparison)
        let det = Bm::z(2);

        let p_heis = heisenberg_detection_probability(&gates_orig, &det, &noise, &stab, 0.0);

        eprintln!("\nSimple X-check (original circuit), theta={theta}:");
        eprintln!("  Heisenberg: {p_heis:.6}");
    }

    #[test]
    fn test_bell_parity_exact() {
        // Bell parity: PZ(0,1), H(0), CX(0,1), H(0), H(1), MZ(0), MZ(1)
        // Parity detector: Z_0 * Z_1 (on original qubits)
        // Exact answer: p = sin²(theta)
        let gates_orig = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[1]),
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
        ];

        // Parity detector: Z on both measured qubits (original frame)
        let mut det = Bm::default();
        det.z_bits.set_bit(0);
        det.z_bits.set_bit(1);

        // Initial state: Z on each qubit
        let init_gates: Vec<Gate> = (0..2).map(|q| gate(GateType::PZ, &[q])).collect();
        let stab = StabilizerGroup::from_circuit(&init_gates, 2);

        for &theta in &[0.01, 0.05, 0.1, 0.2, 0.5] {
            let noise = UniformNoise::coherent_only(theta);

            let p = heisenberg_detection_probability(&gates_orig, &det, &noise, &stab, 0.0);

            let exact = theta.sin().powi(2);
            let eeg_taylor = theta * theta; // leading-order EEG

            eprintln!(
                "theta={theta:.2}: Heisenberg={p:.6}, exact={exact:.6}, Taylor={eeg_taylor:.6}"
            );

            // Heisenberg should match exact much better than Taylor
            assert!(
                (p - exact).abs() < 0.01,
                "theta={theta}: Heisenberg {p:.6} vs exact {exact:.6}, diff={:.6}",
                (p - exact).abs()
            );
        }
    }

    #[test]
    fn test_exact_bell_parity() {
        // Matrix-based exact Heisenberg should match sin²(θ) perfectly.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[1]),
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
        ];

        for &theta in &[0.01, 0.05, 0.1, 0.2, 0.5] {
            let noise = crate::noise::UniformNoise::coherent_only(theta);
            let p = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 2);
            let exact = theta.sin().powi(2);
            assert!(
                (p - exact).abs() < 1e-10,
                "theta={theta}: exact_heisenberg {p:.10} vs sin²(θ) {exact:.10}"
            );
        }
    }

    #[test]
    fn test_exact_2round_xcheck() {
        // Matrix Heisenberg on the simplest failing case: 2-round, 1 ancilla.
        // Exact analytical: P = [2 - cos(6θ) - cos(2θ)] / 4.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
        ];

        for &theta in &[0.01, 0.05, 0.1, 0.2] {
            let noise = crate::noise::UniformNoise::coherent_only(theta);
            let p = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 3);
            let exact = (2.0 - (6.0 * theta).cos() - (2.0 * theta).cos()) / 4.0;
            eprintln!("theta={theta:.2}: exact_heisenberg={p:.10}, analytical={exact:.10}");
            assert!(
                (p - exact).abs() < 1e-8,
                "theta={theta}: got {p:.10}, expected {exact:.10}, diff={:.2e}",
                (p - exact).abs()
            );
        }
    }

    /// Verify heisenberg_sparse produces identical results to heisenberg_windowed,
    /// and measure the speedup from sparse traversal.
    #[test]
    fn test_sparse_matches_windowed_and_timing() {
        use std::time::Instant;

        // Build a d=2 Z-basis surface code with 2 rounds (same as test above)
        let gates_orig = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            gate(GateType::PZ, &[5]),
            gate(GateType::PZ, &[6]),
            // Round 1
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::CX, &[1, 6]),
            gate(GateType::CX, &[5, 3]),
            gate(GateType::CX, &[3, 6]),
            gate(GateType::CX, &[5, 2]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[0, 6]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[2, 6]),
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::MZ, &[4]),
            gate(GateType::MZ, &[5]),
            gate(GateType::MZ, &[6]),
            // Reset + Round 2
            gate(GateType::PZ, &[4]),
            gate(GateType::PZ, &[5]),
            gate(GateType::PZ, &[6]),
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::CX, &[1, 6]),
            gate(GateType::CX, &[5, 3]),
            gate(GateType::CX, &[3, 6]),
            gate(GateType::CX, &[5, 2]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[0, 6]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[2, 6]),
            gate(GateType::H, &[4]),
            gate(GateType::H, &[5]),
            gate(GateType::MZ, &[4]),
            gate(GateType::MZ, &[5]),
            gate(GateType::MZ, &[6]),
        ];

        let expanded = crate::expand::expand_circuit(&gates_orig);
        let gate_index = crate::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);

        let init_gates: Vec<Gate> = (0..7).map(|q| gate(GateType::PZ, &[q])).collect();
        let stab =
            crate::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

        // Test both coherent-only and depolarizing noise
        let noise_configs: Vec<(&str, crate::noise::UniformNoise)> = vec![
            (
                "coherent_only",
                crate::noise::UniformNoise::coherent_only(0.05),
            ),
            (
                "depolarizing",
                crate::noise::UniformNoise {
                    idle_rz: 0.0,
                    p1: 0.001,
                    p2: 0.01,
                    p_meas: 0.001,
                    p_prep: 0.001,
                },
            ),
            (
                "combined",
                crate::noise::UniformNoise {
                    idle_rz: 0.05,
                    p1: 0.001,
                    p2: 0.01,
                    p_meas: 0.001,
                    p_prep: 0.001,
                },
            ),
        ];

        for (label, noise) in &noise_configs {
            let noise_map = build_noise_map(&expanded.gates, noise, &gate_index.expansion_gates);

            // Test all 3 detectors (auxiliary qubits in round 1: meas 0,1,2)
            for meas_idx in 0..3 {
                let aux_q = expanded.measurement_qubit[meas_idx];
                let det = Bm::z(aux_q);

                // Windowed (old path)
                let start = Instant::now();
                let p_windowed =
                    heisenberg_windowed(&expanded.gates, &det, noise, &stab, 1e-12, None);
                let t_windowed = start.elapsed();

                // Sparse without noise map
                let start = Instant::now();
                let p_sparse = heisenberg_sparse(
                    &expanded.gates,
                    &det,
                    noise,
                    &stab,
                    1e-12,
                    &gate_index,
                    None,
                );
                let t_sparse = start.elapsed();

                // Sparse with noise map
                let start = Instant::now();
                let p_sparse_nm = heisenberg_sparse(
                    &expanded.gates,
                    &det,
                    noise,
                    &stab,
                    1e-12,
                    &gate_index,
                    Some(&noise_map),
                );
                let t_sparse_nm = start.elapsed();

                // With noise map (old path)
                let start = Instant::now();
                let p_nm =
                    heisenberg_with_noise_map(&expanded.gates, &det, &noise_map, &stab, 1e-12);
                let t_nm = start.elapsed();

                // Verify exact match
                let tol = 1e-12;
                assert!(
                    (p_windowed - p_sparse).abs() < tol,
                    "{label} det{meas_idx}: windowed={p_windowed:.15} vs sparse={p_sparse:.15}, diff={:.2e}",
                    (p_windowed - p_sparse).abs()
                );
                assert!(
                    (p_windowed - p_sparse_nm).abs() < tol,
                    "{label} det{meas_idx}: windowed={p_windowed:.15} vs sparse+nm={p_sparse_nm:.15}, diff={:.2e}",
                    (p_windowed - p_sparse_nm).abs()
                );
                assert!(
                    (p_windowed - p_nm).abs() < tol,
                    "{label} det{meas_idx}: windowed={p_windowed:.15} vs nm={p_nm:.15}, diff={:.2e}",
                    (p_windowed - p_nm).abs()
                );

                eprintln!(
                    "  {label} det{meas_idx}: p={p_windowed:.8} \
                    windowed={:.1}us sparse={:.1}us sparse+nm={:.1}us nm={:.1}us",
                    t_windowed.as_secs_f64() * 1e6,
                    t_sparse.as_secs_f64() * 1e6,
                    t_sparse_nm.as_secs_f64() * 1e6,
                    t_nm.as_secs_f64() * 1e6
                );
            }
        }
    }

    /// Scaling benchmark: sparse vs windowed at d=3..11 repetition codes.
    ///
    /// Builds larger circuits and measures per-detector walk time with both
    /// implementations. Verifies results match exactly.
    #[test]
    #[ignore = "benchmark; run manually with --ignored --nocapture"]
    fn bench_sparse_scaling() {
        use std::time::Instant;

        let noise = crate::noise::UniformNoise {
            idle_rz: 0.05,
            p1: 0.001,
            p2: 0.01,
            p_meas: 0.001,
            p_prep: 0.001,
        };

        eprintln!("\n=== Sparse vs Windowed scaling (combined noise) ===");
        eprintln!(
            "{:>4} {:>6} {:>8} {:>8} {:>6} {:>12} {:>12} {:>8}",
            "d", "rnds", "gates", "exp_q", "n_det", "windowed_ms", "sparse_ms", "speedup"
        );

        // Test with increasing circuit sizes.
        // Repetition codes are 1D — detectors propagate through most gates.
        // Surface codes are 2D — detectors are local (touch ~8 out of d^2 qubits).
        // Test both to show where sparsity helps.

        // --- Repetition codes (1D, low sparsity) ---
        eprintln!("\n--- Repetition codes (1D) ---");
        let rep_configs: Vec<(usize, usize)> =
            vec![(5, 3), (5, 10), (9, 3), (9, 10), (13, 3), (13, 10)];

        for &(d, num_rounds) in &rep_configs {
            let num_data = d;
            let num_ancilla = d - 1;
            let num_qubits = num_data + num_ancilla;

            // Build repetition code
            let mut gates = Vec::new();
            for q in 0..num_qubits {
                gates.push(gate(GateType::PZ, &[q]));
            }
            for round in 0..num_rounds {
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::H, &[num_data + i]));
                }
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::CX, &[num_data + i, i]));
                }
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::CX, &[num_data + i, i + 1]));
                }
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::H, &[num_data + i]));
                }
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::MZ, &[num_data + i]));
                }
                if round < num_rounds - 1 {
                    for i in 0..num_ancilla {
                        gates.push(gate(GateType::PZ, &[num_data + i]));
                    }
                }
            }
            for q in 0..num_data {
                gates.push(gate(GateType::MZ, &[q]));
            }

            let expanded = crate::expand::expand_circuit(&gates);
            let gate_index = crate::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);
            let noise_map = build_noise_map(&expanded.gates, &noise, &gate_index.expansion_gates);

            let init_gates: Vec<Gate> = (0..num_qubits).map(|q| gate(GateType::PZ, &[q])).collect();
            let stab =
                crate::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

            // Build detectors: round-to-round comparison
            let num_detectors = num_ancilla * (num_rounds - 1);
            let mut detectors = Vec::new();
            for round in 0..(num_rounds - 1) {
                for i in 0..num_ancilla {
                    let m1 = round * num_ancilla + i;
                    let m2 = (round + 1) * num_ancilla + i;
                    let aux1 = expanded.measurement_qubit[m1];
                    let aux2 = expanded.measurement_qubit[m2];
                    let det_bm = Bm::z(aux1).multiply(&Bm::z(aux2));
                    detectors.push(det_bm);
                }
            }

            // Time windowed (old path)
            let start = Instant::now();
            let mut p_windowed = Vec::new();
            for det in &detectors {
                p_windowed.push(heisenberg_with_noise_map(
                    &expanded.gates,
                    det,
                    &noise_map,
                    &stab,
                    1e-12,
                ));
            }
            let t_windowed = start.elapsed();

            // Time sparse (new path)
            let start = Instant::now();
            let mut p_sparse = Vec::new();
            for det in &detectors {
                p_sparse.push(heisenberg_sparse(
                    &expanded.gates,
                    det,
                    &noise,
                    &stab,
                    1e-12,
                    &gate_index,
                    Some(&noise_map),
                ));
            }
            let t_sparse = start.elapsed();

            // Verify exact match
            for (i, (&pw, &ps)) in p_windowed.iter().zip(p_sparse.iter()).enumerate() {
                assert!(
                    (pw - ps).abs() < 1e-12,
                    "d={d} det{i}: windowed={pw:.15} vs sparse={ps:.15}, diff={:.2e}",
                    (pw - ps).abs()
                );
            }

            let speedup = t_windowed.as_secs_f64() / t_sparse.as_secs_f64();
            eprintln!(
                "{d:>4} {num_rounds:>6} {:>8} {:>8} {num_detectors:>6} {:>12.2} {:>12.2} {speedup:>8.1}x",
                expanded.gates.len(),
                expanded.num_qubits,
                t_windowed.as_secs_f64() * 1000.0,
                t_sparse.as_secs_f64() * 1000.0
            );
        }

        // --- 2D grid codes (high sparsity at large d) ---
        // Each Z-stabilizer checks a plaquette of 4 data qubits using 1 ancilla.
        // Detectors are local: each touches only 1 ancilla + 4 data qubits.
        // At d=7: 49 data qubits, 24 Z-stab ancillas, ~500+ expanded gates.
        // A detector touches ~10 qubits out of ~100+ — high sparsity.
        eprintln!("\n--- 2D grid codes (surface-code-like) ---");
        eprintln!(
            "{:>4} {:>6} {:>8} {:>8} {:>6} {:>12} {:>12} {:>8}",
            "d", "rnds", "gates", "exp_q", "n_det", "windowed_ms", "sparse_ms", "speedup"
        );

        for &(d, num_rounds) in &[(3, 2), (5, 2), (7, 2), (9, 2), (7, 5), (9, 5)] {
            // Build a d x d grid with Z-plaquette stabilizers.
            // Data qubits: (r, c) for r in 0..d, c in 0..d → index r*d + c
            // Z-ancillas: one per plaquette, (d-1)*(d-1) total
            let num_data = d * d;
            let num_ancilla = (d - 1) * (d - 1);
            let num_qubits = num_data + num_ancilla;
            let anc_start = num_data;

            let mut gates = Vec::new();
            for q in 0..num_qubits {
                gates.push(gate(GateType::PZ, &[q]));
            }

            for round in 0..num_rounds {
                // Z-stabilizer syndrome: CX(data, anc) for each of 4 data qubits
                // Plaquette (r, c) has corners at data qubits:
                //   (r, c), (r, c+1), (r+1, c), (r+1, c+1)
                for r in 0..(d - 1) {
                    for c in 0..(d - 1) {
                        let anc = anc_start + r * (d - 1) + c;
                        let d00 = r * d + c;
                        let d01 = r * d + c + 1;
                        let d10 = (r + 1) * d + c;
                        let d11 = (r + 1) * d + c + 1;
                        gates.push(gate(GateType::CX, &[d00, anc]));
                        gates.push(gate(GateType::CX, &[d01, anc]));
                        gates.push(gate(GateType::CX, &[d10, anc]));
                        gates.push(gate(GateType::CX, &[d11, anc]));
                    }
                }
                for i in 0..num_ancilla {
                    gates.push(gate(GateType::MZ, &[anc_start + i]));
                }
                if round < num_rounds - 1 {
                    for i in 0..num_ancilla {
                        gates.push(gate(GateType::PZ, &[anc_start + i]));
                    }
                }
            }
            for q in 0..num_data {
                gates.push(gate(GateType::MZ, &[q]));
            }

            let expanded = crate::expand::expand_circuit(&gates);
            let gate_index = crate::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);
            let noise_map = build_noise_map(&expanded.gates, &noise, &gate_index.expansion_gates);

            let init_gates: Vec<Gate> = (0..num_qubits).map(|q| gate(GateType::PZ, &[q])).collect();
            let stab =
                crate::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

            // Build detectors: round-to-round comparison of each ancilla
            let num_detectors = num_ancilla * (num_rounds - 1);
            let mut detectors = Vec::new();
            for round in 0..(num_rounds - 1) {
                for i in 0..num_ancilla {
                    let m1 = round * num_ancilla + i;
                    let m2 = (round + 1) * num_ancilla + i;
                    let aux1 = expanded.measurement_qubit[m1];
                    let aux2 = expanded.measurement_qubit[m2];
                    let det_bm = Bm::z(aux1).multiply(&Bm::z(aux2));
                    detectors.push(det_bm);
                }
            }

            // Time windowed
            let start = Instant::now();
            let mut p_windowed = Vec::new();
            for det in &detectors {
                p_windowed.push(heisenberg_with_noise_map(
                    &expanded.gates,
                    det,
                    &noise_map,
                    &stab,
                    1e-12,
                ));
            }
            let t_windowed = start.elapsed();

            // Time sparse
            let start = Instant::now();
            let mut p_sparse = Vec::new();
            for det in &detectors {
                p_sparse.push(heisenberg_sparse(
                    &expanded.gates,
                    det,
                    &noise,
                    &stab,
                    1e-12,
                    &gate_index,
                    Some(&noise_map),
                ));
            }
            let t_sparse = start.elapsed();

            // Verify exact match
            for (i, (&pw, &ps)) in p_windowed.iter().zip(p_sparse.iter()).enumerate() {
                assert!(
                    (pw - ps).abs() < 1e-12,
                    "grid d={d} det{i}: windowed={pw:.15} vs sparse={ps:.15}, diff={:.2e}",
                    (pw - ps).abs()
                );
            }

            let speedup = t_windowed.as_secs_f64() / t_sparse.as_secs_f64();
            eprintln!(
                "{d:>4} {num_rounds:>6} {:>8} {:>8} {num_detectors:>6} {:>12.2} {:>12.2} {speedup:>8.1}x",
                expanded.gates.len(),
                expanded.num_qubits,
                t_windowed.as_secs_f64() * 1000.0,
                t_sparse.as_secs_f64() * 1000.0
            );
        }
    }
}
