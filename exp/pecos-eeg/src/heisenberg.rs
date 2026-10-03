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
//! Independent stochastic (S-type) injections have physical probability p=-s:
//!   D → D          (when D and P commute)
//!   D → (1-2p)·D   (when D and P anticommute)
//! Categorical depolarizing is a separate channel, with nonidentity eigenvalue
//! 1-4p/3 (one qubit) or 1-16p/15 (two qubits).
//!
//! The cost is exponential in the number of anticommuting H-type noise
//! sources per detector (2^m terms), but this is typically manageable
//! for QEC circuits (m ~ 5-15). S-type noise does not increase term count.
//!
//! The Pauli-tracking walks support these coherent and stochastic channels.
//! A dense-matrix reference ([`heisenberg_exact_from_circuit`], up to ~20
//! qubits) applies the same noise and checks the walks.

use crate::Bm;
use crate::noise::{GateNoise, NoiseSpec};
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

/// Mark `q` active, growing the bitmap: noise may act on qubits no gate touches.
fn activate(active: &mut Vec<bool>, q: usize) {
    if q >= active.len() {
        active.resize(q + 1, false);
    }
    active[q] = true;
}

fn is_active(active: &[bool], q: usize) -> bool {
    active.get(q).copied().unwrap_or(false)
}

/// Whether any of the noise's injections or channels acts on an active qubit.
/// Noise acting only on inactive qubits commutes with every term.
fn noise_touches_active(noise: &GateNoise, active: &[bool]) -> bool {
    noise.qubits().any(|q| is_active(active, q))
}

fn activate_qubit(
    q: u16,
    before_gate: u32,
    active: &mut Vec<bool>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<u32>,
    gate_index: &crate::expand::GateIndex,
) {
    let qu = q as usize;
    activate(active, qu);
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

/// Build exact gate noise, skipping gates introduced by measurement expansion.
/// Returns `None` for gates with neither injections nor categorical channels.
///
/// # Panics
/// Panics if the provenance flags do not match the gate count.
pub fn build_noise_map(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    expansion_gates: &[bool],
) -> Vec<Option<GateNoise>> {
    crate::expand::assert_one_per_gate("expansion_gates", expansion_gates.len(), gates.len());

    gates
        .iter()
        .enumerate()
        .map(|(i, gate)| {
            if expansion_gates[i] {
                return None;
            }
            let qubits: SmallVec<[usize; 4]> =
                gate.qubits.iter().map(pecos_core::QubitId::index).collect();
            let exact = noise.exact_noise_after_gate(i, gate.gate_type, &qubits);
            (!exact.injections.is_empty() || !exact.depolarizing.is_empty()).then_some(exact)
        })
        .collect()
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

fn apply_depolarizing(
    terms: &mut [HeisenbergTerm],
    channels: &[crate::noise::DepolarizingChannel],
) {
    for channel in channels {
        let scale = channel.eigenvalue();
        for term in terms.iter_mut() {
            if channel
                .qubits()
                .iter()
                .any(|&q| term.pauli.has_x(q as u16) || term.pauli.has_z(q as u16))
            {
                // Zero and negative eigenvalues are valid, including p=3/4
                // (1q), p=15/16 (2q), and stronger depolarizing channels.
                term.coeff_re *= scale;
                term.coeff_im *= scale;
            }
        }
    }
}

/// Compute detection probability via backward Heisenberg propagation.
///
/// Operates on the EXPANDED circuit (from [`crate::expand`]). Expansion
/// gates marked in `expansion_gates` are skipped for noise injection.
///
/// Handles both H-type (coherent) and S-type (stochastic) noise.
///
/// # Arguments
/// * `gates` - The expanded circuit gates
/// * `detector` - Detector as Z on auxiliary qubit(s) (expanded frame)
/// * `noise` - Noise specification
/// * `initial_stab` - Stabilizer group of |0...0⟩
/// * `prune_threshold` - Drop terms with |coefficient| below this (0 for exact)
/// * `expansion_gates` - Provenance flags parallel to `gates`; all false for an unexpanded circuit
///
/// # Panics
/// Panics if the provenance flags do not match the gate count.
pub fn heisenberg_detection_probability(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
    expansion_gates: &[bool],
) -> f64 {
    heisenberg_windowed(
        gates,
        detector,
        noise,
        initial_stab,
        prune_threshold,
        None,
        expansion_gates,
    )
}

/// Backward Heisenberg with precomputed noise map and BTreeMap-based merging.
///
/// Uses BTreeMap<SparsePauli, (re, im)> for continuous dedup — no separate
/// merge step. Terms are merged on insert via BTreeMap's O(log n) lookup.
/// Applies explicit categorical channels from the precomputed noise map.
///
/// # Panics
/// Panics if the noise map does not match the gate count.
#[must_use]
pub fn heisenberg_with_noise_map(
    gates: &[Gate],
    detector: &Bm,
    noise_map: &[Option<GateNoise>],
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
) -> f64 {
    crate::expand::assert_one_per_gate("noise_map", noise_map.len(), gates.len());

    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    // Conservative active-qubit bitmap
    let mut active_qubits = Vec::new();
    for &q in terms[0]
        .pauli
        .x_qubits
        .iter()
        .chain(terms[0].pauli.z_qubits.iter())
    {
        activate(&mut active_qubits, q as usize);
    }

    let mut last_merge_count = 1usize;
    let mut sin_branches: Vec<HeisenbergTerm> = Vec::new();

    for i in (0..gates.len()).rev() {
        let gate = &gates[i];
        let gate_qs: SmallVec<[u16; 4]> = gate.qubits.iter().map(|q| q.index() as u16).collect();

        // Noise is relevant by its own support, which may lie outside the gate.
        let gate_noise = noise_map[i]
            .as_ref()
            .filter(|gn| noise_touches_active(gn, &active_qubits));

        let noise_applied = gate_noise.is_some();
        if let Some(gn) = gate_noise {
            // Individual injections in their original order
            for inj in &gn.injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }
                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> =
                            if inj.label.x_bits.is_zero() && inj.label.weight() == 1 {
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
                                activate(&mut active_qubits, q as usize);
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
                        // Custom independent S-type injection.
                        let s = inj.rate;
                        if s.abs() < 1e-20 {
                            continue;
                        }
                        let p = -s;
                        let scale = 1.0 - 2.0 * p;
                        let single_q: Option<u16> = if inj.label.weight() == 1 {
                            inj.label
                                .x_bits
                                .highest_set_bit()
                                .or_else(|| inj.label.z_bits.highest_set_bit())
                                .map(|q| q as u16)
                        } else {
                            None
                        };
                        if let Some(q) = single_q {
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    != (has_x_in_noise && term.pauli.has_z(q));
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

            apply_depolarizing(&mut terms, &gn.depolarizing);
        }

        // Step 2: Backward Clifford conjugation. Checked after the noise,
        // which may have branched terms onto the gate's qubits.
        let gate_relevant = gate_qs
            .iter()
            .any(|&q| is_active(&active_qubits, q as usize));
        if !noise_applied && !gate_relevant {
            continue;
        }

        if gate_relevant {
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
                            activate(&mut active_qubits, q as usize);
                        }
                    }
                }
            }
        }

        // Prune and merge after noise alone too: it can add or shrink terms.
        // Prune
        if prune_threshold > 0.0 {
            let thresh_sq = prune_threshold * prune_threshold;
            terms.retain(|t| t.coeff_re * t.coeff_re + t.coeff_im * t.coeff_im > thresh_sq);
        }

        // Merge: sort + dedup. In-place, no allocation, cache-friendly.
        // For typical term counts (~50-100), this beats both HashMap and
        // BTreeMap due to zero allocation overhead and sequential access.
        let should_merge = match gate.gate_type {
            // Reset and measurement remove terms, so merge eagerly, but only
            // when the gate acted; noise alone grows terms like any gate.
            GateType::PZ | GateType::QAlloc | GateType::MZ if gate_relevant => terms.len() > 4,
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
///
/// # Panics
/// Panics if the provenance flags do not match the gate count.
pub fn heisenberg_windowed(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
    gate_window: Option<(usize, usize)>,
    expansion_gates: &[bool],
) -> f64 {
    crate::expand::assert_one_per_gate("expansion_gates", expansion_gates.len(), gates.len());

    // Start with the detector as a single sparse term
    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    let mut last_merge_count = 1usize;
    // #3: Pre-allocate sin branches buffer, reused across noise sources
    let mut sin_branches: Vec<HeisenbergTerm> = Vec::new();

    // Conservative active-qubit bitmap: once a qubit is active, stays active.
    // This avoids the expensive per-term scan for gate relevance.
    let mut active_qubits = Vec::new();
    // Seed from detector
    for &q in terms[0]
        .pauli
        .x_qubits
        .iter()
        .chain(terms[0].pauli.z_qubits.iter())
    {
        activate(&mut active_qubits, q as usize);
    }

    // Walk backward through the circuit (optionally windowed)
    let (walk_start, walk_end) = gate_window.unwrap_or((0, gates.len()));
    for i in (walk_start..walk_end).rev() {
        let gate = &gates[i];
        let gate_qs: SmallVec<[u16; 4]> = gate.qubits.iter().map(|q| q.index() as u16).collect();

        // Step 1: Apply noise adjoint (skip expansion gates). Noise is relevant
        // by its own support, which may lie outside the gate.
        let gate_noise = (!expansion_gates[i])
            .then(|| {
                let qubits_usize: SmallVec<[usize; 4]> =
                    gate_qs.iter().map(|&q| q as usize).collect();
                noise.exact_noise_after_gate(i, gate.gate_type, &qubits_usize)
            })
            .filter(|exact| noise_touches_active(exact, &active_qubits));
        let noise_applied = gate_noise.is_some();
        if let Some(exact) = gate_noise {
            for inj in &exact.injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }

                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> =
                            if inj.label.x_bits.is_zero() && inj.label.weight() == 1 {
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
                                activate(&mut active_qubits, q as usize);
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
                        let single_q: Option<u16> = if inj.label.weight() == 1 {
                            inj.label
                                .x_bits
                                .highest_set_bit()
                                .or_else(|| inj.label.z_bits.highest_set_bit())
                                .map(|q| q as u16)
                        } else {
                            None
                        };

                        if let Some(q) = single_q {
                            // Single-qubit S noise: check just the one qubit
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                // Anticommutes when exactly one symplectic overlap is present
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    != (has_x_in_noise && term.pauli.has_z(q));
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
            apply_depolarizing(&mut terms, &exact.depolarizing);
        }

        // Step 2: Conjugate backward through the gate.
        // #2: Skip gates that don't touch active qubits. Checked after the
        // noise, which may have branched terms onto the gate's qubits.
        let gate_relevant = gate_qs
            .iter()
            .any(|&q| is_active(&active_qubits, q as usize));
        if !noise_applied && !gate_relevant {
            continue;
        }

        if gate_relevant {
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
                            activate(&mut active_qubits, q as usize);
                        }
                    }
                }
            }
        }

        // Merge after noise alone too: it can add duplicate terms.
        // Merge duplicate Pauli terms by sorting + linear scan.
        let should_merge = match gate.gate_type {
            // Reset and measurement remove terms, so merge eagerly, but only
            // when the gate acted; noise alone grows terms like any gate.
            GateType::PZ | GateType::QAlloc | GateType::MZ if gate_relevant => terms.len() > 4,
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
/// Accepts an optional precomputed noise map. Gates without a map entry call
/// `noise.exact_noise_after_gate()`.
///
/// # Panics
/// Panics if the index provenance flags or a supplied noise map do not match the gate count.
pub fn heisenberg_sparse(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial_stab: &StabilizerGroup,
    prune_threshold: f64,
    gate_index: &crate::expand::GateIndex,
    noise_map: Option<&[Option<GateNoise>]>,
) -> f64 {
    crate::expand::assert_one_per_gate(
        "gate_index.expansion_gates",
        gate_index.expansion_gates.len(),
        gates.len(),
    );

    if let Some(noise_map) = noise_map {
        crate::expand::assert_one_per_gate("noise_map", noise_map.len(), gates.len());
    }
    let mut terms = vec![HeisenbergTerm {
        pauli: SparsePauli::from_bm(detector),
        coeff_re: 1.0,
        coeff_im: 0.0,
    }];

    // Active qubit set: union of all terms' support.
    // Use a Vec<bool> for O(1) check (faster than BTreeSet for small qubit counts).
    let mut active = Vec::new();

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
            let dynamic_noise;
            let gate_noise = if let Some(gn) = noise_map.and_then(|nm| nm[i].as_ref()) {
                gn
            } else {
                let qubits_usize: SmallVec<[usize; 4]> =
                    gate_qs.iter().map(|&q| q as usize).collect();
                dynamic_noise = noise.exact_noise_after_gate(i, gate.gate_type, &qubits_usize);
                &dynamic_noise
            };

            for inj in &gate_noise.injections {
                match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let h = inj.rate;
                        if h.abs() < 1e-20 {
                            continue;
                        }

                        let cos2h = (2.0 * h).cos();
                        let sin2h = (2.0 * h).sin();

                        let single_z_qubit: Option<u16> =
                            if inj.label.x_bits.is_zero() && inj.label.weight() == 1 {
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
                        // Custom S injections retain independent flip semantics.
                        let s = inj.rate;
                        if s.abs() < 1e-20 {
                            continue;
                        }
                        let p = -s;
                        let scale = 1.0 - 2.0 * p;

                        let single_q: Option<u16> = if inj.label.weight() == 1 {
                            inj.label
                                .x_bits
                                .highest_set_bit()
                                .or_else(|| inj.label.z_bits.highest_set_bit())
                                .map(|q| q as u16)
                        } else {
                            None
                        };

                        if let Some(q) = single_q {
                            let has_x_in_noise = inj.label.x_bits.highest_set_bit().is_some();
                            let has_z_in_noise = inj.label.z_bits.highest_set_bit().is_some();
                            for term in &mut terms {
                                let anti = (has_z_in_noise && term.pauli.has_x(q))
                                    != (has_x_in_noise && term.pauli.has_z(q));
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

            apply_depolarizing(&mut terms, &gate_noise.depolarizing);
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

        // Merge duplicate Pauli terms. Reset and measurement merge eagerly only
        // when the gate acted: this walk also pops gates whose noise alone
        // reaches an active qubit.
        let gate_acted = gate_qs.iter().any(|&q| is_active(&active, q as usize));
        let should_merge = match gate.gate_type {
            GateType::PZ | GateType::QAlloc | GateType::MZ if gate_acted => terms.len() > 4,
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
) -> Result<f64, crate::expand::EegBuildError> {
    let expanded = crate::expand::expand_circuit(original_gates)?;

    let mut detector = Bm::default();
    for &m in detector_meas_indices {
        // Single resolver: an out-of-range record is an error, never a
        // silently thinner detector.
        detector.z_bits.set_bit(expanded.aux_qubit_for_record(m)?);
    }

    let init_gates: Vec<Gate> = (0..num_original_qubits)
        .map(|q| crate::expand::make_gate(GateType::PZ, &[q]))
        .collect();
    let stab = StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    Ok(heisenberg_detection_probability(
        &expanded.gates,
        &detector,
        noise,
        &stab,
        prune_threshold,
        &expanded.expansion_gates,
    ))
}

/// Detection probability via a dense-matrix reference for the exact walks.
///
/// Computes the backward adjoint using dense 2^n × 2^n complex matrix
/// operations, with O(4^n) memory. Supports PZ, QAlloc, MZ, H, and CX in the
/// expanded circuit. Noise is the physical view from
/// [`NoiseSpec::exact_noise_after_gate`]: H-type rotations U = exp(-i h P) and
/// S-type Pauli channels with probability p = -s, for arbitrary Pauli strings
/// P, followed by categorical depolarizing channels, each applied as the
/// explicit sum over its 3 or 15 nonidentity Paulis.
/// Identity labels have no effect. Expansion gates receive no noise.
/// Useful as a reference/validation for the faster
/// Pauli-tracking walk ([`heisenberg_detection_probability_from_circuit`]).
///
/// # Errors
///
/// Returns an error for C/A injections (even at zero rate), a label or
/// categorical channel acting on a qubit outside the expanded circuit, any
/// unimplemented expanded gate adjoint, or an expansion/measurement-record
/// resolution error.
///
/// # Panics
///
/// Panics if the expanded circuit has more than 20 qubits.
pub fn heisenberg_exact_from_circuit(
    original_gates: &[Gate],
    detector_meas_indices: &[usize],
    noise: &dyn NoiseSpec,
    _num_original_qubits: usize,
) -> Result<f64, crate::expand::EegBuildError> {
    let expanded = crate::expand::expand_circuit(original_gates)?;
    let n = expanded.num_qubits;

    assert!(
        n <= 20,
        "Matrix Heisenberg requires 2^n memory; {n} qubits is too large. Use the Pauli-tracking walk for approximate results."
    );

    let dim = 1usize << n;

    // Build detector matrix: diagonal with Z eigenvalues on the detector aux qubits.
    let mut obs_re = vec![0.0f64; dim * dim];
    let obs_im = vec![0.0f64; dim * dim];
    // Pre-resolve outside the matrix loop: out-of-range is an error, and the
    // resolver must not run 2^n times.
    let mut detector_aux = Vec::with_capacity(detector_meas_indices.len());
    for &m in detector_meas_indices {
        detector_aux.push(expanded.aux_qubit_for_record(m)?);
    }
    for i in 0..dim {
        let mut eigenvalue = 1.0f64;
        for &aux in &detector_aux {
            {
                if (i >> aux) & 1 == 1 {
                    eigenvalue = -eigenvalue;
                }
            }
        }
        obs_re[i * dim + i] = eigenvalue;
    }

    // Provenance recorded by measurement expansion
    let expansion_gates = &expanded.expansion_gates;

    // Walk backward, applying adjoints via matrix multiplication.
    let mut im = obs_im;
    for idx in (0..expanded.gates.len()).rev() {
        let g = &expanded.gates[idx];
        let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();

        // Noise adjoint (skip expansion gates)
        if !expansion_gates[idx] {
            let exact = noise.exact_noise_after_gate(idx, g.gate_type, &qs);
            for inj in &exact.injections {
                let weights = match inj.eeg_type {
                    crate::eeg::EegType::H => {
                        let (s, c) = inj.rate.sin_cos();
                        (c * c, s * s, s * c)
                    }
                    crate::eeg::EegType::S => {
                        let p = -inj.rate;
                        (1.0 - p, p, 0.0)
                    }
                    eeg_type => {
                        return Err(crate::expand::EegBuildError::UnsupportedExactNoise {
                            eeg_type,
                        });
                    }
                };
                if let Some(qubit) = inj
                    .label
                    .x_bits
                    .highest_set_bit()
                    .max(inj.label.z_bits.highest_set_bit())
                    .filter(|&q| q >= n)
                {
                    return Err(crate::expand::EegBuildError::ExactLabelOutOfRange {
                        qubit,
                        num_qubits: n,
                    });
                }
                if inj.rate.abs() < 1e-20 || inj.label.is_identity() {
                    continue;
                }
                matrix_pauli_adjoint(&mut obs_re, &mut im, &inj.label, weights, n);
            }
            for channel in &exact.depolarizing {
                if let Some(&qubit) = channel.qubits().iter().find(|&&q| q >= n) {
                    return Err(crate::expand::EegBuildError::ExactLabelOutOfRange {
                        qubit,
                        num_qubits: n,
                    });
                }
                matrix_depolarizing_adjoint(&mut obs_re, &mut im, channel, n);
            }
        }

        // Gate adjoint
        match g.gate_type {
            GateType::PZ | GateType::QAlloc => {
                for &q in &qs {
                    matrix_pz_adjoint(&mut obs_re, &mut im, q, n);
                }
            }
            GateType::MZ => {
                for &q in &qs {
                    matrix_mz_adjoint(&mut obs_re, &mut im, q, n);
                }
            }
            GateType::H => {
                for &q in &qs {
                    matrix_h_adjoint(&mut obs_re, &mut im, q, n);
                }
            }
            GateType::CX if qs.len() >= 2 && qs.len().is_multiple_of(2) => {
                for pair in qs.rchunks_exact(2) {
                    matrix_cx_adjoint(&mut obs_re, &mut im, pair[0], pair[1], n);
                }
            }
            GateType::I | GateType::Idle => {}
            gate_type => {
                return Err(crate::expand::EegBuildError::UnsupportedExactGate { gate_type });
            }
        }
    }

    // ⟨0...0|O_backward|0...0⟩ = obs_re[0]
    let expectation = obs_re[0];
    let prob = 0.5 * (1.0 - expectation);
    Ok(prob.clamp(0.0, 1.0))
}

// --- Matrix helpers for exact Heisenberg ---

fn matrix_phase(re: f64, im: f64, phase: u8) -> (f64, f64) {
    match phase % 4 {
        0 => (re, im),
        1 => (-im, re),
        2 => (-re, -im),
        3 => (im, -re),
        _ => unreachable!(),
    }
}

/// Apply a O + b P O P + i c (P O - O P), using P's monomial action.
fn matrix_pauli_adjoint(
    re: &mut [f64],
    im: &mut [f64],
    label: &Bm,
    (a, b, c): (f64, f64, f64),
    n: usize,
) {
    let dim = 1usize << n;
    let action: Vec<_> = (0..dim)
        .map(|i| {
            let state = SmallVec::from_slice(&[u64::try_from(i).expect("basis index fits in u64")]);
            let (image, phase) = label.apply_to_basis_state(&state);
            // The dense walk is limited to 20 qubits, so one word suffices.
            let index = usize::try_from(image[0]).expect("basis image fits in usize");
            (index, phase)
        })
        .collect();
    let mut new_re = vec![0.0; dim * dim];
    let mut new_im = vec![0.0; dim * dim];
    for (i, &(pi, phase_i)) in action.iter().enumerate() {
        for (j, &(pj, phase_j)) in action.iter().enumerate() {
            // P is Hermitian: <i|P = conjugate(i^phase_i) <pi|.
            let po = pi * dim + j;
            let op = i * dim + pj;
            let pop = pi * dim + pj;
            let (po_re, po_im) = matrix_phase(re[po], im[po], 4 - phase_i);
            let (op_re, op_im) = matrix_phase(re[op], im[op], phase_j);
            let (pop_re, pop_im) = matrix_phase(re[pop], im[pop], 4 - phase_i + phase_j);
            let idx = i * dim + j;
            new_re[idx] = a * re[idx] + b * pop_re - c * (po_im - op_im);
            new_im[idx] = a * im[idx] + b * pop_im + c * (po_re - op_re);
        }
    }
    re.copy_from_slice(&new_re);
    im.copy_from_slice(&new_im);
}

/// Apply (1-p) O + (p/k) Σ P O P over the channel's k nonidentity Paulis.
fn matrix_depolarizing_adjoint(
    re: &mut [f64],
    im: &mut [f64],
    channel: &crate::noise::DepolarizingChannel,
    n: usize,
) {
    use crate::noise::DepolarizingChannel;
    let local = |q: usize| [Bm::default(), Bm::x(q), Bm::y(q), Bm::z(q)];
    let (paulis, probability): (Vec<Bm>, f64) = match *channel {
        DepolarizingChannel::OneQubit { qubit, probability } => {
            (local(qubit)[1..].to_vec(), probability)
        }
        DepolarizingChannel::TwoQubit {
            qubits: [qa, qb],
            probability,
        } => {
            let paulis = local(qa)
                .iter()
                .flat_map(|a| local(qb).map(|b| a.multiply(&b)))
                .skip(1) // I ⊗ I
                .collect();
            (paulis, probability)
        }
    };
    let weight = probability / paulis.len() as f64;
    let mut sum_re: Vec<f64> = re.iter().map(|v| (1.0 - probability) * v).collect();
    let mut sum_im: Vec<f64> = im.iter().map(|v| (1.0 - probability) * v).collect();
    for pauli in &paulis {
        let mut pop_re = re.to_vec();
        let mut pop_im = im.to_vec();
        matrix_pauli_adjoint(&mut pop_re, &mut pop_im, pauli, (0.0, 1.0, 0.0), n);
        for (sum, pop) in sum_re.iter_mut().zip(&pop_re) {
            *sum += weight * pop;
        }
        for (sum, pop) in sum_im.iter_mut().zip(&pop_im) {
            *sum += weight * pop;
        }
    }
    re.copy_from_slice(&sum_re);
    im.copy_from_slice(&sum_im);
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

    struct PauliAfterGate {
        gate_index: usize,
        label: Bm,
        probability: f64,
    }

    impl NoiseSpec for PauliAfterGate {
        fn noise_after_gate(
            &self,
            gate_index: usize,
            _gate_type: GateType,
            _qubits: &[usize],
        ) -> Vec<crate::noise::NoiseInjection> {
            if gate_index == self.gate_index {
                vec![crate::noise::NoiseInjection {
                    eeg_type: crate::eeg::EegType::S,
                    label: self.label.clone(),
                    label2: None,
                    rate: -self.probability,
                }]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn test_s_injection_y_eigenstate() {
        // Issue #942: Y noise on the Y eigenstate prepared by SX changes
        // only its global phase, so undoing SX must always measure zero.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::SX, &[0]),
            gate(GateType::SXdg, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let stab = StabilizerGroup::from_circuit(&gates[..1], 1);
        for probability in [0.01, 0.2] {
            let noise = PauliAfterGate {
                gate_index: 1,
                label: Bm::y(0),
                probability,
            };
            let actual = heisenberg_detection_probability(
                &gates,
                &Bm::z(0),
                &noise,
                &stab,
                0.0,
                &vec![false; gates.len()],
            );
            assert!(actual.abs() < 1e-12, "p={probability}: got {actual}");
        }
    }

    fn check_s_injection_pauli_matrix(walk: &str) {
        // Prepare an eigenstate of X, Y, or Z, inject a Pauli, then undo
        // the preparation. The independent analytic oracle is Pauli algebra:
        // equal nonidentity Paulis commute (zero detection probability),
        // while distinct ones anticommute (detection probability p).
        let bases = [
            ("X", GateType::H, GateType::H),
            ("Y", GateType::SX, GateType::SXdg),
            ("Z", GateType::Z, GateType::Z),
        ];
        for (term, prepare, undo) in bases {
            let gates = vec![
                gate(GateType::PZ, &[0]),
                gate(prepare, &[0]),
                gate(undo, &[0]),
                gate(GateType::MZ, &[0]),
            ];
            let stab = StabilizerGroup::from_circuit(&gates[..1], 1);
            for (injection, label) in [("X", Bm::x(0)), ("Y", Bm::y(0)), ("Z", Bm::z(0))] {
                for probability in [0.0, 0.01, 0.2, 0.5, 0.75, 1.0] {
                    let noise = PauliAfterGate {
                        gate_index: 1,
                        label: label.clone(),
                        probability,
                    };
                    let gate_index = crate::expand::GateIndex::build(
                        &gates,
                        1,
                        &noise,
                        &vec![false; gates.len()],
                    );
                    let noise_map = build_noise_map(&gates, &noise, &gate_index.expansion_gates);
                    let actual = match walk {
                        "windowed" => heisenberg_detection_probability(
                            &gates,
                            &Bm::z(0),
                            &noise,
                            &stab,
                            0.0,
                            &vec![false; gates.len()],
                        ),
                        "noise_map" => {
                            heisenberg_with_noise_map(&gates, &Bm::z(0), &noise_map, &stab, 0.0)
                        }
                        "sparse" | "sparse_noise_map" => heisenberg_sparse(
                            &gates,
                            &Bm::z(0),
                            &noise,
                            &stab,
                            0.0,
                            &gate_index,
                            (walk == "sparse_noise_map").then_some(noise_map.as_slice()),
                        ),
                        _ => unreachable!(),
                    };
                    let expected = if injection == term { 0.0 } else { probability };
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "{walk}: injection={injection}, term={term}, p={probability}: \
                         expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_s_injection_pauli_matrix_windowed() {
        check_s_injection_pauli_matrix("windowed");
    }

    #[test]
    fn test_s_injection_pauli_matrix_noise_map() {
        check_s_injection_pauli_matrix("noise_map");
    }

    #[test]
    fn test_s_injection_pauli_matrix_sparse() {
        check_s_injection_pauli_matrix("sparse");
    }

    #[test]
    fn test_s_injection_pauli_matrix_sparse_noise_map() {
        check_s_injection_pauli_matrix("sparse_noise_map");
    }

    #[test]
    fn test_s_injection_two_qubit_labels() {
        // Two-qubit Paulis whose highest X and Z bits share a qubit must not
        // take the single-qubit fast path. Z0 passes both CXs unchanged, so
        // the detection probability is p exactly when the injected Pauli
        // anticommutes with Z0, i.e. has X or Y on qubit 0.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::MZ, &[0]),
        ];
        let stab = StabilizerGroup::from_circuit(&gates[..2], 2);
        let labels = [
            ("X0X1", Bm::x(0).multiply(&Bm::x(1)), true),
            ("X0Y1", Bm::x(0).multiply(&Bm::y(1)), true),
            ("Y0X1", Bm::y(0).multiply(&Bm::x(1)), true),
            ("Y0Y1", Bm::y(0).multiply(&Bm::y(1)), true),
            ("Z0X1", Bm::z(0).multiply(&Bm::x(1)), false),
            ("Z0Y1", Bm::z(0).multiply(&Bm::y(1)), false),
        ];
        for (name, label, anticommutes) in labels {
            let probability = 0.1;
            let noise = PauliAfterGate {
                gate_index: 2,
                label,
                probability,
            };
            let gate_index =
                crate::expand::GateIndex::build(&gates, 2, &noise, &vec![false; gates.len()]);
            let noise_map = build_noise_map(&gates, &noise, &gate_index.expansion_gates);
            let expected = if anticommutes { probability } else { 0.0 };
            let results = [
                (
                    "windowed",
                    heisenberg_detection_probability(
                        &gates,
                        &Bm::z(0),
                        &noise,
                        &stab,
                        0.0,
                        &vec![false; gates.len()],
                    ),
                ),
                (
                    "noise_map",
                    heisenberg_with_noise_map(&gates, &Bm::z(0), &noise_map, &stab, 0.0),
                ),
                (
                    "sparse",
                    heisenberg_sparse(&gates, &Bm::z(0), &noise, &stab, 0.0, &gate_index, None),
                ),
                (
                    "sparse_noise_map",
                    heisenberg_sparse(
                        &gates,
                        &Bm::z(0),
                        &noise,
                        &stab,
                        0.0,
                        &gate_index,
                        Some(noise_map.as_slice()),
                    ),
                ),
            ];
            for (walk, actual) in results {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{walk}: label={name}: expected {expected}, got {actual}"
                );
            }
        }
    }

    struct CoherentAfterGate {
        gate_index: usize,
        label: Bm,
        angle: f64,
    }

    impl NoiseSpec for CoherentAfterGate {
        fn noise_after_gate(
            &self,
            gate_index: usize,
            _gate_type: GateType,
            _qubits: &[usize],
        ) -> Vec<crate::noise::NoiseInjection> {
            if gate_index == self.gate_index {
                vec![crate::noise::NoiseInjection {
                    eeg_type: crate::eeg::EegType::H,
                    label: self.label.clone(),
                    label2: None,
                    rate: self.angle,
                }]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn test_h_injection_two_qubit_z_label() {
        // A Z0Z1 rotation must not take the single-qubit Z fast path.
        // The rotation follows the first two-qubit gate. With CZs on |++>
        // the detector Z0 is X0Z1 there, which anticommutes with Z0Z1, so the
        // detection probability is sin^2(angle). With CXs on the Bell state
        // it is X0X1, which commutes with Z0Z1, so it is zero.
        let plus_plus = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[1]),
            gate(GateType::CZ, &[0, 1]),
            gate(GateType::CZ, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let bell = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let label = Bm::z(0).multiply(&Bm::z(1));
        for angle in [0.1_f64, 0.3] {
            for (state, gates, injection_gate, expected) in [
                ("plus_plus", &plus_plus, 4, angle.sin().powi(2)),
                ("bell", &bell, 3, 0.0),
            ] {
                let stab = StabilizerGroup::from_circuit(&gates[..2], 2);
                let noise = CoherentAfterGate {
                    gate_index: injection_gate,
                    label: label.clone(),
                    angle,
                };
                let gate_index =
                    crate::expand::GateIndex::build(gates, 2, &noise, &vec![false; gates.len()]);
                let noise_map = build_noise_map(gates, &noise, &gate_index.expansion_gates);
                let results = [
                    (
                        "windowed",
                        heisenberg_detection_probability(
                            gates,
                            &Bm::z(0),
                            &noise,
                            &stab,
                            0.0,
                            &vec![false; gates.len()],
                        ),
                    ),
                    (
                        "noise_map",
                        heisenberg_with_noise_map(gates, &Bm::z(0), &noise_map, &stab, 0.0),
                    ),
                    (
                        "sparse",
                        heisenberg_sparse(gates, &Bm::z(0), &noise, &stab, 0.0, &gate_index, None),
                    ),
                    (
                        "sparse_noise_map",
                        heisenberg_sparse(
                            gates,
                            &Bm::z(0),
                            &noise,
                            &stab,
                            0.0,
                            &gate_index,
                            Some(noise_map.as_slice()),
                        ),
                    ),
                ];
                for (walk, actual) in results {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "{walk}: state={state}, angle={angle}: expected {expected}, got {actual}"
                    );
                }
            }
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

        let expanded = expand::expand_circuit(&gates_orig).expect("supported circuit");
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
        let p1_heis = heisenberg_detection_probability(
            &expanded.gates,
            &det1,
            &noise,
            &stab,
            1e-10,
            &expanded.expansion_gates,
        );
        let p2_heis = heisenberg_detection_probability(
            &expanded.gates,
            &det2,
            &noise,
            &stab,
            1e-10,
            &expanded.expansion_gates,
        );

        // For comparison: forward EEG
        let eeg_result =
            crate::circuit::analyze_with_noise(&expanded.gates, &noise, &expanded.expansion_gates);
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

        let p_heis = heisenberg_detection_probability(
            &gates_orig,
            &det,
            &noise,
            &stab,
            0.0,
            &vec![false; gates_orig.len()],
        );

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

            let p = heisenberg_detection_probability(
                &gates_orig,
                &det,
                &noise,
                &stab,
                0.0,
                &vec![false; gates_orig.len()],
            );

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
            let p = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 2)
                .expect("supported circuit");
            let exact = theta.sin().powi(2);
            assert!(
                (p - exact).abs() < 1e-10,
                "theta={theta}: exact_heisenberg {p:.10} vs sin²(θ) {exact:.10}"
            );
        }
    }

    #[test]
    fn test_exact_categorical_depolarizing() {
        // Two H locations each attenuate the detector by 1-4p/3; at p=3/4 the
        // channel is fully depolarizing and the detector is a coin flip.
        let one_qubit = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        // CX preserves |00>; each nontrivial Z-type detector anticommutes
        // with 8 of the 15 exclusive two-qubit Paulis.
        let two_qubit = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
        ];
        for p in [0.0_f64, 0.1, 0.3, 0.75, 0.9375, 1.0] {
            let mut noise = crate::noise::UniformNoise::coherent_only(0.0);
            noise.p1 = p;
            let actual = heisenberg_exact_from_circuit(&one_qubit, &[0], &noise, 1).unwrap();
            let expected = (1.0 - (1.0 - 4.0 * p / 3.0).powi(2)) / 2.0;
            assert!(
                (actual - expected).abs() < 1e-12,
                "p1={p}: {actual} vs {expected}"
            );

            let mut noise = crate::noise::UniformNoise::coherent_only(0.0);
            noise.p2 = p;
            for detector in [&[0][..], &[1], &[0, 1]] {
                let actual =
                    heisenberg_exact_from_circuit(&two_qubit, detector, &noise, 2).unwrap();
                let expected = 8.0 * p / 15.0;
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "p2={p}, detector {detector:?}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_exact_matches_walk_under_mixed_uniform_noise() {
        // Two rounds of an X check on two data qubits, with every UniformNoise
        // channel active, so coherent, categorical and measurement noise mix.
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
        let noise = crate::noise::UniformNoise {
            idle_rz: 0.13,
            p1: 0.2,
            p2: 0.3,
            p_meas: 0.05,
            p_prep: 0.04,
        };
        let exact = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 3).unwrap();
        let walk =
            heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise, 3, 0.0).unwrap();
        assert!(
            (exact - walk).abs() < 1e-10,
            "matrix {exact} vs walk {walk}"
        );
    }

    struct ExactTestNoise(Vec<(usize, crate::noise::NoiseInjection)>);

    impl NoiseSpec for ExactTestNoise {
        fn noise_after_gate(
            &self,
            gate_index: usize,
            _gate_type: GateType,
            _qubits: &[usize],
        ) -> Vec<crate::noise::NoiseInjection> {
            self.0
                .iter()
                .filter(|(idx, _)| *idx == gate_index)
                .map(|(_, inj)| inj.clone())
                .collect()
        }
    }

    fn exact_test_injection(
        eeg_type: crate::eeg::EegType,
        label: Bm,
        rate: f64,
    ) -> crate::noise::NoiseInjection {
        crate::noise::NoiseInjection {
            eeg_type,
            label,
            label2: None,
            rate,
        }
    }

    #[test]
    fn test_exact_single_qubit_pauli_rotations() {
        let gates = [gate(GateType::PZ, &[0]), gate(GateType::MZ, &[0])];
        for h in [-0.7_f64, -0.3, 0.0, 0.13, 0.3, 0.8] {
            for (label, expected) in [
                (Bm::x(0), h.sin().powi(2)),
                (Bm::y(0), h.sin().powi(2)),
                (Bm::z(0), 0.0),
            ] {
                let noise = ExactTestNoise(vec![(
                    0,
                    exact_test_injection(crate::eeg::EegType::H, label.clone(), h),
                )]);
                let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 1).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "label={label:?}, h={h}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn test_exact_multi_qubit_zz_rotation() {
        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        for h in [-0.4_f64, 0.3, 0.7] {
            // Z0Z1 on |+0>, followed by H0: P(q0=1) = sin²(h).
            // Keeping only Z1 leaves |+0> unchanged and gives zero.
            let noise = ExactTestNoise(vec![(
                2,
                exact_test_injection(crate::eeg::EegType::H, Bm::z(0).multiply(&Bm::z(1)), h),
            )]);
            let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 2).unwrap();
            assert!((actual - h.sin().powi(2)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exact_multi_qubit_xx_rotation() {
        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::MZ, &[0]),
        ];
        for h in [-0.4_f64, 0.3, 0.7] {
            // exp(-i h X0X1)|00> = cos(h)|00> - i sin(h)|11>:
            // P(q0=1) = sin²(h); acting on q1 alone gives zero.
            let noise = ExactTestNoise(vec![(
                1,
                exact_test_injection(crate::eeg::EegType::H, Bm::x(0).multiply(&Bm::x(1)), h),
            )]);
            let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 2).unwrap();
            assert!((actual - h.sin().powi(2)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exact_noncommuting_rotations_forward_oracle() {
        use pecos_core::Angle64;
        use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};

        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::H, &[1]),
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
        ];
        let noise = ExactTestNoise(
            [
                (2, Bm::y(0), 0.23),
                (3, Bm::x(1), -0.31),
                (4, Bm::z(0), 0.41),
                (5, Bm::y(0).multiply(&Bm::z(1)), -0.19),
                (6, Bm::x(0).multiply(&Bm::x(1)), 0.27),
                (7, Bm::z(1), -0.37),
                (8, Bm::y(0).multiply(&Bm::y(1)), 0.17),
            ]
            .into_iter()
            .map(|(idx, label, h)| (idx, exact_test_injection(crate::eeg::EegType::H, label, h)))
            .collect(),
        );

        // Independent forward evolution on the original two qubits. Each
        // simulator rotation uses angle 2h for U = exp(-i h P).
        let mut state = StateVec::new(2);
        let q0 = [QubitId(0)];
        let q1 = [QubitId(1)];
        let pair = [(QubitId(0), QubitId(1))];
        state.h(&q0);
        state.ry(Angle64::from_radians(2.0 * 0.23), &q0);
        state.cx(&pair);
        state.rx(Angle64::from_radians(2.0 * -0.31), &q1);
        state.h(&q1);
        state.rz(Angle64::from_radians(2.0 * 0.41), &q0);
        state.cx(&[(QubitId(1), QubitId(0))]);
        // RX(pi/2) maps Y to Z; undo it after the ZZ rotation.
        state.rx(Angle64::from_radians(std::f64::consts::FRAC_PI_2), &q0);
        state.rzz(Angle64::from_radians(2.0 * -0.19), &pair);
        state.rx(Angle64::from_radians(-std::f64::consts::FRAC_PI_2), &q0);
        state.h(&q0);
        state.rxx(Angle64::from_radians(2.0 * 0.27), &pair);
        state.cx(&pair);
        state.rz(Angle64::from_radians(2.0 * -0.37), &q1);
        state.h(&q1);
        state.ryy(Angle64::from_radians(2.0 * 0.17), &pair);

        for (detector, expected) in [
            (vec![0], state.probability(1) + state.probability(3)),
            (vec![1], state.probability(2) + state.probability(3)),
            (vec![0, 1], state.probability(1) + state.probability(2)),
        ] {
            let actual = heisenberg_exact_from_circuit(&gates, &detector, &noise, 2).unwrap();
            assert!(
                (actual - expected).abs() < 1e-10,
                "detector={detector:?}: matrix={actual}, forward={expected}"
            );
        }
    }

    #[test]
    fn test_exact_stochastic_pauli_channels() {
        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::MZ, &[0]),
        ];
        for p in [0.0, 0.2, 0.5, 1.0] {
            for (label, expected) in [
                (Bm::x(0), p),
                (Bm::y(0), p),
                (Bm::z(0), 0.0),
                (Bm::x(0).multiply(&Bm::x(1)), p),
                (Bm::y(0).multiply(&Bm::y(1)), p),
                (Bm::z(0).multiply(&Bm::z(1)), 0.0),
            ] {
                let noise = ExactTestNoise(vec![(
                    1,
                    exact_test_injection(crate::eeg::EegType::S, label.clone(), -p),
                )]);
                let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 2).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "label={label:?}, p={p}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn test_exact_identity_injections() {
        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        for eeg_type in [crate::eeg::EegType::H, crate::eeg::EegType::S] {
            let noise = ExactTestNoise(vec![(
                1,
                exact_test_injection(eeg_type, Bm::default(), -0.7),
            )]);
            let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 1).unwrap();
            assert!((actual - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exact_unsupported_injections() {
        let gates = [gate(GateType::PZ, &[0]), gate(GateType::MZ, &[0])];
        for eeg_type in [crate::eeg::EegType::C, crate::eeg::EegType::A] {
            for rate in [0.0, 0.3] {
                for label in [Bm::default(), Bm::x(0)] {
                    let mut inj = exact_test_injection(eeg_type, label, rate);
                    inj.label2 = Some(Bm::z(0));
                    let noise = ExactTestNoise(vec![(0, inj)]);
                    assert_eq!(
                        heisenberg_exact_from_circuit(&gates, &[0], &noise, 1),
                        Err(expand::EegBuildError::UnsupportedExactNoise { eeg_type })
                    );
                }
            }
        }
    }

    #[test]
    fn test_exact_label_out_of_range() {
        /// A depolarizing channel on qubit 5 after gate 0.
        struct FarChannel;
        impl NoiseSpec for FarChannel {
            fn noise_after_gate(
                &self,
                _: usize,
                _: GateType,
                _: &[usize],
            ) -> Vec<crate::noise::NoiseInjection> {
                Vec::new()
            }
            fn exact_noise_after_gate(
                &self,
                gate_index: usize,
                _: GateType,
                _: &[usize],
            ) -> GateNoise {
                GateNoise {
                    injections: Vec::new(),
                    depolarizing: (gate_index == 0)
                        .then_some(crate::noise::DepolarizingChannel::OneQubit {
                            qubit: 5,
                            probability: 0.1,
                        })
                        .into_iter()
                        .collect(),
                }
            }
        }
        let gates = [gate(GateType::PZ, &[0]), gate(GateType::MZ, &[0])];
        // Expansion adds one aux qubit, so qubit 2 lies outside the circuit.
        for eeg_type in [crate::eeg::EegType::H, crate::eeg::EegType::S] {
            let noise = ExactTestNoise(vec![(0, exact_test_injection(eeg_type, Bm::x(2), -0.1))]);
            assert_eq!(
                heisenberg_exact_from_circuit(&gates, &[0], &noise, 1),
                Err(expand::EegBuildError::ExactLabelOutOfRange {
                    qubit: 2,
                    num_qubits: 2
                })
            );
        }

        // A categorical channel past the circuit is reported the same way.
        assert_eq!(
            heisenberg_exact_from_circuit(&gates, &[0], &FarChannel, 1),
            Err(expand::EegBuildError::ExactLabelOutOfRange {
                qubit: 5,
                num_qubits: 2
            })
        );
    }

    #[test]
    fn test_exact_unsupported_gates() {
        for unsupported in [
            gate(GateType::SZ, &[0]),
            gate(GateType::CZ, &[0, 1]),
            gate(GateType::CX, &[0, 1, 2]),
        ] {
            let gate_type = unsupported.gate_type;
            let gates = [
                gate(GateType::PZ, &[0]),
                gate(GateType::PZ, &[1]),
                unsupported,
                gate(GateType::MZ, &[0]),
            ];
            assert_eq!(
                heisenberg_exact_from_circuit(&gates, &[0], &ExactTestNoise(vec![]), 2),
                Err(expand::EegBuildError::UnsupportedExactGate { gate_type })
            );
        }
    }

    #[test]
    fn test_exact_skips_expansion_noise() {
        let gates = [
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        // Expansion inserts QAlloc, CX, PZ at indices 1..=3 and 5..=7.
        let noise = ExactTestNoise(
            [1, 2, 3, 5, 6, 7]
                .into_iter()
                .map(|idx| {
                    (
                        idx,
                        exact_test_injection(crate::eeg::EegType::C, Bm::x(0), 0.3),
                    )
                })
                .collect(),
        );
        let actual = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 1).unwrap();
        assert!(actual.abs() < 1e-10);
    }

    #[test]
    fn test_exact_batched_supported_gates() {
        let gates = [
            gate(GateType::QAlloc, &[0, 1, 2, 3]),
            gate(GateType::H, &[1]),
            gate(GateType::PZ, &[0, 1]),
            gate(GateType::H, &[0, 1]),
            gate(GateType::CX, &[0, 2, 1, 3]),
            gate(GateType::MZ, &[0, 1, 2, 3]),
        ];
        // The reset removes the first H1. The batched H and CX then prepare
        // two Bell pairs, (0,2) and (1,3): each bit is fair, each pair even.
        let noise = ExactTestNoise(vec![]);
        for (detector, expected) in [
            (vec![0], 0.5),
            (vec![1], 0.5),
            (vec![2], 0.5),
            (vec![3], 0.5),
            (vec![0, 2], 0.0),
            (vec![1, 3], 0.0),
        ] {
            let actual = heisenberg_exact_from_circuit(&gates, &detector, &noise, 4).unwrap();
            assert!(
                (actual - expected).abs() < 1e-10,
                "detector={detector:?}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn test_exact_batched_cx_pairs_apply_in_order() {
        // CX [0,1,1,2] is CX(0,1) then CX(1,2): after H0 that makes a GHZ
        // state, so q2 is a fair coin. In the other order CX(1,2) acts on
        // |0> first and q2 stays 0.
        let gates = [
            gate(GateType::QAlloc, &[0, 1, 2]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1, 1, 2]),
            gate(GateType::MZ, &[0, 1, 2]),
        ];
        let noise = ExactTestNoise(vec![]);
        for (detector, expected) in [(vec![2], 0.5), (vec![0, 2], 0.0), (vec![1, 2], 0.0)] {
            let actual = heisenberg_exact_from_circuit(&gates, &detector, &noise, 3).unwrap();
            assert!(
                (actual - expected).abs() < 1e-10,
                "detector={detector:?}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn test_exact_identity_gates_carry_noise() {
        // I and Idle act trivially but still carry noise: an X0 flip with
        // probability p after either one flips the measurement with p.
        let p = 0.2;
        for idle in [GateType::I, GateType::Idle] {
            let gates = [
                gate(GateType::PZ, &[0]),
                gate(idle, &[0]),
                gate(GateType::MZ, &[0]),
            ];
            let noise = ExactTestNoise(vec![(
                1,
                exact_test_injection(crate::eeg::EegType::S, Bm::x(0), -p),
            )]);
            let actual = heisenberg_exact_from_circuit(&gates, &[0], &noise, 1).unwrap();
            assert!((actual - p).abs() < 1e-12, "{idle:?}: {actual} != {p}");
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
            let p = heisenberg_exact_from_circuit(&gates, &[0, 1], &noise, 3)
                .expect("supported circuit");
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

        let expanded = crate::expand::expand_circuit(&gates_orig).expect("supported circuit");

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
            let gate_index = crate::expand::GateIndex::build(
                &expanded.gates,
                expanded.num_qubits,
                noise,
                &expanded.expansion_gates,
            );
            let noise_map = build_noise_map(&expanded.gates, noise, &expanded.expansion_gates);

            // Test all 3 detectors (auxiliary qubits in round 1: meas 0,1,2)
            for meas_idx in 0..3 {
                let aux_q = expanded.measurement_qubit[meas_idx];
                let det = Bm::z(aux_q);

                // Windowed (old path)
                let start = Instant::now();
                let p_windowed = heisenberg_windowed(
                    &expanded.gates,
                    &det,
                    noise,
                    &stab,
                    1e-12,
                    None,
                    &expanded.expansion_gates,
                );
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

            let expanded = crate::expand::expand_circuit(&gates).expect("supported circuit");
            let gate_index = crate::expand::GateIndex::build(
                &expanded.gates,
                expanded.num_qubits,
                &noise,
                &expanded.expansion_gates,
            );
            let noise_map = build_noise_map(&expanded.gates, &noise, &expanded.expansion_gates);

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

            let expanded = crate::expand::expand_circuit(&gates).expect("supported circuit");
            let gate_index = crate::expand::GateIndex::build(
                &expanded.gates,
                expanded.num_qubits,
                &noise,
                &expanded.expansion_gates,
            );
            let noise_map = build_noise_map(&expanded.gates, &noise, &expanded.expansion_gates);

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
