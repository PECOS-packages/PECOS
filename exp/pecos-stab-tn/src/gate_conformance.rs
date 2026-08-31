// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Central, tiered conformance roster for every PECOS gate simulator.
//!
//! The phase-exact tier is for representations that carry amplitudes. It compares
//! a gate applied to a fixed nontrivial state with the corresponding
//! [`GateType::canonical_1q_matrix`] or [`Clifford::canonical_1q_matrix`], including global
//! phase. CPU f64 state vectors use `1e-12`, f32 state vectors use `2e-5`, GPU f64 uses
//! `1e-10`, and stabilizer/MPS amplitude reconstruction uses `1e-10`. The f64 bounds are
//! well above machine epsilon for these short circuits; the f32 bound allows a small number
//! of shader/SIMD operations without accepting a gate-scale error. The MPS bound accounts for
//! dense reconstruction and normalization without making global-phase differences invisible.
//! Gate coverage comes from [`Clifford::all_1q`] and
//! [`pecos_core::gate_type::NAMED_SINGLE_QUBIT_GATES`]; registration selects a trait bound, never
//! a runtime gate-name filter. A new oracle entry therefore becomes a test case automatically.
//!
//! The projective tier is for representations that provably cannot retain a quantum-state
//! global phase. It compares `G P G^dagger`, for `P` in `{X, Y, Z}`, with conjugation computed
//! from the same canonical matrix oracle, including the Pauli sign. Density matrices belong in
//! this tier permanently: their evolution is `rho -> U rho U^dagger`, so a scalar phase on `U`
//! cancels identically and cannot be observed by construction. Numeric density comparisons use
//! `1e-10` for f64 and `2e-5` for f32.
//!
//! `CircuitRunner`, `CoinToss`, `CustomScoreCriterion`, `GateSystemRegistry`, and
//! `ThresholdCriterion` are orchestration, sampling, scoring, or registry utilities rather than
//! quantum-state/Pauli representations. `ForeignSimulator` is a delegator whose conformance is
//! the responsibility of the concrete simulator behind its vtable. They are intentionally absent
//! from the roster below.

#![allow(clippy::missing_panics_doc)]

use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

use num_complex::Complex64;
use pecos_core::gate_type::{GateType, NAMED_SINGLE_QUBIT_GATES, SingleQubitGateMatrix};
use pecos_core::{Clifford, QubitId};
use pecos_cppsparsestab::CppSparseStab;
use pecos_gpu_sims::{
    DefaultGpuStab, GpuDensityMatrix32, GpuDensityMatrix64, GpuDensityMatrixHostAccess,
    GpuStateVec32, GpuStateVec64, GpuStateVecAuto, GpuStateVecBackend, GpuStateVectorHostAccess,
};
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, BitmaskPauliProp, CHFormGeneric, CliffordGateable, DenseStab,
    DenseStabColOnly, DenseStabRowOnly, DensityMatrix, DensityMatrixAccess, GpuStab, GpuStabOpt,
    GpuStabParallel, GraphStateSim, PauliProp, SparseColOnly, SparseRowOnly, SparseStab,
    SparseStabGeneric, SparseStabHybrid, SparseStabY, SparseStabYGeneric, SparseStateVecAoS,
    SparseStateVecSoA, StabVecGeneric, Stabilizer, StateVecAoS, StateVecSoA, StateVecSoA32,
    StateVectorAccess,
};

use crate::stab_mps::StabMps;
use crate::stab_mps::compile::StabMpsCompile;
use crate::stab_mps::mast::Mast;

const F64_TOLERANCE: f64 = 1e-12;
const F32_TOLERANCE: f64 = 2e-5;
const GPU_F64_TOLERANCE: f64 = 1e-10;
const RECONSTRUCTED_TOLERANCE: f64 = 1e-10;
const PROJECTIVE_F64_TOLERANCE: f64 = 1e-10;
const PROJECTIVE_F32_TOLERANCE: f64 = 2e-5;

#[derive(Debug)]
struct Failure {
    simulator: String,
    gate: String,
    tier: &'static str,
    expected: String,
    actual: String,
    algebra: String,
}

#[derive(Debug)]
struct Skip {
    simulator: String,
    tier: &'static str,
    reason: String,
}

#[derive(Default)]
struct Report {
    rostered: usize,
    checks: usize,
    passes: usize,
    failures: Vec<Failure>,
    skips: Vec<Skip>,
}

impl Report {
    fn passed(&mut self) {
        self.checks += 1;
        self.passes += 1;
    }

    fn failed(&mut self, failure: Failure) {
        self.checks += 1;
        self.failures.push(failure);
    }

    fn skip(&mut self, simulator: &str, tier: &'static str, reason: impl Into<String>) {
        self.skips.push(Skip {
            simulator: simulator.to_string(),
            tier,
            reason: reason.into(),
        });
    }

    fn print(&self) {
        for failure in &self.failures {
            eprintln!(
                "CONFORMANCE_FAILURE\nsimulator: {}\ngate: {}\ntier: {}\nexpected: {}\nactual: {}\nalgebra: {}\nEND_CONFORMANCE_FAILURE",
                failure.simulator,
                failure.gate,
                failure.tier,
                failure.expected,
                failure.actual,
                failure.algebra,
            );
        }
        for skip in &self.skips {
            eprintln!(
                "CONFORMANCE_SKIP simulator={} tier={} reason={}",
                skip.simulator, skip.tier, skip.reason
            );
        }
        eprintln!(
            "CONFORMANCE_TALLY rostered={} checks={} passed={} failed={} skipped_simulators={}",
            self.rostered,
            self.checks,
            self.passes,
            self.failures.len(),
            self.skips.len(),
        );
    }
}

trait Construct: Sized {
    fn construct(num_qubits: usize) -> Result<Self, String>;
}

trait PhaseSnapshot: Construct + CliffordGateable {
    const TOLERANCE: f64;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String>;
}

trait DensitySnapshot: Construct + CliffordGateable {
    const TOLERANCE: f64;

    fn construct_pauli_input(input: PauliAxis) -> Result<Self, String>;

    fn density_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, String>;
}

macro_rules! impl_construct_infallible {
    ($($ty:ty => $constructor:path),+ $(,)?) => {
        $(
            impl Construct for $ty {
                fn construct(num_qubits: usize) -> Result<Self, String> {
                    Ok($constructor(num_qubits))
                }
            }
        )+
    };
}

impl_construct_infallible!(
    StateVecSoA => StateVecSoA::new,
    StateVecSoA32 => StateVecSoA32::new,
    StateVecAoS => StateVecAoS::new,
    SparseStateVecAoS => SparseStateVecAoS::new,
    SparseStateVecSoA => SparseStateVecSoA::new,
    StabVecGeneric => StabVecGeneric::new,
    CHFormGeneric => CHFormGeneric::new,
    StabMpsCompile => StabMpsCompile::new,
    SparseStabGeneric => SparseStab::new,
    SparseStabYGeneric => SparseStabY::new,
    SparseStabHybrid => SparseStabHybrid::new,
    SparseColOnly => SparseColOnly::new,
    SparseRowOnly => SparseRowOnly::new,
    DenseStab => DenseStab::new,
    DenseStabColOnly => DenseStabColOnly::new,
    DenseStabRowOnly => DenseStabRowOnly::new,
    GpuStab => GpuStab::new,
    GpuStabOpt => GpuStabOpt::new,
    GpuStabParallel => GpuStabParallel::new,
    CppSparseStab => CppSparseStab::new,
    GraphStateSim => GraphStateSim::new,
    Stabilizer => Stabilizer::new,
    DensityMatrix => DensityMatrix::new,
);

impl Construct for StabMps {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Ok(Self::builder(num_qubits).merge_rz(false).build())
    }
}

impl Construct for Mast {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Ok(Self::new(num_qubits, 2))
    }
}

impl Construct for GpuStateVec32 {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits as u32).map_err(|error| error.to_string())
    }
}

impl Construct for GpuStateVec64 {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits as u32).map_err(|error| error.to_string())
    }
}

impl Construct for GpuStateVecAuto {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits as u32).map_err(|error| error.to_string())
    }
}

impl Construct for DefaultGpuStab {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits).map_err(|error| error.clone())
    }
}

impl Construct for GpuDensityMatrix32 {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits).map_err(|error| error.to_string())
    }
}

impl Construct for GpuDensityMatrix64 {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Self::new(num_qubits).map_err(|error| error.to_string())
    }
}

impl Construct for PauliProp {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Ok(Self::with_sign_tracking(num_qubits))
    }
}

impl Construct for BitmaskPauliProp {
    fn construct(num_qubits: usize) -> Result<Self, String> {
        Ok(Self::with_num_qubits(num_qubits))
    }
}

macro_rules! impl_cpu_phase_snapshot {
    ($($ty:ty => $tolerance:expr),+ $(,)?) => {
        $(
            impl PhaseSnapshot for $ty {
                const TOLERANCE: f64 = $tolerance;

                fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
                    StateVectorAccess::state_vector(self).map_err(|error| error.to_string())
                }
            }
        )+
    };
}

impl_cpu_phase_snapshot!(
    StateVecSoA => F64_TOLERANCE,
    StateVecSoA32 => F32_TOLERANCE,
    StateVecAoS => F64_TOLERANCE,
    SparseStateVecAoS => F64_TOLERANCE,
    SparseStateVecSoA => F64_TOLERANCE,
);

impl PhaseSnapshot for StabVecGeneric {
    const TOLERANCE: f64 = RECONSTRUCTED_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        Ok(self.state_vector())
    }
}

impl PhaseSnapshot for CHFormGeneric {
    const TOLERANCE: f64 = RECONSTRUCTED_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        Ok(self.state_vector())
    }
}

impl PhaseSnapshot for StabMps {
    const TOLERANCE: f64 = RECONSTRUCTED_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        Ok(self.state_vector())
    }
}

impl PhaseSnapshot for StabMpsCompile {
    const TOLERANCE: f64 = RECONSTRUCTED_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        Ok(self.conformance_state_vector())
    }
}

impl PhaseSnapshot for Mast {
    const TOLERANCE: f64 = RECONSTRUCTED_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        Ok(self.conformance_state_vector())
    }
}

impl PhaseSnapshot for GpuStateVec32 {
    const TOLERANCE: f64 = F32_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        self.state_vector_host_snapshot()
            .map_err(|error| error.to_string())
    }
}

impl PhaseSnapshot for GpuStateVec64 {
    const TOLERANCE: f64 = GPU_F64_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        self.state_vector_host_snapshot()
            .map_err(|error| error.to_string())
    }
}

impl PhaseSnapshot for GpuStateVecAuto {
    const TOLERANCE: f64 = F32_TOLERANCE;

    fn snapshot(&mut self) -> Result<Vec<Complex64>, String> {
        match self {
            Self::F64(simulator) => simulator
                .state_vector_host_snapshot()
                .map_err(|error| error.to_string()),
            Self::F32(simulator) => simulator
                .state_vector_host_snapshot()
                .map_err(|error| error.to_string()),
        }
    }
}

impl DensitySnapshot for DensityMatrix {
    const TOLERANCE: f64 = PROJECTIVE_F64_TOLERANCE;

    fn construct_pauli_input(input: PauliAxis) -> Result<Self, String> {
        let mut simulator = Self::construct(1)?;
        *simulator.state_vector_mut() = StateVecSoA::from_state(
            &pauli_eigenstate_choi(input),
            PecosRng::seed_from_u64(0x5eed),
        );
        Ok(simulator)
    }

    fn density_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, String> {
        DensityMatrixAccess::density_matrix(self).map_err(|error| error.to_string())
    }
}

impl DensitySnapshot for GpuDensityMatrix32 {
    const TOLERANCE: f64 = PROJECTIVE_F32_TOLERANCE;

    fn construct_pauli_input(input: PauliAxis) -> Result<Self, String> {
        let mut simulator = Self::construct(1)?;
        simulator
            .state_vector_mut()
            .write_state_f64(&complex_state_to_pairs(&pauli_eigenstate_choi(input)));
        Ok(simulator)
    }

    fn density_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, String> {
        self.density_matrix_host_snapshot()
            .map_err(|error| error.to_string())
    }
}

impl DensitySnapshot for GpuDensityMatrix64 {
    const TOLERANCE: f64 = PROJECTIVE_F64_TOLERANCE;

    fn construct_pauli_input(input: PauliAxis) -> Result<Self, String> {
        let mut simulator = Self::construct(1)?;
        simulator
            .state_vector_mut()
            .write_state_f64(&complex_state_to_pairs(&pauli_eigenstate_choi(input)));
        Ok(simulator)
    }

    fn density_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, String> {
        self.density_matrix_host_snapshot()
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy)]
enum GateSpec {
    Clifford(Clifford),
    Named(GateType),
}

impl GateSpec {
    fn label(self) -> String {
        match self {
            Self::Clifford(gate) => gate.to_string(),
            Self::Named(gate) => format!("{gate:?}"),
        }
    }

    fn matrix(self) -> SingleQubitGateMatrix {
        match self {
            Self::Clifford(gate) => gate
                .canonical_1q_matrix()
                .expect("registered Clifford must be single-qubit"),
            Self::Named(gate) => gate
                .canonical_1q_matrix()
                .expect("registered named gate must be single-qubit"),
        }
    }
}

fn non_clifford_named_gates() -> impl Iterator<Item = GateType> {
    NAMED_SINGLE_QUBIT_GATES.into_iter().filter(|gate| {
        !Clifford::all_1q()
            .iter()
            .filter_map(|clifford| clifford.to_gate_type())
            .any(|clifford_gate| clifford_gate == *gate)
    })
}

fn apply_clifford<S: CliffordGateable>(simulator: &mut S, gate: Clifford, qubits: &[QubitId]) {
    match gate {
        Clifford::I => simulator.identity(qubits),
        Clifford::X => simulator.x(qubits),
        Clifford::Y => simulator.y(qubits),
        Clifford::Z => simulator.z(qubits),
        Clifford::H => simulator.h(qubits),
        Clifford::H2 => simulator.h2(qubits),
        Clifford::H3 => simulator.h3(qubits),
        Clifford::H4 => simulator.h4(qubits),
        Clifford::H5 => simulator.h5(qubits),
        Clifford::H6 => simulator.h6(qubits),
        Clifford::SX => simulator.sx(qubits),
        Clifford::SXdg => simulator.sxdg(qubits),
        Clifford::SY => simulator.sy(qubits),
        Clifford::SYdg => simulator.sydg(qubits),
        Clifford::SZ => simulator.sz(qubits),
        Clifford::SZdg => simulator.szdg(qubits),
        Clifford::F => simulator.f(qubits),
        Clifford::Fdg => simulator.fdg(qubits),
        Clifford::F2 => simulator.f2(qubits),
        Clifford::F2dg => simulator.f2dg(qubits),
        Clifford::F3 => simulator.f3(qubits),
        Clifford::F3dg => simulator.f3dg(qubits),
        Clifford::F4 => simulator.f4(qubits),
        Clifford::F4dg => simulator.f4dg(qubits),
        _ => panic!("two-qubit Clifford in single-qubit conformance sweep"),
    };
}

fn apply_gate<S: ArbitraryRotationGateable>(simulator: &mut S, gate: GateSpec, qubits: &[QubitId]) {
    match gate {
        GateSpec::Clifford(gate) => apply_clifford(simulator, gate, qubits),
        GateSpec::Named(GateType::T) => {
            simulator.t(qubits);
        }
        GateSpec::Named(GateType::Tdg) => {
            simulator.tdg(qubits);
        }
        GateSpec::Named(gate) => panic!("unsupported non-Clifford named gate {gate:?}"),
    }
}

fn prepare_nontrivial<S: CliffordGateable>(simulator: &mut S) {
    simulator
        .h(&[QubitId(0), QubitId(2)])
        .sz(&[QubitId(0)])
        .h(&[QubitId(1)])
        .cx(&[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))])
        .cz(&[(QubitId(1), QubitId(2))]);
}

fn apply_matrix_to_state(state: &mut [Complex64], matrix: SingleQubitGateMatrix, target: QubitId) {
    let matrix = unpack_matrix(matrix);
    let step = 1_usize << target.index();
    for block in state.chunks_exact_mut(step * 2) {
        let (zero, one) = block.split_at_mut(step);
        for (zero, one) in zero.iter_mut().zip(one) {
            let input_zero = *zero;
            let input_one = *one;
            *zero = matrix[0] * input_zero + matrix[1] * input_one;
            *one = matrix[2] * input_zero + matrix[3] * input_one;
        }
    }
}

fn max_state_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max)
}

fn phase_case<S>(gate: GateSpec, targets: &[QubitId]) -> Result<PhaseObservation, String>
where
    S: PhaseSnapshot + ArbitraryRotationGateable,
{
    let mut simulator = S::construct(4)?;
    prepare_nontrivial(&mut simulator);
    let input = simulator.snapshot()?;
    let mut expected = input.clone();
    for &target in targets {
        apply_matrix_to_state(&mut expected, gate.matrix(), target);
    }
    apply_gate(&mut simulator, gate, targets);
    let actual = simulator.snapshot()?;
    Ok(PhaseObservation {
        input,
        expected,
        actual,
    })
}

fn clifford_phase_case<S>(gate: Clifford, targets: &[QubitId]) -> Result<PhaseObservation, String>
where
    S: PhaseSnapshot,
{
    let mut simulator = S::construct(4)?;
    prepare_nontrivial(&mut simulator);
    let input = simulator.snapshot()?;
    let mut expected = input.clone();
    let matrix = gate
        .canonical_1q_matrix()
        .expect("registered Clifford must be single-qubit");
    for &target in targets {
        apply_matrix_to_state(&mut expected, matrix, target);
    }
    apply_clifford(&mut simulator, gate, targets);
    let actual = simulator.snapshot()?;
    Ok(PhaseObservation {
        input,
        expected,
        actual,
    })
}

struct PhaseObservation {
    input: Vec<Complex64>,
    expected: Vec<Complex64>,
    actual: Vec<Complex64>,
}

fn run_phase_exact_clifford<S>(report: &mut Report, simulator_name: &str)
where
    S: PhaseSnapshot,
{
    if let Some(reason) = construction_skip::<S>(4) {
        report.skip(simulator_name, "phase-exact", reason);
        return;
    }

    for &gate in Clifford::all_1q() {
        for targets in [&[QubitId(1)][..], &[QubitId(1), QubitId(0)][..]] {
            let context = if targets.len() == 1 {
                "single-target q1"
            } else {
                "multi-target slice [q1,q0]"
            };
            let label = format!("{gate} ({context})");
            let outcome =
                catch_unwind(AssertUnwindSafe(|| clifford_phase_case::<S>(gate, targets)));
            record_phase_outcome::<S>(
                report,
                simulator_name,
                &label,
                gate.into(),
                targets,
                outcome,
            );
        }
    }

    run_issue_613_probe::<S>(report, simulator_name);
}

fn run_phase_exact_arbitrary<S>(report: &mut Report, simulator_name: &str)
where
    S: PhaseSnapshot + ArbitraryRotationGateable,
{
    if let Some(reason) = construction_skip::<S>(4) {
        report.skip(simulator_name, "phase-exact", reason);
        return;
    }

    for &gate in Clifford::all_1q() {
        for targets in [&[QubitId(1)][..], &[QubitId(1), QubitId(0)][..]] {
            let context = if targets.len() == 1 {
                "single-target q1"
            } else {
                "multi-target slice [q1,q0]"
            };
            let label = format!("{gate} ({context})");
            let spec = GateSpec::Clifford(gate);
            let outcome = catch_unwind(AssertUnwindSafe(|| phase_case::<S>(spec, targets)));
            record_phase_outcome::<S>(report, simulator_name, &label, spec, targets, outcome);
        }
    }

    for gate in non_clifford_named_gates() {
        for targets in [&[QubitId(1)][..], &[QubitId(1), QubitId(0)][..]] {
            let context = if targets.len() == 1 {
                "single-target q1"
            } else {
                "multi-target slice [q1,q0]"
            };
            let label = format!("{gate:?} ({context})");
            let spec = GateSpec::Named(gate);
            let outcome = catch_unwind(AssertUnwindSafe(|| phase_case::<S>(spec, targets)));
            record_phase_outcome::<S>(report, simulator_name, &label, spec, targets, outcome);
        }
    }

    run_issue_613_probe::<S>(report, simulator_name);
}

fn construction_skip<S: Construct>(num_qubits: usize) -> Option<String> {
    match catch_unwind(AssertUnwindSafe(|| S::construct(num_qubits))) {
        Ok(Ok(_)) => None,
        Ok(Err(error)) => Some(error),
        Err(payload) => Some(format!(
            "constructor panicked: {}",
            panic_message(payload.as_ref())
        )),
    }
}

fn record_phase_outcome<S: PhaseSnapshot>(
    report: &mut Report,
    simulator_name: &str,
    label: &str,
    gate: GateSpec,
    targets: &[QubitId],
    outcome: std::thread::Result<Result<PhaseObservation, String>>,
) {
    match outcome {
        Ok(Ok(observation)) => {
            let error = max_state_error(&observation.actual, &observation.expected);
            if error <= S::TOLERANCE {
                report.passed();
                return;
            }
            report.failed(Failure {
                simulator: simulator_name.to_string(),
                gate: label.to_string(),
                tier: "phase-exact",
                expected: format!(
                    "matrix={} output={}",
                    format_matrix(gate.matrix()),
                    format_state(&observation.expected)
                ),
                actual: format!(
                    "max_error={error:.6e} output={}",
                    format_state(&observation.actual)
                ),
                algebra: simplify_state_action(
                    &observation.input,
                    &observation.expected,
                    &observation.actual,
                    targets,
                    S::TOLERANCE,
                ),
            });
        }
        Ok(Err(error)) => report.failed(Failure {
            simulator: simulator_name.to_string(),
            gate: label.to_string(),
            tier: "phase-exact",
            expected: format!("matrix={}", format_matrix(gate.matrix())),
            actual: format!("backend error: {error}"),
            algebra: "the backend did not produce a state to simplify".to_string(),
        }),
        Err(payload) => report.failed(Failure {
            simulator: simulator_name.to_string(),
            gate: label.to_string(),
            tier: "phase-exact",
            expected: format!("matrix={}", format_matrix(gate.matrix())),
            actual: format!("panic: {}", panic_message(payload.as_ref())),
            algebra: "the gate/read path panicked before an operator could be classified"
                .to_string(),
        }),
    }
}

impl From<Clifford> for GateSpec {
    fn from(value: Clifford) -> Self {
        Self::Clifford(value)
    }
}

fn run_issue_613_probe<S>(report: &mut Report, simulator_name: &str)
where
    S: PhaseSnapshot,
{
    let outcome = catch_unwind(AssertUnwindSafe(issue_613_case::<S>));
    match outcome {
        Ok(Ok((expected, actual))) => {
            let error = max_state_error(&actual, &expected);
            if error <= S::TOLERANCE {
                report.passed();
            } else {
                report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: "H5 multi-target slice in uninterrupted #613 circuit".to_string(),
                    tier: "phase-exact",
                    expected: format!(
                        "F3(q3); SXdg(q2); canonical H5([q1,q0]); CX(q3,q0) => {}",
                        format_state(&expected)
                    ),
                    actual: format!(
                        "max_error={error:.6e} uninterrupted output={}",
                        format_state(&actual)
                    ),
                    algebra: simplify_issue_613_output(&expected, &actual, S::TOLERANCE),
                });
            }
        }
        Ok(Err(error)) => report.failed(Failure {
            simulator: simulator_name.to_string(),
            gate: "H5 multi-target slice in uninterrupted #613 circuit".to_string(),
            tier: "phase-exact",
            expected: "the canonical #613 four-qubit state".to_string(),
            actual: format!("backend error: {error}"),
            algebra: "the backend did not produce a state to simplify".to_string(),
        }),
        Err(payload) => report.failed(Failure {
            simulator: simulator_name.to_string(),
            gate: "H5 multi-target slice in uninterrupted #613 circuit".to_string(),
            tier: "phase-exact",
            expected: "the canonical #613 four-qubit state".to_string(),
            actual: format!("panic: {}", panic_message(payload.as_ref())),
            algebra: "the uninterrupted probe panicked".to_string(),
        }),
    }
}

fn simplify_issue_613_output(
    expected: &[Complex64],
    actual: &[Complex64],
    tolerance: f64,
) -> String {
    let mut best = (f64::INFINITY, String::new(), Complex64::new(1.0, 0.0));
    update_best_fit(actual, expected, "the canonical #613 circuit", &mut best);
    if best.0 <= tolerance {
        return format!(
            "simplifies to ({:.9}{:+.9}i) * the canonical #613 circuit; this is a phase-only failure, distinct from #613's wrong sparse state",
            best.2.re, best.2.im
        );
    }

    let missing = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual.norm() <= tolerance && expected.norm() > tolerance)
        .count();
    format!(
        "does not simplify to a global phase times the canonical circuit (closest projective error {:.6e}); {missing} expected nonzero amplitudes simplify to zero, matching #613's non-unitary sparse-state truncation signature",
        best.0
    )
}

fn issue_613_case<S>() -> Result<(Vec<Complex64>, Vec<Complex64>), String>
where
    S: PhaseSnapshot,
{
    let mut simulator = S::construct(4)?;
    apply_clifford(&mut simulator, Clifford::F3, &[QubitId(3)]);
    apply_clifford(&mut simulator, Clifford::SXdg, &[QubitId(2)]);
    apply_clifford(&mut simulator, Clifford::H5, &[QubitId(1), QubitId(0)]);
    simulator.cx(&[(QubitId(3), QubitId(0))]);
    let actual = simulator.snapshot()?;

    let mut expected = vec![Complex64::new(0.0, 0.0); 16];
    expected[0] = Complex64::new(1.0, 0.0);
    for (gate, targets) in [
        (Clifford::F3, &[QubitId(3)][..]),
        (Clifford::SXdg, &[QubitId(2)][..]),
        (Clifford::H5, &[QubitId(1), QubitId(0)][..]),
    ] {
        let matrix = gate
            .canonical_1q_matrix()
            .expect("#613 probe uses single-qubit Cliffords");
        for &target in targets {
            apply_matrix_to_state(&mut expected, matrix, target);
        }
    }
    apply_cx_permutation(&mut expected, QubitId(3), QubitId(0));
    Ok((expected, actual))
}

fn apply_cx_permutation(state: &mut [Complex64], control: QubitId, target: QubitId) {
    let control_mask = 1_usize << control.index();
    let target_mask = 1_usize << target.index();
    for basis in 0..state.len() {
        if basis & control_mask != 0 && basis & target_mask == 0 {
            state.swap(basis, basis | target_mask);
        }
    }
}

fn simplify_state_action(
    input: &[Complex64],
    expected: &[Complex64],
    actual: &[Complex64],
    targets: &[QubitId],
    tolerance: f64,
) -> String {
    let mut intended = (f64::INFINITY, String::new(), Complex64::new(1.0, 0.0));
    update_best_fit(
        actual,
        expected,
        "the intended canonical action",
        &mut intended,
    );
    if intended.0 <= tolerance {
        return format!(
            "simplifies to ({:.9}{:+.9}i) * the intended canonical action",
            intended.2.re, intended.2.im
        );
    }

    let candidates: Vec<(String, SingleQubitGateMatrix)> = Clifford::all_1q()
        .iter()
        .map(|&gate| {
            (
                gate.to_string(),
                gate.canonical_1q_matrix()
                    .expect("candidate Clifford must be single-qubit"),
            )
        })
        .chain(non_clifford_named_gates().map(|gate| {
            (
                format!("{gate:?}"),
                gate.canonical_1q_matrix()
                    .expect("candidate named gate must be single-qubit"),
            )
        }))
        .collect();

    let mut best = (f64::INFINITY, String::new(), Complex64::new(1.0, 0.0));
    if targets.len() == 1 {
        for (label, matrix) in &candidates {
            let mut candidate = input.to_vec();
            apply_matrix_to_state(&mut candidate, *matrix, targets[0]);
            update_best_fit(actual, &candidate, label, &mut best);
        }
    } else if targets.len() == 2 {
        for (left_label, left_matrix) in &candidates {
            for (right_label, right_matrix) in &candidates {
                let mut candidate = input.to_vec();
                apply_matrix_to_state(&mut candidate, *left_matrix, targets[0]);
                apply_matrix_to_state(&mut candidate, *right_matrix, targets[1]);
                update_best_fit(
                    actual,
                    &candidate,
                    &format!(
                        "{left_label}(q{}) * {right_label}(q{})",
                        targets[0].index(),
                        targets[1].index()
                    ),
                    &mut best,
                );
            }
        }
    }

    if best.0 <= tolerance {
        if (best.2 - Complex64::new(1.0, 0.0)).norm() <= tolerance {
            format!("simplifies to {}", best.1)
        } else {
            format!(
                "simplifies to ({:.9}{:+.9}i) * {}",
                best.2.re, best.2.im, best.1
            )
        }
    } else {
        format!(
            "does not simplify to a tensor product of canonical named gates; closest projective fit is ({:.9}{:+.9}i) * {} with max error {:.6e}",
            best.2.re, best.2.im, best.1, best.0
        )
    }
}

fn update_best_fit(
    actual: &[Complex64],
    candidate: &[Complex64],
    label: &str,
    best: &mut (f64, String, Complex64),
) {
    let overlap = candidate
        .iter()
        .zip(actual)
        .map(|(candidate, actual)| candidate.conj() * actual)
        .sum::<Complex64>();
    let phase = if overlap.norm() > 1e-15 {
        overlap / overlap.norm()
    } else {
        Complex64::new(1.0, 0.0)
    };
    let error = actual
        .iter()
        .zip(candidate)
        .map(|(actual, candidate)| (*actual - phase * candidate).norm())
        .fold(0.0, f64::max);
    if error < best.0 {
        *best = (error, label.to_string(), phase);
    }
}

fn format_state(state: &[Complex64]) -> String {
    let mut output = String::from("[");
    for (index, amplitude) in state.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{:.8}{:+.8}i", amplitude.re, amplitude.im).unwrap();
    }
    output.push(']');
    output
}

fn format_matrix(matrix: SingleQubitGateMatrix) -> String {
    let matrix = unpack_matrix(matrix);
    format!(
        "[[{:.8}{:+.8}i, {:.8}{:+.8}i], [{:.8}{:+.8}i, {:.8}{:+.8}i]]",
        matrix[0].re,
        matrix[0].im,
        matrix[1].re,
        matrix[1].im,
        matrix[2].re,
        matrix[2].im,
        matrix[3].re,
        matrix[3].im,
    )
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PauliAxis {
    X,
    Y,
    Z,
}

impl PauliAxis {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    const fn gate_type(self) -> GateType {
        match self {
            Self::X => GateType::X,
            Self::Y => GateType::Y,
            Self::Z => GateType::Z,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SignedPauli {
    negative: bool,
    axis: PauliAxis,
}

fn prepare_plus_pauli<S: CliffordGateable>(simulator: &mut S, input: PauliAxis) {
    match input {
        PauliAxis::X => {
            simulator.h(&[QubitId(0)]);
        }
        PauliAxis::Y => {
            simulator.h(&[QubitId(0)]).sz(&[QubitId(0)]);
        }
        PauliAxis::Z => {}
    }
}

fn measure_pauli<S: CliffordGateable>(
    simulator: &mut S,
    axis: PauliAxis,
) -> pecos_simulators::MeasurementResult {
    let mut results = match axis {
        PauliAxis::X => simulator.mx(&[QubitId(0)]),
        PauliAxis::Y => simulator.my(&[QubitId(0)]),
        PauliAxis::Z => simulator.mz(&[QubitId(0)]),
    };
    results
        .pop()
        .expect("one-qubit measurement must return one result")
}

fn observed_state_conjugation<S>(
    gate: Clifford,
    input: PauliAxis,
) -> Result<(String, Option<SignedPauli>), String>
where
    S: Construct + CliffordGateable,
{
    let mut observations = Vec::with_capacity(3);
    for output in PauliAxis::ALL {
        let mut simulator = S::construct(1)?;
        prepare_plus_pauli(&mut simulator, input);
        apply_clifford(&mut simulator, gate, &[QubitId(0)]);
        let measurement = measure_pauli(&mut simulator, output);
        observations.push((output, measurement.outcome, measurement.is_deterministic));
    }
    let deterministic: Vec<_> = observations
        .iter()
        .filter(|(_, _, is_deterministic)| *is_deterministic)
        .collect();
    let actual = if deterministic.len() == 1 {
        let (axis, outcome, _) = *deterministic[0];
        Some(SignedPauli {
            negative: outcome,
            axis,
        })
    } else {
        None
    };
    Ok((format!("measurements={observations:?}"), actual))
}

fn observed_cpp_conjugation(
    gate: Clifford,
    input: PauliAxis,
) -> Result<(String, Option<SignedPauli>), String> {
    let mut simulator = CppSparseStab::construct(1)?;
    prepare_plus_pauli(&mut simulator, input);
    apply_clifford(&mut simulator, gate, &[QubitId(0)]);
    let tableau = simulator.stab_tableau();
    let actual = match tableau.trim() {
        "+X" => Some(SignedPauli {
            negative: false,
            axis: PauliAxis::X,
        }),
        "-X" => Some(SignedPauli {
            negative: true,
            axis: PauliAxis::X,
        }),
        // The C++ sparse tableau stores a Y generator as iXZ and renders that
        // internal phase explicitly. Normalize the storage convention here;
        // the physical Hermitian generators are still signed Y Paulis.
        "+Y" | "iY" => Some(SignedPauli {
            negative: false,
            axis: PauliAxis::Y,
        }),
        "-Y" | "-iY" => Some(SignedPauli {
            negative: true,
            axis: PauliAxis::Y,
        }),
        "+Z" => Some(SignedPauli {
            negative: false,
            axis: PauliAxis::Z,
        }),
        "-Z" => Some(SignedPauli {
            negative: true,
            axis: PauliAxis::Z,
        }),
        _ => None,
    };
    Ok((format!("stabilizer tableau={tableau:?}"), actual))
}

fn expected_signed_pauli(gate: GateSpec, input: PauliAxis) -> (Matrix2, SignedPauli) {
    let conjugation = conjugate_matrix(gate.matrix(), input);
    let signed = classify_signed_pauli(&conjugation, 1e-10)
        .expect("a Clifford canonical matrix must conjugate a Pauli to a signed Pauli");
    (conjugation, signed)
}

fn run_projective_state<S>(report: &mut Report, simulator_name: &str)
where
    S: Construct + CliffordGateable,
{
    if let Some(reason) = construction_skip::<S>(1) {
        report.skip(simulator_name, "projective", reason);
        return;
    }
    for &gate in Clifford::all_1q() {
        for input in PauliAxis::ALL {
            let (expected_matrix, expected) = expected_signed_pauli(gate.into(), input);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                observed_state_conjugation::<S>(gate, input)
            }));
            match outcome {
                Ok(Ok((_detail, Some(actual)))) if actual == expected => report.passed(),
                Ok(Ok((detail, actual))) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format!(
                        "{} -> {}{}; matrix={}",
                        input.label(),
                        if expected.negative { "-" } else { "+" },
                        expected.axis.label(),
                        format_matrix2(&expected_matrix),
                    ),
                    actual: match actual {
                        Some(actual) => format!(
                            "{}{}; {detail}",
                            if actual.negative { "-" } else { "+" },
                            actual.axis.label(),
                        ),
                        None => format!("not a uniquely signed Pauli; {detail}"),
                    },
                    algebra: actual.map_or_else(
                        || "the measured action does not simplify to one signed Pauli".to_string(),
                        |actual| {
                            format!(
                                "simplifies to {}{}",
                                if actual.negative { "-" } else { "+" },
                                actual.axis.label()
                            )
                        },
                    ),
                }),
                Ok(Err(error)) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected_matrix),
                    actual: format!("backend error: {error}"),
                    algebra: "the backend did not produce a conjugation action".to_string(),
                }),
                Err(payload) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected_matrix),
                    actual: format!("panic: {}", panic_message(payload.as_ref())),
                    algebra: "the gate/measurement path panicked before classification".to_string(),
                }),
            }
        }
    }
}

fn run_projective_cpp(report: &mut Report, simulator_name: &str) {
    if let Some(reason) = construction_skip::<CppSparseStab>(1) {
        report.skip(simulator_name, "projective", reason);
        return;
    }
    for &gate in Clifford::all_1q() {
        for input in PauliAxis::ALL {
            let (expected_matrix, expected) = expected_signed_pauli(gate.into(), input);
            let outcome = catch_unwind(AssertUnwindSafe(|| observed_cpp_conjugation(gate, input)));
            match outcome {
                Ok(Ok((_detail, Some(actual)))) if actual == expected => report.passed(),
                Ok(Ok((detail, actual))) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format!(
                        "{} -> {}{}; matrix={}",
                        input.label(),
                        if expected.negative { "-" } else { "+" },
                        expected.axis.label(),
                        format_matrix2(&expected_matrix),
                    ),
                    actual: match actual {
                        Some(actual) => format!(
                            "{}{}; {detail}",
                            if actual.negative { "-" } else { "+" },
                            actual.axis.label(),
                        ),
                        None => format!("not a uniquely signed Pauli; {detail}"),
                    },
                    algebra: actual.map_or_else(
                        || "the tableau does not simplify to one signed Pauli".to_string(),
                        |actual| {
                            format!(
                                "simplifies to {}{}",
                                if actual.negative { "-" } else { "+" },
                                actual.axis.label()
                            )
                        },
                    ),
                }),
                Ok(Err(error)) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected_matrix),
                    actual: format!("backend error: {error}"),
                    algebra: "the backend did not produce a conjugation action".to_string(),
                }),
                Err(payload) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected_matrix),
                    actual: format!("panic: {}", panic_message(payload.as_ref())),
                    algebra: "the gate/tableau path panicked before classification".to_string(),
                }),
            }
        }
    }
}

fn track_pauli(simulator: &mut PauliProp, input: PauliAxis) {
    match input {
        PauliAxis::X => simulator.track_x(&[0]),
        PauliAxis::Y => simulator.track_y(&[0]),
        PauliAxis::Z => simulator.track_z(&[0]),
    }
}

fn track_bitmask_pauli(simulator: &mut BitmaskPauliProp, input: PauliAxis) {
    match input {
        PauliAxis::X => simulator.track_x(&[0]),
        PauliAxis::Y => simulator.track_y(&[0]),
        PauliAxis::Z => simulator.track_z(&[0]),
    }
}

fn pauli_prop_axis(simulator: &PauliProp) -> Option<PauliAxis> {
    match (simulator.contains_x(0), simulator.contains_z(0)) {
        (true, false) => Some(PauliAxis::X),
        (true, true) => Some(PauliAxis::Y),
        (false, true) => Some(PauliAxis::Z),
        (false, false) => None,
    }
}

fn bitmask_axis(simulator: &BitmaskPauliProp) -> Option<PauliAxis> {
    match (simulator.contains_x(0), simulator.contains_z(0)) {
        (true, false) => Some(PauliAxis::X),
        (true, true) => Some(PauliAxis::Y),
        (false, true) => Some(PauliAxis::Z),
        (false, false) => None,
    }
}

fn run_pauli_prop(report: &mut Report, simulator_name: &str) {
    for &gate in Clifford::all_1q() {
        for input in PauliAxis::ALL {
            let (expected_matrix, expected) = expected_signed_pauli(gate.into(), input);
            let mut simulator = PauliProp::with_sign_tracking(1);
            track_pauli(&mut simulator, input);
            apply_clifford(&mut simulator, gate, &[QubitId(0)]);
            let actual = pauli_prop_axis(&simulator).map(|axis| SignedPauli {
                negative: simulator.get_sign(),
                axis,
            });
            if actual == Some(expected) {
                report.passed();
            } else {
                report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{gate}: {} conjugation", input.label()),
                    tier: "projective",
                    expected: format!(
                        "{}{}; matrix={}",
                        if expected.negative { "-" } else { "+" },
                        expected.axis.label(),
                        format_matrix2(&expected_matrix),
                    ),
                    actual: actual.map_or_else(
                        || "identity/invalid Pauli".to_string(),
                        |actual| {
                            format!(
                                "{}{}",
                                if actual.negative { "-" } else { "+" },
                                actual.axis.label()
                            )
                        },
                    ),
                    algebra: actual.map_or_else(
                        || "simplifies to the identity or an invalid label".to_string(),
                        |actual| {
                            format!(
                                "simplifies to {}{}",
                                if actual.negative { "-" } else { "+" },
                                actual.axis.label()
                            )
                        },
                    ),
                });
            }
        }
    }
}

fn run_bitmask_pauli_prop(report: &mut Report, simulator_name: &str) {
    for &gate in Clifford::all_1q() {
        for input in PauliAxis::ALL {
            let (expected_matrix, expected) = expected_signed_pauli(gate.into(), input);
            let mut simulator = BitmaskPauliProp::with_num_qubits(1);
            track_bitmask_pauli(&mut simulator, input);
            apply_clifford(&mut simulator, gate, &[QubitId(0)]);
            let actual_axis = bitmask_axis(&simulator);
            report.failed(Failure {
                simulator: simulator_name.to_string(),
                gate: format!("{gate}: {} conjugation", input.label()),
                tier: "projective",
                expected: format!(
                    "{}{}; matrix={}",
                    if expected.negative { "-" } else { "+" },
                    expected.axis.label(),
                    format_matrix2(&expected_matrix),
                ),
                actual: actual_axis.map_or_else(
                    || "identity/invalid Pauli with sign discarded".to_string(),
                    |axis| format!("±{} (sign is not represented)", axis.label()),
                ),
                algebra: actual_axis.map_or_else(
                    || "simplifies to the identity label with no recoverable sign".to_string(),
                    |axis| {
                        format!(
                            "simplifies only to the projective Pauli class ±{}; the required sign is discarded",
                            axis.label()
                        )
                    },
                ),
            });
        }
    }
}

fn density_conjugation<S>(gate: GateSpec, input: PauliAxis) -> Result<Matrix2, String>
where
    S: DensitySnapshot + ArbitraryRotationGateable,
{
    let mut simulator = S::construct_pauli_input(input)?;
    apply_gate(&mut simulator, gate, &[QubitId(0)]);
    let density = simulator.density_snapshot()?;
    if density.len() != 2 || density.iter().any(|row| row.len() != 2) {
        return Err(format!("unexpected density shape: {density:?}"));
    }
    let identity = unpack_matrix(
        GateType::I
            .canonical_1q_matrix()
            .expect("identity must have a canonical matrix"),
    );
    Ok([
        density[0][0] * 2.0 - identity[0],
        density[0][1] * 2.0 - identity[1],
        density[1][0] * 2.0 - identity[2],
        density[1][1] * 2.0 - identity[3],
    ])
}

fn pauli_eigenstate_choi(input: PauliAxis) -> Vec<Complex64> {
    let mut eigenstate = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    match input {
        PauliAxis::X => apply_matrix_to_state(
            &mut eigenstate,
            Clifford::H
                .canonical_1q_matrix()
                .expect("H must have a canonical matrix"),
            QubitId(0),
        ),
        PauliAxis::Y => {
            apply_matrix_to_state(
                &mut eigenstate,
                Clifford::H
                    .canonical_1q_matrix()
                    .expect("H must have a canonical matrix"),
                QubitId(0),
            );
            apply_matrix_to_state(
                &mut eigenstate,
                Clifford::SZ
                    .canonical_1q_matrix()
                    .expect("SZ must have a canonical matrix"),
                QubitId(0),
            );
        }
        PauliAxis::Z => {}
    }
    vec![
        eigenstate[0],
        Complex64::new(0.0, 0.0),
        eigenstate[1],
        Complex64::new(0.0, 0.0),
    ]
}

fn complex_state_to_pairs(state: &[Complex64]) -> Vec<[f64; 2]> {
    state.iter().map(|value| [value.re, value.im]).collect()
}

fn run_projective_density<S>(report: &mut Report, simulator_name: &str)
where
    S: DensitySnapshot + ArbitraryRotationGateable,
{
    if let Some(reason) = construction_skip::<S>(1) {
        report.skip(simulator_name, "projective", reason);
        return;
    }
    let gates = Clifford::all_1q()
        .iter()
        .copied()
        .map(GateSpec::Clifford)
        .chain(non_clifford_named_gates().map(GateSpec::Named));
    for gate in gates {
        for input in PauliAxis::ALL {
            let expected = conjugate_matrix(gate.matrix(), input);
            let outcome = catch_unwind(AssertUnwindSafe(|| density_conjugation::<S>(gate, input)));
            match outcome {
                Ok(Ok(actual)) => {
                    let error = max_matrix_error(&actual, &expected);
                    if error <= S::TOLERANCE {
                        report.passed();
                    } else {
                        report.failed(Failure {
                            simulator: simulator_name.to_string(),
                            gate: format!("{}: {} conjugation", gate.label(), input.label()),
                            tier: "projective",
                            expected: format_matrix2(&expected),
                            actual: format!("max_error={error:.6e}; {}", format_matrix2(&actual)),
                            algebra: simplify_pauli_combination(&actual),
                        });
                    }
                }
                Ok(Err(error)) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{}: {} conjugation", gate.label(), input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected),
                    actual: format!("backend error: {error}"),
                    algebra: "the backend did not produce a conjugated matrix".to_string(),
                }),
                Err(payload) => report.failed(Failure {
                    simulator: simulator_name.to_string(),
                    gate: format!("{}: {} conjugation", gate.label(), input.label()),
                    tier: "projective",
                    expected: format_matrix2(&expected),
                    actual: format!("panic: {}", panic_message(payload.as_ref())),
                    algebra: "the density path panicked before matrix simplification".to_string(),
                }),
            }
        }
    }
}

type Matrix2 = [Complex64; 4];

fn unpack_matrix(matrix: SingleQubitGateMatrix) -> Matrix2 {
    [
        Complex64::new(matrix[0], matrix[1]),
        Complex64::new(matrix[2], matrix[3]),
        Complex64::new(matrix[4], matrix[5]),
        Complex64::new(matrix[6], matrix[7]),
    ]
}

fn matrix_multiply(left: &Matrix2, right: &Matrix2) -> Matrix2 {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

fn matrix_dagger(matrix: &Matrix2) -> Matrix2 {
    [
        matrix[0].conj(),
        matrix[2].conj(),
        matrix[1].conj(),
        matrix[3].conj(),
    ]
}

fn conjugate_matrix(matrix: SingleQubitGateMatrix, input: PauliAxis) -> Matrix2 {
    let unitary = unpack_matrix(matrix);
    let pauli = unpack_matrix(
        input
            .gate_type()
            .canonical_1q_matrix()
            .expect("Pauli must have a canonical matrix"),
    );
    matrix_multiply(&matrix_multiply(&unitary, &pauli), &matrix_dagger(&unitary))
}

fn max_matrix_error(actual: &Matrix2, expected: &Matrix2) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max)
}

fn classify_signed_pauli(matrix: &Matrix2, tolerance: f64) -> Option<SignedPauli> {
    for axis in PauliAxis::ALL {
        let pauli = unpack_matrix(
            axis.gate_type()
                .canonical_1q_matrix()
                .expect("Pauli must have a canonical matrix"),
        );
        for negative in [false, true] {
            let sign = if negative { -1.0 } else { 1.0 };
            let candidate = pauli.map(|value| value * sign);
            if max_matrix_error(matrix, &candidate) <= tolerance {
                return Some(SignedPauli { negative, axis });
            }
        }
    }
    None
}

fn simplify_pauli_combination(matrix: &Matrix2) -> String {
    let identity = unpack_matrix(
        GateType::I
            .canonical_1q_matrix()
            .expect("identity must have a canonical matrix"),
    );
    let mut terms = Vec::new();
    for (label, basis) in std::iter::once(("I", identity)).chain(PauliAxis::ALL.map(|axis| {
        (
            axis.label(),
            unpack_matrix(
                axis.gate_type()
                    .canonical_1q_matrix()
                    .expect("Pauli must have a canonical matrix"),
            ),
        )
    })) {
        let coefficient = basis
            .iter()
            .zip(matrix)
            .map(|(basis, matrix)| basis.conj() * matrix)
            .sum::<Complex64>()
            / 2.0;
        if coefficient.norm() > 1e-9 {
            terms.push(format!(
                "({:.9}{:+.9}i){label}",
                coefficient.re, coefficient.im
            ));
        }
    }
    if terms.is_empty() {
        "simplifies to the zero matrix".to_string()
    } else {
        format!("simplifies to {}", terms.join(" + "))
    }
}

fn format_matrix2(matrix: &Matrix2) -> String {
    format!(
        "[[{:.8}{:+.8}i, {:.8}{:+.8}i], [{:.8}{:+.8}i, {:.8}{:+.8}i]]",
        matrix[0].re,
        matrix[0].im,
        matrix[1].re,
        matrix[1].im,
        matrix[2].re,
        matrix[2].im,
        matrix[3].re,
        matrix[3].im,
    )
}

fn record_trait_coverage_failure(report: &mut Report, simulator_name: &str, detail: &str) {
    report.failed(Failure {
        simulator: simulator_name.to_string(),
        gate: "gate-trait-driven coverage".to_string(),
        tier: "projective",
        expected: "a gate trait whose bounds enumerate every supported named gate".to_string(),
        actual: detail.to_string(),
        algebra:
            "no gate action can be certified without reintroducing a hand-maintained gate roster"
                .to_string(),
    });
}

macro_rules! register_simulator {
    ($report:ident, $name:literal, $ty:ty, phase_exact, arbitrary) => {{
        $report.rostered += 1;
        run_phase_exact_arbitrary::<$ty>(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, phase_exact, clifford) => {{
        $report.rostered += 1;
        run_phase_exact_clifford::<$ty>(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, state_clifford) => {{
        $report.rostered += 1;
        run_projective_state::<$ty>(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, cpp_tableau) => {{
        $report.rostered += 1;
        let _: Option<$ty> = None;
        run_projective_cpp(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, density_arbitrary) => {{
        $report.rostered += 1;
        run_projective_density::<$ty>(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, pauli_prop) => {{
        $report.rostered += 1;
        let _ = stringify!($ty);
        run_pauli_prop(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, bitmask_pauli_prop) => {{
        $report.rostered += 1;
        let _ = stringify!($ty);
        run_bitmask_pauli_prop(&mut $report, $name);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, missing_gate_trait, $reason:literal) => {{
        $report.rostered += 1;
        let _ = stringify!($ty);
        record_trait_coverage_failure(&mut $report, $name, $reason);
    }};
    ($report:ident, $name:literal, $ty:ty, phase_exact, unavailable, $reason:literal) => {{
        $report.rostered += 1;
        let _ = stringify!($ty);
        $report.skip($name, "phase-exact", $reason);
    }};
    ($report:ident, $name:literal, $ty:ty, projective, unavailable, $reason:literal) => {{
        $report.rostered += 1;
        let _ = stringify!($ty);
        $report.skip($name, "projective", $reason);
    }};
}

#[test]
fn every_registered_simulator_conforms_to_its_declared_gate_tier() {
    let mut report = Report::default();

    // Phase-exact representations. Each line is one simulator declaration.
    register_simulator!(report, "StateVecSoA", StateVecSoA, phase_exact, arbitrary);
    register_simulator!(
        report,
        "StateVecSoA32",
        StateVecSoA32,
        phase_exact,
        arbitrary
    );
    register_simulator!(report, "StateVecAoS", StateVecAoS, phase_exact, arbitrary);
    register_simulator!(
        report,
        "SparseStateVecAoS",
        SparseStateVecAoS,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "SparseStateVecSoA",
        SparseStateVecSoA,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "GpuStateVec32",
        GpuStateVec32,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "GpuStateVec64",
        GpuStateVec64,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "GpuStateVecAuto",
        GpuStateVecAuto,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "CuStateVec",
        pecos_cuquantum::CuStateVec,
        phase_exact,
        unavailable,
        "cuQuantum is not linked in this environment; the CUDA SDK-backed crate cannot be built or run here"
    );
    register_simulator!(
        report,
        "StabVecGeneric",
        StabVecGeneric,
        phase_exact,
        arbitrary
    );
    register_simulator!(
        report,
        "CHFormGeneric",
        CHFormGeneric,
        phase_exact,
        clifford
    );
    register_simulator!(report, "StabMps", StabMps, phase_exact, arbitrary);
    register_simulator!(
        report,
        "StabMpsCompile",
        StabMpsCompile,
        phase_exact,
        arbitrary
    );
    register_simulator!(report, "Mast", Mast, phase_exact, arbitrary);

    // Projective representations. Density matrices stay here because U's global phase cancels.
    register_simulator!(
        report,
        "SparseStabGeneric",
        SparseStabGeneric,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "SparseStabYGeneric",
        SparseStabYGeneric,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "SparseStabHybrid",
        SparseStabHybrid,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "SparseColOnly",
        SparseColOnly,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "SparseRowOnly",
        SparseRowOnly,
        projective,
        state_clifford
    );
    register_simulator!(report, "DenseStab", DenseStab, projective, state_clifford);
    register_simulator!(
        report,
        "DenseStabColOnly",
        DenseStabColOnly,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "DenseStabRowOnly",
        DenseStabRowOnly,
        projective,
        state_clifford
    );
    register_simulator!(report, "GpuStab", GpuStab, projective, state_clifford);
    register_simulator!(report, "GpuStabOpt", GpuStabOpt, projective, state_clifford);
    register_simulator!(
        report,
        "GpuStabParallel",
        GpuStabParallel,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "pecos_gpu_sims::GpuStab",
        DefaultGpuStab,
        projective,
        state_clifford
    );
    register_simulator!(
        report,
        "CuStabilizer",
        pecos_cuquantum::CuStabilizer,
        projective,
        unavailable,
        "cuQuantum is not linked in this environment; the CUDA SDK-backed crate cannot be built or run here"
    );
    register_simulator!(
        report,
        "CppSparseStab",
        CppSparseStab,
        projective,
        cpp_tableau
    );
    register_simulator!(report, "PauliProp", PauliProp, projective, pauli_prop);
    register_simulator!(
        report,
        "BitmaskPauliProp",
        BitmaskPauliProp,
        projective,
        bitmask_pauli_prop
    );
    register_simulator!(
        report,
        "GraphStateSim",
        GraphStateSim,
        projective,
        state_clifford
    );
    register_simulator!(report, "Stabilizer", Stabilizer, projective, state_clifford);
    register_simulator!(
        report,
        "SymbolicSparseStab",
        pecos_simulators::SymbolicSparseStab,
        projective,
        missing_gate_trait,
        "the symbolic bitset simulator has inherent usize gate methods but implements only QuantumSimulator, not a gate trait"
    );
    register_simulator!(
        report,
        "SymbolicSparseStabVecSet",
        pecos_simulators::SymbolicSparseStabVecSet,
        projective,
        missing_gate_trait,
        "the symbolic VecSet simulator has inherent usize gate methods but implements only QuantumSimulator, not a gate trait"
    );
    register_simulator!(
        report,
        "DensityMatrix",
        DensityMatrix,
        projective,
        density_arbitrary
    );
    register_simulator!(
        report,
        "GpuDensityMatrix32",
        GpuDensityMatrix32,
        projective,
        density_arbitrary
    );
    register_simulator!(
        report,
        "GpuDensityMatrix64",
        GpuDensityMatrix64,
        projective,
        density_arbitrary
    );

    report.print();
    assert!(
        report.failures.is_empty(),
        "gate conformance found {} failures across {} checks ({} simulator skips); see CONFORMANCE_FAILURE records above",
        report.failures.len(),
        report.checks,
        report.skips.len(),
    );
}
