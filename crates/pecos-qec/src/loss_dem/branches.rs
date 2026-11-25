//! First-order branch catalog. Later roots are enumerated even if an earlier root
//! could block their draw. No survival factors, posteriors, or normalization occur here.
use super::branch_replay::{ReplayContext, ReplayOptions};
use super::trajectory::Program;
use super::{Location, LossDemError, OutputTable, ProbeId, Span, invalid};
use crate::fault_tolerance::dem_builder::FaultContribution;
use crate::{DetectorErrorModel, FaultMechanism};
use pecos_core::BitSet;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_quantum::TickCircuit;
use std::collections::{BTreeMap, BTreeSet};

/// Source of an independent first-order root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootKind {
    /// An events label containing L, including correlated partner Paulis.
    Events,
    /// Explicit Idle operation.
    Idle,
    /// Implicit idle following all after-two-qubit hooks.
    AfterTwoQubitIdle,
    /// Post-readout loss (including record-free Z heralds).
    Measurement,
    /// Loss after retained preparation or MPZ's reload.
    Preparation,
}

/// Ordered stage within one stored command. Hook indices are positions in the
/// built hook order, not positions in the configured channel list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootPhase {
    /// Before-site hook position.
    BeforeHook(usize),
    /// The scheduled operation (or explicit idle loss).
    Operation,
    /// MPZ/MZ measurement loss precedes any reload.
    MeasurementLoss,
    /// Preparation/reload bookkeeping.
    Reload,
    /// Preparation loss follows reload.
    PreparationLoss,
    /// After-site hook position.
    AfterHook(usize),
    /// Idle after all after-site hooks.
    AfterTwoQubitIdle,
}

/// Stable root identity; scalar sources have no channel index and label L.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootKey {
    /// Source family.
    pub kind: RootKind,
    /// Index in the built loss channel list, for events.
    pub channel: Option<usize>,
    /// Entire joint label, never separated into independent legs.
    pub label: String,
    /// First operand of the event, or the scalar-source operand.
    pub location: Location,
    /// Stage of the engine command.
    pub phase: RootPhase,
}

/// One recovery channel at one operand. Channel identity distinguishes repeated
/// recoveries at the same location. Removal omits contributions before merging;
/// the XOR inverse is not used, since it fails at `p_tag = 1/2` (including FP = 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FalseAlarmTag {
    /// Index in the built loss channel list.
    pub channel: usize,
    /// Recovery site and operand.
    pub location: Location,
}

/// Lost-set transition, ordered by engine action (including within a command).
#[derive(Debug, Clone)]
pub struct LossMoment {
    /// Stored circuit coordinates.
    pub location: Location,
    /// Stage at these coordinates.
    pub phase: RootPhase,
    /// Lost qubits after the transition, sorted.
    pub lost: BTreeSet<usize>,
}

/// Affine branch distribution, uniform on `constant + span`.
/// Coordinates are detector IDs followed by observable
/// IDs, offset by `num_detectors` (including gaps in IDs).
#[derive(Debug, Clone)]
pub struct BranchNormalForm {
    /// Reduced coset representative.
    pub constant: BitSet,
    /// GF(2) span of all visible random signatures.
    pub span: Span,
}

/// One resolution of recovery-FN, spread, and non-coset partner Pauli outcomes.
#[derive(Debug, Clone)]
pub struct LossBranch {
    /// Root prior times conditional trajectory factors.
    pub weight: f64,
    /// Exact sorted probe set assuming nominal detect readout. Configured detect
    /// readout rows accepted by E1 change engine flags, but not these catalog flags.
    pub flags: BTreeSet<ProbeId>,
    /// Every lost-set change, starting with the root.
    pub lost_over_time: Vec<LossMoment>,
    /// Emission indices read while lost.
    pub lost_records: BTreeSet<usize>,
    /// First operand of each dropped operation instance.
    pub dropped_gates: BTreeSet<Location>,
    /// Recovery channels visited while that operand was lost, including FN outcomes.
    pub false_alarm_removals: BTreeSet<FalseAlarmTag>,
    /// Branch DEM in affine normal form.
    pub normal_form: BranchNormalForm,
}

/// First-order root and its mutually exclusive branches.
#[derive(Debug, Clone)]
pub struct LossRoot {
    /// Identity of the site and joint label.
    pub key: RootKey,
    /// Raw scaled site probability times label weight; no survival correction.
    pub prior: f64,
    /// Number of explored trajectory segments (every pending-path pop, even if
    /// subsequently pruned). This is bounded by `max_branches`, not by the number
    /// of retained leaves.
    pub explored_paths: usize,
    /// Retained trajectory resolutions.
    pub branches: Vec<LossBranch>,
    /// Absolute mass pruned at this root's branching nodes.
    pub dropped_mass: f64,
    /// Pruned mass keyed by flags accumulated at pruning. These sets are lower
    /// bounds on final flag sets, not exact final-pattern masses.
    pub dropped_mass_by_flags: BTreeMap<BTreeSet<ProbeId>, f64>,
}

/// Index into `LossCatalog::roots[root].branches[branch]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BranchId {
    /// Root index.
    pub root: usize,
    /// Branch index within its root.
    pub branch: usize,
}

/// Why a branch's independent-mechanism representation is approximated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Approximation {
    /// Correlated rank exceeds the caller's bound.
    RankLimit,
    /// Nonzero reduced constant with positive rank at absolute weight below one.
    WeightedCoset,
}

/// Catalog-wide diagnostics. Dropped mass is additionally recorded per root.
#[derive(Debug, Clone, Default)]
pub struct BranchReport {
    /// Total absolute mass pruned, summed across first-order roots.
    pub dropped_mass: f64,
    /// Lower-bound flag sets at pruning, accumulated across roots.
    pub dropped_mass_by_flags: BTreeMap<BTreeSet<ProbeId>, f64>,
    /// Branches requiring approximations and their reasons.
    pub approximated: BTreeMap<BranchId, BTreeSet<Approximation>>,
    /// Number of retained branches with a random observable.
    pub random_observable_count: usize,
    /// Sum of their absolute weights (each branch counted once).
    pub random_observable_weight: f64,
}

#[derive(Debug, Clone)]
enum ContributionIndex {
    Base(usize),
    Tagged(FalseAlarmTag, usize),
}

/// Precomputed loss-conditioned branch layer. E3 omits tagged contributions
/// before merging per shot; no inverse of merged probabilities is used.
#[derive(Debug, Clone)]
pub struct LossCatalog {
    /// E1 Pauli background plus unflagged loss branches.
    pub background: DetectorErrorModel,
    /// Independent background contributions excluding false alarms: E1 Pauli
    /// noise, pure-Pauli events, then unflagged branches, in stable insertion order.
    base: Vec<FaultContribution>,
    /// E1 output/record layout of the original circuit.
    pub outputs: OutputTable,
    /// Per-tag independent contributions in insertion order, before any merging.
    false_alarm_tags: BTreeMap<FalseAlarmTag, Vec<FaultContribution>>,
    /// Roots in engine-action order, then label order.
    pub roots: Vec<LossRoot>,
    /// Exact sorted flag set to retained branch references, including the empty set.
    pub by_flags: BTreeMap<BTreeSet<ProbeId>, Vec<BranchId>>,
    /// Pruning, approximation, and logical-randomness diagnostics.
    pub report: BranchReport,
    /// Detector coordinate width; observable bits start here in each normal form.
    pub num_detectors: usize,
    /// Observable coordinate width.
    pub num_observables: usize,
    // Preserve the ordinary builder's floating-point XOR accumulation order,
    // including interleaved event and recovery channels.
    contribution_order: Vec<ContributionIndex>,
}

impl LossCatalog {
    /// Build first-order roots, trajectories, affine DEMs, and background.
    /// `min_branch_weight` is finite in [0, 1]. `max_branches` bounds the
    /// explored paths for each root: every pending-path pop counts, including
    /// segments subsequently pruned. Zero permits no explored path. Non-coset
    /// partner Pauli outcomes use the same cutoff and cap as recovery and spread;
    /// single Paulis and cosets retain their coin form.
    ///
    /// # Errors
    /// Names unsupported settings, invalid cutoff, nondeterministic lossless
    /// definitions, or the root and count exceeding the branch cap.
    pub fn build(
        circuit: &TickCircuit,
        noise_model: &GeneralNoiseModel,
        min_branch_weight: f64,
        max_correlated_rank: usize,
        max_branches: usize,
    ) -> Result<Self, LossDemError> {
        Self::build_with_options(
            circuit,
            noise_model,
            min_branch_weight,
            max_correlated_rank,
            max_branches,
            ReplayOptions::default(),
        )
    }

    /// Independent base sources, without false alarms, in stable insertion order.
    /// Whole contributions retain source metadata and decomposition components.
    /// The collection cannot be mutated independently of its private order index.
    ///
    /// ```compile_fail,E0616
    /// fn mutate(catalog: &mut pecos_qec::loss_dem::LossCatalog) {
    ///     catalog.base.clear();
    /// }
    /// ```
    #[must_use]
    pub fn base(&self) -> &[FaultContribution] {
        &self.base
    }

    /// Per-tag independent sources before merging, in stable insertion order.
    /// This read-only view keeps the private contribution-order indices valid.
    ///
    /// ```compile_fail,E0616
    /// fn mutate(catalog: &mut pecos_qec::loss_dem::LossCatalog) {
    ///     catalog.false_alarm_tags.clear();
    /// }
    /// ```
    #[must_use]
    pub fn false_alarm_tags(&self) -> &BTreeMap<FalseAlarmTag, Vec<FaultContribution>> {
        &self.false_alarm_tags
    }

    /// Merge the base and all tags outside `removals`, omitting contributions
    /// before independent XOR combination. This also works at FP = 1. Original
    /// contribution order, all DEM renderings, counts and residuals match `background` exactly
    /// when `removals` is empty.
    #[must_use]
    pub fn background_without(&self, removals: &BTreeSet<FalseAlarmTag>) -> DetectorErrorModel {
        // False alarms are direct exact location channels: they contribute no
        // graphlike counts or fit residuals. All sources that do stay in base.
        let mut dem = self.background.clone_without_contributions();
        for index in &self.contribution_order {
            let contribution = match *index {
                ContributionIndex::Base(index) => &self.base[index],
                ContributionIndex::Tagged(tag, index) => {
                    if removals.contains(&tag) {
                        continue;
                    }
                    &self.false_alarm_tags[&tag][index]
                }
            };
            dem.push_contribution(contribution.clone());
        }
        dem
    }

    pub(super) fn build_with_options(
        circuit: &TickCircuit,
        noise_model: &GeneralNoiseModel,
        min_branch_weight: f64,
        max_correlated_rank: usize,
        max_branches: usize,
        options: ReplayOptions,
    ) -> Result<Self, LossDemError> {
        if !min_branch_weight.is_finite() || !(0.0..=1.0).contains(&min_branch_weight) {
            return Err(invalid(format!(
                "min_branch_weight = {min_branch_weight} is invalid; supply a finite cutoff in [0, 1]"
            )));
        }
        let tagged = super::channels::tagged_background(circuit, noise_model)?;
        let mut base = Vec::new();
        let mut false_alarm_tags = BTreeMap::new();
        let mut tagged_indices = BTreeMap::new();
        for (tag, range) in tagged.tags {
            let contributions = &tagged.dem.contributions()[range.clone()];
            false_alarm_tags.insert(tag, contributions.to_vec());
            for (within_tag, index) in range.enumerate() {
                tagged_indices.insert(index, (tag, within_tag));
            }
        }
        let mut contribution_order = Vec::new();
        for (index, contribution) in tagged.dem.contributions().iter().enumerate() {
            if let Some(&(tag, within_tag)) = tagged_indices.get(&index) {
                contribution_order.push(ContributionIndex::Tagged(tag, within_tag));
            } else {
                contribution_order.push(ContributionIndex::Base(base.len()));
                base.push(contribution.clone());
            }
        }
        let outputs = OutputTable::new(circuit)?;
        let replay = ReplayContext::new(circuit)?;
        let program = Program::new(circuit, &noise_model.description(), &outputs)?;
        let mut catalog = Self {
            background: tagged.dem,
            base,
            outputs,
            false_alarm_tags,
            contribution_order,
            roots: Vec::new(),
            by_flags: BTreeMap::new(),
            report: BranchReport::default(),
            num_detectors: replay.num_detectors,
            num_observables: replay.num_observables,
        };
        for seed in &program.roots {
            let (paths, dropped, explored_paths) =
                program.walk(seed, min_branch_weight, max_branches, options)?;
            let mut root = LossRoot {
                key: seed.key.clone(),
                prior: seed.prior,
                explored_paths,
                branches: Vec::new(),
                dropped_mass: dropped.values().sum(),
                dropped_mass_by_flags: dropped,
            };
            catalog.report.dropped_mass += root.dropped_mass;
            for (flags, weight) in &root.dropped_mass_by_flags {
                *catalog
                    .report
                    .dropped_mass_by_flags
                    .entry(flags.clone())
                    .or_default() += weight;
            }
            for path in paths {
                let normal_form = replay.normal_form(&program, &path, options)?;
                let id = BranchId {
                    root: catalog.roots.len(),
                    branch: root.branches.len(),
                };
                let mut approximations = BTreeSet::new();
                if normal_form
                    .span
                    .basis()
                    .iter()
                    .any(|v| v.iter().any(|b| b >= replay.num_detectors))
                {
                    catalog.report.random_observable_count += 1;
                    catalog.report.random_observable_weight += path.weight;
                }
                if path.flags.is_empty() {
                    let start = catalog.background.num_contributions();
                    approximations.extend(emit_normal_form(
                        &mut catalog.background,
                        &normal_form,
                        path.weight,
                        max_correlated_rank,
                        replay.num_detectors,
                    )?);
                    for contribution in &catalog.background.contributions()[start..] {
                        catalog
                            .contribution_order
                            .push(ContributionIndex::Base(catalog.base.len()));
                        catalog.base.push(contribution.clone());
                    }
                }
                if !approximations.is_empty() {
                    catalog.report.approximated.insert(id, approximations);
                }
                catalog
                    .by_flags
                    .entry(path.flags.clone())
                    .or_default()
                    .push(id);
                root.branches.push(LossBranch {
                    weight: path.weight,
                    flags: path.flags,
                    lost_over_time: path.history,
                    lost_records: path.records,
                    dropped_gates: path.dropped,
                    false_alarm_removals: path.removals,
                    normal_form,
                });
            }
            catalog.roots.push(root);
        }
        Ok(catalog)
    }
}

pub(super) fn effect(
    vector: &BitSet,
    detector_width: usize,
) -> Result<FaultMechanism, LossDemError> {
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    for bit in vector {
        let (target, index) = if bit < detector_width {
            (&mut detectors, bit)
        } else {
            (&mut observables, bit - detector_width)
        };
        target.push(u32::try_from(index).map_err(|_| {
            invalid(format!(
                "effect index {index} exceeds u32; reduce the detector/observable ID range"
            ))
        })?);
    }
    Ok(FaultMechanism::from_unsorted(detectors, observables))
}

// Stable at small w; rank zero has no nonzero span elements and is never used.
pub(super) fn correlated_probability(weight: f64, rank: usize) -> f64 {
    let mut exponent = 2.0;
    for _ in 0..rank {
        exponent *= 0.5;
    }
    if weight >= 1.0 {
        0.5
    } else {
        -((-weight).ln_1p() * exponent).exp_m1() / 2.0
    }
}

pub(super) fn span_elements(span: &Span) -> Result<Vec<BitSet>, LossDemError> {
    let shift = u32::try_from(span.rank()).map_err(|_| {
        invalid(format!(
            "span rank {} cannot be enumerated; lower max_correlated_rank",
            span.rank()
        ))
    })?;
    let count = 1usize.checked_shl(shift).ok_or_else(|| {
        invalid(format!(
            "span rank {} overflows the element count; lower max_correlated_rank",
            span.rank()
        ))
    })?;
    let mut elements = Vec::new();
    elements.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "span rank {} needs {count} elements: {error}; lower max_correlated_rank",
            span.rank()
        ))
    })?;
    elements.push(BitSet::new());
    for row in span.basis() {
        let n = elements.len();
        for i in 0..n {
            let mut next = elements[i].clone();
            next ^= row;
            elements.push(next);
        }
    }
    Ok(elements)
}

pub(super) fn emit_normal_form(
    dem: &mut DetectorErrorModel,
    form: &BranchNormalForm,
    weight: f64,
    max_rank: usize,
    detector_width: usize,
) -> Result<BTreeSet<Approximation>, LossDemError> {
    let mut approximations = BTreeSet::new();
    let rank = form.span.rank();
    let mut add = |v: &BitSet, p| -> Result<(), LossDemError> {
        dem.add_direct_contribution(effect(v, detector_width)?, p);
        Ok(())
    };
    if rank > max_rank {
        approximations.insert(Approximation::RankLimit);
        for v in form.span.basis() {
            add(v, correlated_probability(weight, 1))?;
        }
        add(&form.constant, weight)?;
    } else if form.constant.is_empty() {
        for v in span_elements(&form.span)?.iter().skip(1) {
            add(v, correlated_probability(weight, rank))?;
        }
    } else if weight < 1.0 {
        if rank > 0 {
            approximations.insert(Approximation::WeightedCoset);
        }
        let mut p = weight;
        for _ in 0..rank {
            p *= 0.5;
        }
        for mut v in span_elements(&form.span)? {
            v ^= &form.constant;
            add(&v, p)?;
        }
    } else {
        add(&form.constant, 1.0)?;
        for v in form.span.basis() {
            add(v, 0.5)?;
        }
    }
    Ok(approximations)
}

#[cfg(test)]
#[test]
fn structured_catalog_sources_preserve_decomposition_and_counts() {
    use super::Side;
    use crate::fault_tolerance::dem_builder::{
        DirectSourceFamily, NoiseChannelKind, NoiseChannelResidual,
    };
    use crate::fault_tolerance::propagator::Pauli;
    use pecos_core::gate_type::GateType;

    let mut circuit = TickCircuit::new();
    for record in circuit.tick().mz(&[0, 1]) {
        circuit.detector(&[record]).unwrap();
    }
    let mut catalog =
        LossCatalog::build(&circuit, &GeneralNoiseModel::default(), 0.0, 12, 100).unwrap();
    let x = FaultMechanism::from_unsorted(vec![0], vec![]);
    let z = FaultMechanism::from_unsorted(vec![1], vec![]);
    let mut y = FaultContribution::y_decomposed(x.xor(&z), &x, &z, 0.07);
    y.location_indices = vec![3].into();
    y.paulis = vec![Pauli::Y].into();
    y.source_gate_types = vec![GateType::H].into();
    y.source_before_flags = vec![false].into();
    y.replacement_branch = true;
    let mut components = FaultContribution::source_decomposed([x.clone(), z], 0.11);
    components.location_indices = vec![4, 5].into();
    components.paulis = vec![Pauli::X, Pauli::Z].into();
    components.source_gate_types = vec![GateType::CX, GateType::CX].into();
    components.source_before_flags = vec![false, false].into();
    components.direct_source_family = Some(DirectSourceFamily::Other);
    catalog.base = vec![y, components];
    for contribution in &catalog.base {
        catalog.background.push_contribution(contribution.clone());
        catalog.background.mark_graphlike_decomposable(0, 1);
    }
    catalog
        .background
        .add_idle_noise_residual(NoiseChannelResidual {
            location_index: 3,
            channel_kind: NoiseChannelKind::SingleQubitGate,
            effect: x.clone(),
            magnitude: 0.02,
            channel_weight: 0.2,
        });
    let without_tag = catalog.background.clone();
    let tag = FalseAlarmTag {
        channel: 0,
        location: Location {
            tick: 0,
            batch: 0,
            operand: 0,
            side: Side::Before,
        },
    };
    let mut tagged = FaultContribution::direct(x, 0.13);
    tagged.location_indices = vec![9].into();
    tagged.paulis = vec![Pauli::X].into();
    tagged.source_gate_types = vec![GateType::H].into();
    tagged.source_before_flags = vec![false].into();
    catalog.background.push_contribution(tagged.clone());
    catalog.false_alarm_tags.insert(tag, vec![tagged]);
    catalog.contribution_order = vec![
        ContributionIndex::Base(0),
        ContributionIndex::Base(1),
        ContributionIndex::Tagged(tag, 0),
    ];
    // The expected sources do not pass through push_contribution: a broken
    // cloning helper must not corrupt both the reference and the reconstruction.
    let expected_sources = catalog
        .base
        .iter()
        .chain(&catalog.false_alarm_tags[&tag])
        .cloned()
        .collect::<Vec<_>>();
    super::branch_tests::assert_same_sources(
        catalog.background_without(&BTreeSet::new()).contributions(),
        &expected_sources,
    );
    assert_eq!(catalog.background.graphlike_decomposable_count(0, 1), 2);
    super::branch_tests::assert_same_dem_renderings(
        &catalog.background_without(&BTreeSet::new()),
        &catalog.background,
    );
    super::branch_tests::assert_same_dem_renderings(
        &catalog.background_without(&BTreeSet::from([tag])),
        &without_tag,
    );
}
