//! Flag assignments and pure, component-wise posterior mixing.
//!
//! For each engine draw d, `f_d` is the absolute prior mass of its nonempty-flag
//! branches. Assignment weights multiply `w_b / (1 - f_d)`, accounting for the
//! other draws not flagging; the shared product of `(1 - f_d)` cancels. A draw
//! with `f_d >= 1` has no finite odds representation and is explicitly rejected.
//! Physically blocked roots are excluded before weights or covers contribute.
//! These odds use catalog draw marginals; they do not recompute the no-flag
//! likelihood when different assignments block different sets of later draws.
//! `BlockedDrawOdds` conservatively marks catalog-built shots with an ambiguous
//! region.
//! A region has multiple assignments exactly when one of its factored groups
//! has multiple alternatives; this test avoids enumerating joint assignments
//! or comparing their blocked-draw sets. It can mark shots with identical
//! blocked-draw likelihoods as well. Single-assignment shots are not marked.
use super::branches::{effect, emit_normal_form};
use super::{
    Approximation, BranchId, BranchNormalForm, FalseAlarmTag, LossCatalog, LossDemError, ProbeId,
    Span, invalid,
};
use crate::DetectorErrorModel;
use crate::fault_tolerance::dem_builder::{DemOutput, DetectorDef, FaultContribution};
use pecos_core::BitSet;
use pecos_decoder_core::obs_mask::ObsMask;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// Required bounds for shot construction. Independent components are bounded separately.
#[derive(Debug, Clone, Copy)]
pub struct ShotLimits {
    /// Maximum valid covers per region, alternatives per connected component,
    /// and alternatives in each region's flattened cover remainder.
    pub max_assignments: usize,
    /// Maximum visited cover and candidate-search nodes combined, per region.
    /// Counts recursive entries, including memo hits; excludes index/region setup.
    pub max_search_nodes: usize,
    /// Maximum rank enumerated by E2's correlated emitter.
    pub max_correlated_rank: usize,
}

/// One exclusive alternative within an independent group.
#[derive(Debug, Clone)]
pub struct ShotAlternative {
    /// Branches chosen from distinct engine draws (empty for constructed forms).
    pub branches: Vec<BranchId>,
    /// Product of branch odds `w_b / (1 - f_d)`, before posterior normalisation.
    pub weight: f64,
    /// Posterior probability within this group.
    pub pi: f64,
    /// Reduced constant and sum of branch spans.
    pub normal_form: BranchNormalForm,
    /// Union of the chosen branches' false-alarm removals.
    pub removals: BTreeSet<FalseAlarmTag>,
}

/// Pure mixing inputs: background with the union of removals omitted, and the
/// omitted channels' unmerged, metadata-bearing contributions. Invisible Paulis
/// need not be retained; each tag originates in exactly one uniform X/Y/Z channel.
#[derive(Debug, Clone)]
pub struct ShotBackground {
    /// Background after removing every tag that any alternative removes.
    pub dem: DetectorErrorModel,
    /// Contributions of precisely those removed tags, before merging.
    pub removed: BTreeMap<FalseAlarmTag, Vec<FaultContribution>>,
}

/// Normalised decoder input and its affine offset.
#[derive(Debug, Clone)]
pub struct ShotDem {
    /// Standard DEM, with all probabilities in (0, 1/2].
    pub dem: DetectorErrorModel,
    /// XOR into the measured syndrome before decoding.
    pub detector_bias: Vec<u8>,
    /// XOR into the decoder's observable prediction.
    pub observable_bias: ObsMask,
    /// Independent groups of assignments, with posterior probabilities. The
    /// joint assignment posterior is their product; it is never materialised.
    pub assignments: Vec<Vec<ShotAlternative>>,
    /// Representation approximations, including those inherited from E2.
    pub approximations: BTreeSet<Approximation>,
    /// UPPER bound on absolute pruned prior mass, not posterior mass.
    /// Pruned keys are lower bounds on final flag sets.
    pub pruned_mass_upper_bound: f64,
}

/// Assignment failures are cacheable and retain the offending probe set.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AssignmentError {
    /// Cover and candidate traversal exceeded the region's work budget.
    #[error(
        "search node limit for probes {probes:?}: visited {visited} nodes, max_search_nodes = {max}; increase max_search_nodes or simplify the catalog's ambiguous regions"
    )]
    SearchNodeLimit {
        /// Region's probe set.
        probes: BTreeSet<ProbeId>,
        /// Count including the first node beyond the bound.
        visited: u128,
        /// Caller-supplied per-region bound.
        max: usize,
    },
    /// A flag does not name a catalog probe.
    #[error("invalid probes {probes:?}; catalog has {probe_count} probes")]
    InvalidProbe {
        /// Requested flags.
        probes: BTreeSet<ProbeId>,
        /// Number of valid probe IDs.
        probe_count: usize,
    },
    /// No positive-weight assignment explains this flag set.
    #[error("no assignment for probes {probes:?}")]
    NoAssignment {
        /// Requested flags.
        probes: BTreeSet<ProbeId>,
    },
    /// A component or the flattened remainder exceeds the caller's bound.
    #[error("at least max + 1 assignments (max = {max}) for probes {probes:?}")]
    TooManyAssignments {
        /// Requested flags.
        probes: BTreeSet<ProbeId>,
        /// Caller-supplied cap.
        max: usize,
    },
}

/// Failures produced by shot construction, without decoder-layer variants.
#[derive(Debug, thiserror::Error)]
pub enum ShotBuildError {
    /// Invalid probes, incompatible assignments or an exceeded search limit.
    #[error(transparent)]
    Assignment(#[from] AssignmentError),
    /// Invalid probabilities, dimensions or catalog/background data.
    #[error(transparent)]
    Dem(#[from] LossDemError),
}

pub(super) fn reduce(vector: &BitSet, span: &Span) -> Result<BitSet, LossDemError> {
    Ok(Span::new(span.width(), std::slice::from_ref(vector))?
        .quotient(span)?
        .basis()
        .first()
        .cloned()
        .unwrap_or_default())
}

/// Apply the E3 mixing rule to independent groups, then add the exact on/off
/// false-alarm marginals and normalise. This function has no catalog or engine
/// dependency. The builder represents empty flags by one group containing one
/// empty alternative at pi=1. Zero groups are also accepted as the equivalent
/// identity (background-only) input; an empty group is invalid.
/// Posterior sums are checked with a count-scaled floating-point error bound.
///
/// # Errors
/// Rejects invalid probabilities, dimensions, or missing false-alarm data.
pub fn mix_shot_dem(
    groups: &[Vec<ShotAlternative>],
    background: &ShotBackground,
    max_correlated_rank: usize,
) -> Result<ShotDem, LossDemError> {
    let (detector_width, observable_width) = dimensions(&background.dem)?;
    let width = detector_width + observable_width;
    let mut dem = background.dem.clone();
    let mut bias = BitSet::new();
    let mut approximations = BTreeSet::new();
    for (group_index, group) in groups.iter().enumerate() {
        let (total, roundoff) = group
            .iter()
            .fold((0.0_f64, 0.0_f64), |(sum, tolerance), a| {
                (sum + a.pi, tolerance + 8.0 * f64::EPSILON)
            });
        if group.is_empty()
            || group
                .iter()
                .any(|a| !a.pi.is_finite() || !(0.0..=1.0).contains(&a.pi))
            || (total - 1.0).abs() > roundoff
        {
            return Err(invalid(format!(
                "group {group_index} has total posterior mass {total} and probabilities {:?}; require finite probabilities in [0, 1] summing to one",
                group.iter().map(|a| a.pi).collect::<Vec<_>>()
            )));
        }
        let mut common = group[0].normal_form.span.clone();
        let mut reference = 0;
        for (index, alternative) in group.iter().enumerate() {
            if alternative.normal_form.span.width() != width {
                return Err(invalid(format!(
                    "group {group_index}, alternative {index}: normal-form width {} differs from background width {width}",
                    alternative.normal_form.span.width()
                )));
            }
            common = common.intersection(&alternative.normal_form.span)?;
            if alternative.pi > group[reference].pi {
                reference = index;
            }
        }
        for vector in common.basis() {
            dem.add_direct_contribution(effect(vector, detector_width)?, 0.5);
        }
        let reference_constant = reduce(
            &group[reference].normal_form.constant,
            &group[reference].normal_form.span,
        )?;
        bias ^= &reference_constant;
        for (index, alternative) in group.iter().enumerate() {
            let form = &alternative.normal_form;
            let constant = if index == reference {
                BitSet::new()
            } else {
                let mut delta = form.constant.clone();
                delta ^= &reference_constant;
                reduce(&delta, &form.span)?
            };
            approximations.extend(emit_normal_form(
                &mut dem,
                &BranchNormalForm {
                    constant,
                    span: form.span.quotient(&common)?,
                },
                alternative.pi,
                max_correlated_rank,
                detector_width,
            )?);
        }
    }
    let removals: BTreeSet<_> = groups
        .iter()
        .flatten()
        .flat_map(|a| &a.removals)
        .copied()
        .collect();
    if removals != background.removed.keys().copied().collect() {
        return Err(invalid(format!(
            "false-alarm removal tags {removals:?} differ from supplied tags {:?}",
            background.removed.keys().collect::<Vec<_>>()
        )));
    }
    for tag in removals {
        let keep: f64 = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .filter(|a| !a.removals.contains(&tag))
                    .map(|a| a.pi)
                    .sum::<f64>()
            })
            .product();
        if keep > 0.0 && keep < 1.0 {
            approximations.insert(Approximation::FalseAlarmMixture);
        }
        for contribution in &background.removed[&tag] {
            let mut retained = contribution.clone();
            let p = contribution.probability;
            // Rationalised form of (1 - sqrt(1 - keep * (1 - (1 - 2p)^2))) / 2.
            let channel = keep * 4.0 * p * (1.0 - p);
            retained.probability = channel / (2.0 * (1.0 + (1.0 - channel).max(0.0).sqrt()));
            dem.push_contribution(retained);
        }
    }
    let mut result = normalise(&dem, &bias)?;
    result.assignments = groups.to_vec();
    result.approximations = approximations;
    Ok(result)
}

// num_detectors() counts definitions, whereas decoder coordinates use max ID + 1.
// Include implicit targets before folding, so the pure and catalog paths agree.
fn dimensions(dem: &DetectorErrorModel) -> Result<(usize, usize), LossDemError> {
    let detectors = dem
        .detectors
        .iter()
        .map(|d| d.id)
        .chain(
            dem.contributions()
                .iter()
                .flat_map(|c| c.effect.detectors.iter().copied()),
        )
        .max();
    let observables = dem
        .observables()
        .map(|o| o.id)
        .chain(
            dem.contributions()
                .iter()
                .flat_map(|c| c.effect.dem_outputs.iter().copied()),
        )
        .max();
    let width = |last: Option<u32>| -> Result<usize, LossDemError> {
        last.map_or(Ok(0), |id| {
            usize::try_from(u64::from(id) + 1)
                .map_err(|_| invalid(format!("DEM dimension {} exceeds usize", u64::from(id) + 1)))
        })
    };
    Ok((width(detectors)?, width(observables)?))
}

pub(super) fn normalise(dem: &DetectorErrorModel, bias: &BitSet) -> Result<ShotDem, LossDemError> {
    let (detector_width, observable_width) = dimensions(dem)?;
    let mut detector_bias = vec![0; detector_width];
    let mut observable_bias = ObsMask::new();
    for bit in bias {
        if bit < detector_bias.len() {
            detector_bias[bit] ^= 1;
        } else {
            observable_bias.toggle(bit - detector_bias.len());
        }
    }
    let mut normalised = dem.clone_without_contributions();
    // A source may be the only mention of the highest ID. Keep its declaration
    // when folding it, including gaps in detector and observable IDs.
    if detector_width > 0 {
        let id = u32::try_from(detector_width - 1)
            .map_err(|_| invalid(format!("detector ID {} exceeds u32", detector_width - 1)))?;
        if !normalised.detectors.iter().any(|d| d.id == id) {
            normalised.add_detector(DetectorDef::new(id));
        }
    }
    if observable_width > normalised.num_observables() {
        let id = u32::try_from(observable_width - 1).map_err(|_| {
            invalid(format!(
                "observable ID {} exceeds u32",
                observable_width - 1
            ))
        })?;
        normalised.add_observable(DemOutput::new(id));
    }
    for source in dem.contributions() {
        let mut source = source.clone();
        if !source.probability.is_finite() || !(0.0..=1.0).contains(&source.probability) {
            return Err(invalid(format!(
                "invalid mechanism probability {}",
                source.probability
            )));
        }
        if source.probability > 0.5 {
            for &d in &source.effect.detectors {
                let bit = detector_bias.get_mut(d as usize).ok_or_else(|| {
                    invalid(format!(
                        "mechanism detector {d} exceeds width {detector_width}"
                    ))
                })?;
                *bit ^= 1;
            }
            for &o in &source.effect.dem_outputs {
                observable_bias.toggle(o as usize);
            }
            source.probability = 1.0 - source.probability;
        }
        if source.probability > 0.0 {
            normalised.push_contribution(source);
        }
    }
    Ok(ShotDem {
        dem: normalised,
        detector_bias,
        observable_bias,
        assignments: Vec::new(),
        approximations: BTreeSet::new(),
        pruned_mass_upper_bound: 0.0,
    })
}

/// Per-catalog index, borrowed immutably so the index cannot become stale.
/// Build once and reuse for all flag patterns in a run. Flags are partitioned
/// into regions by overlapping parts, shared positive-weight candidate draws,
/// and physical incompatibilities. Covers stream through a capped visitor in
/// each region; independent region products are never materialised. Search
/// is bounded by the required per-region node budget. Structurally infeasible
/// remaining probe sets are memoised; candidate-dependent failures are not.
/// Draw masses are validated once at construction. Lazy pair caches are reused
/// across covers and shots and can grow quadratically in parts and branches.
#[derive(Debug)]
pub struct ShotDemBuilder<'a> {
    catalog: &'a LossCatalog,
    parts: Vec<&'a BTreeSet<ProbeId>>,
    by_probe: Vec<Vec<usize>>,
    nonflagging: BTreeMap<Draw, f64>,
    compatible_cache: RefCell<BTreeMap<(BranchId, BranchId), bool>>,
    coupled_cache: RefCell<BTreeMap<(usize, usize), bool>>,
}

#[cfg(test)]
thread_local! {
    pub(super) static PAIR_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static DRAW_MASS_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static SEARCH_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static COVER_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

type Component = BTreeSet<usize>;
type Draw = (
    super::RootKind,
    Option<usize>,
    super::Location,
    super::RootPhase,
);

struct Search<'a> {
    probes: &'a BTreeSet<ProbeId>,
    visited: usize,
    max_nodes: usize,
    max_assignments: usize,
}

impl Search<'_> {
    fn visit(&mut self) -> Result<(), AssignmentError> {
        #[cfg(test)]
        SEARCH_VISITS.with(|count| count.set(count.get() + 1));
        if self.visited >= self.max_nodes {
            return Err(AssignmentError::SearchNodeLimit {
                probes: self.probes.clone(),
                visited: self.visited as u128 + 1,
                max: self.max_nodes,
            });
        }
        self.visited += 1;
        Ok(())
    }
}

impl<'a> ShotDemBuilder<'a> {
    /// Index flag keys and aggregate draw masses once for all subsequent shots.
    ///
    /// # Errors
    /// Rejects nonfinite, negative, or at-least-one flagging mass by draw.
    pub fn new(catalog: &'a LossCatalog) -> Result<Self, LossDemError> {
        let parts: Vec<_> = catalog
            .by_flags
            .keys()
            .filter(|key| !key.is_empty())
            .collect();
        let mut by_probe = vec![Vec::new(); catalog.outputs.probes.len()];
        for (index, part) in parts.iter().enumerate() {
            for probe in *part {
                if let Some(keys) = by_probe.get_mut(probe.0) {
                    keys.push(index);
                }
            }
        }
        Ok(Self {
            catalog,
            parts,
            by_probe,
            nonflagging: Self::nonflagging_mass(catalog)?,
            compatible_cache: RefCell::new(BTreeMap::new()),
            coupled_cache: RefCell::new(BTreeMap::new()),
        })
    }

    fn draw(&self, id: BranchId) -> Draw {
        let key = &self.catalog.roots[id.root].key;
        (key.kind, key.channel, key.location, key.phase)
    }

    fn nonflagging_mass(catalog: &LossCatalog) -> Result<BTreeMap<Draw, f64>, LossDemError> {
        #[cfg(test)]
        DRAW_MASS_EVALUATIONS.with(|c| c.set(c.get() + 1));
        let mut flagging = BTreeMap::<Draw, f64>::new();
        for entry in &catalog.roots {
            for leaf in &entry.branches {
                if !leaf.flags.is_empty() {
                    *flagging
                        .entry((
                            entry.key.kind,
                            entry.key.channel,
                            entry.key.location,
                            entry.key.phase,
                        ))
                        .or_default() += leaf.weight;
                }
            }
        }
        flagging
            .into_iter()
            .map(|(draw, mass)| {
                if !mass.is_finite() || !(0.0..1.0).contains(&mass) {
                    Err(LossDemError::InvalidDrawMass {
                        draw: format!("{draw:?}"),
                        mass,
                    })
                } else {
                    Ok((draw, 1.0 - mass))
                }
            })
            .collect()
    }

    // Histories are in Program action order. Only changes strictly BEFORE the
    // target root can suppress its draw. Events require both operands present,
    // including an I/Z partner, not just the operand labeled L.
    fn blocks(&self, branch: BranchId, root: usize) -> bool {
        let target = &self.catalog.roots[root];
        let history = &self.catalog.roots[branch.root].branches[branch.branch].lost_over_time;
        let before = history.partition_point(|moment| moment.action_index < target.action_index);
        before.checked_sub(1).is_some_and(|index| {
            target
                .qubits
                .iter()
                .any(|q| history[index].lost.contains(q))
        })
    }

    fn compatible(&self, left: BranchId, right: BranchId) -> bool {
        let key = (left.min(right), left.max(right));
        if let Some(&value) = self.compatible_cache.borrow().get(&key) {
            return value;
        }
        #[cfg(test)]
        PAIR_EVALUATIONS.with(|c| c.set(c.get() + 1));
        let value = !self.blocks(left, right.root) && !self.blocks(right, left.root);
        self.compatible_cache.borrow_mut().insert(key, value);
        value
    }

    fn candidates(&self, part: usize) -> impl Iterator<Item = BranchId> + '_ {
        self.catalog.by_flags[self.parts[part]]
            .iter()
            .copied()
            .filter(|id| self.catalog.roots[id.root].branches[id.branch].weight > 0.0)
    }

    fn coupled(&self, left: usize, right: usize) -> bool {
        let key = (left.min(right), left.max(right));
        if let Some(&value) = self.coupled_cache.borrow().get(&key) {
            return value;
        }
        #[cfg(test)]
        PAIR_EVALUATIONS.with(|c| c.set(c.get() + 1));
        let value = self.candidates(left).any(|a| {
            self.candidates(right)
                .any(|b| self.draw(a) == self.draw(b) || !self.compatible(a, b))
        });
        self.coupled_cache.borrow_mut().insert(key, value);
        value
    }

    // Physical exclusion, like shared draws, is a dependency: it must not be
    // factored across independent groups or regions. Zero-weight candidates
    // never create either kind of dependency.
    fn components(&self, parts: &Component, overlap: bool) -> Vec<Component> {
        let mut result: Vec<Component> = Vec::new();
        for &part in parts {
            let mut component = BTreeSet::from([part]);
            let mut index = 0;
            while index < result.len() {
                let connected = component.iter().any(|&a| {
                    result[index].iter().any(|&b| {
                        (overlap && !self.parts[a].is_disjoint(self.parts[b])) || self.coupled(a, b)
                    })
                });
                if connected {
                    component.extend(result.remove(index));
                    index = 0;
                } else {
                    index += 1;
                }
            }
            result.push(component);
        }
        result
    }

    fn covers(
        &self,
        remaining: &BTreeSet<ProbeId>,
        chosen: &mut Component,
        infeasible: &mut BTreeSet<BTreeSet<ProbeId>>,
        search: &mut Search<'_>,
        visit: &mut impl FnMut(&Component, &mut Search<'_>) -> Result<(), AssignmentError>,
    ) -> Result<bool, AssignmentError> {
        search.visit()?;
        if infeasible.contains(remaining) {
            return Ok(false);
        }
        let Some(probe) = remaining.first() else {
            #[cfg(test)]
            COVER_VISITS.with(|count| count.set(count.get() + 1));
            visit(chosen, search)?;
            return Ok(true);
        };
        let mut covered = false;
        for &part in &self.by_probe[probe.0] {
            if self.parts[part].is_subset(remaining) && self.candidates(part).next().is_some() {
                chosen.insert(part);
                covered |= self.covers(
                    &remaining.difference(self.parts[part]).copied().collect(),
                    chosen,
                    infeasible,
                    search,
                    visit,
                )?;
                chosen.remove(&part);
            }
        }
        // Only structural infeasibility is independent of the chosen prefix.
        // A complete cover rejected by candidate compatibility still returns true.
        if !covered {
            infeasible.insert(remaining.clone());
        }
        Ok(covered)
    }

    fn enumerate(
        &self,
        parts: &Component,
        chosen: &mut Vec<BranchId>,
        used: &mut BTreeSet<Draw>,
        out: &mut Vec<Vec<BranchId>>,
        search: &mut Search<'_>,
    ) -> Result<bool, AssignmentError> {
        search.visit()?;
        // Each part contains disjoint probes. Choose the lowest unexplained one.
        let Some(&part) = parts.iter().min_by_key(|&&p| self.parts[p].first()) else {
            out.push(chosen.clone());
            return Ok(out.len() <= search.max_assignments);
        };
        let remaining = parts.difference(&BTreeSet::from([part])).copied().collect();
        for &id in &self.catalog.by_flags[self.parts[part]] {
            if self.catalog.roots[id.root].branches[id.branch].weight <= 0.0 {
                continue;
            }
            let draw = self.draw(id);
            if chosen.iter().any(|&other| !self.compatible(id, other)) {
                continue;
            }
            if used.insert(draw) {
                chosen.push(id);
                if !self.enumerate(&remaining, chosen, used, out, search)? {
                    return Ok(false);
                }
                chosen.pop();
                used.remove(&draw);
            }
        }
        Ok(true)
    }

    /// Enumerate valid distinct-draw assignments, factored into independent groups.
    /// The empty flag set has exactly one empty assignment.
    ///
    /// # Errors
    /// Invalid probes, no assignment, or exceeded assignment/search-node bounds.
    pub fn assignments(
        &self,
        flags: &BTreeSet<ProbeId>,
        max_assignments: usize,
        max_search_nodes: usize,
    ) -> Result<Vec<Vec<Vec<BranchId>>>, AssignmentError> {
        if flags.iter().any(|p| p.0 >= self.by_probe.len()) {
            return Err(AssignmentError::InvalidProbe {
                probes: flags.clone(),
                probe_count: self.by_probe.len(),
            });
        }
        if flags.is_empty() {
            return self.region_assignments(flags, max_assignments, max_search_nodes);
        }
        let parts: Component = flags
            .iter()
            .flat_map(|p| &self.by_probe[p.0])
            .copied()
            .filter(|&part| {
                self.parts[part].is_subset(flags) && self.candidates(part).next().is_some()
            })
            .collect();
        let covered: BTreeSet<_> = parts.iter().flat_map(|&p| self.parts[p]).copied().collect();
        if covered != *flags {
            return Err(AssignmentError::NoAssignment {
                probes: flags.clone(),
            });
        }
        let mut groups = Vec::new();
        for region in self.components(&parts, true) {
            let probes = region
                .iter()
                .flat_map(|&p| self.parts[p])
                .copied()
                .collect();
            groups.extend(self.region_assignments(&probes, max_assignments, max_search_nodes)?);
        }
        Ok(groups)
    }

    fn region_assignments(
        &self,
        flags: &BTreeSet<ProbeId>,
        max_assignments: usize,
        max_search_nodes: usize,
    ) -> Result<Vec<Vec<Vec<BranchId>>>, AssignmentError> {
        let too_many = || AssignmentError::TooManyAssignments {
            probes: flags.clone(),
            max: max_assignments,
        };
        let mut cache = BTreeMap::new();
        let mut valid = Vec::new();
        let mut visit = |cover: &Component, search: &mut Search<'_>| {
            let components = self.components(cover, false);
            let mut possible = true;
            for component in &components {
                if !cache.contains_key(component) {
                    let mut assignments = Vec::new();
                    self.enumerate(
                        component,
                        &mut Vec::new(),
                        &mut BTreeSet::new(),
                        &mut assignments,
                        search,
                    )?;
                    cache.insert(component.clone(), assignments);
                }
                possible &= !cache[component].is_empty();
            }
            if possible {
                if components.iter().any(|c| cache[c].len() > max_assignments) {
                    return Err(too_many());
                }
                valid.push(components);
                if valid.len() > max_assignments {
                    return Err(too_many());
                }
            }
            Ok(())
        };
        let mut search = Search {
            probes: flags,
            visited: 0,
            max_nodes: max_search_nodes,
            max_assignments,
        };
        self.covers(
            flags,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut search,
            &mut visit,
        )?;
        let Some(first) = valid.first() else {
            return Err(AssignmentError::NoAssignment {
                probes: flags.clone(),
            });
        };
        let common: Vec<_> = first
            .iter()
            .filter(|c| valid.iter().all(|cover| cover.contains(c)))
            .cloned()
            .collect();
        let mut groups: Vec<_> = common.iter().map(|c| cache[c].clone()).collect();
        let mut flattened = Vec::new();
        for cover in &valid {
            let mut remainder = vec![Vec::new()];
            for component in cover.iter().filter(|c| !common.contains(c)) {
                let mut next = Vec::new();
                for prefix in &remainder {
                    for alternative in &cache[component] {
                        let mut assignment = prefix.clone();
                        assignment.extend(alternative);
                        next.push(assignment);
                        if next.len() > max_assignments {
                            return Err(too_many());
                        }
                    }
                }
                remainder = next;
            }
            flattened.extend(remainder);
            if flattened.len() > max_assignments {
                return Err(too_many());
            }
        }
        if valid.len() > 1 || common.is_empty() {
            groups.push(flattened);
        }
        Ok(groups)
    }

    fn alternative(&self, branches: Vec<BranchId>) -> Result<ShotAlternative, LossDemError> {
        let mut weight = 1.0;
        let mut constant = BitSet::new();
        let mut vectors = Vec::new();
        let mut removals = BTreeSet::new();
        for id in &branches {
            let branch = &self.catalog.roots[id.root].branches[id.branch];
            weight *= branch.weight / self.nonflagging[&self.draw(*id)];
            constant ^= &branch.normal_form.constant;
            vectors.extend_from_slice(branch.normal_form.span.basis());
            removals.extend(&branch.false_alarm_removals);
        }
        let span = Span::new(
            self.catalog.num_detectors + self.catalog.num_observables,
            &vectors,
        )?;
        Ok(ShotAlternative {
            branches,
            weight,
            pi: 0.0,
            normal_form: BranchNormalForm {
                constant: reduce(&constant, &span)?,
                span,
            },
            removals,
        })
    }

    /// Construct a normalised DEM for one flag pattern.
    ///
    /// # Errors
    /// Returns assignment errors or invalid affine/background inputs.
    pub fn build(
        &self,
        flags: &BTreeSet<ProbeId>,
        limits: ShotLimits,
    ) -> Result<ShotDem, ShotBuildError> {
        let assignments =
            self.assignments(flags, limits.max_assignments, limits.max_search_nodes)?;
        let blocked_draw_odds = assignments.iter().any(|group| group.len() > 1);
        let mut groups = Vec::new();
        for group in assignments {
            let mut alternatives = group
                .into_iter()
                .map(|ids| self.alternative(ids))
                .collect::<Result<Vec<_>, _>>()?;
            let total: f64 = alternatives.iter().map(|a| a.weight).sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(invalid(format!(
                    "assignment weight total {total} for probes {flags:?} is nonpositive or nonfinite; reduce the region complexity or correct the catalog branch weights to avoid underflow/overflow"
                ))
                .into());
            }
            for alternative in &mut alternatives {
                alternative.pi = alternative.weight / total;
            }
            groups.push(alternatives);
        }
        let removals = groups
            .iter()
            .flatten()
            .flat_map(|a| &a.removals)
            .copied()
            .collect();
        let background = self.catalog.shot_background(&removals)?;
        let mut result = mix_shot_dem(&groups, &background, limits.max_correlated_rank)?;
        result
            .approximations
            .extend(self.catalog.report.approximated.values().flatten().copied());
        if blocked_draw_odds {
            result.approximations.insert(Approximation::BlockedDrawOdds);
        }
        result
            .approximations
            .insert(Approximation::UnflaggedBackground);
        result.pruned_mass_upper_bound = self
            .catalog
            .report
            .dropped_mass_by_flags
            .iter()
            .filter(|(key, _)| key.is_subset(flags))
            .map(|(_, mass)| mass)
            .sum();
        Ok(result)
    }
}
