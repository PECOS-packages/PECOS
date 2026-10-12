//! Pure graphlike projection using only standalone mechanisms of the raw DEM.
//!
//! Disjoint detector partitions preserve each hyperedge's complete XOR effect;
//! family preference precedes likelihood so declared stabilizer families stay
//! separate when possible. This module does not change sampling or decoders.

use crate::correlation_table::bernoulli_xor;
use crate::dem::grammar::xor_indices;
use crate::obs_mask::ObsMask;
use crate::window::{StructuredDem, StructuredDemComponent, StructuredDemError};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use thiserror::Error;

/// Exhaustive partitions are bounded explicitly, never silently truncated.
pub const MAX_PROJECTION_DETECTORS: usize = 12;

/// Stabilizer-family declarations, separate from an initialization basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorFamily {
    /// Records from X stabilizers.
    X,
    /// Records from Z stabilizers.
    Z,
    /// Records spanning both families, for example across a transversal H.
    Mixed,
}

/// Irreducible mechanisms require an explicit caller policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrreduciblePolicy {
    /// Report all irreducible lines after checking the entire input.
    Refuse,
    /// Omit irreducible lines and record their input indices.
    Drop,
}

/// Diagnostics retain input-line multiplicity, even when choices are cached.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectionReport {
    /// Input errors with more than two detector targets.
    pub input_hyperedges: usize,
    /// Hyperedges replaced by disjoint standalone components.
    pub projected: usize,
    /// Omitted input error indices, in input order.
    pub dropped_indices: Vec<usize>,
    /// X/Z pairs in all emitted components, including passthrough errors.
    pub cross_family_components: usize,
    /// Standalone detector sets with multiple distinct observable sets.
    /// Labels may be fault IDs, so this is diagnostic rather than an error.
    pub label_conflicts: usize,
    /// Detectors with no family declaration, including unused detectors.
    pub undeclared_detectors: usize,
    /// Detectors explicitly spanning both stabilizer families.
    pub mixed_detectors: usize,
}

/// Projected model and diagnostics; source dimensions and coordinates survive.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphlikeProjection {
    /// Graphlike errors in input order, excluding explicitly dropped errors.
    pub model: StructuredDem,
    /// Projection and family diagnostics.
    pub report: ProjectionReport,
}

/// Projection failures distinguish malformed input from unsupported reduction.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GraphlikeProjectionError {
    /// A model-wide mismatch has no associated error-line index.
    #[error("family declaration count {actual} does not match detector count {expected}")]
    FamiliesLength { expected: usize, actual: usize },
    /// Zero components are malformed; multiple components are already decomposed.
    #[error("error {error_index} has {count} components; projection requires exactly one")]
    ComponentCount { error_index: usize, count: usize },
    /// Endpoints and nonfinite probabilities cannot define standalone weights.
    #[error("error {error_index} has probability {probability}; expected finite 0 < p < 1")]
    InvalidProbability {
        error_index: usize,
        probability: f64,
    },
    /// Bounds are checked before any family lookup.
    #[error("error {error_index} detector D{detector} is outside detector count {num_detectors}")]
    DetectorOutOfRange {
        error_index: usize,
        detector: u32,
        num_detectors: usize,
    },
    /// Observable support must fit the declared dimension too.
    #[error(
        "error {error_index} observable L{observable} is outside observable count {num_observables}"
    )]
    ObservableOutOfRange {
        error_index: usize,
        observable: u32,
        num_observables: usize,
    },
    /// Written duplicates are rejected, never cancelled during validation.
    #[error("error {error_index} repeats detector D{detector}")]
    RepeatedDetector { error_index: usize, detector: u32 },
    /// Detector and observable target namespaces are distinct.
    #[error("error {error_index} repeats observable L{observable}")]
    RepeatedObservable { error_index: usize, observable: u32 },
    /// An exhaustive search above the advertised limit is not attempted.
    #[error(
        "error {error_index} detectors {detectors:?} exceed projection limit {MAX_PROJECTION_DETECTORS}"
    )]
    SearchBudget {
        error_index: usize,
        detectors: Vec<u32>,
    },
    /// Span membership alone is insufficient: output components must be disjoint.
    #[error(
        "error {error_index} effect ({detectors:?}, {observables:?}) is reducible only with overlapping components"
    )]
    OverlapOnly {
        error_index: usize,
        detectors: Vec<u32>,
        observables: Vec<u32>,
    },
    /// Refusal summarizes the complete scan without an unbounded diagnostic.
    #[error("{count} irreducible errors; first input indices: {first_indices:?}")]
    Irreducible {
        count: usize,
        first_indices: Vec<usize>,
    },
    /// Checked on every successful call, including release builds.
    #[error("internal projection invariant failed at error {error_index}, detectors {detectors:?}")]
    InternalInvariant {
        error_index: usize,
        detectors: Vec<u32>,
    },
}

// Tuple ordering is exactly the contract's canonical component ordering.
type Effect = (Vec<u32>, Vec<u32>);
type Standalone = BTreeMap<Vec<u32>, BTreeMap<Vec<u32>, f64>>;

/// Project raw errors using an exhaustive, cached search of disjoint partitions.
///
/// Passthrough errors retain written target order and probability bits. Only
/// existing standalone detector/observable pairs may replace hyperedges.
///
/// # Errors
/// Checks family length, then validates all error lines before building the
/// catalogue or searching. The first malformed line in input order takes
/// precedence over every reduction outcome. Once validation succeeds, returns
/// the first budget or overlap failure in input order. Under
/// [`IrreduciblePolicy::Refuse`], irreducibility is reported only after that
/// reduction scan completes.
pub fn project_graphlike(
    dem: &StructuredDem,
    families: &[Option<DetectorFamily>],
    policy: IrreduciblePolicy,
) -> Result<GraphlikeProjection, GraphlikeProjectionError> {
    if families.len() != dem.num_detectors {
        return Err(GraphlikeProjectionError::FamiliesLength {
            expected: dem.num_detectors,
            actual: families.len(),
        });
    }
    // Malformed input takes precedence over unsupported reduction. Only build
    // the complete standalone catalogue after every line passes validation.
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(index, error)| validate(dem, index, error))
        .collect::<Result<_, _>>()?;
    let standalone = standalone_mechanisms(dem, &effects);
    let mut report = ProjectionReport {
        label_conflicts: standalone
            .values()
            .filter(|labels| labels.len() > 1)
            .count(),
        undeclared_detectors: families.iter().filter(|family| family.is_none()).count(),
        mixed_detectors: families
            .iter()
            .filter(|&&family| family == Some(DetectorFamily::Mixed))
            .count(),
        ..ProjectionReport::default()
    };
    let mut cache = BTreeMap::new();
    let mut errors = Vec::with_capacity(dem.errors.len());
    let mut replacements = Vec::new();
    for (index, (input, effect)) in dem.errors.iter().zip(effects).enumerate() {
        let output = if effect.0.len() <= 2 {
            input.clone()
        } else {
            report.input_hyperedges += 1;
            if effect.0.len() > MAX_PROJECTION_DETECTORS {
                return Err(GraphlikeProjectionError::SearchBudget {
                    error_index: index,
                    detectors: effect.0,
                });
            }
            if !cache.contains_key(&effect) {
                let blocks = local_blocks(&effect.0, &standalone, families);
                let choice = choose(&effect, &blocks);
                if choice.is_none() && in_span(&effect, &blocks) {
                    return Err(GraphlikeProjectionError::OverlapOnly {
                        error_index: index,
                        detectors: effect.0,
                        observables: effect.1,
                    });
                }
                cache.insert(effect.clone(), choice);
            }
            let Some(components) = &cache[&effect] else {
                report.dropped_indices.push(index);
                continue;
            };
            report.projected += 1;
            replacements.push((index, errors.len()));
            StructuredDemError {
                probability: input.probability,
                components: components
                    .iter()
                    .map(|(detectors, observables)| StructuredDemComponent {
                        detectors: detectors.clone(),
                        observables: observables.clone(),
                    })
                    .collect(),
            }
        };
        report.cross_family_components += output
            .components
            .iter()
            .filter(|part| is_cross(&part.detectors, families))
            .count();
        errors.push(output);
    }
    if policy == IrreduciblePolicy::Refuse && !report.dropped_indices.is_empty() {
        return Err(GraphlikeProjectionError::Irreducible {
            count: report.dropped_indices.len(),
            first_indices: report.dropped_indices.iter().take(5).copied().collect(),
        });
    }
    for (input_index, output_index) in replacements {
        assert_replacement(
            input_index,
            &dem.errors[input_index],
            &errors[output_index],
            &standalone,
        )?;
    }
    Ok(GraphlikeProjection {
        model: StructuredDem {
            errors,
            detector_coords: dem.detector_coords.clone(),
            num_detectors: dem.num_detectors,
            num_observables: dem.num_observables,
        },
        report,
    })
}

fn validate(
    dem: &StructuredDem,
    error_index: usize,
    error: &StructuredDemError,
) -> Result<Effect, GraphlikeProjectionError> {
    use GraphlikeProjectionError as E;
    if error.components.len() != 1 {
        return Err(E::ComponentCount {
            error_index,
            count: error.components.len(),
        });
    }
    if !error.probability.is_finite() || error.probability <= 0.0 || error.probability >= 1.0 {
        return Err(E::InvalidProbability {
            error_index,
            probability: error.probability,
        });
    }
    let part = &error.components[0];
    for &detector in &part.detectors {
        if detector as usize >= dem.num_detectors {
            return Err(E::DetectorOutOfRange {
                error_index,
                detector,
                num_detectors: dem.num_detectors,
            });
        }
    }
    for &observable in &part.observables {
        if observable as usize >= dem.num_observables {
            return Err(E::ObservableOutOfRange {
                error_index,
                observable,
                num_observables: dem.num_observables,
            });
        }
    }
    let mut detectors = part.detectors.clone();
    let mut observables = part.observables.clone();
    detectors.sort_unstable();
    observables.sort_unstable();
    if let Some(pair) = detectors.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(E::RepeatedDetector {
            error_index,
            detector: pair[0],
        });
    }
    if let Some(pair) = observables.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(E::RepeatedObservable {
            error_index,
            observable: pair[0],
        });
    }
    Ok((detectors, observables))
}

fn standalone_mechanisms(dem: &StructuredDem, effects: &[Effect]) -> Standalone {
    let mut standalone = Standalone::new();
    for (error, (detectors, observables)) in dem.errors.iter().zip(effects) {
        if (1..=2).contains(&detectors.len()) {
            let probability = standalone
                .entry(detectors.clone())
                .or_default()
                .entry(observables.clone())
                .or_default();
            *probability = bernoulli_xor(*probability, error.probability);
        }
    }
    for labels in standalone.values_mut() {
        for probability in labels.values_mut() {
            *probability = ((1.0 - *probability) / *probability).ln();
        }
    }
    standalone
}

fn is_cross(detectors: &[u32], families: &[Option<DetectorFamily>]) -> bool {
    use DetectorFamily::{X, Z};
    detectors.len() == 2
        && matches!(
            (
                families[detectors[0] as usize],
                families[detectors[1] as usize]
            ),
            (Some(X), Some(Z)) | (Some(Z), Some(X))
        )
}

struct Block {
    mask: u16,
    effect: Effect,
    weight: f64,
    cross: usize,
}

fn local_blocks(
    detectors: &[u32],
    standalone: &Standalone,
    families: &[Option<DetectorFamily>],
) -> Vec<Vec<Block>> {
    detectors
        .iter()
        .enumerate()
        .map(|(i, &first)| {
            let mut blocks = Vec::new();
            for (j, &second) in detectors.iter().enumerate().skip(i) {
                let support = if i == j {
                    vec![first]
                } else {
                    vec![first, second]
                };
                if let Some(labels) = standalone.get(&support) {
                    for (observables, &weight) in labels {
                        blocks.push(Block {
                            mask: (1 << i) | (1 << j),
                            cross: usize::from(is_cross(&support, families)),
                            effect: (support.clone(), observables.clone()),
                            weight,
                        });
                    }
                }
            }
            blocks
        })
        .collect()
}

struct Candidate<'a> {
    cross: usize,
    weight: f64,
    components: Vec<&'a Effect>,
}

fn choose(effect: &Effect, blocks: &[Vec<Block>]) -> Option<Vec<Effect>> {
    let mut best = None;
    let mut current = Candidate {
        cross: 0,
        weight: 0.0,
        components: Vec::new(),
    };
    let search = Search::new(&effect.1, blocks);
    search.visit(
        (1 << effect.0.len()) - 1,
        &mut current,
        &mut search.target.clone(),
        &mut best,
    );
    best.map(|candidate| candidate.components.into_iter().cloned().collect())
}

struct SearchBlock<'a> {
    block: &'a Block,
    observables: ObsMask,
}

struct Search<'a> {
    blocks: Vec<Vec<SearchBlock<'a>>>,
    target: ObsMask,
    nonnegative_weights: bool,
}

impl<'a> Search<'a> {
    fn new(observables: &[u32], blocks: &'a [Vec<Block>]) -> Self {
        // Compact only the labels used by this search: even sparse u32 IDs
        // consume space proportional to the number of local labels.
        let labels: std::collections::BTreeSet<_> = observables
            .iter()
            .copied()
            .chain(
                blocks
                    .iter()
                    .flatten()
                    .flat_map(|block| block.effect.1.iter().copied()),
            )
            .collect();
        let positions: BTreeMap<_, _> = labels
            .into_iter()
            .enumerate()
            .map(|(position, label)| (label, position))
            .collect();
        let mask = |labels: &[u32]| {
            let mut mask = ObsMask::new();
            for label in labels {
                mask.set(positions[label]);
            }
            mask
        };
        Self {
            blocks: blocks
                .iter()
                .map(|blocks| {
                    blocks
                        .iter()
                        .map(|block| SearchBlock {
                            block,
                            observables: mask(&block.effect.1),
                        })
                        .collect()
                })
                .collect(),
            target: mask(observables),
            nonnegative_weights: blocks.iter().flatten().all(|block| block.weight >= 0.0),
        }
    }

    fn visit(
        &self,
        remaining: u16,
        current: &mut Candidate<'a>,
        residual: &mut ObsMask,
        best: &mut Option<Candidate<'a>>,
    ) {
        // Cross counts never decrease. Weight is monotone only when *all*
        // local weights are nonnegative. Strict inequalities preserve ties.
        if best.as_ref().is_some_and(|old| {
            current.cross > old.cross
                || (self.nonnegative_weights
                    && current.cross == old.cross
                    && current.weight > old.weight)
        }) {
            return;
        }
        if remaining == 0 {
            if !residual.is_zero() {
                return;
            }
            if best.as_ref().is_none_or(|old| {
                (current.cross, current.weight, &current.components).partial_cmp(&(
                    old.cross,
                    old.weight,
                    &old.components,
                )) == Some(Ordering::Less)
            }) {
                *best = Some(Candidate {
                    cross: current.cross,
                    weight: current.weight,
                    components: current.components.clone(),
                });
            }
            return;
        }
        // Taking the least remaining detector makes component order canonical,
        // so weight accumulation follows the required left-to-right f64 order.
        for SearchBlock { block, observables } in &self.blocks[remaining.trailing_zeros() as usize]
        {
            if remaining & block.mask == block.mask {
                let old_weight = current.weight;
                current.weight += block.weight;
                current.cross += block.cross;
                current.components.push(&block.effect);
                *residual ^= observables;
                self.visit(remaining ^ block.mask, current, residual, best);
                *residual ^= observables;
                current.components.pop();
                current.cross -= block.cross;
                current.weight = old_weight;
            }
        }
    }
}

// Tagged sparse coordinates avoid allocating storage proportional to global
// detector/observable dimensions and keep Dn and Ln in separate namespaces.
fn vector(effect: &Effect) -> Vec<(bool, u32)> {
    effect
        .0
        .iter()
        .map(|&id| (false, id))
        .chain(effect.1.iter().map(|&id| (true, id)))
        .collect()
}

fn in_span(effect: &Effect, blocks: &[Vec<Block>]) -> bool {
    let mut basis: BTreeMap<(bool, u32), Vec<(bool, u32)>> = BTreeMap::new();
    for block in blocks.iter().flatten() {
        let mut column = vector(&block.effect);
        while let Some(&pivot) = column.first() {
            if let Some(row) = basis.get(&pivot) {
                column = xor_indices(column.into_iter().chain(row.iter().copied()));
            } else {
                basis.insert(pivot, column);
                break;
            }
        }
    }
    let mut target = vector(effect);
    while let Some(pivot) = target.first() {
        let Some(row) = basis.get(pivot) else {
            return false;
        };
        target = xor_indices(target.into_iter().chain(row.iter().copied()));
    }
    true
}

fn assert_replacement(
    index: usize,
    input: &StructuredDemError,
    output: &StructuredDemError,
    standalone: &Standalone,
) -> Result<(), GraphlikeProjectionError> {
    let detectors = xor_indices(
        output
            .components
            .iter()
            .flat_map(|part| part.detectors.iter().copied()),
    );
    let observables = xor_indices(
        output
            .components
            .iter()
            .flat_map(|part| part.observables.iter().copied()),
    );
    let original = &input.components[0];
    if output.probability.to_bits() != input.probability.to_bits()
        || detectors != xor_indices(original.detectors.iter().copied())
        || observables != xor_indices(original.observables.iter().copied())
        // Combined with equal XOR support, equal cardinality excludes every
        // repeated detector across components.
        || output.components.iter().map(|part| part.detectors.len()).sum::<usize>()
            != original.detectors.len()
        || output.components.windows(2).any(|pair| {
            (&pair[0].detectors, &pair[0].observables)
                >= (&pair[1].detectors, &pair[1].observables)
        })
        || output.components.iter().any(|part| {
            !(1..=2).contains(&part.detectors.len())
                || !standalone
                    .get(&part.detectors)
                    .is_some_and(|labels| labels.contains_key(&part.observables))
        })
    {
        return Err(GraphlikeProjectionError::InternalInvariant {
            error_index: index,
            detectors: original.detectors.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
