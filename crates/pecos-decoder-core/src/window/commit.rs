// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Component provenance and checked graphs for whole-component window commits.

use super::{StructuredDem, StructuredDemComponent, StructuredDemError, invalid};
use crate::dem::grammar::xor_indices;
use crate::{DecoderError, DemMatchingGraph, MatchingEdge};
use std::collections::BTreeMap;
use std::ops::Range;

/// A global component-column. Its index is its position in flattened error order.
#[derive(Clone, Debug, PartialEq)]
pub struct CommitColumn {
    /// Parent error index in the source model.
    pub error_index: usize,
    /// Component index within the parent error.
    pub component_index: usize,
    /// Full global detector incidence, including projected detectors.
    pub detectors: Vec<u32>,
    /// Observable incidence.
    pub observables: Vec<u32>,
    /// Parent error probability, used to select representatives.
    pub probability: f64,
    /// Minimum detector time; absent for a detector-free component.
    pub owner: Option<u32>,
    /// Maximum detector time.
    pub last: u32,
}

/// One global column represented by a local edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitMember {
    /// Global flattened column index.
    pub column: usize,
    /// Whether a far detector was removed from this column locally.
    pub projected: bool,
}

/// A merged local edge with fixed global representatives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitEdge {
    /// First local endpoint.
    pub node1: u32,
    /// Second local endpoint, absent for a real or projected boundary.
    pub node2: Option<u32>,
    /// All columns merged into this edge.
    pub members: Vec<CommitMember>,
    /// True if any member is projected.
    pub future: bool,
    /// Highest-probability non-projected member, ties by global index.
    pub rep_nonprojected: Option<usize>,
    /// Highest-probability member, ties by global index.
    pub rep_any: usize,
}

/// Checked window model and provenance, in matching-graph edge order.
#[derive(Clone, Debug)]
pub struct CommitWindow {
    /// Surviving columns, still grouped by parent error for correlated backends.
    pub model: StructuredDem,
    /// Local detector index to global detector index.
    pub local_to_global_detector: Vec<u32>,
    /// Local edges in the order used to construct the backend graph.
    pub edges: Vec<CommitEdge>,
    /// Whether each local row is in the commit region.
    pub is_commit_row: Vec<bool>,
    graph: DemMatchingGraph,
}

impl CommitWindow {
    /// The graph built alongside this window's edge provenance.
    #[must_use]
    pub fn matching_graph(&self) -> &DemMatchingGraph {
        &self.graph
    }

    /// Endpoint lookup for backends that return pairs instead of edge indices.
    /// Build this mapping before constructing the backend, never from its output graph.
    #[must_use]
    pub fn edge_indices(&self) -> BTreeMap<(u32, Option<u32>), usize> {
        self.edges
            .iter()
            .enumerate()
            .map(|(index, edge)| ((edge.node1, edge.node2), index))
            .collect()
    }
}

impl StructuredDem {
    /// Validate raw third coordinates and return integral detector times.
    pub fn commit_detector_times(&self) -> Result<Vec<u32>, DecoderError> {
        if self.detector_coords.len() != self.num_detectors {
            return Err(invalid(
                "detector coordinate count does not match detector count",
            ));
        }
        self.detector_coords.iter().enumerate().map(|(row, coords)| {
            let time = coords.as_ref().and_then(|values| values.get(2)).copied();
            match time {
                Some(t) if t.is_finite() && t >= 0.0 && t.fract() == 0.0 && t < f64::from(u32::MAX) => {
                    u32::try_from(t as i64).map_err(|_| invalid(format!("detector {row} time is out of range")))
                }
                _ => Err(invalid(format!("detector {row} needs a non-negative integer third time coordinate below u32::MAX"))),
            }
        }).collect()
    }

    /// Index all global columns once, retaining detector-free positions in the index.
    pub fn commit_columns(&self) -> Result<Vec<CommitColumn>, DecoderError> {
        let times = self.commit_detector_times()?;
        let mut columns = Vec::new();
        for (error_index, error) in self.errors.iter().enumerate() {
            if !error.probability.is_finite() || !(0.0..=1.0).contains(&error.probability) {
                return Err(invalid(format!(
                    "error {error_index} has an invalid probability"
                )));
            }
            for (component_index, part) in error.components.iter().enumerate() {
                let detectors = xor_indices(part.detectors.iter().copied());
                let observables = xor_indices(part.observables.iter().copied());
                if detectors.len() > 2 {
                    return Err(invalid(format!(
                        "error {error_index} component {component_index} has {} detectors; commit windows require graphlike columns",
                        detectors.len()
                    )));
                }
                if detectors.iter().any(|&d| d as usize >= self.num_detectors)
                    || observables
                        .iter()
                        .any(|&o| o as usize >= self.num_observables)
                {
                    return Err(invalid(format!(
                        "error {error_index} component {component_index} has an out-of-range target"
                    )));
                }
                let owner = detectors.iter().map(|&d| times[d as usize]).min();
                let last = detectors
                    .iter()
                    .map(|&d| times[d as usize])
                    .max()
                    .unwrap_or(0);
                columns.push(CommitColumn {
                    error_index,
                    component_index,
                    detectors,
                    observables,
                    probability: error.probability,
                    owner,
                    last,
                });
            }
        }
        Ok(columns)
    }

    /// Build a complete, solvable local graph with fixed global column representatives.
    ///
    /// Bounds are detector times, with a hard back boundary and projected forward boundary.
    pub fn commit_window(
        &self,
        rows: Range<u32>,
        commit: Range<u32>,
    ) -> Result<CommitWindow, DecoderError> {
        if rows.start >= rows.end
            || commit.start >= commit.end
            || commit.start < rows.start
            || commit.end > rows.end
        {
            return Err(invalid(
                "commit-window ranges must be increasing and commit must lie inside rows",
            ));
        }
        let times = self.commit_detector_times()?;
        let columns = self.commit_columns()?;
        let mut local_to_global_detector = Vec::new();
        let mut global_to_local = vec![None; self.num_detectors];
        for (global, &time) in times.iter().enumerate() {
            if rows.contains(&time) {
                let local = u32::try_from(local_to_global_detector.len())
                    .map_err(|_| invalid("too many window detectors"))?;
                global_to_local[global] = Some(local);
                local_to_global_detector
                    .push(u32::try_from(global).map_err(|_| invalid("too many global detectors"))?);
            }
        }
        let mut projected_columns = Vec::new();
        let mut graph_edges = Vec::new();
        for (index, column) in columns.iter().enumerate() {
            let Some(owner) = column.owner else { continue };
            if !rows.contains(&owner) || column.probability == 0.0 {
                continue;
            }
            if owner < commit.end && column.last >= rows.end {
                return Err(invalid(format!(
                    "completeness error: column {index} (error {}) crosses the forward boundary",
                    column.error_index
                )));
            }
            let local: Vec<_> = column
                .detectors
                .iter()
                .filter_map(|&d| global_to_local[d as usize])
                .collect();
            let key = (local[0], local.get(1).copied());
            projected_columns.push((index, local));
            graph_edges.push(MatchingEdge {
                node1: key.0,
                node2: key.1,
                probability: column.probability,
                weight: if column.probability < 1.0 {
                    ((1.0 - column.probability) / column.probability).ln()
                } else {
                    0.0
                },
                observables: column.observables.clone(),
                fault_id: column.error_index,
            });
        }
        let graph_edges = DemMatchingGraph::fold_correlated_edges(graph_edges);
        let surviving: std::collections::BTreeSet<_> = graph_edges
            .iter()
            .map(|edge| (edge.fault_id, edge.node1, edge.node2))
            .collect();
        let mut groups: BTreeMap<usize, Vec<StructuredDemComponent>> = BTreeMap::new();
        let mut members: BTreeMap<(u32, Option<u32>), Vec<CommitMember>> = BTreeMap::new();
        for (index, local) in projected_columns {
            let column = &columns[index];
            let key = (local[0], local.get(1).copied());
            if !surviving.contains(&(column.error_index, key.0, key.1)) {
                continue;
            }
            members.entry(key).or_default().push(CommitMember {
                column: index,
                projected: local.len() != column.detectors.len(),
            });
            groups
                .entry(column.error_index)
                .or_default()
                .push(StructuredDemComponent {
                    detectors: local,
                    observables: column.observables.clone(),
                });
        }
        let graph_edges = DemMatchingGraph::merge_independent_edges(graph_edges);
        let edges = graph_edges
            .iter()
            .map(|edge| {
                let mem = members
                    .remove(&(edge.node1, edge.node2))
                    .expect("provenance was built with graph edges");
                let mut ranked: Vec<_> = mem.iter().map(|m| m.column).collect();
                ranked.sort_by(|&a, &b| {
                    columns[b]
                        .probability
                        .total_cmp(&columns[a].probability)
                        .then(a.cmp(&b))
                });
                let rep_any = ranked[0];
                let rep_nonprojected = ranked
                    .into_iter()
                    .find(|&id| mem.iter().any(|m| m.column == id && !m.projected));
                if let Some(rep) = rep_nonprojected {
                    for other in mem.iter().filter(|m| !m.projected) {
                        if columns[rep].observables != columns[other.column].observables {
                            return Err(invalid(format!(
                                "non-projected columns {rep} and {} disagree on observables; \
                                 a matching edge carries one observable label, so the two mechanisms \
                                 cannot both be represented. This usually comes from a graphlike \
                                 decomposition that placed observables inconsistently; terminal \
                                 graphlike decomposition is a known source (issue #780). Source \
                                 graphlike decomposition avoids it.",
                                other.column
                            )));
                        }
                    }
                }
                Ok(CommitEdge {
                    node1: edge.node1,
                    node2: edge.node2,
                    future: mem.iter().any(|m| m.projected),
                    members: mem,
                    rep_nonprojected,
                    rep_any,
                })
            })
            .collect::<Result<Vec<_>, DecoderError>>()?;
        check_solvable(local_to_global_detector.len(), &edges)?;
        let model = StructuredDem {
            errors: groups
                .into_iter()
                .map(|(index, components)| StructuredDemError {
                    probability: self.errors[index].probability,
                    components,
                })
                .collect(),
            detector_coords: local_to_global_detector
                .iter()
                .map(|&d| self.detector_coords[d as usize].clone())
                .collect(),
            num_detectors: local_to_global_detector.len(),
            num_observables: self.num_observables,
        };
        let graph = DemMatchingGraph {
            edges: graph_edges,
            num_detectors: model.num_detectors,
            num_observables: model.num_observables,
            skipped_hyperedges: 0,
            detector_coords: model.detector_coords.clone(),
        };
        let is_commit_row = local_to_global_detector
            .iter()
            .map(|&d| commit.contains(&times[d as usize]))
            .collect();
        Ok(CommitWindow {
            model,
            local_to_global_detector,
            edges,
            is_commit_row,
            graph,
        })
    }
}

/// Smallest buffer that guarantees complete commit-relevant columns.
pub fn min_buffer_rounds(dem: &StructuredDem) -> Result<u32, DecoderError> {
    Ok(dem
        .commit_columns()?
        .iter()
        .filter(|c| c.probability != 0.0)
        .filter_map(|c| c.owner.map(|owner| c.last - owner))
        .max()
        .unwrap_or(0))
}

fn check_solvable(rows: usize, edges: &[CommitEdge]) -> Result<(), DecoderError> {
    let mut adjacency = vec![Vec::new(); rows];
    let mut reachable = vec![false; rows];
    let mut stack = Vec::new();
    for edge in edges {
        if let Some(node2) = edge.node2 {
            adjacency[edge.node1 as usize].push(node2 as usize);
            adjacency[node2 as usize].push(edge.node1 as usize);
        } else {
            reachable[edge.node1 as usize] = true;
            stack.push(edge.node1 as usize);
        }
    }
    while let Some(row) = stack.pop() {
        for &next in &adjacency[row] {
            if !reachable[next] {
                reachable[next] = true;
                stack.push(next);
            }
        }
    }
    if let Some(row) = reachable.iter().position(|&r| !r) {
        return Err(invalid(format!(
            "window solvability: row {row} has no path to a boundary"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_time_coordinates_are_required() {
        for coords in ["", "(0,0)", "(0,0,-1)", "(0,0,0.5)", "(0,0,4294967295)"] {
            let dem = StructuredDem::from_dem_str(&format!("error(0.1) D0\ndetector{coords} D0\n"))
                .unwrap();
            let error = dem.commit_window(0..2, 0..1).unwrap_err();
            assert!(error.to_string().contains("detector 0"));
        }
    }

    #[test]
    fn representatives_groups_and_backend_order_are_fixed_at_construction() {
        let dem = StructuredDem::from_dem_str("error(0.1) D0 ^ D1 D2 L0\nerror(0.3) D1 D2 L0\nerror(0.3) D1 D2 L0\nerror(0.01) D1\nerror(0.01) D2\ndetector(0,0,0) D0\ndetector(0,0,1) D1\ndetector(0,0,2) D2\nlogical_observable L0\n").unwrap();
        let window = dem.commit_window(0..2, 0..1).unwrap();
        assert_eq!(window.model.errors[0].components.len(), 2);
        assert_eq!(window.local_to_global_detector, [0, 1]);
        assert_eq!(window.is_commit_row, [true, false]);
        let edge = &window.edges[1];
        assert!(edge.future);
        assert_eq!(
            edge.rep_any, 2,
            "highest parent probability, then lowest global column index"
        );
        assert_eq!(edge.rep_nonprojected, Some(4));
        assert_eq!(
            edge.members.iter().map(|m| m.column).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        let graph = DemMatchingGraph::from_dem_str(&window.model.to_dem_string()).unwrap();
        for (edge, backend) in window.edges.iter().zip(graph.edges) {
            assert_eq!((edge.node1, edge.node2), (backend.node1, backend.node2));
        }
        let later = dem.commit_window(1..3, 1..3).unwrap();
        assert_eq!(
            later.model.errors[0].components.len(),
            1,
            "drop old columns individually, preserving their surviving sibling"
        );
        assert_eq!(later.edges[1].rep_nonprojected, Some(2));
    }

    #[test]
    fn cancelled_projected_columns_leave_neither_groups_nor_members() {
        for targets in ["D2 D5 ^ D2 D6", "D2 D3 ^ D3 D4 ^ D2 D4"] {
            let dem = StructuredDem::from_dem_str(&format!(
                "error(0.4) {targets}\nerror(0.1) D2\nerror(0.1) D3\nerror(0.1) D4\n\
                 detector(0,0,2) D0\ndetector(0,0,2) D1\ndetector(0,0,1) D2\n\
                 detector(0,0,1) D3\ndetector(0,0,1) D4\ndetector(0,0,2) D5\n\
                 detector(0,0,2) D6\n"
            ))
            .unwrap();
            let window = dem.commit_window(0..2, 0..1).unwrap();
            assert_eq!(window.local_to_global_detector, [2, 3, 4]);
            assert_eq!(
                window.model.errors,
                dem.errors[1..]
                    .iter()
                    .enumerate()
                    .map(|(local, error)| {
                        StructuredDemError {
                            probability: error.probability,
                            components: vec![StructuredDemComponent {
                                detectors: vec![u32::try_from(local).unwrap()],
                                observables: vec![],
                            }],
                        }
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(window.edges.len(), 3);
            let first_survivor = dem.errors[0].components.len();
            for (index, edge) in window.edges.iter().enumerate() {
                assert_eq!(
                    (edge.node1, edge.node2),
                    (u32::try_from(index).unwrap(), None)
                );
                assert_eq!(edge.members.len(), 1);
                assert_eq!(edge.members[0].column, first_survivor + index);
                assert!(!edge.future);
                assert_eq!(edge.rep_any, first_survivor + index);
                assert_eq!(edge.rep_nonprojected, Some(first_survivor + index));
            }
            let reparsed = DemMatchingGraph::from_dem_str(&window.model.to_dem_string()).unwrap();
            assert_eq!(reparsed.edges.len(), window.graph.edges.len());
            for (actual, expected) in reparsed.edges.iter().zip(&window.graph.edges) {
                assert_eq!(
                    (actual.node1, actual.node2),
                    (expected.node1, expected.node2)
                );
                assert!((actual.probability - expected.probability).abs() < f64::EPSILON);
                assert_eq!(actual.observables, expected.observables);
            }
        }
    }

    #[test]
    fn nonprojected_observable_disagreement_names_both_columns() {
        let dem =
            StructuredDem::from_dem_str("error(0.1) D0\nerror(0.2) D0 L0\ndetector(0,0,0) D0\n")
                .unwrap();
        let error = dem.commit_window(0..1, 0..1).unwrap_err().to_string();
        assert!(error.contains("columns 1 and 0"), "{error}");
        assert!(
            error.contains("a matching edge carries one observable label"),
            "{error}"
        );
        assert!(error.contains("cannot both be represented"), "{error}");
        assert!(
            error.contains("terminal graphlike decomposition"),
            "{error}"
        );
        assert!(error.contains("issue #780"), "{error}");
        assert!(
            error.contains("Source graphlike decomposition avoids it"),
            "{error}"
        );
    }

    #[test]
    fn constructor_checks_completeness_even_without_engine() {
        let dem = StructuredDem::from_dem_str("error(0.1) D0 D1\nerror(0.1) D0\nerror(0.1) D1\ndetector(0,0,0) D0\ndetector(0,0,2) D1\n").unwrap();
        assert!(
            dem.commit_window(0..2, 0..1)
                .unwrap_err()
                .to_string()
                .contains("completeness")
        );
        assert!(dem.commit_window(0..3, 0..1).is_ok());
    }

    #[test]
    fn zero_probability_columns_do_not_require_buffer() {
        assert_eq!(min_buffer_rounds(&StructuredDem::from_dem_str("error(0) D0 D1\nerror(0.1) D0\nerror(0.1) D1\ndetector(0,0,0) D0\ndetector(0,0,2) D1\n").unwrap()).unwrap(), 0);
    }
}
