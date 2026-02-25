// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Pauli web computation and classification.
//!
//! Implements the detection webs algorithm from Borghans' thesis, fixing a bug in
//! QuiZX's `ordered_nodes` where B-type (boundary) vertices are inadvertently included
//! in the adjacency matrix, causing panics in the `pw()` function.
//!
//! Reference: <https://www.cs.ox.ac.uk/people/aleks.kissinger/papers/borghans-thesis.pdf>
//! (pages 32-37)

use std::collections::{BTreeSet, HashMap};

use quizx::detection_webs::{Pauli, PauliWeb};
use quizx::graph::{GraphLike, VType};
use quizx::hash_graph::Graph as HashGraph;
use quizx::linalg::Mat2;

use crate::ZxGraph;

/// Result of Pauli web computation.
#[derive(Debug, Clone)]
pub struct PauliWebResult {
    /// The computed Pauli webs.
    pub webs: Vec<PauliWeb>,
    /// Vertex IDs of input boundary (B-type) vertices.
    pub input_ids: Vec<usize>,
    /// Vertex IDs of output boundary (B-type) vertices.
    pub output_ids: Vec<usize>,
}

/// Classification of a Pauli web.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebClassification {
    /// No boundary legs -- a detector.
    Detector,
    /// Has only input boundary legs -- an input stabilizer.
    InputStabilizer,
    /// Has only output boundary legs -- an output stabilizer.
    OutputStabilizer,
    /// Has both input and output boundary legs -- a propagated operator.
    Propagated,
}

/// A classified detector extracted from a Pauli web.
#[derive(Debug, Clone)]
pub struct Detector {
    /// Index of this detector in the web list.
    pub index: usize,
    /// The underlying Pauli web.
    pub web: PauliWeb,
}

/// Compute Pauli webs for a ZX graph.
///
/// Returns both detection webs (detectors) and propagated operators (observables).
/// Detection webs are found using a constrained nullspace computation that forces
/// boundary-adjacent spider variables to zero. Propagated operators are found from
/// the unconstrained nullspace -- those webs whose edges span both input and output
/// boundaries.
///
/// This is a reimplementation of QuiZX's `detection_webs()` that fixes a bug where
/// B-type (boundary) vertices were included in the node ordering, causing panics.
///
/// **Limitation**: Only works for Clifford diagrams (phases that are multiples of pi).
#[must_use]
pub fn compute_pauli_webs(graph: &ZxGraph) -> PauliWebResult {
    let input_ids = graph.inputs().clone();
    let output_ids = graph.outputs().clone();

    let mut hg = vec_to_hash_graph(graph);

    // Get detection webs (constrained: boundary-adjacent variables forced to zero)
    let detection_webs = compute_detection_webs(&mut hg);

    // Get all webs (unconstrained nullspace)
    let all_webs = compute_all_webs(&mut hg);

    // From the unconstrained set, extract webs that have boundary edges.
    // These are stabilizers and propagated operators that the constrained
    // computation suppresses. We skip "detector-like" webs from the unconstrained
    // set because they include redundant stabilizer products; the constrained
    // computation gives the correct independent set of detectors.
    let non_detector_webs: Vec<PauliWeb> = all_webs
        .into_iter()
        .filter(|web| has_boundary_edges(web, &input_ids, &output_ids))
        .collect();

    // Combine: detection webs first, then non-detector webs
    let mut webs = detection_webs;
    webs.extend(non_detector_webs);

    PauliWebResult {
        webs,
        input_ids,
        output_ids,
    }
}

/// Check if a web has any edges touching boundary (B-type) vertices.
fn has_boundary_edges(web: &PauliWeb, input_ids: &[usize], output_ids: &[usize]) -> bool {
    for &(from, to) in web.edge_operators.keys() {
        if input_ids.contains(&from)
            || input_ids.contains(&to)
            || output_ids.contains(&from)
            || output_ids.contains(&to)
        {
            return true;
        }
    }
    false
}

/// Compute all Pauli webs (unconstrained) on a hash graph.
///
/// Converts the graph to bipartite form, then builds a constraint matrix whose nullspace
/// gives all valid spider "firings". Unlike [`compute_detection_webs`], this does NOT
/// apply the `no_output` constraint, so the result includes detection webs, stabilizers,
/// and propagated operators (observables).
fn compute_all_webs(g: &mut HashGraph) -> Vec<PauliWeb> {
    g.make_bipartite();

    let old_inputs = g.inputs().clone();
    let old_outputs = g.outputs().clone();

    // Collect boundary-adjacent spiders (neighbors of B vertices)
    let mut boundary_adj = Vec::new();
    for v in g.vertices() {
        if g.vertex_type(v) == VType::B {
            for w in g.neighbors(v) {
                boundary_adj.push(w);
            }
        }
    }
    let outs = boundary_adj.len();

    // Temporarily redefine inputs/outputs for the matrix computation
    g.set_inputs(vec![]);
    g.set_outputs(boundary_adj);

    // Build ordered node list, EXCLUDING B vertices.
    let (nodelist, index_map) = ordered_nodes_excluding_boundaries(g);

    // Build constraint matrix: md = [mdl | adjacency]
    let big_n = g.adjacency_matrix(Some(&nodelist));
    let mdl = Mat2::id(outs).vstack(&Mat2::zeros(big_n.num_rows() - outs, outs));
    let md = mdl.hstack(&big_n);

    // Compute nullspace WITHOUT no_output constraint -- finds all Pauli webs
    let basis_vectors = md.nullspace();
    let webs = basis_vectors
        .iter()
        .map(|v| firing_to_pauli_web(&index_map, v, g))
        .collect();

    // Restore original boundary information
    g.set_inputs(old_inputs);
    g.set_outputs(old_outputs);

    webs
}

/// Compute detection webs only on a hash graph.
///
/// Converts the graph to bipartite form, then builds a constraint matrix whose nullspace
/// gives the valid spider "firings". Applies a `no_output` constraint that forces
/// boundary-adjacent spider variables to zero, restricting results to detection webs only
/// (no boundary legs).
fn compute_detection_webs(g: &mut HashGraph) -> Vec<PauliWeb> {
    g.make_bipartite();

    let old_inputs = g.inputs().clone();
    let old_outputs = g.outputs().clone();

    // Collect boundary-adjacent spiders (neighbors of B vertices)
    let mut boundary_adj = Vec::new();
    for v in g.vertices() {
        if g.vertex_type(v) == VType::B {
            for w in g.neighbors(v) {
                boundary_adj.push(w);
            }
        }
    }
    let outs = boundary_adj.len();

    // Temporarily redefine inputs/outputs for the matrix computation
    g.set_inputs(vec![]);
    g.set_outputs(boundary_adj);

    // Build ordered node list, EXCLUDING B vertices.
    // Internal (non-boundary-adjacent) spiders come first, boundary-adjacent spiders last.
    let (nodelist, index_map) = ordered_nodes_excluding_boundaries(g);

    // Build constraint matrix: md = [mdl | adjacency]
    let big_n = g.adjacency_matrix(Some(&nodelist));
    let mdl = Mat2::id(outs).vstack(&Mat2::zeros(big_n.num_rows() - outs, outs));
    let md = mdl.hstack(&big_n);

    // Add no-output constraint [I_{2*outs} | 0] to force boundary-adjacent variables to zero
    let no_output = Mat2::id(2 * outs).hstack(&Mat2::zeros(2 * outs, md.num_cols() - 2 * outs));
    let md_constrained = md.vstack(&no_output);

    // Each nullspace basis vector is a detection web
    let basis_vectors = md_constrained.nullspace();
    let webs = basis_vectors
        .iter()
        .map(|v| firing_to_pauli_web(&index_map, v, g))
        .collect();

    // Restore original boundary information
    g.set_inputs(old_inputs);
    g.set_outputs(old_outputs);

    webs
}

/// Build an ordered node list excluding B-type vertices.
///
/// Returns nodes ordered as [boundary-adjacent spiders, internal spiders], along with
/// an index map from position to vertex ID. B-type vertices are excluded entirely,
/// fixing a bug in QuiZX's `ordered_nodes` where they were inadvertently included.
///
/// Boundary-adjacent nodes must come first so the `[I_{2*outs} | 0]` no-output
/// constraint correctly forces their firing variables to zero.
fn ordered_nodes_excluding_boundaries(g: &HashGraph) -> (Vec<usize>, HashMap<usize, usize>) {
    let mut all: Vec<usize> = g.vertices().collect();
    all.sort();

    // Boundary-adjacent spiders first: in g.outputs() and not B-type
    let boundary_adj: Vec<usize> = all
        .iter()
        .filter(|&&v| g.vertex_type(v) != VType::B && g.outputs().contains(&v))
        .copied()
        .collect();

    // Internal spiders second: not in g.outputs() and not B-type
    let internal: Vec<usize> = all
        .iter()
        .filter(|&&v| g.vertex_type(v) != VType::B && !g.outputs().contains(&v))
        .copied()
        .collect();

    let mut vertices = boundary_adj;
    vertices.extend(internal);

    let index_map: HashMap<usize, usize> =
        vertices.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    (vertices, index_map)
}

/// Convert a nullspace basis vector (firing pattern) into a `PauliWeb`.
///
/// Each non-zero entry in the firing vector indicates a spider that "fires" --
/// a Z spider firing contributes X (green) edges, an X spider contributes Z (red) edges.
/// Edges with both colors become Y (blue).
fn firing_to_pauli_web(index_map: &HashMap<usize, usize>, v: &Mat2, g: &HashGraph) -> PauliWeb {
    let n_outs = g.inputs().len() + g.outputs().len();
    let mut red_edges = BTreeSet::new();
    let mut green_edges = BTreeSet::new();
    let mut web = PauliWeb::new();

    // Skip the first n_outs columns -- those are auxiliary output variables from the
    // constraint matrix, not spider firing indices.
    for col in n_outs..v.num_cols() {
        if v[(0, col)] == 1 {
            let node = *index_map
                .get(&(col - n_outs))
                .expect("Node index not found in index map");
            let node_color = g.vertex_type(node);

            for edge in g.edges() {
                if node == edge.0 || node == edge.1 {
                    match node_color {
                        VType::Z => {
                            green_edges.insert(edge);
                        }
                        VType::X => {
                            red_edges.insert(edge);
                        }
                        _ => unreachable!("unexpected vertex type in firing: {node_color:?}"),
                    }
                }
            }
        }
    }

    for e in &red_edges {
        if green_edges.contains(e) {
            web.set_edge(e.0, e.1, Pauli::Y);
        } else {
            web.set_edge(e.0, e.1, Pauli::Z);
        }
    }
    for e in &green_edges {
        if !red_edges.contains(e) {
            web.set_edge(e.0, e.1, Pauli::X);
        }
    }

    web
}

/// Extract detectors from a Pauli web result.
///
/// Detectors are Pauli webs that have no boundary legs (neither input nor output).
#[must_use]
pub fn extract_detectors(result: &PauliWebResult) -> Vec<Detector> {
    result
        .webs
        .iter()
        .enumerate()
        .filter(|(_, web)| web.edge_operators.is_empty() || is_detector(web, result))
        .map(|(i, web)| Detector {
            index: i,
            web: web.clone(),
        })
        .collect()
}

/// Classify all webs in a result.
#[must_use]
pub fn classify_webs(result: &PauliWebResult) -> Vec<WebClassification> {
    result
        .webs
        .iter()
        .map(|web| classify_single_web(web, result))
        .collect()
}

fn is_detector(web: &PauliWeb, result: &PauliWebResult) -> bool {
    classify_single_web(web, result) == WebClassification::Detector
}

fn classify_single_web(web: &PauliWeb, result: &PauliWebResult) -> WebClassification {
    let mut has_input = false;
    let mut has_output = false;

    for &(from, to) in web.edge_operators.keys() {
        if result.input_ids.contains(&from) || result.input_ids.contains(&to) {
            has_input = true;
        }
        if result.output_ids.contains(&from) || result.output_ids.contains(&to) {
            has_output = true;
        }
    }

    match (has_input, has_output) {
        (false, false) => WebClassification::Detector,
        (true, false) => WebClassification::InputStabilizer,
        (false, true) => WebClassification::OutputStabilizer,
        (true, true) => WebClassification::Propagated,
    }
}

/// Convert a `vec_graph::Graph` to a `hash_graph::Graph`.
///
/// This is needed because `detection_webs()` requires a `hash_graph::Graph`.
fn vec_to_hash_graph(vg: &ZxGraph) -> HashGraph {
    let mut hg = HashGraph::new();

    // Copy vertices
    for v in vg.vertices() {
        let d = vg.vertex_data(v).clone();
        let new_v = hg.add_vertex_with_data(d);
        // Vertex IDs should match since we add in order on fresh graph
        assert_eq!(v, new_v, "vertex ID mismatch during graph conversion");
    }

    // Copy edges
    for (s, t, ety) in vg.edges() {
        hg.add_edge_with_type(s, t, ety);
    }

    // Copy inputs/outputs
    hg.set_inputs(vg.inputs().clone());
    hg.set_outputs(vg.outputs().clone());

    hg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_web_result_construction() {
        // Create a minimal graph to test the infrastructure
        let g = ZxGraph::new();
        let result = PauliWebResult {
            webs: vec![],
            input_ids: vec![],
            output_ids: vec![],
        };
        assert_eq!(result.webs.len(), 0);
        let _ = g;
    }

    #[test]
    fn test_web_classification_empty() {
        let result = PauliWebResult {
            webs: vec![PauliWeb::new()],
            input_ids: vec![0, 1],
            output_ids: vec![2, 3],
        };
        let classifications = classify_webs(&result);
        assert_eq!(classifications[0], WebClassification::Detector);
    }
}
