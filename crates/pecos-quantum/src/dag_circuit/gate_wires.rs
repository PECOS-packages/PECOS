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

//! Per-gate index of labeled wires by qubit.

use std::collections::BTreeMap;

use pecos_core::QubitId;
use smallvec::SmallVec;

/// Lists longer than this move from the inline vector to the map.
const INLINE_LIMIT: usize = 16;

/// The labeled wires at one end of a gate (all into it, or all out of it),
/// looked up by qubit.
///
/// Within a qubit, wires are stored oldest to newest and read newest first,
/// the graph's own adjacency order, so removing the newest shifts nothing.
/// A short list is a vector sorted by qubit, so ordinary gates allocate
/// nothing. A longer one is a map from qubit to that qubit's wires, so a wide
/// gate inserts and removes in O(log W) whatever order its qubits come in.
#[derive(Debug, Clone)]
pub(super) enum GateWires {
    Inline(SmallVec<[(QubitId, usize); 2]>),
    Map(BTreeMap<QubitId, SmallVec<[usize; 1]>>),
}

impl Default for GateWires {
    fn default() -> Self {
        Self::Inline(SmallVec::new())
    }
}

impl GateWires {
    /// Add `edge` as the newest wire on `qubit`.
    pub(super) fn insert(&mut self, qubit: QubitId, edge: usize) {
        match self {
            Self::Inline(wires) => {
                let end = wires.partition_point(|&(q, _)| q <= qubit);
                wires.insert(end, (qubit, edge));
                if wires.len() > INLINE_LIMIT {
                    let mut map: BTreeMap<QubitId, SmallVec<[usize; 1]>> = BTreeMap::new();
                    for &(q, e) in wires.iter() {
                        map.entry(q).or_default().push(e);
                    }
                    *self = Self::Map(map);
                }
            }
            Self::Map(map) => map.entry(qubit).or_default().push(edge),
        }
    }

    /// Remove `edge` from `qubit`'s wires. Panics if it is not there.
    pub(super) fn remove(&mut self, qubit: QubitId, edge: usize) {
        match self {
            Self::Inline(wires) => {
                let end = wires.partition_point(|&(q, _)| q <= qubit);
                let at = wires[..end]
                    .iter()
                    .rposition(|&wire| wire == (qubit, edge))
                    .expect("indexed wire");
                wires.remove(at);
            }
            Self::Map(map) => {
                let edges = map.get_mut(&qubit).expect("indexed qubit");
                let at = edges
                    .iter()
                    .rposition(|&wire| wire == edge)
                    .expect("indexed wire");
                edges.remove(at);
                if edges.is_empty() {
                    map.remove(&qubit);
                }
            }
        }
    }

    /// The wires on `qubit`, newest first.
    pub(super) fn on_qubit(&self, qubit: QubitId) -> OnQubit<'_> {
        match self {
            Self::Inline(wires) => {
                let start = wires.partition_point(|&(q, _)| q < qubit);
                let end = start + wires[start..].partition_point(|&(q, _)| q == qubit);
                OnQubit::Inline(wires[start..end].iter().rev())
            }
            Self::Map(map) => {
                OnQubit::Map(map.get(&qubit).map_or(&[][..], |edges| edges).iter().rev())
            }
        }
    }

    /// Calls `f` with every wire as `(qubit, edge)`, by qubit and newest first
    /// within a qubit.
    pub(super) fn for_each(&self, mut f: impl FnMut(QubitId, usize)) {
        match self {
            Self::Inline(wires) => {
                for run in wires.chunk_by(|a, b| a.0 == b.0) {
                    for &(qubit, edge) in run.iter().rev() {
                        f(qubit, edge);
                    }
                }
            }
            Self::Map(map) => {
                for (&qubit, edges) in map {
                    for &edge in edges.iter().rev() {
                        f(qubit, edge);
                    }
                }
            }
        }
    }
}

/// Iterator over one qubit's wires, newest first. See [`GateWires::on_qubit`].
pub(super) enum OnQubit<'a> {
    Inline(std::iter::Rev<std::slice::Iter<'a, (QubitId, usize)>>),
    Map(std::iter::Rev<std::slice::Iter<'a, usize>>),
}

impl OnQubit<'_> {
    /// No wires, for a gate slot beyond the index.
    pub(super) fn empty() -> Self {
        Self::Map([].iter().rev())
    }
}

impl Iterator for OnQubit<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Inline(wires) => wires.next().map(|&(_, edge)| edge),
            Self::Map(edges) => edges.next().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_random::{PecosRng, RngExt};

    /// Compare against a plain list of `(qubit, edge)` in insertion order,
    /// across the switch from the inline vector to the map and back to empty.
    #[test]
    fn matches_a_plain_list_across_the_inline_limit() {
        for seed in [0, 7, 99] {
            let mut rng = PecosRng::seed_from_u64(seed);
            let mut wires = GateWires::default();
            let mut model: Vec<(QubitId, usize)> = Vec::new();
            let mut next_edge = 0;
            for step in 0..2000 {
                // Grow past the limit, then drain, so both forms are exercised.
                let grow = (step / 250) % 2 == 0;
                if grow || model.is_empty() {
                    let qubit = QubitId(rng.random_range(0..6));
                    wires.insert(qubit, next_edge);
                    model.push((qubit, next_edge));
                    next_edge += 1;
                } else {
                    let (qubit, edge) = model.remove(rng.random_range(0..model.len()));
                    wires.remove(qubit, edge);
                }
                for q in 0..6 {
                    let expected: Vec<usize> = model
                        .iter()
                        .rev()
                        .filter(|&&(qubit, _)| qubit == QubitId(q))
                        .map(|&(_, edge)| edge)
                        .collect();
                    assert_eq!(wires.on_qubit(QubitId(q)).collect::<Vec<_>>(), expected);
                }
                let mut all = Vec::new();
                wires.for_each(|qubit, edge| all.push((qubit, edge)));
                let mut expected: Vec<_> = model.iter().rev().copied().collect();
                expected.sort_by_key(|&(qubit, _)| qubit);
                assert_eq!(all, expected);
            }
            assert!(matches!(wires, GateWires::Map(_)));
        }
    }
}
