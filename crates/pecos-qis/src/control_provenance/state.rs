use super::{ControlManifest, EVENT_KEY, MAX_EVENTS};
use pecos_core::errors::PecosError;
use pecos_qis_ffi_types::Operation;
use std::sync::Arc;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ControlEvent {
    pub site_id: usize,
    pub occurrence: u64,
    pub value: u64,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ControlLoop {
    pub header: usize,
    pub occurrence: u64,
    pub iteration: u64,
}
#[derive(Debug, serde::Serialize)]
pub struct ControlSnapshot {
    pub version: u8,
    pub manifest: ControlManifest,
    pub decode_site: usize,
    pub events: Vec<ControlEvent>,
    pub active_loops: Vec<ControlLoop>,
    pub runtime_shot_id: u64,
    pub first_batch_index: usize,
    pub next_batch_index: usize,
}
#[derive(Clone)]
pub(crate) struct ControlState {
    manifest: Arc<ControlManifest>,
    counts: Vec<u64>,
    events: Vec<ControlEvent>,
    block: Option<usize>,
    expected: Option<usize>,
    branch_seen: bool,
    decode_site: Option<usize>,
    loops: Vec<Option<ControlLoop>>,
    loop_counts: Vec<u64>,
}
fn invalid() -> PecosError {
    PecosError::Input("compiler control provenance sequence or interval limit invalid".into())
}
impl ControlState {
    pub fn new(manifest: ControlManifest) -> Self {
        let counts = vec![0; manifest.sites.len()];
        let loops = vec![None; manifest.loop_headers.len()];
        let loop_counts = vec![0; loops.len()];
        Self {
            manifest: Arc::new(manifest),
            counts,
            events: Vec::new(),
            block: None,
            expected: None,
            branch_seen: false,
            decode_site: None,
            loops,
            loop_counts,
        }
    }
    pub fn fresh(&self) -> Self {
        Self::new((*self.manifest).clone())
    }
    pub fn filter(&mut self, ops: &[Operation]) -> Result<Vec<Operation>, PecosError> {
        let mut filtered = Vec::with_capacity(ops.len());
        for op in ops {
            if let Operation::TraceMetadata {
                metadata,
                qubit: None,
            } = op
                && let Some(data) = metadata.get(EVENT_KEY)
            {
                if metadata.len() != 1 {
                    return Err(invalid());
                }
                let (site, value) = data.split_once(':').ok_or_else(invalid)?;
                self.record(
                    site.parse().map_err(|_| invalid())?,
                    value.parse().map_err(|_| invalid())?,
                )?;
            } else {
                filtered.push(op.clone());
            }
        }
        Ok(filtered)
    }
    fn record(&mut self, id: usize, value: u64) -> Result<(), PecosError> {
        let site = self.manifest.sites.get(id).ok_or_else(invalid)?;
        if self.events.len() >= MAX_EVENTS || (site.scalar_type == "bool" && value > 1) {
            return Err(invalid());
        }
        if site.kind == "block" {
            if value != 0 || self.decode_site.is_some() {
                return Err(invalid());
            }
            let next = site.block;
            if let Some(previous) = self.block {
                let successors = &self.manifest.successors[previous];
                if successors.len() == 2 && !self.branch_seen {
                    return Err(invalid());
                }
                if self.expected.is_some_and(|expected| expected != next)
                    || !successors.contains(&next)
                {
                    return Err(invalid());
                }
            } else if next != 0 {
                return Err(invalid());
            }
            for (index, members) in self.manifest.loop_members.iter().enumerate() {
                if !members.contains(&next) {
                    self.loops[index] = None;
                } else if self.loops[index].is_none() {
                    if next != self.manifest.loop_headers[index] {
                        return Err(invalid());
                    }
                    self.loops[index] = Some(ControlLoop {
                        header: next,
                        occurrence: self.loop_counts[index],
                        iteration: 0,
                    });
                    self.loop_counts[index] =
                        self.loop_counts[index].checked_add(1).ok_or_else(invalid)?;
                } else if next == self.manifest.loop_headers[index] {
                    let active = self.loops[index].as_mut().ok_or_else(invalid)?;
                    active.iteration = active.iteration.checked_add(1).ok_or_else(invalid)?;
                }
            }
            self.block = Some(next);
            self.expected = None;
            self.branch_seen = false;
        } else {
            if self.block != Some(site.block) {
                return Err(invalid());
            }
            if site.kind == "branch" {
                if self.branch_seen {
                    return Err(invalid());
                }
                self.expected = Some(self.manifest.successors[site.block][usize::from(value == 0)]);
                self.branch_seen = true;
            } else if site.kind == "decode" && self.decode_site.replace(id).is_some() {
                return Err(invalid());
            }
        }
        let count = self.counts.get_mut(id).ok_or_else(invalid)?;
        self.events.push(ControlEvent {
            site_id: id,
            occurrence: *count,
            value,
        });
        *count = count.checked_add(1).ok_or_else(invalid)?;
        Ok(())
    }
    pub fn take(
        &mut self,
        batches: &crate::decoder_batches::DecoderBatchPrefix,
    ) -> Result<ControlSnapshot, PecosError> {
        Ok(ControlSnapshot {
            version: 1,
            manifest: (*self.manifest).clone(),
            decode_site: self.decode_site.take().ok_or_else(invalid)?,
            events: std::mem::take(&mut self.events),
            active_loops: self.loops.iter().flatten().cloned().collect(),
            runtime_shot_id: batches.runtime_shot_id,
            first_batch_index: batches.first_batch_index,
            next_batch_index: batches.next_batch_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{control_provenance::ControlSite, decoder_batches::DecoderBatchPrefix};
    use std::collections::BTreeSet;

    fn nested() -> ControlState {
        // 0 -> outer(1) -> inner(2 -> 3 -> 2) -> 4 -> outer, then exit(5).
        let successors = vec![vec![1], vec![2, 5], vec![3], vec![2, 4], vec![1], vec![]];
        let mut sites: Vec<_> = (0..6)
            .map(|block| ControlSite {
                site_id: block,
                block,
                kind: "block".into(),
                scalar_type: "u64".into(),
                input_dependencies: BTreeSet::new(),
            })
            .collect();
        for (block, kind) in [(1, "branch"), (3, "branch"), (3, "decode"), (5, "decode")] {
            sites.push(ControlSite {
                site_id: sites.len(),
                block,
                kind: kind.into(),
                scalar_type: if kind == "branch" { "bool" } else { "u64" }.into(),
                input_dependencies: BTreeSet::new(),
            });
        }
        ControlState::new(ControlManifest {
            module_sha256: "0".repeat(64),
            instrumenter: "pecos-qis-control-v1".into(),
            sites,
            successors,
            loop_headers: vec![1, 2],
            loop_members: vec![BTreeSet::from([1, 2, 3, 4]), BTreeSet::from([2, 3])],
        })
    }
    fn prefix(index: usize) -> DecoderBatchPrefix {
        DecoderBatchPrefix {
            version: 1,
            runtime_shot_id: 17,
            first_batch_index: index,
            next_batch_index: index,
            batches: vec![],
        }
    }
    fn record(state: &mut ControlState, events: &[(usize, u64)]) {
        for &(id, value) in events {
            state.record(id, value).unwrap();
        }
    }
    #[test]
    fn control_state_nested_loop_exit_reentry_and_empty_batch_intervals() {
        let mut state = nested();
        record(
            &mut state,
            &[(0, 0), (1, 0), (6, 1), (2, 0), (3, 0), (8, 0)],
        );
        let first = state.take(&prefix(0)).unwrap();
        assert_eq!(
            first
                .active_loops
                .iter()
                .map(|l| (l.header, l.occurrence, l.iteration))
                .collect::<Vec<_>>(),
            [(1, 0, 0), (2, 0, 0)]
        );
        record(&mut state, &[(7, 1), (2, 0), (3, 0), (8, 0)]);
        let second = state.take(&prefix(0)).unwrap();
        assert_eq!(second.active_loops[1].iteration, 1);
        assert_eq!(second.events.last().unwrap().occurrence, 1);
        record(
            &mut state,
            &[(7, 0), (4, 0), (1, 0), (6, 1), (2, 0), (3, 0), (8, 0)],
        );
        let third = state.take(&prefix(0)).unwrap();
        assert_eq!(
            third
                .active_loops
                .iter()
                .map(|l| (l.occurrence, l.iteration))
                .collect::<Vec<_>>(),
            [(0, 1), (1, 0)]
        );
        record(
            &mut state,
            &[(7, 0), (4, 0), (1, 0), (6, 0), (5, 0), (9, 0)],
        );
        let exit = state.take(&prefix(0)).unwrap();
        assert!(exit.active_loops.is_empty());
        assert_eq!(exit.runtime_shot_id, 17);
        let mut fresh = state.fresh();
        record(&mut fresh, &[(0, 0), (1, 0), (6, 0), (5, 0), (9, 0)]);
        assert!(
            fresh
                .take(&prefix(0))
                .unwrap()
                .events
                .iter()
                .all(|e| e.occurrence == 0)
        );
    }
    #[test]
    fn control_state_rejects_wrong_edges_missing_predicates_and_overflow() {
        let mut state = nested();
        record(&mut state, &[(0, 0), (1, 0)]);
        assert!(state.record(2, 0).is_err());
        let mut state = nested();
        record(&mut state, &[(0, 0), (1, 0), (6, 0)]);
        assert!(state.record(2, 0).is_err());
        let mut state = nested();
        state.counts[0] = u64::MAX;
        assert!(state.record(0, 0).is_err());
        let mut state = nested();
        record(&mut state, &[(0, 0), (1, 0), (6, 1), (2, 0), (3, 0)]);
        // More than one bounded interval of control without a decoder fence.
        for _ in 0..(MAX_EVENTS / 3) {
            for &(id, value) in &[(7, 1), (2, 0), (3, 0)] {
                if state.record(id, value).is_err() {
                    assert_eq!(state.events.len(), MAX_EVENTS);
                    return;
                }
            }
        }
        panic!("missing event cap");
    }
    #[test]
    fn control_state_filters_only_reserved_events() {
        let mut state = nested();
        let physical = Operation::AllocateQubit { id: 0 };
        let event = Operation::TraceMetadata {
            metadata: std::collections::BTreeMap::from([(EVENT_KEY.into(), "0:0".into())]),
            qubit: None,
        };
        assert_eq!(
            state.filter(&[event, physical.clone()]).unwrap(),
            [physical]
        );
        assert_eq!(state.events.len(), 1);
        assert!(state.take(&prefix(0)).is_err());
    }
}
