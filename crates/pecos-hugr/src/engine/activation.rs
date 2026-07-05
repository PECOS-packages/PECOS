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

//! Container activation as a mechanism: the two-phase discipline.
//!
//! Every container (re-)activation -- CFG entry blocks and transitions,
//! Conditional case expansion, `TailLoop` expansion and continuation, and
//! Call frame re-activation -- must clear ALL stale state (processed flags,
//! output wires) BEFORE any readiness check queues work. Readiness
//! (`all_predecessors_ready`) consults the processed set, so interleaving
//! clear-and-queue per op category lets a consumer pass its readiness check
//! against a not-yet-cleared producer's previous-iteration flags, fire
//! early, and copy stale or missing values (the historical loop-freeze
//! class of bugs).
//!
//! Sites build a [`ContainerActivation`] batch with their own selection
//! logic (which nodes, in which order, under which queue policy);
//! [`HugrEngine::run_activation`] enforces the phase ordering so no site
//! can get it wrong again.

use std::collections::BTreeSet;

use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;

/// Whether a node queues unconditionally or only once its predecessors are
/// processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueuePolicy {
    /// Queue only when `all_predecessors_ready` passes. For nodes that copy
    /// their inputs at fire time (Calls, classical/extension ops): firing
    /// early copies missing values and starves everything downstream.
    IfReady,
    /// Queue unconditionally. For nodes that defer internally until their
    /// inputs resolve (Conditionals, `TailLoops`, bool reads) or that have no
    /// dataflow inputs at all (`LoadConstants`).
    Always,
}

/// A batched container activation. Build with the site's selection logic,
/// then apply with [`HugrEngine::run_activation`].
#[derive(Debug, Default)]
pub(crate) struct ContainerActivation {
    /// Nodes whose processed flag clears in phase 1.
    reset_processed: Vec<Node>,
    /// Nodes whose stale output wires clear in phase 1.
    reset_wires: Vec<Node>,
    /// Wire-clear exemptions (nodes holding freshly propagated values,
    /// e.g. a block or loop-body Input node).
    keep_wires: BTreeSet<Node>,
    /// Nodes released from the CFG-block gate only, without queueing
    /// (e.g. block-level tracking of ops that an inner `TailLoop` still
    /// owns and will queue at its own expansion).
    ungate_block_only: Vec<Node>,
    /// Nodes released from every container gate without queueing.
    ungate_all: Vec<Node>,
    /// Nodes to queue in phase 2, in submission order.
    queue: Vec<(Node, QueuePolicy)>,
}

impl ContainerActivation {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Phase 1: clear this node's processed flag AND stale output wires.
    pub(crate) fn reset(&mut self, node: Node) {
        self.reset_processed.push(node);
        self.reset_wires.push(node);
    }

    /// Phase 1: clear only this node's processed flag.
    pub(crate) fn reset_processed(&mut self, node: Node) {
        self.reset_processed.push(node);
    }

    /// Phase 1: clear only this node's stale output wires.
    pub(crate) fn reset_wires(&mut self, node: Node) {
        self.reset_wires.push(node);
    }

    /// Exempt a node from wire clearing (it holds freshly propagated
    /// values).
    pub(crate) fn keep_wires(&mut self, node: Node) {
        self.keep_wires.insert(node);
    }

    /// Release a node from the CFG-block gate without queueing it: an
    /// inner container still owns it and queues it at its own expansion.
    pub(crate) fn ungate_block_only(&mut self, node: Node) {
        self.ungate_block_only.push(node);
    }

    /// Release a node from EVERY container gate without queueing it (e.g.
    /// case quantum ops queued separately as entry nodes, or ops that only
    /// the retry path may queue).
    pub(crate) fn ungate(&mut self, node: Node) {
        self.ungate_all.push(node);
    }

    /// Phase 2: release this node from every container gate and queue it
    /// under the given policy.
    pub(crate) fn queue(&mut self, node: Node, policy: QueuePolicy) {
        self.queue.push((node, policy));
    }
}

impl HugrEngine {
    /// Apply a batched container activation with the two-phase discipline:
    /// every reset happens before any readiness check.
    pub(crate) fn run_activation(&mut self, hugr: &Hugr, act: &ContainerActivation) {
        // PHASE 1: clear processed flags first, then stale output wires --
        // readiness checks and value resolution must not see either.
        for node in &act.reset_processed {
            self.processed.remove(node);
        }
        for node in &act.reset_wires {
            if act.keep_wires.contains(node) {
                continue;
            }
            for port in 0..hugr.num_outputs(*node) {
                self.wire_state.classical_values.remove(&(*node, port));
                self.wire_state.wire_to_qubit.remove(&(*node, port));
            }
        }

        // Gates: a queued node leaves every container gate (it is being
        // activated now); an ungate-only node leaves just the block gate.
        for node in &act.ungate_block_only {
            self.nodes_inside_cfg_blocks.remove(node);
        }
        for node in &act.ungate_all {
            self.nodes_inside_cfg_blocks.remove(node);
            self.nodes_inside_cases.remove(node);
            self.nodes_inside_tailloops.remove(node);
        }
        for (node, _) in &act.queue {
            self.nodes_inside_cfg_blocks.remove(node);
            self.nodes_inside_cases.remove(node);
            self.nodes_inside_tailloops.remove(node);
        }

        // PHASE 2: queue, respecting each node's policy. Nodes that fail
        // their readiness check here are queued later by
        // queue_ready_successors when their producers complete.
        for &(node, policy) in &act.queue {
            if self.work_queue.contains(node) || self.processed.contains(&node) {
                continue;
            }
            if policy == QueuePolicy::IfReady && !self.all_predecessors_ready(hugr, node) {
                continue;
            }
            self.work_queue.push_back(node);
        }
    }
}
