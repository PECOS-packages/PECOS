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

//! The engine's work queue: FIFO with a membership mirror.
//!
//! The queue is consulted with `contains` on every activation and retry
//! wave (dozens of sites); a raw `VecDeque` made each of those an O(n)
//! scan, quadratic-shaped across a retry storm. The membership set makes
//! them O(log n) and gives dedup on push for free: a node is never queued
//! twice, matching the `if !contains { push }` guards every call site
//! already carried.

use std::collections::{BTreeSet, VecDeque};

use tket::hugr::Node;

/// FIFO node queue with O(log n) membership and dedup on push.
#[derive(Debug, Default, Clone)]
pub(crate) struct WorkQueue {
    queue: VecDeque<Node>,
    members: BTreeSet<Node>,
}

impl WorkQueue {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Enqueue at the back; a node already queued stays where it is.
    pub(crate) fn push_back(&mut self, node: Node) {
        if self.members.insert(node) {
            self.queue.push_back(node);
        }
    }

    /// Enqueue at the front (priority); a node already queued stays where
    /// it is.
    pub(crate) fn push_front(&mut self, node: Node) {
        if self.members.insert(node) {
            self.queue.push_front(node);
        }
    }

    pub(crate) fn pop_front(&mut self) -> Option<Node> {
        let node = self.queue.pop_front();
        if let Some(node) = node {
            self.members.remove(&node);
        }
        node
    }

    pub(crate) fn contains(&self, node: Node) -> bool {
        self.members.contains(&node)
    }

    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.members.clear();
    }
}

impl<'a> IntoIterator for &'a WorkQueue {
    type Item = &'a Node;
    type IntoIter = std::collections::vec_deque::Iter<'a, Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.queue.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tket::hugr::HugrView;

    #[test]
    fn dedup_and_order() {
        // Real node ids from a trivial hugr (Node is not directly
        // constructible).
        let hugr = tket::hugr::Hugr::default();
        let nodes: Vec<Node> = hugr.nodes().take(1).collect();
        let n1 = nodes[0];

        let mut q = WorkQueue::new();
        q.push_back(n1);
        q.push_back(n1); // dedup: stays queued once
        assert_eq!(q.len(), 1);
        assert!(q.contains(n1));
        assert_eq!(q.pop_front(), Some(n1));
        assert!(!q.contains(n1));
        assert!(q.is_empty());
        q.push_front(n1);
        assert_eq!(q.len(), 1);
        q.clear();
        assert!(q.is_empty());
    }
}
