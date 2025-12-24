//! Iterator implementations for `PyMatching` decoder

use super::decoder::{EdgeData, PyMatchingDecoder};

/// Iterator over all edges in the matching graph
pub type EdgeIterator = std::vec::IntoIter<EdgeData>;

/// Iterator over boundary nodes
pub type BoundaryIterator = std::vec::IntoIter<usize>;

/// Extension methods for `PyMatchingDecoder`
impl PyMatchingDecoder {
    /// Returns an iterator over all edges in the graph
    #[must_use]
    pub fn edges(&self) -> EdgeIterator {
        self.get_all_edges().into_iter()
    }

    /// Returns an iterator over boundary node indices
    #[must_use]
    pub fn boundary_nodes(&self) -> BoundaryIterator {
        self.get_boundary().into_iter()
    }

    /// Get edge data between two nodes (if it exists)
    #[must_use]
    pub fn get_edge(&self, node1: usize, node2: usize) -> Option<EdgeData> {
        if self.has_edge(node1, node2) {
            self.get_edge_data(node1, node2).ok()
        } else {
            None
        }
    }

    /// Get boundary edge data for a node (if it exists)
    #[must_use]
    pub fn get_boundary_edge(&self, node: usize) -> Option<EdgeData> {
        if self.has_boundary_edge(node) {
            self.get_boundary_edge_data(node).ok()
        } else {
            None
        }
    }
}
