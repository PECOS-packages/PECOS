//! Builder pattern for Fusion Blossom decoder
//!
//! This module provides an ergonomic builder for constructing Fusion Blossom decoders.
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use pecos_fusion_blossom::{FusionBlossomDecoder, SolverType};
//! use ndarray::arr2;
//!
//! let h = arr2(&[[1u8, 1, 0], [0, 1, 1]]);
//!
//! let decoder = FusionBlossomDecoder::builder()
//!     .num_nodes(2)
//!     .num_observables(3)
//!     .solver_type(SolverType::Serial)
//!     .from_check_matrix(&h, None)?;
//! # Ok(())
//! # }
//! ```

use crate::{
    decoder::{FusionBlossomConfig, FusionBlossomDecoder, SolverType, StandardCode},
    errors::Result,
};
use ndarray::Array2;

/// Builder for `FusionBlossomDecoder`
#[must_use]
pub struct FusionBlossomBuilder {
    num_nodes: Option<usize>,
    num_observables: usize,
    solver_type: SolverType,
    max_tree_size: Option<usize>,
    edges: Vec<EdgeSpec>,
    boundary_edges: Vec<BoundaryEdgeSpec>,
}

struct EdgeSpec {
    node1: usize,
    node2: usize,
    observables: Vec<usize>,
    weight: Option<f64>,
}

struct BoundaryEdgeSpec {
    node: usize,
    observables: Vec<usize>,
    weight: Option<f64>,
}

impl Default for FusionBlossomBuilder {
    fn default() -> Self {
        Self {
            num_nodes: None,
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
            edges: Vec::new(),
            boundary_edges: Vec::new(),
        }
    }
}

impl FusionBlossomBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nodes (detectors)
    pub fn num_nodes(mut self, num_nodes: usize) -> Self {
        self.num_nodes = Some(num_nodes);
        self
    }

    /// Set the number of observables
    pub fn num_observables(mut self, num_observables: usize) -> Self {
        self.num_observables = num_observables;
        self
    }

    /// Set the solver type (Legacy or Serial)
    pub fn solver_type(mut self, solver_type: SolverType) -> Self {
        self.solver_type = solver_type;
        self
    }

    /// Set the maximum tree size for union-find
    pub fn max_tree_size(mut self, size: usize) -> Self {
        self.max_tree_size = Some(size);
        self
    }

    /// Add an edge between two nodes
    pub fn add_edge(
        mut self,
        node1: usize,
        node2: usize,
        observables: impl Into<Vec<usize>>,
        weight: Option<f64>,
    ) -> Self {
        self.edges.push(EdgeSpec {
            node1,
            node2,
            observables: observables.into(),
            weight,
        });
        self
    }

    /// Add a boundary edge from a node
    pub fn add_boundary_edge(
        mut self,
        node: usize,
        observables: impl Into<Vec<usize>>,
        weight: Option<f64>,
    ) -> Self {
        self.boundary_edges.push(BoundaryEdgeSpec {
            node,
            observables: observables.into(),
            weight,
        });
        self
    }

    /// Build from a check matrix
    ///
    /// # Errors
    ///
    /// Returns `FusionBlossomError` if the matrix is invalid or decoder creation fails.
    pub fn from_check_matrix(
        self,
        check_matrix: &Array2<u8>,
        weights: Option<&[f64]>,
    ) -> Result<FusionBlossomDecoder> {
        let config = FusionBlossomConfig {
            num_nodes: self.num_nodes,
            num_observables: self.num_observables,
            solver_type: self.solver_type,
            max_tree_size: self.max_tree_size,
        };
        FusionBlossomDecoder::from_check_matrix(check_matrix, weights, config)
    }

    /// Build from a standard QEC code
    ///
    /// # Errors
    ///
    /// Returns `FusionBlossomError` if decoder creation fails.
    pub fn from_standard_code(self, code: StandardCode) -> Result<FusionBlossomDecoder> {
        let config = FusionBlossomConfig {
            num_nodes: self.num_nodes,
            num_observables: self.num_observables,
            solver_type: self.solver_type,
            max_tree_size: self.max_tree_size,
        };
        FusionBlossomDecoder::from_standard_code(code, config)
    }

    /// Build the decoder from manually specified edges
    ///
    /// # Errors
    ///
    /// Returns `FusionBlossomError` if:
    /// - `num_nodes` was not set
    /// - Adding an edge fails
    pub fn build(self) -> Result<FusionBlossomDecoder> {
        let config = FusionBlossomConfig {
            num_nodes: self.num_nodes,
            num_observables: self.num_observables,
            solver_type: self.solver_type,
            max_tree_size: self.max_tree_size,
        };

        let mut decoder = FusionBlossomDecoder::new(config)?;

        // Add all edges
        for edge in self.edges {
            decoder.add_edge(edge.node1, edge.node2, &edge.observables, edge.weight)?;
        }

        // Add boundary edges
        for edge in self.boundary_edges {
            decoder.add_boundary_edge(edge.node, &edge.observables, edge.weight)?;
        }

        Ok(decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_builder_from_check_matrix() {
        let h = arr2(&[[1, 1, 0], [0, 1, 1]]);
        let decoder = FusionBlossomBuilder::new()
            .num_observables(3)
            .solver_type(SolverType::Serial)
            .from_check_matrix(&h, None);

        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.num_nodes(), 2);
    }

    #[test]
    fn test_builder_manual_edges() {
        let decoder = FusionBlossomBuilder::new()
            .num_nodes(3)
            .num_observables(2)
            .add_edge(0, 1, vec![0], Some(1.0))
            .add_edge(1, 2, vec![1], Some(1.0))
            .add_boundary_edge(0, vec![0], Some(1.0))
            .add_boundary_edge(2, vec![1], Some(1.0))
            .build();

        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.num_nodes(), 3);
        assert_eq!(decoder.num_edges(), 4);
    }
}
