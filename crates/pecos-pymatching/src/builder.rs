//! Improved builder pattern for `PyMatching` decoder

use super::decoder::{
    CheckMatrix, CheckMatrixConfig, DEFAULT_OBSERVABLES, PyMatchingConfig, PyMatchingDecoder,
};
use super::errors::Result;
use std::collections::HashSet;

/// Builder for constructing `PyMatching` decoders with a fluent API
#[must_use]
pub struct PyMatchingBuilder {
    num_nodes: Option<usize>,
    num_observables: usize,
    num_neighbours: Option<i32>,
    edges: Vec<EdgeSpec>,
    boundary_edges: Vec<BoundaryEdgeSpec>,
    boundary_nodes: HashSet<usize>,
}

struct EdgeSpec {
    node1: usize,
    node2: usize,
    observables: Vec<usize>,
    weight: f64,
    error_probability: Option<f64>,
}

struct BoundaryEdgeSpec {
    node: usize,
    observables: Vec<usize>,
    weight: f64,
    error_probability: Option<f64>,
}

impl Default for PyMatchingBuilder {
    fn default() -> Self {
        Self {
            num_nodes: None,
            num_observables: DEFAULT_OBSERVABLES,
            num_neighbours: None,
            edges: Vec::new(),
            boundary_edges: Vec::new(),
            boundary_nodes: HashSet::new(),
        }
    }
}

impl PyMatchingBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nodes in the graph
    pub fn nodes(mut self, num_nodes: usize) -> Self {
        self.num_nodes = Some(num_nodes);
        self
    }

    /// Set the number of observables
    pub fn observables(mut self, num_observables: usize) -> Self {
        self.num_observables = num_observables;
        self
    }

    /// Set the number of neighbours for matching
    pub fn neighbours(mut self, num_neighbours: i32) -> Self {
        self.num_neighbours = Some(num_neighbours);
        self
    }

    /// Set a default error probability for all edges
    pub fn with_error_probability(mut self, p: f64) -> Self {
        // Apply to all existing edges
        for edge in &mut self.edges {
            if edge.error_probability.is_none() {
                edge.error_probability = Some(p);
            }
        }
        for edge in &mut self.boundary_edges {
            if edge.error_probability.is_none() {
                edge.error_probability = Some(p);
            }
        }
        self
    }

    /// Add an edge to the graph
    pub fn add_edge(
        mut self,
        node1: usize,
        node2: usize,
        observables: impl Into<Vec<usize>>,
        weight: f64,
        error_probability: Option<f64>,
    ) -> Self {
        self.edges.push(EdgeSpec {
            node1,
            node2,
            observables: observables.into(),
            weight,
            error_probability,
        });
        self
    }

    /// Add a chain of edges connecting consecutive nodes
    pub fn add_edge_chain(
        mut self,
        nodes: impl IntoIterator<Item = usize>,
        weight: f64,
        error_probability: Option<f64>,
    ) -> Self {
        let nodes: Vec<_> = nodes.into_iter().collect();
        for i in 0..nodes.len().saturating_sub(1) {
            self.edges.push(EdgeSpec {
                node1: nodes[i],
                node2: nodes[i + 1],
                observables: vec![i],
                weight,
                error_probability,
            });
        }
        self
    }

    /// Add a boundary edge
    pub fn add_boundary_edge(
        mut self,
        node: usize,
        observables: impl Into<Vec<usize>>,
        weight: f64,
        error_probability: Option<f64>,
    ) -> Self {
        self.boundary_edges.push(BoundaryEdgeSpec {
            node,
            observables: observables.into(),
            weight,
            error_probability,
        });
        self
    }

    /// Add nodes to the boundary set
    pub fn add_boundary_nodes(mut self, nodes: impl IntoIterator<Item = usize>) -> Self {
        self.boundary_nodes.extend(nodes);
        self
    }

    /// Create a repetition code with the specified size
    pub fn repetition_code(mut self, size: usize, error_probability: f64) -> Self {
        self.num_nodes = Some(size);
        self.num_observables = size - 1;

        // Add chain of edges
        for i in 0..size - 1 {
            self.edges.push(EdgeSpec {
                node1: i,
                node2: i + 1,
                observables: vec![i],
                weight: 1.0,
                error_probability: Some(error_probability),
            });
        }

        self
    }

    /// Create a simple square lattice
    pub fn square_lattice(mut self, width: usize, height: usize, error_probability: f64) -> Self {
        let num_nodes = width * height;
        self.num_nodes = Some(num_nodes);

        let mut obs_idx = 0;

        // Horizontal edges
        for y in 0..height {
            for x in 0..width - 1 {
                let node1 = y * width + x;
                let node2 = node1 + 1;
                self.edges.push(EdgeSpec {
                    node1,
                    node2,
                    observables: vec![obs_idx],
                    weight: 1.0,
                    error_probability: Some(error_probability),
                });
                obs_idx += 1;
            }
        }

        // Vertical edges
        for y in 0..height - 1 {
            for x in 0..width {
                let node1 = y * width + x;
                let node2 = (y + 1) * width + x;
                self.edges.push(EdgeSpec {
                    node1,
                    node2,
                    observables: vec![obs_idx],
                    weight: 1.0,
                    error_probability: Some(error_probability),
                });
                obs_idx += 1;
            }
        }

        self.num_observables = obs_idx;

        // Set boundary as the perimeter
        for x in 0..width {
            self.boundary_nodes.insert(x); // Top row
            self.boundary_nodes.insert((height - 1) * width + x); // Bottom row
        }
        for y in 1..height - 1 {
            self.boundary_nodes.insert(y * width); // Left column
            self.boundary_nodes.insert(y * width + width - 1); // Right column
        }

        self
    }

    /// Add edges from a `CheckMatrix`
    ///
    /// This is a convenience method to populate the builder from a check matrix.
    /// Note: this will set the number of nodes and observables based on the matrix.
    ///
    /// # Errors
    ///
    /// Returns a [`PyMatchingError`](crate::PyMatchingError) if the decoder creation fails.
    ///
    /// # Panics
    ///
    /// This function will not panic. The internal `unwrap()` is safe because
    /// `config` is checked for `is_none()` before use.
    pub fn from_check_matrix(
        self,
        matrix: &CheckMatrix,
        config: Option<CheckMatrixConfig>,
    ) -> Result<PyMatchingDecoder> {
        // If this is a simple case, just use the direct API
        let Some(config) = config else {
            return PyMatchingDecoder::from_check_matrix(matrix);
        };
        if config.repetitions == 1 {
            return PyMatchingDecoder::from_check_matrix(matrix);
        }
        PyMatchingDecoder::from_check_matrix_with_config(matrix, config)
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns a [`PyMatchingError`](crate::PyMatchingError) if:
    /// - The decoder creation fails
    /// - Adding an edge fails (e.g., invalid node indices)
    /// - Adding a boundary edge fails
    pub fn build(self) -> Result<PyMatchingDecoder> {
        let config = PyMatchingConfig {
            num_nodes: self.num_nodes,
            num_observables: self.num_observables,
            num_neighbours: self.num_neighbours,
        };

        let mut decoder = PyMatchingDecoder::new(config)?;

        // Add all edges
        for edge in self.edges {
            decoder.add_edge(
                edge.node1,
                edge.node2,
                &edge.observables,
                Some(edge.weight),
                edge.error_probability,
                None,
            )?;
        }

        // Add boundary edges
        for edge in self.boundary_edges {
            decoder.add_boundary_edge(
                edge.node,
                &edge.observables,
                Some(edge.weight),
                edge.error_probability,
                None,
            )?;
        }

        // Set boundary nodes
        if !self.boundary_nodes.is_empty() {
            let boundary: Vec<_> = self.boundary_nodes.into_iter().collect();
            decoder.set_boundary(&boundary);
        }

        Ok(decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_code_builder() {
        let decoder = PyMatchingDecoder::builder()
            .repetition_code(5, 0.1)
            .build()
            .unwrap();

        assert_eq!(decoder.num_nodes(), 5);
        assert_eq!(decoder.num_edges(), 4);
        // PyMatching always reports at least 64 observables
        assert!(decoder.num_observables() >= 4);
    }

    #[test]
    fn test_chain_builder() {
        let decoder = PyMatchingDecoder::builder()
            .nodes(6)
            .observables(5)
            .add_edge_chain(0..6, 1.0, Some(0.1))
            .add_boundary_nodes([0, 5])
            .build()
            .unwrap();

        assert_eq!(decoder.num_nodes(), 6);
        assert_eq!(decoder.num_edges(), 5);
        assert_eq!(decoder.num_detectors(), 4); // 6 nodes - 2 boundary nodes
    }

    #[test]
    fn test_square_lattice_builder() {
        let decoder = PyMatchingDecoder::builder()
            .square_lattice(3, 3, 0.1)
            .build()
            .unwrap();

        assert_eq!(decoder.num_nodes(), 9);
        // 3x3 lattice has 12 edges: 6 horizontal + 6 vertical
        assert_eq!(decoder.num_edges(), 12);
        // PyMatching always reports at least 64 observables
        assert!(decoder.num_observables() >= 12);

        // Perimeter has 8 nodes
        let boundary_count = decoder.boundary_nodes().count();
        assert_eq!(boundary_count, 8);
    }

    #[test]
    fn test_custom_builder() {
        let decoder = PyMatchingDecoder::builder()
            .nodes(4)
            .observables(10)
            .add_edge(0, 1, vec![0, 1], 1.0, Some(0.1))
            .add_edge(1, 2, vec![2], 2.0, Some(0.2))
            .add_boundary_edge(3, vec![3], 1.5, Some(0.15))
            .add_boundary_nodes([0, 3])
            .build()
            .unwrap();

        assert_eq!(decoder.num_nodes(), 4);
        assert_eq!(decoder.num_edges(), 3); // 2 regular + 1 boundary
        // PyMatching always reports at least 64 observables
        assert!(decoder.num_observables() >= 10);
    }
}
