//! Complete `PyMatching` decoder implementation with full API surface

use super::bridge::ffi;
use super::errors::{PyMatchingError, Result};
use cxx::UniquePtr;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

// Type aliases for clarity
pub type NodeId = usize;
pub type ObservableId = usize;
pub type DetectorId = i64;

// Constants
pub const DEFAULT_OBSERVABLES: usize = 64;
pub const OPTIMIZED_OBSERVABLE_LIMIT: usize = 64;
pub const BITS_PER_BYTE: usize = 8;
pub const BOUNDARY_NODE_MARKER: usize = usize::MAX;
pub const BOUNDARY_DETECTOR_MARKER: i64 = -1;

/// Decoding result
#[derive(Debug, Clone, PartialEq)]
pub struct DecodingResult {
    pub observable: Vec<u8>,
    pub weight: f64,
}

impl fmt::Display for DecodingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DecodingResult {{ observables: {:?}, weight: {:.6} }}",
            self.observable, self.weight
        )
    }
}

/// Sparse check matrix representation for `PyMatching` following COO format
///
/// This struct provides a clean API for representing sparse parity check matrices
/// using coordinate (COO) format with optional edge weights for quantum error correction.
///
/// # Examples
///
/// ## Basic COO format usage
/// ```rust
/// use pecos_pymatching::{CheckMatrix, PyMatchingDecoder};
///
/// // Create a simple repetition code matrix: H = [[1, 1, 0], [0, 1, 1]]
/// // COO format: specify non-zero positions directly
/// let matrix = CheckMatrix::new(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2])
///     .with_weights(vec![1.0, 2.0, 1.0])  // Different weights for each qubit
///     .unwrap();
///
/// let mut decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();
///
/// // Decode a syndrome
/// let syndrome = vec![1, 0];  // First check fires
/// let result = decoder.decode(&syndrome).unwrap();
/// println!("Correction: {:?}", result.observable);
/// ```
///
/// ## Migration from triplet format
/// ```rust
/// use pecos_pymatching::{CheckMatrix, PyMatchingDecoder};
///
/// let entries = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];
/// let matrix = CheckMatrix::from_triplets(entries, 2, 3)
///     .with_weights(vec![1.0, 2.0, 1.0])
///     .unwrap();
/// println!("Matrix has {} rows and {} columns", matrix.rows(), matrix.cols());
/// ```
///
/// ## Dense matrix conversion
/// ```rust
/// use pecos_pymatching::{CheckMatrix, PyMatchingDecoder};
///
/// let dense = vec![vec![1, 1, 0], vec![0, 1, 1]];
/// let matrix = CheckMatrix::from_dense_vec(&dense).unwrap();
/// println!("Matrix has {} non-zero entries", matrix.nnz());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CheckMatrix {
    /// Number of rows (detectors/checks) in the matrix
    rows: usize,
    /// Number of columns (errors/qubits) in the matrix
    cols: usize,
    /// Row indices of non-zero entries (COO format)
    row_indices: Vec<usize>,
    /// Column indices of non-zero entries (COO format)
    col_indices: Vec<usize>,
    /// Optional edge weights for each column (error)
    weights: Option<Vec<f64>>,
}

impl CheckMatrix {
    /// Create a new sparse check matrix using COO format
    ///
    /// # Arguments
    /// * `rows` - Number of rows (detectors/checks) in the matrix
    /// * `cols` - Number of columns (errors/qubits) in the matrix
    /// * `row_indices` - Row indices of non-zero entries
    /// * `col_indices` - Column indices of non-zero entries
    ///
    /// # Example
    /// ```rust
    /// use pecos_pymatching::CheckMatrix;
    ///
    /// // H = [[1, 1, 0], [0, 1, 1]]
    /// let matrix = CheckMatrix::new(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]);
    /// assert_eq!(matrix.rows(), 2);
    /// assert_eq!(matrix.cols(), 3);
    /// ```
    #[must_use]
    pub fn new(rows: usize, cols: usize, row_indices: Vec<usize>, col_indices: Vec<usize>) -> Self {
        Self {
            rows,
            cols,
            row_indices,
            col_indices,
            weights: None,
        }
    }

    /// Create a sparse check matrix from triplets (row, col, value)
    /// This provides compatibility with the old format
    #[must_use]
    pub fn from_triplets(entries: Vec<(usize, usize, u8)>, rows: usize, cols: usize) -> Self {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for (row, col, val) in entries {
            if val != 0 {
                row_indices.push(row);
                col_indices.push(col);
            }
        }

        Self {
            rows,
            cols,
            row_indices,
            col_indices,
            weights: None,
        }
    }

    /// Create a sparse check matrix from a dense matrix represented as Vec<Vec<u8>>
    ///
    /// # Errors
    /// Returns an error if rows have inconsistent column counts.
    pub fn from_dense_vec(matrix: &[Vec<u8>]) -> Result<Self> {
        if matrix.is_empty() {
            return Ok(Self {
                rows: 0,
                cols: 0,
                row_indices: Vec::new(),
                col_indices: Vec::new(),
                weights: None,
            });
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        // Validate consistent column count
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != cols {
                return Err(PyMatchingError::Configuration(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    cols
                )));
            }
        }

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for (row_idx, row) in matrix.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                if val != 0 {
                    row_indices.push(row_idx);
                    col_indices.push(col_idx);
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            row_indices,
            col_indices,
            weights: None,
        })
    }

    /// Set weights for the matrix columns using fluent API
    ///
    /// # Errors
    /// Returns an error if the weights length doesn't match the number of columns.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        if weights.len() != self.cols {
            return Err(PyMatchingError::Configuration(format!(
                "weights length {} doesn't match number of columns {}",
                weights.len(),
                self.cols
            )));
        }
        self.weights = Some(weights);
        Ok(self)
    }

    /// Get the number of rows (detectors/checks)
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns (errors/qubits)
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the weights if they exist
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Get the number of non-zero entries
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    /// Convert to triplet format for internal use
    #[must_use]
    pub fn to_triplets(&self) -> Vec<(usize, usize, u8)> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .map(|(&row, &col)| (row, col, 1u8))
            .collect()
    }

    /// Validate the matrix structure and constraints
    ///
    /// # Errors
    /// Returns an error if indices are mismatched, out of bounds, or QEC constraints are violated.
    pub fn validate(&self) -> Result<()> {
        // Check that row and column indices have the same length
        if self.row_indices.len() != self.col_indices.len() {
            return Err(PyMatchingError::Configuration(format!(
                "Row indices length {} doesn't match column indices length {}",
                self.row_indices.len(),
                self.col_indices.len()
            )));
        }

        // Check that all indices are within bounds
        for &row in &self.row_indices {
            if row >= self.rows {
                return Err(PyMatchingError::Configuration(format!(
                    "Row index {} out of bounds (matrix has {} rows)",
                    row, self.rows
                )));
            }
        }

        for &col in &self.col_indices {
            if col >= self.cols {
                return Err(PyMatchingError::Configuration(format!(
                    "Column index {} out of bounds (matrix has {} columns)",
                    col, self.cols
                )));
            }
        }

        // Check that weights length matches number of columns if present
        if let Some(ref weights) = self.weights
            && weights.len() != self.cols
        {
            return Err(PyMatchingError::Configuration(format!(
                "weights length {} doesn't match number of columns {}",
                weights.len(),
                self.cols
            )));
        }

        // Check QEC constraint: each column has at most 2 non-zero entries (for matching decoder)
        let mut col_counts = vec![0; self.cols];
        for &col in &self.col_indices {
            col_counts[col] += 1;
        }

        for (col_idx, &count) in col_counts.iter().enumerate() {
            if count > 2 {
                return Err(PyMatchingError::Configuration(format!(
                    "Column {col_idx} has {count} non-zero entries, expected at most 2 for matching decoder"
                )));
            }
        }

        Ok(())
    }

    /// Get all row indices for a specific column
    #[must_use]
    pub fn get_column_entries(&self, col: usize) -> Vec<usize> {
        self.col_indices
            .iter()
            .enumerate()
            .filter_map(|(idx, &c)| {
                if c == col {
                    Some(self.row_indices[idx])
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Configuration for `PyMatching` decoder
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyMatchingConfig {
    /// Maximum number of neighbours to consider during matching
    pub num_neighbours: Option<i32>,
    /// Initial number of nodes (required unless loading from DEM)
    pub num_nodes: Option<usize>,
    /// Number of observables
    pub num_observables: usize,
}

impl Default for PyMatchingConfig {
    fn default() -> Self {
        Self {
            num_neighbours: None,
            num_nodes: None,
            num_observables: DEFAULT_OBSERVABLES,
        }
    }
}

/// Complete `PyMatching` decoder with full API
pub struct PyMatchingDecoder {
    graph: UniquePtr<ffi::PyMatchingGraph>,
    config: PyMatchingConfig,
}

impl fmt::Display for PyMatchingDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.graph_summary())
    }
}

impl PyMatchingDecoder {
    /// Normalize edge parameters to their default values
    fn normalize_edge_params(
        weight: Option<f64>,
        error_probability: Option<f64>,
        merge_strategy: Option<MergeStrategy>,
    ) -> (f64, f64, MergeStrategy) {
        (
            weight.unwrap_or(1.0),
            error_probability.unwrap_or(f64::NAN),
            merge_strategy.unwrap_or(MergeStrategy::SmallestWeight),
        )
    }

    /// Create a new builder for constructing a decoder
    pub fn builder() -> crate::builder::PyMatchingBuilder {
        crate::builder::PyMatchingBuilder::new()
    }

    /// Add spacelike edges from check matrix columns
    fn add_spacelike_edges_from_check_matrix(
        &mut self,
        col_entries: &[Vec<usize>],
        weights: Option<&[f64]>,
        error_probabilities: Option<&[f64]>,
        repetitions: usize,
        num_rows: usize,
    ) -> Result<()> {
        for (col_idx, rows) in col_entries.iter().enumerate() {
            let weight = weights.map_or(1.0, |w| w[col_idx]);
            let error_prob = error_probabilities.map(|p| p[col_idx]);

            match rows.len() {
                0 => {
                    // No edge for this error
                }
                1 => {
                    // Single detector - create boundary edge
                    let node = rows[0];
                    for rep in 0..repetitions {
                        let actual_node = node + rep * num_rows;
                        self.add_boundary_edge(
                            actual_node,
                            &[col_idx],
                            Some(weight),
                            error_prob,
                            Some(MergeStrategy::SmallestWeight),
                        )?;
                    }
                }
                2 => {
                    // Two detectors - create edge between them
                    let node1 = rows[0];
                    let node2 = rows[1];

                    // Add spacelike edges
                    for rep in 0..repetitions {
                        let actual_node1 = node1 + rep * num_rows;
                        let actual_node2 = node2 + rep * num_rows;
                        self.add_edge(
                            actual_node1,
                            actual_node2,
                            &[col_idx],
                            Some(weight),
                            error_prob,
                            Some(MergeStrategy::SmallestWeight),
                        )?;
                    }
                }
                _ => {
                    return Err(PyMatchingError::Configuration(format!(
                        "Column {} has {} non-zero entries, expected 1 or 2",
                        col_idx,
                        rows.len()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Add timelike edges between repetitions
    fn add_timelike_edges(
        &mut self,
        repetitions: usize,
        num_rows: usize,
        timelike_weights: Option<&[f64]>,
        measurement_error_probabilities: Option<&[f64]>,
    ) -> Result<()> {
        if repetitions <= 1 {
            return Ok(());
        }

        // Validate timelike weights and measurement error probabilities
        if let Some(t_weights) = timelike_weights
            && t_weights.len() != num_rows
        {
            return Err(PyMatchingError::Configuration(format!(
                "timelike_weights has length {} but must equal number of rows ({})",
                t_weights.len(),
                num_rows
            )));
        }

        if let Some(m_errors) = measurement_error_probabilities
            && m_errors.len() != num_rows
        {
            return Err(PyMatchingError::Configuration(format!(
                "measurement_error_probabilities has length {} but must equal number of rows ({})",
                m_errors.len(),
                num_rows
            )));
        }

        // Add timelike edges between consecutive rounds
        for rep in 0..(repetitions - 1) {
            for row in 0..num_rows {
                let node1 = row + rep * num_rows;
                let node2 = row + (rep + 1) * num_rows;

                let weight = timelike_weights.map_or(1.0, |w| w[row]);
                let error_prob = measurement_error_probabilities.map(|p| p[row]);

                self.add_edge(
                    node1,
                    node2,
                    &[], // No observables for timelike edges
                    Some(weight),
                    error_prob,
                    Some(MergeStrategy::SmallestWeight),
                )?;
            }
        }

        Ok(())
    }

    /// Create a new decoder from configuration
    ///
    /// # Errors
    /// Returns an error if `num_nodes` is not specified in the configuration.
    pub fn new(config: PyMatchingConfig) -> Result<Self> {
        let graph = if let Some(num_nodes) = config.num_nodes {
            if config.num_observables <= OPTIMIZED_OBSERVABLE_LIMIT {
                ffi::create_pymatching_graph(num_nodes)
            } else {
                ffi::create_pymatching_graph_with_observables(num_nodes, config.num_observables)
            }
        } else {
            return Err(PyMatchingError::Configuration(
                "num_nodes must be specified in config".to_string(),
            ));
        };

        Ok(Self { graph, config })
    }

    /// Create a decoder from a Detector Error Model (DEM) string
    ///
    /// # Errors
    /// Returns an error if the DEM string is invalid or cannot be parsed.
    pub fn from_dem(dem_string: &str) -> Result<Self> {
        let graph = ffi::create_pymatching_graph_from_dem(dem_string)?;

        // Query graph for configuration
        let num_nodes = ffi::pymatching_get_num_nodes(&graph);
        let num_observables = ffi::pymatching_get_num_observables(&graph);

        let config = PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(num_nodes),
            num_observables,
        };

        Ok(Self { graph, config })
    }

    /// Create a decoder from a check matrix
    ///
    /// The check matrix should be in sparse format where:
    /// - Each row represents a detector/check
    /// - Each column represents a potential error
    /// - Each column should have 1 or 2 non-zero entries
    ///
    /// # Arguments
    /// * `check_matrix` - Sparse representation as (`row_indices`, `col_indices`, values)
    /// * `num_rows` - Number of rows (detectors) in the matrix
    /// * `num_cols` - Number of columns (errors) in the matrix
    /// * `weights` - Optional weights for each column (error)
    /// * `error_probabilities` - Optional error probabilities for each column
    /// * `repetitions` - Number of syndrome extraction rounds (for timelike edges)
    /// * `timelike_weights` - Optional weights for timelike edges (between rounds)
    /// * `measurement_error_probabilities` - Optional error probabilities for timelike edges
    /// * `use_virtual_boundary` - If true, use virtual boundary node for single-detector errors
    ///
    /// Internal method for creating decoder from check matrix with configuration struct
    ///
    /// This works directly with `CheckMatrix` data without conversion to triplets.
    fn from_check_matrix_with_config_internal(
        matrix: &CheckMatrix,
        config: &CheckMatrixConfig,
    ) -> Result<Self> {
        let total_nodes = matrix.rows * config.repetitions;

        // Create decoder with appropriate number of nodes
        let decoder_config = PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(total_nodes),
            num_observables: matrix.cols,
        };

        let mut decoder = Self::new(decoder_config)?;

        // Set boundary if not using virtual boundary
        if !config.use_virtual_boundary {
            let boundary_nodes: Vec<usize> =
                (matrix.rows * (config.repetitions - 1)..total_nodes).collect();
            decoder.set_boundary(&boundary_nodes);
        }

        // Group matrix entries by column
        let mut col_entries: Vec<Vec<usize>> = vec![Vec::new(); matrix.cols];
        for (&row, &col) in matrix.row_indices.iter().zip(matrix.col_indices.iter()) {
            col_entries[col].push(row);
        }

        // Add spacelike edges
        decoder.add_spacelike_edges_from_check_matrix(
            &col_entries,
            config.weights.as_deref(),
            config.error_probabilities.as_deref(),
            config.repetitions,
            matrix.rows,
        )?;

        // Add timelike edges
        decoder.add_timelike_edges(
            config.repetitions,
            matrix.rows,
            config.timelike_weights.as_deref(),
            config.measurement_error_probabilities.as_deref(),
        )?;

        Ok(decoder)
    }

    /// Create decoder from a `CheckMatrix`
    ///
    /// This is the new clean API for creating a decoder from a check matrix.
    ///
    /// # Arguments
    /// - `matrix`: `CheckMatrix` containing the matrix structure and optional weights
    ///
    /// # Errors
    /// Returns an error if the matrix validation fails or decoder creation fails.
    ///
    /// # Example
    /// ```rust
    /// use pecos_pymatching::{CheckMatrix, PyMatchingDecoder};
    ///
    /// let matrix = CheckMatrix::new(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2])
    ///     .with_weights(vec![1.0, 2.0, 1.0])
    ///     .unwrap();
    /// let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();
    /// assert!(decoder.num_nodes() >= 2);
    /// ```
    pub fn from_check_matrix(matrix: &CheckMatrix) -> Result<Self> {
        matrix.validate()?;

        let config = CheckMatrixConfig {
            weights: matrix.weights.clone(),
            ..Default::default()
        };

        Self::from_check_matrix_with_config_internal(matrix, &config)
    }

    /// Create decoder from a `CheckMatrix` with additional configuration
    ///
    /// This is the advanced API for creating a decoder from a check matrix with custom configuration.
    ///
    /// # Arguments
    /// - `matrix`: `CheckMatrix` containing the matrix structure and optional weights
    /// - `config`: Additional configuration options
    ///
    /// # Errors
    /// Returns an error if matrix validation fails, configuration is invalid, or decoder creation fails.
    ///
    /// # Example
    /// ```rust
    /// use pecos_pymatching::{CheckMatrix, PyMatchingDecoder, CheckMatrixConfig};
    ///
    /// let matrix = CheckMatrix::new(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2])
    ///     .with_weights(vec![1.0, 2.0, 1.0])
    ///     .unwrap();
    /// let config = CheckMatrixConfig {
    ///     repetitions: 3,
    ///     ..Default::default()
    /// };
    /// let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();
    /// assert!(decoder.num_nodes() >= 6); // 2 detectors * 3 repetitions
    /// ```
    pub fn from_check_matrix_with_config(
        matrix: &CheckMatrix,
        mut config: CheckMatrixConfig,
    ) -> Result<Self> {
        matrix.validate()?;

        // Use weights from matrix if not provided in config
        if config.weights.is_none() && matrix.weights.is_some() {
            config.weights.clone_from(&matrix.weights);
        }

        Self::from_check_matrix_with_config_internal(matrix, &config)
    }

    /// Add an edge between two nodes with configuration
    ///
    /// # Errors
    /// Returns an error if the edge cannot be added due to graph constraints.
    pub fn add_edge_with_config(
        &mut self,
        node1: NodeId,
        node2: NodeId,
        observables: &[ObservableId],
        config: EdgeConfig,
    ) -> Result<()> {
        let error_prob = config.error_probability.unwrap_or(f64::NAN);

        ffi::add_edge(
            self.graph.pin_mut(),
            node1,
            node2,
            observables,
            config.weight,
            error_prob,
            config.merge_strategy.into(),
        )?;

        Ok(())
    }

    /// Add an edge between two nodes (compatibility method)
    ///
    /// # Errors
    /// Returns an error if the edge cannot be added due to graph constraints.
    pub fn add_edge(
        &mut self,
        node1: NodeId,
        node2: NodeId,
        observables: &[ObservableId],
        weight: Option<f64>,
        error_probability: Option<f64>,
        merge_strategy: Option<MergeStrategy>,
    ) -> Result<()> {
        let config = EdgeConfig {
            weight: weight.unwrap_or(1.0),
            error_probability,
            merge_strategy: merge_strategy.unwrap_or(MergeStrategy::SmallestWeight),
        };

        self.add_edge_with_config(node1, node2, observables, config)
    }

    /// Add a boundary edge from a node
    ///
    /// # Errors
    /// Returns an error if the boundary edge cannot be added due to graph constraints.
    pub fn add_boundary_edge(
        &mut self,
        node: NodeId,
        observables: &[ObservableId],
        weight: Option<f64>,
        error_probability: Option<f64>,
        merge_strategy: Option<MergeStrategy>,
    ) -> Result<()> {
        // Note: PyMatching auto-expands nodes and observables, so we don't validate bounds here
        // The C++ layer will handle expansion as needed

        let (weight, error_probability, merge_strategy) =
            Self::normalize_edge_params(weight, error_probability, merge_strategy);

        ffi::add_boundary_edge(
            self.graph.pin_mut(),
            node,
            observables,
            weight,
            error_probability,
            merge_strategy.into(),
        )?;

        Ok(())
    }

    /// Get the number of nodes in the graph
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        ffi::pymatching_get_num_nodes(&self.graph)
    }

    /// Get the number of detectors
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        ffi::pymatching_get_num_detectors(&self.graph)
    }

    /// Get the number of edges
    #[must_use]
    pub fn num_edges(&self) -> usize {
        ffi::pymatching_get_num_edges(&self.graph)
    }

    /// Get the number of observables
    #[must_use]
    pub fn num_observables(&self) -> usize {
        ffi::pymatching_get_num_observables(&self.graph)
    }

    /// Ensure the graph has at least the specified number of observables
    /// This is useful when you need to add edges with observable indices higher than current max
    ///
    /// # Errors
    /// Returns an error if the observable count cannot be expanded.
    pub fn ensure_num_observables(&mut self, min_num_observables: usize) -> Result<()> {
        ffi::pymatching_set_min_num_observables(self.graph.pin_mut(), min_num_observables);
        self.config.num_observables = min_num_observables.max(self.config.num_observables);
        Ok(())
    }

    /// Check if an edge exists between two nodes
    #[must_use]
    pub fn has_edge(&self, node1: NodeId, node2: NodeId) -> bool {
        ffi::has_edge(&self.graph, node1, node2)
    }

    /// Check if a boundary edge exists from a node
    #[must_use]
    pub fn has_boundary_edge(&self, node: NodeId) -> bool {
        ffi::has_boundary_edge(&self.graph, node)
    }

    /// Get edge data
    ///
    /// # Errors
    /// Returns an error if no edge exists between the specified nodes.
    pub fn get_edge_data(&self, node1: NodeId, node2: NodeId) -> Result<EdgeData> {
        Ok(ffi::pymatching_get_edge_data(&self.graph, node1, node2)?.into())
    }

    /// Get boundary edge data
    ///
    /// # Errors
    /// Returns an error if no boundary edge exists from the specified node.
    pub fn get_boundary_edge_data(&self, node: NodeId) -> Result<EdgeData> {
        Ok(ffi::pymatching_get_boundary_edge_data(&self.graph, node)?.into())
    }

    /// Get all edges in the graph
    #[must_use]
    pub fn get_all_edges(&self) -> Vec<EdgeData> {
        ffi::pymatching_get_all_edges(&self.graph)
            .into_iter()
            .map(std::convert::Into::into)
            .collect()
    }

    /// Get boundary nodes
    #[must_use]
    pub fn get_boundary(&self) -> Vec<NodeId> {
        ffi::pymatching_get_boundary(&self.graph)
            .into_iter()
            .collect()
    }

    /// Set boundary nodes
    pub fn set_boundary(&mut self, boundary: &[NodeId]) {
        // PyMatching will auto-expand nodes as needed
        ffi::pymatching_set_boundary(self.graph.pin_mut(), boundary);
    }

    /// Check if a node is a boundary node
    #[must_use]
    pub fn is_boundary_node(&self, node: NodeId) -> bool {
        ffi::pymatching_is_boundary_node(&self.graph, node)
    }

    /// Decode detection events
    ///
    /// Automatically uses the appropriate method based on the number of observables
    ///
    /// # Errors
    ///
    /// Returns an error if detection events are invalid or decoding fails.
    ///
    /// # Panics
    ///
    /// This function will not panic. The internal `expect()` is safe because
    /// the value `(obs_mask >> i) & 1` is always 0 or 1, which fits in a `u8`.
    #[must_use = "The decoding result should be used"]
    pub fn decode(&mut self, detection_events: &[u8]) -> Result<DecodingResult> {
        // Validate detection events length
        let num_detectors = self.num_detectors();
        if detection_events.len() > num_detectors {
            return Err(PyMatchingError::InvalidSyndrome {
                expected: num_detectors,
                actual: detection_events.len(),
            });
        }

        // Use optimized method for ≤64 observables, extended method otherwise
        if self.config.num_observables <= OPTIMIZED_OBSERVABLE_LIMIT {
            let result = ffi::decode_detection_events_64(self.graph.pin_mut(), detection_events)?;

            // The first 8 bytes of observables contain the packed obs_mask
            let num_obs = self.config.num_observables;
            let mut observables = vec![0u8; num_obs];

            // Unpack the obs_mask from the result
            if !result.observables.is_empty() {
                let mut obs_mask = 0u64;
                for i in 0..8.min(result.observables.len()) {
                    obs_mask |= u64::from(result.observables[i]) << (i * BITS_PER_BYTE);
                }

                for (i, obs) in observables[..num_obs].iter_mut().enumerate() {
                    *obs =
                        u8::try_from((obs_mask >> i) & 1).expect("Value 0 or 1 should fit in u8");
                }
            }

            Ok(DecodingResult {
                observable: observables,
                weight: result.weight,
            })
        } else {
            let result =
                ffi::decode_detection_events_extended(self.graph.pin_mut(), detection_events)?;

            Ok(DecodingResult {
                observable: result.observables.into_iter().collect(),
                weight: result.weight,
            })
        }
    }

    /// Decode to matched detection event pairs
    /// Returns pairs of detection events that are matched together
    /// A value of -1 in detector2 indicates matching to boundary
    ///
    /// # Errors
    /// Returns an error if detection events are invalid or matching fails.
    #[must_use = "The matched pairs should be used"]
    pub fn decode_to_matched_pairs(&mut self, detection_events: &[u8]) -> Result<Vec<MatchedPair>> {
        let pairs = ffi::decode_to_matched_pairs(self.graph.pin_mut(), detection_events)?;
        Ok(pairs.into_iter().map(std::convert::Into::into).collect())
    }

    /// Decode to matched detection event pairs as a dictionary/map
    /// Returns a `HashMap` where keys are detection event indices and values are their matched partners
    /// If a detection event is matched to boundary, it maps to None
    ///
    /// This is similar to `PyMatching`'s `decode_to_matched_dets_dict` method
    ///
    /// # Errors
    /// Returns an error if detection events are invalid or matching fails.
    #[must_use = "The matched pairs dictionary should be used"]
    pub fn decode_to_matched_pairs_dict(
        &mut self,
        detection_events: &[u8],
    ) -> Result<HashMap<i64, Option<i64>>> {
        let pairs = self.decode_to_matched_pairs(detection_events)?;
        let mut match_dict = HashMap::new();

        for pair in pairs {
            // Add both directions of the matching
            match_dict.insert(pair.detector1, pair.detector2);
            if let Some(det2) = pair.detector2 {
                match_dict.insert(det2, Some(pair.detector1));
            }
        }

        Ok(match_dict)
    }

    /// Decode to matched detection event pairs as a structured dictionary object
    /// This provides additional convenience methods for working with the matches
    ///
    /// # Errors
    /// Returns an error if detection events are invalid or matching fails.
    pub fn decode_to_matched_dict(&mut self, detection_events: &[u8]) -> Result<MatchedPairsDict> {
        let matches = self.decode_to_matched_pairs_dict(detection_events)?;
        Ok(MatchedPairsDict { matches })
    }

    /// Decode to edges in the matching
    /// Returns the actual edges (pairs of detectors) used in the matching solution
    /// These are detector pairs that form edges, not detection event pairs
    ///
    /// # Errors
    /// Returns an error if detection events are invalid or edge extraction fails.
    pub fn decode_to_edges(&mut self, detection_events: &[u8]) -> Result<Vec<MatchedPair>> {
        let edges = ffi::decode_to_edges(self.graph.pin_mut(), detection_events)?;
        Ok(edges.into_iter().map(std::convert::Into::into).collect())
    }

    /// Batch decode multiple shots (new API)
    ///
    /// # Arguments
    /// * `shots` - Detection events for all shots (flat array)
    /// * `num_shots` - Number of shots to decode
    /// * `num_detectors` - Number of detectors per shot
    /// * `config` - Configuration for batch decoding
    ///
    /// # Errors
    /// Returns an error if shot data is invalid, parameters are inconsistent, or batch decoding fails.
    pub fn decode_batch_with_config(
        &mut self,
        shots: &[u8],
        num_shots: usize,
        num_detectors: usize,
        config: BatchConfig,
    ) -> Result<BatchDecodingResult> {
        // Validate input parameters
        if num_shots == 0 {
            return Ok(BatchDecodingResult {
                predictions: vec![],
                weights: vec![],
                bit_packed: config.bit_packed_output,
            });
        }

        // Validate that num_detectors doesn't exceed actual detector count
        let actual_detectors = self.num_detectors();
        if num_detectors > actual_detectors {
            return Err(PyMatchingError::InvalidSyndrome {
                expected: actual_detectors,
                actual: num_detectors,
            });
        }

        // Calculate expected shots array size
        let expected_size = if config.bit_packed_input {
            num_shots * num_detectors.div_ceil(8)
        } else {
            num_shots * num_detectors
        };

        if shots.len() != expected_size {
            return Err(PyMatchingError::Configuration(format!(
                "shots array length {} doesn't match expected size {} \
                        (num_shots={}, num_detectors={}, bit_packed={})",
                shots.len(),
                expected_size,
                num_shots,
                num_detectors,
                config.bit_packed_input
            )));
        }

        let result = ffi::decode_batch(
            self.graph.pin_mut(),
            shots,
            num_shots,
            num_detectors,
            config.bit_packed_input,
            config.bit_packed_output,
        )?;

        let mut batch_result = BatchDecodingResult::from(result);
        batch_result.bit_packed = config.bit_packed_output;

        // If not returning weights, clear them
        if !config.return_weights {
            batch_result.weights.clear();
        }

        Ok(batch_result)
    }

    /// Find shortest path between two nodes
    /// Returns the sequence of nodes along the shortest path from source to target
    /// If no path exists, returns an empty vector
    ///
    /// # Errors
    /// Returns an error if nodes are out of bounds, graph is empty, or nodes are in different components.
    pub fn get_shortest_path(&mut self, source: usize, target: usize) -> Result<Vec<usize>> {
        // Validate node indices
        let num_nodes = self.num_nodes();
        if source >= num_nodes {
            return Err(PyMatchingError::Configuration(format!(
                "Source node {source} out of bounds. Must be < {num_nodes}"
            )));
        }
        if target >= num_nodes {
            return Err(PyMatchingError::Configuration(format!(
                "Target node {target} out of bounds. Must be < {num_nodes}"
            )));
        }

        // Check if graph has any edges
        if self.num_edges() == 0 {
            return Err(PyMatchingError::Configuration(
                "Cannot find shortest path in empty graph".to_string(),
            ));
        }

        // Quick check: if source == target, return trivial path
        if source == target {
            return Ok(vec![source]);
        }

        // Check connectivity before calling PyMatching to avoid segfault
        if !self.check_nodes_connected(source, target) {
            return Err(PyMatchingError::Configuration(format!(
                "No path exists between nodes {source} and {target}. They are in different connected components."
            )));
        }

        let path = ffi::get_shortest_path(self.graph.pin_mut(), source, target)?;
        Ok(path.into_iter().collect())
    }

    /// Check if two nodes are in the same connected component
    /// This prevents segfaults when calling `shortest_path` on disconnected graphs
    fn check_nodes_connected(&self, source: usize, target: usize) -> bool {
        use std::collections::{HashSet, VecDeque};

        // Get all edges to build adjacency information
        let edges = self.get_all_edges();
        let num_nodes = self.num_nodes();

        // Build adjacency list
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_nodes];

        for edge in edges {
            // Skip boundary edges (node2 is None for boundary edges)
            if let Some(node2) = edge.node2
                && node2 < num_nodes
            {
                adj[edge.node1].insert(node2);
                adj[node2].insert(edge.node1);
            }
        }

        // BFS from source to find if target is reachable
        let mut visited = vec![false; num_nodes];
        let mut queue = VecDeque::new();

        queue.push_back(source);
        visited[source] = true;

        while let Some(node) = queue.pop_front() {
            if node == target {
                return true;
            }

            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        false
    }

    /// Simulate noise on the graph
    /// Returns (errors, syndromes) for the specified number of samples
    /// Note: The `BatchDecodingResult` is repurposed here - predictions contain errors,
    /// and weights contain syndromes (as f64 values)
    ///
    /// # Errors
    /// Returns an error if noise simulation fails or parameters are invalid.
    pub fn add_noise(&self, num_samples: usize, rng_seed: u64) -> Result<NoiseResult> {
        let result = ffi::add_noise(&self.graph, num_samples, rng_seed)?;

        // Convert BatchDecodingResult to proper noise result
        let num_observables = self.num_observables();
        let num_detectors = self.num_detectors();

        let mut errors = Vec::with_capacity(num_samples);
        let mut syndromes = Vec::with_capacity(num_samples);

        // Unpack the results
        for sample in 0..num_samples {
            let error_start = sample * num_observables;
            let error_end = error_start + num_observables;
            let error_vec: Vec<u8> = result.predictions[error_start..error_end].to_vec();
            errors.push(error_vec);

            let syndrome_start = sample * num_detectors;
            let syndrome_end = syndrome_start + num_detectors;
            let syndrome_vec: Vec<u8> = result.weights[syndrome_start..syndrome_end]
                .iter()
                .map(|&w| w.round() as u8)
                .collect();
            syndromes.push(syndrome_vec);
        }

        Ok(NoiseResult { errors, syndromes })
    }

    /// Get edge weight normalising constant
    #[must_use]
    pub fn get_edge_weight_normalising_constant(&self, num_distinct_weights: usize) -> f64 {
        ffi::get_edge_weight_normalising_constant(&self.graph, num_distinct_weights)
    }

    /// Check if all edges have error probabilities
    #[must_use]
    pub fn all_edges_have_error_probabilities(&self) -> bool {
        ffi::all_edges_have_error_probabilities(&self.graph)
    }

    /// Validate detector indices
    ///
    /// # Errors
    /// Returns an error if detection events are invalid or indices are out of bounds.
    pub fn validate_detector_indices(&self, detection_events: &[u8]) -> Result<()> {
        ffi::validate_detector_indices(&self.graph, detection_events)?;
        Ok(())
    }

    /// Load a decoder from a DEM file
    /// This is a convenience method that reads the file and calls `from_dem`
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the DEM is invalid.
    pub fn from_dem_file(path: &Path) -> Result<Self> {
        let dem_string = std::fs::read_to_string(path).map_err(|e| {
            PyMatchingError::Configuration(format!(
                "Failed to read DEM file '{}': {}",
                path.display(),
                e
            ))
        })?;
        Self::from_dem(&dem_string)
    }

    /// Create decoder from a Stim circuit file
    /// Note: This requires the circuit to have detectors and observables defined
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the circuit is invalid.
    pub fn from_stim_circuit_file(path: &Path) -> Result<Self> {
        // For now, we treat this the same as a DEM file
        // In the future, we could add proper Stim circuit parsing if needed
        Self::from_dem_file(path)
    }

    // ===== Random Number Generation =====

    /// Set the random seed for reproducible results
    /// This affects noise simulation and any randomized operations
    ///
    /// # Errors
    /// Returns an error if the seed cannot be set.
    pub fn set_seed(seed: u32) -> Result<()> {
        ffi::pymatching_set_seed(seed)?;
        Ok(())
    }

    /// Randomize the seed using system entropy
    /// This ensures different random sequences in each run
    ///
    /// # Errors
    /// Returns an error if randomization fails.
    pub fn randomize() -> Result<()> {
        ffi::pymatching_randomize()?;
        Ok(())
    }

    /// Generate a random float in the given range [from, to)
    /// Uses the internal MT19937 random number generator
    ///
    /// # Errors
    /// Returns an error if random number generation fails.
    pub fn rand_float(from: f64, to: f64) -> Result<f64> {
        Ok(ffi::pymatching_rand_float(from, to)?)
    }

    // Convenience methods

    /// Get edge data between two nodes if edge exists
    #[must_use]
    pub fn get_edge_between(&self, node1: usize, node2: usize) -> Option<EdgeData> {
        if self.has_edge(node1, node2) {
            self.get_edge_data(node1, node2).ok()
        } else {
            None
        }
    }

    /// Check if the graph is connected
    ///
    /// # Errors
    /// Returns an error if connectivity check fails.
    pub fn is_connected(&self) -> Result<bool> {
        // A graph is connected if there's at most one component
        // (excluding isolated nodes)
        let num_nodes = self.num_nodes();
        if num_nodes <= 1 {
            return Ok(true);
        }

        // Check connectivity from node 0 to all others
        for target in 1..num_nodes {
            if self.check_nodes_connected(0, target) {
                continue;
            }
            // If we can't reach this node, graph is disconnected
            return Ok(false);
        }
        Ok(true)
    }

    /// Get the number of connected components
    ///
    /// # Errors
    /// Returns an error if component counting fails.
    pub fn count_components(&self) -> Result<usize> {
        let num_nodes = self.num_nodes();
        if num_nodes == 0 {
            return Ok(0);
        }

        let mut visited = vec![false; num_nodes];
        let mut components = 0;

        for start in 0..num_nodes {
            if visited[start] {
                continue;
            }

            // Start a new component
            components += 1;
            visited[start] = true;

            // Mark all nodes connected to start
            for (target, visit_status) in visited.iter_mut().enumerate().skip(start + 1) {
                if !*visit_status && self.check_nodes_connected(start, target) {
                    *visit_status = true;
                }
            }
        }

        Ok(components)
    }

    /// Create a decoder with uniform error probability on all edges
    ///
    /// # Errors
    /// Returns an error if decoder creation or edge addition fails.
    pub fn with_uniform_error_rate(
        num_nodes: usize,
        edges: &[(NodeId, NodeId)],
        error_rate: f64,
    ) -> Result<Self> {
        let config = PyMatchingConfig {
            num_nodes: Some(num_nodes),
            num_observables: edges.len(),
            num_neighbours: None,
        };

        let mut decoder = Self::new(config)?;

        for (i, &(n1, n2)) in edges.iter().enumerate() {
            decoder.add_edge(n1, n2, &[i], None, Some(error_rate), None)?;
        }

        Ok(decoder)
    }

    /// Get a summary of the graph structure
    #[must_use]
    pub fn graph_summary(&self) -> String {
        let num_nodes = self.num_nodes();
        let num_edges = self.num_edges();
        let num_detectors = self.num_detectors();
        let num_boundary = self.boundary_nodes().count();
        let num_observables = self.num_observables();
        let connected = if num_nodes > 0 {
            self.is_connected().unwrap_or(false)
        } else {
            true
        };

        format!(
            "PyMatchingDecoder {{ nodes: {num_nodes}, edges: {num_edges}, detectors: {num_detectors}, boundary: {num_boundary}, observables: {num_observables}, connected: {connected} }}"
        )
    }
}

/// Merge strategy for handling parallel edges
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Disallow parallel edges (error if edge already exists)
    Disallow,
    /// Treat parallel edges as independent error mechanisms
    Independent,
    /// Keep the edge with smallest weight
    SmallestWeight,
    /// Keep the original edge
    KeepOriginal,
    /// Replace with the new edge
    Replace,
}

impl From<MergeStrategy> for ffi::MergeStrategy {
    fn from(strategy: MergeStrategy) -> Self {
        match strategy {
            MergeStrategy::Disallow => ffi::MergeStrategy::Disallow,
            MergeStrategy::Independent => ffi::MergeStrategy::Independent,
            MergeStrategy::SmallestWeight => ffi::MergeStrategy::SmallestWeight,
            MergeStrategy::KeepOriginal => ffi::MergeStrategy::KeepOriginal,
            MergeStrategy::Replace => ffi::MergeStrategy::Replace,
        }
    }
}

/// Configuration for adding an edge
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeConfig {
    pub weight: f64,
    pub error_probability: Option<f64>,
    pub merge_strategy: MergeStrategy,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            error_probability: None,
            merge_strategy: MergeStrategy::SmallestWeight,
        }
    }
}

/// Edge data structure
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeData {
    pub node1: usize,
    pub node2: Option<usize>, // None for boundary edges
    pub observables: Vec<usize>,
    pub weight: f64,
    pub error_probability: f64,
}

impl fmt::Display for EdgeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node2 {
            Some(n2) => write!(
                f,
                "Edge({} <-> {}, w={:.3}, p={:.3}, obs={:?})",
                self.node1, n2, self.weight, self.error_probability, self.observables
            ),
            None => write!(
                f,
                "BoundaryEdge({} <-> boundary, w={:.3}, p={:.3}, obs={:?})",
                self.node1, self.weight, self.error_probability, self.observables
            ),
        }
    }
}

impl From<ffi::EdgeData> for EdgeData {
    fn from(data: ffi::EdgeData) -> Self {
        Self {
            node1: data.node1,
            node2: if data.node2 == usize::MAX {
                None
            } else {
                Some(data.node2)
            },
            observables: data.observables.into_iter().collect(),
            weight: data.weight,
            error_probability: data.error_probability,
        }
    }
}

/// Matched pair structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchedPair {
    pub detector1: i64,
    pub detector2: Option<i64>, // None for boundary
}

impl From<ffi::MatchedPair> for MatchedPair {
    fn from(pair: ffi::MatchedPair) -> Self {
        Self {
            detector1: pair.detector1,
            detector2: if pair.detector2 == -1 {
                None
            } else {
                Some(pair.detector2)
            },
        }
    }
}

/// Configuration for batch decoding
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchConfig {
    /// Whether input shots are bit-packed
    pub bit_packed_input: bool,
    /// Whether output predictions should be bit-packed
    pub bit_packed_output: bool,
    /// Whether to return weights for each shot
    pub return_weights: bool,
}

/// Configuration for creating decoder from check matrix
#[derive(Debug, Clone)]
pub struct CheckMatrixConfig {
    /// Number of repetitions (for temporal codes)
    pub repetitions: usize,
    /// Error probabilities for each column
    pub error_probabilities: Option<Vec<f64>>,
    /// Timelike weights for repetition rounds
    pub timelike_weights: Option<Vec<f64>>,
    /// Measurement error probabilities for each detector
    pub measurement_error_probabilities: Option<Vec<f64>>,
    /// Whether to use virtual boundary nodes
    pub use_virtual_boundary: bool,
    /// Internal field for weights (used by legacy APIs)
    #[doc(hidden)]
    pub weights: Option<Vec<f64>>,
}

impl Default for CheckMatrixConfig {
    fn default() -> Self {
        Self {
            repetitions: 1,
            error_probabilities: None,
            timelike_weights: None,
            measurement_error_probabilities: None,
            use_virtual_boundary: true,
            weights: None,
        }
    }
}

/// Batch decoding result
#[derive(Debug)]
pub struct BatchDecodingResult {
    pub predictions: Vec<Vec<u8>>, // Predictions for each shot
    pub weights: Vec<f64>,         // Weight for each shot (empty if not requested)
    pub bit_packed: bool,          // Whether predictions are bit-packed
}

/// Noise simulation result
#[derive(Debug)]
pub struct NoiseResult {
    pub errors: Vec<Vec<u8>>,    // Error patterns for each sample
    pub syndromes: Vec<Vec<u8>>, // Resulting syndromes for each sample
}

/// Alternative matched pairs representation using indices
/// This provides a more convenient format for some use cases
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedPairsDict {
    /// Map from detection event index to its matched partner (or None for boundary)
    pub matches: HashMap<i64, Option<i64>>,
}

impl MatchedPairsDict {
    /// Get the match for a specific detection event
    #[must_use]
    pub fn get_match(&self, detection_event: i64) -> Option<Option<i64>> {
        self.matches.get(&detection_event).copied()
    }

    /// Check if a detection event is matched to boundary
    #[must_use]
    pub fn is_matched_to_boundary(&self, detection_event: i64) -> bool {
        matches!(self.matches.get(&detection_event), Some(None))
    }

    /// Get all detection events matched to boundary
    #[must_use]
    pub fn boundary_matches(&self) -> Vec<i64> {
        self.matches
            .iter()
            .filter_map(|(&k, &v)| if v.is_none() { Some(k) } else { None })
            .collect()
    }
}

impl From<ffi::BatchDecodingResult> for BatchDecodingResult {
    fn from(result: ffi::BatchDecodingResult) -> Self {
        // The result from FFI is already in the requested format
        // We just need to reshape it by shots
        let num_shots = result.weights.len();
        let bytes_per_shot = result.predictions.len().checked_div(num_shots).unwrap_or(0);

        let predictions = if bytes_per_shot > 0 {
            result
                .predictions
                .chunks(bytes_per_shot)
                .map(<[u8]>::to_vec)
                .collect()
        } else {
            vec![]
        };

        Self {
            predictions,
            weights: result.weights.into_iter().collect(),
            bit_packed: true, // The C++ implementation determines this
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let config = PyMatchingConfig {
            num_nodes: Some(10),
            num_observables: 2,
            ..Default::default()
        };

        let decoder = PyMatchingDecoder::new(config).unwrap();
        assert_eq!(decoder.num_nodes(), 10);
        // PyMatching defaults to 64 observables if num_observables <= 64
        assert!(decoder.num_observables() >= 2);
    }

    #[test]
    fn test_add_edges() {
        let config = PyMatchingConfig {
            num_nodes: Some(5),
            num_observables: 2,
            ..Default::default()
        };

        let mut decoder = PyMatchingDecoder::new(config).unwrap();

        // Add regular edge
        decoder.add_edge(0, 1, &[0], Some(1.5), None, None).unwrap();
        assert!(decoder.has_edge(0, 1));

        // Add boundary edge
        decoder
            .add_boundary_edge(2, &[1], Some(2.0), None, None)
            .unwrap();
        assert!(decoder.has_boundary_edge(2));
    }

    #[test]
    fn test_batch_decode_formats() {
        let config = PyMatchingConfig {
            num_nodes: Some(5),
            num_observables: 2,
            ..Default::default()
        };

        let mut decoder = PyMatchingDecoder::new(config).unwrap();

        // Create a simple matching graph with boundary
        decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
        decoder.add_edge(2, 3, &[1], Some(1.0), None, None).unwrap();
        decoder
            .add_boundary_edge(4, &[], Some(1.0), None, None)
            .unwrap();
        decoder.set_boundary(&[4]);

        // Test unpacked format with valid syndromes
        let shots = vec![0, 0, 0, 0, 0, 0]; // 2 shots of 3 detectors each (all zero)
        let config = BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: true,
        };
        let result = decoder
            .decode_batch_with_config(
                &shots, 2, // num_shots
                3, // num_detectors
                config,
            )
            .unwrap();

        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.weights.len(), 2);
        assert!(!result.bit_packed);

        // Test bit-packed shots
        let packed_shots = vec![0b000, 0b000]; // Same as above but bit-packed
        let packed_config = BatchConfig {
            bit_packed_input: true,
            bit_packed_output: true,
            return_weights: true,
        };
        let result_packed = decoder
            .decode_batch_with_config(
                &packed_shots,
                2, // num_shots
                3, // num_detectors
                packed_config,
            )
            .unwrap();

        assert_eq!(result_packed.predictions.len(), 2);
        assert_eq!(result_packed.weights.len(), 2);
        assert!(result_packed.bit_packed);
    }

    #[test]
    fn test_from_check_matrix() {
        // Test creating decoder from parity check matrix
        // H = [[1, 1, 0],
        //      [0, 1, 1]]
        let entries = vec![
            (0, 0, 1), // H[0,0] = 1
            (0, 1, 1), // H[0,1] = 1
            (1, 1, 1), // H[1,1] = 1
            (1, 2, 1), // H[1,2] = 1
        ];

        let weights = vec![1.0, 2.0, 3.0];
        let matrix = CheckMatrix::from_triplets(entries, 2, 3)
            .with_weights(weights)
            .unwrap();
        let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

        // Check basic properties
        // PyMatching defaults to 64 observables if num_observables <= 64
        assert!(decoder.num_observables() >= 3);
        // The actual number of nodes depends on whether boundary edges were created
        // For this check matrix, we have 2 detector nodes
        assert!(decoder.num_nodes() >= 2);
    }

    #[test]
    fn test_from_check_matrix_with_repetitions() {
        // Test creating decoder with repetitions (timelike edges)
        let entries = vec![
            (0, 0, 1), // H[0,0] = 1
            (0, 1, 1), // H[0,1] = 1
            (1, 1, 1), // H[1,1] = 1
            (1, 2, 1), // H[1,2] = 1
        ];

        let matrix = CheckMatrix::from_triplets(entries, 2, 3)
            .with_weights(vec![1.0, 2.0, 3.0])
            .unwrap();

        let config = CheckMatrixConfig {
            repetitions: 3,
            error_probabilities: None,
            timelike_weights: Some(vec![0.5, 1.5]), // timelike weights for each row
            measurement_error_probabilities: Some(vec![0.1, 0.2]), // measurement error probabilities
            use_virtual_boundary: false,
            weights: None, // Now in the matrix
        };
        let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

        // Check basic properties
        // PyMatching defaults to 64 observables if num_observables <= 64
        assert!(decoder.num_observables() >= 3);
        // 2 detectors * 3 repetitions = 6 nodes minimum
        assert!(decoder.num_nodes() >= 6);
        // Should have spacelike edges + timelike edges
        assert!(decoder.num_edges() > 0);
    }
}

use pecos_decoder_core::DecodingResultTrait;

impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        // PyMatching always returns a result, success is implicit
        true
    }

    fn cost(&self) -> Option<f64> {
        Some(self.weight)
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_check_matrix_config_api() {
        // Test the new config-based API
        let entries = vec![
            (0, 0, 1),
            (0, 1, 1), // Check 0: qubits 0,1
            (1, 1, 1),
            (1, 2, 1), // Check 1: qubits 1,2
        ];

        // Test with explicit config and matrix with weights
        let matrix = CheckMatrix::from_triplets(entries.clone(), 2, 3)
            .with_weights(vec![1.0, 2.0, 1.0])
            .unwrap();
        let config = CheckMatrixConfig {
            repetitions: 1,
            error_probabilities: None,
            timelike_weights: None,
            measurement_error_probabilities: None,
            use_virtual_boundary: true,
            weights: None,
        };

        let mut decoder =
            PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

        let syndrome = vec![1, 0];
        let result = decoder.decode(&syndrome).unwrap();
        // Should have 3 observables (num_cols)
        assert_eq!(result.observable.len(), 3);

        // Test with default config
        let matrix2 = CheckMatrix::from_triplets(entries.clone(), 2, 3);
        let default_config = CheckMatrixConfig::default();
        let mut decoder2 =
            PyMatchingDecoder::from_check_matrix_with_config(&matrix2, default_config).unwrap();

        let result2 = decoder2.decode(&syndrome).unwrap();
        assert_eq!(result2.observable.len(), 3);

        // Verify default config with virtual boundary
        let matrix3 = CheckMatrix::from_triplets(entries, 2, 3);
        let mut decoder_default = PyMatchingDecoder::from_check_matrix_with_config(
            &matrix3,
            CheckMatrixConfig {
                use_virtual_boundary: true,
                ..Default::default()
            },
        )
        .unwrap();

        let result_default = decoder_default.decode(&syndrome).unwrap();
        assert_eq!(result_default.observable.len(), 3);
    }

    #[test]
    fn test_from_check_matrix_simple() {
        // Test the new simple API
        let entries = vec![
            (0, 0, 1),
            (0, 1, 1), // Check 0: qubits 0,1
            (1, 1, 1),
            (1, 2, 1), // Check 1: qubits 1,2
        ];

        // Using the simple API with uniform weights
        let weights = vec![1.0; 3]; // uniform weights
        let matrix = CheckMatrix::from_triplets(entries.clone(), 2, 3)
            .with_weights(weights)
            .unwrap();
        let mut decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

        let syndrome = vec![1, 0];
        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable.len(), 3);

        // Compare with using default config explicitly
        let matrix2 = CheckMatrix::from_triplets(entries, 2, 3);
        let mut decoder2 = PyMatchingDecoder::from_check_matrix_with_config(
            &matrix2,
            CheckMatrixConfig::default(),
        )
        .unwrap();

        let result2 = decoder2.decode(&syndrome).unwrap();
        assert_eq!(result2.observable.len(), 3);

        // Both results should be valid (may not be identical due to different weights)
        assert_eq!(result.observable.len(), result2.observable.len());
    }

    #[test]
    fn test_new_sparse_check_matrix_api() {
        // Test the new SparseCheckMatrix API
        let entries = vec![
            (0, 0, 1),
            (0, 1, 1), // Check 0: qubits 0,1
            (1, 1, 1),
            (1, 2, 1), // Check 1: qubits 1,2
        ];

        // Test creating matrix without weights
        let matrix_no_weights = CheckMatrix::from_triplets(entries.clone(), 2, 3);
        let mut decoder1 = PyMatchingDecoder::from_check_matrix(&matrix_no_weights).unwrap();

        // Test creating matrix with weights using fluent API
        let matrix_with_weights = CheckMatrix::from_triplets(entries.clone(), 2, 3)
            .with_weights(vec![1.0, 2.0, 3.0])
            .unwrap();
        let mut decoder2 = PyMatchingDecoder::from_check_matrix(&matrix_with_weights).unwrap();

        // Test that both work
        let syndrome = vec![1, 0];
        let result1 = decoder1.decode(&syndrome).unwrap();
        let result2 = decoder2.decode(&syndrome).unwrap();

        assert_eq!(result1.observable.len(), 3);
        assert_eq!(result2.observable.len(), 3);

        // Test validation
        matrix_no_weights.validate().unwrap();
        matrix_with_weights.validate().unwrap();

        // Test accessors
        assert_eq!(matrix_no_weights.rows(), 2);
        assert_eq!(matrix_no_weights.cols(), 3);
        assert!(matrix_no_weights.weights().is_none());

        assert_eq!(matrix_with_weights.rows(), 2);
        assert_eq!(matrix_with_weights.cols(), 3);
        assert_eq!(matrix_with_weights.weights().unwrap(), &[1.0, 2.0, 3.0]);

        // Test with configuration
        let config = CheckMatrixConfig {
            repetitions: 2,
            ..Default::default()
        };
        let decoder3 =
            PyMatchingDecoder::from_check_matrix_with_config(&matrix_with_weights, config).unwrap();
        assert!(decoder3.num_nodes() >= 4); // 2 checks * 2 repetitions
    }
}
