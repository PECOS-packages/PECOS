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

    /// The error for a column count too large to hold one entry per column.
    fn too_many_columns(cols: usize) -> PyMatchingError {
        PyMatchingError::Configuration(format!(
            "cannot allocate one entry per column for {cols} columns; pass the matrix's real \
             column count"
        ))
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
        let mut col_counts = Vec::new();
        col_counts
            .try_reserve_exact(self.cols)
            .map_err(|_| Self::too_many_columns(self.cols))?;
        col_counts.resize(self.cols, 0usize);
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

/// Allocate `rows` zeroed rows of `row_len` entries for the C++ bridge to fill.
///
/// A `rust::Vec` that fails to grow inside the bridge ends the process, so
/// output sized by the graph or by a caller count is allocated here, where a
/// failure is an error. `row_len` comes from the graph's count of `row_unit`s:
/// if one row cannot be allocated on its own the graph is too large, a
/// [`PyMatchingError::InvalidGraph`]; otherwise only `rows` is too large, a
/// [`PyMatchingError::Configuration`] naming the `rows_name` argument.
pub(crate) fn zeroed_rows<T: Copy + Default>(
    row_len: usize,
    rows: usize,
    row_unit: &str,
    rows_name: &str,
) -> Result<Vec<T>> {
    let mut buffer = Vec::new();
    if let Some(len) = row_len.checked_mul(rows)
        && buffer.try_reserve_exact(len).is_ok()
    {
        buffer.resize(len, T::default());
        return Ok(buffer);
    }
    if Vec::<T>::new().try_reserve_exact(row_len).is_err() {
        return Err(PyMatchingError::InvalidGraph(format!(
            "cannot allocate one row of {row_len} output entries, sized by the graph's \
             {row_unit} count; build the graph with fewer {row_unit}s"
        )));
    }
    Err(PyMatchingError::Configuration(format!(
        "cannot allocate {rows} rows of {row_len} output entries; pass a smaller {rows_name}"
    )))
}

/// Split `flat`, `rows` rows of `row_len` entries, into one `Vec` per row that
/// keeps the first `width` entries.
///
/// Every allocation is fallible: the split doubles the output's memory, and
/// the row count comes from the caller.
///
/// # Errors
///
/// Returns [`PyMatchingError::Configuration`] naming the `rows_name` argument
/// if the rows cannot be allocated.
fn split_rows<T: Copy>(
    flat: &[T],
    row_len: usize,
    width: usize,
    rows: usize,
    rows_name: &str,
) -> Result<Vec<Vec<T>>> {
    // Free the rows split so far before the error message is allocated.
    let split = || -> Option<Vec<Vec<T>>> {
        let kept = width.min(row_len);
        let mut split = Vec::new();
        split.try_reserve_exact(rows).ok()?;
        for index in 0..rows {
            let mut row = Vec::new();
            row.try_reserve_exact(kept).ok()?;
            row.extend_from_slice(&flat[index * row_len..][..kept]);
            split.push(row);
        }
        Some(split)
    };
    split().ok_or_else(|| {
        PyMatchingError::Configuration(format!(
            "cannot allocate {rows} output rows; pass a smaller {rows_name}"
        ))
    })
}

/// Complete `PyMatching` decoder with full API
pub struct PyMatchingDecoder {
    graph: UniquePtr<ffi::PyMatchingGraph>,
    num_detectors: usize,
    num_nodes: usize,
    config: PyMatchingConfig,
}

impl fmt::Display for PyMatchingDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.graph_summary())
    }
}

impl PyMatchingDecoder {
    fn validate_error_probability(error_probability: f64) -> Result<()> {
        if !(0.0..1.0).contains(&error_probability) || error_probability == 0.0 {
            return Err(PyMatchingError::Configuration(
                "error_probability must be finite and strictly between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }

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
    /// Returns an error if `num_nodes` is not specified in the configuration, or
    /// [`PyMatchingError::InvalidGraph`] if the graph cannot be created, for
    /// example a node count too large to allocate.
    pub fn new(config: PyMatchingConfig) -> Result<Self> {
        let graph = if let Some(num_nodes) = config.num_nodes {
            if config.num_observables <= OPTIMIZED_OBSERVABLE_LIMIT {
                ffi::create_pymatching_graph(num_nodes)?
            } else {
                ffi::create_pymatching_graph_with_observables(num_nodes, config.num_observables)?
            }
        } else {
            return Err(PyMatchingError::Configuration(
                "num_nodes must be specified in config".to_string(),
            ));
        };

        let num_detectors = ffi::pymatching_get_num_detectors(&graph);
        let num_nodes = ffi::pymatching_get_num_nodes(&graph);
        Ok(Self {
            graph,
            num_detectors,
            num_nodes,
            config,
        })
    }

    /// Create a decoder from a Detector Error Model (DEM) string
    ///
    /// # Errors
    /// Returns an error if the DEM string is invalid or cannot be parsed.
    pub fn from_dem(dem_string: &str) -> Result<Self> {
        pecos_decoder_core::dem::grammar::validate_dem_text(dem_string)?;
        let graph = ffi::create_pymatching_graph_from_dem(dem_string)?;

        // Query graph for configuration
        let num_nodes = ffi::pymatching_get_num_nodes(&graph);
        let num_observables = ffi::pymatching_get_num_observables(&graph);

        let config = PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(num_nodes),
            num_observables,
        };

        let num_detectors = ffi::pymatching_get_num_detectors(&graph);
        Ok(Self {
            graph,
            num_detectors,
            num_nodes,
            config,
        })
    }

    /// Create a decoder from a DEM and override every graph edge's error probability.
    ///
    /// The graph structure is preserved. Each matching weight is recomputed
    /// from the supplied probability, matching the builder's probability semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the DEM is invalid, the probability is outside
    /// `(0, 1)`, or an edge cannot be updated.
    pub fn from_dem_with_error_probability(
        dem_string: &str,
        error_probability: f64,
    ) -> Result<Self> {
        let mut decoder = Self::from_dem(dem_string)?;
        decoder.set_all_error_probabilities(error_probability)?;
        Ok(decoder)
    }

    /// Replace the error probability and derived matching weight on every edge.
    ///
    /// # Errors
    ///
    /// Returns an error if `error_probability` is outside `(0, 1)` or an edge
    /// cannot be updated.
    pub fn set_all_error_probabilities(&mut self, error_probability: f64) -> Result<()> {
        Self::validate_error_probability(error_probability)?;
        ffi::pymatching_set_all_error_probabilities(self.graph.pin_mut(), error_probability);
        Ok(())
    }

    /// Create a decoder from a DEM string with correlation support
    ///
    /// When `enable_correlations` is true, the decoder tracks edge correlations
    /// during graph construction and uses them during decoding.
    ///
    /// # Errors
    /// Returns an error if the DEM string is invalid or cannot be parsed, or
    /// [`PyMatchingError::Configuration`] if correlations are enabled and an
    /// error mechanism has probability above 0.5, which upstream `PyMatching`
    /// does not support with correlations.
    pub fn from_dem_with_correlations(dem_string: &str, enable_correlations: bool) -> Result<Self> {
        use pecos_decoder_core::dem::grammar::{Kind, parse_line};
        for line in dem_string.lines() {
            if let Some(instruction) = parse_line(line)?
                && enable_correlations
                && instruction.kind == Kind::Error
                && let Some(&probability) = instruction.args.first()
                && probability > 0.5
            {
                return Err(PyMatchingError::Configuration(format!(
                    "DEM error probability {probability} is above 0.5, which correlated \
                     PyMatching does not support; keep every error probability at most 0.5 \
                     or disable correlations"
                )));
            }
        }
        let graph = ffi::create_pymatching_graph_from_dem_with_correlations(
            dem_string,
            enable_correlations,
        )?;

        let num_nodes = ffi::pymatching_get_num_nodes(&graph);
        let num_observables = ffi::pymatching_get_num_observables(&graph);

        let config = PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(num_nodes),
            num_observables,
        };

        let num_detectors = ffi::pymatching_get_num_detectors(&graph);
        Ok(Self {
            graph,
            num_detectors,
            num_nodes,
            config,
        })
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
        if config.repetitions == 0 {
            return Err(PyMatchingError::Configuration(
                "repetitions must be at least 1, got 0".to_string(),
            ));
        }
        // Every node index below (`row + rep * rows` with `row < rows` and
        // `rep < repetitions`) is below `total_nodes`, so this is the only
        // product that can overflow.
        let total_nodes = matrix.rows.checked_mul(config.repetitions).ok_or_else(|| {
            PyMatchingError::Configuration(format!(
                "{} rows times {} repetitions overflows usize; pass fewer repetitions",
                matrix.rows, config.repetitions
            ))
        })?;
        for (name, values) in [
            ("weights", &config.weights),
            ("error_probabilities", &config.error_probabilities),
        ] {
            if let Some(values) = values
                && values.len() != matrix.cols
            {
                return Err(PyMatchingError::Configuration(format!(
                    "{name} has length {} but must equal the number of columns ({})",
                    values.len(),
                    matrix.cols
                )));
            }
        }

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
            decoder.set_boundary(&boundary_nodes)?;
        }

        // Group matrix entries by column
        let mut col_entries: Vec<Vec<usize>> = Vec::new();
        col_entries
            .try_reserve_exact(matrix.cols)
            .map_err(|_| CheckMatrix::too_many_columns(matrix.cols))?;
        col_entries.resize_with(matrix.cols, Vec::new);
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
    /// Returns [`PyMatchingError::Configuration`] if `repetitions` is 0, if rows
    /// times `repetitions` overflows `usize`, or if `weights` or
    /// `error_probabilities` does not have one entry per column.
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

    /// Reject `usize::MAX` as an edge's node.
    ///
    /// Upstream grows its node list to `node + 1`, which wraps for `usize::MAX`,
    /// and then indexes past the end of the list.
    fn check_edge_node(node: NodeId) -> Result<()> {
        if node == BOUNDARY_NODE_MARKER {
            return Err(PyMatchingError::InvalidGraph(format!(
                "node {node} is out of range: BOUNDARY_NODE_MARKER (usize::MAX) is not a node; \
                 use add_boundary_edge to connect a node to the boundary"
            )));
        }
        Ok(())
    }

    /// The observable count that covers `observables`, or `None` if empty.
    ///
    /// # Errors
    ///
    /// Returns [`PyMatchingError::InvalidGraph`] for observable `usize::MAX`,
    /// whose count `usize::MAX + 1` does not exist.
    fn observable_count_for(observables: &[ObservableId]) -> Result<Option<usize>> {
        observables
            .iter()
            .max()
            .map(|&max| {
                max.checked_add(1).ok_or_else(|| {
                    PyMatchingError::InvalidGraph(format!(
                        "observable {max} is out of range; observable indices must be below \
                         usize::MAX"
                    ))
                })
            })
            .transpose()
    }

    /// Widen the decoded observable count to `count`, from
    /// [`Self::observable_count_for`].
    ///
    /// `decode` returns `config.num_observables` entries, so it must cover every
    /// observable an edge carries or flips on it would be dropped. The backend graph
    /// grows on its own only when a new edge is added, not when a parallel edge is
    /// merged, so both counts are raised together through `ensure_num_observables`.
    fn include_observables(&mut self, count: Option<usize>) -> Result<()> {
        match count {
            Some(count) if count > self.config.num_observables => {
                self.ensure_num_observables(count)
            }
            _ => Ok(()),
        }
    }

    /// Add an edge between two nodes with configuration
    ///
    /// `PyMatching` grows the graph to hold both nodes and every observable.
    ///
    /// # Errors
    /// Returns [`PyMatchingError::InvalidGraph`], before the edge is added, if a
    /// node is [`BOUNDARY_NODE_MARKER`] (`usize::MAX`), which is not a node (use
    /// [`Self::add_boundary_edge`] for a boundary edge), or an observable is
    /// `usize::MAX`. Also returns [`PyMatchingError::InvalidGraph`] if upstream
    /// rejects the edge, for example a node too large to allocate, or an error if
    /// the edge conflicts with the merge strategy.
    pub fn add_edge_with_config(
        &mut self,
        node1: NodeId,
        node2: NodeId,
        observables: &[ObservableId],
        config: EdgeConfig,
    ) -> Result<()> {
        Self::check_edge_node(node1)?;
        Self::check_edge_node(node2)?;
        let observable_count = Self::observable_count_for(observables)?;
        let error_prob = config.error_probability.unwrap_or(f64::NAN);

        let result = ffi::add_edge(
            self.graph.pin_mut(),
            node1,
            node2,
            observables,
            config.weight,
            error_prob,
            config.merge_strategy.into(),
        );
        self.num_detectors = ffi::pymatching_get_num_detectors(&self.graph);
        self.num_nodes = ffi::pymatching_get_num_nodes(&self.graph);
        result?;
        self.include_observables(observable_count)?;

        Ok(())
    }

    /// Add an edge between two nodes (compatibility method)
    ///
    /// # Errors
    /// Returns the errors of [`Self::add_edge_with_config`].
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
    /// `PyMatching` grows the graph to hold the node and every observable.
    ///
    /// # Errors
    /// Returns [`PyMatchingError::InvalidGraph`], before the edge is added, if
    /// `node` is [`BOUNDARY_NODE_MARKER`] (`usize::MAX`), which is not a node, or
    /// an observable is `usize::MAX`. Also returns
    /// [`PyMatchingError::InvalidGraph`] if upstream rejects the edge, for example
    /// a node too large to allocate, or an error if the edge conflicts with the
    /// merge strategy.
    pub fn add_boundary_edge(
        &mut self,
        node: NodeId,
        observables: &[ObservableId],
        weight: Option<f64>,
        error_probability: Option<f64>,
        merge_strategy: Option<MergeStrategy>,
    ) -> Result<()> {
        Self::check_edge_node(node)?;
        let observable_count = Self::observable_count_for(observables)?;
        let (weight, error_probability, merge_strategy) =
            Self::normalize_edge_params(weight, error_probability, merge_strategy);

        let result = ffi::add_boundary_edge(
            self.graph.pin_mut(),
            node,
            observables,
            weight,
            error_probability,
            merge_strategy.into(),
        );
        self.num_detectors = ffi::pymatching_get_num_detectors(&self.graph);
        self.num_nodes = ffi::pymatching_get_num_nodes(&self.graph);
        result?;
        self.include_observables(observable_count)?;

        Ok(())
    }

    /// Get the number of nodes in the graph
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get the number of detectors
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Get the number of edges
    #[must_use]
    pub fn num_edges(&self) -> usize {
        ffi::pymatching_get_num_edges(&self.graph)
    }

    /// Get the number of observables
    ///
    /// This is the graph's count. For a graph built with [`Self::new`] or from
    /// a check matrix with at most 64 observables it is padded to 64; use
    /// [`Self::num_decoded_observables`] for the length [`Self::decode`] returns.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        ffi::pymatching_get_num_observables(&self.graph)
    }

    /// Number of entries in [`DecodingResult::observable`] from [`Self::decode`]
    ///
    /// With at most 64 configured observables, `decode` returns the configured
    /// count; otherwise it returns the graph's [`Self::num_observables`].
    #[must_use]
    pub fn num_decoded_observables(&self) -> usize {
        if self.config.num_observables <= OPTIMIZED_OBSERVABLE_LIMIT {
            self.config.num_observables
        } else {
            self.num_observables()
        }
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

    /// Set boundary nodes. `PyMatching` grows the graph to hold every node listed.
    ///
    /// # Errors
    ///
    /// Returns [`PyMatchingError::InvalidGraph`] if a node is `usize::MAX`, whose
    /// node count `usize::MAX + 1` does not exist, or if the graph cannot grow to
    /// hold a node.
    pub fn set_boundary(&mut self, boundary: &[NodeId]) -> Result<()> {
        if boundary.contains(&usize::MAX) {
            return Err(PyMatchingError::InvalidGraph(format!(
                "boundary node {} is out of range; boundary nodes must be below usize::MAX",
                usize::MAX
            )));
        }
        let result = ffi::pymatching_set_boundary(self.graph.pin_mut(), boundary);
        self.num_detectors = ffi::pymatching_get_num_detectors(&self.graph);
        self.num_nodes = ffi::pymatching_get_num_nodes(&self.graph);
        Ok(result?)
    }

    /// Check if a node is a boundary node.
    ///
    /// A node outside the graph is not a boundary node, as [`Self::has_edge`]
    /// reports no edge for one; `usize::MAX` is upstream's virtual boundary node.
    #[must_use]
    pub fn is_boundary_node(&self, node: NodeId) -> bool {
        // Upstream indexes its node list unchecked, so check the range here.
        node == usize::MAX
            || (node < ffi::pymatching_get_num_nodes(&self.graph)
                && ffi::pymatching_is_boundary_node(&self.graph, node))
    }

    /// Reject a dense syndrome width outside the range upstream `PyMatching` accepts.
    ///
    /// Dense entry `i` is node `i`, and [`Self::num_detectors`] counts only the
    /// non-boundary nodes. When a boundary node sits below a detector, the
    /// detector is addressable only by a syndrome longer than `num_detectors`,
    /// so, as upstream does, any width from `num_detectors` to
    /// [`Self::num_nodes`] (boundary entries included) is accepted. Upstream
    /// creates no detection event at a boundary node.
    pub(crate) fn validate_syndrome_width(&self, width: usize) -> Result<()> {
        let num_nodes = self.num_nodes();
        if (self.num_detectors..=num_nodes).contains(&width) {
            return Ok(());
        }
        if num_nodes == self.num_detectors {
            return Err(PyMatchingError::InvalidSyndrome {
                expected: self.num_detectors,
                actual: width,
            });
        }
        Err(PyMatchingError::Dem(
            pecos_decoder_core::DecoderError::InvalidSyndrome(format!(
                "dense syndrome has {width} entries, but this graph accepts {} (one per \
                 detector) to {num_nodes} (one per node, boundary nodes included)",
                self.num_detectors
            )),
        ))
    }

    /// Reject a dense syndrome whose length [`Self::validate_syndrome_width`] rejects.
    pub(crate) fn validate_syndrome(&self, syndrome: &[u8]) -> Result<()> {
        self.validate_syndrome_width(syndrome.len())
    }

    /// Decode detection events
    ///
    /// Automatically uses the appropriate method based on the number of observables
    ///
    /// # Errors
    ///
    /// Returns an invalid-syndrome error unless `detection_events` has between
    /// [`Self::num_detectors`] and [`Self::num_nodes`] entries.
    /// Entry `i` is node `i`, so with a boundary node below the highest
    /// detector, the entries past `num_detectors` address the top detectors.
    /// Returns [`PyMatchingError::InvalidGraph`] if upstream rejects the graph when
    /// the matcher is first built, for example an edge weight above its maximum,
    /// or if the result, one entry per graph observable past 64 observables,
    /// cannot be allocated. Also returns an error if decoding fails.
    ///
    /// # Panics
    ///
    /// This function will not panic. The internal `expect()` is safe because
    /// the value `(obs_mask >> i) & 1` is always 0 or 1, which fits in a `u8`.
    #[must_use = "The decoding result should be used"]
    pub fn decode(&mut self, detection_events: &[u8]) -> Result<DecodingResult> {
        self.validate_syndrome(detection_events)?;

        // Use optimized method for ≤64 observables, extended method otherwise
        if self.config.num_observables <= OPTIMIZED_OBSERVABLE_LIMIT {
            let result = ffi::decode_detection_events_64(self.graph.pin_mut(), detection_events)?;

            // The first 8 bytes of observables contain the packed obs_mask
            let num_obs = self.num_decoded_observables();
            let mut observables = vec![0u8; num_obs];

            // Unpack the obs_mask from the result
            if !result.observables.is_empty() {
                let mut obs_mask = 0u64;
                for (i, &observable) in result.observables.iter().enumerate().take(8) {
                    obs_mask |= u64::from(observable) << (i * BITS_PER_BYTE);
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
            let mut observable = zeroed_rows(self.num_observables(), 1, "observable", "num_shots")?;
            let weight = ffi::decode_detection_events_extended(
                self.graph.pin_mut(),
                detection_events,
                &mut observable,
            )?;

            Ok(DecodingResult { observable, weight })
        }
    }

    /// Decode to matched detection event pairs
    /// Returns pairs of detection events that are matched together
    /// A value of -1 in detector2 indicates matching to boundary
    ///
    /// # Errors
    /// Returns an invalid-syndrome error unless `detection_events` has a length
    /// [`Self::decode`] accepts, or an error if matching fails.
    #[must_use = "The matched pairs should be used"]
    pub fn decode_to_matched_pairs(&mut self, detection_events: &[u8]) -> Result<Vec<MatchedPair>> {
        self.validate_syndrome(detection_events)?;
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
    /// Returns an invalid-syndrome error unless `detection_events` has a length
    /// [`Self::decode`] accepts, or an error if edge extraction fails.
    pub fn decode_to_edges(&mut self, detection_events: &[u8]) -> Result<Vec<MatchedPair>> {
        self.validate_syndrome(detection_events)?;
        let edges = ffi::decode_to_edges(self.graph.pin_mut(), detection_events)?;
        Ok(edges.into_iter().map(std::convert::Into::into).collect())
    }

    /// Batch decode multiple shots (new API)
    ///
    /// # Arguments
    /// * `shots` - Detection events for all shots (flat array)
    /// * `num_shots` - Number of shots to decode
    /// * `num_detectors` - Dense entries per shot, any width [`Self::decode`] accepts
    /// * `config` - Configuration for batch decoding
    ///
    /// # Errors
    /// Returns an error if shot data is invalid, parameters are inconsistent, or batch decoding fails.
    /// Returns [`PyMatchingError::InvalidGraph`] if one shot's predictions, sized
    /// by the graph's observable count, cannot be allocated, and
    /// [`PyMatchingError::Configuration`] if only `num_shots` makes the output too
    /// large.
    pub fn decode_batch_with_config(
        &mut self,
        shots: &[u8],
        num_shots: usize,
        num_detectors: usize,
        config: BatchConfig,
    ) -> Result<BatchDecodingResult> {
        // Each shot is a dense syndrome, with the widths `decode` accepts.
        self.validate_syndrome_width(num_detectors)?;
        let stride = if config.bit_packed_input {
            num_detectors.div_ceil(8)
        } else {
            num_detectors
        };
        let expected_size = num_shots.checked_mul(stride).ok_or_else(|| {
            PyMatchingError::Configuration("batch dimensions overflow usize".to_string())
        })?;
        if shots.len() != expected_size {
            return Err(PyMatchingError::Dem(
                pecos_decoder_core::DecoderError::BatchSizeMismatch {
                    expected: expected_size,
                    actual: shots.len(),
                },
            ));
        }
        if num_shots == 0 {
            return Ok(BatchDecodingResult {
                predictions: vec![],
                weights: vec![],
                bit_packed: config.bit_packed_output,
            });
        }

        // The backend writes one row per shot, padded to the graph's observable count.
        let packed_width = |observables: usize| {
            if config.bit_packed_output {
                observables.div_ceil(BITS_PER_BYTE)
            } else {
                observables
            }
        };
        let row_len = packed_width(self.num_observables());
        let mut predictions = zeroed_rows(row_len, num_shots, "observable", "num_shots")?;
        let mut weights = Vec::new();
        weights.try_reserve_exact(num_shots).map_err(|_| {
            PyMatchingError::Configuration(format!(
                "cannot allocate {num_shots} shot weights; pass a smaller num_shots"
            ))
        })?;
        weights.resize(num_shots, 0.0);

        ffi::decode_batch(
            self.graph.pin_mut(),
            shots,
            num_detectors,
            config.bit_packed_input,
            config.bit_packed_output,
            &mut predictions,
            &mut weights,
        )?;

        // Trim each row to the width `decode` returns.
        let width = packed_width(self.num_decoded_observables());
        let predictions = split_rows(&predictions, row_len, width, num_shots, "num_shots")?;

        Ok(BatchDecodingResult {
            predictions,
            weights: if config.return_weights {
                weights
            } else {
                Vec::new()
            },
            bit_packed: config.bit_packed_output,
        })
    }

    /// Find shortest path between two nodes
    ///
    /// Returns the nodes along the shortest path, in order from `source` to
    /// `target`; the path is never empty. Between two detectors it starts at
    /// `source` and ends at `target`. When one node is a boundary node,
    /// upstream `PyMatching` searches from the other node to the nearest
    /// boundary (any boundary node or boundary edge, not necessarily the given
    /// one) and the boundary node is left out, so the path ends (or, for a
    /// boundary `source`, starts) at the detector next to that boundary. For a
    /// detector `source == target` on a graph with edges it is `[source]`,
    /// returned without building the matcher.
    ///
    /// # Errors
    ///
    /// Checked in this order, each before upstream is called:
    /// [`PyMatchingError::Configuration`] if a node is out of bounds, if the
    /// graph has no edges (also for `source == target`), if both nodes are
    /// boundary nodes, or if no path exists: `target` is not reachable from
    /// `source` without passing through a boundary node, or, when one node is a
    /// boundary node, no boundary is reachable from the other.
    /// [`PyMatchingError::InvalidGraph`] if the reachability search, sized by
    /// the graph, cannot be allocated. Upstream would dereference a null pointer
    /// for an unreachable target and for `source == target`, which is why these
    /// are checked first. Then [`PyMatchingError::InvalidGraph`] if upstream
    /// rejects the graph when it builds the matcher, for example an edge weight
    /// above its maximum, and [`PyMatchingError::Ffi`] if the upstream search
    /// fails.
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

        let source_is_boundary = self.is_boundary_node(source);
        let target_is_boundary = self.is_boundary_node(target);
        if source_is_boundary && target_is_boundary {
            return Err(PyMatchingError::Configuration(format!(
                "Source node {source} and target node {target} are both boundary nodes; at \
                 least one must be a detector"
            )));
        }

        if source == target {
            return Ok(vec![source]);
        }

        let (start, end) = if source_is_boundary {
            (target, source)
        } else {
            (source, target)
        };
        if !self.path_exists(start, end)? {
            return Err(PyMatchingError::Configuration(
                if self.is_boundary_node(end) {
                    format!(
                        "No path exists between nodes {source} and {target}: no boundary is \
                     reachable from node {start}"
                    )
                } else {
                    format!(
                        "No path exists between nodes {source} and {target}. They are in different connected components."
                    )
                },
            ));
        }

        let path = ffi::get_shortest_path(self.graph.pin_mut(), source, target)?;
        Ok(path.into_iter().collect())
    }

    /// Whether upstream's search from detector `start` reaches `end`.
    ///
    /// Upstream searches a graph without boundary nodes: an edge to a boundary
    /// node is a boundary edge, and the search never passes through one. A
    /// detector `end` must be reachable over edges between detectors; a boundary
    /// `end` is reached through any boundary edge.
    ///
    /// # Errors
    ///
    /// Returns [`PyMatchingError::InvalidGraph`] if the search state, sized by
    /// the graph's node and edge counts, cannot be allocated.
    fn path_exists(&self, start: usize, end: usize) -> Result<bool> {
        let num_nodes = self.num_nodes();
        let too_large = || {
            PyMatchingError::InvalidGraph(format!(
                "cannot allocate the shortest-path search over the graph's {num_nodes} nodes; \
                 build the graph with fewer nodes"
            ))
        };
        let falses = || -> Result<Vec<bool>> {
            let mut flags = Vec::new();
            flags
                .try_reserve_exact(num_nodes)
                .map_err(|_| too_large())?;
            flags.resize(num_nodes, false);
            Ok(flags)
        };
        let end_is_boundary = self.is_boundary_node(end);
        let mut adjacent: Vec<Vec<usize>> = Vec::new();
        adjacent
            .try_reserve_exact(num_nodes)
            .map_err(|_| too_large())?;
        adjacent.resize_with(num_nodes, Vec::new);
        let mut touches_boundary = falses()?;
        // The bridge's copy of the edges is at most the size of the graph's own.
        for edge in ffi::pymatching_get_all_edges(&self.graph) {
            // A boundary edge's `node2` is `usize::MAX`, which is a boundary node.
            let node2_is_detector = !self.is_boundary_node(edge.node2);
            match (self.is_boundary_node(edge.node1), node2_is_detector) {
                (false, true) => {
                    for (from, to) in [(edge.node1, edge.node2), (edge.node2, edge.node1)] {
                        adjacent[from].try_reserve(1).map_err(|_| too_large())?;
                        adjacent[from].push(to);
                    }
                }
                (false, false) => touches_boundary[edge.node1] = true,
                (true, true) => touches_boundary[edge.node2] = true,
                (true, false) => {}
            }
        }

        let mut visited = falses()?;
        // Each node is queued at most once, so the queue never grows past this.
        let mut queue = std::collections::VecDeque::new();
        queue
            .try_reserve_exact(num_nodes)
            .map_err(|_| too_large())?;
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            if node == end || (end_is_boundary && touches_boundary[node]) {
                return Ok(true);
            }
            for &neighbor in &adjacent[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(false)
    }

    /// Check if two nodes are in the same connected component, counting
    /// boundary nodes as ordinary nodes and ignoring boundary edges
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
    ///
    /// Samples each edge with its error probability and returns, per sample, the
    /// flipped observables ([`Self::num_observables`] entries) and the syndrome.
    /// The syndrome has one entry per node, [`Self::num_nodes`] entries, as
    /// upstream `PyMatching`'s `add_noise` returns; a boundary node's entry is
    /// always 0. With no boundary that is [`Self::num_detectors`] entries.
    /// [`Self::decode`] accepts a syndrome of that length. Unless every edge has
    /// an error probability, every sample is all zeros.
    ///
    /// # Errors
    ///
    /// Returns [`PyMatchingError::InvalidGraph`] if one sample's output, sized by
    /// the graph's observable or node count, cannot be allocated,
    /// [`PyMatchingError::Configuration`] if only `num_samples` makes the output
    /// too large, or [`PyMatchingError::Ffi`] if sampling fails.
    pub fn add_noise(&self, num_samples: usize, rng_seed: u64) -> Result<NoiseResult> {
        let num_observables = self.num_observables();
        let num_nodes = self.num_nodes();
        let mut errors = zeroed_rows(num_observables, num_samples, "observable", "num_samples")?;
        let mut syndromes = zeroed_rows(num_nodes, num_samples, "node", "num_samples")?;
        ffi::add_noise(
            &self.graph,
            num_samples,
            rng_seed,
            &mut errors,
            &mut syndromes,
        )?;

        Ok(NoiseResult {
            errors: split_rows(
                &errors,
                num_observables,
                num_observables,
                num_samples,
                "num_samples",
            )?,
            syndromes: split_rows(&syndromes, num_nodes, num_nodes, num_samples, "num_samples")?,
        })
    }

    /// Get edge weight normalising constant
    ///
    /// # Errors
    ///
    /// Returns [`PyMatchingError::InvalidGraph`] if `PyMatching` rejects the graph,
    /// for example an edge weight above its maximum absolute edge weight.
    pub fn get_edge_weight_normalising_constant(&self, num_distinct_weights: usize) -> Result<f64> {
        Ok(ffi::get_edge_weight_normalising_constant(
            &self.graph,
            num_distinct_weights,
        )?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_edge_observable_past_64_is_decoded() {
        let config = PyMatchingConfig {
            num_nodes: Some(2),
            num_observables: 1,
            ..Default::default()
        };
        let mut decoder = PyMatchingDecoder::new(config).unwrap();
        decoder
            .add_edge(0, 1, &[3], Some(2.0), Some(0.1), None)
            .unwrap();
        // A lighter parallel edge replaces the observables of the existing edge, so
        // the backend graph does not grow on its own.
        decoder
            .add_edge(0, 1, &[70], Some(1.0), Some(0.1), None)
            .unwrap();
        assert_eq!(decoder.num_decoded_observables(), 71);
        let observable = decoder.decode(&[1, 1]).unwrap().observable;
        assert_eq!(observable.len(), 71);
        assert_eq!(observable[70], 1);
        assert_eq!(observable.iter().filter(|&&bit| bit != 0).count(), 1);
    }

    #[test]
    fn added_edge_observables_are_decoded_in_single_and_batch_paths() {
        let config = PyMatchingConfig {
            num_nodes: Some(2),
            num_observables: 1,
            ..Default::default()
        };
        let mut decoder = PyMatchingDecoder::new(config).unwrap();
        // Observable 5 lies past the configured count of 1.
        decoder
            .add_edge(0, 1, &[5], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_boundary_edge(0, &[], Some(3.0), Some(0.01), None)
            .unwrap();
        assert_eq!(decoder.num_decoded_observables(), 6);

        let single = decoder.decode(&[1, 1]).unwrap().observable;
        assert_eq!(single, [0, 0, 0, 0, 0, 1]);

        let unpacked = decoder
            .decode_batch_with_config(&[1, 1], 1, 2, BatchConfig::default())
            .unwrap();
        assert_eq!(unpacked.predictions, std::slice::from_ref(&single));

        let packed_config = BatchConfig {
            bit_packed_output: true,
            ..BatchConfig::default()
        };
        let packed = decoder
            .decode_batch_with_config(&[1, 1], 1, 2, packed_config)
            .unwrap();
        // Six observables pack into one byte, with observable 5 in bit 5.
        assert_eq!(packed.predictions, [vec![1_u8 << 5]]);
    }

    #[test]
    fn batch_validates_empty_buffers_and_size_overflow() {
        let mut decoder = PyMatchingDecoder::from_dem("error(0.1) D0 L0\ndetector D1").unwrap();
        for bit_packed_input in [false, true] {
            let config = BatchConfig {
                bit_packed_input,
                ..BatchConfig::default()
            };
            assert!(matches!(
                decoder.decode_batch_with_config(&[0], 0, 2, config),
                Err(PyMatchingError::Dem(
                    pecos_decoder_core::DecoderError::BatchSizeMismatch {
                        expected: 0,
                        actual: 1
                    }
                ))
            ));
            assert!(decoder.decode_batch_with_config(&[], 0, 2, config).is_ok());
        }
        assert!(matches!(
            decoder.decode_batch_with_config(&[], usize::MAX, 2, BatchConfig::default()),
            Err(PyMatchingError::Configuration(_))
        ));
    }

    #[test]
    fn dem_validation_precedes_backend_parsing() {
        for text in [
            "error(0.1) D0 D0",
            "error(0.1) D0 D1 ^ D1 D0",
            "repeat 2 {\nerror(0.1) D0 D0\n}",
            "@bad",
        ] {
            let expected = pecos_decoder_core::dem::grammar::validate_dem_text(text).unwrap_err();
            let error = PyMatchingDecoder::from_dem(text).err().unwrap();
            assert!(matches!(
                error,
                PyMatchingError::Dem(pecos_decoder_core::DecoderError::InvalidDemSyntax(_))
            ));
            assert_eq!(error.to_string(), expected.to_string());
            let correlated = PyMatchingDecoder::from_dem_with_correlations(text, true)
                .err()
                .unwrap();
            assert_eq!(correlated.to_string(), expected.to_string());
        }
    }

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
        decoder.set_boundary(&[4]).unwrap();

        // Test unpacked format with valid syndromes: nodes 0..=3 are detectors
        // and node 4 is the boundary.
        let shots = vec![0; 8]; // 2 shots of 4 detectors each (all zero)
        let config = BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: true,
        };
        let result = decoder
            .decode_batch_with_config(
                &shots, 2, // num_shots
                4, // num_detectors
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
                4, // num_detectors
                packed_config,
            )
            .unwrap();

        assert_eq!(result_packed.predictions.len(), 2);
        assert_eq!(result_packed.weights.len(), 2);
        assert!(result_packed.bit_packed);
    }

    #[test]
    fn test_batch_decode_without_observables() {
        for correlated in [false, true] {
            let mut decoder =
                PyMatchingDecoder::from_dem_with_correlations("error(0.1) D0 D1", correlated)
                    .unwrap();
            assert_eq!(decoder.num_observables(), 0);
            for bit_packed_input in [false, true] {
                for bit_packed_output in [false, true] {
                    for return_weights in [false, true] {
                        for num_shots in [0, 2] {
                            let shots: &[u8] = if num_shots == 0 {
                                &[]
                            } else if bit_packed_input {
                                &[0, 3]
                            } else {
                                &[0, 0, 1, 1]
                            };
                            let result = decoder
                                .decode_batch_with_config(
                                    shots,
                                    num_shots,
                                    2,
                                    BatchConfig {
                                        bit_packed_input,
                                        bit_packed_output,
                                        return_weights,
                                    },
                                )
                                .unwrap();
                            assert_eq!(result.predictions, vec![Vec::<u8>::new(); num_shots]);
                            assert_eq!(result.bit_packed, bit_packed_output);
                            assert_eq!(
                                result.weights.len(),
                                if return_weights { num_shots } else { 0 }
                            );
                        }
                    }
                }
            }
        }
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
    fn test_dem_error_probability_override_reaches_every_edge() {
        let dem =
            "error(0.1) D0\nerror(0.2) D1 L0\ndetector D0\ndetector D1\nlogical_observable L0";
        let baseline = PyMatchingDecoder::from_dem(dem).unwrap().get_all_edges();
        let overridden = PyMatchingDecoder::from_dem_with_error_probability(dem, 0.35)
            .unwrap()
            .get_all_edges();

        assert_eq!(baseline.len(), overridden.len());
        let expected_weight = ((1.0_f64 - 0.35) / 0.35).ln();
        for after in &overridden {
            assert!((after.weight - expected_weight).abs() < f64::EPSILON);
            assert!((after.error_probability - 0.35).abs() < f64::EPSILON);
        }

        let error = PyMatchingDecoder::from_dem_with_error_probability(dem, f64::NAN)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("error_probability"));

        for endpoint in [0.0, 1.0] {
            let error = PyMatchingDecoder::from_dem_with_error_probability(dem, endpoint)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("error_probability"));
        }
    }

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
