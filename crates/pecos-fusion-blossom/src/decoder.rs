//! Fusion Blossom decoder implementation

use super::errors::{FusionBlossomError, Result};
use fusion_blossom::{
    example_codes::{
        CircuitLevelPlanarCode, CodeCapacityPlanarCode, CodeCapacityRotatedCode, ExampleCode,
        PhenomenologicalPlanarCode, PhenomenologicalRotatedCode,
    },
    mwpm_solver::{LegacySolverSerial, PrimalDualSolver, SolverSerial},
    util::{EdgeIndex, SolverInitializer, SyndromePattern, VertexIndex, Weight},
};
use ndarray::{Array2, ArrayView1};
use std::collections::HashMap;
use std::fmt;

/// Solver type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    /// Legacy solver (original implementation)
    Legacy,
    /// Serial solver (improved performance)
    #[default]
    Serial,
}

/// Configuration for Fusion Blossom decoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionBlossomConfig {
    /// Number of nodes in the graph
    pub num_nodes: Option<usize>,
    /// Number of observables
    pub num_observables: usize,
    /// Solver type to use
    pub solver_type: SolverType,
    /// Maximum tree size for union-find decoder (currently not supported in Rust API)
    pub max_tree_size: Option<usize>,
}

impl Default for FusionBlossomConfig {
    fn default() -> Self {
        Self {
            num_nodes: None,
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        }
    }
}

/// Options for decoding
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodingOptions {
    /// Whether to include perfect matching details in the result
    pub include_perfect_matching: bool,
}

/// Syndrome data with optional erasures and dynamic weights
#[derive(Debug, Clone, Default)]
pub struct SyndromeData {
    /// Defect vertices (syndrome)
    pub defects: Vec<usize>,
    /// Erasure edges (known errors)
    pub erasures: Option<Vec<usize>>,
    /// Dynamic weight adjustments: (`edge_index`, `new_weight`)
    pub dynamic_weights: Option<Vec<(usize, i32)>>,
}

impl SyndromeData {
    /// Create syndrome data from just defects
    #[must_use]
    pub fn from_defects(defects: Vec<usize>) -> Self {
        Self {
            defects,
            erasures: None,
            dynamic_weights: None,
        }
    }

    /// Create syndrome data with erasures
    #[must_use]
    pub fn with_erasures(defects: Vec<usize>, erasures: Vec<usize>) -> Self {
        Self {
            defects,
            erasures: Some(erasures),
            dynamic_weights: None,
        }
    }
}

/// Perfect matching information
#[derive(Debug, Clone, PartialEq)]
pub struct PerfectMatchingInfo {
    /// Matched vertex pairs: (vertex1, vertex2, `is_virtual`)
    pub matched_pairs: Vec<(VertexIndex, VertexIndex, bool)>,
    /// Total number of matches
    pub match_count: usize,
}

/// Decoding result from Fusion Blossom
#[derive(Debug, Clone, PartialEq)]
pub struct DecodingResult {
    /// The decoded observable errors
    pub observable: Vec<u8>,
    /// Total weight of the matching
    pub weight: f64,
    /// The matched edge indices
    pub matched_edges: Vec<EdgeIndex>,
    /// Perfect matching details (if requested)
    pub perfect_matching: Option<PerfectMatchingInfo>,
}

impl fmt::Display for DecodingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DecodingResult {{ observables: {:?}, weight: {:.6}, edges: {} }}",
            self.observable,
            self.weight,
            self.matched_edges.len()
        )
    }
}

/// Standard QEC code types
#[derive(Debug, Clone, Copy)]
pub enum StandardCode {
    /// Code capacity planar code
    CodeCapacityPlanar {
        /// Distance of the code
        d: usize,
        /// Physical error rate
        p: f64,
        /// Maximum half weight for edges
        max_half_weight: i32,
    },
    /// Phenomenological planar code
    PhenomenologicalPlanar {
        /// Distance of the code
        d: usize,
        /// Physical error rate
        p: f64,
        /// Measurement error rate
        p_measurement: f64,
        /// Maximum half weight for edges
        max_half_weight: i32,
    },
    /// Circuit-level planar code
    CircuitLevelPlanar {
        /// Distance of the code
        d: usize,
        /// Physical error rate
        p: f64,
        /// Maximum half weight for edges
        max_half_weight: i32,
    },
    /// Code capacity rotated code
    CodeCapacityRotated {
        /// Distance of the code
        d: usize,
        /// Physical error rate
        p: f64,
        /// Maximum half weight for edges
        max_half_weight: i32,
    },
    /// Phenomenological rotated code
    PhenomenologicalRotated {
        /// Distance of the code
        d: usize,
        /// Physical error rate
        p: f64,
        /// Measurement error rate
        p_measurement: f64,
        /// Maximum half weight for edges
        max_half_weight: i32,
    },
}

/// Internal solver enum to hold different solver types
enum Solver {
    Legacy(LegacySolverSerial),
    Serial(SolverSerial),
}

/// Fusion Blossom decoder
pub struct FusionBlossomDecoder {
    config: FusionBlossomConfig,
    /// Map from edge index to observable mask
    edge_observables: HashMap<EdgeIndex, Vec<usize>>,
    /// Number of nodes (detectors)
    num_nodes: usize,
    /// Virtual boundary node (if used)
    boundary_node: Option<VertexIndex>,
    /// Edges to be added to the initializer
    weighted_edges: Vec<(VertexIndex, VertexIndex, Weight)>,
    /// Virtual vertices
    virtual_vertices: Vec<VertexIndex>,
    /// Cached solver instance for reuse
    solver: Option<Solver>,
    /// Cached initializer
    initializer: Option<SolverInitializer>,
}

impl FusionBlossomDecoder {
    /// Create a builder for configuring a new Fusion Blossom decoder
    ///
    /// This is the recommended way to construct a decoder:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use pecos_fusion_blossom::FusionBlossomDecoder;
    ///
    /// let decoder = FusionBlossomDecoder::builder()
    ///     .num_nodes(4)
    ///     .num_observables(2)
    ///     .add_edge(0, 1, vec![0], Some(1.0))
    ///     .add_edge(1, 2, vec![1], Some(1.0))
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> crate::builder::FusionBlossomBuilder {
        crate::builder::FusionBlossomBuilder::new()
    }

    /// Create a new decoder with the given configuration
    ///
    /// # Errors
    ///
    /// Returns [`FusionBlossomError::Configuration`] if `num_nodes` is not specified in the config.
    pub fn new(config: FusionBlossomConfig) -> Result<Self> {
        let num_nodes = config.num_nodes.ok_or_else(|| {
            FusionBlossomError::Configuration("num_nodes must be specified".to_string())
        })?;

        Ok(Self {
            config,
            edge_observables: HashMap::new(),
            num_nodes,
            boundary_node: None,
            weighted_edges: Vec::new(),
            virtual_vertices: Vec::new(),
            solver: None,
            initializer: None,
        })
    }

    /// Create decoder from a standard QEC code
    ///
    /// # Errors
    ///
    /// This function currently does not return errors, but returns `Result` for API
    /// consistency and future extensibility.
    pub fn from_standard_code(code: StandardCode, config: FusionBlossomConfig) -> Result<Self> {
        let example_code: Box<dyn ExampleCode> = match code {
            StandardCode::CodeCapacityPlanar {
                d,
                p,
                max_half_weight,
            } => Box::new(CodeCapacityPlanarCode::new(
                d as VertexIndex,
                p,
                max_half_weight as Weight,
            )),
            StandardCode::PhenomenologicalPlanar {
                d,
                p,
                p_measurement: _,
                max_half_weight,
            } => {
                // Note: PhenomenologicalPlanarCode takes noisy_measurements count, not probability
                // Using d-1 as a reasonable default for number of measurement rounds
                Box::new(PhenomenologicalPlanarCode::new(
                    d as VertexIndex,
                    (d - 1) as VertexIndex,
                    p,
                    max_half_weight as Weight,
                ))
            }
            StandardCode::CircuitLevelPlanar {
                d,
                p,
                max_half_weight,
            } => {
                // CircuitLevelPlanarCode also needs noisy_measurements count
                Box::new(CircuitLevelPlanarCode::new(
                    d as VertexIndex,
                    (d - 1) as VertexIndex,
                    p,
                    max_half_weight as Weight,
                ))
            }
            StandardCode::CodeCapacityRotated {
                d,
                p,
                max_half_weight,
            } => Box::new(CodeCapacityRotatedCode::new(
                d as VertexIndex,
                p,
                max_half_weight as Weight,
            )),
            StandardCode::PhenomenologicalRotated {
                d,
                p,
                p_measurement: _,
                max_half_weight,
            } => {
                // Using d-1 measurement rounds
                Box::new(PhenomenologicalRotatedCode::new(
                    d as VertexIndex,
                    (d - 1) as VertexIndex,
                    p,
                    max_half_weight as Weight,
                ))
            }
        };

        let initializer = example_code.get_initializer();
        let num_nodes = initializer.vertex_num;

        // Extract edge observables from the code
        let mut edge_observables = HashMap::new();
        // Note: Fusion Blossom's example codes don't directly expose observables,
        // so we'll use a simple mapping based on edge index
        for (i, _) in initializer.weighted_edges.iter().enumerate() {
            edge_observables.insert(i as EdgeIndex, vec![i % config.num_observables]);
        }

        let mut decoder = Self {
            config: FusionBlossomConfig {
                num_nodes: Some(num_nodes),
                ..config
            },
            edge_observables,
            num_nodes,
            boundary_node: None,
            weighted_edges: initializer.weighted_edges.clone(),
            virtual_vertices: initializer.virtual_vertices.clone(),
            solver: None,
            initializer: Some(initializer),
        };

        // Identify boundary nodes from virtual vertices
        if !decoder.virtual_vertices.is_empty() {
            decoder.boundary_node = Some(decoder.virtual_vertices[0]);
        }

        Ok(decoder)
    }

    /// Add an edge to the graph
    ///
    /// # Errors
    ///
    /// Returns [`FusionBlossomError::InvalidGraph`] if:
    /// - Either node index is out of bounds
    /// - The weight is negative
    pub fn add_edge(
        &mut self,
        node1: usize,
        node2: usize,
        observables: &[usize],
        weight: Option<f64>,
    ) -> Result<()> {
        if node1 >= self.num_nodes || node2 >= self.num_nodes {
            return Err(FusionBlossomError::InvalidGraph(format!(
                "Node indices {} or {} out of bounds (max {})",
                node1,
                node2,
                self.num_nodes - 1
            )));
        }

        let weight_int = if let Some(w) = weight {
            if w < 0.0 {
                return Err(FusionBlossomError::InvalidGraph(
                    "Edge weights must be non-negative".to_string(),
                ));
            }
            // Fusion Blossom requires even weights
            ((w * 1000.0) as Weight / 2) * 2
        } else {
            1000 // Default weight of 1.0
        };

        let edge_idx = self.weighted_edges.len() as EdgeIndex;
        self.weighted_edges
            .push((node1 as VertexIndex, node2 as VertexIndex, weight_int));

        if !observables.is_empty() {
            self.edge_observables.insert(edge_idx, observables.to_vec());
        }

        Ok(())
    }

    /// Add a boundary edge (connects a node to the boundary)
    ///
    /// # Errors
    ///
    /// Returns [`FusionBlossomError::InvalidGraph`] if:
    /// - The node index is out of bounds
    /// - The weight is negative
    ///
    /// # Panics
    ///
    /// This function will not panic. The internal `unwrap()` is safe because
    /// `boundary_node` is always set before use (either already `Some` or set
    /// in the same code path).
    pub fn add_boundary_edge(
        &mut self,
        node: usize,
        observables: &[usize],
        weight: Option<f64>,
    ) -> Result<()> {
        if node >= self.num_nodes {
            return Err(FusionBlossomError::InvalidGraph(format!(
                "Node index {} out of bounds (max {})",
                node,
                self.num_nodes - 1
            )));
        }

        // Create a virtual boundary node if not already created
        if self.boundary_node.is_none() {
            self.boundary_node = Some(self.num_nodes as VertexIndex);
            self.virtual_vertices.push(self.num_nodes as VertexIndex);
        }

        let boundary_node = self.boundary_node.expect("boundary_node is set above");

        let weight_int = if let Some(w) = weight {
            if w < 0.0 {
                return Err(FusionBlossomError::InvalidGraph(
                    "Edge weights must be non-negative".to_string(),
                ));
            }
            // Fusion Blossom requires even weights
            ((w * 1000.0) as Weight / 2) * 2
        } else {
            1000
        };

        let edge_idx = self.weighted_edges.len() as EdgeIndex;
        self.weighted_edges
            .push((node as VertexIndex, boundary_node, weight_int));

        if !observables.is_empty() {
            self.edge_observables.insert(edge_idx, observables.to_vec());
        }

        Ok(())
    }

    /// Create decoder from a check matrix
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`FusionBlossomError::Configuration`] if `num_nodes` cannot be set
    /// - [`FusionBlossomError::InvalidCheckMatrix`] if a column has more than 2 non-zero entries
    /// - [`FusionBlossomError::InvalidGraph`] if edge addition fails
    pub fn from_check_matrix(
        check_matrix: &Array2<u8>,
        weights: Option<&[f64]>,
        config: FusionBlossomConfig,
    ) -> Result<Self> {
        let num_rows = check_matrix.nrows();
        let num_cols = check_matrix.ncols();

        let mut decoder = Self::new(FusionBlossomConfig {
            num_nodes: Some(num_rows),
            ..config
        })?;

        // Process each column (error)
        for col in 0..num_cols {
            let mut non_zero_rows = Vec::new();
            for row in 0..num_rows {
                if check_matrix[[row, col]] != 0 {
                    non_zero_rows.push(row);
                }
            }

            let weight = weights.map(|w| w[col]);

            match non_zero_rows.len() {
                0 => {
                    // No edge for this error
                }
                1 => {
                    // Boundary edge
                    decoder.add_boundary_edge(non_zero_rows[0], &[col], weight)?;
                }
                2 => {
                    // Regular edge between two nodes
                    decoder.add_edge(non_zero_rows[0], non_zero_rows[1], &[col], weight)?;
                }
                _ => {
                    return Err(FusionBlossomError::InvalidCheckMatrix(format!(
                        "Column {} has {} non-zero entries, expected 1 or 2",
                        col,
                        non_zero_rows.len()
                    )));
                }
            }
        }

        Ok(decoder)
    }

    /// Clear the solver state for reuse
    pub fn clear(&mut self) {
        // For Fusion Blossom, we need to recreate the solver to clear state
        self.solver = None;
    }

    /// Get or create the initializer
    fn get_or_create_initializer(&mut self) -> SolverInitializer {
        if let Some(ref initializer) = self.initializer {
            initializer.clone()
        } else {
            let vertex_num = if self.boundary_node.is_some() {
                (self.num_nodes + 1) as VertexIndex
            } else {
                self.num_nodes as VertexIndex
            };

            let initializer = SolverInitializer::new(
                vertex_num,
                self.weighted_edges.clone(),
                self.virtual_vertices.clone(),
            );

            self.initializer = Some(initializer.clone());
            initializer
        }
    }

    /// Get or create the solver
    fn get_or_create_solver(&mut self) -> &mut Solver {
        if self.solver.is_none() {
            let initializer = self.get_or_create_initializer();

            let solver = match self.config.solver_type {
                SolverType::Legacy => Solver::Legacy(LegacySolverSerial::new(&initializer)),
                SolverType::Serial => Solver::Serial(SolverSerial::new(&initializer)),
            };

            self.solver = Some(solver);
        }

        self.solver.as_mut().expect("solver is initialized above")
    }

    /// Decode a syndrome with advanced options and decoding options
    ///
    /// # Errors
    ///
    /// This function currently does not return errors, but returns `Result` for API
    /// consistency and future extensibility.
    pub fn decode_with_options(
        &mut self,
        syndrome_data: SyndromeData,
        options: DecodingOptions,
    ) -> Result<DecodingResult> {
        // Convert defects to VertexIndex
        let defect_vertices: Vec<VertexIndex> = syndrome_data
            .defects
            .iter()
            .map(|&v| v as VertexIndex)
            .collect();

        if defect_vertices.is_empty() {
            // No defects, return empty result
            return Ok(DecodingResult {
                observable: vec![0; self.config.num_observables],
                weight: 0.0,
                matched_edges: Vec::new(),
                perfect_matching: None,
            });
        }

        // Create syndrome pattern with optional erasures and dynamic weights
        let syndrome_pattern =
            if syndrome_data.erasures.is_some() || syndrome_data.dynamic_weights.is_some() {
                let erasures = syndrome_data
                    .erasures
                    .unwrap_or_default()
                    .iter()
                    .map(|&idx| idx as EdgeIndex)
                    .collect();

                let dynamic_weights = syndrome_data
                    .dynamic_weights
                    .unwrap_or_default()
                    .iter()
                    .map(|&(idx, w)| (idx as EdgeIndex, w as Weight))
                    .collect();

                SyndromePattern::new_dynamic_weights(defect_vertices, erasures, dynamic_weights)
            } else {
                SyndromePattern::new_vertices(defect_vertices)
            };

        // Get or create solver
        let solver = self.get_or_create_solver();

        // Solve and get perfect matching if requested
        let (matched_edges, perfect_matching_info) = match solver {
            Solver::Legacy(s) => {
                let edges = s.solve_subgraph(&syndrome_pattern);
                let pm_info = if options.include_perfect_matching {
                    // Legacy solver doesn't have easy access to perfect matching
                    None
                } else {
                    None
                };
                (edges, pm_info)
            }
            Solver::Serial(s) => {
                s.solve(&syndrome_pattern);
                let edges = s.subgraph();

                let pm_info = if options.include_perfect_matching {
                    // For Serial solver, we can't easily get perfect matching details
                    // without accessing internal structures
                    None
                } else {
                    None
                };

                (edges, pm_info)
            }
        };

        // Calculate observables
        let mut observable = vec![0u8; self.config.num_observables];
        let mut total_weight = 0.0;

        for &edge_idx in &matched_edges {
            if let Some(obs_indices) = self.edge_observables.get(&edge_idx) {
                for &obs_idx in obs_indices {
                    if obs_idx < self.config.num_observables {
                        observable[obs_idx] ^= 1;
                    }
                }
            }

            // Get edge weight
            if let Some((_, _, weight)) = self.weighted_edges.get(edge_idx) {
                total_weight += (*weight as f64) / 1000.0; // Convert back from milliunits
            }
        }

        Ok(DecodingResult {
            observable,
            weight: total_weight,
            matched_edges,
            perfect_matching: perfect_matching_info,
        })
    }

    /// Decode a syndrome with advanced options (backwards compatibility)
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::decode_with_options`].
    pub fn decode_advanced(&mut self, syndrome_data: SyndromeData) -> Result<DecodingResult> {
        self.decode_with_options(syndrome_data, DecodingOptions::default())
    }

    /// Decode a syndrome (simple interface)
    ///
    /// # Errors
    ///
    /// Returns [`FusionBlossomError::InvalidSyndrome`] if the syndrome length doesn't
    /// match the number of nodes in the decoder.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult> {
        if syndrome.len() != self.num_nodes {
            return Err(FusionBlossomError::InvalidSyndrome(format!(
                "Syndrome length {} doesn't match number of nodes {}",
                syndrome.len(),
                self.num_nodes
            )));
        }

        // Find defect vertices
        let mut defects = Vec::new();
        for (i, &val) in syndrome.iter().enumerate() {
            if val != 0 {
                defects.push(i);
            }
        }

        self.decode_advanced(SyndromeData::from_defects(defects))
    }

    /// Get a summary of the graph structure
    #[must_use]
    pub fn graph_summary(&self) -> String {
        format!(
            "FusionBlossomDecoder: {} nodes, {} edges, {} observables",
            self.num_nodes,
            self.weighted_edges.len(),
            self.config.num_observables
        )
    }

    /// Clear solver cache for weight reset
    pub fn clear_solver_cache(&mut self) {
        self.solver = None;
        self.initializer = None;
    }

    /// Get number of nodes
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get number of edges
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.weighted_edges.len()
    }
}
