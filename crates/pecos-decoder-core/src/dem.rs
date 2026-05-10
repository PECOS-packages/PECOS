//! Common traits and types for Detector Error Model (DEM) based decoders
//!
//! This module provides standardized interfaces for decoders that work
//! with Stim's detector error model format.

use crate::errors::DecoderError;

/// Trait for decoders that can be constructed from detector error models
pub trait DemDecoder: super::Decoder {
    /// Configuration type for DEM construction
    type DemConfig: Default;

    /// Create decoder from a DEM string
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM string is malformed or invalid
    /// - The detector/observable indices are out of bounds
    /// - The decoder cannot be constructed from the given DEM
    fn from_dem(dem: &str) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        Self::from_dem_with_config(dem, Default::default())
    }

    /// Create decoder from a DEM string with configuration
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM string is malformed or invalid
    /// - The configuration is invalid
    /// - The decoder cannot be constructed with the given parameters
    fn from_dem_with_config(dem: &str, config: Self::DemConfig) -> Result<Self, DecoderError>
    where
        Self: Sized;

    /// Create decoder from a DEM file
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The file cannot be read (I/O error)
    /// - The file contents are not valid DEM format
    /// - The decoder cannot be constructed from the DEM
    fn from_dem_file(path: &str) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        let dem = std::fs::read_to_string(path).map_err(DecoderError::IoError)?;
        Self::from_dem(&dem)
    }

    /// Create decoder from a DEM file with configuration
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The file cannot be read (I/O error)
    /// - The file contents are not valid DEM format
    /// - The configuration is invalid
    /// - The decoder cannot be constructed with the given parameters
    fn from_dem_file_with_config(path: &str, config: Self::DemConfig) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        let dem = std::fs::read_to_string(path).map_err(DecoderError::IoError)?;
        Self::from_dem_with_config(&dem, config)
    }

    /// Get the number of detectors in the model
    fn detector_count(&self) -> usize;

    /// Get the number of observables in the model
    fn observable_count(&self) -> usize;
}

/// Common configuration for DEM-based decoders
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DemConfig {
    /// Random seed for deterministic behavior
    pub seed: Option<u64>,
    /// Whether to use a compressed representation
    pub compressed: bool,
    /// Custom detector coordinates (if any)
    pub detector_coordinates: Option<Vec<Vec<f64>>>,
    /// Maximum number of errors to consider per detector
    pub max_errors_per_detector: Option<usize>,
}

/// Utilities for working with detector error models
pub mod utils {
    use super::DecoderError;

    /// Parse basic DEM metadata without full parsing
    ///
    /// Returns (`detector_count`, `observable_count`)
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if the DEM format is invalid
    pub fn parse_dem_metadata(dem: &str) -> Result<(usize, usize), DecoderError> {
        let mut max_detector = None;
        let mut observables = std::collections::BTreeSet::new();

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // Handle commands with probability parameters like "error(0.01)"
            let command = if parts[0].starts_with("error(") {
                "error"
            } else {
                parts[0]
            };

            match command {
                "error" => {
                    // Parse error line for detector and observable indices
                    for part in &parts[1..] {
                        if let Some(d_str) = part.strip_prefix('D') {
                            if let Ok(d) = d_str.parse::<usize>() {
                                max_detector = Some(max_detector.map_or(d, |m: usize| m.max(d)));
                            }
                        } else if let Some(l_str) = part.strip_prefix('L')
                            && let Ok(l) = l_str.parse::<usize>()
                        {
                            observables.insert(l);
                        }
                    }
                }
                "detector" => {
                    // Parse detector declarations
                    for part in &parts[1..] {
                        if let Some(d_str) = part.strip_prefix('D')
                            && let Ok(d) = d_str.parse::<usize>()
                        {
                            max_detector = Some(max_detector.map_or(d, |m: usize| m.max(d)));
                        }
                    }
                }
                _ => {}
            }
        }

        let detector_count = max_detector.map_or(0, |m| m + 1);
        let observable_count = observables.len();

        Ok((detector_count, observable_count))
    }

    /// Validate DEM format
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM is empty
    /// - The DEM contains invalid commands or syntax
    /// - Detector/observable indices are invalid
    pub fn validate_dem(dem: &str) -> Result<(), DecoderError> {
        if dem.trim().is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "DEM cannot be empty".to_string(),
            ));
        }

        // Basic validation - check for valid DEM commands
        let valid_commands = ["error", "detector", "logical_observable", "repeat"];

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let first_word = line.split_whitespace().next().unwrap_or("");
            // Handle commands with probability parameters like "error(0.01)"
            let command = if first_word.starts_with("error(") {
                "error"
            } else {
                first_word
            };
            if !valid_commands.contains(&command) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "Invalid DEM command: {first_word}"
                )));
            }
        }

        Ok(())
    }
}

/// Check matrix representation extracted from a Detector Error Model.
///
/// Converts a DEM string into the matrices needed by check-matrix-based
/// decoders (BP+OSD, `UnionFind`, `RelayBP`, etc.):
///
/// - **`check_matrix`** `H[d][m]`: 1 if error mechanism `m` flips detector `d`
/// - **`observable_matrix`** `L[o][m]`: 1 if mechanism `m` flips observable `o`
/// - **`error_priors`** `p[m]`: probability of mechanism `m`
///
/// # Example
///
/// ```
/// use pecos_decoder_core::dem::DemCheckMatrix;
///
/// let dem = "error(0.01) D0 D1 L0\nerror(0.02) D1 D2";
/// let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();
/// assert_eq!(dcm.num_detectors, 3);
/// assert_eq!(dcm.num_observables, 1);
/// assert_eq!(dcm.num_mechanisms, 2);
/// assert_eq!(dcm.error_priors, vec![0.01, 0.02]);
/// ```
#[derive(Debug, Clone)]
pub struct DemCheckMatrix {
    /// Check matrix: rows = detectors, columns = error mechanisms.
    pub check_matrix: ndarray::Array2<u8>,
    /// Observable matrix: rows = observables, columns = error mechanisms.
    pub observable_matrix: ndarray::Array2<u8>,
    /// Error probability per mechanism.
    pub error_priors: Vec<f64>,
    /// Number of detectors (rows of `check_matrix`).
    pub num_detectors: usize,
    /// Number of observables (rows of `observable_matrix`).
    pub num_observables: usize,
    /// Number of error mechanisms (columns of both matrices).
    pub num_mechanisms: usize,
}

impl DemCheckMatrix {
    /// Parse a DEM string into check matrix form.
    ///
    /// Each `error(p) D_i D_j ... L_k ...` line becomes one column in the
    /// check matrix (for the D entries) and one column in the observable
    /// matrix (for the L entries). Decomposed mechanisms (`D0 ^ D1`) are
    /// combined by XOR.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if the DEM string is malformed.
    pub fn from_dem_str(dem: &str) -> Result<Self, DecoderError> {
        // First pass: collect mechanisms and find dimensions.
        let mut mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)> = Vec::new();
        let mut max_detector: Option<u32> = None;
        let mut max_observable: Option<u32> = None;

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if !line.starts_with("error(") {
                // Skip non-error lines (detector, logical_observable, etc.)
                continue;
            }

            // Parse "error(p) D0 D1 ... L0 ..." or "error(p) D0 ^ D1 ..."
            let close_paren = line.find(')').ok_or_else(|| {
                DecoderError::InvalidConfiguration(
                    "Missing closing parenthesis in error line".into(),
                )
            })?;
            let prob_str = &line[6..close_paren];
            let probability: f64 = prob_str.parse().map_err(|_| {
                DecoderError::InvalidConfiguration(format!("Invalid probability: {prob_str}"))
            })?;

            let tokens_str = &line[close_paren + 1..];

            // Handle decomposed mechanisms (with ^) by XOR-ing components.
            let mut det_set = std::collections::BTreeSet::new();
            let mut obs_set = std::collections::BTreeSet::new();

            for component in tokens_str.split('^') {
                for token in component.split_whitespace() {
                    if let Some(d_str) = token.strip_prefix('D') {
                        let d: u32 = d_str.parse().map_err(|_| {
                            DecoderError::InvalidConfiguration(format!("Invalid detector: {token}"))
                        })?;
                        // XOR: toggle membership
                        if !det_set.remove(&d) {
                            det_set.insert(d);
                        }
                        max_detector = Some(max_detector.map_or(d, |m| m.max(d)));
                    } else if let Some(l_str) = token.strip_prefix('L') {
                        let l: u32 = l_str.parse().map_err(|_| {
                            DecoderError::InvalidConfiguration(format!(
                                "Invalid observable: {token}"
                            ))
                        })?;
                        if !obs_set.remove(&l) {
                            obs_set.insert(l);
                        }
                        max_observable = Some(max_observable.map_or(l, |m| m.max(l)));
                    }
                }
            }

            let detectors: Vec<u32> = det_set.into_iter().collect();
            let observables: Vec<u32> = obs_set.into_iter().collect();
            mechanisms.push((probability, detectors, observables));
        }

        let num_detectors = max_detector.map_or(0, |m| m as usize + 1);
        let num_observables = max_observable.map_or(0, |m| m as usize + 1);
        let num_mechanisms = mechanisms.len();

        // Build matrices.
        let mut check_matrix = ndarray::Array2::<u8>::zeros((num_detectors, num_mechanisms));
        let mut observable_matrix = ndarray::Array2::<u8>::zeros((num_observables, num_mechanisms));
        let mut error_priors = Vec::with_capacity(num_mechanisms);

        for (col, (prob, detectors, observables)) in mechanisms.iter().enumerate() {
            error_priors.push(*prob);
            for &d in detectors {
                check_matrix[[d as usize, col]] = 1;
            }
            for &o in observables {
                observable_matrix[[o as usize, col]] = 1;
            }
        }

        Ok(Self {
            check_matrix,
            observable_matrix,
            error_priors,
            num_detectors,
            num_observables,
            num_mechanisms,
        })
    }

    /// Compute the observable prediction from a correction vector.
    ///
    /// Given a binary correction vector (one entry per mechanism, from a
    /// check-matrix decoder), returns the observable mask as
    /// `observable_matrix @ correction (mod 2)`.
    #[must_use]
    pub fn observables_from_correction(&self, correction: &[u8]) -> Vec<u8> {
        let mut obs = vec![0u8; self.num_observables];
        for (o, row) in self.observable_matrix.rows().into_iter().enumerate() {
            let mut sum = 0u8;
            for (m, &val) in row.iter().enumerate() {
                if val != 0 && m < correction.len() && correction[m] != 0 {
                    sum ^= 1;
                }
            }
            obs[o] = sum;
        }
        obs
    }

    /// Pack observable predictions into a bitmask (u64).
    ///
    /// Bit `i` is set if observable `i` is predicted to flip.
    #[must_use]
    pub fn observables_mask_from_correction(&self, correction: &[u8]) -> u64 {
        let obs = self.observables_from_correction(correction);
        let mut mask = 0u64;
        for (i, &v) in obs.iter().enumerate() {
            if v != 0 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

/// An edge in a matching graph extracted from a DEM.
#[derive(Debug, Clone)]
pub struct MatchingEdge {
    /// First detector node (always present).
    pub node1: u32,
    /// Second detector node, or `None` for a boundary edge.
    pub node2: Option<u32>,
    /// Weight for MWPM: `ln((1-p) / p)`.
    pub weight: f64,
    /// Observable indices flipped by this error.
    pub observables: Vec<u32>,
    /// Original error probability.
    pub probability: f64,
    /// Fault mechanism ID (DEM line number). Components from the same
    /// decomposed mechanism share the same `fault_id`.
    pub fault_id: usize,
}

/// Matching graph representation extracted from a Detector Error Model.
///
/// Parses a DEM into edges suitable for MWPM decoders (`PyMatching`, Fusion
/// Blossom). Each graphlike error mechanism (1-2 detectors) becomes one edge.
/// Decomposed mechanisms (`D0 ^ D1`) are split into their components.
/// Hyperedges (3+ detectors after resolution) are skipped with a warning.
///
/// # Example
///
/// ```
/// use pecos_decoder_core::dem::DemMatchingGraph;
///
/// let dem = "error(0.01) D0 D1 L0\nerror(0.02) D1";
/// let graph = DemMatchingGraph::from_dem_str(dem).unwrap();
/// assert_eq!(graph.edges.len(), 2);
/// assert_eq!(graph.num_detectors, 2);
/// ```
#[derive(Debug, Clone)]
pub struct DemMatchingGraph {
    /// Edges in the matching graph.
    pub edges: Vec<MatchingEdge>,
    /// Number of detectors (max detector ID + 1).
    pub num_detectors: usize,
    /// Number of observables (max observable ID + 1).
    pub num_observables: usize,
    /// Number of hyperedges skipped (3+ detectors).
    pub skipped_hyperedges: usize,
    /// Detector coordinates (from `detector(x,y,t) D_i` declarations).
    /// Indexed by detector ID. Empty if no detector declarations in DEM.
    pub detector_coords: Vec<Option<Vec<f64>>>,
}

impl DemMatchingGraph {
    /// Parse a DEM string into a matching graph.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if the DEM string is malformed.
    pub fn from_dem_str(dem: &str) -> Result<Self, DecoderError> {
        let mut edges = Vec::new();
        let mut max_detector: Option<u32> = None;
        let mut max_observable: Option<u32> = None;
        let mut skipped = 0usize;
        let mut fault_id = 0usize;

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || !line.starts_with("error(") {
                continue;
            }

            let close_paren = line.find(')').ok_or_else(|| {
                DecoderError::InvalidConfiguration("Missing closing parenthesis".into())
            })?;
            let prob_str = &line[6..close_paren];
            let probability: f64 = prob_str.parse().map_err(|_| {
                DecoderError::InvalidConfiguration(format!("Invalid probability: {prob_str}"))
            })?;

            if probability <= 0.0 {
                continue;
            }

            let weight = if probability < 1.0 {
                ((1.0 - probability) / probability).ln()
            } else {
                0.0
            };

            let tokens_str = &line[close_paren + 1..];

            // For decomposed mechanisms (with ^), each component is a separate edge.
            // For non-decomposed mechanisms, there's one component.
            let components: Vec<&str> = tokens_str.split('^').collect();

            for component in &components {
                let mut detectors = Vec::new();
                let mut observables = Vec::new();

                for token in component.split_whitespace() {
                    if let Some(d_str) = token.strip_prefix('D') {
                        let d: u32 = d_str.parse().map_err(|_| {
                            DecoderError::InvalidConfiguration(format!("Invalid detector: {token}"))
                        })?;
                        detectors.push(d);
                        max_detector = Some(max_detector.map_or(d, |m| m.max(d)));
                    } else if let Some(l_str) = token.strip_prefix('L') {
                        let l: u32 = l_str.parse().map_err(|_| {
                            DecoderError::InvalidConfiguration(format!(
                                "Invalid observable: {token}"
                            ))
                        })?;
                        observables.push(l);
                        max_observable = Some(max_observable.map_or(l, |m| m.max(l)));
                    }
                }

                match detectors.len() {
                    0 => {} // Pure observable error, skip
                    1 => {
                        edges.push(MatchingEdge {
                            node1: detectors[0],
                            node2: None, // boundary
                            weight,
                            observables,
                            probability,
                            fault_id,
                        });
                    }
                    2 => {
                        edges.push(MatchingEdge {
                            node1: detectors[0],
                            node2: Some(detectors[1]),
                            weight,
                            observables,
                            probability,
                            fault_id,
                        });
                    }
                    _ => {
                        skipped += 1;
                    }
                }
            }
            fault_id += 1;
        }

        let num_detectors = max_detector.map_or(0, |m| m as usize + 1);
        let num_observables = max_observable.map_or(0, |m| m as usize + 1);

        let edges = Self::merge_parallel_edges(edges);

        // Parse detector coordinates
        let coords = parse_detector_coords(dem);
        let mut detector_coords = vec![None; num_detectors];
        for dc in coords {
            if (dc.id as usize) < num_detectors {
                detector_coords[dc.id as usize] = Some(dc.coords);
            }
        }

        Ok(Self {
            edges,
            num_detectors,
            num_observables,
            skipped_hyperedges: skipped,
            detector_coords,
        })
    }

    /// Merge edges with independent fault-ID-aware probability combination.
    ///
    /// Components from the same fault mechanism (same `fault_id`) that land on
    /// the same edge pair are NOT merged -- they're part of one correlated event.
    /// Components from different fault mechanisms (different `fault_id`) are
    /// combined using: `p_combined = p_a*(1-p_b) + p_b*(1-p_a)`.
    ///
    /// This matches `PyMatching`'s "independent" merge strategy with fault ID tracking.
    pub(crate) fn merge_parallel_edges(edges: Vec<MatchingEdge>) -> Vec<MatchingEdge> {
        use std::collections::BTreeMap;

        type EdgeKey = (u32, Option<u32>);

        // First, deduplicate: for each (edge_key, fault_id), keep only one entry.
        // Multiple components from the same fault_id on the same edge just confirm
        // that the fault affects this edge -- don't double-count the probability.
        let mut per_fault: BTreeMap<(EdgeKey, usize), MatchingEdge> = BTreeMap::new();

        for edge in edges {
            let key = match edge.node2 {
                Some(n2) if edge.node1 > n2 => (n2, Some(edge.node1)),
                _ => (edge.node1, edge.node2),
            };
            let fault_key = (key, edge.fault_id);
            // First occurrence of this (edge, fault_id) wins
            per_fault.entry(fault_key).or_insert(MatchingEdge {
                node1: key.0,
                node2: key.1,
                ..edge
            });
        }

        // Now merge across different fault_ids for the same edge pair
        let mut merged: BTreeMap<EdgeKey, MatchingEdge> = BTreeMap::new();

        for ((edge_key, _fault_id), edge) in per_fault {
            if let Some(existing) = merged.get_mut(&edge_key) {
                // Independent combination: p_ab = p_a*(1-p_b) + p_b*(1-p_a)
                let p_a = existing.probability;
                let p_b = edge.probability;
                let p_combined = p_a * (1.0 - p_b) + p_b * (1.0 - p_a);
                existing.probability = p_combined;
                existing.weight = if p_combined > 0.0 && p_combined < 1.0 {
                    ((1.0 - p_combined) / p_combined).ln()
                } else if p_combined >= 1.0 {
                    0.0
                } else {
                    1e6
                };

                // Keep the first edge's observables (matching PyMatching's
                // INDEPENDENT strategy). If observables differ between parallel
                // edges on the same node pair, the code has distance 2.
            } else {
                merged.insert(edge_key, edge);
            }
        }

        merged.into_values().collect()
    }
}

/// Generic wrapper that combines any [`Decoder`] with a [`DemCheckMatrix`]
/// to implement [`ObservableDecoder`].
///
/// This is the proper way to use check-matrix decoders (BP+OSD, `UnionFind`,
/// `RelayBP`, etc.) in a sample+decode loop. The wrapper:
/// 1. Passes the syndrome to the inner decoder
/// 2. Gets back a correction vector
/// 3. Multiplies by the observable matrix to get the observable prediction
///
/// # Example
///
/// ```
/// use ndarray::ArrayView1;
///
/// use pecos_decoder_core::{
///     CheckMatrixObservableDecoder, Decoder, DecoderError, DecodingResultTrait, DemCheckMatrix,
///     ObservableDecoder,
/// };
///
/// struct CorrectionResult {
///     correction: Vec<u8>,
/// }
///
/// impl DecodingResultTrait for CorrectionResult {
///     fn is_successful(&self) -> bool {
///         true
///     }
///
///     fn correction(&self) -> &[u8] {
///         &self.correction
///     }
/// }
///
/// struct FirstMechanismDecoder {
///     checks: usize,
///     bits: usize,
/// }
///
/// impl Decoder for FirstMechanismDecoder {
///     type Result = CorrectionResult;
///     type Error = DecoderError;
///
///     fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
///         assert_eq!(input.len(), self.checks);
///         let mut correction = vec![0; self.bits];
///         correction[0] = 1;
///         Ok(CorrectionResult { correction })
///     }
///
///     fn check_count(&self) -> usize {
///         self.checks
///     }
///
///     fn bit_count(&self) -> usize {
///         self.bits
///     }
/// }
///
/// let dem_str = "error(0.01) D0 L0\nerror(0.02) D0";
/// let dcm = DemCheckMatrix::from_dem_str(dem_str).unwrap();
/// let inner_decoder = FirstMechanismDecoder {
///     checks: dcm.num_detectors,
///     bits: dcm.num_mechanisms,
/// };
/// let mut decoder = CheckMatrixObservableDecoder::new(inner_decoder, dcm);
///
/// let mask = decoder.decode_to_observables(&[1]).unwrap();
/// assert_eq!(mask, 0b1);
/// ```
pub struct CheckMatrixObservableDecoder<D> {
    /// The inner check-matrix decoder.
    pub decoder: D,
    /// The DEM check matrix (holds observable matrix for prediction).
    pub dem: DemCheckMatrix,
    /// Reusable syndrome buffer (avoids per-shot ndarray allocation).
    syndrome_arr: ndarray::Array1<u8>,
}

impl<D> CheckMatrixObservableDecoder<D> {
    /// Create a new wrapper from a decoder and its DEM check matrix.
    pub fn new(decoder: D, dem: DemCheckMatrix) -> Self {
        let len = dem.num_detectors;
        Self {
            decoder,
            dem,
            syndrome_arr: ndarray::Array1::zeros(len),
        }
    }
}

impl<D> super::ObservableDecoder for CheckMatrixObservableDecoder<D>
where
    D: super::Decoder,
{
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        use super::DecodingResultTrait;

        // Copy syndrome into reusable buffer (no allocation after first call)
        let len = syndrome.len();
        if self.syndrome_arr.len() != len {
            self.syndrome_arr = ndarray::Array1::zeros(len);
        }
        self.syndrome_arr
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(syndrome);
        let result = self
            .decoder
            .decode(&self.syndrome_arr.view())
            .map_err(|e| DecoderError::DecodingFailed(e.to_string()))?;

        let correction = result.correction();
        Ok(self.dem.observables_mask_from_correction(correction))
    }
}

/// Detector coordinate parsed from a DEM `detector(x, y, t) D_i` line.
#[derive(Debug, Clone)]
pub struct DetectorCoord {
    /// Detector ID.
    pub id: u32,
    /// Coordinates (typically x, y, t for surface codes).
    pub coords: Vec<f64>,
}

/// Parse detector coordinates from a DEM string.
///
/// Returns a list of `DetectorCoord` for each `detector(...)` declaration.
#[must_use]
pub fn parse_detector_coords(dem: &str) -> Vec<DetectorCoord> {
    let mut result = Vec::new();
    for line in dem.lines() {
        let line = line.trim();
        if !line.starts_with("detector(") {
            continue;
        }
        if let Some(close) = line.find(')') {
            let coord_str = &line[9..close];
            let coords: Vec<f64> = coord_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            // Find D_i after the closing paren
            let rest = &line[close + 1..];
            for token in rest.split_whitespace() {
                if let Some(d_str) = token.strip_prefix('D')
                    && let Ok(id) = d_str.parse::<u32>()
                {
                    result.push(DetectorCoord {
                        id,
                        coords: coords.clone(),
                    });
                }
            }
        }
    }
    result
}

/// Information about a detector error model
#[derive(Debug, Clone, PartialEq)]
pub struct DemInfo {
    /// Number of detectors
    pub detector_count: usize,
    /// Number of logical observables
    pub observable_count: usize,
    /// Number of error mechanisms
    pub error_count: usize,
    /// Detector coordinates (if specified)
    pub detector_coordinates: Option<Vec<Vec<f64>>>,
}

/// Builder pattern for DEM configuration
pub struct DemConfigBuilder {
    config: DemConfig,
}

impl DemConfigBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DemConfig::default(),
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Enable compression
    #[must_use]
    pub fn compressed(mut self, compressed: bool) -> Self {
        self.config.compressed = compressed;
        self
    }

    /// Set detector coordinates
    #[must_use]
    pub fn detector_coordinates(mut self, coords: Vec<Vec<f64>>) -> Self {
        self.config.detector_coordinates = Some(coords);
        self
    }

    /// Set maximum errors per detector
    #[must_use]
    pub fn max_errors_per_detector(mut self, max: usize) -> Self {
        self.config.max_errors_per_detector = Some(max);
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> DemConfig {
        self.config
    }
}

impl Default for DemConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dem_validation() {
        let valid_dem = r"
            error(0.01) D0
            error(0.01) D1 D2
            error(0.01) D0 L0
        ";
        assert!(utils::validate_dem(valid_dem).is_ok());

        let invalid_dem = r"
            invalid_command D0
        ";
        assert!(utils::validate_dem(invalid_dem).is_err());
    }

    #[test]
    fn test_dem_metadata_parsing() {
        let dem = r"
            error(0.01) D0
            error(0.01) D1 D2
            error(0.01) D3 L0
            error(0.01) D4 L1
        ";

        let (detectors, observables) = utils::parse_dem_metadata(dem).unwrap();
        assert_eq!(detectors, 5); // D0 through D4
        assert_eq!(observables, 2); // L0 and L1
    }

    #[test]
    fn test_dem_check_matrix_basic() {
        let dem = "error(0.01) D0 D1 L0\nerror(0.02) D1 D2\nerror(0.03) D0 D2 L0";
        let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();

        assert_eq!(dcm.num_detectors, 3);
        assert_eq!(dcm.num_observables, 1);
        assert_eq!(dcm.num_mechanisms, 3);
        assert_eq!(dcm.error_priors, vec![0.01, 0.02, 0.03]);

        // Check matrix: mechanism 0 -> D0,D1; mechanism 1 -> D1,D2; mechanism 2 -> D0,D2
        assert_eq!(dcm.check_matrix[[0, 0]], 1); // D0, mech 0
        assert_eq!(dcm.check_matrix[[1, 0]], 1); // D1, mech 0
        assert_eq!(dcm.check_matrix[[2, 0]], 0); // D2, mech 0
        assert_eq!(dcm.check_matrix[[0, 1]], 0); // D0, mech 1
        assert_eq!(dcm.check_matrix[[1, 1]], 1); // D1, mech 1
        assert_eq!(dcm.check_matrix[[2, 1]], 1); // D2, mech 1

        // Observable matrix: mechanism 0 -> L0; mechanism 1 -> none; mechanism 2 -> L0
        assert_eq!(dcm.observable_matrix[[0, 0]], 1);
        assert_eq!(dcm.observable_matrix[[0, 1]], 0);
        assert_eq!(dcm.observable_matrix[[0, 2]], 1);
    }

    #[test]
    fn test_dem_check_matrix_observables_from_correction() {
        let dem = "error(0.01) D0 L0\nerror(0.01) D1 L1\nerror(0.01) D0 D1 L0 L1";
        let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();

        // Correction activates mechanism 0 -> L0 flips
        assert_eq!(dcm.observables_mask_from_correction(&[1, 0, 0]), 0b01);
        // Correction activates mechanism 2 -> L0 and L1 flip
        assert_eq!(dcm.observables_mask_from_correction(&[0, 0, 1]), 0b11);
        // Correction activates mechanisms 0 and 2 -> L0 xor L0 = 0, L1 flips
        assert_eq!(dcm.observables_mask_from_correction(&[1, 0, 1]), 0b10);
    }

    #[test]
    fn test_dem_check_matrix_decomposed() {
        // Decomposed mechanism: D0 ^ D1 means XOR
        let dem = "error(0.01) D0 D1 ^ D1 D2";
        let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();

        // D1 appears in both components -> XOR cancels it
        assert_eq!(dcm.check_matrix[[0, 0]], 1); // D0
        assert_eq!(dcm.check_matrix[[1, 0]], 0); // D1 cancels
        assert_eq!(dcm.check_matrix[[2, 0]], 1); // D2
    }

    #[test]
    fn test_dem_check_matrix_empty() {
        let dem = "";
        let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();
        assert_eq!(dcm.num_mechanisms, 0);
        assert_eq!(dcm.num_detectors, 0);
    }

    #[test]
    fn test_dem_config_builder() {
        let config = DemConfigBuilder::new()
            .seed(42)
            .compressed(true)
            .max_errors_per_detector(2)
            .build();

        assert_eq!(config.seed, Some(42));
        assert!(config.compressed);
        assert_eq!(config.max_errors_per_detector, Some(2));
    }
}
