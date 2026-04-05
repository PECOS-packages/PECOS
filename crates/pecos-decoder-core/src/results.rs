//! Common result types for PECOS decoders
//!
//! Standardized result structures that all decoders
//! can use or convert to for interoperability.

/// Standard decoding result that all decoders can use
#[derive(Debug, Clone, PartialEq)]
pub struct StandardDecodingResult {
    /// The decoded observable outcome (standardized as Vec<u8>)
    pub observable: Vec<u8>,
    /// Weight/cost of the solution
    pub weight: f64,
    /// Whether the decoder converged (if applicable)
    pub converged: Option<bool>,
    /// Number of iterations performed (if applicable)
    pub iterations: Option<usize>,
    /// Confidence in the result (if applicable)
    pub confidence: Option<f64>,
}

impl StandardDecodingResult {
    /// Create a new standard result with minimal information
    #[must_use]
    pub fn new(observable: Vec<u8>, weight: f64) -> Self {
        Self {
            observable,
            weight,
            converged: None,
            iterations: None,
            confidence: None,
        }
    }

    /// Create a result with convergence information
    #[must_use]
    pub fn with_convergence(
        observable: Vec<u8>,
        weight: f64,
        converged: bool,
        iterations: usize,
    ) -> Self {
        Self {
            observable,
            weight,
            converged: Some(converged),
            iterations: Some(iterations),
            confidence: None,
        }
    }

    /// Add confidence information
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// Trait that all decoding results should implement
pub trait DecodingResultTrait {
    /// Whether the decoding was successful
    fn is_successful(&self) -> bool;

    /// Get the cost of the decoding (if available)
    fn cost(&self) -> Option<f64> {
        None
    }

    /// Get the number of iterations used (if applicable)
    fn iterations(&self) -> Option<usize> {
        None
    }

    /// Convert to standardized result format
    fn to_standard(&self) -> StandardDecodingResult {
        // Default implementation - individual decoders should override this
        StandardDecodingResult {
            observable: vec![],
            weight: self.cost().unwrap_or(0.0),
            converged: None,
            iterations: self.iterations(),
            confidence: None,
        }
    }
}

impl DecodingResultTrait for StandardDecodingResult {
    fn is_successful(&self) -> bool {
        self.converged.unwrap_or(true)
    }

    fn cost(&self) -> Option<f64> {
        Some(self.weight)
    }

    fn iterations(&self) -> Option<usize> {
        self.iterations
    }

    fn to_standard(&self) -> StandardDecodingResult {
        self.clone()
    }
}

/// Batch decoding result
#[derive(Debug, Clone)]
pub struct BatchDecodingResult {
    /// Individual results for each input
    pub results: Vec<StandardDecodingResult>,
    /// Total time taken (if measured)
    pub total_time: Option<std::time::Duration>,
    /// Number of successful decodings
    pub successful_count: usize,
}

impl BatchDecodingResult {
    /// Create a new batch result
    #[must_use]
    pub fn new(results: Vec<StandardDecodingResult>) -> Self {
        let successful_count = results
            .iter()
            .filter(|r| r.converged.unwrap_or(true))
            .count();
        Self {
            results,
            total_time: None,
            successful_count,
        }
    }

    /// Add timing information
    #[must_use]
    pub fn with_timing(mut self, duration: std::time::Duration) -> Self {
        self.total_time = Some(duration);
        self
    }

    /// Get success rate
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            self.successful_count as f64 / self.results.len() as f64
        }
    }

    /// Get average weight
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_weight(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            let sum: f64 = self.results.iter().map(|r| r.weight).sum();
            sum / self.results.len() as f64
        }
    }
}

/// Result builder for fluent API
pub struct ResultBuilder {
    observable: Vec<u8>,
    weight: f64,
    converged: Option<bool>,
    iterations: Option<usize>,
    confidence: Option<f64>,
}

impl ResultBuilder {
    /// Create a new result builder
    #[must_use]
    pub fn new(observable: Vec<u8>, weight: f64) -> Self {
        Self {
            observable,
            weight,
            converged: None,
            iterations: None,
            confidence: None,
        }
    }

    /// Set convergence status
    #[must_use]
    pub fn converged(mut self, converged: bool) -> Self {
        self.converged = Some(converged);
        self
    }

    /// Set iteration count
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = Some(iterations);
        self
    }

    /// Set confidence
    #[must_use]
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Build the result
    #[must_use]
    pub fn build(self) -> StandardDecodingResult {
        StandardDecodingResult {
            observable: self.observable,
            weight: self.weight,
            converged: self.converged,
            iterations: self.iterations,
            confidence: self.confidence,
        }
    }
}
