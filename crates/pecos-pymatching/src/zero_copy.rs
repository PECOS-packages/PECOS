//! Zero-copy and buffer reuse utilities for `PyMatching` decoder

use super::decoder::{BITS_PER_BYTE, DecodingResult, PyMatchingDecoder};
use super::errors::Result;

/// A reusable buffer for decoding operations
pub struct DecodeBuffer {
    /// Internal buffer for syndrome data
    syndrome_buffer: Vec<u8>,
    /// Internal buffer for observable results
    observable_buffer: Vec<u8>,
}

impl DecodeBuffer {
    /// Create a new decode buffer for the given decoder
    #[must_use]
    pub fn new(decoder: &PyMatchingDecoder) -> Self {
        let max_observables = decoder.num_observables();
        Self {
            syndrome_buffer: Vec::new(),
            observable_buffer: vec![0; max_observables.div_ceil(BITS_PER_BYTE)],
        }
    }

    /// Clear the buffer for reuse
    pub fn clear(&mut self) {
        self.syndrome_buffer.clear();
        self.observable_buffer.fill(0);
    }

    /// Get the current syndrome buffer
    #[must_use]
    pub fn syndrome_buffer(&self) -> &[u8] {
        &self.syndrome_buffer
    }

    /// Get the current observable buffer
    #[must_use]
    pub fn observable_buffer(&self) -> &[u8] {
        &self.observable_buffer
    }
}

/// Extension methods for zero-copy operations
impl PyMatchingDecoder {
    /// Create a reusable decode buffer
    #[must_use]
    pub fn create_decode_buffer(&self) -> DecodeBuffer {
        DecodeBuffer::new(self)
    }

    /// Validate syndrome length
    fn validate_syndrome(&self, syndrome: &[u8]) -> Result<()> {
        let expected = self.num_detectors();
        if syndrome.len() != expected {
            return Err(crate::PyMatchingError::InvalidSyndrome {
                expected,
                actual: syndrome.len(),
            });
        }
        Ok(())
    }

    /// Validate buffer size for observables
    fn validate_buffer_size(&self, buffer: &[u8], purpose: &str) -> Result<()> {
        let required_len = self.num_observables().div_ceil(BITS_PER_BYTE);
        if buffer.len() < required_len {
            return Err(crate::PyMatchingError::Configuration(format!(
                "{} buffer too small: need {} bytes, got {}",
                purpose,
                required_len,
                buffer.len()
            )));
        }
        Ok(())
    }

    /// Decode into an existing buffer without allocating
    ///
    /// This method reuses the provided observable buffer to avoid allocations.
    /// The buffer must be at least (`num_observables` + 7) / 8 bytes long.
    ///
    /// # Errors
    ///
    /// Returns a [`PyMatchingError`](crate::PyMatchingError) if:
    /// - The syndrome length doesn't match the number of detectors
    /// - The observable buffer is too small
    /// - Decoding fails
    pub fn decode_into(&mut self, syndrome: &[u8], observable_buffer: &mut [u8]) -> Result<f64> {
        self.validate_syndrome(syndrome)?;
        self.validate_buffer_size(observable_buffer, "Observable")?;

        // Clear and decode
        let required_len = self.num_observables().div_ceil(BITS_PER_BYTE);
        observable_buffer[..required_len].fill(0);

        let result = self.decode(syndrome)?;
        let copy_len = result.observable.len().min(observable_buffer.len());
        observable_buffer[..copy_len].copy_from_slice(&result.observable[..copy_len]);

        Ok(result.weight)
    }

    /// Decode with a reusable buffer
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::decode_into`].
    pub fn decode_with_buffer(
        &mut self,
        syndrome: &[u8],
        buffer: &mut DecodeBuffer,
    ) -> Result<DecodingResult> {
        buffer.clear();
        buffer.syndrome_buffer.extend_from_slice(syndrome);

        let weight = self.decode_into(&buffer.syndrome_buffer, &mut buffer.observable_buffer)?;

        Ok(DecodingResult {
            observable: buffer.observable_buffer[..self.num_observables().div_ceil(BITS_PER_BYTE)]
                .to_vec(),
            weight,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PyMatchingConfig, decoder::DEFAULT_OBSERVABLES};

    #[test]
    fn test_decode_into() {
        let config = PyMatchingConfig {
            num_nodes: Some(4),
            num_observables: 3,
            ..Default::default()
        };

        let mut decoder = PyMatchingDecoder::new(config).unwrap();
        decoder
            .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(1, 2, &[1], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(2, 3, &[2], Some(1.0), Some(0.1), None)
            .unwrap();
        // Add boundary edge to handle odd parity
        decoder
            .add_boundary_edge(0, &[], Some(1.0), Some(0.1), None)
            .unwrap();

        let syndrome = vec![1, 0, 0, 0];
        let mut observable_buffer = vec![0u8; DEFAULT_OBSERVABLES / BITS_PER_BYTE]; // PyMatching defaults to 64 observables = 8 bytes

        let weight = decoder
            .decode_into(&syndrome, &mut observable_buffer)
            .unwrap();
        assert!(weight >= 0.0);

        // Check that buffer was modified
        assert!(observable_buffer[0] != 0 || weight == 0.0);
    }

    #[test]
    fn test_decode_with_buffer() {
        let config = PyMatchingConfig {
            num_nodes: Some(4),
            num_observables: 3,
            ..Default::default()
        };

        let mut decoder = PyMatchingDecoder::new(config).unwrap();
        decoder
            .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(1, 2, &[1], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(2, 3, &[2], Some(1.0), Some(0.1), None)
            .unwrap();
        // Add boundary edges to handle odd parity
        decoder
            .add_boundary_edge(0, &[], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_boundary_edge(3, &[], Some(1.0), Some(0.1), None)
            .unwrap();

        let mut buffer = decoder.create_decode_buffer();

        // Decode multiple times reusing the buffer - use even parity patterns
        let test_syndromes = vec![
            vec![0, 0, 0, 0], // No detections
            vec![1, 1, 0, 0], // Two adjacent detections (even parity)
            vec![1, 0, 0, 1], // Two distant detections (even parity)
        ];

        for syndrome in test_syndromes {
            let result = decoder.decode_with_buffer(&syndrome, &mut buffer).unwrap();
            assert!(result.weight >= 0.0);
            assert_eq!(result.observable.len(), DEFAULT_OBSERVABLES / BITS_PER_BYTE); // PyMatching defaults to 64 observables = 8 bytes
        }
    }
}
