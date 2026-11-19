//! Zero-copy and buffer reuse utilities for `PyMatching` decoder

use super::decoder::{DecodingResult, PyMatchingDecoder};
use super::errors::Result;

/// A reusable buffer for decoding operations
pub struct DecodeBuffer {
    /// Internal buffer for syndrome data
    syndrome_buffer: Vec<u8>,
    /// Internal buffer for observable results, one byte per observable
    observable_buffer: Vec<u8>,
}

impl DecodeBuffer {
    /// Create a new decode buffer for the given decoder
    #[must_use]
    pub fn new(decoder: &PyMatchingDecoder) -> Self {
        let max_observables = decoder.num_observables();
        Self {
            syndrome_buffer: Vec::new(),
            observable_buffer: vec![0; max_observables],
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

    /// Validate buffer size for observables
    fn validate_buffer_size(&self, buffer: &[u8], purpose: &str) -> Result<()> {
        let required_len = self.num_observables();
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
    /// The buffer holds one byte per observable, in the same layout as
    /// [`DecodingResult::observable`], so it must be at least `num_observables`
    /// bytes long. Bytes past `num_observables` are left untouched.
    ///
    /// # Errors
    ///
    /// Returns a [`PyMatchingError`](crate::PyMatchingError), checked in this
    /// order and before the buffer is written, if:
    /// - The syndrome length is outside the range [`Self::decode`] accepts
    /// - The observable buffer is too small
    /// - Decoding fails
    pub fn decode_into(&mut self, syndrome: &[u8], observable_buffer: &mut [u8]) -> Result<f64> {
        self.validate_syndrome(syndrome)?;
        self.validate_buffer_size(observable_buffer, "Observable")?;

        let result = self.decode(syndrome)?;
        let observables = &mut observable_buffer[..self.num_observables()];
        observables.fill(0);
        let copy_len = result.observable.len().min(observables.len());
        observables[..copy_len].copy_from_slice(&result.observable[..copy_len]);

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
        // Keep the caller's buffer intact when the syndrome is rejected.
        self.validate_syndrome(syndrome)?;
        buffer.clear();
        buffer.syndrome_buffer.extend_from_slice(syndrome);

        let weight = self.decode_into(&buffer.syndrome_buffer, &mut buffer.observable_buffer)?;

        Ok(DecodingResult {
            observable: buffer.observable_buffer[..self.num_observables()].to_vec(),
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
        let mut observable_buffer = vec![0u8; decoder.num_observables()];

        let weight = decoder
            .decode_into(&syndrome, &mut observable_buffer)
            .unwrap();
        assert!(weight >= 0.0);

        // Check that buffer was modified
        assert!(observable_buffer[0] != 0 || weight == 0.0);
    }

    #[test]
    fn buffered_decodes_return_every_observable() {
        // Observable 10 lies past the first ceil(11 / 8) = 2 bytes, so a
        // bit-packed buffer size would drop it.
        let mut decoder =
            PyMatchingDecoder::from_dem("error(0.1) D0 L10\nerror(0.1) D0 D1").unwrap();
        let expected = decoder.decode(&[1, 0]).unwrap().observable;
        assert_eq!(expected[10], 1);

        let mut observable_buffer = vec![0u8; decoder.num_observables()];
        decoder
            .decode_into(&[1, 0], &mut observable_buffer)
            .unwrap();
        assert_eq!(observable_buffer, expected);

        let mut buffer = decoder.create_decode_buffer();
        let result = decoder.decode_with_buffer(&[1, 0], &mut buffer).unwrap();
        assert_eq!(result.observable, expected);
    }

    #[test]
    fn rejected_syndrome_leaves_buffers_untouched() {
        let mut decoder =
            PyMatchingDecoder::from_dem("error(0.1) D0 L0\nerror(0.1) D0 D1").unwrap();
        let mut observable_buffer = vec![0xAB_u8; 1];
        assert!(matches!(
            decoder.decode_into(&[1], &mut observable_buffer),
            Err(crate::PyMatchingError::InvalidSyndrome {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(observable_buffer, [0xAB]);
        // A wrong syndrome is reported before a too-small buffer.
        assert!(matches!(
            decoder.decode_into(&[1], &mut []),
            Err(crate::PyMatchingError::InvalidSyndrome { .. })
        ));

        let mut buffer = decoder.create_decode_buffer();
        decoder.decode_with_buffer(&[1, 0], &mut buffer).unwrap();
        let (syndrome, observables) = (
            buffer.syndrome_buffer().to_vec(),
            buffer.observable_buffer().to_vec(),
        );
        assert!(decoder.decode_with_buffer(&[1], &mut buffer).is_err());
        assert_eq!(buffer.syndrome_buffer(), syndrome);
        assert_eq!(buffer.observable_buffer(), observables);
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
            assert_eq!(result.observable.len(), DEFAULT_OBSERVABLES); // PyMatching defaults to 64 observables
        }
    }
}
