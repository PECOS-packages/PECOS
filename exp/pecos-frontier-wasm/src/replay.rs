// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Shared, validated FWR1 v2 reader for build-time and native/Wasm replay.

#[derive(Clone, Copy, Debug)]
pub struct Replay<'a> {
    bytes: &'a [u8],
    pub shots: usize,
    pub detectors: usize,
    pub observables: usize,
    record_bytes: usize,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

impl<'a> Replay<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < 20 || &bytes[..4] != b"FWR1" {
            return Err("invalid FWR1 header");
        }
        if read_u32(bytes, 4) != 2 {
            return Err("FWR1 requires version 2");
        }
        let shots = read_u32(bytes, 8) as usize;
        let detectors = read_u32(bytes, 12) as usize;
        let observables = read_u32(bytes, 16) as usize;
        if observables > 128 {
            return Err("FWR1 exceeds the 128-observable result ABI");
        }
        let record_bytes = (detectors.div_ceil(32) + observables.div_ceil(32)) * 4;
        let length = shots
            .checked_mul(record_bytes)
            .and_then(|n| n.checked_add(20));
        if length != Some(bytes.len()) {
            return Err("invalid FWR1 length");
        }
        // Reject padding bits instead of silently discarding malformed records.
        for shot in 0..shots {
            let start = 20 + shot * record_bytes;
            for (width, offset) in [
                (detectors, start),
                (observables, start + detectors.div_ceil(32) * 4),
            ] {
                if width % 32 != 0
                    && read_u32(bytes, offset + (width / 32) * 4) >> (width % 32) != 0
                {
                    return Err("nonzero FWR1 padding bits");
                }
            }
        }
        Ok(Self {
            bytes,
            shots,
            detectors,
            observables,
            record_bytes,
        })
    }

    pub fn record(&self, index: usize) -> Option<(Vec<u8>, [i32; 4])> {
        if index >= self.shots {
            return None;
        }
        let start = 20 + index * self.record_bytes;
        let syndrome = (0..self.detectors)
            .map(|bit| ((read_u32(self.bytes, start + (bit / 32) * 4) >> (bit % 32)) & 1) as u8)
            .collect();
        let mut expected = [0; 4];
        for (word, value) in expected
            .iter_mut()
            .enumerate()
            .take(self.observables.div_ceil(32))
        {
            *value = read_u32(self.bytes, start + (self.detectors.div_ceil(32) + word) * 4)
                .cast_signed();
        }
        Some((syndrome, expected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_header_length_width_and_padding() {
        let mut bytes = b"FWR1".to_vec();
        for field in [2_u32, 1, 33, 33, 0, 1, 0, 1] {
            bytes.extend(field.to_le_bytes());
        }
        let replay = Replay::parse(&bytes).unwrap();
        let (syndrome, expected) = replay.record(0).unwrap();
        assert_eq!(syndrome[32], 1);
        assert_eq!(expected, [0, 1, 0, 0]);
        assert!(replay.record(1).is_none());
        assert!(Replay::parse(&bytes[..bytes.len() - 1]).is_err());
        bytes[4] = 1;
        assert!(Replay::parse(&bytes).is_err());
        bytes[4] = 2;
        bytes[16] = 129;
        assert!(Replay::parse(&bytes).is_err());
        bytes[16] = 33;
        bytes[24] = 2;
        assert!(Replay::parse(&bytes).is_err());
        assert!(Replay::parse(b"FWR1").is_err());
    }
}
