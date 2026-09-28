// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Bare-WebAssembly adapter for the PECOS Frontier decoder.
//!
//! The module has no imports. All exported parameters and results are WebAssembly
//! `i32` values, with at most one result per function. This lowest-common-denominator
//! ABI runs on Quantinuum hardware, which requires those integer-only signatures.
//! The live-call adapter supports at most 128 detectors and 128 observables.
//! Embedded replay fixtures may contain more detectors. Bits are packed little-endian:
//! word `w`, bit `b` represents index `32*w + b`.

use pecos_frontier::{
    FrontierConfig, ObsMask, SparseDem, TrellisStreamingDecoder, deadline_column_order,
};
use std::cell::RefCell;

const MODEL_DEM: &str = include_str!(concat!(env!("OUT_DIR"), "/model.dem"));
const REPLAY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/replay.fwr"));
const LIVE_MAX_DETECTORS: usize = 128;
const MAX_OBSERVABLES: usize = 128;
const REPLAY_HEADER_BYTES: usize = 20;

pub const STATUS_OK: i32 = 0;
pub const STATUS_MODEL_ERROR: i32 = 1;
pub const STATUS_MODEL_TOO_WIDE: i32 = 2;
pub const STATUS_DECODE_ERROR: i32 = 3;
pub const STATUS_REPLAY_ERROR: i32 = 4;

struct State {
    decoder: Option<TrellisStreamingDecoder>,
    detector_count: usize,
    stream_columns_processed: usize,
    result: [i32; 4],
    status: i32,
    replay_errors: i32,
    replay_checksum: u32,
    prepared_replay: Option<PreparedReplay>,
}

struct PreparedReplay {
    bit_count: usize,
    words: [i32; 4],
    expected: [i32; 4],
}

impl State {
    const fn empty() -> Self {
        Self {
            decoder: None,
            detector_count: 0,
            stream_columns_processed: 0,
            result: [0; 4],
            status: STATUS_MODEL_ERROR,
            replay_errors: 0,
            replay_checksum: 0,
            prepared_replay: None,
        }
    }

    fn initialize(&mut self, dem_source: &str) {
        self.decoder = None;
        self.detector_count = 0;
        self.stream_columns_processed = 0;
        self.result = [0; 4];
        self.replay_errors = 0;
        self.replay_checksum = 0;
        self.prepared_replay = None;

        let Ok(dem) = SparseDem::from_dem_str(dem_source) else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        if dem.num_observables > MAX_OBSERVABLES {
            self.status = STATUS_MODEL_TOO_WIDE;
            return;
        }
        if replay_shot_count() > 0
            && (replay_u32(12) as usize != dem.num_detectors
                || replay_u32(16) as usize != dem.num_observables)
        {
            self.status = STATUS_REPLAY_ERROR;
            return;
        }
        self.detector_count = dem.num_detectors;
        let Ok(column_order) = deadline_column_order(&dem) else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        let config = FrontierConfig {
            column_order: Some(column_order),
            ..FrontierConfig::default()
        };
        match TrellisStreamingDecoder::from_sparse_dem(&dem, config) {
            Ok(decoder) => {
                self.decoder = Some(decoder);
                self.status = STATUS_OK;
            }
            Err(_) => self.status = STATUS_MODEL_ERROR,
        }
    }

    fn decode(&mut self, words: [i32; 4]) -> bool {
        if self.detector_count > LIVE_MAX_DETECTORS {
            self.status = STATUS_MODEL_TOO_WIDE;
            return false;
        }
        if has_bits_at_or_above(words, self.detector_count) {
            self.status = STATUS_DECODE_ERROR;
            return false;
        }
        let syndrome = (0..self.detector_count)
            .map(|detector| {
                let word = words[detector / 32].cast_unsigned();
                ((word >> (detector % 32)) & 1) as u8
            })
            .collect::<Vec<_>>();
        self.decode_syndrome(&syndrome)
    }

    fn decode_syndrome(&mut self, syndrome: &[u8]) -> bool {
        self.result = [0; 4];
        if syndrome.len() != self.detector_count {
            self.status = STATUS_REPLAY_ERROR;
            return false;
        }

        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return false;
        };
        decoder.reset();
        if decoder.feed_dense(syndrome).is_err() {
            self.status = STATUS_DECODE_ERROR;
            return false;
        }
        if let Ok(predicted) = decoder.flush_prediction() {
            self.store_result(&predicted);
            self.status = STATUS_OK;
            true
        } else {
            self.status = STATUS_DECODE_ERROR;
            false
        }
    }

    fn store_result(&mut self, predicted: &ObsMask) {
        self.result = [0; 4];
        for observable in predicted.iter_set_bits() {
            self.result[observable / 32] |= (1_u32 << (observable % 32)).cast_signed();
        }
    }

    fn begin_stream(&mut self) -> bool {
        self.result = [0; 4];
        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return false;
        };
        decoder.reset();
        self.stream_columns_processed = 0;
        self.status = STATUS_OK;
        true
    }

    fn push_stream_round(&mut self, bit_count: usize, words: [i32; 4]) -> Option<usize> {
        if bit_count > LIVE_MAX_DETECTORS {
            self.status = STATUS_MODEL_TOO_WIDE;
            return None;
        }
        let bits = (0..bit_count)
            .map(|detector| {
                let word = words[detector / 32].cast_unsigned();
                ((word >> (detector % 32)) & 1) as u8
            })
            .collect::<Vec<_>>();
        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return None;
        };
        if decoder.feed_prefix(&bits).is_err() {
            self.status = STATUS_DECODE_ERROR;
            return None;
        }
        if let Ok(columns_processed) = decoder.advance_prediction() {
            let processed = columns_processed.saturating_sub(self.stream_columns_processed);
            self.stream_columns_processed = columns_processed;
            self.status = STATUS_OK;
            Some(processed)
        } else {
            self.status = STATUS_DECODE_ERROR;
            None
        }
    }

    fn finish_stream(&mut self) -> bool {
        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return false;
        };
        if let Ok(predicted) = decoder.flush_prediction() {
            self.store_result(&predicted);
            self.status = STATUS_OK;
            true
        } else {
            self.status = STATUS_DECODE_ERROR;
            false
        }
    }

    fn replay_shot(&mut self, index: usize) -> Option<i32> {
        let (syndrome, expected) = replay_record(index)?;
        if !self.decode_syndrome(&syndrome) {
            return None;
        }
        let mismatch = i32::from(self.result != expected);
        for word in self.result {
            self.replay_checksum = self.replay_checksum.rotate_left(5) ^ word.cast_unsigned();
        }
        Some(mismatch)
    }

    fn replay_stream_shot(&mut self, index: usize, rounds: usize) -> Option<i32> {
        if rounds == 0 {
            return None;
        }
        let (syndrome, expected) = replay_record(index)?;
        if !self.begin_stream() {
            return None;
        }
        for round in 0..rounds {
            let start = round * syndrome.len() / rounds;
            let end = (round + 1) * syndrome.len() / rounds;
            let bit_count = end - start;
            if bit_count > LIVE_MAX_DETECTORS {
                self.status = STATUS_MODEL_TOO_WIDE;
                return None;
            }
            let mut words = [0_i32; 4];
            for (bit, &value) in syndrome[start..end].iter().enumerate() {
                if value != 0 {
                    words[bit / 32] |= (1_u32 << (bit % 32)).cast_signed();
                }
            }
            self.push_stream_round(bit_count, words)?;
        }
        if !self.finish_stream() {
            return None;
        }
        let mismatch = i32::from(self.result != expected);
        for word in self.result {
            self.replay_checksum = self.replay_checksum.rotate_left(5) ^ word.cast_unsigned();
        }
        Some(mismatch)
    }

    fn prepare_replay_stream(&mut self, index: usize, rounds: usize) -> bool {
        self.prepared_replay = None;
        if rounds == 0 {
            return false;
        }
        let Some((syndrome, expected)) = replay_record(index) else {
            return false;
        };
        if !self.begin_stream() {
            return false;
        }
        for round in 0..rounds {
            let start = round * syndrome.len() / rounds;
            let end = (round + 1) * syndrome.len() / rounds;
            let bit_count = end - start;
            if bit_count > LIVE_MAX_DETECTORS {
                self.status = STATUS_MODEL_TOO_WIDE;
                return false;
            }
            let mut words = [0_i32; 4];
            for (bit, &value) in syndrome[start..end].iter().enumerate() {
                if value != 0 {
                    words[bit / 32] |= (1_u32 << (bit % 32)).cast_signed();
                }
            }
            if round + 1 == rounds {
                self.prepared_replay = Some(PreparedReplay {
                    bit_count,
                    words,
                    expected,
                });
            } else if self.push_stream_round(bit_count, words).is_none() {
                return false;
            }
        }
        self.prepared_replay.is_some()
    }

    fn finish_prepared_replay_stream(&mut self) -> Option<i32> {
        let prepared = self.prepared_replay.take()?;
        self.push_stream_round(prepared.bit_count, prepared.words)?;
        if !self.finish_stream() {
            return None;
        }
        let mismatch = i32::from(self.result != prepared.expected);
        for word in self.result {
            self.replay_checksum = self.replay_checksum.rotate_left(5) ^ word.cast_unsigned();
        }
        Some(mismatch)
    }
}

fn has_bits_at_or_above(words: [i32; 4], detector_count: usize) -> bool {
    let complete_words = detector_count / i32::BITS as usize;
    let remaining_bits = detector_count % i32::BITS as usize;
    let first_padding_word = if remaining_bits == 0 {
        complete_words
    } else {
        let valid_mask = (1_u32 << remaining_bits) - 1;
        if words[complete_words].cast_unsigned() & !valid_mask != 0 {
            return true;
        }
        complete_words + 1
    };
    words[first_padding_word..].iter().any(|word| *word != 0)
}

fn replay_u32(offset: usize) -> u32 {
    u32::from_le_bytes(REPLAY[offset..offset + 4].try_into().unwrap())
}

fn replay_shot_count() -> usize {
    replay_u32(8) as usize
}

fn replay_detector_count() -> usize {
    replay_u32(12) as usize
}

fn replay_observable_count() -> usize {
    replay_u32(16) as usize
}

fn replay_record_bytes() -> usize {
    if replay_u32(4) == 1 {
        32
    } else {
        (replay_detector_count().div_ceil(32) + replay_observable_count().div_ceil(32)) * 4
    }
}

fn replay_record(index: usize) -> Option<(Vec<u8>, [i32; 4])> {
    if index >= replay_shot_count() {
        return None;
    }
    let offset = REPLAY_HEADER_BYTES + index * replay_record_bytes();
    let detector_words = if replay_u32(4) == 1 {
        4
    } else {
        replay_detector_count().div_ceil(32)
    };
    let observable_offset = offset + detector_words * 4;
    let mut syndrome = vec![0_u8; replay_detector_count()];
    let mut expected = [0_i32; 4];
    for (detector, value) in syndrome.iter_mut().enumerate() {
        let word = replay_u32(offset + (detector / 32) * 4);
        *value = ((word >> (detector % 32)) & 1) as u8;
    }
    for (word, value) in expected
        .iter_mut()
        .enumerate()
        .take(replay_observable_count().div_ceil(32))
    {
        *value = replay_u32(observable_offset + word * 4).cast_signed();
    }
    Some((syndrome, expected))
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State::empty()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn init() {
    STATE.with_borrow_mut(|state| state.initialize(MODEL_DEM));
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_decode(s0: i32, s1: i32, s2: i32, s3: i32) {
    STATE.with_borrow_mut(|state| state.decode([s0, s1, s2, s3]));
}

/// Begin an online decode whose syndrome arrives in contiguous detector rounds.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_begin() {
    STATE.with_borrow_mut(|state| {
        state.begin_stream();
    });
}

/// Append up to 128 detector bits and return the number of DEM columns advanced.
/// Returns -1 on an invalid round or decoder failure.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_push(bit_count: i32, s0: i32, s1: i32, s2: i32, s3: i32) -> i32 {
    let Ok(bit_count) = usize::try_from(bit_count) else {
        return -1;
    };
    STATE.with_borrow_mut(|state| {
        state
            .push_stream_round(bit_count, [s0, s1, s2, s3])
            .and_then(|processed| i32::try_from(processed).ok())
            .unwrap_or(-1)
    })
}

/// Complete the online decode after all detector rounds have arrived.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_finish() {
    STATE.with_borrow_mut(|state| {
        state.finish_stream();
    });
}

/// Return the number of hardware shots compiled into this module.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_shot_count() -> i32 {
    replay_shot_count().try_into().unwrap_or(i32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_detector_count() -> i32 {
    STATE.with_borrow(|state| state.detector_count.try_into().unwrap_or(i32::MAX))
}

/// Decode one compiled-in hardware shot and return 0/1 for correct/incorrect.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_shot(index: i32) -> i32 {
    let Ok(index) = usize::try_from(index) else {
        return -1;
    };
    STATE.with_borrow_mut(|state| {
        if let Some(mismatch) = state.replay_shot(index) {
            state.replay_errors = mismatch;
            mismatch
        } else {
            state.status = STATUS_REPLAY_ERROR;
            -1
        }
    })
}

/// Stream one compiled-in shot through evenly divided detector rounds.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_shot(index: i32, rounds: i32) -> i32 {
    let (Ok(index), Ok(rounds)) = (usize::try_from(index), usize::try_from(rounds)) else {
        return -1;
    };
    STATE.with_borrow_mut(|state| {
        if let Some(mismatch) = state.replay_stream_shot(index, rounds) {
            state.replay_errors = mismatch;
            mismatch
        } else {
            state.status = STATUS_REPLAY_ERROR;
            -1
        }
    })
}

/// Process all but the final detector round of an embedded shot.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_prepare(index: i32, rounds: i32) -> i32 {
    let (Ok(index), Ok(rounds)) = (usize::try_from(index), usize::try_from(rounds)) else {
        return -1;
    };
    STATE.with_borrow_mut(|state| -i32::from(!state.prepare_replay_stream(index, rounds)))
}

/// Process the prepared final round and return the correction's mismatch bit.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_finish() -> i32 {
    STATE.with_borrow_mut(|state| {
        if let Some(mismatch) = state.finish_prepared_replay_stream() {
            state.replay_errors = mismatch;
            mismatch
        } else {
            state.status = STATUS_REPLAY_ERROR;
            -1
        }
    })
}

/// Decode a contiguous range of compiled-in shots and return its error count.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_range(start: i32, count: i32) -> i32 {
    let (Ok(start), Ok(count)) = (usize::try_from(start), usize::try_from(count)) else {
        return -1;
    };
    let Some(end) = start.checked_add(count) else {
        return -1;
    };
    if end > replay_shot_count() {
        STATE.with_borrow_mut(|state| state.status = STATUS_REPLAY_ERROR);
        return -1;
    }

    STATE.with_borrow_mut(|state| {
        state.replay_errors = 0;
        state.replay_checksum = 0;
        state.prepared_replay = None;
        for index in start..end {
            let Some(mismatch) = state.replay_shot(index) else {
                state.status = STATUS_REPLAY_ERROR;
                return -1;
            };
            state.replay_errors += mismatch;
        }
        state.replay_errors
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_last_errors() -> i32 {
    STATE.with_borrow(|state| state.replay_errors)
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_checksum() -> i32 {
    STATE.with_borrow(|state| state.replay_checksum.cast_signed())
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_0() -> i32 {
    STATE.with_borrow(|state| state.result[0])
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_1() -> i32 {
    STATE.with_borrow(|state| state.result[1])
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_2() -> i32 {
    STATE.with_borrow(|state| state.result[2])
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_3() -> i32 {
    STATE.with_borrow(|state| state.result[3])
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_status() -> i32 {
    STATE.with_borrow(|state| state.status)
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_reset() {
    STATE.with_borrow_mut(|state| {
        state.result = [0; 4];
        state.replay_errors = 0;
        state.replay_checksum = 0;
        state.status = if state.decoder.is_some() {
            STATUS_OK
        } else {
            STATUS_MODEL_ERROR
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_initializes_and_decodes() {
        init();
        assert_eq!(frontier_status(), STATUS_OK);
        frontier_decode(1, 0, 0, 0);
        assert_eq!(frontier_status(), STATUS_OK);
        assert_eq!(frontier_result_0() & 1, 1);
        frontier_reset();
        assert_eq!(frontier_result_0(), 0);
    }

    #[test]
    fn malformed_and_wide_models_report_status() {
        let mut state = State::empty();
        state.initialize("error(not-a-probability) D0");
        assert_eq!(state.status, STATUS_MODEL_ERROR);

        state.initialize("error(0.1) D128");
        assert_eq!(state.status, STATUS_OK);
        state.decode([0; 4]);
        assert_eq!(state.status, STATUS_MODEL_TOO_WIDE);

        state.initialize("error(0.1) L128");
        assert_eq!(state.status, STATUS_MODEL_TOO_WIDE);
    }

    #[test]
    fn packing_crosses_i32_word_boundaries() {
        let mut state = State::empty();
        state.initialize("error(0.1) D32 L32");
        assert_eq!(state.status, STATUS_OK);

        state.decode([0, 1, 0, 0]);
        assert_eq!(state.status, STATUS_OK);
        assert_eq!(state.result, [0, 1, 0, 0]);
    }

    #[test]
    fn syndrome_bits_beyond_model_width_are_rejected() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0\nerror(0.2) D1");

        state.decode([1 | (1 << 5), 0, 0, 0]);

        assert_eq!(state.status, STATUS_DECODE_ERROR);
        assert_eq!(state.result, [0; 4]);

        state.initialize("error(0.1) D31");
        state.decode([0, 1, 0, 0]);
        assert_eq!(state.status, STATUS_DECODE_ERROR);
    }

    #[test]
    fn streaming_abi_accepts_detector_rounds() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0");
        assert!(state.begin_stream());
        assert_eq!(state.push_stream_round(1, [1, 0, 0, 0]), Some(1));
        assert!(state.finish_stream());
        assert_eq!(state.status, STATUS_OK);
        assert_eq!(state.result[0] & 1, 1);
    }

    #[test]
    fn prepared_replay_separates_prior_rounds_from_correction_latency() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0");
        assert!(state.begin_stream());
        state.prepared_replay = Some(PreparedReplay {
            bit_count: 1,
            words: [1, 0, 0, 0],
            expected: [1, 0, 0, 0],
        });
        assert_eq!(state.finish_prepared_replay_stream(), Some(0));
        assert_eq!(state.status, STATUS_OK);
    }

    #[test]
    fn empty_embedded_replay_is_well_formed() {
        if frontier_replay_shot_count() != 0 {
            return;
        }
        assert_eq!(frontier_replay_shot_count(), 0);
        init();
        assert_eq!(frontier_replay_range(0, 0), 0);
        assert_eq!(frontier_replay_shot(0), -1);
        assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
    }

    #[test]
    fn embedded_replay_summary() {
        let shots = frontier_replay_shot_count();
        if shots == 0 {
            return;
        }
        init();
        assert_eq!(frontier_status(), STATUS_OK);
        let selected = shots.min(32);
        let errors = frontier_replay_range(0, selected);
        assert!(errors >= 0);
        eprintln!(
            "native replay: fixture_shots={shots}, selected={selected}, errors={errors}, checksum={}",
            frontier_replay_checksum().cast_unsigned()
        );
    }
}
