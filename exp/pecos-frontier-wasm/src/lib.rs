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
//! The module has no imports. The original API uses WebAssembly `i32` values;
//! Helios callers use the explicit `_i64` streaming wrappers. Every function
//! returns at most one value. Decoder words remain 32-bit on both interfaces.
//! Each live call carries at most 128 detector bits; streaming and replay support
//! wider models. Corrections contain at most 128 observables. Bits are packed little-endian:
//! word `w`, bit `b` represents index `32*w + b`.

use pecos_frontier::{
    FrontierConfig, ObsMask, SparseDem, TrellisOrdering, TrellisStreamingDecoder,
};
use std::cell::RefCell;

const MODEL_DEM: &str = include_str!(concat!(env!("OUT_DIR"), "/model.dem"));
#[cfg(all(feature = "replay", not(test)))]
const REPLAY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/replay.fwr"));
const LIVE_MAX_DETECTORS: usize = 128;
const MAX_OBSERVABLES: usize = 128;
#[cfg(any(feature = "replay", test))]
mod replay;
// Deterministic records for the two-detector model: D0 -> L0, D1 -> no flip.
#[cfg(test)]
const REPLAY: &[u8] = &[
    70, 87, 82, 49, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0,
];

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
    #[cfg(any(feature = "replay", test))]
    prepared_replay: Option<PreparedReplay>,
}

#[cfg(any(feature = "replay", test))]
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
            #[cfg(any(feature = "replay", test))]
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
        #[cfg(any(feature = "replay", test))]
        {
            self.prepared_replay = None;
        }

        let Ok(dem) = SparseDem::from_dem_str(dem_source) else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        if dem.num_observables > MAX_OBSERVABLES {
            self.status = STATUS_MODEL_TOO_WIDE;
            return;
        }
        self.detector_count = dem.num_detectors;
        let Ok(column_order) = TrellisOrdering::Deadline.resolve(&dem) else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        let config = FrontierConfig {
            column_order,
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
        self.result = [0; 4];
        if self.detector_count > LIVE_MAX_DETECTORS {
            self.status = STATUS_MODEL_TOO_WIDE;
            return false;
        }
        if has_bits_at_or_above(words, self.detector_count) {
            self.status = STATUS_DECODE_ERROR;
            return false;
        }
        let bits = unpack_words(words);
        self.decode_syndrome(&bits[..self.detector_count])
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
        self.result = [0; 4];
        if bit_count > LIVE_MAX_DETECTORS {
            self.status = STATUS_MODEL_TOO_WIDE;
            return None;
        }
        if has_bits_at_or_above(words, bit_count) {
            self.status = STATUS_DECODE_ERROR;
            return None;
        }
        let bits = unpack_words(words);
        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return None;
        };
        if decoder.feed_prefix(&bits[..bit_count]).is_err() {
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
        self.result = [0; 4];
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

    fn finish_stream_round(&mut self, bit_count: usize, words: [i32; 4]) -> Option<i32> {
        self.push_stream_round(bit_count, words)?;
        if !self.finish_stream() {
            return None;
        }
        Some(self.result[0])
    }

    #[cfg(any(feature = "replay", test))]
    fn replay_mismatch(&mut self, expected: [i32; 4]) -> i32 {
        for word in self.result {
            self.replay_checksum = self.replay_checksum.rotate_left(5) ^ word.cast_unsigned();
        }
        i32::from(self.result != expected)
    }

    #[cfg(any(feature = "replay", test))]
    fn replay_shot(&mut self, index: usize) -> Option<i32> {
        let (syndrome, expected) = replay_record(index)?;
        if !self.decode_syndrome(&syndrome) {
            return None;
        }
        Some(self.replay_mismatch(expected))
    }

    #[cfg(any(feature = "replay", test))]
    fn replay_stream_shot(&mut self, index: usize, rounds: usize) -> Option<i32> {
        if !self.prepare_replay_stream(index, rounds) {
            return None;
        }
        self.finish_prepared_replay_stream()
    }

    #[cfg(any(feature = "replay", test))]
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

    #[cfg(any(feature = "replay", test))]
    fn finish_prepared_replay_stream(&mut self) -> Option<i32> {
        let prepared = self.prepared_replay.take()?;
        self.push_stream_round(prepared.bit_count, prepared.words)?;
        if !self.finish_stream() {
            return None;
        }
        Some(self.replay_mismatch(prepared.expected))
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

fn unpack_words(words: [i32; 4]) -> [u8; LIVE_MAX_DETECTORS] {
    std::array::from_fn(|bit| ((words[bit / 32].cast_unsigned() >> (bit % 32)) & 1) as u8)
}

#[cfg(any(feature = "replay", test))]
fn replay_shot_count() -> usize {
    REPLAY_FIXTURE.with(|fixture| fixture.shots)
}

#[cfg(any(feature = "replay", test))]
fn replay_record(index: usize) -> Option<(Vec<u8>, [i32; 4])> {
    REPLAY_FIXTURE.with(|fixture| fixture.record(index))
}

#[cfg(any(feature = "replay", test))]
thread_local! {
    static REPLAY_FIXTURE: replay::Replay<'static> = replay::Replay::parse(REPLAY).expect("build-validated fixture");
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State::empty()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn init() {
    #[cfg(any(feature = "replay", test))]
    let _ = replay_shot_count();
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
        STATE.with_borrow_mut(|state| {
            state.status = STATUS_DECODE_ERROR;
            state.result = [0; 4];
        });
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

/// Append the final detector round, flush the decoder, and return the first
/// observable-mask word. This makes the hardware timing boundary exactly one
/// Wasm call from the last syndrome block to the returned correction.
/// Returns -1 on failure, but -1 is also a valid correction. Always check
/// `frontier_status()` to distinguish success from failure.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_finish_round(
    bit_count: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
) -> i32 {
    let Ok(bit_count) = usize::try_from(bit_count) else {
        STATE.with_borrow_mut(|state| {
            state.status = STATUS_DECODE_ERROR;
            state.result = [0; 4];
        });
        return -1;
    };
    STATE.with_borrow_mut(|state| {
        state
            .finish_stream_round(bit_count, [s0, s1, s2, s3])
            .unwrap_or(-1)
    })
}

/// Convert a signed or unsigned 32-bit packed word carried by a Helios i64.
fn helios_word(value: i64) -> Option<i32> {
    i32::try_from(value)
        .ok()
        .or_else(|| u32::try_from(value).ok().map(u32::cast_signed))
}

fn helios_round(bit_count: i64, words: [i64; 4]) -> Option<(i32, [i32; 4])> {
    let bit_count = i32::try_from(bit_count).ok()?;
    Some((
        bit_count,
        [
            helios_word(words[0])?,
            helios_word(words[1])?,
            helios_word(words[2])?,
            helios_word(words[3])?,
        ],
    ))
}

fn reject_helios_call() -> i64 {
    STATE.with_borrow_mut(|state| {
        state.status = STATUS_DECODE_ERROR;
        state.result = [0; 4];
    });
    -1
}

/// Helios-compatible i64 call boundary; the decoder retains its i32 word layout.
/// Packed words may be sign-extended i32 or zero-extended u32 values.
/// Invalid values return -1, set decode-error status, and clear stale results.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_push_i64(
    bit_count: i64,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) -> i64 {
    let Some((count, [w0, w1, w2, w3])) = helios_round(bit_count, [s0, s1, s2, s3]) else {
        return reject_helios_call();
    };
    i64::from(frontier_stream_push(count, w0, w1, w2, w3))
}

/// Helios final-block-to-correction call.
/// Successful correction words are zero-extended from u32; errors return -1.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_stream_finish_round_i64(
    bit_count: i64,
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
) -> i64 {
    let Some((count, [w0, w1, w2, w3])) = helios_round(bit_count, [s0, s1, s2, s3]) else {
        return reject_helios_call();
    };
    let correction = frontier_stream_finish_round(count, w0, w1, w2, w3);
    if frontier_status() == STATUS_OK {
        i64::from(correction.cast_unsigned())
    } else {
        -1
    }
}

/// Helios-compatible status query.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_status_i64() -> i64 {
    i64::from(frontier_status())
}

/// Helios-compatible correction word 0, zero-extended from u32.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_0_i64() -> i64 {
    i64::from(frontier_result_0().cast_unsigned())
}

/// Helios-compatible correction word 1, zero-extended from u32.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_1_i64() -> i64 {
    i64::from(frontier_result_1().cast_unsigned())
}

/// Helios-compatible correction word 2, zero-extended from u32.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_2_i64() -> i64 {
    i64::from(frontier_result_2().cast_unsigned())
}

/// Helios-compatible correction word 3, zero-extended from u32.
#[unsafe(no_mangle)]
pub extern "C" fn frontier_result_3_i64() -> i64 {
    i64::from(frontier_result_3().cast_unsigned())
}

/// Return the number of hardware shots compiled into this module.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_shot_count() -> i32 {
    replay_shot_count().try_into().unwrap_or(i32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn frontier_detector_count() -> i32 {
    STATE.with_borrow(|state| state.detector_count.try_into().unwrap_or(i32::MAX))
}

/// Apply the same status and stale-state policy at every replay call boundary.
#[cfg(any(feature = "replay", test))]
fn with_replay_call(call: impl FnOnce(&mut State) -> Option<i32>) -> i32 {
    STATE.with_borrow_mut(|state| {
        if let Some(result) = call(state) {
            state.status = STATUS_OK;
            result
        } else {
            state.status = STATUS_REPLAY_ERROR;
            state.result = [0; 4];
            state.prepared_replay = None;
            -1
        }
    })
}

/// Decode one compiled-in hardware shot and return 0/1 for correct/incorrect.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_shot(index: i32) -> i32 {
    with_replay_call(|state| {
        let mismatch = state.replay_shot(usize::try_from(index).ok()?)?;
        state.replay_errors = mismatch;
        Some(mismatch)
    })
}

/// Stream one compiled-in shot through evenly divided detector rounds.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_shot(index: i32, rounds: i32) -> i32 {
    with_replay_call(|state| {
        let mismatch = state
            .replay_stream_shot(usize::try_from(index).ok()?, usize::try_from(rounds).ok()?)?;
        state.replay_errors = mismatch;
        Some(mismatch)
    })
}

/// Process all but the final detector round of an embedded shot.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_prepare(index: i32, rounds: i32) -> i32 {
    with_replay_call(|state| {
        state
            .prepare_replay_stream(usize::try_from(index).ok()?, usize::try_from(rounds).ok()?)
            .then_some(0)
    })
}

/// Process the prepared final round and return the correction's mismatch bit.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_stream_finish() -> i32 {
    with_replay_call(|state| {
        let mismatch = state.finish_prepared_replay_stream()?;
        state.replay_errors = mismatch;
        Some(mismatch)
    })
}

/// Decode a contiguous range of compiled-in shots and return its error count.
#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_range(start: i32, count: i32) -> i32 {
    with_replay_call(|state| {
        let (start, count) = (usize::try_from(start).ok()?, usize::try_from(count).ok()?);
        let end = start.checked_add(count)?;
        if end > replay_shot_count() {
            return None;
        }
        state.result = [0; 4];
        state.replay_errors = 0;
        state.replay_checksum = 0;
        state.prepared_replay = None;
        for index in start..end {
            state.replay_errors += state.replay_shot(index)?;
        }
        Some(state.replay_errors)
    })
}

#[cfg(any(feature = "replay", test))]
#[unsafe(no_mangle)]
pub extern "C" fn frontier_replay_last_errors() -> i32 {
    STATE.with_borrow(|state| state.replay_errors)
}

#[cfg(any(feature = "replay", test))]
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
        #[cfg(any(feature = "replay", test))]
        {
            state.prepared_replay = None;
        }
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
    fn helios_words_preserve_all_32_bits_without_truncation() {
        assert_eq!(helios_word(i64::from(i32::MIN)), Some(i32::MIN));
        assert_eq!(helios_word(0x8000_0000), Some(i32::MIN));
        assert_eq!(helios_word(0xffff_ffff), Some(-1));
        assert_eq!(helios_word(-1), Some(-1));
        assert_eq!(helios_word(1_i64 << 32), None);
        assert_eq!(helios_word(i64::from(i32::MIN) - 1), None);
        STATE.with_borrow_mut(|state| {
            state.initialize("error(0.1) D0 L0");
            state.result = [7; 4];
        });
        frontier_stream_begin();
        assert_eq!(frontier_stream_push_i64(1_i64 << 32, 0, 0, 0, 0), -1);
        assert_eq!(frontier_status_i64(), i64::from(STATUS_DECODE_ERROR));
        assert_eq!(STATE.with_borrow(|state| state.result), [0; 4]);

        frontier_stream_begin();
        STATE.with_borrow_mut(|state| state.result = [7; 4]);
        assert_eq!(
            frontier_stream_finish_round_i64(1, 1_i64 << 32, 0, 0, 0),
            -1
        );
        assert_eq!(frontier_status_i64(), i64::from(STATUS_DECODE_ERROR));
        assert_eq!(STATE.with_borrow(|state| state.result), [0; 4]);
    }

    #[test]
    fn helios_streaming_matches_i32_exports() {
        // A synthetic 33-detector model exercises both high bits and word boundaries.
        STATE.with_borrow_mut(|state| state.initialize("error(0.1) D31 L0\nerror(0.1) D32 L1"));
        frontier_stream_begin();
        assert!(frontier_stream_push_i64(32, 0x8000_0000, 0, 0, 0) >= 0);
        let unsigned = frontier_stream_finish_round_i64(1, 1, 0, 0, 0);
        assert_eq!(unsigned, 3);
        assert_eq!(frontier_status_i64(), 0);
        frontier_stream_begin();
        assert!(frontier_stream_push_i64(32, i64::from(i32::MIN), 0, 0, 0) >= 0);
        assert_eq!(frontier_stream_finish_round_i64(1, 1, 0, 0, 0), unsigned);
        frontier_stream_begin();
        assert!(frontier_stream_push(32, i32::MIN, 0, 0, 0) >= 0);
        assert_eq!(
            i64::from(frontier_stream_finish_round(1, 1, 0, 0, 0)),
            unsigned
        );
    }

    #[test]
    fn helios_final_correction_is_zero_extended() {
        let observables = (0..32)
            .map(|observable| format!("L{observable}"))
            .collect::<Vec<_>>()
            .join(" ");
        STATE.with_borrow_mut(|state| {
            state.initialize(&format!("error(0.1) D0 {observables}"));
        });

        frontier_stream_begin();
        assert_eq!(
            frontier_stream_finish_round_i64(1, 1, 0, 0, 0),
            i64::from(u32::MAX)
        );
        assert_eq!(frontier_status_i64(), i64::from(STATUS_OK));

        // A decoder error must stay negative rather than zero-extending -1.
        frontier_stream_begin();
        assert_eq!(frontier_stream_finish_round_i64(1, 2, 0, 0, 0), -1);
        assert_eq!(frontier_status_i64(), i64::from(STATUS_DECODE_ERROR));
    }

    #[test]
    fn helios_result_words_are_zero_extended() {
        STATE.with_borrow_mut(|state| {
            state.initialize("error(0.1) D0 L31 L32 L33 L63 L66 L95 L96 L127");
        });

        frontier_stream_begin();
        assert_eq!(frontier_stream_finish_round_i64(1, 1, 0, 0, 0), 0x8000_0000);
        assert_eq!(frontier_status_i64(), i64::from(STATUS_OK));
        assert_eq!(
            [
                frontier_result_0_i64(),
                frontier_result_1_i64(),
                frontier_result_2_i64(),
                frontier_result_3_i64(),
            ],
            [0x8000_0000, 0x8000_0003, 0x8000_0004, 0x8000_0001]
        );
    }

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
    fn final_round_returns_correction_in_one_call() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0");
        assert!(state.begin_stream());
        assert_eq!(state.finish_stream_round(1, [1, 0, 0, 0]), Some(1));
        assert_eq!(state.status, STATUS_OK);
        assert_eq!(state.result[0], 1);
    }

    #[test]
    fn prepared_replay_separates_prior_rounds_from_correction_latency() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0\nerror(0.2) D1");
        assert!(state.prepare_replay_stream(0, 2));
        assert_eq!(state.prepared_replay.as_ref().unwrap().bit_count, 1);
        assert_eq!(state.finish_prepared_replay_stream(), Some(0));
        assert_eq!(state.result, [1, 0, 0, 0]);
        let checksum = state.replay_checksum;
        assert!(state.finish_prepared_replay_stream().is_none());
        assert_eq!(state.replay_stream_shot(1, 2), Some(0));
        assert_eq!(state.result, [0; 4]);
        assert_eq!(state.replay_checksum, checksum.rotate_left(20));
        assert!(!state.prepare_replay_stream(2, 2));
        assert!(!state.prepare_replay_stream(0, 0));
    }

    #[test]
    fn replay_failures_set_status_and_discard_stale_preparations() {
        let invalid_calls: &[fn() -> i32] = &[
            || frontier_replay_shot(-1),
            || frontier_replay_shot(2),
            || frontier_replay_stream_shot(-1, 2),
            || frontier_replay_stream_shot(0, -1),
            || frontier_replay_stream_shot(0, 0),
            || frontier_replay_stream_shot(2, 2),
            || frontier_replay_stream_prepare(-1, 2),
            || frontier_replay_stream_prepare(0, -1),
            || frontier_replay_stream_prepare(0, 0),
            || frontier_replay_stream_prepare(2, 2),
            || frontier_replay_range(-1, 1),
            || frontier_replay_range(0, -1),
            || frontier_replay_range(0, 3),
            || frontier_replay_range(i32::MAX, i32::MAX),
        ];
        for call in invalid_calls {
            init();
            assert_eq!(frontier_replay_shot(0), 0);
            assert_eq!(frontier_result_0(), 1);
            assert_eq!(call(), -1);
            assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
            assert_eq!(frontier_result_0(), 0);

            assert_eq!(frontier_replay_stream_prepare(0, 2), 0);
            assert_eq!(frontier_status(), STATUS_OK);
            assert_eq!(call(), -1);
            assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
            assert_eq!(frontier_replay_stream_finish(), -1);
            assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
            assert_eq!(frontier_replay_range(0, 0), 0);
            assert_eq!(frontier_status(), STATUS_OK);
        }
    }

    #[test]
    fn reset_cancels_prepared_replay_and_allows_a_fresh_shot() {
        init();
        assert_eq!(frontier_replay_stream_prepare(0, 2), 0);
        frontier_reset();
        assert_eq!(frontier_status(), STATUS_OK);
        assert_eq!(frontier_replay_stream_finish(), -1);
        assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
        assert_eq!(frontier_result_0(), 0);
        assert_eq!(frontier_replay_checksum(), 0);
        assert_eq!(frontier_replay_last_errors(), 0);
        assert_eq!(frontier_replay_stream_prepare(0, 2), 0);
        assert_eq!(frontier_replay_stream_finish(), 0);
        assert_eq!(frontier_status(), STATUS_OK);
        assert_eq!(frontier_result_0(), 1);
    }

    #[test]
    fn embedded_replay_summary() {
        STATE.with_borrow_mut(|state| state.initialize("error(0.1) D0 L0\nerror(0.2) D1"));
        assert_eq!(frontier_replay_shot_count(), 2);
        assert_eq!(frontier_replay_range(0, 2), 0);
        assert_eq!(
            frontier_replay_checksum().cast_unsigned(),
            1_u32.rotate_left(3)
        );
        assert_eq!(frontier_replay_shot(2), -1);
        assert_eq!(frontier_status(), STATUS_REPLAY_ERROR);
    }

    #[test]
    fn invalid_live_calls_clear_results_and_reject_padding() {
        let mut state = State::empty();
        state.initialize("error(0.1) D0 L0");
        assert!(state.decode([1, 0, 0, 0]));
        assert!(!state.decode([3, 0, 0, 0]));
        assert_eq!(state.result, [0; 4]);
        for (bits, words) in [
            (0, [1, 0, 0, 0]),
            (1, [3, 0, 0, 0]),
            (32, [0, 1, 0, 0]),
            (127, [0, 0, 0, i32::MIN]),
        ] {
            assert!(state.begin_stream());
            assert!(state.push_stream_round(bits, words).is_none());
            assert_eq!(state.status, STATUS_DECODE_ERROR);
            assert_eq!(state.result, [0; 4]);
        }
        assert!(state.begin_stream());
        assert_eq!(state.finish_stream_round(1, [1, 0, 0, 0]), Some(1));
    }

    #[test]
    fn wide_models_stream_across_calls_but_reject_single_call_decode() {
        let mut state = State::empty();
        state.initialize("error(0.1) D128 L69");
        assert_eq!(state.status, STATUS_OK);
        assert!(state.begin_stream());
        assert!(state.push_stream_round(128, [0; 4]).is_some());
        assert_eq!(state.finish_stream_round(1, [1, 0, 0, 0]), Some(0));
        assert_eq!(state.result, [0, 0, 32, 0]);
        assert!(!state.decode([0; 4]));
        assert_eq!(state.status, STATUS_MODEL_TOO_WIDE);
        assert_eq!(state.result, [0; 4]);
    }

    #[test]
    fn all_ones_correction_is_success_not_error_sentinel() {
        let dem = format!(
            "error(0.1) D0 {}",
            (0..32)
                .map(|i| format!("L{i}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        STATE.with_borrow_mut(|state| state.initialize(&dem));
        frontier_stream_begin();
        assert_eq!(frontier_stream_finish_round(1, 1, 0, 0, 0), -1);
        assert_eq!(frontier_status(), STATUS_OK);
        assert_eq!(frontier_stream_finish_round(-1, 0, 0, 0, 0), -1);
        assert_eq!(frontier_status(), STATUS_DECODE_ERROR);
        assert_eq!(frontier_result_0(), 0);
    }
}
