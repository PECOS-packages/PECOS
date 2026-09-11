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
//! The adapter supports at most 128 detectors and 128 observables. Bits are packed
//! little-endian: word `w`, bit `b` represents index `32*w + b`.

use pecos_frontier::{FrontierConfig, FrontierDecoder, SparseDem};
use std::cell::RefCell;

const MODEL_DEM: &str = include_str!(concat!(env!("OUT_DIR"), "/model.dem"));
const MAX_BITS: usize = 128;

pub const STATUS_OK: i32 = 0;
pub const STATUS_MODEL_ERROR: i32 = 1;
pub const STATUS_MODEL_TOO_WIDE: i32 = 2;
pub const STATUS_DECODE_ERROR: i32 = 3;

struct State {
    decoder: Option<FrontierDecoder>,
    detector_count: usize,
    result: [i32; 4],
    status: i32,
}

impl State {
    const fn empty() -> Self {
        Self {
            decoder: None,
            detector_count: 0,
            result: [0; 4],
            status: STATUS_MODEL_ERROR,
        }
    }

    fn initialize(&mut self, dem_source: &str) {
        self.decoder = None;
        self.detector_count = 0;
        self.result = [0; 4];

        let Ok(dem) = SparseDem::from_dem_str(dem_source) else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        if dem.num_detectors > MAX_BITS || dem.num_observables > MAX_BITS {
            self.status = STATUS_MODEL_TOO_WIDE;
            return;
        }
        self.detector_count = dem.num_detectors;
        match FrontierDecoder::from_sparse_dem(&dem, FrontierConfig::default()) {
            Ok(decoder) => {
                self.decoder = Some(decoder);
                self.status = STATUS_OK;
            }
            Err(_) => self.status = STATUS_MODEL_ERROR,
        }
    }

    fn decode(&mut self, words: [i32; 4]) {
        self.result = [0; 4];
        let mut syndrome = vec![0_u8; self.detector_count];
        for (detector, value) in syndrome.iter_mut().enumerate() {
            let word = words[detector / 32].cast_unsigned();
            *value = ((word >> (detector % 32)) & 1) as u8;
        }

        let Some(decoder) = self.decoder.as_mut() else {
            self.status = STATUS_MODEL_ERROR;
            return;
        };
        match decoder.decode(syndrome.as_slice()) {
            Ok(decoded) => {
                for observable in decoded.predicted.iter_set_bits() {
                    self.result[observable / 32] |= (1_u32 << (observable % 32)).cast_signed();
                }
                self.status = STATUS_OK;
            }
            Err(_) => self.status = STATUS_DECODE_ERROR,
        }
    }
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
}
