//! Quantinuum WAVM ABI for the PECOS Frontier decoder.
//!
//! All exported parameters and results are WebAssembly `i32` values, as
//! required by Quantinuum. Detector and observable bits are packed little-endian:
//! word `w`, bit `b` represents index `32*w + b`.

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

    fn initialize(&mut self) {
        self.decoder = None;
        self.detector_count = 0;
        self.result = [0; 4];

        let Ok(dem) = SparseDem::from_dem_str(MODEL_DEM) else {
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
                    if observable >= MAX_BITS {
                        self.status = STATUS_MODEL_TOO_WIDE;
                        return;
                    }
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
    STATE.with_borrow_mut(State::initialize);
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
}
