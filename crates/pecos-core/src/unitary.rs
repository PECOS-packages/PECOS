// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Unitary gate algebra namespace.
//!
//! Re-exports the full unitary-level API so users can write:
//!
//! ```
//! use pecos_core::unitary::*;
//! use pecos_core::Angle64;
//!
//! let circuit = T(1) * CX(0, 1) * H(0);
//! let layer = RZ(Angle64::HALF_TURN / 4, 0) & H(1);
//! ```

pub use crate::unitary_rep::{
    CCX, CX, CXs, CY, CYs, CZ, CZs, Commutativity, H, Hs, I, Is, ParseUnitaryRepError, PhaseValue,
    QubitPairs, Qubits, RX, RXX, RXXs, RXs, RY, RYY, RYYs, RYs, RZ, RZZ, RZZs, RZs, RotationType,
    SWAP, SWAPs, SX, SXs, SY, SYs, SZ, SZs, T, Ts, Unitary, UnitaryRep, X, Xs, Y, Ys, Z, Zs, phase,
};
