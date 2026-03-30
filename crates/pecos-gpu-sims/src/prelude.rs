// Copyright 2025 The PECOS Developers
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

//! Convenient re-exports for common usage.
//!
//! # Example
//!
//! ```
//! use pecos_gpu_sims::prelude::*;
//!
//! let mut sim = GpuStateVec::new(4).unwrap();
//! sim.h(&qid(0));
//! sim.cx(&[(QubitId(0), QubitId(1))]);
//! ```

pub use crate::{GpuError, GpuStateVec};
pub use pecos_core::{QubitId, qid};
pub use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
