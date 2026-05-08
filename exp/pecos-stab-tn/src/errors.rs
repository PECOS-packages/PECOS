// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MpsError {
    #[error("site index {index} out of bounds (num_sites = {num_sites})")]
    SiteOutOfBounds { index: usize, num_sites: usize },

    #[error("gate dimension mismatch: expected {expected}x{expected}, got {rows}x{cols}")]
    GateDimMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
    },

    #[error("SVD failed to converge")]
    SvdFailed,

    #[error("sites {q0} and {q1} are not adjacent")]
    NonAdjacentSites { q0: usize, q1: usize },
}
