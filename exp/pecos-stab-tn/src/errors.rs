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

/// Failures reported by checked matrix-product-state operations.
#[derive(Clone, Error, Debug, PartialEq, Eq)]
pub enum MpsError {
    /// A requested site does not exist in the MPS chain.
    #[error("site index {index} out of bounds (num_sites = {num_sites})")]
    SiteOutOfBounds {
        /// Requested zero-based site index.
        index: usize,
        /// Number of sites in the MPS chain.
        num_sites: usize,
    },

    /// A gate matrix does not have the dimension required by the target site or sites.
    #[error("gate dimension mismatch: expected {expected}x{expected}, got {rows}x{cols}")]
    GateDimMismatch {
        /// Required number of rows and columns.
        expected: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },

    /// The singular-value decomposition failed to converge.
    #[error("SVD failed to converge")]
    SvdFailed,

    /// An adjacent-site operation received sites that are not ordered neighbors.
    #[error("sites {q0} and {q1} are not adjacent")]
    NonAdjacentSites {
        /// First requested site.
        q0: usize,
        /// Second requested site.
        q1: usize,
    },
}
