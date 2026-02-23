//! Conversion utilities between PECOS (ndarray 0.17) and relay-bp (ndarray 0.16) types
//!
//! relay-bp pins ndarray >=0.15, <0.17 while PECOS uses ndarray 0.17.
//! These are different types to the Rust compiler, so all data must cross
//! the version boundary via raw slices. The overhead is negligible compared
//! to decoding time.

use crate::errors::{RelayBpError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::sync::Arc;

/// Convert a PECOS ndarray 0.17 `ArrayView1<u8>` syndrome to a relay-bp
/// ndarray 0.16 `Array1<u8>` by copying through a `Vec`.
pub(crate) fn syndrome_to_relay(syndrome: &ArrayView1<u8>) -> ndarray_016::Array1<u8> {
    let data: Vec<u8> = syndrome.iter().copied().collect();
    ndarray_016::Array1::from_vec(data)
}

/// Convert a relay-bp ndarray 0.16 `Array1<u8>` decoding result back to
/// PECOS ndarray 0.17 `Array1<u8>`.
pub(crate) fn relay_array1_to_pecos(arr: &ndarray_016::Array1<u8>) -> Array1<u8> {
    let data: Vec<u8> = arr.iter().copied().collect();
    Array1::from_vec(data)
}

/// Convert a `Vec<f64>` to relay-bp's ndarray 0.16 `Array1<f64>`.
pub(crate) fn vec_to_relay_array1_f64(v: &[f64]) -> ndarray_016::Array1<f64> {
    ndarray_016::Array1::from_vec(v.to_vec())
}

/// Convert a PECOS dense check matrix (ndarray 0.17 `ArrayView2<u8>`) to
/// relay-bp's `Arc<SparseBitMatrix>` via `sparse_bartite_graph_from_dense`.
pub(crate) fn check_matrix_to_relay(
    check_matrix: &ArrayView2<u8>,
) -> Result<Arc<relay_bp::decoder::SparseBitMatrix>> {
    let nrows = check_matrix.nrows();
    let ncols = check_matrix.ncols();

    if nrows == 0 || ncols == 0 {
        return Err(RelayBpError::InvalidMatrix(
            "Check matrix must have non-zero dimensions".to_string(),
        ));
    }

    // Copy data row-by-row to relay-bp's ndarray 0.16 Array2
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in check_matrix.rows() {
        data.extend(row.iter().copied());
    }
    let relay_matrix = ndarray_016::Array2::from_shape_vec((nrows, ncols), data).map_err(|e| {
        RelayBpError::InvalidMatrix(format!("Failed to create relay-bp matrix: {e}"))
    })?;

    let sparse = relay_bp::bipartite_graph::sparse_bartite_graph_from_dense(
        relay_matrix,
        sprs::CompressedStorage::CSR,
    );

    Ok(Arc::new(sparse))
}
