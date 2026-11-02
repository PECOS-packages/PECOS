//! Exercise the patched archive dependency through relay's actual NPZ consumer.

use ndarray_016::{Array1, array};
use ndarray_npy::{NpzReader, NpzWriter};
use relay_bp::utilities::sparse::load_sparse_npz;
use std::fs::File;

fn check_sparse_npz(compressed: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checks.npz");
    let file = File::create(&path).unwrap();
    let mut writer = if compressed {
        NpzWriter::new_compressed(file)
    } else {
        NpzWriter::new(file)
    };
    // CSR representation of the three-bit repetition code's check matrix.
    writer.add_array("shape", &array![2_i64, 3]).unwrap();
    writer.add_array("data", &array![1_u8, 1, 1, 1]).unwrap();
    writer
        .add_array("indices", &array![0_i32, 1, 1, 2])
        .unwrap();
    writer.add_array("indptr", &array![0_i32, 2, 4]).unwrap();
    writer.finish().unwrap();

    let mut reader = NpzReader::new(File::open(&path).unwrap()).unwrap();
    assert_eq!(
        reader.names().unwrap(),
        ["shape", "data", "indices", "indptr"]
    );
    let shape: Array1<i64> = reader.by_name("shape").unwrap();
    let shape_with_suffix: Array1<i64> = reader.by_name("shape.npy").unwrap();
    assert_eq!(shape, array![2_i64, 3]);
    assert_eq!(shape_with_suffix, shape);
    assert!(
        reader
            .by_name::<ndarray_016::OwnedRepr<u8>, ndarray_016::Ix1>("missing")
            .is_err()
    );

    let checks = load_sparse_npz::<u8>(path).unwrap();
    assert_eq!(checks.shape(), (2, 3));
    assert_eq!(checks.to_dense(), array![[1_u8, 1, 0], [0, 1, 1]]);
}

#[test]
fn relay_reads_stored_npz() {
    check_sparse_npz(false);
}

#[test]
fn relay_reads_compressed_npz() {
    check_sparse_npz(true);
}
