//! Public builder argument imports must work through the metacrate facade.
#![cfg(feature = "sim")]

use pecos::noise::*;

#[test]
fn readout_matrix_is_exported_with_noise_builder_arguments() {
    let rows = std::collections::BTreeMap::from([(
        "0".into(),
        std::collections::BTreeMap::from([("0".into(), 1.0)]),
    )]);
    let matrix: pecos_engines::prelude::ReadoutMatrix =
        ReadoutMatrix::new(pecos_core::gate_type::GateType::MZ, &rows).unwrap();
    assert!(
        GeneralNoiseModel::builder()
            .with_readout(matrix)
            .try_build()
            .is_ok()
    );
}
