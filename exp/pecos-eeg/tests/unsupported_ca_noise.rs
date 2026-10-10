// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Consumers restricted to H/S must reject C/A before rate or support filtering.

use pecos_core::{Gate, gate_type::GateType};
use pecos_eeg::coherent_dem::{build_coherent_dem, build_coherent_dem_decomposable};
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{GateIndex, make_gate};
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_sparse, heisenberg_windowed, heisenberg_with_noise_map,
};
use pecos_eeg::noise_compression::compress_noise_to_boundaries;
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, NoiseInjection, NoiseSpec};

struct UnsupportedNoise(EegType);

impl NoiseSpec for UnsupportedNoise {
    fn noise_after_gate(&self, _: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        vec![NoiseInjection {
            eeg_type: self.0,
            // Neither label touches the detector; even zero rate must be rejected.
            label: Bm::x(1),
            label2: Some(Bm::y(1)),
            rate: 0.0,
        }]
    }
}

fn run(consumer: &str, eeg_type: EegType) {
    let gates = [make_gate(GateType::I, &[0])];
    let noise = UnsupportedNoise(eeg_type);
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], 2);
    let detector = Bm::z(0);
    match consumer {
        "map" => {
            build_noise_map(&gates, &noise, &[false]);
        }
        "mapped_walk" => {
            // Construct directly so map validation cannot mask a walk defect.
            let map = [Some(noise.exact_noise_after_gate(0, GateType::I, &[0]))];
            let _ = heisenberg_with_noise_map(&gates, &detector, &map, &initial, 0.0);
        }
        "windowed" => {
            heisenberg_windowed(&gates, &detector, &noise, &initial, 0.0, None, &[false]);
        }
        "sparse" | "sparse_mapped" => {
            let index = GateIndex::build(&gates, 2, &noise, &[false]);
            let map = [Some(noise.exact_noise_after_gate(0, GateType::I, &[0]))];
            heisenberg_sparse(
                &gates,
                &detector,
                &noise,
                &initial,
                0.0,
                &index,
                (consumer == "sparse_mapped").then_some(&map[..]),
            );
        }
        "coherent" => {
            build_coherent_dem(&gates, &noise, &[], &[], &[false]);
        }
        "decomposable" => {
            build_coherent_dem_decomposable(&gates, &noise, &[], &[], &[false]);
        }
        "compression" => {
            compress_noise_to_boundaries(&gates, &noise, &[false]);
        }
        _ => unreachable!(),
    }
}

macro_rules! rejection_test {
    ($name:ident, $consumer:literal, $kind:ident, $message:literal) => {
        #[test]
        #[should_panic(expected = $message)]
        fn $name() {
            run($consumer, EegType::$kind);
        }
    };
}

rejection_test!(map_c, "map", C, "unsupported EEG type C");
rejection_test!(map_a, "map", A, "unsupported EEG type A");
rejection_test!(mapped_walk_c, "mapped_walk", C, "unsupported EEG type C");
rejection_test!(mapped_walk_a, "mapped_walk", A, "unsupported EEG type A");
rejection_test!(windowed_c, "windowed", C, "unsupported EEG type C");
rejection_test!(windowed_a, "windowed", A, "unsupported EEG type A");
rejection_test!(sparse_c, "sparse", C, "unsupported EEG type C");
rejection_test!(sparse_a, "sparse", A, "unsupported EEG type A");
rejection_test!(
    sparse_mapped_c,
    "sparse_mapped",
    C,
    "unsupported EEG type C"
);
rejection_test!(
    sparse_mapped_a,
    "sparse_mapped",
    A,
    "unsupported EEG type A"
);
rejection_test!(coherent_c, "coherent", C, "unsupported EEG type C");
rejection_test!(coherent_a, "coherent", A, "unsupported EEG type A");
rejection_test!(decomposable_c, "decomposable", C, "unsupported EEG type C");
rejection_test!(decomposable_a, "decomposable", A, "unsupported EEG type A");
rejection_test!(compression_c, "compression", C, "unsupported EEG type C");
rejection_test!(compression_a, "compression", A, "unsupported EEG type A");
