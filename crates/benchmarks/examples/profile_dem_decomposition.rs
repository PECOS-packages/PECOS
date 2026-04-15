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

//! Standalone hot-loop around `DetectorErrorModel::to_string_decomposed_maximally()`,
//! used to drive `samply` / `cargo flamegraph` while investigating the
//! HashMap/BTreeMap/Vec tradeoff in `dem_builder/types.rs`.

use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, DetectorErrorModel};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use std::fmt::Write;
use std::hint::black_box;

fn build_surface_code_dem(distance: usize, rounds: usize) -> DetectorErrorModel {
    let num_data = distance * distance;
    let num_ancilla = num_data - 1;

    let mut dag = DagCircuit::new();

    for q in 0..num_data {
        dag.pz(&[q]);
        dag.h(&[q]);
    }

    for _round in 0..rounds {
        for a in 0..num_ancilla {
            dag.pz(&[num_data + a]);
        }
        for a in 0..num_ancilla {
            let ancilla = num_data + a;
            let d1 = a % num_data;
            let d2 = (a + 1) % num_data;
            dag.cx(&[(ancilla, d1)]);
            dag.cx(&[(ancilla, d2)]);
        }
        for a in 0..num_ancilla {
            dag.mz(&[num_data + a]);
        }
    }

    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let mut detectors = String::from("[");
    let mut first = true;
    let mut det_id: u32 = 0;
    for round in 1..rounds {
        for a in 0..num_ancilla {
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let curr = -((round * num_ancilla - a) as i32);
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let prev = -(((round + 1) * num_ancilla - a) as i32);
            if !first {
                detectors.push(',');
            }
            write!(detectors, r#"{{"id":{det_id},"records":[{curr},{prev}]}}"#)
                .expect("writing to String cannot fail");
            det_id += 1;
            first = false;
        }
    }
    detectors.push(']');

    DemBuilder::new(&influence_map)
        .with_noise(0.001, 0.001, 0.001, 0.001)
        .with_detectors_json(&detectors)
        .expect("detectors json should parse")
        .with_observables_json("[]")
        .expect("observables json should parse")
        .build()
}

fn main() {
    // d=7 r=5 matches the slowest bench case (~17 ms per iter).
    // 200 iterations -> ~3.4 seconds of hot-path work, enough for samply.
    let dem = build_surface_code_dem(7, 5);
    for _ in 0..200 {
        let out = dem.to_string_decomposed_maximally();
        black_box(out);
    }
}
