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

//! The X/Z detector mask that the XZ DEM views need, read from detector fault
//! sectors, and its use on bivariate-bicycle memory DEMs.

use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::dem_views::{DetectorBasis, gari_transform, init_dets_view};
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, DetectorDef, DetectorErrorModel};
use pecos_qec::{BbMemoryBasis, bb_memory_circuit};
use pecos_quantum::Attribute;

fn detector(id: u32, sector: Option<&str>) -> DetectorDef {
    let mut detector = DetectorDef::new(id);
    detector.fault_sector = sector.map(str::to_string);
    detector
}

fn error_text(dem: &DetectorErrorModel) -> String {
    dem.detector_bases().unwrap_err().to_string()
}

#[test]
fn detector_bases_follow_detector_ids() {
    let mut dem = DetectorErrorModel::new();
    assert_eq!(dem.detector_bases().unwrap(), vec![]);
    dem.add_detector(detector(1, Some("Z")));
    dem.add_detector(detector(0, Some("X")));
    assert_eq!(
        dem.detector_bases().unwrap(),
        vec![DetectorBasis::X, DetectorBasis::Z]
    );
}

#[test]
fn detector_bases_reject_detectors_without_an_xz_sector() {
    let mut dem = DetectorErrorModel::new();
    dem.add_detector(detector(0, Some("X")));
    dem.add_detector(detector(1, None));
    assert!(error_text(&dem).contains("detector D1 is undefined or has no fault sector"));

    // A deformed-code family is a valid fault sector but not an X/Z family.
    dem.detectors[1].fault_sector = Some("XZ-rotated".into());
    assert!(error_text(&dem).contains(r#"detector D1 has fault sector "XZ-rotated""#));

    // A repeated id leaves another id undefined.
    dem.detectors[1] = detector(0, Some("Z"));
    assert!(error_text(&dem).contains("detector D1 is undefined or has no fault sector"));

    dem.detectors[1].id = 2;
    assert_eq!(
        error_text(&dem),
        dem.to_structured_decoder_dem().unwrap_err().to_string()
    );
}

fn check_bb_memory(basis: BbMemoryBasis, init: DetectorBasis) {
    let circuit = bb_memory_circuit(
        6,
        6,
        &[(3, 0), (0, 1), (0, 2)],
        &[(0, 3), (1, 0), (2, 0)],
        3,
        basis,
    )
    .unwrap();
    let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.001, 0.001, 0.001, 0.001).unwrap();
    let mask = dem.detector_bases().unwrap();
    assert_eq!(mask.len(), dem.num_detectors());

    // The generator labels each detector by its check family; the mask must
    // agree with every label.
    let Some(Attribute::String(json)) = circuit.get_meta("detectors") else {
        panic!("missing detector metadata")
    };
    let entries: serde_json::Value = serde_json::from_str(json).unwrap();
    for entry in entries.as_array().unwrap() {
        let id = usize::try_from(entry["id"].as_u64().unwrap()).unwrap();
        let expected = match entry["label"].as_str().unwrap().chars().next().unwrap() {
            'X' => DetectorBasis::X,
            'Z' => DetectorBasis::Z,
            _ => panic!("unexpected detector label"),
        };
        assert_eq!(mask[id], expected);
    }

    let sparse = SparseDem::from_structured_dem(&dem.to_structured_decoder_dem().unwrap());
    gari_transform(&sparse, &mask, init).unwrap();
    init_dets_view(&sparse, &mask, init).unwrap();

    let flip = |basis: &DetectorBasis| match basis {
        DetectorBasis::X => DetectorBasis::Z,
        DetectorBasis::Z => DetectorBasis::X,
    };
    let swapped: Vec<_> = mask.iter().map(flip).collect();
    assert!(gari_transform(&sparse, &swapped, init).is_err());
    assert!(gari_transform(&sparse, &mask, flip(&init)).is_err());
}

#[test]
fn bb72_x_memory_mask_drives_the_xz_views() {
    check_bb_memory(BbMemoryBasis::X, DetectorBasis::X);
}

#[test]
fn bb72_z_memory_mask_drives_the_xz_views() {
    check_bb_memory(BbMemoryBasis::Z, DetectorBasis::Z);
}
