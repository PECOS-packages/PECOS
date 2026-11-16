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

use pecos_decoder_core::{dem::SparseDem, feasibility::DemFeasibility, obs_mask::ObsMask};

#[test]
fn solver_matches_bruteforce_support_and_returns_valid_witnesses() {
    let dem = SparseDem::from_dem_str("error(0) D4 L0\nerror(1) D1 L70\nerror(0.8) D0 D2 L1\nerror(0.2) D1 D2 ^ D0 L0\nerror(0.3) D3 L2\nerror(0.4) L71\nerror(0.5) D0 D3 L0").unwrap();
    let prepared = DemFeasibility::new(&dem).unwrap();
    assert_eq!(prepared.rank(), 4);
    for bits in 0..32 {
        let syndrome: Vec<u8> = (0..5).map(|i| u8::from(bits & (1 << i) != 0)).collect();
        let mut possible = false;
        for faults in 0..(1 << dem.mechanisms.len()) {
            let mut check = vec![0; 5];
            let mut supported = true;
            for (i, (p, ds, _)) in dem.mechanisms.iter().enumerate() {
                let active = faults & (1 << i) != 0;
                if (*p == 0.0 && active) || (p.to_bits() == 1.0_f64.to_bits() && !active) {
                    supported = false;
                }
                if active {
                    for &d in ds {
                        check[d as usize] ^= 1;
                    }
                }
            }
            possible |= supported && check == syndrome;
        }
        let result = prepared.solve(&syndrome).unwrap();
        assert_eq!(result.is_some(), possible);
        if let Some(explanation) = result {
            assert!(explanation.mechanisms.contains(&1));
            assert!(!explanation.mechanisms.contains(&0));
            let mut check = vec![0; 5];
            let mut logical = ObsMask::new();
            for i in explanation.mechanisms {
                for &d in &dem.mechanisms[i].1 {
                    check[d as usize] ^= 1;
                }
                for &l in &dem.mechanisms[i].2 {
                    logical.toggle(l as usize);
                }
            }
            assert_eq!(check, syndrome);
            assert_eq!(logical, explanation.observables);
        }
    }
}

#[test]
fn probability_order_and_modal_baseline_are_deterministic() {
    let dem = SparseDem::from_dem_str("error(0.01) D0 L0\nerror(0.1) D0 L70").unwrap();
    let solution = DemFeasibility::new(&dem)
        .unwrap()
        .solve(&[1])
        .unwrap()
        .unwrap();
    assert_eq!(solution.mechanisms, [1]);
    assert!(solution.observables.get(70));
    let dem = SparseDem::from_dem_str("error(0.9) D0 L0\nerror(0.2) D0 L70").unwrap();
    let solution = DemFeasibility::new(&dem)
        .unwrap()
        .solve(&[0])
        .unwrap()
        .unwrap();
    assert_eq!(solution.mechanisms, [0, 1]);
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\nerror(0.1) D0 L1").unwrap();
    assert_eq!(
        DemFeasibility::new(&dem)
            .unwrap()
            .solve(&[1])
            .unwrap()
            .unwrap()
            .mechanisms,
        [0]
    );
}

#[test]
fn declarations_empty_models_and_manual_invalid_models_are_checked() {
    for text in ["", "logical_observable L70", "error(0) L70", "error(1) L70"] {
        let dem = SparseDem::from_dem_str(text).unwrap();
        assert!(
            DemFeasibility::new(&dem)
                .unwrap()
                .solve(&[])
                .unwrap()
                .is_some()
        );
    }
    let mut dem = SparseDem::from_dem_str("detector D0").unwrap();
    let prepared = DemFeasibility::new(&dem).unwrap();
    assert!(prepared.solve(&[]).is_err());
    assert!(prepared.solve(&[1]).unwrap().is_none());
    for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        dem.mechanisms = vec![(p, vec![0], vec![])];
        assert!(DemFeasibility::new(&dem).is_err());
    }
    dem.mechanisms = vec![(0.1, vec![1], vec![])];
    assert!(DemFeasibility::new(&dem).is_err());
    dem.mechanisms = vec![(0.1, vec![0], vec![0])];
    assert!(DemFeasibility::new(&dem).is_err());
}

#[test]
fn detector_and_observable_words_cross_boundaries() {
    let dem = SparseDem::from_dem_str("error(0.2) D0 D65 L130\nerror(0.1) D64 L64").unwrap();
    let mut syndrome = vec![0; 66];
    syndrome[0] = 1;
    syndrome[64] = 1;
    syndrome[65] = 1;
    let result = DemFeasibility::new(&dem)
        .unwrap()
        .solve(&syndrome)
        .unwrap()
        .unwrap();
    assert_eq!(result.mechanisms, [0, 1]);
    assert!(result.observables.get(64));
    assert!(result.observables.get(130));
}
