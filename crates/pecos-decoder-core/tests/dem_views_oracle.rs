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

//! External oracle for the DEM views: the init-basis-detector restriction,
//! the GARI transform (arXiv:2510.14060), and the canonical XZ-memory
//! detector mask. Expected values in `tests/data` were produced by the
//! reference Python implementation shipped with arXiv:2607.28795 on
//! distance-3 surface-code X- and Z-memory DEMs (see `tests/data/README.md`).
//! Priors are compared as f64 bit patterns.

use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::dem_views::{
    DetectorBasis, GariColumnBlock, GariModel, gari_transform, init_dets_view,
    xz_memory_detector_bases,
};
use serde_json::Value;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{DATA}{name}")).unwrap()
}

fn json(name: &str) -> Value {
    serde_json::from_str(&read(name)).unwrap()
}

fn basis_of(c: char) -> DetectorBasis {
    match c {
        'X' => DetectorBasis::X,
        'Z' => DetectorBasis::Z,
        other => panic!("bad mask char {other}"),
    }
}

fn fixture(dem_name: &str, mask_name: &str) -> (SparseDem, Vec<DetectorBasis>) {
    let dem = SparseDem::from_dem_str(&read(dem_name)).unwrap();
    let bases: Vec<DetectorBasis> = read(mask_name).trim().chars().map(basis_of).collect();
    assert_eq!(bases.len(), dem.num_detectors);
    (dem, bases)
}

fn x_toy() -> (SparseDem, Vec<DetectorBasis>) {
    fixture("toy_xz_d3.dem", "toy_xz_d3_mask.txt")
}

fn z_toy() -> (SparseDem, Vec<DetectorBasis>) {
    fixture("toy_xz_d3_zinit.dem", "toy_xz_d3_zinit_mask.txt")
}

fn usize_list(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| usize::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn bits_list(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().parse::<u64>().unwrap())
        .collect()
}

fn sorted_rows(v: &Value) -> Vec<Vec<usize>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let mut row = usize_list(r);
            row.sort_unstable();
            row
        })
        .collect()
}

/// Every mechanism's supports must be strictly increasing (canonical form).
fn assert_canonical(dem: &SparseDem) {
    for (col, (p, dets, obs)) in dem.mechanisms.iter().enumerate() {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(p),
            "column {col} prior {p}"
        );
        assert!(
            dets.windows(2).all(|w| w[0] < w[1]),
            "column {col} detectors {dets:?}"
        );
        assert!(
            obs.windows(2).all(|w| w[0] < w[1]),
            "column {col} observables {obs:?}"
        );
        assert!(dets.iter().all(|&d| (d as usize) < dem.num_detectors));
        assert!(obs.iter().all(|&o| (o as usize) < dem.num_observables));
    }
}

/// Row -> sorted column indices of a `SparseDem` viewed as an H matrix.
fn h_rows(dem: &SparseDem) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); dem.num_detectors];
    for (col, (_, dets, _)) in dem.mechanisms.iter().enumerate() {
        for &d in dets {
            rows[d as usize].push(col);
        }
    }
    rows
}

/// Observable -> sorted column indices.
fn l_rows(dem: &SparseDem) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); dem.num_observables];
    for (col, (_, _, obs)) in dem.mechanisms.iter().enumerate() {
        for &o in obs {
            rows[o as usize].push(col);
        }
    }
    rows
}

fn syndrome_of(dem: &SparseDem, err: &[u8]) -> Vec<u8> {
    let mut s = vec![0u8; dem.num_detectors];
    for (col, (_, dets, _)) in dem.mechanisms.iter().enumerate() {
        if err[col] == 1 {
            for &det in dets {
                s[det as usize] ^= 1;
            }
        }
    }
    s
}

fn flips_of(dem: &SparseDem, err: &[u8]) -> Vec<u8> {
    let mut s = vec![0u8; dem.num_observables];
    for (col, (_, _, obs)) in dem.mechanisms.iter().enumerate() {
        if err[col] == 1 {
            for &o in obs {
                s[o as usize] ^= 1;
            }
        }
    }
    s
}

/// Compare a GARI model with a reference fixture, exactly.
fn check_gari_against(gari: &GariModel, expected: &Value, n_det: usize, init: DetectorBasis) {
    assert_eq!(gari.num_detectors, n_det);
    assert_eq!(gari.init_basis, init);
    let answer = match init {
        DetectorBasis::X => GariColumnBlock::EbarZ,
        DetectorBasis::Z => GariColumnBlock::EbarX,
    };
    assert_eq!(gari.answer_block, answer);
    assert_canonical(&gari.dem);

    let cb = &expected["col_blocks"];
    let bounds = |name: &str| {
        let b = usize_list(&cb[name]);
        b[0]..b[1]
    };
    assert_eq!(gari.columns(GariColumnBlock::EZ), bounds("eZ"));
    assert_eq!(gari.columns(GariColumnBlock::EX), bounds("eX"));
    assert_eq!(gari.columns(GariColumnBlock::EY), bounds("eY"));
    assert_eq!(gari.columns(GariColumnBlock::EbarZ), bounds("ebarZ"));
    assert_eq!(gari.columns(GariColumnBlock::EbarX), bounds("ebarX"));
    let rb = &expected["row_blocks"];
    assert_eq!(gari.detector_rows(), 0..usize_list(&rb["detectors"])[1]);
    assert_eq!(
        gari.u_rows(),
        usize_list(&rb["U"])[0]..usize_list(&rb["U"])[1]
    );
    assert_eq!(
        gari.v_rows(),
        usize_list(&rb["V"])[0]..usize_list(&rb["V"])[1]
    );

    let h_shape = usize_list(&expected["h_shape"]);
    assert_eq!(gari.dem.num_detectors, h_shape[0]);
    assert_eq!(gari.dem.mechanisms.len(), h_shape[1]);
    assert_eq!(
        gari.dem.num_observables,
        usize_list(&expected["l_shape"])[0]
    );
    assert_eq!(
        h_rows(&gari.dem),
        sorted_rows(&expected["h_rows"]),
        "Hbar rows differ"
    );
    assert_eq!(
        l_rows(&gari.dem),
        sorted_rows(&expected["l_rows"]),
        "Lbar rows differ"
    );

    let priors: Vec<u64> = gari.dem.mechanisms.iter().map(|m| m.0.to_bits()).collect();
    assert_eq!(
        priors,
        bits_list(&expected["priors_bits"]),
        "priors differ bitwise"
    );
    assert_eq!(
        gari.relevant_rows
            .iter()
            .map(|&r| r as usize)
            .collect::<Vec<_>>(),
        usize_list(&expected["relevant_rows"])
    );
    let rel: Vec<u64> = gari.relevant_priors.iter().map(|p| p.to_bits()).collect();
    assert_eq!(
        rel,
        bits_list(&expected["relevant_priors_bits"]),
        "relevant priors differ"
    );
    assert_eq!(
        gari.u_map.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        usize_list(&expected["u_map"])
    );
    assert_eq!(
        gari.v_map.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        usize_list(&expected["v_map"])
    );
    assert!(gari.relevant_rows.windows(2).all(|w| w[0] < w[1]));
}

/// Independent equivalence check from the public block structure, over
/// every physical unit vector (complete, since the equations are linear).
fn check_gari_equivalence(gari: &GariModel) {
    let physical = gari.physical_columns();
    assert_canonical(physical);
    let n_phys = physical.mechanisms.len();
    let ez = gari.columns(GariColumnBlock::EZ);
    let ex = gari.columns(GariColumnBlock::EX);
    let ey = gari.columns(GariColumnBlock::EY);
    let ebarz = gari.columns(GariColumnBlock::EbarZ);
    let ebarx = gari.columns(GariColumnBlock::EbarX);
    assert_eq!(n_phys, ey.end);
    let n_det = gari.num_detectors;
    for j in 0..n_phys {
        let mut e = vec![0u8; n_phys];
        e[j] = 1;
        let mut e_hat = vec![0u8; gari.dem.mechanisms.len()];
        e_hat[j] = 1;
        if ez.contains(&j) {
            e_hat[ebarz.start + j - ez.start] ^= 1;
        } else if ex.contains(&j) {
            e_hat[ebarx.start + j - ex.start] ^= 1;
        } else {
            let k = j - ey.start;
            e_hat[ebarz.start + gari.u_map[k] as usize] ^= 1;
            e_hat[ebarx.start + gari.v_map[k] as usize] ^= 1;
        }
        let lhs = syndrome_of(&gari.dem, &e_hat);
        let rhs = syndrome_of(physical, &e);
        assert_eq!(&lhs[..n_det], &rhs[..], "column {j}: detector rows");
        assert!(
            lhs[n_det..].iter().all(|&b| b == 0),
            "column {j}: consistency rows"
        );
        assert_eq!(
            flips_of(&gari.dem, &e_hat),
            flips_of(physical, &e),
            "column {j}: observables"
        );
    }
    let (original, transformed) = gari.edge_counts();
    assert_eq!(
        original,
        physical.mechanisms.iter().map(|m| m.1.len()).sum::<usize>()
    );
    assert_eq!(
        transformed,
        gari.dem.mechanisms.iter().map(|m| m.1.len()).sum::<usize>()
    );
}

/// Compare an init-dets view with a reference fixture, exactly.
fn check_init_against(
    dem: &SparseDem,
    bases: &[DetectorBasis],
    init: DetectorBasis,
    expected: &Value,
) {
    let view = init_dets_view(dem, bases, init).unwrap();
    assert_canonical(&view.dem);
    assert_eq!(
        view.detector_index
            .iter()
            .map(|&i| i as usize)
            .collect::<Vec<_>>(),
        usize_list(&expected["init_idx"])
    );
    assert!(
        view.detector_index
            .iter()
            .all(|&i| bases[i as usize] == init)
    );
    let h_shape = usize_list(&expected["h_shape"]);
    assert_eq!(view.dem.num_detectors, h_shape[0]);
    assert_eq!(view.dem.mechanisms.len(), h_shape[1]);
    assert_eq!(
        view.dem.num_observables,
        usize_list(&expected["l_shape"])[0]
    );
    for (col, exp) in expected["columns"].as_array().unwrap().iter().enumerate() {
        let (p, dets, obs) = &view.dem.mechanisms[col];
        let got_dets: Vec<usize> = dets.iter().map(|&d| d as usize).collect();
        let got_obs: Vec<usize> = obs.iter().map(|&o| o as usize).collect();
        let mut exp_dets = usize_list(&exp["dets"]);
        exp_dets.sort_unstable();
        let mut exp_obs = usize_list(&exp["obs"]);
        exp_obs.sort_unstable();
        assert_eq!(got_dets, exp_dets, "column {col} detectors");
        assert_eq!(got_obs, exp_obs, "column {col} observables");
        let want: u64 = exp["p_bits"].as_str().unwrap().parse().unwrap();
        assert_eq!(p.to_bits(), want, "column {col} prior {p} differs bitwise");
    }
    let full: Vec<u8> = (0..dem.num_detectors)
        .map(|i| u8::from(i % 3 == 0))
        .collect();
    let projected = view.project_syndrome(&full).unwrap();
    assert_eq!(projected.len(), view.dem.num_detectors);
    for (row, &orig) in view.detector_index.iter().enumerate() {
        assert_eq!(projected[row], full[orig as usize]);
    }
    assert!(view.project_syndrome(&full[..full.len() - 1]).is_err());
    let mut longer = full.clone();
    longer.push(0);
    assert!(view.project_syndrome(&longer).is_err());
}

#[test]
fn canonical_xz_memory_masks_match_reference() {
    let cases = json("xz_memory_masks_expected.json");
    for case in cases.as_array().unwrap() {
        let n_x = usize::try_from(case["n_x"].as_u64().unwrap()).unwrap();
        let n_z = usize::try_from(case["n_z"].as_u64().unwrap()).unwrap();
        let rounds = usize::try_from(case["rounds"].as_u64().unwrap()).unwrap();
        let init = basis_of(case["init_basis"].as_str().unwrap().chars().next().unwrap());
        let got: String = xz_memory_detector_bases(n_x, n_z, rounds, init)
            .iter()
            .map(|b| match b {
                DetectorBasis::X => 'X',
                DetectorBasis::Z => 'Z',
            })
            .collect();
        assert_eq!(got, case["mask"].as_str().unwrap(), "{case}");
    }
}

#[test]
fn x_memory_gari_matches_reference_exactly() {
    let (dem, bases) = x_toy();
    let gari = gari_transform(&dem, &bases, DetectorBasis::X).unwrap();
    check_gari_against(
        &gari,
        &json("toy_xz_d3_gari_expected.json"),
        24,
        DetectorBasis::X,
    );
    check_gari_equivalence(&gari);
    let syndrome: Vec<u8> = (0..24).map(|i| u8::from(i % 5 == 0)).collect();
    let extended = gari.extend_syndrome(&syndrome).unwrap();
    assert_eq!(extended.len(), gari.dem.num_detectors);
    assert_eq!(&extended[..24], &syndrome[..]);
    assert!(extended[24..].iter().all(|&b| b == 0));
    assert!(gari.extend_syndrome(&syndrome[..23]).is_err());
    assert!(gari.extend_syndrome(&extended).is_err());
}

#[test]
fn z_memory_gari_matches_reference_exactly() {
    let (dem, bases) = z_toy();
    assert_eq!(bases.iter().filter(|&&b| b == DetectorBasis::X).count(), 8);
    let gari = gari_transform(&dem, &bases, DetectorBasis::Z).unwrap();
    check_gari_against(
        &gari,
        &json("toy_xz_d3_zinit_gari_expected.json"),
        24,
        DetectorBasis::Z,
    );
    check_gari_equivalence(&gari);
    let ebarx = gari.columns(GariColumnBlock::EbarX);
    for (col, (_, _, obs)) in gari.dem.mechanisms.iter().enumerate() {
        assert!(
            obs.is_empty() || ebarx.contains(&col),
            "observable outside ebarX at {col}"
        );
    }
    assert!(gari_transform(&dem, &bases, DetectorBasis::X).is_err());
}

#[test]
fn duplicated_lines_merge_exactly_like_the_reference() {
    // The same X-memory model with every error line repeated (every third
    // one three times): 124 lines merge to the same 53 columns with
    // combined priors.
    let (dem, bases) = fixture("toy_xz_d3_dup.dem", "toy_xz_d3_mask.txt");
    assert_eq!(dem.mechanisms.len(), 124);
    let gari = gari_transform(&dem, &bases, DetectorBasis::X).unwrap();
    check_gari_against(
        &gari,
        &json("toy_xz_d3_dup_gari_expected.json"),
        24,
        DetectorBasis::X,
    );
    check_gari_equivalence(&gari);
    let single = gari_transform(&x_toy().0, &bases, DetectorBasis::X).unwrap();
    assert_eq!(h_rows(&gari.dem), h_rows(&single.dem));
    assert_ne!(
        gari.dem.mechanisms[0].0.to_bits(),
        single.dem.mechanisms[0].0.to_bits()
    );
    check_init_against(
        &dem,
        &bases,
        DetectorBasis::X,
        &json("toy_xz_d3_dup_init_dets_expected.json"),
    );
}

#[test]
fn unsorted_supports_are_canonicalized_before_keying() {
    // Reverse every support list: the transform must treat it as the same
    // model, and its output must be canonical.
    let (dem, bases) = x_toy();
    let mut scrambled = dem.clone();
    for m in &mut scrambled.mechanisms {
        m.1.reverse();
        m.2.reverse();
    }
    let a = gari_transform(&dem, &bases, DetectorBasis::X).unwrap();
    let b = gari_transform(&scrambled, &bases, DetectorBasis::X).unwrap();
    assert_eq!(h_rows(&a.dem), h_rows(&b.dem));
    assert_eq!(l_rows(&a.dem), l_rows(&b.dem));
    assert_eq!(a.u_map, b.u_map);
    let pa: Vec<u64> = a.dem.mechanisms.iter().map(|m| m.0.to_bits()).collect();
    let pb: Vec<u64> = b.dem.mechanisms.iter().map(|m| m.0.to_bits()).collect();
    assert_eq!(pa, pb);
    assert_canonical(&b.dem);
    let va = init_dets_view(&dem, &bases, DetectorBasis::X).unwrap();
    let vb = init_dets_view(&scrambled, &bases, DetectorBasis::X).unwrap();
    assert_eq!(va.dem.mechanisms, vb.dem.mechanisms);
}

#[test]
fn gari_transform_rejects_structural_violations() {
    let (dem, bases) = x_toy();
    // Wrong init basis: the X-memory observable lives on eZ columns, which
    // are the off side for Z initialization.
    assert!(gari_transform(&dem, &bases, DetectorBasis::Z).is_err());
    // Mask length mismatch.
    assert!(gari_transform(&dem, &bases[..23], DetectorBasis::X).is_err());
    // A mixed column (one X-basis and one Z-basis detector) with no pure
    // partner on either side.
    let x_det = bases.iter().position(|&b| b == DetectorBasis::X).unwrap();
    let z_det = bases.iter().position(|&b| b == DetectorBasis::Z).unwrap();
    let mut broken = dem.clone();
    broken.mechanisms.push((
        0.01,
        vec![u32::try_from(x_det).unwrap(), u32::try_from(z_det).unwrap()],
        vec![],
    ));
    assert!(gari_transform(&broken, &bases, DetectorBasis::X).is_err());
    // A detector-silent column.
    let mut silent = dem.clone();
    silent.mechanisms.push((0.01, vec![], vec![0]));
    assert!(gari_transform(&silent, &bases, DetectorBasis::X).is_err());
    // Identical detector support with different observable support.
    let mut collide = dem.clone();
    let (p, dets, obs) = collide.mechanisms[1].clone();
    let other_obs = if obs.is_empty() { vec![0] } else { vec![] };
    collide.mechanisms.push((p, dets, other_obs));
    assert!(gari_transform(&collide, &bases, DetectorBasis::X).is_err());
    // A mixed column whose X restriction matches an existing eZ column but
    // whose observables differ: no partner.
    let ey_col = dem
        .mechanisms
        .iter()
        .position(|(_, dets, _)| {
            dets.iter().any(|&d| bases[d as usize] == DetectorBasis::X)
                && dets.iter().any(|&d| bases[d as usize] == DetectorBasis::Z)
        })
        .unwrap();
    let (_, dets, obs) = dem.mechanisms[ey_col].clone();
    let flipped_obs = if obs.is_empty() { vec![0] } else { vec![] };
    let spare_z = bases
        .iter()
        .enumerate()
        .find(|&(i, &b)| b == DetectorBasis::Z && !dets.contains(&u32::try_from(i).unwrap()))
        .map(|(i, _)| u32::try_from(i).unwrap())
        .unwrap();
    let mut new_dets = dets.clone();
    new_dets.push(spare_z);
    new_dets.sort_unstable();
    let mut mismatched = dem.clone();
    mismatched.mechanisms.push((0.01, new_dets, flipped_obs));
    assert!(gari_transform(&mismatched, &bases, DetectorBasis::X).is_err());
    // Malformed priors and indices.
    let mut nan = dem.clone();
    nan.mechanisms[0].0 = f64::NAN;
    assert!(gari_transform(&nan, &bases, DetectorBasis::X).is_err());
    let mut big = dem.clone();
    big.mechanisms[0].0 = 1.5;
    assert!(gari_transform(&big, &bases, DetectorBasis::X).is_err());
    let mut oob = dem.clone();
    oob.mechanisms[0].1.push(24);
    assert!(gari_transform(&oob, &bases, DetectorBasis::X).is_err());
    assert!(init_dets_view(&oob, &bases, DetectorBasis::X).is_err());
    assert!(init_dets_view(&nan, &bases, DetectorBasis::X).is_err());
}

#[test]
fn gari_transform_handles_degenerate_shapes() {
    // No mixed columns at all: a pure-X-detector model (eX and eY empty).
    let (dem, bases) = x_toy();
    let x_only = SparseDem {
        mechanisms: dem
            .mechanisms
            .iter()
            .filter(|(_, dets, _)| dets.iter().all(|&d| bases[d as usize] == DetectorBasis::X))
            .cloned()
            .collect(),
        detector_coords: dem.detector_coords.clone(),
        num_detectors: dem.num_detectors,
        num_observables: dem.num_observables,
    };
    let gari = gari_transform(&x_only, &bases, DetectorBasis::X).unwrap();
    assert!(gari.columns(GariColumnBlock::EX).is_empty());
    assert!(gari.columns(GariColumnBlock::EY).is_empty());
    assert!(gari.columns(GariColumnBlock::EbarX).is_empty());
    assert!(gari.u_map.is_empty() && gari.v_map.is_empty());
    assert_eq!(
        gari.dem.num_detectors,
        24 + gari.columns(GariColumnBlock::EZ).len()
    );
    check_gari_equivalence(&gari);

    // Zero and one-half priors survive the transform unchanged.
    let mut edge = dem.clone();
    edge.mechanisms[0].0 = 0.0;
    edge.mechanisms[1].0 = 0.5;
    let gari = gari_transform(&edge, &bases, DetectorBasis::X).unwrap();
    let physical = gari.physical_columns();
    assert!(
        physical
            .mechanisms
            .iter()
            .any(|m| m.0.to_bits() == 0.0_f64.to_bits())
    );
    assert!(
        physical
            .mechanisms
            .iter()
            .any(|m| m.0.to_bits() == 0.5_f64.to_bits())
    );
    check_gari_equivalence(&gari);

    // A detector touched by no mechanism is an all-zero row; still fine.
    let mut orphan = dem.clone();
    orphan.num_detectors += 1;
    let mut orphan_bases = bases.clone();
    orphan_bases.push(DetectorBasis::X);
    let gari = gari_transform(&orphan, &orphan_bases, DetectorBasis::X).unwrap();
    assert_eq!(gari.num_detectors, 25);
    assert_eq!(gari.relevant_rows.len(), 17);
    check_gari_equivalence(&gari);
}

#[test]
fn x_memory_init_dets_views_match_reference_exactly() {
    let (dem, bases) = x_toy();
    check_init_against(
        &dem,
        &bases,
        DetectorBasis::X,
        &json("toy_xz_d3_init_dets_expected.json"),
    );
    check_init_against(
        &dem,
        &bases,
        DetectorBasis::Z,
        &json("toy_xz_d3_zfamily_init_dets_expected.json"),
    );
}

#[test]
fn z_memory_init_dets_view_matches_reference_exactly() {
    let (dem, bases) = z_toy();
    check_init_against(
        &dem,
        &bases,
        DetectorBasis::Z,
        &json("toy_xz_d3_zinit_init_dets_expected.json"),
    );
}

#[test]
fn init_dets_view_with_an_empty_family_is_empty() {
    let (dem, _) = x_toy();
    let all_z = vec![DetectorBasis::Z; dem.num_detectors];
    let view = init_dets_view(&dem, &all_z, DetectorBasis::X).unwrap();
    assert_eq!(view.dem.num_detectors, 0);
    assert_eq!(view.detector_index, Vec::<u32>::new());
    // Only columns that still touch an observable survive the projection.
    assert!(
        view.dem
            .mechanisms
            .iter()
            .all(|(_, dets, obs)| dets.is_empty() && !obs.is_empty())
    );
    assert_eq!(view.project_syndrome(&[0u8; 24]).unwrap(), Vec::<u8>::new());
}
