//! Independent oracle parity for graphlike projection; fixture files are immutable.

use pecos_decoder_core::graphlike_projection::{
    GraphlikeProjectionError, IrreduciblePolicy, project_graphlike,
};
use pecos_decoder_core::window::StructuredDem;
use std::collections::BTreeMap;
use std::path::Path;

fn read(path: &Path, extension: &str) -> String {
    std::fs::read_to_string(path.with_extension(extension)).unwrap()
}

fn check_fixture(path: &Path) {
    let mut dem = StructuredDem::from_dem_str(&read(path, "dem")).unwrap();
    dem.detector_fault_sectors = read(path, "sectors")
        .split_whitespace()
        .map(|token| (token != "-").then(|| token.to_owned()))
        .collect();
    let expected: BTreeMap<usize, _> = read(path, "expected")
        .lines()
        .map(|line| {
            let (index, components) = line.split_once('\t').unwrap();
            let components = (components != "drop").then(|| {
                StructuredDem::from_dem_str(&format!("error(0.1) {components}"))
                    .unwrap()
                    .errors
                    .remove(0)
                    .components
            });
            (index.parse().unwrap(), components)
        })
        .collect();
    assert_eq!(
        expected.len(),
        dem.errors
            .iter()
            .filter(|error| error.components[0].detectors.len() > 2)
            .count()
    );
    let result = project_graphlike(&dem, IrreduciblePolicy::Drop).unwrap();
    assert_eq!(result.report.input_hyperedges, expected.len());
    assert_eq!(
        result.report.projected,
        expected
            .values()
            .filter(|components| components.is_some())
            .count()
    );
    let mut output = result.model.errors.iter();
    let mut dropped = Vec::new();
    for (index, input) in dem.errors.iter().enumerate() {
        let components = match expected.get(&index) {
            Some(None) => {
                dropped.push(index);
                continue;
            }
            Some(Some(components)) => components,
            None => &input.components,
        };
        let actual = output.next().unwrap();
        assert_eq!(
            &actual.components,
            components,
            "{} error {index}",
            path.display()
        );
        assert_eq!(actual.probability.to_bits(), input.probability.to_bits());
    }
    assert!(output.next().is_none());
    assert_eq!(result.report.dropped_indices, dropped);
    assert_eq!(result.model.detector_coords, dem.detector_coords);
    assert_eq!(result.model.num_detectors, dem.num_detectors);
    assert_eq!(result.model.num_observables, dem.num_observables);
    // contract: projection preserves declarations; Stim text preserves only DEM fields.
    assert_eq!(
        result.model.detector_fault_sectors,
        dem.detector_fault_sectors
    );
    let reparsed = StructuredDem::from_dem_str(&result.model.to_dem_string()).unwrap();
    assert_eq!(reparsed.errors, result.model.errors);
    assert_eq!(reparsed.detector_coords, result.model.detector_coords);
    assert_eq!(reparsed.num_detectors, result.model.num_detectors);
    assert_eq!(reparsed.num_observables, result.model.num_observables);
    assert_eq!(
        reparsed.detector_fault_sectors,
        vec![None; dem.num_detectors]
    );
    let refusal = project_graphlike(&dem, IrreduciblePolicy::Refuse);
    match read(path, "refuse.txt").trim() {
        "ok" => assert_eq!(refusal.unwrap(), result),
        "irreducible" => assert_eq!(
            refusal.unwrap_err(),
            GraphlikeProjectionError::Irreducible {
                count: dropped.len(),
                first_indices: dropped.into_iter().take(5).collect(),
            }
        ),
        other => panic!("unknown refusal {other}"),
    }
}

// fixture: discover every oracle case and compare parsed components, probability bits, and round trips.
#[test]
fn every_oracle_fixture() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/graphlike_v4");
    let mut cases: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "dem"))
        .collect();
    cases.sort();
    // The README lists six cases; a missing fixture must fail, not shrink the test.
    assert_eq!(cases.len(), 6, "{cases:?}");
    for path in cases {
        check_fixture(&path);
    }
}
