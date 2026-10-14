//! Independent oracle parity for graphlike projection; fixture files are immutable.

use pecos_decoder_core::graphlike_projection::{
    DetectorFamily, GraphlikeProjectionError, IrreduciblePolicy, project_graphlike,
};
use pecos_decoder_core::window::StructuredDem;
use std::collections::BTreeMap;
use std::path::Path;

fn read(path: &Path, extension: &str) -> String {
    std::fs::read_to_string(path.with_extension(extension)).unwrap()
}

fn check_fixture(path: &Path) {
    let dem = StructuredDem::from_dem_str(&read(path, "dem")).unwrap();
    let families: Vec<_> = read(path, "families")
        .split_whitespace()
        .map(|token| match token {
            "X" => Some(DetectorFamily::X),
            "Z" => Some(DetectorFamily::Z),
            "M" => Some(DetectorFamily::Mixed),
            "-" => None,
            other => panic!("unknown family {other}"),
        })
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
    let result = project_graphlike(&dem, &families, IrreduciblePolicy::Drop).unwrap();
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
    assert_eq!(
        StructuredDem::from_dem_str(&result.model.to_dem_string()).unwrap(),
        result.model
    );
    let refusal = project_graphlike(&dem, &families, IrreduciblePolicy::Refuse);
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
    // The README lists five cases; a missing fixture must fail, not shrink the test.
    assert_eq!(cases.len(), 5, "{cases:?}");
    for path in cases {
        check_fixture(&path);
    }
}
