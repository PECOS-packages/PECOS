// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_qec::fault_tolerance::dem_builder::ParsedDem;

#[test]
fn parsed_dem_reports_index_overflow() {
    let error = ParsedDem::parse("error(0.1) D4294967296").unwrap_err();
    assert!(matches!(
        error,
        pecos_qec::fault_tolerance::dem_builder::DemParseError::UnsupportedIndex(_)
    ));
    assert!(
        error
            .to_string()
            .contains("detector index 4294967296 exceeds the supported maximum 4294967295")
    );
}

#[test]
fn pecos_metadata_comments_preserve_hashes_in_json_strings() {
    let parsed = ParsedDem::parse("error[tag](0.1) d0 TP0 # ignored\npecos_tracked_pauli {\"id\":0,\"label\":\"a#b\"} # comment").unwrap();
    assert_eq!(parsed.mechanisms.len(), 1);
    assert_eq!(
        parsed.tracked_paulis[0].as_ref().unwrap().label.as_deref(),
        Some("a#b")
    );
}

#[test]
fn parsed_dem_accepts_tracked_paulis_and_metadata() {
    let parsed = ParsedDem::parse(
        "error(0.1) D0 L0 TP1\npecos_observable {\"id\":0,\"label\":\"logical\"}\npecos_tracked_pauli {\"id\":1,\"label\":\"tracked\"}",
    ).unwrap();
    assert_eq!(parsed.num_detectors, 1);
    assert_eq!(parsed.num_observables(), 1);
    assert_eq!(parsed.tracked_paulis.len(), 2);
    assert_eq!(
        parsed.dem_outputs[0].as_ref().unwrap().label.as_deref(),
        Some("logical")
    );
    assert_eq!(
        parsed.tracked_paulis[1].as_ref().unwrap().label.as_deref(),
        Some("tracked")
    );
}
