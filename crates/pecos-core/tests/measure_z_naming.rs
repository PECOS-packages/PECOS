use pecos_core::{Gate, gate_type::GateType};

#[test]
fn measurement_spellings_preserve_wire_values() {
    for name in [
        "MeasureLeaked",
        "MEASURELEAKED",
        "MeasureZLeaked",
        "MEASUREZLEAKED",
    ] {
        assert_eq!(name.parse::<GateType>(), Ok(GateType::MeasureZLeaked));
    }
    let gate = GateType::MeasureZLeaked;
    assert_eq!(gate as u8, 105);
    assert_eq!(GateType::try_from(105), Ok(gate));
    assert_eq!(gate.to_string(), "MeasureZLeaked");
    assert_eq!(gate.to_string().parse::<GateType>(), Ok(gate));
    for name in ["Measure", "MEASUREZ", "MeasureZ", "MZ", "measure Z"] {
        assert_eq!(name.parse::<GateType>(), Ok(GateType::MZ));
    }
    assert_eq!(GateType::MZ.to_string(), "MZ");
}

#[test]
#[allow(deprecated)]
fn deprecated_measurement_aliases_preserve_values_and_patterns() {
    assert_eq!(GateType::MeasureLeaked, GateType::MeasureZLeaked);
    assert!(matches!(GateType::MeasureZLeaked, GateType::MeasureLeaked));
    assert!(matches!(GateType::MeasureLeaked, GateType::MeasureZLeaked));
    assert_eq!(
        Gate::measure_leaked(&[0, 3]),
        Gate::measure_z_leaked(&[0, 3])
    );
}
