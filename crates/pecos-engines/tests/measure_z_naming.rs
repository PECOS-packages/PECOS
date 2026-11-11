use pecos_engines::{ByteMessage, Gate, GateType};

#[test]
#[allow(deprecated)]
fn deprecated_measurement_builder_preserves_bytes() {
    let mut old = ByteMessage::quantum_operations_builder();
    old.measure_leakages(&[0, 3]);
    let mut new = ByteMessage::quantum_operations_builder();
    new.measure_z_leaked(&[0, 3]);
    let old = old.build();
    let new = new.build();
    assert_eq!(old.as_bytes(), new.as_bytes());
    assert_eq!(
        new.quantum_ops().unwrap(),
        [Gate::measure_z_leaked(&[0]), Gate::measure_z_leaked(&[3])]
    );
    assert_eq!(
        new.quantum_ops().unwrap()[0].gate_type,
        GateType::MeasureZLeaked
    );
}
