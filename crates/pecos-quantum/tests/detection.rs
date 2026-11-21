//! Detection preserves outcome order and acts as an optimization barrier.
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, MeasId};
use pecos_quantum::pass::{
    AbsorbBasisGates, CancelInverses, CircuitPass, MergeAdjacentRotations, PeepholeOptimize,
    SimplifyRotations, SimplifySingleQubitCliffordChains, StripIdentities,
};
use pecos_quantum::{DagCircuit, TickCircuit};

const FLAVORS: [GateType; 3] = [
    GateType::DetectLeaked,
    GateType::DetectLost,
    GateType::DetectLeakedLost,
];
fn gates(tc: &TickCircuit) -> Vec<Gate> {
    tc.iter_gate_batches()
        .map(|g| g.as_gate().clone())
        .collect()
}
#[test]
fn detect_survives_each_optimization_on_tick_and_dag() {
    let passes: Vec<Box<dyn CircuitPass>> = vec![
        Box::new(AbsorbBasisGates),
        Box::new(MergeAdjacentRotations),
        Box::new(StripIdentities),
        Box::new(SimplifyRotations),
        Box::new(CancelInverses),
        Box::new(PeepholeOptimize),
        Box::new(SimplifySingleQubitCliffordChains),
    ];
    for flavor in FLAVORS {
        for explicit in [false, true] {
            for pass in &passes {
                let mut tc = TickCircuit::new();
                tc.tick().h(&[0, 1]);
                let mut detect = Gate::simple(flavor, vec![0.into(), 1.into()]);
                if explicit {
                    detect
                        .meas_ids
                        .extend([MeasId::from_raw(9), MeasId::from_raw(3)]);
                }
                tc.tick().try_add_gate(detect.clone()).unwrap();
                tc.tick().h(&[0, 1]);
                let mut dag = DagCircuit::new();
                for gate in gates(&tc) {
                    dag.add_gate_auto_wire(gate);
                }
                pass.apply_tick(&mut tc);
                pass.apply_dag(&mut dag);
                for output in [&tc, &TickCircuit::from(&dag)] {
                    let sequence = gates(output);
                    assert_eq!(
                        sequence
                            .iter()
                            .filter(|g| g.gate_type.is_detection())
                            .collect::<Vec<_>>(),
                        [&detect]
                    );
                    for qubit in [0.into(), 1.into()] {
                        assert_eq!(
                            sequence
                                .iter()
                                .filter(|g| g.qubits.contains(&qubit))
                                .map(|g| g.gate_type)
                                .collect::<Vec<_>>(),
                            [GateType::H, flavor, GateType::H]
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn dag_emission_barrier_includes_heralds_and_detection_without_minting_records() {
    for flavor in FLAVORS.into_iter().chain([
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ]) {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);
        dag.add_gate_auto_wire(Gate::simple(flavor, vec![1.into()]));
        let tc = TickCircuit::from(&dag);
        let outcomes: Vec<_> = gates(&tc)
            .into_iter()
            .filter(|g| g.gate_type == GateType::MZ || g.gate_type == flavor)
            .collect();
        assert_eq!(
            outcomes.iter().map(|g| g.gate_type).collect::<Vec<_>>(),
            [GateType::MZ, flavor]
        );
        assert!(outcomes[1].meas_ids.is_empty());
    }
}
#[test]
fn detection_ticks_have_no_idle_duration() {
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0, 1]);
    for flavor in FLAVORS {
        tc.tick()
            .try_add_gate(Gate::simple(flavor, vec![0.into()]))
            .unwrap();
    }
    tc.tick().h(&[0]);
    tc.fill_idle_gates();
    for tick in tc.ticks().iter().skip(1).take(3) {
        assert_eq!(tick.gate_batches().len(), 1);
        assert!(tick.gate_batches()[0].gate_type.is_detection());
    }
    assert!(
        tc.ticks()
            .last()
            .unwrap()
            .gate_batches()
            .iter()
            .any(|g| g.gate_type == GateType::Idle)
    );
}
#[test]
#[should_panic(expected = "emit detect gates in their own tick")]
fn mixed_detection_physical_tick_rejects_idle_filling() {
    let mut tc = TickCircuit::new();
    tc.tick()
        .try_add_gate(Gate::detect_lost(&[0]))
        .unwrap()
        .h(&[1]);
    tc.fill_idle_gates();
}

#[test]
fn detection_with_zero_duration_metadata_is_counted_once() {
    use pecos_quantum::Attribute;
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0, 1]);
    let idx = tc.tick().index();
    let tick = tc.get_tick_mut(idx).unwrap();
    let batch = tick.try_add_gate(Gate::detect_lost(&[0])).unwrap();
    tick.set_gate_attr(batch, "_physical_duration", Attribute::Int(0));
    tc.fill_idle_gates();
    assert_eq!(tc.ticks()[idx].gate_batches().len(), 1);
}

#[test]
fn detection_with_metadata_and_zero_duration_batches_has_no_idle() {
    use pecos_quantum::Attribute;
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0, 1, 2, 3]);
    let idx = tc.tick().index();
    let tick = tc.get_tick_mut(idx).unwrap();
    tick.try_add_gate(Gate::detect_lost(&[0])).unwrap();
    let meta = tick
        .try_add_gate(Gate::simple(GateType::TrackedPauliMeta, vec![1.into()]))
        .unwrap();
    tick.set_gate_attr(meta, "_physical_duration", Attribute::Int(0));
    let zero = tick.try_add_gate(Gate::z(&[2])).unwrap();
    tick.set_gate_attr(zero, "_physical_duration", Attribute::Int(0));
    tc.fill_idle_gates();
    assert_eq!(tc.ticks()[idx].gate_batches().len(), 3);
}
