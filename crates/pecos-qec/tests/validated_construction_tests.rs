//! Circuit-owning analysis APIs reject unsupported gates at construction.
use pecos_core::{Angle64, Gate, QubitId};
use pecos_qec::fault_tolerance::*;
use pecos_quantum::{DagCircuit, GateType, TickCircuit};

macro_rules! constructor_rejections {
    ($($name:ident => |$tick:ident, $dag:ident| $construct:expr),* $(,)?) => { $(
        #[test]
        fn $name() {
            for gate in [Gate::simple(GateType::T, vec![QubitId(0)]), Gate::rz(Angle64::from_radians(0.3), &[0])] {
                let mut $tick = TickCircuit::new();
                $tick.tick().h(&[0]);
                $tick.tick().try_add_gate(gate.clone()).unwrap();
                let mut $dag = DagCircuit::new();
                $dag.h(&[0]);
                $dag.add_gate_auto_wire(gate.clone());
                let error = ($construct).err().expect("constructor must reject unsupported gate");
                assert_eq!(error.gate_type, gate.gate_type);
                assert_eq!(error.angles, gate.angles.to_vec());
                assert_eq!(error.qubits, vec![0]);
                assert_eq!(error.location, if stringify!($name).starts_with("influence") {
                    UnsupportedGateLocation::DagNode { node: 1 }
                } else { UnsupportedGateLocation::Tick { tick: 1, gate_in_tick: 0 } });
            }
        }
    )* };
}
constructor_rejections! {
    pauli_checker_constructor => |tick, dag| PauliPropChecker::new(&tick),
    gadget_constructor => |tick, dag| GadgetChecker::new(&tick, GadgetConfig::new()),
    gadget_auto_constructor => |tick, dag| GadgetChecker::from_circuit(&tick),
    fault_checker_constructor => |tick, dag| FaultChecker::new(&tick),
    correction_checker_constructor => |tick, dag| ErrorCorrectionChecker::new(&tick),
    tick_analyzer_constructor => |tick, dag| TickFaultAnalyzer::new(&tick),
    influence_builder_constructor => |tick, dag| InfluenceBuilder::new(&dag),
}
