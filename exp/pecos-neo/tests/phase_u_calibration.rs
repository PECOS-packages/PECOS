use pecos_core::{Angle64, QubitId};
use pecos_neo::noise::{
    gate_id_dependent::GateIdDependentChannel, per_gate_pauli::PerGatePauliChannel,
};
use pecos_neo::prelude::*;
use pecos_random::PecosRng;

#[test]
fn phase_u_scheduled_calibration_precedes_inherited_exemption_in_all_channels() {
    let channels: Vec<Box<dyn NoiseChannel>> = vec![
        Box::new(GateDependentChannel::new().with_gate_error(GateType::U, 1.0)),
        Box::new(GateIdDependentChannel::new().with_gate_type_error(GateType::U, 1.0)),
        Box::new(PerGatePauliChannel::new().with_1q_rates(GateType::U, [1.0, 0.0, 0.0])),
        Box::new(PerGatePauliChannel::new().with_1q_rates_for_qubit(
            GateType::U,
            QubitId(0),
            [1.0, 0.0, 0.0],
        )),
    ];
    let qubits = [QubitId(0)];
    let angles = [Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN];
    let event = NoiseEvent::AfterGate {
        gate_type: GateType::U,
        qubits: &qubits,
        angles: &angles,
        gate_id: Some(GateType::U.to_gate_id()),
    };
    let mut failures = Vec::new();
    for channel in channels {
        let mut ctx = NoiseContext::new();
        ctx.add_noiseless_gate(GateType::RZ);
        let mut rng = PecosRng::seed_from_u64(42);
        if !matches!(
            channel.apply(&event, &mut ctx, &mut rng),
            NoiseResponse::InjectGates(_)
        ) {
            failures.push(channel.name());
        }
        ctx.add_noiseless_gate(GateType::U);
        assert!(
            matches!(
                channel.apply(&event, &mut ctx, &mut rng),
                NoiseResponse::None
            ),
            "explicit scheduled exemption must win for {}",
            channel.name()
        );
    }
    assert!(
        failures.is_empty(),
        "channels ignored scheduled U calibration: {failures:?}"
    );
}
