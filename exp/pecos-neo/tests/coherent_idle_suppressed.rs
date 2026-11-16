//! Keep the Python bridge's coherent idle contract covered by pecos-neo CI.
#[path = "../../../python/pecos-rslib-exp/src/coherent_idle_channel.rs"]
mod coherent_idle_channel;

use coherent_idle_channel::CoherentIdleChannel;
use pecos_core::{Angle64, QubitId};
use pecos_neo::command::GateType;
use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_random::PecosRng;

#[test]
fn coherent_idle_uses_healthy_retained_operands() {
    let channel = CoherentIdleChannel::new(Angle64::QUARTER_TURN);
    let mut ctx = NoiseContext::new();
    // q2 leaked during a child after the parent's execution partition was fixed.
    ctx.mark_leaked(QubitId(0));
    ctx.mark_leaked(QubitId(2));
    let event = NoiseEvent::AfterGate {
        gate_type: GateType::CX,
        gate_id: None,
        qubits: &[QubitId(2), QubitId(3)],
        leaked_suppressed: &[QubitId(0), QubitId(1)],
        group_size: std::num::NonZeroUsize::new(2).unwrap(),
        angles: &[],
    };
    let NoiseResponse::InjectGates(gates) =
        channel.apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42))
    else {
        panic!("healthy retained members must idle");
    };
    assert_eq!(gates.len(), 2);
    assert_eq!(gates[0].qubits.as_slice(), &[QubitId(3)]);
    assert_eq!(gates[1].qubits.as_slice(), &[QubitId(1)]);
    assert!(
        gates
            .iter()
            .all(|g| g.gate_type == GateType::RZ && g.angles() == [Angle64::QUARTER_TURN])
    );
}

#[test]
fn coherent_idle_returns_none_when_all_operands_leak() {
    let channel = CoherentIdleChannel::new(Angle64::QUARTER_TURN);
    let pair = [QubitId(0), QubitId(1)];
    let mut ctx = NoiseContext::new();
    for q in pair {
        ctx.mark_leaked(q);
    }
    for suppressed in [false, true] {
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            gate_id: None,
            qubits: if suppressed { &[] } else { &pair },
            leaked_suppressed: if suppressed { &pair } else { &[] },
            group_size: pecos_neo::noise::gate_group_size(2, pair.len()),
            angles: &[],
        };
        assert!(matches!(
            channel.apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42)),
            NoiseResponse::None
        ));
    }
}
