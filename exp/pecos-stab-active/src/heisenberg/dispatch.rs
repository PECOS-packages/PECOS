// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use crate::PauliKindForDecomp;
use pecos_core::{QubitId, gate_type::GateType};
use pecos_simulators::CliffordGateable;

pub(super) fn rotation_axis(gate: GateType) -> Option<PauliKindForDecomp> {
    match gate {
        GateType::RX | GateType::RXX => Some(PauliKindForDecomp::X),
        GateType::RY | GateType::RYY => Some(PauliKindForDecomp::Y),
        GateType::RZ | GateType::RZZ => Some(PauliKindForDecomp::Z),
        _ => None,
    }
}

pub(super) fn named_clifford(gate: GateType) -> bool {
    matches!(
        gate,
        GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
    )
}

pub(super) fn apply_clifford<S: CliffordGateable>(sim: &mut S, gate: GateType, qs: &[QubitId]) {
    let pairs: Vec<_> = qs.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
    match gate {
        GateType::H => {
            sim.h(qs);
        }
        GateType::F => {
            sim.f(qs);
        }
        GateType::Fdg => {
            sim.fdg(qs);
        }
        GateType::X => {
            sim.x(qs);
        }
        GateType::Y => {
            sim.y(qs);
        }
        GateType::Z => {
            sim.z(qs);
        }
        GateType::SX => {
            sim.sx(qs);
        }
        GateType::SXdg => {
            sim.sxdg(qs);
        }
        GateType::SY => {
            sim.sy(qs);
        }
        GateType::SYdg => {
            sim.sydg(qs);
        }
        GateType::SZ => {
            sim.sz(qs);
        }
        GateType::SZdg => {
            sim.szdg(qs);
        }
        GateType::CX => {
            sim.cx(&pairs);
        }
        GateType::CY => {
            sim.cy(&pairs);
        }
        GateType::CZ => {
            sim.cz(&pairs);
        }
        GateType::SXX => {
            sim.sxx(&pairs);
        }
        GateType::SXXdg => {
            sim.sxxdg(&pairs);
        }
        GateType::SYY => {
            sim.syy(&pairs);
        }
        GateType::SYYdg => {
            sim.syydg(&pairs);
        }
        GateType::SZZ => {
            sim.szz(&pairs);
        }
        GateType::SZZdg => {
            sim.szzdg(&pairs);
        }
        GateType::SWAP => {
            sim.swap(&pairs);
        }
        _ => unreachable!("validated named Clifford"),
    }
}
