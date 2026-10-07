// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Independent peephole passes on signed virtual Pauli operations.

use super::{HeisenbergOp, HeisenbergProgram, VirtualPauli};
use crate::structure::clifford_turns;
use pecos_core::Angle64;

#[derive(Clone, Copy, Debug)]
pub(crate) enum FusionEvent {
    Fused,
    ZeroCancelled,
    HalfTurnRemoved,
    QuarterTurnResult,
    NonCliffordResult,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RemovalEvent {
    MeasurementBefore,
    MeasurementAfter,
}

fn body(operation: &HeisenbergOp) -> &VirtualPauli {
    match operation {
        HeisenbergOp::Rotation { pauli, .. } | HeisenbergOp::Measurement { pauli, .. } => pauli,
    }
}

/// Fuse equal-body rotations with equal symbolic sign sets across commuting operations.
/// Preserve the joint distribution of all measurement-symbol outcomes and records,
/// up to global phase; final-state equality is not preserved. Widths and per-seed
/// samples may change.
///
/// # Panics
/// Panics only if a pass bug produces an invalid program.
#[must_use]
pub fn fuse_rotations(program: HeisenbergProgram) -> HeisenbergProgram {
    fuse_rotations_observed(program, |_, _| {})
}

fn first_fusion(operations: &[HeisenbergOp], start: usize) -> Option<(usize, usize, Angle64)> {
    for (a, operation) in operations.iter().enumerate().skip(start) {
        let HeisenbergOp::Rotation { pauli, angle, sign } = operation else {
            continue;
        };
        for (b, other) in operations.iter().enumerate().skip(a + 1) {
            if !pauli.bits().commutes_with(body(other).bits()) {
                break;
            }
            if let HeisenbergOp::Rotation {
                pauli: other_pauli,
                angle: other_angle,
                sign: other_sign,
            } = other
                && pauli.bits() == other_pauli.bits()
                && sign.noise == other_sign.noise
                && sign.measurements == other_sign.measurements
            {
                // Equal symbol sets mean B reads only symbols already read
                // by valid A, hence produced before A. R2 needs no check.
                let fused = if sign.constant ^ other_sign.constant {
                    *angle - *other_angle
                } else {
                    *angle + *other_angle
                };
                return Some((a, b, fused));
            }
        }
    }
    None
}

// Scan A left to right, then B left to right. If A stays, resume at A: any
// earlier scan blocked by B is still blocked by A, which has the same body.
// If A is removed, restart at zero because earlier pairs may be unblocked.
// Constants never decide which pair fuses, only the fused angle. A half-turn
// removal can unblock a pair straddling A and toggle only its later member,
// so finish the toggles before the next scan.
// The second observer argument counts toggled constants (zero for other events).
pub(crate) fn fuse_rotations_observed(
    mut program: HeisenbergProgram,
    mut observe: impl FnMut(FusionEvent, usize),
) -> HeisenbergProgram {
    let mut operations = std::mem::take(&mut program.operations);
    let mut start = 0;
    while let Some((a, b, fused)) = first_fusion(&operations, start) {
        operations.remove(b);
        observe(FusionEvent::Fused, 0);
        start = 0;
        match clifford_turns(fused) {
            Some(0) => {
                operations.remove(a);
                observe(FusionEvent::ZeroCancelled, 0);
            }
            Some(2) => {
                let removed = operations.remove(a);
                let mut toggles = 0;
                // All operations between A and B commute with this body, so
                // absorbing the half turn at either position has the same effect.
                for later in &mut operations[a..] {
                    if !body(&removed).bits().commutes_with(body(later).bits()) {
                        let (HeisenbergOp::Rotation { sign, .. }
                        | HeisenbergOp::Measurement { sign, .. }) = later;
                        sign.constant ^= true;
                        toggles += 1;
                    }
                }
                observe(FusionEvent::HalfTurnRemoved, toggles);
            }
            turns => {
                let HeisenbergOp::Rotation { angle, .. } = &mut operations[a] else {
                    unreachable!("fusion starts at a rotation");
                };
                *angle = fused;
                start = a;
                observe(
                    if turns.is_some() {
                        FusionEvent::QuarterTurnResult
                    } else {
                        FusionEvent::NonCliffordResult
                    },
                    0,
                );
            }
        }
    }
    program
        .with_operations(operations)
        .expect("fuse_rotations produced an invalid program: pass bug")
}

/// Delete rotations separated from a same-body measurement only by commuting operations.
/// Signs do not restrict deletion. Preserve the joint distribution of all
/// measurement-symbol outcomes and records, up to global phase; final-state
/// equality is not preserved. Widths and per-seed samples may change.
///
/// # Panics
/// Panics only if a pass bug produces an invalid program.
#[must_use]
pub fn drop_measured_rotations(program: HeisenbergProgram) -> HeisenbergProgram {
    drop_measured_rotations_observed(program, |_| {})
}

fn reaches_measurement<'a>(
    pauli: &VirtualPauli,
    operations: impl Iterator<Item = &'a HeisenbergOp>,
) -> bool {
    for operation in operations {
        if !pauli.bits().commutes_with(body(operation).bits()) {
            return false;
        }
        if matches!(operation, HeisenbergOp::Measurement { .. })
            && pauli.bits() == body(operation).bits()
        {
            return true;
        }
    }
    false
}

pub(crate) fn drop_measured_rotations_observed(
    mut program: HeisenbergProgram,
    mut observe: impl FnMut(RemovalEvent),
) -> HeisenbergProgram {
    let mut operations = std::mem::take(&mut program.operations);
    let mut i = 0;
    // Deterministic left-to-right scan, preferring a measurement before R.
    // This is already a fixpoint. A later removable blocker B, with body H,
    // blocks R with body K. B's same-body measurement cannot precede R or
    // follow R's target measurement M_K: R and M_K anticommute with H and
    // would block B. It must therefore lie between R and M_K, where it
    // anticommutes with K and itself blocks R. Measurements are never deleted.
    // Earlier removable blockers are removed before R is examined in this scan.
    while i < operations.len() {
        if let HeisenbergOp::Rotation { pauli, .. } = &operations[i] {
            // After M measures (-1)^s H, the branch lies in an H-eigenspace.
            // Commuting intervening operations preserve that eigenspace, so R
            // is the scalar cos(theta/2) - i (-1)^s_R lambda sin(theta/2).
            // This holds even if s_R reads intervening measurement symbols:
            // R is deleted, not moved, and R2 never arises. Before a later M,
            // R commutes through the intervening operations into its projector.
            let event = if reaches_measurement(pauli, operations[..i].iter().rev()) {
                Some(RemovalEvent::MeasurementBefore)
            } else if reaches_measurement(pauli, operations[i + 1..].iter()) {
                Some(RemovalEvent::MeasurementAfter)
            } else {
                None
            };
            if let Some(event) = event {
                operations.remove(i);
                observe(event);
                continue;
            }
        }
        i += 1;
    }
    program
        .with_operations(operations)
        .expect("drop_measured_rotations produced an invalid program: pass bug")
}
