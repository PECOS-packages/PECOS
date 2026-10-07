// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Greedy width scheduling and sampling-plan selection.

use super::passes::body;
use super::plan::Instruction;
use super::{HeisenbergOp, HeisenbergProgram, PlanError, SamplingPlan, replay_width_operation};
use crate::ActiveStructure;
use pecos_stab_tn::stab_mps::coordinate_tableau::MeasurementCase;

/// Reorder commuting operations greedily to reduce active width.
/// Preserves the joint distribution of measurement-symbol outcomes and records,
/// but can change per-seed samples. Never raises peak width; it is deliberately
/// not an optimal scheduler. Ties within each priority use original order.
///
/// # Panics
/// Panics if a pass bug produces an invalid program.
#[must_use]
pub fn schedule_for_width(program: HeisenbergProgram) -> HeisenbergProgram {
    let source = program.operations();
    let mut successors = vec![Vec::new(); source.len()];
    let mut indegrees = vec![0; source.len()];
    // O(m^2) pair checks build the DAG. The ready set is maintained incrementally
    // through indegrees. Classification adds O(|ready| * decompose) per step,
    // followed by one shared structural replay of the chosen operation.
    for (later, b) in source.iter().enumerate() {
        let (HeisenbergOp::Rotation { sign, .. } | HeisenbergOp::Measurement { sign, .. }) = b;
        for (earlier, a) in source[..later].iter().enumerate() {
            let r1 = !body(a).bits().commutes_with(body(b).bits());
            let r2 = matches!(a, HeisenbergOp::Measurement { symbol, .. }
                if sign.measurements.contains(symbol));
            if r1 || r2 {
                successors[earlier].push(later);
                indegrees[later] += 1;
            }
        }
    }
    let mut ready: Vec<_> = (0..source.len()).filter(|&i| indegrees[i] == 0).collect();
    let mut structure = ActiveStructure::with_seed(program.num_qubits(), 0);
    let mut operations = Vec::with_capacity(source.len());
    // Exchange argument: when only promoting rotations are ready, the earliest
    // remaining operation is ready. Choosing it lets non-growing operations move
    // earlier without raising peak width, including kept quarter turns. This
    // needs the lowest-index tie-break only within priority 3; any deterministic
    // tie-break within priorities 1 and 2 preserves the bound. Keep original
    // order in all three anyway: refining priorities needs a benchmark.
    while !ready.is_empty() {
        let position = (0..ready.len())
            .min_by_key(|&position| {
                let index = ready[position];
                (priority(&mut structure, &source[index]), index)
            })
            .expect("nonempty ready set");
        let index = ready.swap_remove(position);
        replay_width_operation(&mut structure, &source[index]);
        operations.push(source[index].clone());
        for &later in &successors[index] {
            indegrees[later] -= 1;
            if indegrees[later] == 0 {
                ready.push(later);
            }
        }
    }
    assert_eq!(
        operations.len(),
        source.len(),
        "scheduler DAG cycle: pass bug"
    );
    program
        .with_operations(operations)
        .expect("schedule_for_width produced an invalid program: pass bug")
}

fn priority(structure: &mut ActiveStructure, operation: &HeisenbergOp) -> u8 {
    match operation {
        HeisenbergOp::Rotation { pauli, angle, .. } => {
            if structure
                .rotation(*angle, pauli.factors(), false)
                .needs_promotion()
            {
                3
            } else {
                2
            }
        }
        HeisenbergOp::Measurement { pauli, .. } => {
            if structure.measurement(pauli.factors(), false).data().case()
                == MeasurementCase::Active
            {
                1
            } else {
                2
            }
        }
    }
}

/// Plan the given and greedily scheduled orders and return the better candidate.
/// Compare feasibility, peak width, then amplitude work; keep the given order on
/// ties. Work charges non-Clifford rotations at their post-operation width and
/// Active measurements at their pre-operation width, using exact `u128` sums.
/// Reordering preserves joint distributions but can change per-seed samples.
///
/// # Errors
/// Returns the given program's error if both exceed the width limit. Other
/// planning errors are not infeasibility: propagate the given candidate's error.
///
/// # Panics
/// A non-width planning error on the scheduled candidate only is a pass bug and
/// panics, as does an invalid scheduled program.
pub fn plan_scheduled(
    program: &HeisenbergProgram,
    max_width: usize,
) -> Result<SamplingPlan, PlanError> {
    let given = program.plan(max_width);
    let scheduled = schedule_for_width(program.clone()).plan(max_width);
    select_plan(given, scheduled)
}

pub(super) fn select_plan(
    given: Result<SamplingPlan, PlanError>,
    scheduled: Result<SamplingPlan, PlanError>,
) -> Result<SamplingPlan, PlanError> {
    match (given, scheduled) {
        (Err(error), _) if !matches!(error, PlanError::WidthExceeded { .. }) => Err(error),
        (_, Err(error)) if !matches!(error, PlanError::WidthExceeded { .. }) => {
            panic!("scheduled candidate planning failed: pass bug: {error}")
        }
        (Ok(given), Ok(scheduled)) => {
            if plan_cost(&scheduled) < plan_cost(&given) {
                Ok(scheduled)
            } else {
                Ok(given)
            }
        }
        (Ok(plan), Err(_)) | (Err(_), Ok(plan)) => Ok(plan),
        (Err(given), Err(_)) => Err(given),
    }
}

pub(super) fn plan_cost(plan: &SamplingPlan) -> (usize, u128) {
    let mut before = 0;
    let mut work = 0_u128;
    for (operation, &after) in plan.operations.iter().zip(plan.width_profile()) {
        let width = match operation.instruction {
            Instruction::Rotation { .. } => Some(after),
            Instruction::Measurement {
                case: MeasurementCase::Active,
                ..
            } => Some(before),
            _ => None,
        };
        if let Some(width) = width {
            // A valid plan's width is bounded by complex-vector byte addressing.
            // On a 64-bit platform each term is <= 2^59 and there are fewer than
            // 2^64 instructions, so u128 is exact. Checked arithmetic guards this
            // invariant on future platforms; never saturate into a false tie.
            let term = 1_u128
                .checked_shl(u32::try_from(width).expect("plan width fits u32"))
                .expect("plan work term fits u128");
            work = work.checked_add(term).expect("plan work sum fits u128");
        }
        before = after;
    }
    (
        plan.width_profile().iter().copied().max().unwrap_or(0),
        work,
    )
}
