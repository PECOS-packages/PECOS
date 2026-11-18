# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic QEC abstractions.

Code-agnostic utilities for stabilizer checks, scheduling,
and Pauli operators that work across different QEC code families.
"""

from pecos.qec.generic.check import (
    CheckSchedule,
    PauliOperator,
    PauliType,
    StabilizerCheck,
)
from pecos.qec.generic.fault_explanations import FaultSignatureExplanation
from pecos.qec.generic.gadget_flows import GadgetFlowAudit, PauliFlow
from pecos.qec.generic.gadgets import Gadget, GadgetCall, GadgetFlow, GadgetProgram, Gate, NamedParity, NamedPauli
from pecos.qec.generic.guppy_audit import GuppyImplementationAudit
from pecos.qec.generic.guppy_comparison import GuppyScheduleComparison
from pecos.qec.generic.guppy_lifetimes import bind_runtime_lifetime_windows
from pecos.qec.generic.guppy_ordering import GuppyFence, GuppyLifetimeFence, GuppyOrdering, GuppyOrderingPlan
from pecos.qec.generic.guppy_schedule import GuppyScheduleAudit, summarize_runtime_batches
from pecos.qec.generic.native_noise import NativeScheduledNoise
from pecos.qec.generic.noise_comparison import ScheduledNoiseComparison
from pecos.qec.generic.parity_audit import ScheduledParityAudit
from pecos.qec.generic.scheduled import (
    CircuitOp,
    MeasurementParity,
    OperationOrigin,
    ScheduledCircuit,
    SchedulePolicy,
    ScheduleResources,
    ScheduleRow,
)
from pecos.qec.generic.scheduled_faults import (
    FaultDistanceBounds,
    LogicalFaultWitness,
    PauliFault,
    fault_distance_bounds,
    remap_operation_faults,
    verify_logical_fault,
)
from pecos.qec.generic.scheduled_noise import SI1000, NoiseChannel, NoisyScheduledCircuit
from pecos.qec.generic.tick_lowering import evaluate_tick_circuit

__all__ = [
    "SI1000",
    "CheckSchedule",
    "CircuitOp",
    "FaultDistanceBounds",
    "FaultSignatureExplanation",
    "Gadget",
    "GadgetCall",
    "GadgetFlow",
    "GadgetFlowAudit",
    "GadgetProgram",
    "Gate",
    "GuppyFence",
    "GuppyImplementationAudit",
    "GuppyLifetimeFence",
    "GuppyOrdering",
    "GuppyOrderingPlan",
    "GuppyScheduleAudit",
    "GuppyScheduleComparison",
    "LogicalFaultWitness",
    "MeasurementParity",
    "NamedParity",
    "NamedPauli",
    "NativeScheduledNoise",
    "NoiseChannel",
    "NoisyScheduledCircuit",
    "OperationOrigin",
    "PauliFault",
    "PauliFlow",
    "PauliOperator",
    "PauliType",
    "SchedulePolicy",
    "ScheduleResources",
    "ScheduleRow",
    "ScheduledCircuit",
    "ScheduledNoiseComparison",
    "ScheduledParityAudit",
    "StabilizerCheck",
    "bind_runtime_lifetime_windows",
    "evaluate_tick_circuit",
    "fault_distance_bounds",
    "remap_operation_faults",
    "summarize_runtime_batches",
    "verify_logical_fault",
]
