# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Native execution and fault propagation for explicit scheduled Pauli channels.

The schedule and SI1000 rates come from ``scheduled_noise``. Python selects
mutually exclusive noise branches; PECOS SparseStab executes the trajectory.
PECOS DagFaultAnalyzer independently propagates faults to measurement records.
No Stim import, simulator, or DEM conversion is used by this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from math import expm1, log1p
from random import Random
from typing import TYPE_CHECKING

from pecos.qec.generic.scheduled_noise import NoiseChannel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pecos_rslib.qec import DetectorErrorModel

    from pecos.qec.generic.fault_explanations import FaultSignatureExplanation
    from pecos.qec.generic.scheduled import CircuitOp
    from pecos.qec.generic.scheduled_faults import PauliFault
    from pecos.qec.generic.scheduled_noise import NoisyScheduledCircuit

_SYMBOLS = {"R": "PZ", "M": "MZ", "S": "SZ", "S_DAG": "SZdg", "C_XYZ": "F", "C_ZYX": "Fdg", "SQRT_X": "SX"}


def _branches(channel: NoiseChannel) -> tuple[str, ...]:
    if channel.kind == "X_ERROR":
        return ("X",)
    return tuple("".join(p) for p in product("IXYZ", repeat=len(channel.qubits)) if any(x != "I" for x in p))


def _independent_branch_probability(channel: NoiseChannel) -> float:
    """Convert one categorical channel to equal independent nonidentity factors."""
    if channel.kind == "X_ERROR":
        return channel.probability
    size = 4 ** len(channel.qubits)
    return -expm1(log1p(-channel.probability * size / (size - 1)) / (size / 2)) / 2


@dataclass(frozen=True)
class NativeScheduledNoise:
    """A selected schedule executed through PECOS native Clifford kernels.

    This adapter retains each independent channel and its physical moment.
    Its internal identity markers locate faults in an analysis DAG only: they
    are neither hardware gates nor extra ticks. Use ``noisy.ideal`` for ideal
    TickCircuit/Guppy exports. Passing those exports to a generic runtime does
    not implicitly apply this adapter's noise.
    """

    noisy: NoisyScheduledCircuit
    _instructions: tuple[CircuitOp | NoiseChannel, ...] = field(init=False, repr=False)
    _branches: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Snapshot the instruction order without changing the physical schedule."""
        noisy = self.noisy
        at = defaultdict(list)
        for channel in noisy.channels:
            at[channel.moment, channel.operation, channel.placement].append(channel)
        instructions = []
        for m, moment in enumerate(noisy.ideal.moments):
            for j, op in enumerate(moment):
                instructions.extend(at[m, j, "before"])
                instructions.append(op)
                instructions.extend(at[m, j, "after"])
            instructions.extend(at[m, None, "end"])
        object.__setattr__(self, "_instructions", tuple(instructions))
        object.__setattr__(self, "_branches", {c.id: _branches(c) for c in noisy.channels})

    def _record(self, rng: Random, faults: dict[str, str] | None) -> tuple[bool, ...]:
        from pecos_rslib import SparseStab  # noqa: PLC0415

        sim = SparseStab(self.noisy.ideal.num_qubits)
        sim.set_seed(rng.getrandbits(64))
        record = []
        for inst in self._instructions:
            if isinstance(inst, NoiseChannel):
                word = (
                    faults.get(inst.id)
                    if faults is not None
                    else (rng.choice(self._branches[inst.id]) if rng.random() < inst.probability else None)
                )
                if word is not None:
                    for q, p in zip(inst.qubits, word, strict=True):
                        if p != "I":
                            sim.run_1q_gate(p, q)
            elif inst.name == "CZ":
                sim.run_2q_gate("CZ", inst.qubits, None)
            else:
                result = sim.run_1q_gate(_SYMBOLS.get(inst.name, inst.name), inst.qubits[0])
                if inst.name == "M":
                    record.append(bool(result))
        return tuple(record)

    def sample_records(self, shots: int, *, seed: int | None = None) -> tuple[tuple[bool, ...], ...]:
        """Sample physical records with categorical channels and native stabilizers.

        Records follow canonical scheduled measurement order, contain no MPAD,
        and can be passed directly to ``noisy.ideal.evaluate``. That evaluates
        authored affine values, not deviations from an ideal reference.
        """
        if type(shots) is not int or shots < 0:
            msg = "shots must be a nonnegative integer"
            raise ValueError(msg)
        if seed is not None and (type(seed) is not int or seed < 0):
            msg = "seed must be a nonnegative integer or None"
            raise ValueError(msg)
        rng = Random(seed)  # noqa: S311 - reproducible Monte Carlo, not cryptography
        return tuple(self._record(rng, None) for _ in range(shots))

    def replay_faults(self, faults: Sequence[PauliFault], *, seed: int = 0) -> tuple[bool, ...]:
        """Execute selected physical branches with all other noise removed."""
        from pecos.qec.generic.scheduled_faults import _validated_faults  # noqa: PLC0415

        rng = Random(seed)  # noqa: S311 - reproducible trajectory replay
        return self._record(rng, _validated_faults(self.noisy, faults))

    @cached_property
    def _propagation(self) -> tuple[Callable[[int, int], int], dict[str, tuple[int, ...]]]:
        """Build native backward propagation and prove parity determinism.

        A backward parity must commute with Z at every preparation output and
        measurement input. These are precisely the Clifford stabilizer-flow
        constraints for Z preparation/measurement. All live wires start with R.
        The influence map supplies these commutators as Z-fault responses.
        """
        from pecos_rslib.qec import PAULI_Z, DagFaultAnalyzer  # noqa: PLC0415

        from pecos.quantum import DagCircuit, Gate, GateType  # noqa: PLC0415

        dag, tails, markers, measurements = DagCircuit(), {}, {}, {}
        collapse_nodes = set()
        gate_types = {"R": GateType.Prep, "M": GateType.Measure, "S": GateType.S, "S_DAG": GateType.Sdg}
        gate_types.update({"C_XYZ": GateType.F, "C_ZYX": GateType.Fdg, "SQRT_X": GateType.SX})

        def append(gate: Gate) -> int:
            node = dag.add_gate(gate)
            for q in gate.qubits:
                if q in tails:
                    dag.connect(tails[q], node, q)
                tails[q] = node
            return node

        for inst in self._instructions:
            if isinstance(inst, NoiseChannel):
                markers[inst.id] = tuple(append(Gate.i([q])) for q in inst.qubits)
            else:
                kind = gate_types[inst.name] if inst.name in gate_types else getattr(GateType, inst.name)
                node = append(Gate(kind, qubits=inst.qubits))
                if inst.name == "M":
                    measurements[node, inst.qubits[0]] = len(measurements)
                if inst.name in {"R", "M"}:
                    collapse_nodes.add(node)
        influence = DagFaultAnalyzer(dag).build_influence_map()
        order = [measurements[node, q] for node, q, _basis in influence.measurements()]
        parities = (*self.noisy.ideal.detectors, *self.noisy.ideal.observables)
        record_masks = [0] * len(measurements)
        for i, parity in enumerate(parities):
            for m in parity.records:
                record_masks[m] ^= 1 << i

        def effect(location: int, pauli: int) -> int:
            mask = 0
            for m in influence.get_detector_indices(location, pauli):
                mask ^= record_masks[order[m]]
            return mask

        locations = {}
        for i, loc in enumerate(influence.get_locations()):
            locations[loc.node, loc.qubits[0]] = i
            if loc.node in collapse_nodes and effect(i, PAULI_Z):
                msg = "Native DEM/fault analysis requires deterministic ideal detectors and observables"
                raise ValueError(msg)
        return effect, {
            c.id: tuple(locations[node, q] for node, q in zip(markers[c.id], c.qubits, strict=True))
            for c in self.noisy.channels
        }

    def _branch_mask(self, channel: NoiseChannel, word: str) -> int:
        from pecos_rslib.qec import PAULI_X, PAULI_Y, PAULI_Z  # noqa: PLC0415

        effect, locations = self._propagation
        codes = {"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
        mask = 0
        for loc, p in zip(locations[channel.id], word, strict=True):
            if p != "I":
                mask ^= effect(loc, codes[p])
        return mask

    def fault_effect(self, faults: Sequence[PauliFault]) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """Native propagated detector/observable flips relative to the ideal reference.

        Affine constants cancel in deviations. These are not raw parity values.
        """
        from pecos.qec.generic.scheduled_faults import _validated_faults  # noqa: PLC0415

        selected = _validated_faults(self.noisy, faults)
        mask = 0
        for c in self.noisy.channels:
            if c.id in selected:
                mask ^= self._branch_mask(c, selected[c.id])
        nd = len(self.noisy.ideal.detectors)
        return tuple(bool(mask & (1 << i)) for i in range(nd)), tuple(
            bool(mask & (1 << (nd + i))) for i in range(len(self.noisy.ideal.observables))
        )

    def detector_error_model(self) -> DetectorErrorModel:
        """Build a native independent-error DEM with exact depolarizing conversion.

        For an n-qubit depolarizing channel, independently applying each of the
        4**n-1 nonidentity Paulis with probability q gives the same distribution
        when (1-2q)**(4**n/2) = 1 - p*4**n/(4**n-1). SI1000's range satisfies
        this conversion. ``expm1/log1p`` retain small nonzero probabilities.
        Identical effects from independent locations combine by XOR probability.

        The returned PECOS model supports native samplers and decoders. This
        aggregate DEM does not retain categorical physical-location identities;
        use ``fault_effect``/``replay_faults`` for physical witnesses.
        """
        from pecos_rslib.qec import DetectorErrorModel  # noqa: PLC0415

        # Validate ideal determinism even when p=0 and no channels exist.
        _ = self._propagation
        probabilities = {}
        for channel in self.noisy.channels:
            p = _independent_branch_probability(channel)
            if not p:
                continue
            for word in self._branches[channel.id]:
                mask = self._branch_mask(channel, word)
                if mask:
                    previous = probabilities.get(mask, 0.0)
                    probabilities[mask] = previous * (1 - p) + p * (1 - previous)
        nd, no = len(self.noisy.ideal.detectors), len(self.noisy.ideal.observables)
        mechanisms = [
            (p, [i for i in range(nd) if mask & (1 << i)], [i for i in range(no) if mask & (1 << (nd + i))])
            for mask, p in sorted(probabilities.items())
        ]
        return DetectorErrorModel.from_independent_mechanisms(
            mechanisms,
            num_detectors=nd,
            num_observables=no,
            detector_coordinates=[p.coords for p in self.noisy.ideal.detectors],
        )

    def explain_signature(
        self, *, detectors: Sequence[int] = (), observables: Sequence[int] = (),
    ) -> FaultSignatureExplanation:
        """Explain the nonzero independent factors for one detector/logical support.

        Physical branch masses are local to one channel; DEM factors recombine
        by XOR. Neither quantity is a decoded logical error probability.
        """
        from pecos.qec.generic.fault_explanations import explain_fault_signature  # noqa: PLC0415

        return explain_fault_signature(self, detectors=detectors, observables=observables)
