from __future__ import annotations

import pytest
from pecos.qec.surface.decode import _validate_trace_hosted_operations_if_requested
from pecos.quantum.hosted import validate_hosted_operations


class FakeGateType:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeGate:
    def __init__(
        self,
        name: str,
        qubits: list[int],
        *,
        meta: dict[str, object] | None = None,
    ) -> None:
        self.gate_type = FakeGateType(name)
        self.qubits = qubits
        self.meta = meta or {}


class FakeTick:
    def __init__(self, gates: list[FakeGate]) -> None:
        self._gates = gates

    def gate_batches(self) -> list[FakeGate]:
        return self._gates


class FakeTickCircuit:
    def __init__(self, ticks: list[list[FakeGate]]) -> None:
        self._ticks = [FakeTick(gates) for gates in ticks]

    def num_ticks(self) -> int:
        return len(self._ticks)

    def get_tick(self, tick_index: int) -> FakeTick:
        return self._ticks[tick_index]

    def get_gate_meta(self, tick_index: int, gate_index: int, key: str) -> object:
        return self._ticks[tick_index].gate_batches()[gate_index].meta[key]


def test_validate_hosted_operations_binds_local_to_later_host() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [2],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("Idle", [2])],
            [FakeGate("SZZ", [2, 5], meta={"host_id": "host:a"})],
        ],
    )

    bindings = validate_hosted_operations(circuit)

    assert len(bindings) == 1
    assert bindings[0].host_id == "host:a"
    assert bindings[0].local_role == "basis_prefix"
    assert bindings[0].local.gate_name == "H"
    assert bindings[0].host.gate_name == "SZZ"
    assert bindings[0].tick_separation == 2


def test_validate_hosted_operations_selects_later_shared_host_record() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "SX",
                    [2],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("RZ", [9], meta={"host_id": "host:a"})],
            [FakeGate("SZZ", [2, 5], meta={"host_id": "host:a"})],
        ],
    )

    bindings = validate_hosted_operations(circuit)

    assert len(bindings) == 1
    assert bindings[0].host.gate_name == "SZZ"
    assert bindings[0].host.qubits == (2, 5)


def test_validate_hosted_operations_can_require_unique_host_ids() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
        ],
    )

    with pytest.raises(ValueError, match="host_id 'host:a' appears on 2 host gates"):
        validate_hosted_operations(circuit, require_unique_host_id=True)


def test_validate_hosted_operations_unique_host_ids_allow_many_locals() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
                FakeGate(
                    "S",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
        ],
    )

    bindings = validate_hosted_operations(circuit, require_unique_host_id=True)

    assert len(bindings) == 2
    assert {binding.local.gate_name for binding in bindings} == {"H", "S"}
    assert all(binding.host.gate_name == "SZZ" for binding in bindings)


def test_validate_hosted_operations_can_bind_without_shared_qubit_requirement() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "SX",
                    [2],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("RZ", [9], meta={"host_id": "host:a"})],
        ],
    )

    bindings = validate_hosted_operations(circuit, require_shared_qubit=False)

    assert len(bindings) == 1
    assert bindings[0].host.gate_name == "RZ"


def test_validate_hosted_operations_rejects_ordering_drift_by_default() -> None:
    circuit = FakeTickCircuit(
        [
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
        ],
    )

    with pytest.raises(ValueError, match="matching host metadata only before it"):
        validate_hosted_operations(circuit)


def test_validate_hosted_operations_can_bind_prior_host_for_metadata_shape_audit() -> None:
    circuit = FakeTickCircuit(
        [
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
        ],
    )

    bindings = validate_hosted_operations(circuit, require_host_after_local=False)

    assert len(bindings) == 1
    assert bindings[0].host.gate_name == "SZZ"
    assert bindings[0].tick_separation == -1


def test_validate_hosted_operations_rejects_missing_local_host_id() -> None:
    circuit = FakeTickCircuit(
        [[FakeGate("H", [0], meta={"local_role": "basis_prefix"})]],
    )

    with pytest.raises(ValueError, match="no non-empty 'host_id' metadata"):
        validate_hosted_operations(circuit)


def test_validate_hosted_operations_rejects_unbound_local() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "missing", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "other"})],
        ],
    )

    with pytest.raises(ValueError, match="has no host gate sharing a qubit"):
        validate_hosted_operations(circuit)


def test_validate_hosted_operations_rejects_large_tick_separation() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("Idle", [0])],
            [FakeGate("Idle", [0])],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
        ],
    )

    with pytest.raises(ValueError, match="exceeding max_tick_separation=2"):
        validate_hosted_operations(circuit, max_tick_separation=2)


def test_trace_hosted_validation_is_noop_unless_requested() -> None:
    circuit = FakeTickCircuit(
        [[FakeGate("H", [0], meta={"local_role": "basis_prefix"})]],
    )

    _validate_trace_hosted_operations_if_requested(
        circuit,
        require_hosted_operation_order=False,
        max_hosted_tick_separation=None,
        context="test trace validation",
    )


def test_trace_hosted_validation_rejects_ordering_drift_when_requested() -> None:
    circuit = FakeTickCircuit(
        [
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
        ],
    )

    with pytest.raises(ValueError, match="ordering drift"):
        _validate_trace_hosted_operations_if_requested(
            circuit,
            require_hosted_operation_order=True,
            max_hosted_tick_separation=None,
            context="test trace validation",
        )


def test_trace_hosted_validation_rejects_repeated_host_ids_when_requested() -> None:
    circuit = FakeTickCircuit(
        [
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
        ],
    )

    with pytest.raises(ValueError, match="host_id 'host:a' appears on 2 host gates"):
        _validate_trace_hosted_operations_if_requested(
            circuit,
            require_hosted_operation_order=True,
            max_hosted_tick_separation=None,
            context="test trace validation",
        )


def test_trace_hosted_validation_can_check_separation_without_order_guard() -> None:
    circuit = FakeTickCircuit(
        [
            [FakeGate("SZZ", [0, 1], meta={"host_id": "host:a"})],
            [
                FakeGate(
                    "H",
                    [0],
                    meta={"host_id": "host:a", "local_role": "basis_prefix"},
                ),
            ],
        ],
    )

    with pytest.raises(ValueError, match="exceeding max_tick_separation=0"):
        _validate_trace_hosted_operations_if_requested(
            circuit,
            require_hosted_operation_order=False,
            max_hosted_tick_separation=0,
            context="test trace validation",
        )
