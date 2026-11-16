# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Exact ideal parity checks for closed scheduled experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from pecos.qec.generic.gadget_flows import PauliFlow
from pecos.qec.generic.gadgets import Gadget, NamedParity

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


@dataclass(frozen=True)
class ScheduledParityAudit:
    """Signed zero-parity checks with deterministic-one and random failures.

    Verification concerns the concrete ideal schedule, not a compiler's emitted
    circuit. A valid detector/observable expression is identically zero, including
    its authored affine constant. Nothing is silently recentered to a baseline.
    """

    _report_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return detached definitions, provenance, and signed-flow evidence."""
        return json.loads(self._report_json)

    @property
    def all_valid(self) -> bool:
        """Whether every declared detector and observable is ideally zero."""
        return all(row["valid"] for row in self.to_dict()["parities"])

    def require_valid(self) -> None:
        """Raise with stable indices and labels of nonzero or random definitions."""
        failed = [f"{r['id']} ({r['label']})" for r in self.to_dict()["parities"] if not r["valid"]]
        if failed:
            msg = "Ideal parities are not identically zero: " + ", ".join(failed)
            raise ValueError(msg)

    def describe(self) -> str:
        """Show named equations and distinguish sign errors from random checks."""
        report = self.to_dict()
        lines = [
            f"{report['schedule']} schedule: {len(report['parities'])} ideal parity checks; "
            f"all valid={self.all_valid}",
        ]
        for row in report["parities"]:
            status = "always zero" if row["valid"] else "always one" if row["deterministic"] else "nondeterministic"
            lines.append(f"  {row['id']} ({row['label']}): {row['expression']} = {status}")
        lines.append(
            "Closed ideal experiment only; open-gadget action, compiled Guppy, fault tolerance and noise unaudited",
        )
        return "\n".join(lines)


def audit_scheduled_parities(plan: ScheduledCircuit) -> ScheduledParityAudit:
    """Use signed identity-to-identity flows to check authored closed parities.

    Canonical internal role/event names isolate the verifier from author labels
    containing slashes or punctuation. The receipt explicitly binds those names
    to source roles, measurement ordinals, and named parity definitions.
    """
    if not plan.detectors and not plan.observables:
        msg = "A parity audit requires at least one detector or observable definition"
        raise ValueError(msg)
    roles = tuple(f"role[{i}]" for i in range(plan.num_qubits))
    events = tuple(f"event[{i}]" for i in range(plan.num_measurements))
    authored_roles = plan.role_names or tuple(f"q[{i}]" for i in range(plan.num_qubits))
    authored_events = plan.measurement_names or tuple(f"measurement[{i}]" for i in range(plan.num_measurements))
    gadget = Gadget.from_operations(
        "closed scheduled experiment",
        roles,
        plan.moments,
        inputs=(),
        outputs=(),
        outcomes=events,
        citations=plan.citations,
    )
    contracts, definitions = [], []
    for kind, parities in (("detector", plan.detectors), ("observable", plan.observables)):
        for index, parity in enumerate(parities):
            identity = f"{kind}[{index}]"
            local = NamedParity(tuple(events[i] for i in parity.records), parity.constant)
            contracts.append(PauliFlow(identity, parity=local))
            named = NamedParity(tuple(authored_events[i] for i in parity.records), parity.constant)
            definitions.append(
                {
                    "id": identity,
                    "kind": kind,
                    "index": index,
                    "label": parity.label,
                    "records": list(parity.records),
                    "measurement_names": [authored_events[i] for i in parity.records],
                    "constant": parity.constant,
                    "coords": parity.coords,
                    "expression": str(named),
                },
            )
    evidence = gadget.audit_flows(contracts).to_dict()
    for definition, result in zip(definitions, evidence["contracts"], strict=True):
        definition.update(
            valid=result["valid"],
            deterministic=result["support_valid"],
            ideal_value=(0 if result["valid"] else 1) if result["support_valid"] else None,
        )
    report = {
        "schedule": plan.schedule.value,
        "parities": definitions,
        "flow_evidence": evidence,
        "role_bindings": [
            {"internal": internal, "authored": authored, "index": i}
            for i, (internal, authored) in enumerate(zip(roles, authored_roles, strict=True))
        ],
        "measurement_bindings": [
            {"internal": internal, "authored": authored, "index": i}
            for i, (internal, authored) in enumerate(zip(events, authored_events, strict=True))
        ],
        "source_operations": [asdict(row) for row in plan.inspect()],
        "source_citations": list(plan.citations),
        "expected_ideal_value": 0,
        "scope": "All declared parities of this closed ideal scheduled experiment",
        "open_gadget_action_verified": False,
        "compiled_guppy_verified": False,
        "fault_tolerance_verified": False,
        "noise_equivalence_verified": False,
    }
    return ScheduledParityAudit(json.dumps(report))
