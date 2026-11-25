"""Schedule recording preserves runtime-owned custom event interpretation."""

import math
import sys

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import ScheduledCircuit
from selene_simple_runtime_plugin import SimpleRuntimePlugin

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


def test_schedule_audit_uses_explicit_plugins_event_adapter(tmp_path, scheduled_support):
    library, _ = scheduled_support.build_runtime(tmp_path, 100, events=True)
    calls = []

    class Adapter:
        def normalize(self, batch):
            calls.append(batch["batch_index"])
            output = []
            for operation in batch["operations"]:
                if "Custom" in operation:
                    assert operation == {"Custom": {"tag": 4242, "data": [1]}}
                    output.append({"Rz": {"qubit_id": 0, "theta": math.pi}})
                else:
                    output.append(operation)
            return output

    class Plugin:
        library_file = library
        batch_adapter_factory = Adapter

        def get_init_args(self):
            return SimpleRuntimePlugin().get_init_args()

    plan = ScheduledCircuit(1, ((Op("R", (0,)),), (Op("H", (0,)),), (Op("H", (0,)),), (Op("M", (0,)),)), (), ())
    audit = plan.audit_guppy_schedule(runtime=Plugin())
    report = audit.to_dict()
    assert calls == list(range(report["source_runtime"]["batch_count"]))
    assert report["source_runtime"]["operation_counts"]["Custom"] >= 1
    assert "Custom" not in report["normalized_runtime"]["operation_counts"]
    assert report["normalized_runtime"]["operation_counts"]["Rz"] >= 1
    assert report["event_adapter"] == "delegated"
    assert any("Custom" in op for b in audit.evidence()["runtime_batches"] for op in b["operations"])
    assert not report["noise_equivalence_verified"]
