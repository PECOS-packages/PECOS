# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Measurement references from Python must identify distinct measurements.

The binding rebuilds a ``TickMeasRef`` from the ``(tick, gate_idx, qubit)``
tuples ``mz()`` returns. It used to fabricate the record index as ``0``, so an
annotation spanning several measurements collapsed onto one. These tests
exercise that path through the public Python API, which the Rust unit tests
around ``TickCircuit::meas_ref`` do not reach.
"""

from __future__ import annotations

import pytest


def test_observable_over_two_measurements_uses_both() -> None:
    """An observable over two measurements must flip on either one.

    With independent ``p_meas`` errors on both, the observable flips on an odd
    number of them: ``2 * 0.25 * 0.75 = 0.375``. If the two references collapse
    onto a single record, the XOR-parity of the annotation cancels them and the
    observable ends up over nothing or over one measurement, giving a different
    probability -- or no ``L0`` mechanism at all.
    """
    from pecos.qec import DetectorErrorModel
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    refs = tc.tick().mz([0, 1])
    assert len(refs) == 2
    tc.observable(refs)
    tc.set_meta("num_measurements", "2")
    tc.set_meta("detectors", "[]")

    dem = DetectorErrorModel.from_circuit(tc, p1=0.0, p2=0.0, p_meas=0.25, p_prep=0.0)
    text = dem.to_string()

    assert dem.num_observables == 1
    assert "error(0.375) L0" in text, (
        "an observable over two measurements must flip on an odd number of the "
        f"two independent p_meas errors; got:\n{text}"
    )


# Detector annotations are deliberately not covered here: `build_dem_from_circuit`
# takes detectors from the `detectors` metadata JSON, and
# `InfluenceBuilder::with_circuit_annotations` routes detector annotations to the
# sampler path instead. The binding fix corrects that path too, but exercising it
# needs a sampler-level test.


def test_measurement_ref_from_another_circuit_is_rejected() -> None:
    """A reference that does not identify a measurement is an error, not a guess.

    The binding used to fabricate a placeholder for anything it was handed.
    """
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0])
    tc.tick().mz([0])

    with pytest.raises(ValueError, match="does not identify a measurement"):
        tc.observable([(99, 0, 0)])


def test_dag_detector_rejects_plain_int_measurement_refs() -> None:
    """A bare int is ambiguous between a node index and a measurement id.

    Silently reinterpreting it is how annotations end up referencing the wrong
    measurement, so the binding refuses it outright.
    """
    from pecos.quantum import DagCircuit

    dag = DagCircuit()
    dag.pz([0])
    ms = dag.mz([0])

    with pytest.raises(TypeError, match="plain ints are"):
        dag.detector([0])
    # The tuple refs from mz() still work.
    dag.detector(ms)


def test_dag_detector_rejects_a_forged_tuple_ref() -> None:
    """A (node, qubit) tuple that names no measurement raises, never guesses."""
    from pecos.quantum import DagCircuit

    dag = DagCircuit()
    dag.pz([0])
    h_node = dag.gate_count()
    dag.h([0])
    dag.mz([0])

    with pytest.raises(ValueError, match="does not name a measurement gate"):
        dag.detector([(h_node, 0)])


def test_scrambled_mz_with_ids_annotations_survive_to_the_dag() -> None:
    """Non-positional ids (the Guppy pattern) name measurements exactly.

    De-aliased: ids 9 and 5 are neither positional nor contiguous, so any
    consumer that treats the id as an index gives a wrong answer here.
    """
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    refs = tc.tick().mz_with_ids([0, 1], [9, 5])
    assert len(refs) == 2
    tc.detector(refs, label="d")
    tc.observable([refs[1]], label="o")

    anns = tc.annotations()
    by_label = {a["label"]: a for a in anns}
    assert by_label["d"]["measurement_ids"] == [9, 5]
    assert by_label["o"]["measurement_ids"] == [5]

    dag = tc.to_dag_circuit()
    dag_anns = dag.annotations()
    dag_by_label = {a["label"]: a for a in dag_anns}
    assert dag_by_label["d"]["measurement_ids"] == [9, 5]
    assert dag_by_label["o"]["measurement_ids"] == [5]
