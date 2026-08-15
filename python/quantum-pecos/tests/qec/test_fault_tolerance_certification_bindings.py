# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""Discriminating Python cases for circuit fault tooling and distance certification."""

import pytest
from pecos.qec import (
    CircuitFaultAnalyzer,
    DetectorErrorModel,
    DistanceProblem,
    HookError,
    bounded_enumeration_code_distance,
    bounded_enumeration_stabilizer_distance,
    bounded_enumeration_x_distance,
    bounded_enumeration_z_distance,
    certified_classical_distance,
    certified_stabilizer_coset_weight,
    connected_cluster_code_distance,
    logical_generator_coset_weights,
    stabilizer_code_distance,
    x_distance,
    z_distance,
)
from pecos.quantum import ParityCheckMatrix, PauliString, StabilizerCode, StabilizerCodeSpec, TickCircuit


def _hook_ladder() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([3])
    circuit.tick().cx([(3, 0)])
    circuit.tick().cx([(3, 1)])
    circuit.tick().cx([(3, 2)])
    circuit.tick().mz([3])
    return circuit


def _weight_four_x_measurement(*, with_flag: bool) -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([4])
    circuit.tick().h([4])
    if with_flag:
        circuit.tick().pz([5])
    circuit.tick().cx([(4, 0)])
    if with_flag:
        circuit.tick().cx([(4, 5)])
    circuit.tick().cx([(4, 1)])
    circuit.tick().cx([(4, 2)])
    if with_flag:
        circuit.tick().cx([(4, 5)])
    circuit.tick().cx([(4, 3)])
    circuit.tick().h([4])
    circuit.tick().mz([4])
    if with_flag:
        circuit.tick().mz([5])
    return circuit


def _three_qubit_extraction() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([3, 4])
    circuit.tick().cx([(0, 3)])
    circuit.tick().cx([(1, 3)])
    circuit.tick().cx([(1, 4)])
    circuit.tick().cx([(2, 4)])
    circuit.tick().mz([3, 4])
    return circuit


def _unequal_logical_distance_circuit() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().pz([1])
    circuit.tick().h([0, 1])
    circuit.tick().cx([(1, 0)])
    circuit.tick().pz([2])
    circuit.tick().h([2])
    return circuit


def _steane_problem() -> DistanceProblem:
    hamming = ParityCheckMatrix(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
    )
    logical = ParityCheckMatrix([[1, 1, 1, 1, 1, 1, 1]])
    return DistanceProblem.from_css_checks(hamming, logical)


def _triad_dem() -> DetectorErrorModel:
    circuit = TickCircuit()
    circuit.tick().mz([0, 1, 2])
    circuit.add_detector(records=[-3, -2])
    circuit.add_detector(records=[-3, -1])
    circuit.add_observable(records=[-3])
    return DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )


def test_hook_ladder_reports_only_single_qubit_amplifying_faults() -> None:
    report = CircuitFaultAnalyzer(_hook_ladder()).hook_errors([0, 1, 2], [3], [], [], 2)

    hook = next(error for error in report.hook_errors if error.location.tick == 1 and error.fault_paulis == [1, 0])
    assert isinstance(hook, HookError)
    assert hook.location.gate_type == "CX"
    assert hook.location.gate_index == 0
    assert hook.location.qubits == [3, 0]
    assert "tick=1" in repr(hook)
    assert 'gate_type="CX"' in repr(hook)
    assert not any(error.location.tick == 3 and error.fault_paulis == [1, 0] for error in report.hook_errors)
    assert not any(error.location.tick == 2 and error.fault_paulis == [1, 1] for error in report.hook_errors)


def test_flagged_pair_satisfies_condition_and_unflagged_pair_exposes_hook() -> None:
    flagged = CircuitFaultAnalyzer(_weight_four_x_measurement(with_flag=True))
    flagged_report = flagged.flag_fault_condition([0, 1, 2, 3], [5], ([0, 1, 2, 3], []), 1)
    assert flagged_report.fault_condition_satisfied
    assert flagged_report.violations == []

    unflagged = CircuitFaultAnalyzer(_weight_four_x_measurement(with_flag=False))
    unflagged_report = unflagged.flag_fault_condition([0, 1, 2, 3], [], ([0, 1, 2, 3], []), 1)
    assert not unflagged_report.fault_condition_satisfied
    assert any(violation.num_faults == 1 and violation.error_weight == 2 for violation in unflagged_report.violations)


def test_circuit_distance_and_per_logical_distances() -> None:
    extraction_result = CircuitFaultAnalyzer(_three_qubit_extraction()).fault_distance([3, 4], [], [([], [0, 1, 2])], 1)
    assert extraction_result is not None
    assert extraction_result.distance == 1

    analyzer = CircuitFaultAnalyzer(_unequal_logical_distance_circuit())
    logicals = [([2], []), ([0], [])]
    per_logical = analyzer.per_logical_fault_distances([], [1], logicals, 2, x_only=True)
    assert [result.distance if result is not None else None for result in per_logical] == [1, 2]

    overall = analyzer.fault_distance([], [1], logicals, 2, x_only=True)
    assert overall is not None
    assert overall.distance == min(result.distance for result in per_logical if result is not None)


def test_steane_certification_from_checks_and_code_spec() -> None:
    from_checks = _steane_problem()
    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.steane())
    from_spec = DistanceProblem.from_css_code_x_distance(spec)

    for problem in (from_checks, from_spec):
        certified = problem.certified_distance(3)
        assert certified is not None
        assert certified.distance == 3
        assert certified.sat_certified
        assert certified.unsat_trusted_below == 3
        assert problem.verify_witness(certified.witness) == 3

    matrix_result = connected_cluster_code_distance(
        ParityCheckMatrix(
            [
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
            ],
        ),
        ParityCheckMatrix([[1, 1, 1, 1, 1, 1, 1]]),
        3,
    )
    assert matrix_result is not None
    assert matrix_result.distance == 3
    assert len(matrix_result.mechanism_indices) == 3

    x_result = x_distance(spec, 3)
    z_result = z_distance(spec, 3)
    assert x_result is not None
    assert x_result.distance == 3
    assert z_result is not None
    assert z_result.distance == 3

    assert from_checks.certified_distance(2) is None

    certified = from_checks.certified_distance(3)
    assert certified is not None
    corrupted = certified.witness.copy()
    corrupted[0] = not corrupted[0]
    with pytest.raises(ValueError, match="witness violates H row 0"):
        from_checks.verify_witness(corrupted)


def test_certified_distance_repr_uses_verified_qubit_support_weight() -> None:
    code = StabilizerCodeSpec(
        2,
        [PauliString.from_dense_str("YY")],
        [PauliString.from_dense_str("YI")],
        [PauliString.from_dense_str("XZ")],
    )

    certified = certified_stabilizer_coset_weight(code, PauliString.from_dense_str("YI"), 2)

    assert certified is not None
    assert certified.distance == 1
    assert sum(certified.witness) == 2
    assert "witness_weight=1" in repr(certified)


def test_bounded_enumeration_bindings_certify_and_return_intervals() -> None:
    h = ParityCheckMatrix(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
    )
    logicals = ParityCheckMatrix([[1, 1, 1, 1, 1, 1, 1]])
    problem = DistanceProblem.from_css_checks(h, logicals)

    exact = bounded_enumeration_code_distance(h, logicals, 4)
    assert exact is not None
    assert exact.certified
    assert exact.distance == exact.upper_bound == 3
    assert exact.lower_bound >= exact.distance
    assert exact.lb_certified
    assert exact.level is not None
    assert exact.max_level is None
    assert problem.verify_witness(exact.witness) == 3

    interval = bounded_enumeration_code_distance(h, logicals, 0)
    assert interval is not None
    assert not interval.certified
    assert interval.distance is None
    assert (interval.lower_bound, interval.upper_bound) == (1, 3)
    assert interval.max_level == 0
    assert problem.verify_witness(interval.witness) == interval.upper_bound

    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.steane())
    assert bounded_enumeration_x_distance(spec, 4).distance == 3
    assert bounded_enumeration_z_distance(spec, 4).distance == 3


def test_five_qubit_bounded_enumeration_binding_uses_three_mechanisms_per_qubit() -> None:
    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.five_qubit())
    result = bounded_enumeration_stabilizer_distance(spec, 5)

    assert result is not None
    assert result.certified
    assert result.distance == 3
    assert len(result.witness) == 3 * spec.num_qubits
    assert sum(result.witness) == 3


def test_non_css_connected_cluster_binding_returns_logical_pauli() -> None:
    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.five_qubit())
    result = stabilizer_code_distance(spec, 3)

    assert result.certified
    assert result.distance == 3
    assert result.min_weight_operator.weight() == 3
    assert all(result.min_weight_operator.commutes_with(stabilizer) for stabilizer in spec.stabilizers)
    assert any(result.min_weight_operator.anticommutes_with(logical) for logical in spec.logical_zs + spec.logical_xs)


def test_stabilizer_distance_binding_reports_budget_exhaustion_and_no_logicals() -> None:
    spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.five_qubit())
    exhausted = stabilizer_code_distance(spec, 2)

    assert not exhausted.certified
    assert exhausted.distance is None
    assert exhausted.min_weight_operator is None
    assert exhausted.lower_bound == 3
    assert exhausted.max_weight == 2

    stabilizer_state = StabilizerCodeSpec(
        1,
        [PauliString.from_dense_str("Z")],
        [],
        [],
    )
    with pytest.raises(ValueError, match="encodes no logical qubits"):
        stabilizer_code_distance(stabilizer_state, 1)


def test_classical_distance_binding_distinguishes_empty_kernel_from_budget() -> None:
    full_rank = ParityCheckMatrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )
    nonexistent = certified_classical_distance(full_rank, 2)
    assert nonexistent.certified
    assert nonexistent.no_nonzero_codeword
    assert nonexistent.distance is None
    assert nonexistent.witness is None
    assert nonexistent.lower_bound is None
    assert nonexistent.max_weight is None

    repetition = ParityCheckMatrix([[1, 1, 0], [0, 1, 1]])
    exhausted = certified_classical_distance(repetition, 2)
    assert not exhausted.certified
    assert not exhausted.no_nonzero_codeword
    assert exhausted.distance is None
    assert exhausted.witness is None
    assert exhausted.lower_bound == 3
    assert exhausted.max_weight == 2


def test_logical_generator_profile_minimum_is_not_code_distance() -> None:
    spec = StabilizerCodeSpec(
        2,
        [],
        [PauliString.from_dense_str("XX"), PauliString.from_dense_str("YZ")],
        [PauliString.from_dense_str("XY"), PauliString.from_dense_str("ZZ")],
    )
    spec.verify()

    profile = logical_generator_coset_weights(spec, 2)
    assert min(entry.distance for entry in profile if entry is not None) == 2
    distance = stabilizer_code_distance(spec, 1)
    assert distance.certified
    assert distance.distance == 1


def test_triad_dem_certification_agrees_with_exhaustive_distance() -> None:
    dem = _triad_dem()
    problem = DistanceProblem.from_dem(dem)
    certified = problem.certified_distance(3)
    exhaustive = dem.exhaustive_fault_distance(3)

    assert certified is not None
    assert exhaustive is not None
    assert certified.distance == exhaustive.distance == 3
    assert problem.verify_witness(certified.witness) == 3
    assert problem.certified_distance(2) is None


def test_distance_problem_text_formats_have_expected_headers_and_soft_clauses() -> None:
    problem = _steane_problem()
    dimacs = problem.to_dimacs(3)
    dimacs_header = next(line for line in dimacs.splitlines() if not line.startswith("c "))
    assert dimacs_header.startswith("p cnf ")

    wcnf = problem.to_wcnf()
    wcnf_header = next(line for line in wcnf.splitlines() if not line.startswith("c "))
    assert wcnf_header.startswith("p wcnf ")
    assert sum(line.startswith("1 -") for line in wcnf.splitlines()) == problem.num_vars
