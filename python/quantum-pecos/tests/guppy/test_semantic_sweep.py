# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic semantic sweep: execute diverse guppy programs end-to-end.

Every static review of the HUGR engine shared one blind spot: nobody RAN
adversarial programs against it. Each test here is a deterministic program
whose classical computation gates an X on a fresh qubit -- the measurement
is 1 iff the engine computed the exact expected value, so a wrong result,
a stalled loop, or a mis-propagated wire fails loudly. Programs are chosen
to cross-cut the semantics the engine implements: Euclidean division,
logical shifts, comparison chains, nested and zero-iteration loops, while
loops, measurement-conditioned branches, function calls, tuples, and
sequential loops sharing state.
"""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, measure, qubit, x
from pecos import Guppy, sim
from pecos_rslib import state_vector


def _expect_all_ones(prog, shots: int = 3) -> None:
    results = sim(Guppy(prog)).qubits(4).quantum(state_vector()).seed(7).run(shots).to_dict()
    raw_measurements = results["measurements"]
    values = [m[-1] if isinstance(m, list) else m for m in raw_measurements]
    assert values == [1] * shots, f"semantic anchor failed: {values}"


def test_euclidean_matrix() -> None:
    """Several signed division/modulo identities in one predicate."""

    @guppy
    def euclid_matrix() -> bool:
        q = qubit()
        ok = (-7) % 3 == 2
        ok = ok and (-7) // 3 == -3
        ok = ok and 7 % 3 == 1
        ok = ok and 7 // 3 == 2
        ok = ok and (-1) % 2 == 1
        ok = ok and (-9) // 2 == -5
        if ok:
            x(q)
        return measure(q).read()

    _expect_all_ones(euclid_matrix)


def test_shift_chain() -> None:
    """Shift identities on positive values."""

    @guppy
    def shift_chain() -> bool:
        q = qubit()
        a = 1
        ok = (a << 10) == 1024
        ok = ok and (1024 >> 3) == 128
        ok = ok and (0 << 5) == 0
        ok = ok and (7 >> 3) == 0
        if ok:
            x(q)
        return measure(q).read()

    _expect_all_ones(shift_chain)


def test_comparison_chain_negatives() -> None:
    """Signed ordering comparisons across zero."""

    @guppy
    def cmp_chain() -> bool:
        q = qubit()
        a = -5
        b = 3
        ok = a < b
        ok = ok and a <= -5
        ok = ok and b > a
        ok = ok and b >= 3
        ok = ok and a != b
        ok = ok and (a + 8) == b
        if ok:
            x(q)
        return measure(q).read()

    _expect_all_ones(cmp_chain)


def test_nested_loop_accumulation() -> None:
    """A 3x4 nested loop must accumulate exactly 12."""

    @guppy
    def nested_accumulate() -> bool:
        q = qubit()
        count = 0
        for _i in range(3):
            for _j in range(4):
                count = count + 1
        if count == 12:
            x(q)
        return measure(q).read()

    _expect_all_ones(nested_accumulate)


def test_zero_iteration_inner_loop() -> None:
    """A zero-range inner loop must not perturb the outer accumulation."""

    @guppy
    def zero_inner() -> bool:
        q = qubit()
        count = 0
        for _i in range(3):
            count = count + 1
            for _j in range(0):
                count = count + 100
        if count == 3:
            x(q)
        return measure(q).read()

    _expect_all_ones(zero_inner)


def test_while_countdown() -> None:
    """A while loop must run its exact number of iterations."""

    @guppy
    def countdown() -> bool:
        q = qubit()
        n = 5
        steps = 0
        while n > 0:
            n = n - 1
            steps = steps + 1
        if n == 0 and steps == 5:
            x(q)
        return measure(q).read()

    _expect_all_ones(countdown)


def test_measurement_correlated_branch() -> None:
    """A measured bit routed through a branch must correlate exactly."""

    @guppy
    def correlated() -> bool:
        q1 = qubit()
        q2 = qubit()
        h(q1)
        m1 = measure(q1).read()
        if m1:
            x(q2)
        m2 = measure(q2).read()
        q3 = qubit()
        if m1 == m2:
            x(q3)
        return measure(q3).read()

    _expect_all_ones(correlated, shots=10)


def test_function_call_arithmetic() -> None:
    """A called function's return value must flow back exactly."""

    @guppy
    def double(n: int) -> int:
        return n * 2

    @guppy
    def call_arith() -> bool:
        q = qubit()
        if double(21) == 42 and double(-3) == -6:
            x(q)
        return measure(q).read()

    _expect_all_ones(call_arith)


def test_tuple_roundtrip() -> None:
    """Tuple construction and unpacking must preserve both values."""

    @guppy
    def tuple_roundtrip() -> bool:
        q = qubit()
        pair = (3, 4)
        a, b = pair
        if a + b == 7 and a * b == 12:
            x(q)
        return measure(q).read()

    _expect_all_ones(tuple_roundtrip)


def test_sequential_loops_shared_state() -> None:
    """Two sequential loops over the same accumulator must both run."""

    @guppy
    def sequential_loops() -> bool:
        q = qubit()
        total = 0
        for i in range(4):
            total = total + i
        for _j in range(2):
            total = total + 10
        if total == 26:
            x(q)
        return measure(q).read()

    _expect_all_ones(sequential_loops)


def test_branch_chain_on_loop_counter() -> None:
    """An if/elif chain evaluated inside a loop must pick each arm."""

    @guppy
    def branch_chain() -> bool:
        q = qubit()
        low = 0
        mid = 0
        high = 0
        for i in range(6):
            if i < 2:
                low = low + 1
            elif i < 4:
                mid = mid + 1
            else:
                high = high + 1
        if low == 2 and mid == 2 and high == 2:
            x(q)
        return measure(q).read()

    _expect_all_ones(branch_chain)


def test_mixed_arithmetic_expression() -> None:
    """Composite expressions with precedence and negatives."""

    @guppy
    def mixed_expr() -> bool:
        q = qubit()
        ok = (5 * 7 - 3) // 4 == 8
        ok = ok and (-13) % 5 == 2
        ok = ok and (2 + 3) * (7 - 4) == 15
        if ok:
            x(q)
        return measure(q).read()

    _expect_all_ones(mixed_expr)


def test_measured_bits_in_arithmetic() -> None:
    """Measured booleans used as integers must arithmetic correctly."""

    @guppy
    def bits_arith() -> bool:
        q1 = qubit()
        q2 = qubit()
        x(q1)  # deterministic 1
        m1 = measure(q1).read()  # 1
        m2 = measure(q2).read()  # 0
        total = int(m1) + int(m1) + int(m2)
        q3 = qubit()
        if total == 2:
            x(q3)
        return measure(q3).read()

    _expect_all_ones(bits_arith)


def test_loop_carrying_measured_state() -> None:
    """A loop accumulating deterministic measurement outcomes."""

    @guppy
    def loop_measures() -> bool:
        count = 0
        for _i in range(3):
            q = qubit()
            x(q)
            if measure(q).read():
                count = count + 1
        q_out = qubit()
        if count == 3:
            x(q_out)
        return measure(q_out).read()

    _expect_all_ones(loop_measures)


def test_result_label_containing_reserved_words() -> None:
    """Labels are read from the op's typed String arg, so user labels
    containing "result", "Op", or "Report" (which the old Debug-scrape
    heuristics rejected) must survive verbatim as result keys. The array
    variant matters most: its extra BoundedNat arg broke the old primary
    pattern, falling through to the rejecting heuristics."""

    @guppy
    def labeled() -> None:
        q = qubit()
        x(q)
        result("my_result_Report_Op", measure(q).read())

    results = sim(Guppy(labeled)).qubits(2).quantum(state_vector()).seed(7).run(3).to_dict()
    assert results["my_result_Report_Op"] == [1, 1, 1], f"keys: {sorted(results)}"


def test_array_result_label_containing_reserved_words() -> None:
    """Array results carry [String, BoundedNat] type args; the label must
    still come from the typed String arg."""
    from guppylang.std.builtins import array
    from guppylang.std.quantum import collect_measurements, measure_array

    @guppy
    def labeled_array() -> None:
        qs = array(qubit() for _ in range(2))
        x(qs[0])
        x(qs[1])
        result("my_result_Report", collect_measurements(measure_array(qs)))

    results = sim(Guppy(labeled_array)).qubits(3).quantum(state_vector()).seed(7).run(2).to_dict()
    assert results["my_result_Report"] == [[1, 1], [1, 1]], f"keys: {sorted(results)}"


def test_negative_divisor_euclidean_semantics() -> None:
    """Division/modulo with NEGATIVE divisors -- the regime where Python's
    floor semantics and HUGR's Euclidean semantics diverge. HUGR idiv_s
    takes an UNSIGNED divisor, so a negative divisor contributes its
    two's-complement bit pattern (2^64 - 3 here): Euclidean q*m+r=n with
    0 <= r < m gives q=0, r=7. (Python would say 7 // -3 == -3.)"""

    @guppy
    def neg_div() -> int:
        a = 7
        b = -3
        return a // b

    @guppy
    def neg_mod() -> int:
        a = 7
        b = -3
        return a % b

    r = sim(Guppy(neg_div)).qubits(1).quantum(state_vector()).seed(1).run(2).to_dict()
    assert list(r["return"]) == [0, 0], f"7 // -3 per HUGR spec: {r['return']}"
    r = sim(Guppy(neg_mod)).qubits(1).quantum(state_vector()).seed(1).run(2).to_dict()
    assert list(r["return"]) == [7, 7], f"7 %% -3 per HUGR spec: {r['return']}"


def test_logical_shift_on_negative_and_past_width() -> None:
    """ishr is LOGICAL per the spec ("leftmost bits set to zero"), unlike
    Python's arithmetic >>; and shifting by k >= width drops every bit."""

    @guppy
    def shift_negative() -> int:
        a = -8
        return a >> 1

    @guppy
    def shift_past_width() -> int:
        a = 5
        return a >> 70

    r = sim(Guppy(shift_negative)).qubits(1).quantum(state_vector()).seed(1).run(2).to_dict()
    # (-8 as u64) >> 1 = 0x7FFF_FFFF_FFFF_FFFC
    assert list(r["return"]) == [0x7FFF_FFFF_FFFF_FFFC] * 2, f"logical shift: {r['return']}"
    r = sim(Guppy(shift_past_width)).qubits(1).quantum(state_vector()).seed(1).run(2).to_dict()
    assert list(r["return"]) == [0, 0], f"past-width shift: {r['return']}"
