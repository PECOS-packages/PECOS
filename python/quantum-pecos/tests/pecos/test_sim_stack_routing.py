# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Contract tests for routing Python sim() to the pecos-neo stack.

Mirrors the Rust contract tests in crates/pecos/tests/neo_routing_test.rs:
the neo stack must return the same results contract as the engines stack,
with exact equality for deterministic programs and statistical agreement
under noise.
"""

from __future__ import annotations

import pytest
from pecos_rslib import Qasm, depolarizing_noise, sim, state_vector

DETERMINISTIC_CONDITIONAL = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
measure q[0] -> c[0];
if (c == 1) x q[1];
measure q[1] -> c[1];
"""

X_MEASURE = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q[0];
measure q[0] -> c[0];
"""


def test_neo_stack_matches_engines_for_deterministic_qasm() -> None:
    engines = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).seed(42).run(5)
    neo = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).stack("neo").seed(42).run(5)

    assert list(engines["c"]) == list(neo["c"])
    assert all(value == 3 for value in neo["c"])  # c0 = c1 = 1


def test_neo_stack_parallel_matches_engines() -> None:
    engines = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).seed(7).workers(2).run(6)
    neo = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).stack("neo").seed(7).workers(2).run(6)

    assert list(engines["c"]) == list(neo["c"])


def test_neo_stack_measurement_noise_rate_matches_engines() -> None:
    """Measurement-only noise: P(c = 0) = p_meas on both stacks."""
    p_meas = 0.2
    shots = 4000

    def rate_of_zero(stack: str) -> float:
        noise = (
            depolarizing_noise()
            .with_prep_probability(0.0)
            .with_meas_probability(p_meas)
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
        )
        builder = sim(Qasm.from_string(X_MEASURE)).noise(noise).seed(42)
        if stack == "neo":
            builder = builder.stack("neo")
        results = builder.run(shots)
        zeros = sum(1 for value in results["c"] if value == 0)
        return zeros / shots

    engines_rate = rate_of_zero("engines")
    neo_rate = rate_of_zero("neo")

    # ~5 sigma for p = 0.2 at 4000 shots is ~0.032.
    assert abs(engines_rate - p_meas) < 0.035
    assert abs(neo_rate - p_meas) < 0.035


def test_explicit_engines_stack_is_the_default_path() -> None:
    default = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).seed(3).run(4)
    explicit = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).stack("engines").seed(3).run(4)

    assert list(default["c"]) == list(explicit["c"])


def test_unknown_stack_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown simulation stack"):
        sim(Qasm.from_string(X_MEASURE)).stack("warp-drive")


def test_neo_stack_rejects_explicit_quantum_backend() -> None:
    with pytest.raises(RuntimeError, match="not yet routed to the neo stack"):
        sim(Qasm.from_string(X_MEASURE)).stack("neo").quantum(state_vector()).run(5)


def test_neo_stack_rejects_build() -> None:
    with pytest.raises(RuntimeError, match="build"):
        sim(Qasm.from_string(X_MEASURE)).stack("neo").build()


def test_missing_qasm_source_reports_the_real_problem() -> None:
    """A builder with no program must say so, not misreport an unrouted
    .classical() configuration (regression: review finding S2)."""
    from pecos_rslib import qasm_engine

    for stack in ["engines", "neo"]:
        with pytest.raises(RuntimeError, match="No QASM source specified"):
            qasm_engine().to_sim().stack(stack).run(2)


def test_neo_stack_rejects_explicit_classical_engine() -> None:
    """An explicit .classical() engine must be refused on neo rather than
    silently dropped (regression: review finding S9). The engines stack
    still accepts it."""
    from pecos_rslib import qasm_engine

    explicit = qasm_engine().program(Qasm.from_string(X_MEASURE))

    with pytest.raises(RuntimeError, match="not routed to the neo stack"):
        sim(Qasm.from_string(X_MEASURE)).classical(explicit).stack("neo").run(5)

    # Same configuration is fine on the engines stack.
    results = sim(Qasm.from_string(X_MEASURE)).classical(explicit).stack("engines").run(5)
    assert len(list(results["c"])) == 5


def test_missing_source_wins_over_classical_override_on_neo() -> None:
    """A sourceless .classical() builder must report the missing source,
    not the neo classical-override rejection, since the missing source is
    the more fundamental error (re-review S2/S9 ordering gap)."""
    from pecos_rslib import qasm_engine

    for stack in ["engines", "neo"]:
        with pytest.raises(RuntimeError, match="No QASM source specified"):
            qasm_engine().to_sim().classical(qasm_engine()).stack(stack).run(1)


# --- Unified .shots(n) / argless .run() (mirrors the Rust facade) ----------


@pytest.mark.parametrize("stack", ["engines", "neo"])
def test_shots_builder_matches_run_argument(stack: str) -> None:
    """`.shots(n).run()` must equal `.run(n)` on both stacks: shots is a
    builder concern, and the argless run is the unified spelling."""
    via_shots = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).stack(stack).seed(42).shots(5).run()
    via_arg = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).stack(stack).seed(42).run(5)
    assert list(via_shots["c"]) == list(via_arg["c"])
    assert len(list(via_shots["c"])) == 5


def test_run_argument_overrides_shots_builder() -> None:
    """A `run(shots)` argument wins over a prior `.shots(n)`."""
    results = sim(Qasm.from_string(DETERMINISTIC_CONDITIONAL)).seed(42).shots(99).run(5)
    assert len(list(results["c"])) == 5


def test_run_without_shots_fails_fast() -> None:
    """Neither `.shots(n)` nor a `run()` argument -> a loud error, never a
    silent default."""
    with pytest.raises(ValueError, match="No shot count configured"):
        sim(Qasm.from_string(X_MEASURE)).seed(42).run()
