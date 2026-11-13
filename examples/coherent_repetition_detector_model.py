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

"""Compare coherent and Pauli repetition models on the same state-vector shots.

This deliberately small example uses distance three, three rounds, and coherent
angles (0.6, 0.9, 0.94) radians. Select bit/Rx/ZZ or phase/Rz/XX protection.
Preparation, parity extraction, and readout are ideal. The ordinary check
history is the only input to both inferences; final data are scoring truth.

Run from a PECOS environment with Guppy, Selene, and the QuEST plugin:

    python examples/coherent_repetition_detector_model.py --basis phase --shots 2000

The comparison uses the compact models' exact logical posteriors directly.
It does not require or benchmark a decoder adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import selene_sim
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, rx, rz
from pecos.experimental.detector_models import RepetitionConfig, RepetitionModel
from selene_sim import Quest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@guppy
def bit_memory() -> None:
    """Prepare logical zero and extract ZZ checks under coherent Rx noise."""
    d0, d1, d2 = qubit(), qubit(), qubit()
    for _ in range(3):
        # Guppy's angle constructor takes half turns, so divide radians by pi.
        rx(d0, angle(0.6 / 3.141592653589793))
        rx(d1, angle(0.9 / 3.141592653589793))
        rx(d2, angle(0.94 / 3.141592653589793))
        a = qubit()
        cx(d0, a)
        cx(d1, a)
        result("check_0", measure(a).read())
        b = qubit()
        cx(d1, b)
        cx(d2, b)
        result("check_1", measure(b).read())
    result("data_0", measure(d0).read())
    result("data_1", measure(d1).read())
    result("data_2", measure(d2).read())


@guppy
def phase_memory() -> None:
    """Prepare logical plus and extract XX checks under coherent Rz noise."""
    d0, d1, d2 = qubit(), qubit(), qubit()
    h(d0)
    h(d1)
    h(d2)
    for _ in range(3):
        rz(d0, angle(0.6 / 3.141592653589793))
        rz(d1, angle(0.9 / 3.141592653589793))
        rz(d2, angle(0.94 / 3.141592653589793))
        a = qubit()
        h(a)
        cx(a, d0)
        cx(a, d1)
        h(a)
        result("check_0", measure(a).read())
        b = qubit()
        h(b)
        cx(b, d1)
        cx(b, d2)
        h(b)
        result("check_1", measure(b).read())
    h(d0)
    result("data_0", measure(d0).read())
    h(d1)
    result("data_1", measure(d1).read())
    h(d2)
    result("data_2", measure(d2).read())


def parse_record(record: Iterable[tuple[str, object]]) -> tuple[list[list[int]], list[int]]:
    """Preserve repeated check tags and reject missing or nonbinary records."""
    values: dict[str, list[int]] = defaultdict(list)
    expected = {"check_0": 3, "check_1": 3, "data_0": 1, "data_1": 1, "data_2": 1}
    for name, item in record:
        if name not in expected:
            msg = f"Unexpected measurement tag: {name}"
            raise ValueError(msg)
        items = item if isinstance(item, (list, tuple)) else [item]
        for value in items:
            if not isinstance(value, (int, bool)) or value not in (0, 1):
                msg = f"Nonbinary ordinary measurement at {name}: {value!r}"
                raise ValueError(msg)
            values[name].append(int(value))
    if any(len(values[name]) != count for name, count in expected.items()):
        msg = "Each shot requires three values per check and one per final data site"
        raise ValueError(msg)
    checks = [[values["check_0"][t], values["check_1"][t]] for t in range(3)]
    return checks, [values[f"data_{site}"][0] for site in range(3)]


def score_records(
    model: RepetitionModel,
    checks: Sequence[Sequence[Sequence[int]]],
    final_data: Sequence[Sequence[int]],
) -> dict[str, int]:
    """Decode check histories first, then score the held-out logical readout.

    Both bases prepare logical zero in their respective reference basis.
    Final parity checks are verified, but the raw logical readout is never
    passed to the posterior query.
    """
    if len(checks) != len(final_data):
        msg = "Check histories and final data must have matching shot counts"
        raise ValueError(msg)
    if any(
        len(data) != 3
        or any(value not in (0, 1) for value in data)
        or [data[0] ^ data[1], data[1] ^ data[2]] != list(history[-1])
        for history, data in zip(checks, final_data, strict=True)
    ):
        msg = "Final data must be binary and agree with the ideal final checks"
        raise ValueError(msg)
    posteriors = model.query(checks)
    coherent = posteriors["posterior"].tolist()
    canonical = posteriors["canonical_posterior"].tolist()
    coherent_predictions = [int(row[1] > row[0]) for row in coherent]
    canonical_predictions = [int(row[1] > row[0]) for row in canonical]
    targets = [int(data[0]) for data in final_data]
    return {
        "shots": len(targets),
        "canonical_failures": sum(
            prediction != target for prediction, target in zip(canonical_predictions, targets, strict=True)
        ),
        "coherent_failures": sum(
            prediction != target for prediction, target in zip(coherent_predictions, targets, strict=True)
        ),
        "different_corrections": sum(
            left != right for left, right in zip(canonical_predictions, coherent_predictions, strict=True)
        ),
    }


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("quantum-pecos", "pecos-rslib", "guppylang", "hugr", "selene-sim", "selene-hugr-qis-compiler"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    """Return a per-method 95% Wilson interval, including zero-failure cases."""
    if type(failures) is not int or type(shots) is not int or shots <= 0 or not 0 <= failures <= shots:
        msg = "Wilson intervals require positive integer shots and integer failures between zero and shots"
        raise ValueError(msg)
    z = 1.959963984540054
    probability = failures / shots
    denominator = 1 + z * z / shots
    centre = (probability + z * z / (2 * shots)) / denominator
    half_width = z * math.sqrt(probability * (1 - probability) / shots + z * z / (4 * shots * shots)) / denominator
    lower = 0.0 if failures == 0 else max(0.0, centre - half_width)
    upper = 1.0 if failures == shots else min(1.0, centre + half_width)
    return lower, upper


def run_example(*, basis: str = "bit", shots: int = 2000, seed: int = 20261113) -> dict[str, object]:
    """Build fixed models, simulate one cohort, and compare exact decisions."""
    if basis not in ("bit", "phase"):
        msg = "basis must be bit or phase"
        raise ValueError(msg)
    if type(shots) is not int or shots <= 0:
        msg = "shots must be a positive integer"
        raise ValueError(msg)
    if type(seed) is not int or seed < 0:
        msg = "seed must be a nonnegative integer"
        raise ValueError(msg)
    model = RepetitionModel(RepetitionConfig(basis=basis))
    program = bit_memory if basis == "bit" else phase_memory
    package = program.compile()
    quest = Quest()
    with TemporaryDirectory(prefix="pecos-coherent-repetition-") as temporary:
        instance = selene_sim.build(package, name=f"coherent_{basis}_memory", build_dir=Path(temporary))
        checks, final_data = [], []
        for record in instance.run_shots(
            simulator=quest,
            n_qubits=4,
            n_shots=shots,
            random_seed=seed,
            seed_mode="default",
        ):
            history, data = parse_record(record)
            checks.append(history)
            final_data.append(data)
        executable_sha256 = hashlib.sha256(instance.executable.read_bytes()).hexdigest()
    if len(checks) != shots:
        msg = "The simulator did not return the requested number of shots"
        raise ValueError(msg)
    counts = score_records(model, checks, final_data)
    record_bytes = json.dumps({"checks": checks, "final_data": final_data}, separators=(",", ":")).encode()
    return {
        "basis": basis,
        "distance": 3,
        "rounds": 3,
        "angles_radians": [0.6, 0.9, 0.94],
        "seed": seed,
        **counts,
        "canonical_error_rate": counts["canonical_failures"] / shots,
        "coherent_error_rate": counts["coherent_failures"] / shots,
        "canonical_ci95": wilson_interval(counts["canonical_failures"], shots),
        "coherent_ci95": wilson_interval(counts["coherent_failures"], shots),
        "interval_method": "95% Wilson score per method; not an interval for the paired improvement",
        "runtime_route": "Guppy -> Selene -> QuEST pure state vector",
        "maximum_live_qubits": 4,
        "ideal_extraction_and_readout": True,
        "inference_input": "ordinary check history only",
        "scoring_target": "final data site zero in the prepared logical reference basis",
        "versions": _versions(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hugr_sha256": hashlib.sha256(package.to_bytes()).hexdigest(),
        "executable_sha256": executable_sha256,
        "quest_plugin_sha256": hashlib.sha256(Path(quest.library_file).read_bytes()).hexdigest(),
        "observed_records_sha256": hashlib.sha256(record_bytes).hexdigest(),
    }


def main() -> None:
    """Run the small state-vector comparison and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=("bit", "phase"), default="bit")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; an existing file is rejected")
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists():
        parser.error("--output must name a new file")
    report = run_example(basis=arguments.basis, shots=arguments.shots, seed=arguments.seed)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
