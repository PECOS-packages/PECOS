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

"""Exact coherent detector instruments for ideal odd-distance repetition memory.

This experimental generator is specific to single-axis rotations and ideal
commuting parity extraction. A syndrome increment has two complementary error
strings. Keeping their complex amplitudes preserves conditional logical
coherence across rounds. Bit/Rx/ZZ and phase/Rz/XX experiments are Hadamard duals.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pecos as pc
from pecos.experimental.detector_models.process import DetectorDef, DetectorProcess, InstrumentFactor, SystemSpec

if TYPE_CHECKING:
    from collections.abc import Iterator


def _distance(distance: int) -> None:
    if type(distance) is not int or not 3 <= distance <= 15 or distance % 2 == 0:
        message = "distance must be odd and between 3 and 15"
        raise ValueError(message)


def _real_angle(value: object) -> float:
    if isinstance(value, (bool, str, bytes, complex)):
        message = "angles must be real radian values"
        raise TypeError(message)
    return float(value)


@dataclass(frozen=True)
class RepetitionConfig:
    """Declare an ideal repetition memory with one rotation per data site/round.

    Angles are radians. ``bit`` means Rx noise, ZZ checks, and Z readout;
    ``phase`` means Rz noise, XX checks, and X readout. These are separate
    protected bases, rather than simultaneous correction of both error axes.
    """

    distance: int = 3
    rounds: int = 3
    basis: str = "bit"
    angles: tuple[float, ...] = (0.6, 0.9, 0.94)

    def __post_init__(self) -> None:
        """Reject unsupported schedules and nonfinite or mismatched angles."""
        _distance(self.distance)
        if type(self.rounds) is not int or not 1 <= self.rounds <= 1000:
            message = "rounds must be between 1 and 1000"
            raise ValueError(message)
        if self.basis not in ("bit", "phase"):
            message = "basis must be bit or phase"
            raise ValueError(message)
        try:
            angles = tuple(_real_angle(angle) for angle in self.angles)
        except (TypeError, ValueError) as error:
            message = "angles must be real radian values"
            raise ValueError(message) from error
        if len(angles) != self.distance or not all(math.isfinite(angle) for angle in angles):
            message = "provide one finite radian angle per data qubit"
            raise ValueError(message)
        object.__setattr__(self, "angles", angles)

    @property
    def num_detectors(self) -> int:
        """Return the number of increments plus known-zero terminal checks."""
        return (self.distance - 1) * (self.rounds + 1)


def base_angles(distance: int) -> tuple[float, ...]:
    """Interpolate the illustrative (0.6, 0.9, 0.94) spatial angle profile.

    This profile is a reproducible benchmark choice, not a hardware calibration.
    """
    _distance(distance)
    values = []
    for site in range(distance):
        fraction = site / (distance - 1)
        values.append(0.6 + 0.6 * fraction if fraction <= 0.5 else 0.9 + 0.08 * (fraction - 0.5))
    return tuple(values)


def _flatten(values: object) -> Iterator[object]:
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _binary_array(value: object, dimensions: int) -> pc.Array:
    try:
        result = pc.asarray(value)
    except (TypeError, ValueError) as error:
        message = f"expected a binary array with {dimensions} dimensions"
        raise ValueError(message) from error
    if len(result.shape) != dimensions or any(
        isinstance(bit, (str, bytes, complex)) or bit not in (0, 1) for bit in _flatten(result.tolist())
    ):
        message = f"expected a binary array with {dimensions} dimensions"
        raise ValueError(message)
    return result


def detector_history(checks: object, final_data: object | None = None) -> pc.Array:
    """Convert ordinary checks to round-major binary detector increments.

    The initial syndrome is zero. Ideal final extraction makes the terminal
    boundary zero. Optional final data only verify parity consistency: the raw
    logical readout never affects these detectors or the inferred posterior.
    """
    array = _binary_array(checks, 3)
    shots, rounds, width = array.shape
    if rounds < 1 or width < 1:
        message = "checks require at least one round and one adjacent parity"
        raise ValueError(message)
    histories = array.tolist()
    if final_data is not None:
        final = _binary_array(final_data, 2)
        if final.shape != (shots, width + 1):
            message = "final data must have one value per data qubit and shot"
            raise ValueError(message)
        for history, final_row in zip(histories, final.tolist(), strict=True):
            parity = [int(left) ^ int(right) for left, right in pairwise(final_row)]
            if parity != history[-1]:
                message = "final parity disagrees with ideal extraction; extra final noise is unsupported"
                raise ValueError(message)
    rows = []
    for history in histories:
        previous, row = [0] * width, []
        for syndrome in history:
            row.extend(int(left) ^ int(right) for left, right in zip(syndrome, previous, strict=True))
            previous = syndrome
        rows.append(row + [0] * width)
    if not rows:
        return pc.zeros((0, (rounds + 1) * width), dtype="uint8")
    return pc.array(rows, dtype=pc.u8)


def _amplitudes(angles: tuple[float, ...], code: int) -> tuple[complex, complex]:
    representative = [0]
    for site in range(len(angles) - 1):
        representative.append(representative[-1] ^ ((code >> site) & 1))
    first, complement = 1 + 0j, 1 + 0j
    for angle, bit in zip(angles, representative, strict=True):
        cosine, sine = math.cos(angle / 2), -1j * math.sin(angle / 2)
        first *= cosine if bit == 0 else sine
        complement *= sine if bit == 0 else cosine
    return first, complement


def _log_probability(value: float) -> float:
    return math.log(value) if value > 0 else -math.inf


class RepetitionModel:
    """Store a compact round instrument and condition it on ordinary checks.

    Each round has ``2**(distance-1)`` maps on a two-dimensional logical
    reference system, independent of the number of rounds. Queries propagate
    complex logical amplitudes; the canonical comparison propagates twirled
    probabilities. No logical target or initial preparation label is an input.
    Public array properties return copies of the immutable scalar cache.
    """

    __slots__ = ("_amplitudes", "_canonical", "_config", "_probabilities")

    def __init__(self, config: RepetitionConfig) -> None:
        """Construct the exact per-round maps from the declared rotations."""
        self._config = config
        self._amplitudes = tuple(_amplitudes(config.angles, code) for code in range(1 << (config.distance - 1)))
        self._refresh_probabilities()

    def _refresh_probabilities(self) -> None:
        probabilities, canonical = [], []
        for first, complement in self._amplitudes:
            probability = abs(first) ** 2 + abs(complement) ** 2
            probabilities.append(probability)
            canonical.append(
                (abs(first) ** 2 / probability, abs(complement) ** 2 / probability) if probability > 0 else (0.0, 0.0),
            )
            if abs(first.conjugate() * complement + complement.conjugate() * first) > 1e-12:
                message = "round maps are not proportional to isometries"
                raise ValueError(message)
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-12):
            message = "round instrument must preserve trace"
            raise ValueError(message)
        self._probabilities, self._canonical = tuple(probabilities), tuple(canonical)

    @property
    def config(self) -> RepetitionConfig:
        """Return the frozen physical and detector configuration."""
        return self._config

    @property
    def delta_operators(self) -> pc.Array:
        """Return the unnormalized maps ``K_delta = a I + b X``."""
        return pc.array(
            [[[first, complement], [complement, first]] for first, complement in self._amplitudes],
            dtype=pc.complex128,
        )

    @property
    def syndrome_probabilities(self) -> pc.Array:
        """Return the common coherent and twirled syndrome-increment law."""
        return pc.array(self._probabilities, dtype=pc.f64)

    @property
    def canonical_conditional(self) -> pc.Array:
        """Return the twirled logical alternatives conditioned on each increment."""
        return pc.array(self._canonical, dtype=pc.f64)

    @property
    def normalized_operators(self) -> pc.Array:
        """Return conditional unitary maps, or zero for impossible increments."""
        operators = []
        for (first, complement), probability in zip(self._amplitudes, self._probabilities, strict=True):
            factor = math.sqrt(probability) if probability > 0 else 1
            operators.append([[first / factor, complement / factor], [complement / factor, first / factor]])
        return pc.array(operators, dtype=pc.complex128)

    def query(self, checks: object, final_data: object | None = None) -> dict[str, pc.Array]:
        """Return coherent/twirled logical posteriors and original log evidence.

        ``checks`` has shape ``(shots, rounds, distance-1)``. The logical event
        is the protected-basis final readout at data site zero, XOR the initial
        logical bit, under a syndrome representative whose first bit is zero.
        Only scoring needs that initial bit. The optional final data are used
        solely for parity validation. Impossible histories raise ``ValueError``.
        """
        array = _binary_array(checks, 3)
        expected = (self.config.rounds, self.config.distance - 1)
        if array.shape[1:] != expected:
            message = f"checks must have shape (shots, {expected[0]}, {expected[1]})"
            raise ValueError(message)
        detectors = detector_history(array, final_data)
        enhanced, canonical, joint, canonical_joint, evidence = [], [], [], [], []
        for increments in detectors.tolist():
            coherent, classical = (1 + 0j, 0j), (1.0, 0.0)
            log_evidence = 0.0
            for round_index in range(self.config.rounds):
                start = round_index * expected[1]
                code = sum(bit << site for site, bit in enumerate(increments[start : start + expected[1]]))
                probability = self._probabilities[code]
                if probability <= 0:
                    message = "observed syndrome has zero probability under the declared model"
                    raise ValueError(message)
                log_evidence += math.log(probability)
                first, complement = (value / math.sqrt(probability) for value in self._amplitudes[code])
                left = first * coherent[0] + complement * coherent[1]
                right = complement * coherent[0] + first * coherent[1]
                norm = math.sqrt(abs(left) ** 2 + abs(right) ** 2)
                coherent = (left / norm, right / norm)
                keep, flip = self._canonical[code]
                classical = (keep * classical[0] + flip * classical[1], flip * classical[0] + keep * classical[1])
                total = sum(classical)
                classical = (classical[0] / total, classical[1] / total)
            posterior = [abs(value) ** 2 for value in coherent]
            total = sum(posterior)
            posterior = [value / total for value in posterior]
            enhanced.append(posterior)
            canonical.append(list(classical))
            joint.append([_log_probability(value) + log_evidence for value in posterior])
            canonical_joint.append([_log_probability(value) + log_evidence for value in classical])
            evidence.append(log_evidence)
        outputs = {
            name: pc.array(values, dtype=pc.f64) if values else pc.zeros((0, 2), dtype="float64")
            for name, values in (
                ("posterior", enhanced),
                ("canonical_posterior", canonical),
                ("log_joint", joint),
                ("canonical_log_joint", canonical_joint),
            )
        }
        outputs["log_evidence"] = pc.array(evidence, dtype=pc.f64)
        outputs["detectors"] = detectors
        return outputs

    def canonical_dem(self) -> str:
        """Render the independent Pauli-twirled baseline in ordinary DEM text."""
        config, width = self.config, self.config.distance - 1
        lines = [f"# Independent Pauli twirl: {config.basis} repetition d={config.distance}, rounds={config.rounds}"]
        lines.extend(
            f"detector({round_index}, {site}) D{round_index * width + site}"
            for round_index in range(config.rounds + 1)
            for site in range(width)
        )
        for round_index in range(config.rounds):
            for site, angle in enumerate(config.angles):
                targets = [f"D{round_index * width + check}" for check in (site - 1, site) if 0 <= check < width]
                if site == 0:
                    targets.append("L0")
                lines.append(f"error({math.sin(angle / 2) ** 2:.17g}) {' '.join(targets)}")
        return "\n".join(lines) + "\n"

    def as_detector_process(self) -> DetectorProcess:
        """Export the full logical quantum instrument in the generic schema.

        The reference frame is virtual: no conditional correction or sensor is
        added to the physical experiment. The output keeps the complete logical
        density operator, including off-diagonal coherence, for arbitrary inputs.
        """
        config, width = self.config, self.config.distance - 1
        labels = [format(index, f"0{width}b")[::-1] for index in range(1 << width)]
        outcomes = {
            label: [[[first, complement], [complement, first]]]
            for label, (first, complement) in zip(labels, self._amplitudes, strict=True)
        }
        factors, detectors = [], []
        for round_index in range(config.rounds):
            record = f"increment_{round_index}"
            factors.append(InstrumentFactor(f"quantum_round_{round_index}", ("logical_reference",), outcomes, record))
            detectors.extend(
                DetectorDef(
                    f"D{round_index * width + site}",
                    (record,),
                    {label: int(label[site]) for label in labels},
                    expected_parity=0,
                )
                for site in range(width)
            )
        detectors.extend(
            DetectorDef(f"D{config.rounds * width + site}", (), expected_parity=0) for site in range(width)
        )
        return DetectorProcess(
            (SystemSpec("logical_reference", ("0L", "1L"), {"computational": (0, 1)}, role="memory"),),
            tuple(factors),
            tuple(detectors),
            (),
            {
                "kind": "exact repetition detector instrument in a virtual Pauli reference frame",
                "config": asdict(config),
                "reference": "syndrome representative has first bit zero; no physical correction",
                "validity": "odd distance, single-axis rotations, ideal commuting parity extraction",
                "output": "logical quantum state in the declared syndrome reference frame",
                "measurement_records": "derived adjacent-check changes; no additional sensors",
            },
        )

    def save(self, path: str | Path) -> dict[str, object]:
        """Save a self-contained JSON snapshot with exact complex amplitudes.

        Existing files are never overwritten. Complex numbers are explicit
        ``[real, imag]`` pairs; a canonical-payload SHA256 detects corruption.
        This experimental format does not require NumPy or pickle.
        """
        path = Path(path)
        amplitudes = [[[value.real, value.imag] for value in sector] for sector in self._amplitudes]
        encoded = json.dumps(amplitudes, separators=(",", ":"), allow_nan=False)
        manifest = {
            "schema_version": 1,
            "kind": "repetition_quantum_detector_instrument",
            "config": asdict(self.config),
            "round_amplitudes": amplitudes,
            "amplitudes_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "model_size": {
                "round_instrument_outcomes": len(amplitudes),
                "logical_dimension": 2,
                "full_history_table_materialized": False,
            },
            "channel_semantics": "K_delta=a_delta I + b_delta X in the logical reference basis",
            "detector_order": "round-major adjacent parity increments, then known-zero terminal boundary",
            "basis_duality": "bit/Rx/ZZ/Zreadout <-> phase/Rz/XX/Xreadout",
            "inference_inputs": ["ordinary check history", "optional final parity consistency check"],
            "no_input": ["preparation label", "target logical-flip label", "leakage flags"],
            "approximations": [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> RepetitionModel:
        """Load and validate a snapshot, preserving its saved complex phases."""
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "repetition_quantum_detector_instrument":
            message = "unsupported compact model schema"
            raise ValueError(message)
        amplitudes = manifest["round_amplitudes"]
        encoded = json.dumps(amplitudes, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(encoded.encode()).hexdigest() != manifest["amplitudes_sha256"]:
            message = "compact model amplitude hash mismatch"
            raise ValueError(message)
        result = cls(RepetitionConfig(**manifest["config"]))
        try:
            saved = tuple(tuple(complex(*value) for value in sector) for sector in amplitudes)
        except (TypeError, ValueError) as error:
            message = "malformed complex amplitude payload"
            raise ValueError(message) from error
        if len(saved) != len(result._amplitudes) or any(len(sector) != 2 for sector in saved):
            message = "compiled amplitude shape differs from declared parameters"
            raise ValueError(message)
        for expected, actual in zip(result._amplitudes, saved, strict=True):
            for left, right in zip(expected, actual, strict=True):
                if not math.isfinite(right.real) or not math.isfinite(right.imag) or abs(left - right) > 1e-12:
                    message = "compiled instrument differs from declared parameters"
                    raise ValueError(message)
        result._amplitudes = saved
        result._refresh_probabilities()
        return result


def build_model(config: RepetitionConfig) -> RepetitionModel:
    """Construct the exact compact model for a supported repetition experiment."""
    return RepetitionModel(config)
