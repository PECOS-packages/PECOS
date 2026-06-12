"""Configuration objects for Pauli-frame twirling.

`TwirlConfig` carries the twirl-site declaration: scheme, where in the
circuit twirling sites are emitted, how the per-shot mask is encoded into
the runtime result bundle after the corresponding physical Pauli gates
are applied, and how generated Guppy measurement records are framed. The
first three fields are structural for abstract DEM / topology caches.
`frame_output` is runtime-only: raw and canonical Guppy records share the
same abstract DEM and `PauliFrameLookup`.

`GuppyRngMaskConfig` carries the **runtime** mask source: a stream-separator
seed mixed with 32 bits of per-shot quantum entropy when the mask is drawn,
applied to data qubits, and recorded via `result()`. Two abstract circuits
identical except for `seed` or `frame_output` reuse the same DEM but produce
different shot-level runtime records, so those values belong in the Guppy-
module / compiled-shot cache layer but NOT in the abstract DEM cache.

The split mirrors the two-tracks-per-twirl-setting architecture from the
design doc: the abstract circuit (consumer: DEM builder,
`PauliFrameLookup`, decoder structure) consumes `TwirlConfig`; the Guppy
module (consumer: Selene runtime) consumes `TwirlConfig` AND
`GuppyRngMaskConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_SUPPORTED_SCHEMES = ("pauli",)
_SUPPORTED_SITE_SCHEDULES = ("between_rounds",)
_SUPPORTED_RESULT_ENCODINGS = ("bool_array_v1",)
_SUPPORTED_FRAME_OUTPUTS = ("raw", "canonical")


@dataclass(frozen=True)
class TwirlConfig:
    """Structural Pauli-twirl-site declaration.

    All fields are constrained to the values Phase 0a currently supports.
    Future Phase 2 (Clifford twirling) work will extend the `scheme` enum.

    Attributes:
        scheme: Twirling family. Phase 0a supports `"pauli"`; the
            `"clifford"` value is reserved for Phase 2 ({I, H}
            Clifford-frame randomization) and not yet implemented.
        site_schedule: Where twirling sites are emitted in the circuit.
            `"between_rounds"` (the only supported value) emits one site
            between each pair of consecutive syndrome rounds.
        result_encoding: How the per-shot mask is recorded in the
            runtime result bundle. `"bool_array_v1"` packs the
            `2 * num_data` bool bits per round into one tagged array per
            twirl site (the only supported encoding -- the earlier
            shared-tag scalar-bool variant is unimplementable because
            `ShotVec.to_dict()` collapses repeated same-tag calls to the
            last value).
        frame_output: Runtime Guppy measurement-frame convention.
            `"raw"` preserves the landed behavior: measurement tags are
            emitted in the physical/twirled frame and callers can
            canonicalize with `PauliFrameLookup`. `"canonical"` makes the
            generated Guppy program track the Pauli frame classically and
            flip emitted measurement bits into the canonical untwirled DEM
            frame. This does not change the abstract circuit or DEM
            topology; it only changes generated runtime records and must
            therefore be part of the Guppy module cache key.
    """

    scheme: Literal["pauli"] = "pauli"
    site_schedule: Literal["between_rounds"] = "between_rounds"
    result_encoding: Literal["bool_array_v1"] = "bool_array_v1"
    frame_output: Literal["raw", "canonical"] = "raw"

    def validate_runtime_supported(self) -> None:
        """Raise ``ValueError`` if any field is outside the supported runtime set.

        The ``Literal`` annotations are static-only hints; this method
        enforces them at runtime so e.g.
        ``TwirlConfig(result_encoding="bool_scalar_v1")`` (constructed via
        ``object.__setattr__`` or a stale call path) fails loudly at the
        Guppy / harvest boundary rather than silently producing
        encoding-incompatible behavior.
        """
        if self.scheme not in _SUPPORTED_SCHEMES:
            msg = (
                f"TwirlConfig.scheme={self.scheme!r} is not supported; "
                f"expected one of {_SUPPORTED_SCHEMES!r}"
            )
            raise ValueError(msg)
        if self.site_schedule not in _SUPPORTED_SITE_SCHEDULES:
            msg = (
                f"TwirlConfig.site_schedule={self.site_schedule!r} is not "
                f"supported; expected one of {_SUPPORTED_SITE_SCHEDULES!r}"
            )
            raise ValueError(msg)
        if self.result_encoding not in _SUPPORTED_RESULT_ENCODINGS:
            msg = (
                f"TwirlConfig.result_encoding={self.result_encoding!r} is "
                f"not supported; expected one of {_SUPPORTED_RESULT_ENCODINGS!r}"
            )
            raise ValueError(msg)
        if self.frame_output not in _SUPPORTED_FRAME_OUTPUTS:
            msg = (
                f"TwirlConfig.frame_output={self.frame_output!r} is not "
                f"supported; expected one of {_SUPPORTED_FRAME_OUTPUTS!r}"
            )
            raise ValueError(msg)

    def _validate_runtime_supported(self) -> None:
        """Compatibility alias for the public runtime validator."""
        self.validate_runtime_supported()


@dataclass(frozen=True)
class GuppyRngMaskConfig:
    """Runtime Guppy-side mask source.

    The seed separates mask streams. Generated Guppy programs mix it with
    32 measured H-basis entropy bits per shot before drawing the per-shot
    mask at quantum runtime. Excluded from the abstract DEM / topology
    cache key by construction: changing the seed must NOT invalidate the
    abstract circuit's DEM, only the compiled Guppy module + per-shot mask
    buffer.

    The Selene harvest helpers also pass this seed to the Stim simulator
    so mask draws are reproducible in tests. Studies that need mask-stream
    and syndrome-noise randomness to vary independently should expose a
    separate simulator seed.

    Attributes:
        seed: 64-bit unsigned stream-separator seed. Must be representable
            as a 64-bit unsigned integer.
    """

    seed: int
