# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface-code Clifford-deformation metadata.

This module resolves source-level surface-code checks and logical operators
through a concrete local Clifford frame.  It intentionally stops before circuit
emission: renderers should consume the resolved Pauli checks instead of
guessing whether a frame can be represented by the legacy CSS X/Z helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.surface.patch import Stabilizer, SurfacePatch

PauliAxis = Literal["X", "Y", "Z"]
SurfaceFramePolicy = Literal[
    "identity",
    "global_h",
    "global_axis_cycle_f",
    "global_axis_cycle_f2",
]

_SUPPORTED_GLOBAL_FRAME_POLICIES = frozenset(
    {
        "identity",
        "global_h",
        "global_axis_cycle_f",
        "global_axis_cycle_f2",
    },
)


@dataclass(frozen=True, order=True)
class SignedPauli:
    """One signed single-qubit Pauli image."""

    axis: PauliAxis
    sign: int = 1

    def __post_init__(self) -> None:
        axis = str(self.axis).upper()
        if axis not in {"X", "Y", "Z"}:
            msg = f"Pauli axis must be 'X', 'Y', or 'Z', got {self.axis!r}"
            raise ValueError(msg)
        sign = int(self.sign)
        if sign not in {-1, 1}:
            msg = f"Pauli sign must be +/-1, got {self.sign!r}"
            raise ValueError(msg)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "sign", sign)

    def label(self) -> str:
        """Return a compact signed label such as ``X`` or ``-Y``."""
        return self.axis if self.sign > 0 else f"-{self.axis}"


@dataclass(frozen=True)
class LocalCliffordFrame:
    """Images of source X and Z under one local Clifford frame."""

    x_image: SignedPauli
    z_image: SignedPauli

    def image(self, source_axis: str) -> SignedPauli:
        """Return the signed physical Pauli image for source ``X`` or ``Z``."""
        axis = source_axis.upper()
        if axis == "X":
            return self.x_image
        if axis == "Z":
            return self.z_image
        msg = f"source_axis must be 'X' or 'Z', got {source_axis!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ResolvedPauliCheck:
    """A source stabilizer resolved to concrete physical Pauli axes."""

    source_kind: PauliAxis
    stabilizer_index: int
    data_qubits: tuple[int, ...]
    paulis: tuple[SignedPauli, ...]
    is_boundary: bool

    @property
    def axes(self) -> tuple[PauliAxis, ...]:
        """Physical Pauli axes in data-qubit order."""
        return tuple(pauli.axis for pauli in self.paulis)

    @property
    def signs(self) -> tuple[int, ...]:
        """Physical Pauli signs in data-qubit order."""
        return tuple(pauli.sign for pauli in self.paulis)

    @property
    def is_uniform_axis(self) -> bool:
        """Whether every data qubit is checked in the same physical axis."""
        return len(set(self.axes)) <= 1

    @property
    def uniform_axis(self) -> PauliAxis | None:
        """Return the uniform physical axis, or ``None`` for mixed checks."""
        if not self.is_uniform_axis or not self.axes:
            return None
        return self.axes[0]

    @property
    def requires_deformed_check_synthesis(self) -> bool:
        """Whether the legacy CSS helper cannot synthesize this check."""
        return self.uniform_axis not in {"X", "Z"} or not self.is_uniform_axis


@dataclass(frozen=True)
class ResolvedPauliLogical:
    """A source logical operator resolved to concrete physical Pauli axes."""

    source_kind: PauliAxis
    data_qubits: tuple[int, ...]
    paulis: tuple[SignedPauli, ...]

    @property
    def axes(self) -> tuple[PauliAxis, ...]:
        """Physical Pauli axes in data-qubit order."""
        return tuple(pauli.axis for pauli in self.paulis)

    @property
    def is_uniform_axis(self) -> bool:
        """Whether every data qubit is measured in the same physical axis."""
        return len(set(self.axes)) <= 1

    @property
    def uniform_axis(self) -> PauliAxis | None:
        """Return the uniform physical axis, or ``None`` for mixed logicals."""
        if not self.is_uniform_axis or not self.axes:
            return None
        return self.axes[0]


@dataclass(frozen=True)
class ResolvedSurfaceCliffordFrame:
    """Resolved source checks/logicals for one concrete surface-code frame."""

    policy: str
    data_frames: tuple[LocalCliffordFrame, ...]
    x_checks: tuple[ResolvedPauliCheck, ...]
    z_checks: tuple[ResolvedPauliCheck, ...]
    logical_x: ResolvedPauliLogical
    logical_z: ResolvedPauliLogical

    @property
    def checks(self) -> tuple[ResolvedPauliCheck, ...]:
        """All resolved checks in source X-then-Z order."""
        return (*self.x_checks, *self.z_checks)

    @property
    def requires_deformed_check_synthesis(self) -> bool:
        """Whether any check requires the generic deformed-check path."""
        return any(check.requires_deformed_check_synthesis for check in self.checks)

    def css_physical_memory_basis(self, source_basis: str) -> PauliAxis:
        """Return the physical X/Z basis if the CSS helper can represent this frame.

        This is intentionally stricter than asking only where the logical memory
        axis maps.  A global ``F`` frame maps source-Z memory to physical-X
        readout, but it also maps source-X stabilizers to physical-Y checks.
        Such a circuit is not representable by the current CSS helper and must
        use a deformed-check renderer.
        """
        basis = source_basis.upper()
        if basis == "X":
            logical_axis = self.logical_x.uniform_axis
        elif basis == "Z":
            logical_axis = self.logical_z.uniform_axis
        else:
            msg = f"source_basis must be 'X' or 'Z', got {source_basis!r}"
            raise ValueError(msg)

        if self.requires_deformed_check_synthesis:
            msg = (
                f"Frame policy {self.policy!r} requires deformed check synthesis "
                "and cannot be represented by the legacy CSS surface helper."
            )
            raise NotImplementedError(msg)
        if logical_axis not in {"X", "Z"}:
            msg = (
                f"Frame policy {self.policy!r} maps source {basis}-memory to "
                f"physical {logical_axis}; the CSS helper supports only X/Z."
            )
            raise NotImplementedError(msg)
        return logical_axis


def normalize_surface_frame_policy(policy: str) -> str:
    """Normalize and validate a named surface Clifford frame policy."""
    normalized = str(policy).lower().replace("-", "_")
    if normalized not in _SUPPORTED_GLOBAL_FRAME_POLICIES:
        msg = (
            f"unknown surface Clifford frame policy {policy!r}; expected one of "
            f"{sorted(_SUPPORTED_GLOBAL_FRAME_POLICIES)}"
        )
        raise ValueError(msg)
    return normalized


def global_surface_frame(policy: str, num_data: int) -> tuple[LocalCliffordFrame, ...]:
    """Return one of the supported parameter-free global frame maps."""
    if num_data < 0:
        msg = f"num_data must be non-negative, got {num_data}"
        raise ValueError(msg)
    normalized = normalize_surface_frame_policy(policy)
    frame = _global_frame_element(normalized)
    return tuple(frame for _ in range(num_data))


def resolve_surface_clifford_frame(
    patch: SurfacePatch,
    *,
    policy: str = "identity",
    data_frames: Sequence[LocalCliffordFrame] | None = None,
) -> ResolvedSurfaceCliffordFrame:
    """Resolve source surface checks/logicals through a local Clifford frame."""
    normalized = normalize_surface_frame_policy(policy)
    frames = tuple(data_frames) if data_frames is not None else global_surface_frame(normalized, patch.num_data)
    if len(frames) != patch.num_data:
        msg = f"data frame length {len(frames)} does not match patch.num_data={patch.num_data}"
        raise ValueError(msg)

    def resolve_check(stabilizer: Stabilizer) -> ResolvedPauliCheck:
        return ResolvedPauliCheck(
            source_kind=stabilizer.stab_type,  # type: ignore[arg-type]
            stabilizer_index=stabilizer.index,
            data_qubits=tuple(stabilizer.data_qubits),
            paulis=tuple(frames[q].image(stabilizer.stab_type) for q in stabilizer.data_qubits),
            is_boundary=stabilizer.is_boundary,
        )

    geom = patch.geometry
    if geom.logical_x is None or geom.logical_z is None:
        msg = "Surface patch must have both X and Z logical operators"
        raise ValueError(msg)

    return ResolvedSurfaceCliffordFrame(
        policy=normalized,
        data_frames=frames,
        x_checks=tuple(resolve_check(stabilizer) for stabilizer in patch.x_stabilizers),
        z_checks=tuple(resolve_check(stabilizer) for stabilizer in patch.z_stabilizers),
        logical_x=ResolvedPauliLogical(
            source_kind="X",
            data_qubits=tuple(geom.logical_x.data_qubits),
            paulis=tuple(frames[q].image("X") for q in geom.logical_x.data_qubits),
        ),
        logical_z=ResolvedPauliLogical(
            source_kind="Z",
            data_qubits=tuple(geom.logical_z.data_qubits),
            paulis=tuple(frames[q].image("Z") for q in geom.logical_z.data_qubits),
        ),
    )


def _global_frame_element(policy: str) -> LocalCliffordFrame:
    if policy == "identity":
        return LocalCliffordFrame(SignedPauli("X"), SignedPauli("Z"))
    if policy == "global_h":
        return LocalCliffordFrame(SignedPauli("Z"), SignedPauli("X"))
    if policy == "global_axis_cycle_f":
        return LocalCliffordFrame(SignedPauli("Y"), SignedPauli("X"))
    if policy == "global_axis_cycle_f2":
        return LocalCliffordFrame(SignedPauli("Z"), SignedPauli("Y"))
    msg = f"unknown surface Clifford frame policy {policy!r}"
    raise ValueError(msg)
