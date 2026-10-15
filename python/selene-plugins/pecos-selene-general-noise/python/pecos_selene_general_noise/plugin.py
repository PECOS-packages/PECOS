"""User-facing configuration for the PECOS general noise model."""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, cast

from selene_core import ErrorModel

if TYPE_CHECKING:
    from collections.abc import Callable

Distribution = Mapping[str, float]
ONE_QUBIT_PAULIS = frozenset({"X", "Y", "Z"})
ONE_QUBIT_CHANNELS = ONE_QUBIT_PAULIS | {"L"}


class _PecosGeneralNoiseBuilder(Protocol):
    """Structural type for the immutable Python PECOS noise builder."""

    def __getattr__(self, name: str) -> Callable[..., Self]: ...


def _frozen_distribution(value: Distribution | None) -> Distribution | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


def _json_value(value: object) -> object:
    """Convert frozen parameter values to ordinary JSON-compatible containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _probability(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        message = f"{name} must be a finite probability between 0 and 1"
        raise ValueError(message)


def _non_negative(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        message = f"{name} must be finite and non-negative"
        raise ValueError(message)


def _positive(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        message = f"{name} must be finite and positive"
        raise ValueError(message)


def _distribution(
    name: str,
    value: Distribution | None,
    allowed: frozenset[str],
    *,
    normalized: bool = True,
) -> None:
    if value is None:
        return
    if not value:
        message = f"{name} cannot be empty"
        raise ValueError(message)
    invalid = set(value) - allowed
    if invalid:
        message = f"{name} contains unsupported entries: {sorted(invalid)}"
        raise ValueError(message)
    for key, weight in value.items():
        _non_negative(f"{name}[{key!r}]", weight)
    if normalized and value and not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9):
        message = f"{name} weights must sum to 1"
        raise ValueError(message)


def _label_arity(labels: list[str]) -> int:
    arities = {len(label.strip()) for label in labels}
    if not arities or len(arities) != 1 or not arities <= {1, 2}:
        message = "hook labels must have a consistent arity of 1 or 2"
        raise ValueError(message)
    return arities.pop()


@dataclass(frozen=True)
class TransitionHook:
    """Plain transition data; Rust owns label and conditional-row validation."""

    probability: float
    transitions: Mapping[str, float | Mapping[str, float]]
    kind: str = field(default="transition", init=False)

    def __post_init__(self) -> None:
        """Freeze both flat maps and nested rows."""
        object.__setattr__(
            self,
            "transitions",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(value)) if isinstance(value, Mapping) else value
                    for key, value in self.transitions.items()
                },
            ),
        )

    def arity(self) -> int:
        """Infer arity without interpreting state labels."""
        labels = []
        for key, value in self.transitions.items():
            if isinstance(value, Mapping):
                labels.extend((key, *value))
            else:
                labels.extend(key.split("->"))
        return _label_arity(labels)

    def validate(self, arity: int = 2) -> None:
        """Check shape and outer probability for a gate site."""
        _probability("hook.probability", self.probability)
        if self.probability is None or self.kind != "transition" or self.arity() > arity:
            message = "invalid transition hook kind, probability, or arity"
            raise ValueError(message)


@dataclass(frozen=True)
class PauliLeakageHook:
    """Plain Pauli/leakage event data validated by the Rust channel constructor."""

    probability: float
    events: Distribution
    kind: str = field(default="pauli_leakage", init=False)

    def __post_init__(self) -> None:
        """Freeze the event distribution."""
        object.__setattr__(self, "events", _frozen_distribution(self.events))

    def arity(self) -> int:
        """Infer arity from event labels."""
        return _label_arity(list(self.events))

    def validate(self, arity: int = 2) -> None:
        """Check shape and outer probability for a gate site."""
        _probability("hook.probability", self.probability)
        if self.probability is None or self.kind != "pauli_leakage" or self.arity() > arity:
            message = "invalid Pauli-leakage hook kind, probability, or arity"
            raise ValueError(message)


@dataclass(frozen=True)
class IndependentHook:
    """Distinct one-qubit channels of the same family for the two operands."""

    first: TransitionHook | PauliLeakageHook
    second: TransitionHook | PauliLeakageHook
    kind: str = field(default="independent", init=False)

    def validate(self, arity: int = 2) -> None:
        """Require two one-qubit channels from the same family."""
        if (
            arity != 2
            or self.kind != "independent"
            or not isinstance(self.first, (TransitionHook, PauliLeakageHook))
            or type(self.first) is not type(self.second)
        ):
            message = "independent hook requires arity 2 and one family"
            raise ValueError(message)
        self.first.validate(1)
        self.second.validate(1)


GateHook = TransitionHook | PauliLeakageHook | IndependentHook


def _hook_family(hook: GateHook) -> str:
    return hook.first.kind if isinstance(hook, IndependentHook) else hook.kind


@dataclass(frozen=True)
class PreparationNoise:
    """Preparation faults and all-to-all preparation crosstalk."""

    probability: float | None = None
    leakage_ratio: float | None = None
    crosstalk_probability: float | None = None
    average_crosstalk_probability: float | None = None
    scale: float | None = None
    crosstalk_scale: float | None = None

    def validate(self) -> None:
        """Validate preparation parameters."""
        _probability("preparation.probability", self.probability)
        _probability("preparation.leakage_ratio", self.leakage_ratio)
        _probability("preparation.crosstalk_probability", self.crosstalk_probability)
        _probability("preparation.average_crosstalk_probability", self.average_crosstalk_probability)
        if self.crosstalk_probability is not None and self.average_crosstalk_probability is not None:
            message = "preparation accepts crosstalk_probability or average_crosstalk_probability, not both"
            raise ValueError(message)
        if self.average_crosstalk_probability is not None and self.average_crosstalk_probability > 5.0 / 18.0:
            message = "preparation.average_crosstalk_probability cannot exceed 5/18"
            raise ValueError(message)
        _non_negative("preparation.scale", self.scale)
        _non_negative("preparation.crosstalk_scale", self.crosstalk_scale)


@dataclass(frozen=True)
class GateNoise:
    """Noise shared by one- and two-qubit gates.

    Set either ``probability`` (process infidelity) or ``average_infidelity``.
    """

    probability: float | None = None
    average_infidelity: float | None = None
    pauli_model: Distribution | None = None
    emission_ratio: float | None = None
    emission_model: Distribution | None = None
    seepage_probability: float | None = None
    scale: float | None = None

    hooks_before: tuple[GateHook, ...] = ()
    hooks_after: tuple[GateHook, ...] = ()

    _paulis: ClassVar[frozenset[str]] = ONE_QUBIT_PAULIS
    _emissions: ClassVar[frozenset[str]] = ONE_QUBIT_CHANNELS

    def __post_init__(self) -> None:
        """Defensively freeze user-supplied channel distributions."""
        object.__setattr__(self, "hooks_before", tuple(self.hooks_before))
        object.__setattr__(self, "hooks_after", tuple(self.hooks_after))
        object.__setattr__(self, "pauli_model", _frozen_distribution(self.pauli_model))
        object.__setattr__(self, "emission_model", _frozen_distribution(self.emission_model))

    def validate(self, name: str = "single_qubit") -> None:
        """Validate gate-channel parameters."""
        if self.probability is not None and self.average_infidelity is not None:
            message = f"{name} accepts probability or average_infidelity, not both"
            raise ValueError(message)
        for hook in (*self.hooks_before, *self.hooks_after):
            if not isinstance(hook, (TransitionHook, PauliLeakageHook, IndependentHook)):
                message = "unknown gate hook kind"
                raise TypeError(message)
            hook.validate(2 if isinstance(self, TwoQubitGateNoise) else 1)
        _probability(f"{name}.probability", self.probability)
        _probability(f"{name}.average_infidelity", self.average_infidelity)
        _probability(f"{name}.emission_ratio", self.emission_ratio)
        _probability(f"{name}.seepage_probability", self.seepage_probability)
        _non_negative(f"{name}.scale", self.scale)
        _distribution(f"{name}.pauli_model", self.pauli_model, self._paulis)
        _distribution(f"{name}.emission_model", self.emission_model, self._emissions)
        if name == "single_qubit" and self.average_infidelity is not None and self.average_infidelity > 2.0 / 3.0:
            message = "single_qubit.average_infidelity cannot exceed 2/3"
            raise ValueError(message)


@dataclass(frozen=True)
class TwoQubitGateNoise(GateNoise):
    """Two-qubit gate noise, including optional angle-dependent scaling."""

    angle_coefficients: tuple[float, float, float, float] | None = None
    angle_power: float | None = None
    idle_after_gate: float | None = None

    _paulis: ClassVar[frozenset[str]] = frozenset(a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II")
    _emissions: ClassVar[frozenset[str]] = frozenset(a + b for a in "IXYZL" for b in "IXYZL" if a + b != "II")

    def validate(self, name: str = "two_qubit") -> None:
        """Validate two-qubit channel parameters."""
        super().validate(name)
        if self.angle_coefficients is not None and (
            len(self.angle_coefficients) != 4 or not all(math.isfinite(value) for value in self.angle_coefficients)
        ):
            message = "two_qubit.angle_coefficients must contain four finite values"
            raise ValueError(message)
        if self.average_infidelity is not None and self.average_infidelity > 0.8:
            message = "two_qubit.average_infidelity cannot exceed 0.8"
            raise ValueError(message)
        _positive("two_qubit.angle_power", self.angle_power)
        _non_negative("two_qubit.idle_after_gate", self.idle_after_gate)


@dataclass(frozen=True)
class IdleNoise:
    """Time-dependent idle channels; rates use Selene's seconds-based schedule."""

    linear_rate: float | None = None
    linear_model: Distribution | None = None
    sin_squared_rate: float | None = None
    sin_squared_model: Distribution | None = None
    coherent_rate: float | None = None
    coherent_model: Distribution | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        """Defensively freeze user-supplied idle channel models."""
        for name in ("linear_model", "sin_squared_model", "coherent_model"):
            object.__setattr__(self, name, _frozen_distribution(getattr(self, name)))

    def validate(self) -> None:
        """Validate idle-channel parameters."""
        for name in ("linear_rate", "sin_squared_rate", "coherent_rate", "scale"):
            _non_negative(f"idle.{name}", getattr(self, name))
        _distribution("idle.linear_model", self.linear_model, ONE_QUBIT_CHANNELS)
        _distribution(
            "idle.sin_squared_model",
            self.sin_squared_model,
            ONE_QUBIT_CHANNELS,
            normalized=False,
        )
        _distribution(
            "idle.coherent_model",
            self.coherent_model,
            frozenset({"RX", "RY", "RZ"}),
            normalized=False,
        )
        for family in ("linear", "sin_squared", "coherent"):
            if getattr(self, f"{family}_model") is not None and getattr(self, f"{family}_rate") is None:
                message = f"idle.{family}_model requires idle.{family}_rate"
                raise ValueError(message)


@dataclass(frozen=True)
class MeasurementNoise:
    """Readout noise and optional topology-aware measurement crosstalk.

    ``local_groups`` describes device-neutral neighborhoods. Each qubit in a group
    is local to the other members. Global crosstalk applies to every prepared qubit
    outside the measured set.
    """

    p0_to_1: float | None = None
    p1_to_0: float | None = None
    global_crosstalk_probability: float | None = None
    local_crosstalk_probability: float | None = None
    crosstalk_model: Distribution | None = None
    local_groups: tuple[tuple[int, ...], ...] = ()
    scale: float | None = None
    crosstalk_scale: float | None = None

    def __post_init__(self) -> None:
        """Defensively freeze transition models and topology groups."""
        object.__setattr__(self, "crosstalk_model", _frozen_distribution(self.crosstalk_model))
        object.__setattr__(self, "local_groups", tuple(tuple(group) for group in self.local_groups))

    def validate(self) -> None:
        """Validate measurement-channel parameters."""
        for name in (
            "p0_to_1",
            "p1_to_0",
            "global_crosstalk_probability",
            "local_crosstalk_probability",
        ):
            _probability(f"measurement.{name}", getattr(self, name))
        _non_negative("measurement.scale", self.scale)
        _non_negative("measurement.crosstalk_scale", self.crosstalk_scale)
        transitions = self.crosstalk_model
        _distribution(
            "measurement.crosstalk_model",
            transitions,
            frozenset({"0->0", "0->1", "0->L", "1->0", "1->1", "1->L"}),
            normalized=False,
        )
        if transitions is not None:
            for source in ("0", "1"):
                total = sum(weight for key, weight in transitions.items() if key.startswith(f"{source}->"))
                if not math.isclose(total, 1.0, abs_tol=1e-9):
                    message = f"measurement.crosstalk_model {source}->* weights must sum to 1"
                    raise ValueError(message)
        if any(qubit < 0 for group in self.local_groups for qubit in group):
            message = "measurement.local_groups cannot contain negative qubit indices"
            raise ValueError(message)


@dataclass(frozen=True)
class NoiseScaling:
    """Global controls that apply across noise families."""

    overall: float | None = None
    leakage: float | None = None
    emission: float | None = None

    def validate(self) -> None:
        """Validate scale factors."""
        _non_negative("scaling.overall", self.overall)
        _probability("scaling.leakage", self.leakage)
        _non_negative("scaling.emission", self.emission)


@dataclass(frozen=True)
class GeneralNoiseParameters:
    """Immutable, fluent configuration for PECOS ``GeneralNoiseModel``."""

    preparation: PreparationNoise = field(default_factory=PreparationNoise)
    measurement: MeasurementNoise = field(default_factory=MeasurementNoise)
    single_qubit: GateNoise = field(default_factory=GateNoise)
    two_qubit: TwoQubitGateNoise = field(default_factory=TwoQubitGateNoise)
    idle: IdleNoise = field(default_factory=IdleNoise)
    scaling: NoiseScaling = field(default_factory=NoiseScaling)
    noiseless_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the complete configuration after construction."""
        object.__setattr__(self, "noiseless_gates", tuple(self.noiseless_gates))
        self.preparation.validate()
        self.measurement.validate()
        self.single_qubit.validate()
        self.two_qubit.validate()
        self.idle.validate()
        self.scaling.validate()

    def validate_effective_probabilities(self) -> None:
        """Reject scale combinations that leave PECOS's probability domain."""
        overall = self.scaling.overall if self.scaling.overall is not None else 1.0
        prep_scale = self.preparation.scale if self.preparation.scale is not None else 1.0
        meas_scale = self.measurement.scale if self.measurement.scale is not None else 1.0
        p1_scale = self.single_qubit.scale if self.single_qubit.scale is not None else 1.0
        p2_scale = self.two_qubit.scale if self.two_qubit.scale is not None else 1.0
        prep_crosstalk_scale = self.preparation.crosstalk_scale if self.preparation.crosstalk_scale is not None else 1.0
        meas_crosstalk_scale = self.measurement.crosstalk_scale if self.measurement.crosstalk_scale is not None else 1.0

        def require(name: str, probability: float) -> None:
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                message = f"effective {name} probability must be between 0 and 1, got {probability}"
                raise ValueError(message)

        for gate, scale in ((self.single_qubit, p1_scale), (self.two_qubit, p2_scale)):
            for hook in (*gate.hooks_before, *gate.hooks_after):
                legs = (hook.first, hook.second) if isinstance(hook, IndependentHook) else (hook,)
                for leg in legs:
                    require("gate hook", leg.probability * scale * overall)

        prep = self.preparation.probability or 0.0
        meas_0 = self.measurement.p0_to_1 or 0.0
        meas_1 = self.measurement.p1_to_0 or 0.0
        p1 = self.single_qubit.probability
        if p1 is None:
            p1 = (self.single_qubit.average_infidelity or 0.0) * 3.0 / 2.0
        p2 = self.two_qubit.probability
        if p2 is None:
            p2 = (self.two_qubit.average_infidelity or 0.0) * 5.0 / 4.0
        prep_crosstalk = self.preparation.crosstalk_probability
        if prep_crosstalk is None:
            prep_crosstalk = (self.preparation.average_crosstalk_probability or 0.0) * 18.0 / 5.0

        require("preparation", prep * prep_scale * overall)
        require("measurement 0 -> 1", meas_0 * meas_scale * overall)
        require("measurement 1 -> 0", meas_1 * meas_scale * overall)
        require("single-qubit gate", p1 * p1_scale * overall)
        scaled_p2 = p2 * p2_scale * overall
        require("two-qubit gate", scaled_p2)
        require("preparation crosstalk", prep_crosstalk * prep_crosstalk_scale * prep_scale * overall)
        for name, probability in (
            ("global measurement crosstalk", self.measurement.global_crosstalk_probability or 0.0),
            ("local measurement crosstalk", self.measurement.local_crosstalk_probability or 0.0),
        ):
            require(name, probability * meas_crosstalk_scale * meas_scale * overall)

        a, b, c, d = self.two_qubit.angle_coefficients or (0.0, 1.0, 0.0, 1.0)
        for name, multiplier in (
            ("negative-angle two-qubit offset", b),
            ("negative-angle two-qubit endpoint", a + b),
            ("positive-angle two-qubit offset", d),
            ("positive-angle two-qubit endpoint", c + d),
            ("zero-angle two-qubit", (b + d) * 0.5),
        ):
            require(name, scaled_p2 * multiplier)

    @classmethod
    def uniform(cls, probability: float) -> GeneralNoiseParameters:
        """Create a simple device-neutral model with one common error probability."""
        _probability("probability", probability)
        return cls(
            preparation=PreparationNoise(probability=probability),
            measurement=MeasurementNoise(p0_to_1=probability, p1_to_0=probability),
            single_qubit=GateNoise(probability=probability),
            two_qubit=TwoQubitGateNoise(probability=probability),
        )

    def _with_preparation(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, preparation=replace(self.preparation, **changes))

    def _with_measurement(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, measurement=replace(self.measurement, **changes))

    def _with_single_qubit(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, single_qubit=replace(self.single_qubit, **changes))

    def _with_two_qubit(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, two_qubit=replace(self.two_qubit, **changes))

    def _with_idle(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, idle=replace(self.idle, **changes))

    def _with_scaling(self, **changes: object) -> GeneralNoiseParameters:
        return replace(self, scaling=replace(self.scaling, **changes))

    def _with_hooks(
        self,
        site: str,
        timing: str,
        family: str,
        hooks: tuple[GateHook, ...],
        *,
        append: bool,
    ) -> GeneralNoiseParameters:
        gate = getattr(self, site)
        for hook in hooks:
            if not isinstance(hook, (TransitionHook, PauliLeakageHook, IndependentHook)):
                message = "unknown gate hook kind"
                raise TypeError(message)
            hook.validate(1 if site == "single_qubit" else 2)
            if _hook_family(hook) != family:
                message = f"expected {family} hook"
                raise ValueError(message)
        current = getattr(gate, timing)
        retained = current if append else tuple(hook for hook in current if _hook_family(hook) != family)
        return replace(self, **{site: replace(gate, **{timing: (*retained, *hooks)})})

    def with_p1_transition_channels_before_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("single_qubit", "hooks_before", "transition", tuple(hooks), append=False)

    def add_p1_transition_channel_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("single_qubit", "hooks_before", "transition", (hook,), append=True)

    def with_p1_transition_channels_after_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("single_qubit", "hooks_after", "transition", tuple(hooks), append=False)

    def add_p1_transition_channel_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("single_qubit", "hooks_after", "transition", (hook,), append=True)

    def with_p1_pauli_leakage_channels_before_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("single_qubit", "hooks_before", "pauli_leakage", tuple(hooks), append=False)

    def add_p1_pauli_leakage_channel_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("single_qubit", "hooks_before", "pauli_leakage", (hook,), append=True)

    def with_p1_pauli_leakage_channels_after_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("single_qubit", "hooks_after", "pauli_leakage", tuple(hooks), append=False)

    def add_p1_pauli_leakage_channel_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("single_qubit", "hooks_after", "pauli_leakage", (hook,), append=True)

    def with_p2_transition_steps_before_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("two_qubit", "hooks_before", "transition", tuple(hooks), append=False)

    def add_p2_transition_channel_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_before", "transition", (hook,), append=True)

    def add_p2_transition_step_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_before", "transition", (hook,), append=True)

    def with_p2_transition_steps_after_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("two_qubit", "hooks_after", "transition", tuple(hooks), append=False)

    def add_p2_transition_channel_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_after", "transition", (hook,), append=True)

    def add_p2_transition_step_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_after", "transition", (hook,), append=True)

    def with_p2_pauli_leakage_steps_before_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("two_qubit", "hooks_before", "pauli_leakage", tuple(hooks), append=False)

    def add_p2_pauli_leakage_channel_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_before", "pauli_leakage", (hook,), append=True)

    def add_p2_pauli_leakage_step_before_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_before", "pauli_leakage", (hook,), append=True)

    def with_p2_pauli_leakage_steps_after_gate(self, hooks: tuple[GateHook, ...]) -> GeneralNoiseParameters:
        """Replace this family's hooks, appending after the retained other family."""
        return self._with_hooks("two_qubit", "hooks_after", "pauli_leakage", tuple(hooks), append=False)

    def add_p2_pauli_leakage_channel_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_after", "pauli_leakage", (hook,), append=True)

    def add_p2_pauli_leakage_step_after_gate(self, hook: GateHook) -> GeneralNoiseParameters:
        """Append a hook in execution order."""
        return self._with_hooks("two_qubit", "hooks_after", "pauli_leakage", (hook,), append=True)

    def with_noiseless_gate(self, gate: str) -> GeneralNoiseParameters:
        """Add a gate parsed by PECOS to the noiseless gate set."""
        return replace(self, noiseless_gates=(*self.noiseless_gates, gate))

    def with_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the overall noise scale."""
        return self._with_scaling(overall=scale)

    def with_leakage_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the fraction of leakage events retained as leakage."""
        return self._with_scaling(leakage=scale)

    def with_emission_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the spontaneous-emission scale."""
        return self._with_scaling(emission=scale)

    def with_seepage_prob(self, probability: float) -> GeneralNoiseParameters:
        """Set both one- and two-qubit seepage probabilities."""
        return replace(
            self,
            single_qubit=replace(self.single_qubit, seepage_probability=probability),
            two_qubit=replace(self.two_qubit, seepage_probability=probability),
        )

    def with_p_idle_linear(self, rate: float, model: Distribution) -> GeneralNoiseParameters:
        """Set linear stochastic idle noise."""
        return self._with_idle(linear_rate=rate, linear_model=dict(model))

    def with_p_idle_sin_squared(self, rate: float, model: Distribution) -> GeneralNoiseParameters:
        """Set independent sine-squared idle channels."""
        return self._with_idle(sin_squared_rate=rate, sin_squared_model=dict(model))

    def with_p_idle_coherent(self, rate: float, model: Distribution) -> GeneralNoiseParameters:
        """Set coherent idle rotations."""
        return self._with_idle(coherent_rate=rate, coherent_model=dict(model))

    def with_idle_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the idle-noise scale."""
        return self._with_idle(scale=scale)

    def with_p_prep(self, probability: float) -> GeneralNoiseParameters:
        """Set preparation fault probability."""
        return self._with_preparation(probability=probability)

    def with_prep_leak_ratio(self, ratio: float) -> GeneralNoiseParameters:
        """Set the preparation leakage ratio."""
        return self._with_preparation(leakage_ratio=ratio)

    def with_p_prep_crosstalk(self, probability: float) -> GeneralNoiseParameters:
        """Set preparation crosstalk process probability."""
        return self._with_preparation(
            crosstalk_probability=probability,
            average_crosstalk_probability=None,
        )

    def with_average_p_prep_crosstalk(self, probability: float) -> GeneralNoiseParameters:
        """Set preparation crosstalk average probability."""
        return self._with_preparation(
            crosstalk_probability=None,
            average_crosstalk_probability=probability,
        )

    def with_prep_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the preparation-noise scale."""
        return self._with_preparation(scale=scale)

    def with_p_prep_crosstalk_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the preparation-crosstalk scale."""
        return self._with_preparation(crosstalk_scale=scale)

    def with_p1(self, probability: float) -> GeneralNoiseParameters:
        """Set one-qubit process infidelity."""
        return self._with_single_qubit(probability=probability, average_infidelity=None)

    def with_average_p1(self, probability: float) -> GeneralNoiseParameters:
        """Set one-qubit average infidelity."""
        return self._with_single_qubit(probability=None, average_infidelity=probability)

    def with_p1_emission_ratio(self, ratio: float) -> GeneralNoiseParameters:
        """Set the one-qubit spontaneous-emission ratio."""
        return self._with_single_qubit(emission_ratio=ratio)

    def with_p1_emission_model(self, model: Distribution) -> GeneralNoiseParameters:
        """Set the one-qubit emission distribution."""
        return self._with_single_qubit(emission_model=dict(model))

    def with_p1_seepage_prob(self, probability: float) -> GeneralNoiseParameters:
        """Set one-qubit seepage probability."""
        return self._with_single_qubit(seepage_probability=probability)

    def with_p1_pauli_model(self, model: Distribution) -> GeneralNoiseParameters:
        """Set the one-qubit Pauli distribution."""
        return self._with_single_qubit(pauli_model=dict(model))

    def with_p1_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the one-qubit noise scale."""
        return self._with_single_qubit(scale=scale)

    def with_p2(self, probability: float) -> GeneralNoiseParameters:
        """Set two-qubit process infidelity."""
        return self._with_two_qubit(probability=probability, average_infidelity=None)

    def with_average_p2(self, probability: float) -> GeneralNoiseParameters:
        """Set two-qubit average infidelity."""
        return self._with_two_qubit(probability=None, average_infidelity=probability)

    def with_p2_angle_params(self, a: float, b: float, c: float, d: float) -> GeneralNoiseParameters:
        """Set signed angle-dependent two-qubit coefficients."""
        return self._with_two_qubit(angle_coefficients=(a, b, c, d))

    def with_p2_angle_power(self, power: float) -> GeneralNoiseParameters:
        """Set the angle-dependent two-qubit exponent."""
        return self._with_two_qubit(angle_power=power)

    def with_p2_emission_ratio(self, ratio: float) -> GeneralNoiseParameters:
        """Set the two-qubit spontaneous-emission ratio."""
        return self._with_two_qubit(emission_ratio=ratio)

    def with_p2_emission_model(self, model: Distribution) -> GeneralNoiseParameters:
        """Set the two-qubit emission distribution."""
        return self._with_two_qubit(emission_model=dict(model))

    def with_p2_seepage_prob(self, probability: float) -> GeneralNoiseParameters:
        """Set two-qubit seepage probability."""
        return self._with_two_qubit(seepage_probability=probability)

    def with_p2_pauli_model(self, model: Distribution) -> GeneralNoiseParameters:
        """Set the two-qubit Pauli distribution."""
        return self._with_two_qubit(pauli_model=dict(model))

    def with_idle_after_2q(self, duration: float) -> GeneralNoiseParameters:
        """Set the idle duration applied after two-qubit gates."""
        return self._with_two_qubit(idle_after_gate=duration)

    def with_p2_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the two-qubit noise scale."""
        return self._with_two_qubit(scale=scale)

    def with_p_meas_0(self, probability: float) -> GeneralNoiseParameters:
        """Set readout probability for 0 becoming 1."""
        return self._with_measurement(p0_to_1=probability)

    def with_p_meas_1(self, probability: float) -> GeneralNoiseParameters:
        """Set readout probability for 1 becoming 0."""
        return self._with_measurement(p1_to_0=probability)

    def with_p_meas(self, probability: float) -> GeneralNoiseParameters:
        """Set symmetric readout probability."""
        return self._with_measurement(p0_to_1=probability, p1_to_0=probability)

    def with_p_meas_crosstalk_global(self, probability: float) -> GeneralNoiseParameters:
        """Set global measurement-crosstalk probability."""
        return self._with_measurement(global_crosstalk_probability=probability)

    def with_p_meas_crosstalk_local(self, probability: float) -> GeneralNoiseParameters:
        """Set local measurement-crosstalk probability."""
        return self._with_measurement(local_crosstalk_probability=probability)

    def with_p_meas_crosstalk(self, probability: float) -> GeneralNoiseParameters:
        """Set both global and local measurement-crosstalk probabilities."""
        return self._with_measurement(
            global_crosstalk_probability=probability,
            local_crosstalk_probability=probability,
        )

    def with_p_meas_crosstalk_model(self, model: Distribution) -> GeneralNoiseParameters:
        """Set measurement-crosstalk transition distributions."""
        return self._with_measurement(crosstalk_model=dict(model))

    def with_meas_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the measurement-noise scale."""
        return self._with_measurement(scale=scale)

    def with_p_meas_crosstalk_scale(self, scale: float) -> GeneralNoiseParameters:
        """Set the measurement-crosstalk scale."""
        return self._with_measurement(crosstalk_scale=scale)

    def with_local_crosstalk_groups(self, *groups: tuple[int, ...]) -> GeneralNoiseParameters:
        """Set device-neutral local-crosstalk neighborhoods for the Selene adapter."""
        return self._with_measurement(local_groups=tuple(tuple(group) for group in groups))

    def apply_to_pecos_builder(
        self,
        builder: _PecosGeneralNoiseBuilder,
        *,
        seed: int | None = None,
    ) -> _PecosGeneralNoiseBuilder:
        """Apply this configuration to a PECOS ``GeneralNoiseModelBuilder``.

        This is the canonical bridge for running the same general-noise parameters
        through PECOS ``sim()`` and through :class:`GeneralNoisePlugin`. The returned
        builder must be used because PECOS builders are immutable.

        Selene ``local_groups`` are intentionally not applied: they describe how the
        Selene runtime adapter turns a device topology into PECOS crosstalk payloads,
        rather than a parameter of ``GeneralNoiseModel`` itself. End-to-end crosstalk
        parity therefore also requires both runtimes to emit equivalent payloads.
        """
        self.validate_effective_probabilities()
        builder = builder.with_config_json(
            json.dumps(_model_json_value(self), separators=(",", ":"), sort_keys=True),
        )
        if seed is not None:
            if seed < 0:
                message = "seed must be non-negative"
                raise ValueError(message)
            builder = builder.with_seed(seed)
        return builder

    def to_pecos_builder(self, *, seed: int | None = None) -> _PecosGeneralNoiseBuilder:
        """Create a PECOS builder configured from these parameters.

        ``quantum-pecos`` is an optional runtime dependency of this plugin package;
        install it when using this convenience method.
        """
        try:
            pecos = import_module("pecos")
        except ImportError as error:
            message = "to_pecos_builder() requires the quantum-pecos package"
            raise ImportError(message) from error
        general_noise = cast("Callable[[], _PecosGeneralNoiseBuilder]", pecos.general_noise)
        return self.apply_to_pecos_builder(general_noise(), seed=seed)

    def __pecos_noise_builder__(self) -> _PecosGeneralNoiseBuilder:
        """Convert this configuration for direct use by ``sim(...).noise(...)``.

        ``local_groups`` belongs to the Selene adapter rather than PECOS's
        ``GeneralNoiseModel``. Reject it on the direct path instead of silently
        discarding topology that the user may expect ``sim()`` to apply.
        """
        if self.measurement.local_groups:
            message = (
                "measurement.local_groups is Selene-adapter topology and cannot be applied by "
                "sim().noise(GeneralNoiseParameters); pass the parameters to GeneralNoisePlugin "
                "or remove local_groups"
            )
            raise ValueError(message)
        return self.to_pecos_builder()


def _model_json_value(parameters: GeneralNoiseParameters) -> dict[str, object]:
    """Return the engine parameters without Selene's local topology."""
    value = cast("dict[str, object]", _json_value(parameters))
    measurement = cast("dict[str, object]", value["measurement"])
    del measurement["local_groups"]
    return value


@dataclass
class GeneralNoisePlugin(ErrorModel):
    """Thin Selene wrapper around PECOS general-noise parameters."""

    parameters: GeneralNoiseParameters = field(default_factory=GeneralNoiseParameters)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate Selene-owned plugin options."""
        if self.random_seed is not None and self.random_seed < 0:
            message = "random_seed must be non-negative"
            raise ValueError(message)
        self.parameters.validate_effective_probabilities()

    def get_init_args(self) -> list[str]:
        """Serialize the complete configuration as one versionable JSON argument."""
        self.parameters.validate_effective_probabilities()
        payload = {
            "model": _model_json_value(self.parameters),
            "local_groups": [list(group) for group in self.parameters.measurement.local_groups],
        }
        return [json.dumps(payload, separators=(",", ":"), sort_keys=True)]

    @property
    def library_file(self) -> Path:
        """Return the bundled native Selene plugin library."""
        libdir = Path(__file__).parent / "_dist" / "lib"
        system = platform.system().lower()
        if system == "darwin":
            patterns = ["libpecos_selene_general_noise*.dylib"]
        elif system == "windows":
            patterns = ["pecos_selene_general_noise*.dll", "pecos_selene_general_noise*.pyd"]
        else:
            patterns = ["libpecos_selene_general_noise*.so"]
        for pattern in patterns:
            if matches := sorted(libdir.glob(pattern)):
                return matches[0]
        message = f"Could not find PECOS general-noise library in {libdir}"
        raise FileNotFoundError(message)
