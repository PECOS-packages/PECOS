"""Supplied, independently closed DEM windows for live QIS decoding.

This module supplies the checked provider for ``selene_engine().runtime_decoder``.
It does not infer models from a realized branch or handle corrections crossing
calls. A window may itself be composed from cached slices and use a windowed
decoder internally through the existing ``DetectorErrorModel.build_decoder`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from pecos_rslib.qec import DetectorErrorModel

from pecos.qec.runtime_decode_context import RuntimeDecodeBinding, RuntimeDecodeContext
from pecos.qec.runtime_decoder_batches import RuntimeDecodeBatchPrefix
from pecos.qec.runtime_observed_reads import check_observed_reads, validate_read_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pecos.decoders import DecoderSpec
    from pecos.qec.runtime_observed_reads import RuntimeReadSource


@dataclass(frozen=True)
class RuntimeDecodeRequest:
    """Public window identity, shot-local occurrence and optional caller context."""

    window_id: int
    instance_id: int
    context: RuntimeDecodeContext | None = None


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeBatchRequest(RuntimeDecodeRequest):
    """Observed-decode request with a host-captured executed batch interval.

    ``context`` still contains declarations; batches do not authenticate its
    compiler sites, branches, loops, dependencies or physical model.
    """

    batch_prefix: RuntimeDecodeBatchPrefix


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeWindow:
    """A caller-certified independent, closed physical fault region.

    ``detector_ids`` and ``output_ids`` name the model's dense D/L columns.
    ``detector_order[i]`` selects the program byte for model detector i.
    ``output_map[i]`` routes model L_i by XOR into program frame bits 0..63;
    an empty row explicitly projects that output away. All rows are required.
    ``fault_ids`` names the disjoint source groups owned by this region, including
    undetectable sources. Ownership is qualified by the request's instance ID.
    The caller attests that no source or persistent state crosses instances.
    This declaration is checked for consistency, not proved from physical code.

    The supported host commit is every detector; buffer and boundary state must
    be empty. Internal overlap within a single supplied model remains available
    through the existing windowed decoder specifications. Optional ``binding``
    requires the v1 intrinsic and checks declared schema/program identity.
    Optional ``read_sources`` requires the observed intrinsic and checks native
    measurement occurrences and detector parities; an empty tuple is distinct
    from None. Semantic labels and physical independence remain declarations.
    """

    window_id: int
    model: DetectorErrorModel
    decoder: DecoderSpec | str
    detector_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    detector_order: tuple[int, ...]
    output_map: tuple[tuple[int, ...], ...]
    fault_ids: tuple[str, ...]
    independent: bool
    commit: tuple[str, ...]
    buffer: tuple[str, ...] = ()
    boundary_state: tuple[str, ...] = ()
    binding: RuntimeDecodeBinding | None = None
    read_sources: tuple[RuntimeReadSource, ...] | None = None

    def __post_init__(self) -> None:
        """Freeze sequences and reject unsupported or inconsistent contracts."""
        for name in ("detector_ids", "output_ids", "detector_order", "fault_ids", "commit", "buffer", "boundary_state"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "output_map", tuple(tuple(row) for row in self.output_map))
        if self.read_sources is not None:
            object.__setattr__(self, "read_sources", tuple(self.read_sources))
        self.validate()

    def validate(self) -> None:
        """Recheck dimensions before decoding, including caller model mutations."""
        if type(self.window_id) is not int or not 0 <= self.window_id < 2**64:
            msg = "window_id must be an unsigned 64-bit integer"
            raise ValueError(msg)
        if not isinstance(self.model, DetectorErrorModel):
            msg = "model must be a structured DetectorErrorModel (compose slices before handoff)"
            raise TypeError(msg)
        if self.model.num_tracked_paulis:
            msg = "runtime decoding does not support tracked-Pauli outputs"
            raise ValueError(msg)
        if self.independent is not True or self.buffer or self.boundary_state:
            msg = "runtime decoding currently requires independent closed windows with no buffer or boundary state"
            raise ValueError(msg)
        for name in ("detector_ids", "output_ids", "fault_ids"):
            ids = getattr(self, name)
            if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
                msg = f"{name} must contain unique nonempty identities"
                raise ValueError(msg)
        if not self.fault_ids:
            msg = "fault_ids must explicitly declare the owned physical source groups"
            raise ValueError(msg)
        n = self.model.num_detectors
        if len(self.detector_ids) != n or len(self.output_ids) != self.model.num_observables:
            msg = "detector/output identities must cover every model column"
            raise ValueError(msg)
        if self.commit != self.detector_ids:
            msg = "commit must contain every detector in model order; partial commits need boundary carry"
            raise ValueError(msg)
        if any(type(i) is not int for i in self.detector_order) or sorted(self.detector_order) != list(range(n)):
            msg = "detector_order must be a complete permutation of program observations"
            raise ValueError(msg)
        if len(self.output_map) != len(self.output_ids) or any(
            type(bit) is not int or not 0 <= bit < 64 for row in self.output_map for bit in row
        ):
            msg = "output_map must explicitly route every output to frame bits 0..63"
            raise ValueError(msg)

        if self.binding is not None:
            if type(self.binding) is not RuntimeDecodeBinding:
                msg = "binding must be a RuntimeDecodeBinding"
                raise TypeError(msg)
            schema = self.binding.schema
            if (
                len(schema.detectors) != n
                or tuple(schema.detectors[i].detector_id for i in self.detector_order) != self.detector_ids
            ):
                msg = "bound schema detector identities disagree with detector_order"
                raise ValueError(msg)
            if any(bit >= len(schema.frame_ids) for row in self.output_map for bit in row):
                msg = "output_map targets an undeclared schema frame bit"
                raise ValueError(msg)

        if self.read_sources is not None:
            validate_read_sources(self.binding, self.read_sources)


class RuntimeDecoder:
    """Factory of shot-local provider sessions for embedded decoder calls.

    Supply exactly one of a window collection or ``provider_factory``. The latter
    returns a fresh callable ``provider(RuntimeDecodeRequest) -> RuntimeDecodeWindow``
    on each shot's first decode. Only immutable definitions may be shared between
    workers. Providers never receive simulator faults, truth, seeds or futures.
    """

    def __init__(
        self,
        windows: Mapping[int, RuntimeDecodeWindow] | None = None,
        *,
        provider_factory: Callable[[], Callable[[RuntimeDecodeRequest], RuntimeDecodeWindow]] | None = None,
    ) -> None:
        """Configure a supplied collection or a shot-local provider factory."""
        if (windows is None) == (provider_factory is None):
            msg = "supply exactly one of windows or provider_factory"
            raise ValueError(msg)
        self._windows = None if windows is None else MappingProxyType(dict(windows))
        self._provider_factory = provider_factory
        if self._windows is not None:
            for key, window in self._windows.items():
                if not isinstance(window, RuntimeDecodeWindow) or type(key) is not int or key != window.window_id:
                    msg = "window collection keys must match declared window_id"
                    raise ValueError(msg)
        elif not callable(provider_factory):
            msg = "provider_factory must be callable"
            raise TypeError(msg)

    def __call__(self) -> _Session:
        """Start a fresh session; called by the simulation worker, never the builder."""
        if self._windows is not None:
            return _Session(lambda request: self._windows[request.window_id])
        if self._provider_factory is None:
            msg = "runtime decoder has no provider factory"
            raise RuntimeError(msg)
        provider = self._provider_factory()
        if not callable(provider):
            msg = "provider_factory must return a callable window provider"
            raise TypeError(msg)
        return _Session(provider)


class _Session:
    def __init__(self, provider: Callable[[RuntimeDecodeRequest], RuntimeDecodeWindow]) -> None:
        self._provider = provider
        self._next_instance = 0
        self._next_batch_index = 0
        self._runtime_shot_id = None
        self._failed = False

    def decode(self, window_id: int, instance_id: int, syndrome: list[int] | bytes) -> int:
        return self._decode(window_id, instance_id, syndrome, None)

    def decode_v1(self, window_id: int, instance_id: int, syndrome: list[int] | bytes, context: bytes) -> int:
        if self._failed:
            msg = "decoder session failed; start a fresh shot"
            raise RuntimeError(msg)
        # Malformed envelopes latch failure before invoking a provider.
        self._failed = True
        parsed = RuntimeDecodeContext.from_json(context)
        self._failed = False
        return self._decode(window_id, instance_id, syndrome, parsed)

    def decode_observed_v1(
        self,
        window_id: int,
        instance_id: int,
        syndrome: list[int] | bytes,
        context: bytes,
        observations: list[tuple[str, int, int, bool]],
    ) -> int:
        if self._failed:
            msg = "decoder session failed; start a fresh shot"
            raise RuntimeError(msg)
        self._failed = True
        if type(observations) is not list:
            msg = "observations must be a bounded list of native read tuples"
            raise ValueError(msg)
        parsed = RuntimeDecodeContext.from_json(context)
        self._failed = False
        return self._decode(window_id, instance_id, syndrome, parsed, observations)

    def decode_observed_with_batches_v1(
        self,
        window_id: int,
        instance_id: int,
        syndrome: list[int] | bytes,
        context: bytes,
        observations: list[tuple[str, int, int, bool]],
        prefix: bytes,
    ) -> int:
        if self._failed:
            msg = "decoder session failed; start a fresh shot"
            raise RuntimeError(msg)
        self._failed = True
        parsed = RuntimeDecodeContext.from_json(context)
        batch_prefix = RuntimeDecodeBatchPrefix.from_json(prefix)
        if batch_prefix.first_batch_index != self._next_batch_index or (
            self._runtime_shot_id is not None and batch_prefix.runtime_shot_id != self._runtime_shot_id
        ):
            msg = "decoder batch prefix changed shot or skipped/repeated batches"
            raise ValueError(msg)
        if type(observations) is not list:
            msg = "observations must be a bounded list of native read tuples"
            raise ValueError(msg)
        self._failed = False
        result = self._decode(window_id, instance_id, syndrome, parsed, observations, batch_prefix)
        self._next_batch_index = batch_prefix.next_batch_index
        self._runtime_shot_id = batch_prefix.runtime_shot_id
        return result

    def _decode(
        self,
        window_id: int,
        instance_id: int,
        syndrome: list[int] | bytes,
        context: RuntimeDecodeContext | None,
        observations: list[tuple[str, int, int, bool]] | None = None,
        batch_prefix: RuntimeDecodeBatchPrefix | None = None,
    ) -> int:
        if self._failed:
            msg = "decoder session failed; start a fresh shot"
            raise RuntimeError(msg)
        self._failed = True
        if type(window_id) is not int or not 0 <= window_id < 2**64:
            msg = "window_id must be an unsigned 64-bit integer"
            raise ValueError(msg)
        if type(instance_id) is not int or not 0 <= instance_id < 2**64 or instance_id != self._next_instance:
            msg = "decoder instance IDs must be contiguous from zero in each shot"
            raise ValueError(msg)
        if context is not None and any(
            dep.source == "decoder" and dep.occurrence >= instance_id
            for value in context.inputs
            for dep in value.dependencies
        ):
            msg = "decoder dependencies must reference a prior instance"
            raise ValueError(msg)
        request = (
            RuntimeDecodeRequest(window_id, instance_id, context)
            if batch_prefix is None
            else RuntimeDecodeBatchRequest(
                window_id=window_id,
                instance_id=instance_id,
                context=context,
                batch_prefix=batch_prefix,
            )
        )
        window = self._provider(request)
        if not isinstance(window, RuntimeDecodeWindow) or window.window_id != window_id:
            msg = "provider returned a different window identity"
            raise ValueError(msg)
        window.validate()
        if context is None and window.binding is not None:
            msg = "bound window requires a versioned decode context"
            raise ValueError(msg)
        if context is not None and (window.binding is None or not window.binding.matches(context)):
            msg = "runtime decode context does not match the window binding"
            raise ValueError(msg)
        if len(syndrome) != len(window.detector_ids) or any(
            type(bit) is not int or bit not in (0, 1) for bit in syndrome
        ):
            msg = "syndrome must supply exactly one observed bit per declared detector"
            raise ValueError(msg)
        if (window.read_sources is None) != (observations is None):
            msg = "observed decode calls require read_sources; checked windows require observed decode calls"
            raise ValueError(msg)
        if observations is not None:
            check_observed_reads(window.binding, window.read_sources, syndrome, observations)
        # PyStructuredDemDecoder is thread-bound. Construct, use and release it
        # here; no live decoder moves between workers or survives into a new shot.
        decoder = window.model.build_decoder(window.decoder)
        prediction = decoder.decode_syndrome([syndrome[i] for i in window.detector_order])
        if len(prediction) != len(window.output_map):
            msg = "decoder output width differs from the declared mapping"
            raise ValueError(msg)
        frame = 0
        for flip, destinations in zip(prediction, window.output_map, strict=True):
            if flip:
                for bit in destinations:
                    frame ^= 1 << bit
        self._next_instance += 1
        self._failed = False
        return frame
