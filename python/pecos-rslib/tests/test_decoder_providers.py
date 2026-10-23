"""Optional decoder providers use the regular batch planner without native dependencies."""

import subprocess
import sys
import weakref

import pytest
from pecos_rslib.qec import DemSampler, SampleBatch


class Provider:
    _pecos_decoder_api_version = 1
    history_dependent = False
    wall_clock_dependent = False

    def _pecos_build_decoder(self, dem):
        assert dem == "error(0.1) D0 L70\n"
        return Worker()


class Worker:
    num_detectors = 1
    num_observables = 71

    def _pecos_decode_obs(self, syndrome):
        return [0, (1 << 6) if syndrome[0] else 0]


DEM = "error(0.1) D0 L70\n"


@pytest.mark.parametrize("workers", [1, 3])
def test_provider_batch_and_sampler_paths(workers):
    batch = SampleBatch([[0], [1]], [0, 1 << 70])
    result = batch.decode(DEM, Provider(), workers=workers, predictions=True, timing=True)
    assert result.predictions == [0, 1 << 70]
    assert result.num_errors == 0
    assert result.workers_used == min(workers, batch.num_shots)
    assert result.stats.num_timing_samples == 2
    sampler = DemSampler.from_dem_string(DEM)
    result = sampler.decode(DEM, 3073, Provider(), workers=workers, seed=2)
    assert result.num_errors == 0


def test_provider_validation_and_execution_traits():
    batch = SampleBatch([[0]], [0])
    with pytest.raises(TypeError, match="decoder must"):
        batch.decode(DEM, object())
    provider = Provider()
    provider._pecos_decoder_api_version = 99
    with pytest.raises(TypeError, match="version-1"):
        batch.decode(DEM, provider)
    provider._pecos_decoder_api_version = 1
    provider.history_dependent = True
    with pytest.raises(ValueError, match="history|stateful|workers"):
        batch.decode(DEM, provider, workers=3)
    with pytest.raises(ValueError, match="detectors"):
        SampleBatch([[0, 0]], [0]).decode(DEM, Provider())


@pytest.mark.parametrize("trait", [None, "history_dependent", "wall_clock_dependent"])
def test_execution_traits_steer_automatic_planning(trait):
    # Large enough that automatic planning parallelizes a provider with neither trait.
    batch = SampleBatch([[0]] * 3073, [0] * 3073)
    provider = Provider()
    if trait is not None:
        setattr(provider, trait, True)
    expected = "parallel" if trait is None else "sequential"
    assert batch.decode(DEM, provider).execution_path == expected


def test_member_access_errors_other_than_the_protocol_propagate():
    class BrokenProvider(Provider):
        @property
        def history_dependent(self):
            return 1 // 0

    with pytest.raises(ZeroDivisionError):
        SampleBatch([[0]], [0]).decode(DEM, BrokenProvider())


@pytest.mark.parametrize("sampler", [False, True])
@pytest.mark.parametrize(
    ("dem", "words", "observable", "declared"),
    [
        (DEM, [0, 1 << 7], 71, 71),
        ("error(0.1) D0 L0\n", [1 << 5], 5, 1),
    ],
)
def test_prediction_outside_the_model_observables_is_an_error(sampler, dem, words, observable, declared):
    class WideWorker(Worker):
        num_observables = declared

        def _pecos_decode_obs(self, syndrome):
            return words

    class WideProvider(Provider):
        def _pecos_build_decoder(self, dem):
            return WideWorker()

    if sampler:
        source = DemSampler.from_dem_string(dem)
        args = (dem, 8, WideProvider())
    else:
        source = SampleBatch([[0]], [0])
        args = (dem, WideProvider())
    message = f"predicted observable {observable}, but its model has {declared} observables"
    with pytest.raises(RuntimeError, match=message):
        source.decode(*args, workers=1)


@pytest.mark.parametrize(
    ("num_observables", "words", "message"),
    [
        (71, [0, 0, 0], "returned 3 observable words, but 71 observables need at most 2"),
        (0, [1], "returned 1 observable words, but 0 observables need at most 0"),
    ],
)
def test_too_many_prediction_words_is_an_error(num_observables, words, message):
    class LongWorker(Worker):
        def _pecos_decode_obs(self, syndrome):
            return words

    class LongProvider(Provider):
        def _pecos_build_decoder(self, dem):
            worker = LongWorker()
            worker.num_observables = num_observables
            return worker

    with pytest.raises(RuntimeError, match=message):
        SampleBatch([[0]], [0]).decode(DEM, LongProvider(), workers=1)


def test_version_probe_errors_other_than_the_protocol_propagate():
    class LazyProvider:
        @property
        def _pecos_decoder_api_version(self):
            raise ImportError("optional extension missing")

    with pytest.raises(ImportError, match="optional extension missing"):
        SampleBatch([[0]], [0]).decode(DEM, LazyProvider())


def test_standard_decoders_import_without_experimental_package():
    code = """
import sys
sys.modules['pecos_rslib_exp'] = None
import pecos.decoders as decoders
from pecos.decoders import *
from pecos_rslib.qec import SampleBatch
assert SampleBatch([[0]], [0]).decode('error(0.1) D0 L0', decoders.pymatching(correlated=False)).num_errors == 0
for name in ('frontier', 'bp_trellis'):
    assert name not in decoders.__all__
    try:
        getattr(decoders, name)
    except ImportError as error:
        assert 'optional pecos-rslib-exp' in str(error)
    else:
        raise AssertionError('optional dependency was not required')
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 - fixed test program


def test_published_decoder_manifest_has_no_unpublishable_dependencies():
    """crates/pecos-decoders must not transitively path-depend on a publish = false crate."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["dependencies"]
    visited = set()

    def inspect(manifest_path):
        if manifest_path in visited:
            return
        visited.add(manifest_path)
        manifest = tomllib.loads(manifest_path.read_text())
        assert manifest["package"].get("publish") is not False, manifest_path
        for section in ("dependencies", "build-dependencies"):
            for name, entry in manifest.get(section, {}).items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("workspace"):
                    dependency = workspace[name]
                    directory = root
                else:
                    dependency = entry
                    directory = manifest_path.parent
                if isinstance(dependency, dict) and "path" in dependency:
                    inspect((directory / dependency["path"] / "Cargo.toml").resolve())

    inspect(root / "crates/pecos-decoders/Cargo.toml")


def test_oversized_worker_request_builds_one_decoder_per_shot():
    builds = [0]

    class CountingProvider(Provider):
        def _pecos_build_decoder(self, dem):
            builds[0] += 1
            return Worker()

    result = SampleBatch([[0], [1]], [0, 1 << 70]).decode(DEM, CountingProvider(), workers=64)
    assert result.workers_used == 2
    assert builds[0] == 2


def test_provider_workers_decode_concurrently():
    import threading

    workers = 3
    barrier = threading.Barrier(workers, timeout=10)

    class ConcurrentWorker(Worker):
        first = True

        def _pecos_decode_obs(self, syndrome):
            if self.first:
                self.first = False
                barrier.wait()
            return super()._pecos_decode_obs(syndrome)

    class ConcurrentProvider(Provider):
        def _pecos_build_decoder(self, dem):
            assert dem == DEM
            return ConcurrentWorker()

    batch = SampleBatch([[0]] * 3073, [0] * 3073)
    result = batch.decode(DEM, ConcurrentProvider(), workers=workers)
    assert result.num_errors == 0
    assert result.workers_used == workers
    assert not barrier.broken


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("sampler", [False, True])
@pytest.mark.parametrize("failure", ["build", "num_detectors", "_pecos_decode_obs"])
def test_provider_build_exceptions_propagate(workers, sampler, failure):
    import traceback

    error = KeyError("boom")

    def raise_error():
        raise error

    class InvalidWorker:
        num_observables = 71

        @property
        def num_detectors(self):
            if failure == "num_detectors":
                raise_error()
            return 1

        @property
        def _pecos_decode_obs(self):
            raise_error()

    class InvalidProvider(Provider):
        def _pecos_build_decoder(self, dem):
            if failure == "build":
                raise_error()
            return InvalidWorker()

    if sampler:
        source = DemSampler.from_dem_string(DEM)
        args = (DEM, 3073, InvalidProvider())
    else:
        source = SampleBatch([[0]] * 3073, [0] * 3073)
        args = (DEM, InvalidProvider())
    with pytest.raises(KeyError, match="boom") as caught:
        source.decode(*args, workers=workers)
    assert caught.value is error
    assert str(caught.value) == "'boom'"
    assert traceback.extract_tb(caught.value.__traceback__)[-1].name == "raise_error"


def test_fused_provider_build_exception_after_preflight():
    import traceback

    class InvalidProvider(Provider):
        builds = 0

        def _pecos_build_decoder(self, dem):
            self.builds += 1
            if self.builds > 1:
                raise KeyError("boom")
            return super()._pecos_build_decoder(dem)

    provider = InvalidProvider()
    with pytest.raises(KeyError, match="boom") as caught:
        DemSampler.from_dem_string(DEM).decode(DEM, 3073, provider, workers=3)
    assert provider.builds > 1
    assert traceback.extract_tb(caught.value.__traceback__)[-1].name == "_pecos_build_decoder"


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("member", ["factory", "history_dependent", "wall_clock_dependent"])
@pytest.mark.parametrize("missing", [False, True])
def test_required_provider_members(version, member, missing):
    method = "_pecos_build_decoder" if version == 1 else "_pecos_prepare_decoder"
    if member == "factory":
        member = method
    members = {
        "_pecos_decoder_api_version": version,
        method: lambda self, dem: Worker(),
        "history_dependent": False,
        "wall_clock_dependent": False,
    }
    if missing:
        del members[member]
    else:
        members[member] = "wrong type"
    provider = type("InvalidProvider", (), members)()
    with pytest.raises(
        TypeError,
        match=f"version-{version}.*{method}.*history_dependent.*wall_clock_dependent",
    ) as caught:
        SampleBatch([[0]], [0]).decode(DEM, provider)
    assert isinstance(caught.value.__cause__, AttributeError if missing else TypeError)


def test_member_access_error_subclasses_are_protocol_errors():
    class MissingMember(AttributeError):
        pass

    class DynamicProvider:
        _pecos_decoder_api_version = 1

        def __getattr__(self, name):
            raise MissingMember(name)

    with pytest.raises(TypeError, match="version-1") as caught:
        SampleBatch([[0]], [0]).decode(DEM, DynamicProvider())
    assert isinstance(caught.value.__cause__, MissingMember)
    message = str(caught.value)
    assert "version-1" in message
    for required in (
        "_pecos_build_decoder",
        "history_dependent",
        "wall_clock_dependent",
    ):
        assert required in message


def test_provider_per_shot_exception_keeps_shot_context():
    class FailingWorker(Worker):
        def _pecos_decode_obs(self, syndrome):
            raise KeyError("boom")

    class FailingProvider(Provider):
        def _pecos_build_decoder(self, dem):
            return FailingWorker()

    with pytest.raises(RuntimeError, match="decoder failed on shot 0:.*KeyError.*boom"):
        SampleBatch([[0]], [0]).decode(DEM, FailingProvider())


@pytest.mark.parametrize("version", [1, 2])
def test_unlabelled_embedded_corpus_keeps_provider_support(tmp_path, version):
    import hashlib
    import json
    import struct

    path = tmp_path / "provider.pecos"
    SampleBatch([[1], [0]], [1 << 70, 0], num_observables=71).save(path, dem=DEM)
    if version == 1:
        data = path.read_bytes()
        end = 48 + struct.unpack("<I", data[12:16])[0]
        header = json.loads(data[48:end])
        header["format_version"] = 1
        header["detector_fault_sectors"] = ["ignored"]
        encoded = json.dumps(header).encode()
        length = struct.pack("<I", len(encoded))
        payload = data[end:]
        path.write_bytes(data[:12] + length + hashlib.sha256(length + encoded + payload).digest() + encoded + payload)
    loaded = SampleBatch.load(path)
    assert loaded.format_version == version
    assert loaded.dem_fault_sectors == [None]
    assert loaded.decode(None, Provider()).num_errors == 0
    loaded.save(path, dem=loaded.dem)
    assert loaded.format_version == version
    assert SampleBatch.load(path).format_version == 2


def test_labelled_embedded_corpus_refuses_provider(tmp_path):
    from pecos_rslib.qec import DetectorErrorModel
    from pecos_rslib.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.add_detector([-1], detector_id=0, fault_sector="X")
    dem = DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.01, p_prep=0)
    path = tmp_path / "labelled.pecos"
    SampleBatch([[0]], [0], num_observables=0).save(path, dem=dem)
    with pytest.raises(
        RuntimeError,
        match="embedded model carries fault-sector labels.*passing loaded.dem text decodes without them",
    ):
        SampleBatch.load(path).decode(None, Provider())


class OutcomeWorker(Worker):
    def _pecos_decode_outcome(self, syndrome):
        return self._pecos_decode_obs(syndrome), False, None, [1]


class PreparedProvider:
    _pecos_decoder_api_version = 2
    history_dependent = False
    wall_clock_dependent = False

    def __init__(self):
        self.prepares = 0
        self.builds = 0
        self.prepared = []

    def _pecos_prepare_decoder(self, dem):
        import time

        assert dem == DEM
        self.prepares += 1
        # Release the GIL while other workers reach the preparation cache.
        time.sleep(0.01)
        provider = self

        class Prepared:
            def _pecos_build_decoder(self):
                provider.builds += 1
                return OutcomeWorker()

        prepared = Prepared()
        self.prepared.append(weakref.ref(prepared))
        return prepared


@pytest.mark.parametrize("sampler", [False, True])
def test_v2_prepares_once_per_decode_call(sampler):
    import gc

    provider = PreparedProvider()
    if sampler:
        # Enough shots that the fused sampler schedules several build jobs.
        source = DemSampler.from_dem_string(DEM)
        args = (DEM, 65_539, provider)
    else:
        source = SampleBatch([[0]] * 64, [0] * 64)
        args = (DEM, provider)
    for calls in [1, 2]:
        before = provider.builds
        result = source.decode(*args, workers=4)
        assert result.workers_used == 4
        assert result.num_errors == 0
        assert provider.prepares == calls
        assert provider.builds - before > 1
        gc.collect()
        assert provider.prepared[-1]() is None, "the preparation outlived its decode call"


@pytest.mark.parametrize("sampler", [False, True])
def test_v2_prepare_exception_propagates(sampler):
    import traceback

    error = KeyError("prepare boom")

    class FailingProvider(PreparedProvider):
        def _pecos_prepare_decoder(self, dem):
            import time

            self.prepares += 1
            # On the SampleBatch path the other workers queue on this preparation, so
            # a single call proves the failure is cached. The fused sampler's preflight
            # build fails on the calling thread first; there this checks propagation.
            time.sleep(0.01)
            raise error

    provider = FailingProvider()
    if sampler:
        source = DemSampler.from_dem_string(DEM)
        args = (DEM, 65_539, provider)
    else:
        source = SampleBatch([[0]] * 65, [0] * 65)
        args = (DEM, provider)
    with pytest.raises(KeyError, match="prepare boom") as caught:
        source.decode(*args, workers=4)
    assert caught.value is error
    assert traceback.extract_tb(caught.value.__traceback__)[-1].name == "_pecos_prepare_decoder"
    assert provider.prepares == 1


@pytest.mark.parametrize(
    ("words", "fallback", "consistent", "deltas", "message"),
    [
        (
            None,
            True,
            True,
            [1],
            "fallback_used=True requires prediction words, got None",
        ),
        (
            [0],
            False,
            True,
            [1],
            "syndrome_consistent=True requires fallback_used=True, got False",
        ),
        ([0], False, None, [2, 1], "deltas must be strictly increasing, got 2.0 then 1.0"),
        ([0], False, None, [1, 1], "deltas must be strictly increasing, got 1.0 then 1.0"),
        ([0], False, None, [float("nan")], "delta NaN must be nonnegative and not NaN"),
        ([0], False, None, [-1], "delta -1.0 must be nonnegative and not NaN"),
        (
            [0, 128],
            True,
            True,
            [1],
            "predicted observable 71, but its model has 71 observables",
        ),
    ],
)
def test_v2_provider_rejects_invalid_outcomes(words, fallback, consistent, deltas, message):
    class InvalidOutcomeWorker(Worker):
        def _pecos_decode_outcome(self, syndrome):
            return words, fallback, consistent, deltas

    class Prepared:
        def _pecos_build_decoder(self):
            return InvalidOutcomeWorker()

    class OutcomeProvider(PreparedProvider):
        def _pecos_prepare_decoder(self, dem):
            return Prepared()

    with pytest.raises(RuntimeError, match=f"shot 0:.*{message}"):
        SampleBatch([[0]], [0]).decode(DEM, OutcomeProvider(), workers=1)
