"""Optional decoder providers use the regular batch planner without native dependencies."""

import subprocess
import sys

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

    def _pecos_decode_obs(self, syndrome):
        return [0, (1 << 6) if syndrome[0] else 0]


DEM = "error(0.1) D0 L70\n"


@pytest.mark.parametrize("workers", [1, 3])
def test_provider_batch_and_sampler_paths(workers):
    batch = SampleBatch([[0], [1]], [0, 1 << 70])
    result = batch.decode(DEM, Provider(), workers=workers, predictions=True, timing=True)
    assert result.predictions == [0, 1 << 70]
    assert result.num_errors == 0
    assert result.workers_used == workers
    assert result.stats.num_timing_samples == 2
    sampler = DemSampler.from_dem_string(DEM)
    result = sampler.decode(DEM, 3073, Provider(), workers=workers, seed=2)
    assert result.num_errors == 0


def test_provider_validation_and_execution_traits():
    batch = SampleBatch([[0]], [0])
    with pytest.raises(TypeError, match="decoder must"):
        batch.decode(DEM, object())
    provider = Provider()
    provider._pecos_decoder_api_version = 2
    with pytest.raises(TypeError, match="version-1"):
        batch.decode(DEM, provider)
    provider._pecos_decoder_api_version = 1
    provider.history_dependent = True
    with pytest.raises(ValueError, match="history|stateful|workers"):
        batch.decode(DEM, provider, workers=3)
    assert batch.decode(DEM, provider).execution_path == "sequential"
    with pytest.raises(ValueError, match="detectors"):
        SampleBatch([[0, 0]], [0]).decode(DEM, Provider())


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


@pytest.mark.parametrize("member", ["_pecos_build_decoder", "history_dependent", "wall_clock_dependent"])
@pytest.mark.parametrize("missing", [False, True])
def test_required_version_one_members(member, missing):
    members = {
        "_pecos_decoder_api_version": 1,
        "_pecos_build_decoder": lambda self, dem: Worker(),
        "history_dependent": False,
        "wall_clock_dependent": False,
    }
    if missing:
        del members[member]
    else:
        members[member] = "wrong type"
    provider = type("InvalidProvider", (), members)()
    with pytest.raises(TypeError) as caught:
        SampleBatch([[0]], [0]).decode(DEM, provider)
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
