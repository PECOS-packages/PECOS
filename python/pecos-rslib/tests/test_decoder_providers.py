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
    """Optional dependencies must also be publishable for crates.io packaging."""
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
