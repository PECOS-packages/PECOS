"""PECOS runtime, usable anywhere Selene accepts a Runtime plugin."""

import platform
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

from selene_core import Runtime


@dataclass
class PecosRuntimePlugin(Runtime):
    """Lazy virtual-Z runtime with bounded queues and leakage-aware futures.

    Durations are nanoseconds. Batching is deterministic and generic; it does
    not model a particular ion trap. ``library_path`` overrides the packaged
    library for Rust development builds.
    """

    duration_ns_rxy: int = 0
    duration_ns_rzz: int = 0
    duration_ns_measure: int = 0
    duration_ns_measure_leaked: int = 0
    duration_ns_reset: int = 0
    max_batch_size: int = 1
    max_queued_batches: int = 4096
    max_live_results: int = 65536
    library_path: Path | None = None
    native_gates: ClassVar[tuple[str, ...]] = ("rxy", "rz", "rzz")

    def __post_init__(self) -> None:
        """Validate configuration before entering the native plugin."""
        for field in fields(self):
            if field.name in {"library_path", "random_seed"}:
                continue
            value = getattr(self, field.name)
            minimum = 1 if field.name.startswith("max_") else 0
            if type(value) is not int or not minimum <= value <= 2**64 - 1:
                msg = f"{field.name} must be an integer between {minimum} and 2**64 - 1"
                raise ValueError(msg)

    def get_init_args(self) -> list[str]:
        """Return standard Selene runtime initialization arguments."""
        return [
            f"--{field.name.replace('_', '-')}={getattr(self, field.name)}"
            for field in fields(self)
            if field.name not in {"library_path", "random_seed"}
        ]

    @property
    def library_file(self) -> Path:
        """Locate the normal Selene runtime shared library."""
        if self.library_path is not None:
            return Path(self.library_path).resolve()
        filenames = {
            "Linux": "libpecos_selene_runtime_plugin.so",
            "Darwin": "libpecos_selene_runtime_plugin.dylib",
            "Windows": "pecos_selene_runtime_plugin.dll",
        }
        try:
            filename = filenames[platform.system()]
        except KeyError as exc:
            msg = f"Unsupported platform: {platform.system()}"
            raise RuntimeError(msg) from exc
        return Path(__file__).parent / "_dist" / "lib" / filename
