"""Bare syndrome extraction gates for surface codes."""

from pecos.slr.qeclib.surface.patches.patch_base import SurfacePatch


class SurfaceBareSynGates:
    """Collection of bare syndrome extraction gates for surface code patches."""

    @staticmethod
    def syn_extr(*patches: SurfacePatch, rounds: int = 1) -> None:
        """Measure `rounds` number of syndrome extraction of X and Z checks using bare ancillas."""
        # TODO: ...

    @staticmethod
    def qec(*patches: SurfacePatch) -> list[None]:
        """Run distance number of rounds of syndrome extraction."""
        return [SurfaceBareSynGates.syn_extr(rounds=p.distance) for p in patches]
