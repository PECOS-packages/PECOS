"""Measurement and preparation gates for surface codes."""

from pecos.slr import Bit
from pecos.slr.qeclib.surface.macrolibs.preps.project_pauli import PrepProjectZ
from pecos.slr.qeclib.surface.patches.patch_base import SurfacePatch


class SurfaceMeasPrepGates:
    """Collection of measurement and preparation gates for surface code patches."""

    @staticmethod
    def pz(*patches: SurfacePatch) -> list[PrepProjectZ]:
        """Prepare patches in the Z basis.

        Args:
            patches: Surface code patches to prepare in the Z basis.

        Returns:
            List of PrepProjectZ objects for each patch.
        """
        return [PrepProjectZ(p.data) for p in patches]

    @staticmethod
    def mz(
        patches: tuple[SurfacePatch, ...] | SurfacePatch,
        outputs: Bit | tuple[Bit, ...],
    ) -> None:
        """Destructively measure in the Z basis."""
        # TODO: ...
