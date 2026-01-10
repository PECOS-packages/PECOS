"""Transversal gates for surface codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.qeclib.surface.patches.patch_base import SurfacePatch


class SurfaceTransversalGates:
    """A collection of transversal gate implementations for the `Surface4444RotPatch`."""

    @staticmethod
    def x(*patches: SurfacePatch) -> None:
        """Apply logical Pauli X on surface code patches."""
        # TODO: ...

    @staticmethod
    def y(*patches: SurfacePatch) -> None:
        """Apply logical Pauli Y on surface code patches."""
        # TODO: ...

    @staticmethod
    def z(*patches: SurfacePatch) -> None:
        """Apply logical Pauli Z on surface code patches."""
        # TODO: ...

    @staticmethod
    def h(*patches: SurfacePatch, permute: bool = True) -> None:
        """Apply transversal Hadamard on surface code patches followed by a Permutation/Relabeling.

        Args:
            patches: Surface code patches to apply Hadamard gates to.
            permute: Whether to apply permutation/relabeling after Hadamard.

        Examples:
            ```python
            from pecos.slr import Main
            from pecos.slr.qeclib.surface import Surface4444RotPatch as SP

            prog = Main(
                s := [SP.new(3) for _ in range(4)],
                SP.h(s[0], s[3]),
            )
            ```
        """
        # TODO: ...

    @staticmethod
    def cx(
        *patches: SurfacePatch | tuple[SurfacePatch, SurfacePatch],
    ) -> None:
        """Apply transversal CX on surface code patches.

        Args:
            patches: Can either be tuples of two surface code patches or a sequence of surface code patches.
                If tuples, then the first is considered the control and the second the target surface code.
                If a sequence of surface code patches are supplied, it is assumed the sequence is of a control
                surface code followed by a target surface code patch sequence.

        Examples:
             ```python
             from pecos.slr import Main
             from pecos.slr.qeclib.surface import Surface4444RotPatch as SP

             prog = Main(
                 s := [SP.new(3) for _ in range(4)],
                 SP.cx((s[0], s[2]), (s[1], s[3])),
                 SP.cx(s[0], s[2], s[1], s[3]),  # Equivalent to previous line
                 # Equivalent to the following two lines:
                 SP.cx(s[0], s[2]),
                 SP.cx(s[1], s[3]),
             )
             ```
        """
        # TODO: ...
