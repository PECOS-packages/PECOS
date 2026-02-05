# ruff: noqa: N801
# Class names like SurfaceCode_3x3 clearly communicate the code dimensions
"""Surface code patch (dx=3, dz=3) implementation in Guppy.

Auto-generated from SurfacePatch geometry.

Data qubits: 9
X stabilizers: 4
Z stabilizers: 4
Ancilla qubits: 8 (one per stabilizer)
"""

from collections.abc import Callable

from guppylang import guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.quantum import cx, h, measure, measure_array, qubit, x


@guppy.struct
class SurfaceCode_3x3:
    """Surface code patch with dx=3, dz=3 (9 data qubits)."""

    data: array[qubit, 9]


@guppy.struct
class Syndrome_3x3:
    """Syndrome for dx=3, dz=3 patch."""

    synx: array[bool, 4]
    synz: array[bool, 4]


# === State Preparation ===


@guppy
def prep_z_basis() -> SurfaceCode_3x3:
    """Prepare logical |0_L> state."""
    data = array(qubit() for _ in range(9))
    return SurfaceCode_3x3(data)


@guppy
def prep_x_basis() -> SurfaceCode_3x3:
    """Prepare logical |+_L> state."""
    data = array(qubit() for _ in range(9))
    for i in range(9):
        h(data[i])
    return SurfaceCode_3x3(data)


# === Syndrome Extraction ===


@guppy
def syndrome_extraction(surf: SurfaceCode_3x3) -> Syndrome_3x3:
    """Extract full syndrome using 4-round parallel CNOT schedule."""
    # Allocate ancilla qubits (one per stabilizer)
    ax0 = qubit()
    ax1 = qubit()
    ax2 = qubit()
    ax3 = qubit()
    az0 = qubit()
    az1 = qubit()
    az2 = qubit()
    az3 = qubit()

    # Hadamard on X ancillas
    h(ax0)
    h(ax1)
    h(ax2)
    h(ax3)

    # Round 1
    cx(surf.data[3], az0)
    cx(ax1, surf.data[2])
    cx(surf.data[1], az1)
    cx(ax2, surf.data[4])
    cx(surf.data[5], az2)
    cx(ax3, surf.data[8])

    # Round 2
    cx(surf.data[6], az0)
    cx(ax1, surf.data[1])
    cx(surf.data[4], az1)
    cx(ax2, surf.data[3])
    cx(surf.data[8], az2)
    cx(ax3, surf.data[7])

    # Round 3
    cx(ax0, surf.data[1])
    cx(ax1, surf.data[5])
    cx(surf.data[0], az1)
    cx(ax2, surf.data[7])
    cx(surf.data[4], az2)
    cx(surf.data[2], az3)

    # Round 4
    cx(ax0, surf.data[0])
    cx(ax1, surf.data[4])
    cx(surf.data[3], az1)
    cx(ax2, surf.data[6])
    cx(surf.data[7], az2)
    cx(surf.data[5], az3)

    # Hadamard on X ancillas
    h(ax0)
    h(ax1)
    h(ax2)
    h(ax3)

    # Measure ancillas
    sx0 = measure(ax0)
    sx1 = measure(ax1)
    sx2 = measure(ax2)
    sx3 = measure(ax3)
    sz0 = measure(az0)
    sz1 = measure(az1)
    sz2 = measure(az2)
    sz3 = measure(az3)

    synx = array(sx0, sx1, sx2, sx3)
    synz = array(sz0, sz1, sz2, sz3)

    return Syndrome_3x3(synx, synz)


# === Measurement ===


@guppy
def measure_z_basis(surf: SurfaceCode_3x3 @ owned) -> array[bool, 9]:
    """Destructively measure in Z basis."""
    return measure_array(surf.data)


@guppy
def measure_x_basis(surf: SurfaceCode_3x3 @ owned) -> array[bool, 9]:
    """Destructively measure in X basis."""
    for i in range(9):
        h(surf.data[i])
    return measure_array(surf.data)


# === Logical Operators ===


@guppy
def apply_logical_x(surf: SurfaceCode_3x3) -> None:
    """Apply logical X (string along left edge)."""
    x(surf.data[0])
    x(surf.data[3])
    x(surf.data[6])


@guppy
def apply_logical_z(surf: SurfaceCode_3x3) -> None:
    """Apply logical Z (string along top edge)."""
    from guppylang.std.quantum import z

    z(surf.data[0])
    z(surf.data[1])
    z(surf.data[2])


# === Memory Experiments ===


def make_memory_z(num_rounds: int) -> Callable[[], None]:
    """Create Z-basis memory experiment."""
    from guppylang.std.builtins import comptime

    @guppy
    def memory_z() -> None:
        """Z-basis memory experiment for dx=3, dz=3."""
        surf = prep_z_basis()

        for _t in range(comptime(num_rounds)):
            syn = syndrome_extraction(surf)
            result("synx", syn.synx)
            result("synz", syn.synz)

        final = measure_z_basis(surf)
        result("final", final)

    return memory_z


def make_memory_x(num_rounds: int) -> Callable[[], None]:
    """Create X-basis memory experiment."""
    from guppylang.std.builtins import comptime

    @guppy
    def memory_x() -> None:
        """X-basis memory experiment for dx=3, dz=3."""
        surf = prep_x_basis()

        for _t in range(comptime(num_rounds)):
            syn = syndrome_extraction(surf)
            result("synx", syn.synx)
            result("synz", syn.synz)

        final = measure_x_basis(surf)
        result("final", final)

    return memory_x
