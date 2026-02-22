# ruff: noqa: N801
# Class names like SurfaceCode_5x5 clearly communicate the code dimensions
"""Surface code patch (dx=5, dz=5) implementation in Guppy.

Auto-generated from SurfacePatch geometry.

Data qubits: 25
X stabilizers: 12
Z stabilizers: 12
Ancilla qubits: 24 (one per stabilizer)
"""

from collections.abc import Callable

from guppylang import guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.quantum import cx, h, measure, measure_array, qubit, x


@guppy.struct
class SurfaceCode_5x5:
    """Surface code patch with dx=5, dz=5 (25 data qubits)."""

    data: array[qubit, 25]


@guppy.struct
class Syndrome_5x5:
    """Syndrome for dx=5, dz=5 patch."""

    synx: array[bool, 12]
    synz: array[bool, 12]


# === State Preparation ===


@guppy
def prep_z_basis() -> SurfaceCode_5x5:
    """Prepare logical |0_L> state."""
    data = array(qubit() for _ in range(25))
    return SurfaceCode_5x5(data)


@guppy
def prep_x_basis() -> SurfaceCode_5x5:
    """Prepare logical |+_L> state."""
    data = array(qubit() for _ in range(25))
    for i in range(25):
        h(data[i])
    return SurfaceCode_5x5(data)


# === Syndrome Extraction ===


@guppy
def syndrome_extraction(surf: SurfaceCode_5x5) -> Syndrome_5x5:
    """Extract full syndrome using 4-round parallel CNOT schedule."""
    # Allocate ancilla qubits (one per stabilizer)
    ax0 = qubit()
    ax1 = qubit()
    ax2 = qubit()
    ax3 = qubit()
    ax4 = qubit()
    ax5 = qubit()
    ax6 = qubit()
    ax7 = qubit()
    ax8 = qubit()
    ax9 = qubit()
    ax10 = qubit()
    ax11 = qubit()
    az0 = qubit()
    az1 = qubit()
    az2 = qubit()
    az3 = qubit()
    az4 = qubit()
    az5 = qubit()
    az6 = qubit()
    az7 = qubit()
    az8 = qubit()
    az9 = qubit()
    az10 = qubit()
    az11 = qubit()

    # Hadamard on X ancillas
    h(ax0)
    h(ax1)
    h(ax2)
    h(ax3)
    h(ax4)
    h(ax5)
    h(ax6)
    h(ax7)
    h(ax8)
    h(ax9)
    h(ax10)
    h(ax11)

    # Round 1
    cx(surf.data[15], az0)
    cx(surf.data[5], az1)
    cx(ax2, surf.data[2])
    cx(surf.data[11], az2)
    cx(ax3, surf.data[4])
    cx(surf.data[1], az3)
    cx(ax4, surf.data[6])
    cx(surf.data[17], az4)
    cx(ax5, surf.data[8])
    cx(surf.data[7], az5)
    cx(ax6, surf.data[12])
    cx(surf.data[13], az6)
    cx(ax7, surf.data[14])
    cx(surf.data[3], az7)
    cx(ax8, surf.data[16])
    cx(surf.data[19], az8)
    cx(ax9, surf.data[18])
    cx(surf.data[9], az9)
    cx(ax10, surf.data[22])
    cx(ax11, surf.data[24])

    # Round 2
    cx(surf.data[20], az0)
    cx(surf.data[10], az1)
    cx(ax2, surf.data[1])
    cx(surf.data[16], az2)
    cx(ax3, surf.data[3])
    cx(surf.data[6], az3)
    cx(ax4, surf.data[5])
    cx(surf.data[22], az4)
    cx(ax5, surf.data[7])
    cx(surf.data[12], az5)
    cx(ax6, surf.data[11])
    cx(surf.data[18], az6)
    cx(ax7, surf.data[13])
    cx(surf.data[8], az7)
    cx(ax8, surf.data[15])
    cx(surf.data[24], az8)
    cx(ax9, surf.data[17])
    cx(surf.data[14], az9)
    cx(ax10, surf.data[21])
    cx(ax11, surf.data[23])

    # Round 3
    cx(ax0, surf.data[1])
    cx(ax1, surf.data[3])
    cx(ax2, surf.data[7])
    cx(surf.data[10], az2)
    cx(ax3, surf.data[9])
    cx(surf.data[0], az3)
    cx(ax4, surf.data[11])
    cx(surf.data[16], az4)
    cx(ax5, surf.data[13])
    cx(surf.data[6], az5)
    cx(ax6, surf.data[17])
    cx(surf.data[12], az6)
    cx(ax7, surf.data[19])
    cx(surf.data[2], az7)
    cx(ax8, surf.data[21])
    cx(surf.data[18], az8)
    cx(ax9, surf.data[23])
    cx(surf.data[8], az9)
    cx(surf.data[14], az10)
    cx(surf.data[4], az11)

    # Round 4
    cx(ax0, surf.data[0])
    cx(ax1, surf.data[2])
    cx(ax2, surf.data[6])
    cx(surf.data[15], az2)
    cx(ax3, surf.data[8])
    cx(surf.data[5], az3)
    cx(ax4, surf.data[10])
    cx(surf.data[21], az4)
    cx(ax5, surf.data[12])
    cx(surf.data[11], az5)
    cx(ax6, surf.data[16])
    cx(surf.data[17], az6)
    cx(ax7, surf.data[18])
    cx(surf.data[7], az7)
    cx(ax8, surf.data[20])
    cx(surf.data[23], az8)
    cx(ax9, surf.data[22])
    cx(surf.data[13], az9)
    cx(surf.data[19], az10)
    cx(surf.data[9], az11)

    # Hadamard on X ancillas
    h(ax0)
    h(ax1)
    h(ax2)
    h(ax3)
    h(ax4)
    h(ax5)
    h(ax6)
    h(ax7)
    h(ax8)
    h(ax9)
    h(ax10)
    h(ax11)

    # Measure ancillas
    sx0 = measure(ax0)
    sx1 = measure(ax1)
    sx2 = measure(ax2)
    sx3 = measure(ax3)
    sx4 = measure(ax4)
    sx5 = measure(ax5)
    sx6 = measure(ax6)
    sx7 = measure(ax7)
    sx8 = measure(ax8)
    sx9 = measure(ax9)
    sx10 = measure(ax10)
    sx11 = measure(ax11)
    sz0 = measure(az0)
    sz1 = measure(az1)
    sz2 = measure(az2)
    sz3 = measure(az3)
    sz4 = measure(az4)
    sz5 = measure(az5)
    sz6 = measure(az6)
    sz7 = measure(az7)
    sz8 = measure(az8)
    sz9 = measure(az9)
    sz10 = measure(az10)
    sz11 = measure(az11)

    synx = array(sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7, sx8, sx9, sx10, sx11)
    synz = array(sz0, sz1, sz2, sz3, sz4, sz5, sz6, sz7, sz8, sz9, sz10, sz11)

    return Syndrome_5x5(synx, synz)


# === Measurement ===


@guppy
def measure_z_basis(surf: SurfaceCode_5x5 @ owned) -> array[bool, 25]:
    """Destructively measure in Z basis."""
    return measure_array(surf.data)


@guppy
def measure_x_basis(surf: SurfaceCode_5x5 @ owned) -> array[bool, 25]:
    """Destructively measure in X basis."""
    for i in range(25):
        h(surf.data[i])
    return measure_array(surf.data)


# === Logical Operators ===


@guppy
def apply_logical_x(surf: SurfaceCode_5x5) -> None:
    """Apply logical X (string along left edge)."""
    x(surf.data[0])
    x(surf.data[5])
    x(surf.data[10])
    x(surf.data[15])
    x(surf.data[20])


@guppy
def apply_logical_z(surf: SurfaceCode_5x5) -> None:
    """Apply logical Z (string along top edge)."""
    from guppylang.std.quantum import z

    z(surf.data[0])
    z(surf.data[1])
    z(surf.data[2])
    z(surf.data[3])
    z(surf.data[4])


# === Memory Experiments ===


def make_memory_z(num_rounds: int) -> Callable[[], None]:
    """Create Z-basis memory experiment."""
    from guppylang.std.builtins import comptime

    @guppy
    def memory_z() -> None:
        """Z-basis memory experiment for dx=5, dz=5."""
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
        """X-basis memory experiment for dx=5, dz=5."""
        surf = prep_x_basis()

        for _t in range(comptime(num_rounds)):
            syn = syndrome_extraction(surf)
            result("synx", syn.synx)
            result("synz", syn.synz)

        final = measure_x_basis(surf)
        result("final", final)

    return memory_x
