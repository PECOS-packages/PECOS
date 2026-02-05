# ruff: noqa: N801
# Class names like SurfaceCode_7x7 clearly communicate the code dimensions
"""Surface code patch (dx=7, dz=7) implementation in Guppy.

Auto-generated from SurfacePatch geometry.

Data qubits: 49
X stabilizers: 24
Z stabilizers: 24
Ancilla qubits: 48 (one per stabilizer)
"""

from collections.abc import Callable

from guppylang import guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.quantum import cx, h, measure, measure_array, qubit, x


@guppy.struct
class SurfaceCode_7x7:
    """Surface code patch with dx=7, dz=7 (49 data qubits)."""

    data: array[qubit, 49]


@guppy.struct
class Syndrome_7x7:
    """Syndrome for dx=7, dz=7 patch."""

    synx: array[bool, 24]
    synz: array[bool, 24]


# === State Preparation ===


@guppy
def prep_z_basis() -> SurfaceCode_7x7:
    """Prepare logical |0_L> state."""
    data = array(qubit() for _ in range(49))
    return SurfaceCode_7x7(data)


@guppy
def prep_x_basis() -> SurfaceCode_7x7:
    """Prepare logical |+_L> state."""
    data = array(qubit() for _ in range(49))
    for i in range(49):
        h(data[i])
    return SurfaceCode_7x7(data)


# === Syndrome Extraction ===


@guppy
def syndrome_extraction(surf: SurfaceCode_7x7) -> Syndrome_7x7:
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
    ax12 = qubit()
    ax13 = qubit()
    ax14 = qubit()
    ax15 = qubit()
    ax16 = qubit()
    ax17 = qubit()
    ax18 = qubit()
    ax19 = qubit()
    ax20 = qubit()
    ax21 = qubit()
    ax22 = qubit()
    ax23 = qubit()
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
    az12 = qubit()
    az13 = qubit()
    az14 = qubit()
    az15 = qubit()
    az16 = qubit()
    az17 = qubit()
    az18 = qubit()
    az19 = qubit()
    az20 = qubit()
    az21 = qubit()
    az22 = qubit()
    az23 = qubit()

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
    h(ax12)
    h(ax13)
    h(ax14)
    h(ax15)
    h(ax16)
    h(ax17)
    h(ax18)
    h(ax19)
    h(ax20)
    h(ax21)
    h(ax22)
    h(ax23)

    # Round 1
    cx(surf.data[35], az0)
    cx(surf.data[21], az1)
    cx(surf.data[7], az2)
    cx(ax3, surf.data[2])
    cx(surf.data[29], az3)
    cx(ax4, surf.data[4])
    cx(surf.data[15], az4)
    cx(ax5, surf.data[6])
    cx(surf.data[1], az5)
    cx(ax6, surf.data[8])
    cx(surf.data[37], az6)
    cx(ax7, surf.data[10])
    cx(surf.data[23], az7)
    cx(ax8, surf.data[12])
    cx(surf.data[9], az8)
    cx(ax9, surf.data[16])
    cx(surf.data[31], az9)
    cx(ax10, surf.data[18])
    cx(surf.data[17], az10)
    cx(ax11, surf.data[20])
    cx(surf.data[3], az11)
    cx(ax12, surf.data[22])
    cx(surf.data[39], az12)
    cx(ax13, surf.data[24])
    cx(surf.data[25], az13)
    cx(ax14, surf.data[26])
    cx(surf.data[11], az14)
    cx(ax15, surf.data[30])
    cx(surf.data[33], az15)
    cx(ax16, surf.data[32])
    cx(surf.data[19], az16)
    cx(ax17, surf.data[34])
    cx(surf.data[5], az17)
    cx(ax18, surf.data[36])
    cx(surf.data[41], az18)
    cx(ax19, surf.data[38])
    cx(surf.data[27], az19)
    cx(ax20, surf.data[40])
    cx(surf.data[13], az20)
    cx(ax21, surf.data[44])
    cx(ax22, surf.data[46])
    cx(ax23, surf.data[48])

    # Round 2
    cx(surf.data[42], az0)
    cx(surf.data[28], az1)
    cx(surf.data[14], az2)
    cx(ax3, surf.data[1])
    cx(surf.data[36], az3)
    cx(ax4, surf.data[3])
    cx(surf.data[22], az4)
    cx(ax5, surf.data[5])
    cx(surf.data[8], az5)
    cx(ax6, surf.data[7])
    cx(surf.data[44], az6)
    cx(ax7, surf.data[9])
    cx(surf.data[30], az7)
    cx(ax8, surf.data[11])
    cx(surf.data[16], az8)
    cx(ax9, surf.data[15])
    cx(surf.data[38], az9)
    cx(ax10, surf.data[17])
    cx(surf.data[24], az10)
    cx(ax11, surf.data[19])
    cx(surf.data[10], az11)
    cx(ax12, surf.data[21])
    cx(surf.data[46], az12)
    cx(ax13, surf.data[23])
    cx(surf.data[32], az13)
    cx(ax14, surf.data[25])
    cx(surf.data[18], az14)
    cx(ax15, surf.data[29])
    cx(surf.data[40], az15)
    cx(ax16, surf.data[31])
    cx(surf.data[26], az16)
    cx(ax17, surf.data[33])
    cx(surf.data[12], az17)
    cx(ax18, surf.data[35])
    cx(surf.data[48], az18)
    cx(ax19, surf.data[37])
    cx(surf.data[34], az19)
    cx(ax20, surf.data[39])
    cx(surf.data[20], az20)
    cx(ax21, surf.data[43])
    cx(ax22, surf.data[45])
    cx(ax23, surf.data[47])

    # Round 3
    cx(ax0, surf.data[1])
    cx(ax1, surf.data[3])
    cx(ax2, surf.data[5])
    cx(ax3, surf.data[9])
    cx(surf.data[28], az3)
    cx(ax4, surf.data[11])
    cx(surf.data[14], az4)
    cx(ax5, surf.data[13])
    cx(surf.data[0], az5)
    cx(ax6, surf.data[15])
    cx(surf.data[36], az6)
    cx(ax7, surf.data[17])
    cx(surf.data[22], az7)
    cx(ax8, surf.data[19])
    cx(surf.data[8], az8)
    cx(ax9, surf.data[23])
    cx(surf.data[30], az9)
    cx(ax10, surf.data[25])
    cx(surf.data[16], az10)
    cx(ax11, surf.data[27])
    cx(surf.data[2], az11)
    cx(ax12, surf.data[29])
    cx(surf.data[38], az12)
    cx(ax13, surf.data[31])
    cx(surf.data[24], az13)
    cx(ax14, surf.data[33])
    cx(surf.data[10], az14)
    cx(ax15, surf.data[37])
    cx(surf.data[32], az15)
    cx(ax16, surf.data[39])
    cx(surf.data[18], az16)
    cx(ax17, surf.data[41])
    cx(surf.data[4], az17)
    cx(ax18, surf.data[43])
    cx(surf.data[40], az18)
    cx(ax19, surf.data[45])
    cx(surf.data[26], az19)
    cx(ax20, surf.data[47])
    cx(surf.data[12], az20)
    cx(surf.data[34], az21)
    cx(surf.data[20], az22)
    cx(surf.data[6], az23)

    # Round 4
    cx(ax0, surf.data[0])
    cx(ax1, surf.data[2])
    cx(ax2, surf.data[4])
    cx(ax3, surf.data[8])
    cx(surf.data[35], az3)
    cx(ax4, surf.data[10])
    cx(surf.data[21], az4)
    cx(ax5, surf.data[12])
    cx(surf.data[7], az5)
    cx(ax6, surf.data[14])
    cx(surf.data[43], az6)
    cx(ax7, surf.data[16])
    cx(surf.data[29], az7)
    cx(ax8, surf.data[18])
    cx(surf.data[15], az8)
    cx(ax9, surf.data[22])
    cx(surf.data[37], az9)
    cx(ax10, surf.data[24])
    cx(surf.data[23], az10)
    cx(ax11, surf.data[26])
    cx(surf.data[9], az11)
    cx(ax12, surf.data[28])
    cx(surf.data[45], az12)
    cx(ax13, surf.data[30])
    cx(surf.data[31], az13)
    cx(ax14, surf.data[32])
    cx(surf.data[17], az14)
    cx(ax15, surf.data[36])
    cx(surf.data[39], az15)
    cx(ax16, surf.data[38])
    cx(surf.data[25], az16)
    cx(ax17, surf.data[40])
    cx(surf.data[11], az17)
    cx(ax18, surf.data[42])
    cx(surf.data[47], az18)
    cx(ax19, surf.data[44])
    cx(surf.data[33], az19)
    cx(ax20, surf.data[46])
    cx(surf.data[19], az20)
    cx(surf.data[41], az21)
    cx(surf.data[27], az22)
    cx(surf.data[13], az23)

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
    h(ax12)
    h(ax13)
    h(ax14)
    h(ax15)
    h(ax16)
    h(ax17)
    h(ax18)
    h(ax19)
    h(ax20)
    h(ax21)
    h(ax22)
    h(ax23)

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
    sx12 = measure(ax12)
    sx13 = measure(ax13)
    sx14 = measure(ax14)
    sx15 = measure(ax15)
    sx16 = measure(ax16)
    sx17 = measure(ax17)
    sx18 = measure(ax18)
    sx19 = measure(ax19)
    sx20 = measure(ax20)
    sx21 = measure(ax21)
    sx22 = measure(ax22)
    sx23 = measure(ax23)
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
    sz12 = measure(az12)
    sz13 = measure(az13)
    sz14 = measure(az14)
    sz15 = measure(az15)
    sz16 = measure(az16)
    sz17 = measure(az17)
    sz18 = measure(az18)
    sz19 = measure(az19)
    sz20 = measure(az20)
    sz21 = measure(az21)
    sz22 = measure(az22)
    sz23 = measure(az23)

    synx = array(
        sx0,
        sx1,
        sx2,
        sx3,
        sx4,
        sx5,
        sx6,
        sx7,
        sx8,
        sx9,
        sx10,
        sx11,
        sx12,
        sx13,
        sx14,
        sx15,
        sx16,
        sx17,
        sx18,
        sx19,
        sx20,
        sx21,
        sx22,
        sx23,
    )
    synz = array(
        sz0,
        sz1,
        sz2,
        sz3,
        sz4,
        sz5,
        sz6,
        sz7,
        sz8,
        sz9,
        sz10,
        sz11,
        sz12,
        sz13,
        sz14,
        sz15,
        sz16,
        sz17,
        sz18,
        sz19,
        sz20,
        sz21,
        sz22,
        sz23,
    )

    return Syndrome_7x7(synx, synz)


# === Measurement ===


@guppy
def measure_z_basis(surf: SurfaceCode_7x7 @ owned) -> array[bool, 49]:
    """Destructively measure in Z basis."""
    return measure_array(surf.data)


@guppy
def measure_x_basis(surf: SurfaceCode_7x7 @ owned) -> array[bool, 49]:
    """Destructively measure in X basis."""
    for i in range(49):
        h(surf.data[i])
    return measure_array(surf.data)


# === Logical Operators ===


@guppy
def apply_logical_x(surf: SurfaceCode_7x7) -> None:
    """Apply logical X (string along left edge)."""
    x(surf.data[0])
    x(surf.data[7])
    x(surf.data[14])
    x(surf.data[21])
    x(surf.data[28])
    x(surf.data[35])
    x(surf.data[42])


@guppy
def apply_logical_z(surf: SurfaceCode_7x7) -> None:
    """Apply logical Z (string along top edge)."""
    from guppylang.std.quantum import z

    z(surf.data[0])
    z(surf.data[1])
    z(surf.data[2])
    z(surf.data[3])
    z(surf.data[4])
    z(surf.data[5])
    z(surf.data[6])


# === Memory Experiments ===


def make_memory_z(num_rounds: int) -> Callable[[], None]:
    """Create Z-basis memory experiment."""
    from guppylang.std.builtins import comptime

    @guppy
    def memory_z() -> None:
        """Z-basis memory experiment for dx=7, dz=7."""
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
        """X-basis memory experiment for dx=7, dz=7."""
        surf = prep_x_basis()

        for _t in range(comptime(num_rounds)):
            syn = syndrome_extraction(surf)
            result("synx", syn.synx)
            result("synz", syn.synz)

        final = measure_x_basis(surf)
        result("final", final)

    return memory_x
