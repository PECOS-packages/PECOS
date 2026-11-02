# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Export surface-code round panels as standalone SVG and TikZ.

Run with an output directory argument. Existing named files are replaced.
The grouping below is specific to the surface circuit builder's metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
from pecos.visualization import circuit_diagram_panels

if TYPE_CHECKING:
    from pecos.quantum import TickCircuit


def surface_panels(circuit: TickCircuit) -> list[tuple[str, int, int]]:
    """Group contiguous ticks using the surface builder's phase and round tags."""
    panels: list[tuple[str, int, int]] = []
    for tick in range(circuit.num_ticks()):
        phase = circuit.get_tick_meta(tick, "phase")
        round_index = circuit.get_tick_meta(tick, "syndrome_round")
        # Final readout retains the last round tag: phase takes precedence.
        if phase == "prep_data":
            title = "Data preparation"
        elif phase == "measure_data":
            title = "Final data readout"
        elif round_index == -1:
            title = "Initial syndrome"
        elif isinstance(round_index, int) and round_index >= 0:
            title = f"Syndrome round {round_index}"
        else:
            msg = f"Missing surface-builder context at tick {tick}: {phase!r}, {round_index!r}"
            raise ValueError(msg)
        if panels and panels[-1][0] == title:
            panels[-1] = (title, panels[-1][1], tick + 1)
        else:
            panels.append((title, tick, tick + 1))
    return panels


def main() -> None:
    """Write panels for a distance-three surface-code memory circuit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    circuit = generate_tick_circuit_from_patch(SurfacePatch.create(distance=3), 3, "Z")
    selections = surface_panels(circuit)
    for output_format, suffix in [("svg", "svg"), ("tikz", "tex")]:
        for index, panel in enumerate(
            circuit_diagram_panels(circuit, selections, format=output_format, show_measurement_ids=True),
        ):
            panel.save(args.directory / f"surface-panel-{index}.{suffix}")
    for title, start, stop in selections:
        print(f"{title}: ticks {start}..{stop}")


if __name__ == "__main__":
    main()
