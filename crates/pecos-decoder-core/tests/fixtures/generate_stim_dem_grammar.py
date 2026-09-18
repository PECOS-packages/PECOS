# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
r"""Offline generator for the Stim DEM grammar oracle table.

Stim's own parser is the oracle of record for which flat detector-error-model
text PECOS's parsers accept. Run offline with:

    uv run --with stim==1.15.0 python generate_stim_dem_grammar.py > stim_dem_grammar.tsv

Each row is ``input<TAB>verdict`` for rejected text and
``input<TAB>accept<TAB>canonical`` for accepted text, where canonical is Stim's
own rendering. Backslash, newline and tab inside a field are escaped as
``\\``, ``\n`` and ``\t``. The forms cover target spacing, separators, tags,
comments, case, arguments, declarations and the loop instructions PECOS does
not implement.

Do not add an unclosed tag such as ``error[unclosed(0.1) D0``: Stim 1.15.0
exhausts memory on it instead of raising, and the process is killed.
"""

from __future__ import annotations

import stim

FORMS = [
    "error(0.1) D0 D1 L0",
    "error(0.1) D0 D3 ^ D1 D2",
    "error(0.1) D0 D3^D1 D2",
    "error(0.1) D0 D3^ D1 D2",
    "error(0.1) D0 D3 ^D1 D2",
    "error(0.1)D3 D0",
    "error(0.1) D3 D0 ^",
    "error(0.1) ^ D3 D0",
    "error(0.1) D0 ^ ^ D1",
    "error(0.1) D0 ^ D0",
    "error(0.1) D0 L0 ^ L1 D1",
    "error(0.1) D0 ^ L0",
    "error( 0.1 ) D0",
    "error (0.1) D0",
    "error(0.1)  D0\tD1",
    "  error(0.1) D0",
    "error(0.1) D0 # comment",
    "error(0.1) D0\t# c",
    "error(0.1) D0 #c ^ D1",
    "error(0.1)  # only comment",
    "#c\nerror(0.1) D0",
    "error(0.1) D0 D0",
    "error(0.1) L0",
    "error(0.1)",
    "error() D0",
    "error D0",
    "error(0) D0",
    "error(1) D0",
    "error(1.5) D0",
    "error(1e-3) D0",
    "error(.5) D0",
    "error(5e-1) D0",
    "error(0.1,0.2) D0",
    "error[tag](0.1) D0",
    "ERROR(0.1) D0",
    "Error(0.1) D0",
    "error(0.1) d0",
    "error(0.1) D00",
    "error(0.1) D0 L00",
    "error(0.1) D4294967296",
    "error(0.1) D-1",
    "error(0.1) D 0",
    "error(0.1) D0x",
    "error(0.1) D0 \\\n D1",
    "detector D0",
    "detector(1,2) D0",
    "detector(1.5, 2) D0",
    "detector(1,2)  D0 # c",
    "detector(1,2)D0",
    "detector D0 D1",
    "detector",
    "detector D0\ndetector D0",
    "logical_observable L0",
    "logical_observable L0 L1",
    "logical_observable L0 ^ L1",
    "logical_observable",
    "shift_detectors 1",
    "repeat 2 {\n error(0.1) D0\n}",
    "error(0.1) D0\nerror(0.1) D0",
]


def escape(text: str) -> str:
    """Escape backslash, newline and tab so a field stays on one line."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def main() -> None:
    """Print the oracle table for every form in ``FORMS``."""
    print("input\tverdict\tcanonical")
    for form in FORMS:
        try:
            canonical = str(stim.DetectorErrorModel(form)).strip()
        except ValueError:
            print(f"{escape(form)}\treject")
        else:
            print(f"{escape(form)}\taccept\t{escape(canonical)}")


if __name__ == "__main__":
    main()
