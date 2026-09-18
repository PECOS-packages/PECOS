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
own rendering; an accepted row whose canonical rendering is empty has no third
field. Inside a field a backslash is ``\\``, newline is ``\n``, tab is ``\t``,
and any other character outside printable ASCII is ``\u{hex}``. The forms cover target spacing, separators, tags,
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
    "",
    "   ",
    "\x0cerror(0.1) D0",
    "\x0berror(0.1) D0",
    "\u00a0error(0.1) D0",
    "error(0.1)\u00a0D0",
    "error(0.1) D4294967295",
    "error(0.1) D1152921504606846975",
    "error(0.1) D1152921504606846976",
]


def escape(text: str) -> str:
    """Escape a field so it stays on one line of printable ASCII."""
    out = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif " " <= char <= "~":
            out.append(char)
        else:
            out.append(f"\\u{{{ord(char):x}}}")
    return "".join(out)


def main() -> None:
    """Print the oracle table for every form in ``FORMS``."""
    print("input\tverdict\tcanonical")
    for form in FORMS:
        try:
            canonical = str(stim.DetectorErrorModel(form)).strip()
        except (ValueError, IndexError):
            # Stim raises IndexError for some malformed text, such as a leading
            # non-breaking space or an index above its 2**60 - 1 ceiling.
            print(f"{escape(form)}\treject")
        else:
            row = f"{escape(form)}\taccept"
            print(f"{row}\t{escape(canonical)}" if canonical else row)


if __name__ == "__main__":
    main()
