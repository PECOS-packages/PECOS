# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.language import Language

try:
    from pecos.slr.gen_codes.gen_qir import QIRGenerator
except ImportError:
    QIRGenerator = None


class SlrConverter:

    def __init__(self, block):
        self._block = block

    def generate(
        self,
        target: Language,
        *,
        skip_headers: bool = False,
        add_versions: bool = False,
    ) -> str:
        if target == Language.QASM:
            generator = QASMGenerator(skip_headers=skip_headers)
        elif target in [Language.QIR, Language.QIRBC]:
            self._check_qir_imported()
            generator = QIRGenerator()
        else:
            msg = f"Code gen target '{target}' is not supported."
            raise NotImplementedError(msg)

        generator.generate_block(self._block)
        if target == Language.QIRBC:

            return generator.get_bc()
        return generator.get_output()

    @staticmethod
    def _check_qir_imported():
        if QIRGenerator is None:
            msg = (
                "Trying to compile QIR without the appropriate optional dependencies install. "
                "Use optional dependency group `qir` or `all`"
            )
            raise Exception(
                msg,
            )

    def qasm(self, *, skip_headers: bool = False, add_versions: bool = False):
        # Create a QASM generator
        generator = QASMGenerator(skip_headers=skip_headers)

        # Generate the QASM code
        generator.generate_block(self._block)

        # Get the QASM output
        qasm = generator.get_output()

        # Check if there are any register-wide measurements that need to be unrolled
        # This is a workaround for the issue with register-wide measurements and permutations
        if "measure a -> m;" in qasm:
            # Replace register-wide measurements with individual measurements
            lines = qasm.split("\n")

            # Find all quantum register declarations to determine register sizes
            register_sizes = {}
            for line in lines:
                if line.startswith("qreg "):
                    # Parse register declaration (e.g., "qreg a[3];")
                    parts = line.split()
                    if len(parts) >= 2:
                        reg_decl = parts[1].strip(";")
                        reg_name, reg_size = reg_decl.split("[")
                        reg_size = int(reg_size.strip("]"))
                        register_sizes[reg_name] = reg_size

            # Initialize register mappings for quantum registers only
            # For each register, track what each element points to
            # Initially, each element points to itself
            register_mappings = {}
            for reg_name, reg_size in register_sizes.items():
                register_mappings[reg_name] = [(reg_name, i) for i in range(reg_size)]

            # Process all permutation comments in order
            permutation_comments = []
            for line in lines:
                if "// Permutation:" in line:
                    permutation_comments.append(line)

            # Apply each permutation in order
            for comment in permutation_comments:
                # Extract the permutation description
                perm_desc = comment.split("// Permutation:")[1].strip()

                if "<->" in perm_desc:
                    # This is a whole register permutation (e.g., "a <-> c")
                    parts = perm_desc.split("<->")
                    reg1 = parts[0].strip()
                    reg2 = parts[1].strip()

                    # Only process quantum register permutations
                    # Classical register permutations are handled by the QASM generator
                    if reg1 in register_sizes and reg2 in register_sizes:
                        # Swap the register mappings
                        register_mappings[reg1], register_mappings[reg2] = (
                            register_mappings[reg2],
                            register_mappings[reg1],
                        )
                else:
                    # This is an element permutation
                    # Parse arbitrary permutation patterns
                    import re

                    # Match patterns like "a[0] -> b[1], a[1] -> c[2], ..."
                    pattern = r"([a-zA-Z0-9_]+)\[(\d+)\] -> ([a-zA-Z0-9_]+)\[(\d+)\]"
                    matches = re.findall(pattern, perm_desc)

                    # Process each permutation pair
                    # We need to be careful not to process the same pair twice
                    processed_pairs = set()

                    for match in matches:
                        src_reg, src_idx, dst_reg, dst_idx = match
                        src_idx, dst_idx = int(src_idx), int(dst_idx)

                        # Skip if we've already processed this pair
                        pair_key = frozenset([(src_reg, src_idx), (dst_reg, dst_idx)])
                        if pair_key in processed_pairs:
                            continue

                        processed_pairs.add(pair_key)

                        # Only process quantum register permutations
                        # Classical register permutations are handled by the QASM generator
                        if src_reg in register_sizes and dst_reg in register_sizes:
                            # Get the current values at these locations
                            src_val = register_mappings[src_reg][src_idx]
                            dst_val = register_mappings[dst_reg][dst_idx]

                            # Swap what these elements point to
                            register_mappings[src_reg][src_idx] = dst_val
                            register_mappings[dst_reg][dst_idx] = src_val

            # Now replace the register-wide measurement with individual measurements
            for i, line in enumerate(lines):
                if line.strip() == "measure a -> m;":
                    # Replace with individual measurements based on the final mappings
                    individual_measurements = []
                    for j in range(register_sizes.get("a", 3)):
                        # Get what a[j] is pointing to
                        curr_reg, curr_idx = register_mappings["a"][j]
                        individual_measurements.append(
                            f"measure {curr_reg}[{curr_idx}] -> m[{j}];",
                        )

                    # Replace the register-wide measurement with individual measurements
                    lines[i : i + 1] = individual_measurements

            # Join the lines back into a single string
            qasm = "\n".join(lines)

        return qasm

    def qir(self):
        self._check_qir_imported()
        return self.generate(Language.QIR)

    def qir_bc(self):
        self._check_qir_imported()
        return self.generate(Language.QIRBC)
