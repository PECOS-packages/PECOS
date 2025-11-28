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
from pecos.slr.transforms.parallel_optimizer import ParallelOptimizer

try:
    from pecos.slr.gen_codes.gen_qir import QIRGenerator
except ImportError:
    QIRGenerator = None

try:
    from pecos.slr.gen_codes.guppy import IRGuppyGenerator
except ImportError:
    IRGuppyGenerator = None

try:
    from pecos.slr.gen_codes.gen_stim import StimGenerator
except ImportError:
    StimGenerator = None

try:
    from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
except ImportError:
    QuantumCircuitGenerator = None


class SlrConverter:

    def __init__(self, block=None, *, optimize_parallel: bool = True):
        """Initialize the SLR converter.

        Args:
            block: The SLR block to convert (optional for using from_* methods)
            optimize_parallel: Whether to apply ParallelOptimizer transformation (default: True).
                             Only affects blocks containing Parallel() statements.
        """
        self._block = block
        self._optimize_parallel = optimize_parallel

        # Apply transformations if requested and block is provided
        if block is not None and optimize_parallel:
            optimizer = ParallelOptimizer()
            self._block = optimizer.transform(self._block)

    def generate(
        self,
        target: Language,
        *,
        skip_headers: bool = False,
        add_versions: bool = False,
    ) -> str:
        if target == Language.QASM:
            generator = QASMGenerator(
                skip_headers=skip_headers,
                add_versions=add_versions,
            )
        elif target in [Language.QIR, Language.QIRBC]:
            self._check_qir_imported()
            generator = QIRGenerator()
        elif target == Language.GUPPY:
            self._check_guppy_imported()
            generator = IRGuppyGenerator()
        elif target == Language.HUGR:
            # HUGR is handled specially in the hugr() method
            msg = "Use the hugr() method directly to compile to HUGR"
            raise ValueError(msg)
        elif target == Language.STIM:
            self._check_stim_imported()
            generator = StimGenerator()
        elif target == Language.QUANTUM_CIRCUIT:
            generator = QuantumCircuitGenerator()
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
        return self.generate(
            Language.QASM,
            skip_headers=skip_headers,
            add_versions=add_versions,
        )

    def qir(self):
        self._check_qir_imported()
        return self.generate(Language.QIR)

    def qir_bc(self):
        self._check_qir_imported()
        return self.generate(Language.QIRBC)

    @staticmethod
    def _check_guppy_imported():
        if IRGuppyGenerator is None:
            msg = (
                "Trying to compile to Guppy without the IRGuppyGenerator. "
                "Make sure ir_generator.py is available."
            )
            raise Exception(msg)

    def guppy(self):
        self._check_guppy_imported()
        return self.generate(Language.GUPPY)

    def hugr(self):
        """Compile the SLR block to HUGR via Guppy.

        Returns:
            The compiled HUGR module

        Raises:
            ImportError: If guppylang is not available
            RuntimeError: If compilation fails
        """
        self._check_guppy_imported()

        # First generate Guppy code
        generator = IRGuppyGenerator()
        generator.generate_block(self._block)

        # Then compile to HUGR
        try:
            from pecos.slr.gen_codes.guppy.hugr_compiler import HugrCompiler
        except ImportError as e:
            msg = "Failed to import HugrCompiler. Make sure guppylang is installed."
            raise ImportError(msg) from e

        compiler = HugrCompiler(generator)
        return compiler.compile_to_hugr()

    @staticmethod
    def _check_stim_imported():
        if StimGenerator is None:
            msg = (
                "Trying to compile to Stim without the StimGenerator. "
                "Make sure gen_stim.py is available."
            )
            raise Exception(msg)
        # Also check if stim itself is available
        import importlib.util

        if importlib.util.find_spec("stim") is None:
            msg = (
                "Stim is not installed. To use Stim conversion features, install with:\n"
                "  pip install quantum-pecos[stim]\n"
                "or:\n"
                "  pip install stim"
            )
            raise ImportError(msg)

    def stim(self):
        """Generate a Stim circuit from the SLR block.

        Returns:
            stim.Circuit: The generated Stim circuit
        """
        if self._block is None:
            msg = "No SLR block to convert. Use from_* methods first or provide block to constructor."
            raise ValueError(msg)
        self._check_stim_imported()
        generator = StimGenerator()
        generator.generate_block(self._block)
        return generator.get_circuit()

    def quantum_circuit(self):
        """Generate a PECOS QuantumCircuit from the SLR block.

        Returns:
            QuantumCircuit: The generated QuantumCircuit object
        """
        if self._block is None:
            msg = "No SLR block to convert. Use from_* methods first or provide block to constructor."
            raise ValueError(msg)
        generator = QuantumCircuitGenerator()
        generator.generate_block(self._block)
        return generator.get_circuit()

    # ===== Conversion TO SLR from other formats =====

    @classmethod
    def from_stim(cls, circuit, *, optimize_parallel: bool = True):
        """Convert a Stim circuit to SLR format.

        Args:
            circuit: A Stim circuit object
            optimize_parallel: Whether to apply ParallelOptimizer transformation

        Returns:
            Block: The converted SLR block (Main object)

        Note:
            - Stim's measurement record and detector/observable annotations are preserved as comments
            - Noise operations are converted to comments (SLR typically handles noise differently)
            - Some Stim-specific features may not have direct SLR equivalents
        """
        try:
            from pecos.slr.converters.from_stim import stim_to_slr
        except ImportError as e:
            msg = "Failed to import stim_to_slr converter"
            raise ImportError(msg) from e

        slr_block = stim_to_slr(circuit)
        if optimize_parallel:
            from pecos.slr.transforms.parallel_optimizer import ParallelOptimizer

            optimizer = ParallelOptimizer()
            slr_block = optimizer.transform(slr_block)
        return slr_block

    @classmethod
    def from_quantum_circuit(cls, qc, *, optimize_parallel: bool = True):
        """Convert a PECOS QuantumCircuit to SLR format.

        Args:
            qc: A PECOS QuantumCircuit object
            optimize_parallel: Whether to apply ParallelOptimizer transformation

        Returns:
            Block: The converted SLR block (Main object)

        Note:
            - QuantumCircuit's parallel gate structure is preserved
            - Assumes standard gate names from PECOS
        """
        try:
            from pecos.slr.converters.from_quantum_circuit import quantum_circuit_to_slr
        except ImportError as e:
            msg = "Failed to import quantum_circuit_to_slr converter"
            raise ImportError(msg) from e

        slr_block = quantum_circuit_to_slr(qc)
        if optimize_parallel:
            from pecos.slr.transforms.parallel_optimizer import ParallelOptimizer

            optimizer = ParallelOptimizer()
            slr_block = optimizer.transform(slr_block)
        return slr_block
