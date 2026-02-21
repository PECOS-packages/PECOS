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

"""SLR Converter - converts SLR programs to various output formats.

This module uses AST-based code generation for all targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.gen_codes.language import Language
from pecos.slr.transforms.parallel_optimizer import ParallelOptimizer

if TYPE_CHECKING:
    import stim

    from pecos.circuits import QuantumCircuit
    from pecos.slr import Main


class SlrConverter:
    """Convert SLR programs to various output formats.

    Uses AST-based code generation which provides validation,
    analysis, and optimization capabilities.
    """

    def __init__(self, block: Main | None = None, *, optimize_parallel: bool = True):
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

    def _to_ast(self):
        """Convert the SLR block to AST."""
        if self._block is None:
            msg = "No SLR block to convert. Use from_* methods first or provide block to constructor."
            raise ValueError(msg)

        from pecos.slr.ast import slr_to_ast

        return slr_to_ast(self._block)

    def generate(
        self,
        target: Language,
        *,
        skip_headers: bool = False,
        add_versions: bool = False,
    ) -> str:
        """Generate code for the specified target language.

        Args:
            target: The target language (Language enum value)
            skip_headers: For QASM, whether to skip the OPENQASM header
            add_versions: Deprecated, ignored (kept for backwards compatibility)

        Returns:
            Generated code as a string
        """
        del add_versions  # Deprecated parameter, kept for backwards compatibility
        if target == Language.QASM:
            return self._generate_qasm(include_header=not skip_headers)
        if target in [Language.QIR, Language.QIRBC]:
            return self._generate_qir(bytecode=(target == Language.QIRBC))
        if target == Language.GUPPY:
            return self._generate_guppy()
        if target == Language.HUGR:
            msg = "Use the hugr() method directly to compile to HUGR"
            raise ValueError(msg)
        if target == Language.STIM:
            # For backwards compatibility, generate() returns string
            return str(self.stim())
        if target == Language.QUANTUM_CIRCUIT:
            # For backwards compatibility, generate() returns string representation
            return str(self.quantum_circuit())
        msg = f"Code gen target '{target}' is not supported."
        raise NotImplementedError(msg)

    def _generate_qasm(self, *, include_header: bool = True) -> str:
        """Generate QASM code using AST-based codegen."""
        from pecos.slr.ast.codegen.qasm import ast_to_qasm

        ast = self._to_ast()
        return ast_to_qasm(ast, include_header=include_header)

    def _generate_guppy(self) -> str:
        """Generate Guppy code using AST-based codegen."""
        from pecos.slr.ast.codegen.guppy import ast_to_guppy

        ast = self._to_ast()
        return ast_to_guppy(ast)

    def _generate_qir(self, *, bytecode: bool = False) -> str | bytes:
        """Generate QIR code using AST-based codegen."""
        if bytecode:
            # QIR bytecode requires the old generator
            from pecos.slr.gen_codes.gen_qir import QIRGenerator

            if QIRGenerator is None:
                msg = (
                    "Trying to compile QIR without the appropriate optional dependencies install. "
                    "Use optional dependency group `qir` or `all`"
                )
                raise ImportError(msg)

            generator = QIRGenerator(_internal=True)
            generator.generate_block(self._block)
            return generator.get_bc()

        from pecos.slr.ast.codegen.qir import ast_to_qir

        ast = self._to_ast()
        return ast_to_qir(ast)

    def qasm(self, *, skip_headers: bool = False, add_versions: bool = False) -> str:
        """Generate QASM code.

        Args:
            skip_headers: Whether to skip the OPENQASM header
            add_versions: Deprecated, ignored (kept for backwards compatibility)

        Returns:
            Generated QASM code as a string
        """
        del add_versions  # Deprecated parameter, kept for backwards compatibility
        return self._generate_qasm(include_header=not skip_headers)

    def qir(self) -> str:
        """Generate QIR code.

        Returns:
            Generated QIR code as a string
        """
        return self._generate_qir()

    def qir_bc(self) -> bytes:
        """Generate QIR bytecode.

        Returns:
            Generated QIR bytecode
        """
        return self._generate_qir(bytecode=True)

    def guppy(self) -> str:
        """Generate Guppy code.

        Returns:
            Generated Guppy code as a string
        """
        return self._generate_guppy()

    def hugr(self):
        """Compile the SLR block to HUGR via Guppy.

        Returns:
            The compiled HUGR module

        Raises:
            ImportError: If guppylang is not available
            RuntimeError: If compilation fails
        """
        # Generate Guppy code
        self._generate_guppy()

        # Compile to HUGR
        try:
            from pecos.slr.gen_codes.guppy.hugr_compiler import HugrCompiler
        except ImportError as e:
            msg = "Failed to import HugrCompiler. Make sure guppylang is installed."
            raise ImportError(msg) from e

        # HugrCompiler needs the generator object for its internal state
        # For now, fall back to the old path
        from pecos.slr.gen_codes.guppy import IRGuppyGenerator

        generator = IRGuppyGenerator(_internal=True)
        generator.generate_block(self._block)

        compiler = HugrCompiler(generator)
        return compiler.compile_to_hugr()

    def stim(self) -> stim.Circuit:
        """Generate a Stim circuit from the SLR block.

        Returns:
            stim.Circuit: The generated Stim circuit
        """
        from pecos.slr.ast.codegen.stim import ast_to_stim

        ast = self._to_ast()
        return ast_to_stim(ast)

    def quantum_circuit(self) -> QuantumCircuit:
        """Generate a PECOS QuantumCircuit from the SLR block.

        Returns:
            QuantumCircuit: The generated QuantumCircuit object
        """
        from pecos.slr.ast.codegen.quantum_circuit import ast_to_quantum_circuit

        ast = self._to_ast()
        return ast_to_quantum_circuit(ast)

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
            optimizer = ParallelOptimizer()
            slr_block = optimizer.transform(slr_block)
        return slr_block
