"""HUGR to LLVM compiler integration for PECOS.

This module provides a Python interface to compile HUGR files to LLVM IR using
the working quantum compilation pipeline from quantum-compilation-examples.
"""

import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class HugrLlvmCompiler:
    """Compiler that converts HUGR files to LLVM IR with quantum operations.

    This uses the working quantum compilation pipeline from the
    quantum-compilation-examples project.
    """

    def __init__(self, hugr_llvm_binary: Path | None = None) -> None:
        """Initialize the HUGR→LLVM compiler.

        Args:
            hugr_llvm_binary: Path to the hugr-to-llvm compiler binary.
                             If None, will try to find it automatically.
        """
        self.hugr_llvm_binary = hugr_llvm_binary or self._find_hugr_llvm_binary()
        self._temp_dir = None

    def _find_hugr_llvm_binary(self) -> Path | None:
        """Find the hugr-to-llvm compiler binary."""
        # Check common locations relative to PECOS
        base_dir = Path(__file__).parent.parent.parent.parent.parent.parent

        possible_paths = [
            # In quantum-compilation-examples
            base_dir
            / "quantum-compilation-examples/hugr_quantum_llvm/target/release/hugr_quantum_llvm",
            base_dir
            / "quantum-compilation-examples/hugr_quantum_llvm/target/debug/hugr_quantum_llvm",
            # Built versions
            Path("./hugr_quantum_llvm"),
            Path("hugr_quantum_llvm"),
        ]

        for path in possible_paths:
            if path.exists() and os.access(path, os.X_OK):
                return path.resolve()

        return None

    def compile_hugr_to_llvm(
        self,
        hugr_bytes: bytes,
        output_file: Path | None = None,
    ) -> str:
        """Compile HUGR bytes to LLVM IR.

        Args:
            hugr_bytes: HUGR package as bytes
            output_file: Optional output file path

        Returns:
            LLVM IR as string (HUGR convention)

        Raises:
            RuntimeError: If compilation fails
        """
        if not self.hugr_llvm_binary:
            msg = (
                "HUGR→LLVM compiler not found. "
                "Build it from quantum-compilation-examples/hugr_quantum_llvm/"
            )
            raise RuntimeError(
                msg,
            )

        # Create temporary directory
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="hugr_llvm_")

        temp_path = Path(self._temp_dir)

        # Write HUGR to temporary file
        hugr_file = temp_path / "input.hugr"
        with Path(hugr_file).open("wb") as f:
            f.write(hugr_bytes)

        # Determine output file
        llvm_file = temp_path / "output.ll" if output_file is None else output_file

        # Run the compiler
        try:
            cmd = [
                str(self.hugr_llvm_binary),
                str(hugr_file),
                str(llvm_file),
                "hugr",  # Always use HUGR convention
            ]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            # Read the generated LLVM IR
            with Path(llvm_file).open() as f:
                return f.read()

        except subprocess.CalledProcessError as e:
            msg = f"HUGR→LLVM compilation failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            msg = f"Compiler binary not found: {self.hugr_llvm_binary}"
            raise RuntimeError(msg) from e

    def is_available(self) -> bool:
        """Check if the HUGR→LLVM compiler is available."""
        return self.hugr_llvm_binary is not None and self.hugr_llvm_binary.exists()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if (
            hasattr(self, "_temp_dir")
            and self._temp_dir
            and Path(self._temp_dir).exists()
        ):

            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self) -> None:
        """Cleanup on destruction."""
        with contextlib.suppress(Exception):
            self.cleanup()


def compile_hugr_bytes_to_llvm(
    hugr_bytes: bytes,
) -> str:
    """Convenience function to compile HUGR bytes to LLVM IR.

    Args:
        hugr_bytes: HUGR package as bytes

    Returns:
        LLVM IR as string (HUGR convention)
    """
    compiler = HugrLlvmCompiler()
    try:
        return compiler.compile_hugr_to_llvm(hugr_bytes)
    finally:
        compiler.cleanup()


def build_hugr_llvm_compiler() -> bool:
    """Build the HUGR→LLVM compiler if source is available.

    Returns:
        True if build succeeded, False otherwise
    """
    # Find the source directory
    base_dir = Path(__file__).parent.parent.parent.parent.parent.parent
    source_dir = base_dir / "quantum-compilation-examples/hugr_quantum_llvm"

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return False

    try:
        # Build in release mode
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=source_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[ERROR] cargo not found - install Rust toolchain")
        return False
    else:
        binary_path = source_dir / "target/release/hugr_quantum_llvm"
        if binary_path.exists():
            print(f"[PASS] Built HUGR->LLVM compiler: {binary_path}")
            return True
        print("[ERROR] Build succeeded but binary not found")
        return False


if __name__ == "__main__":
    # Test the compiler
    print("Testing HUGR->LLVM compiler...")

    compiler = HugrLlvmCompiler()

    if compiler.is_available():
        print(f"[PASS] Compiler available: {compiler.hugr_llvm_binary}")
    else:
        print("[ERROR] Compiler not available")
        print("Attempting to build...")
        if build_hugr_llvm_compiler():
            compiler = HugrLlvmCompiler()  # Re-initialize to find new binary
            if compiler.is_available():
                print(f"[PASS] Compiler now available: {compiler.hugr_llvm_binary}")
        else:
            print("[ERROR] Failed to build compiler")
