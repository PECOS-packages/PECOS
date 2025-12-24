"""HUGR compiler for Guppy code generation."""

from __future__ import annotations

import tempfile
from typing import Any

from pecos.slr.gen_codes.guppy.hugr_error_handler import HugrErrorHandler

try:
    # Check if guppylang is available by attempting actual imports
    # We need these imports to verify the environment is properly configured
    import guppylang

    # Verify key attributes/modules are accessible
    _ = guppylang.guppy  # This will raise AttributeError if not available

    # Test importing the std modules
    from guppylang.std import quantum
    from guppylang.std.builtins import array, owned, result

    # If we get here, all imports worked
    GUPPY_AVAILABLE = True

    # Clean up namespace - we don't need these imports here
    del guppylang, quantum, array, owned, result

except (ImportError, AttributeError) as e:
    # For debugging - we want to know what specific import failed
    import warnings

    warnings.warn(f"guppylang import failed: {e}")
    GUPPY_AVAILABLE = False


class HugrCompiler:
    """Compiles generated Guppy code to HUGR."""

    def __init__(self, generator):
        """Initialize the HUGR compiler.

        Args:
            generator: A generator instance with generated code (must have get_output() method)
        """
        self.generator = generator

    def compile_to_hugr(self) -> Any:
        """Compile the generated Guppy code to HUGR.

        Returns:
            The compiled HUGR module

        Raises:
            ImportError: If guppylang is not available
            RuntimeError: If compilation fails
        """
        if not GUPPY_AVAILABLE:
            msg = "guppylang is not installed. Install it with: pip install guppylang"
            raise ImportError(
                msg,
            )

        # Get the generated Guppy code
        guppy_code = self.generator.get_output()

        # Create a temporary file to hold the generated code
        # This is necessary because guppy.compile() needs to be able to inspect the source
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_file = f.name
            f.write(guppy_code)

        try:
            # Import the module from the temporary file
            import importlib.util
            import linecache
            import sys

            # Add the source to linecache for better error tracking
            lines = guppy_code.splitlines(keepends=True)
            linecache.cache[temp_file] = (
                len(guppy_code),
                None,
                lines,
                temp_file,
            )

            spec = importlib.util.spec_from_file_location("_guppy_generated", temp_file)
            if spec is None or spec.loader is None:
                msg = "Failed to create module spec"
                raise RuntimeError(msg)

            module = importlib.util.module_from_spec(spec)

            # Ensure the module has proper file tracking
            module.__file__ = temp_file

            # Add to sys.modules temporarily to help with source tracking
            sys.modules["_guppy_generated"] = module

            spec.loader.exec_module(module)

            # Get the main function
            if not hasattr(module, "main"):
                msg = "No main function found in generated code"
                raise RuntimeError(msg)

            main_func = module.main

            # Compile to HUGR
            try:
                # Debug: print the generated code
                # print("DEBUG: Generated Guppy code:")
                # print(guppy_code)
                # print("="*50)

                # Use the new API: func.compile() instead of guppy.compile(func)
                return main_func.compile()
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                # Use the enhanced error handler
                error_handler = HugrErrorHandler(guppy_code)
                detailed_error = error_handler.analyze_error(e)
                raise RuntimeError(detailed_error)

        finally:
            # Clean up
            try:
                # Remove from sys.modules
                import sys

                if "_guppy_generated" in sys.modules:
                    del sys.modules["_guppy_generated"]

                # Clean up the temporary file
                from pathlib import Path

                Path(temp_file).unlink()
            except (OSError, FileNotFoundError):
                # Ignore cleanup errors
                pass
