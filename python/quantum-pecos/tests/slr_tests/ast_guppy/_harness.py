"""Compile harness for AST -> Guppy v1 acceptance tests.

Provides a single primitive: `assert_ast_guppy_compiles(prog)`. Takes an
SLR `Main`/`Block`, runs it through `SlrConverter.guppy()` (which is the
AST path: `slr_to_ast` -> `AstToGuppy`), writes the source to a temp
file, imports it as a fresh module, and calls `main.compile_function()`
on the resulting Guppy function.

Post-cutover, `SlrConverter.hugr()` also routes through the AST path
(wrapping `main(...)` in a no-arg `entry()` and calling
`entry.compile()`). This harness compiles `main.compile_function()`
directly so failures point at the parameterized function, not at the
entry wrapper.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pecos.slr import SlrConverter

if TYPE_CHECKING:
    from pecos.slr import Block


@dataclass(frozen=True)
class CompileFailureError(AssertionError):
    """Raised when generated Guppy source fails to compile.

    Carries the generated source for diagnostics. The exception type
    inherits from AssertionError so pytest renders it as a normal
    test failure instead of an internal error.
    """

    source: str
    cause: BaseException

    def __str__(self) -> str:
        cause_msg = f"{type(self.cause).__name__}: {self.cause}"
        # Truncate the source in repr; full source is on .source for inspection.
        max_lines = 80
        lines = self.source.splitlines()
        shown = "\n".join(lines[:max_lines])
        suffix = f"\n... ({len(lines) - max_lines} more lines truncated)" if len(lines) > max_lines else ""
        return f"{cause_msg}\n--- generated Guppy source ---\n{shown}{suffix}"


def ast_guppy_source(slr_program: Block) -> str:
    """Return the Guppy source the AST path would emit, without compiling."""
    return SlrConverter(slr_program).guppy()


def assert_ast_guppy_compiles(slr_program: Block) -> None:
    """Run SLR -> AST -> Guppy source -> compile_function. Raise on failure.

    The "main" function in the generated source is compiled via Guppy's
    `compile_function()` (works for parameterized functions, unlike
    `compile()` which expects a no-arg entrypoint).
    """
    source = ast_guppy_source(slr_program)

    # Import as a fresh module from a temp file so Guppy can attribute
    # source spans correctly in any error messages.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        path = Path(f.name)
        f.write(source)

    spec = importlib.util.spec_from_file_location(f"_ast_guppy_test_{path.stem}", path)
    if spec is None or spec.loader is None:
        msg = f"Failed to create import spec for generated source at {path}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        raise CompileFailureError(source=source, cause=exc) from exc

    main = getattr(module, "main", None)
    if main is None:
        msg = "Generated Guppy source has no `main` function"
        raise CompileFailureError(
            source=source,
            cause=AttributeError(msg),
        )

    try:
        main.compile_function()
    except BaseException as exc:
        raise CompileFailureError(source=source, cause=exc) from exc
