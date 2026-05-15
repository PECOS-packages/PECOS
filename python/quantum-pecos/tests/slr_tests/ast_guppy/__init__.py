"""AST -> Guppy v1 acceptance tests.

These tests exercise the SLR -> AST -> Guppy lowering path
(`SlrConverter.guppy()` and downstream codegens at
`pecos/slr/ast/codegen/guppy.py`). They are the v1 acceptance
contract: each test is the spec for one feature in the v1 supported
set documented at:

    ~/Repos/pecos-docs/design/slr/v1-feature-matrix.md
    ~/Repos/pecos-docs/design/slr/stage3-synthesis.md
    ~/Repos/pecos-docs/design/slr/stage5-integrity-review.md

Tests start as xfail because the AST Guppy emitter is being rewritten
on this branch (`feat/ast-guppy-v1`). As features land, the xfail
mark comes off the corresponding test.

Post-cutover, `SlrConverter.hugr()` is also AST-routed (wraps `main`
in a no-arg `entry()` and compiles that). Acceptance tests prefer
`SlrConverter.guppy()` / `_harness.assert_ast_guppy_compiles` so
failures point at the parameterized function, not the entry wrapper.
"""
