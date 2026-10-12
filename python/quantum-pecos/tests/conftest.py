"""Shared pytest setup for the Python test suite."""

# The test tree contains ``tests/pecos``. Pytest can import tests from that
# directory as a namespace package named ``pecos`` when an individual file is run
# in isolation. Import the installed/source package first so later
# ``import pecos`` statements resolve to the public PECOS package.
import pecos
import pytest


@pytest.fixture
def sz_program():
    """Compose rendered SZ helpers for signed readout tests."""
    from pecos.guppy_gen._module_loader import load_cached_guppy_source
    from pecos.guppy_gen.gadget_render import render_gadget_function
    from pecos.guppy_gen.protocol_render import render_surface_protocol_module
    from pecos.qec.surface import SurfacePatch, gadgets

    def build(distance, count, num_rounds=2):
        patch = SurfacePatch.create(distance)
        source = render_surface_protocol_module(patch)
        round_gadget = gadgets.syndrome_round_gadget(patch, gadgets.default_allocation(patch), round_index=0)
        source += "\n\n" + "\n".join(render_gadget_function(round_gadget, tag_scope="second"))
        body = ["@guppy", "def experiment() -> None:", "    data = prep_x_basis()"]

        def rounds(scopes):
            for scope in scopes:
                register = "data" if scope == "data" else "anc"
                body.extend(
                    [
                        f"    syn = syndrome_extraction_{scope}({register})",
                        f'    output("synx_{scope}", syn.synx)',
                        f'    output("synz_{scope}", syn.synz)',
                    ],
                )

        for _ in range(num_rounds):
            rounds(("data",))
        for scope in ("anc", "second")[:count]:
            body.append("    anc = prep_y_basis()")
            rounds(("data",))
            body.extend(
                [
                    f"    first = syndrome_extraction_{scope}(anc)",
                    f'    output("synx_{scope}", first.synx)',
                    f'    output("synz_{scope}", first.synz)',
                ],
            )
            for _ in range(num_rounds - 1):
                rounds(("data", scope))
            body.append("    transversal_cx(data, anc)")
            for _ in range(num_rounds):
                rounds(("data", scope))
            body.extend(
                [
                    "    final = measure_z_basis(anc)",
                    f'    output("final_{scope}", final)',
                    "    correct_sz_teleportation(data, sz_teleportation_decision(first, final))",
                ],
            )
            if count == 1:
                body.extend(
                    [
                        "    syn = syndrome_extraction_fold_szdg_data(data)",
                        '    output("synx_data", syn.synx)',
                        '    output("synz_data", syn.synz)',
                        '    output("fold_z", syn.synz)',
                    ],
                )
            for _ in range(num_rounds):
                rounds(("data",))
        body.extend(["    final = measure_x_basis(data)", '    output("final_data", final)'])
        module = load_cached_guppy_source("sz_executed_test", source + "\n\n" + "\n".join(body) + "\n")
        return module["experiment"], patch

    return build


def pytest_configure(config):
    """Register markers at the test-tree root so they are known for ANY
    invocation (e.g. running a single file directly), not only when
    pytest happens to pick a ``pyproject.toml`` whose
    ``[tool.pytest.ini_options].markers`` lists them."""
    config.addinivalue_line(
        "markers",
        "slow: mark tests that provide extra integration coverage but are "
        "excluded from the default fast Python test lane",
    )
