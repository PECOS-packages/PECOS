"""The legacy analysis examples import existing, nondeprecated functions."""

import ast
import importlib
import json
import re
import runpy
import textwrap
from pathlib import Path


def test_monte_carlo_plotting_example(tmp_path, monkeypatch) -> None:
    """Run the displayed sample data and plotting step without opening a window."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    page = Path(__file__).resolve().parents[3] / "docs/examples/monte_carlo_script.rst"
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:    [^\n]*\n|\n)+)", page.read_text())
    source = "\n".join(textwrap.dedent(block) for block in blocks[-2:])
    script = tmp_path / "plot_example.py"
    script.write_text(source)
    try:
        result = runpy.run_path(str(script))
        lines = plt.gca().get_lines()
        assert list(lines[0].get_xdata()) == result["ps"]
        assert list(lines[0].get_ydata()) == result["plog"]
        assert "pecos.analysis" in source
        assert "pecos.tools" not in source
    finally:
        plt.close("all")


def test_color_code_notebook_syndrome_import() -> None:
    """The notebook uses the live syndrome helper without deprecated imports."""
    path = Path(__file__).resolve().parents[5] / "examples/Dusting off color code code.ipynb"
    notebook = json.loads(path.read_text())
    imports = [
        node
        for cell in notebook["cells"]
        for line in cell.get("source", [])
        if "import syn_diff" in line
        for node in ast.walk(ast.parse(line))
        if isinstance(node, ast.ImportFrom)
    ]
    assert len(imports) == 1
    assert imports[0].module == "pecos.analysis.syndromes"
    syn_diff = importlib.import_module(imports[0].module).syn_diff
    assert syn_diff({"a": ["01"], "b": ["11"]}, [("a", "b")]) == {"a_b": ["10"]}
