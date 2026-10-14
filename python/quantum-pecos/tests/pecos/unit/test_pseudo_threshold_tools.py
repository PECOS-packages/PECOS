"""Pseudo-threshold plots accept the documented sequences and PECOS arrays."""

import pecos as pc
import pytest
from pecos.analysis.pseudo_threshold_tools import plot


@pytest.mark.parametrize("container", [list, tuple, pc.array])
@pytest.mark.parametrize("explicit_limits", [False, True])
def test_plot_polynomial_and_samples(container, explicit_limits, tmp_path, monkeypatch) -> None:
    """The fitted curve, input samples, and estimated crossing are plotted."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    plog = [p * p + 0.1 for p in ps]
    limits = {"p_start": 0.0, "p_end": 0.6} if explicit_limits else {}
    try:
        plot(container(ps), container(plog), deg=2, **limits)
        axes = plt.gca()
        fitted, samples, diagonal, crossing = axes.get_lines()
        xs = list(fitted.get_xdata())
        assert len(xs) == 1000
        assert list(fitted.get_ydata()) == pytest.approx([x * x + 0.1 for x in xs])
        assert list(samples.get_xdata()) == ps
        assert list(samples.get_ydata()) == plog
        assert list(diagonal.get_xdata()) == list(diagonal.get_ydata())
        assert list(crossing.get_xdata()) == pytest.approx([(1 - 0.6**0.5) / 2] * 2)
        expected_limits = (0.0, 0.6) if explicit_limits else (min(plog) * 0.9, max(plog) * 1.1)
        assert axes.get_xlim() == pytest.approx(expected_limits)
        assert axes.get_ylim() == pytest.approx(expected_limits)
    finally:
        plt.close("all")
