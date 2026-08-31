"""Collection-time semantic coverage accounting for the general-noise harness."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class ChannelCoverage:
    """Collected sensitive cases and independent oracle kinds for one channel."""

    cases: tuple[str, ...]
    oracles: frozenset[str]


COVERAGE_KEY = pytest.StashKey[dict[str, ChannelCoverage]]()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Record channel marks before the slow/fast selection plugin deselects items."""
    cases: dict[str, set[str]] = defaultdict(set)
    oracles: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for mark in item.iter_markers("noise_channel"):
            if not mark.args:
                message = f"{item.nodeid} has a noise_channel mark without a channel name"
                raise pytest.UsageError(message)
            channel = str(mark.args[0])
            oracle = str(mark.kwargs.get("oracle", "unspecified"))
            evidence = str(mark.kwargs.get("evidence", item.nodeid))
            cases[channel].add(evidence)
            oracles[channel].add(oracle)
    config.stash[COVERAGE_KEY] = {
        channel: ChannelCoverage(tuple(sorted(nodeids)), frozenset(oracles[channel]))
        for channel, nodeids in cases.items()
    }


@pytest.fixture(scope="session")
def noise_channel_coverage(pytestconfig: pytest.Config) -> dict[str, ChannelCoverage]:
    """Expose the complete pre-deselection coverage matrix to contract tests."""
    return pytestconfig.stash.get(COVERAGE_KEY, {})


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:
    """Publish a compact channel summary in local output and CI logs."""
    coverage = config.stash.get(COVERAGE_KEY, {})
    if not coverage:
        return
    terminalreporter.section("general-noise semantic coverage")
    for channel, evidence in sorted(coverage.items()):
        oracle_list = ",".join(sorted(evidence.oracles))
        terminalreporter.write_line(f"{channel}: {len(evidence.cases)} sensitive cases; oracle={oracle_list}")
