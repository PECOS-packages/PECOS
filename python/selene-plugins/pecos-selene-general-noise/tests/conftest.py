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
PASSED_KEY = pytest.StashKey[dict[str, set[str]]]()

# Channels that a complete run must exercise. Kept here rather than in a test module
# because the completeness check runs at session end, after every test has reported.
REQUIRED_CHANNELS = frozenset(
    {
        "combined-channels",
        "gate-leakage",
        "idle-coherent",
        "idle-linear",
        "idle-sine-squared",
        "layered-multiqubit",
        "measurement-crosstalk",
        "measurement-crosstalk-multiqubit",
        "measurement-crosstalk-repeated",
        "preparation",
        "preparation-crosstalk",
        "preparation-leakage",
        "readout",
        "single-qubit-emission",
        "single-qubit-pauli",
        "single-qubit-seepage",
        "two-qubit-angle-scaling",
        "two-qubit-emission",
        "two-qubit-pauli",
        "two-qubit-seepage",
    },
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-full-noise-coverage",
        action="store_true",
        help=(
            "Fail the session unless every required noise channel had a passing test. "
            "Only meaningful for a run that selects the whole suite; the fast lane "
            "deliberately does not."
        ),
    )


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item):
    """Record channels whose evidence actually ran and passed.

    The collection matrix above is deliberately pre-deselection, so it says which
    channels are *claimed*, not which were *checked*. A channel is only genuinely
    covered by a run in which one of its marked tests passed.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.passed:
        return
    passed = item.config.stash.setdefault(PASSED_KEY, defaultdict(set))
    for mark in item.iter_markers("noise_channel"):
        if mark.args:
            passed[str(mark.args[0])].add(str(mark.kwargs.get("evidence", item.nodeid)))


@pytest.fixture(scope="session")
def noise_channel_coverage(pytestconfig: pytest.Config) -> dict[str, ChannelCoverage]:
    """Expose the complete pre-deselection coverage matrix to contract tests."""
    return pytestconfig.stash.get(COVERAGE_KEY, {})


@pytest.fixture(scope="session")
def executed_noise_channels(pytestconfig: pytest.Config) -> dict[str, set[str]]:
    """Expose the channels whose marked tests actually passed in this run."""
    return dict(pytestconfig.stash.get(PASSED_KEY, {}))


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:
    """Publish a compact channel summary in local output and CI logs."""
    coverage = config.stash.get(COVERAGE_KEY, {})
    if not coverage:
        return
    terminalreporter.section("general-noise semantic coverage")
    for channel, evidence in sorted(coverage.items()):
        oracle_list = ",".join(sorted(evidence.oracles))
        terminalreporter.write_line(f"{channel}: {len(evidence.cases)} sensitive cases; oracle={oracle_list}")


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Enforce executed -- not merely declared -- channel coverage on a complete run.

    This has to run at session end rather than as a test: pytest executes files in
    collection order, so a test asserting on executed coverage would only ever see the
    channels from files sorted before it.
    """
    # The option is registered by the pytest_addoption above, but pytest only honours
    # pytest_addoption for conftests between the rootdir and the command-line anchor.
    # This plugin's pyproject.toml carries [tool.pytest.ini_options], so pointing pytest
    # at these tests makes this directory the rootdir and the option registers; pointing
    # it at python/selene-plugins/ makes the repo root the rootdir, leaving this conftest
    # below the cutoff -- its hooks still run, but the option was never added. An absent
    # option means the gate was not requested, which is exactly the fast lane's intent.
    # The gated run (selene-general-noise-semantics.yml) is scoped to this directory and
    # passes the flag explicitly, so a misconfiguration there fails loudly as an unknown
    # option rather than silently skipping the check.
    if not session.config.getoption("--require-full-noise-coverage", default=False):
        return
    passed = session.config.stash.get(PASSED_KEY, {})
    unexecuted = sorted(channel for channel in REQUIRED_CHANNELS if not passed.get(channel))
    if unexecuted:
        session.config.stash[PASSED_KEY] = passed
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"required noise channels had no passing evidence: {unexecuted}",
                red=True,
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
