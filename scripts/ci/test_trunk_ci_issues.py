# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Protect event isolation, missing nightlies, release tags, and issue idempotency.

Run the real Bash tracker against an exact-command gh fake. Unexpected reads
and writes fail loudly, and every issue mutation must match the fixture in full.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-10-09T12:00:00Z"
RECENT = "2026-10-09T10:00:00Z"
OLD = "2026-10-07T23:00:00Z"
SINCE = "2026-09-25T12:00:00Z"
ANCIENT = "2026-09-25T11:59:59Z"
BASE = "repos/{owner}/{repo}"
WORKFLOWS = (
    "cargo-deny.yml",
    "codeql.yml",
    "cuda-build-check.yml",
    "dependency-integrity-check.yml",
    "dependency-review.yml",
    "github-actions-security.yml",
    "julia-release.yml",
    "julia-test.yml",
    "julia-version-consistency.yml",
    "nightly.yml",
    "osv-scanner.yml",
    "pre-commit.yml",
    "python-release.yml",
    "python-test.yml",
    "python-version-consistency.yml",
    "rust-test.yml",
    "rust-version-consistency.yml",
    "selene-general-noise-semantics.yml",
    "selene-plugins.yml",
    "test-docs-examples.yml",
)
DAILY = ("dependency-integrity-check.yml", "nightly.yml")
ALERT_FILTER = (
    '.[] | select(.rule.security_severity_level == "high" or .rule.security_severity_level == "critical")\n'
    '        | "\\(.number)\\t- \\(.html_url) `\\(.rule.id)` (\\(.rule.security_severity_level))'
    ' in `\\(.most_recent_instance.location.path)`"'
)
FAKE_GH = Path(__file__).with_name("fixtures") / "trunk-ci-gh.sh"


def run_object(
    conclusion: str | None = "success",
    *,
    event: str = "push",
    branch: str = "dev",
    status: str = "completed",
    created: str = RECENT,
    number: int = 1,
    attempt: int = 1,
) -> dict:
    return {
        "run_attempt": attempt,
        "conclusion": conclusion,
        "status": status,
        "event": event,
        "head_branch": branch,
        "head_sha": f"sha{number}",
        "html_url": f"https://github.com/test/repo/actions/runs/{number}",
        "created_at": created,
    }


def branch_api(workflow: str, branch: str = "dev", page: int = 1, *, event: str = "push") -> list[str]:
    return [
        "api",
        f"{BASE}/actions/workflows/{workflow}/runs?branch={branch}&event={event}&per_page=100&page={page}",
    ]


def nightly_api(workflow: str) -> list[str]:
    return ["api", f"{BASE}/actions/workflows/{workflow}/runs?branch=dev&event=schedule&per_page=1&page=1"]


def tags_api(workflow: str, tag: str = "py-1.2.3", page: int = 1) -> list[str]:
    return branch_api(workflow, quote(tag, safe=""), page)


def refs_api(prefix: str) -> list[str]:
    return ["api", "--paginate", f"{BASE}/git/matching-refs/tags/{prefix}", "--jq", '.[].ref | ltrimstr("refs/tags/")']


def recent_tag_api(tag: str) -> list[str]:
    return ["api", f"{BASE}/actions/runs?branch={quote(tag, safe='')}&event=push&created=%3E%3D{SINCE}&per_page=1"]


def title_for(workflow: str, branch: str = "dev", *, kind: str = "push") -> str:
    if kind == "missing":
        return f"Nightly missing: {workflow} on dev"
    if kind == "release":
        return f"Release tag CI red: {workflow} on {branch}"
    return f"Trunk CI red: {workflow} on {branch}" + (" (nightly)" if kind == "schedule" else "")


def summary(workflow: str, run: dict) -> str:
    return (
        f"`{workflow}` concluded **{run['conclusion']}** on `{run['head_branch']}` "
        f"at {run['head_sha']}: {run['html_url']}"
    )


def create(title: str, body: str) -> list[str]:
    return [
        "issue",
        "create",
        "--title",
        title,
        "--label",
        "bug",
        "--label",
        "github_actions",
        "--label",
        "severity:high",
        "--body",
        body,
    ]


def red_create(workflow: str, run: dict, *, kind: str = "push") -> list[str]:
    branch = run["head_branch"]
    if kind == "release":
        body = "A release-tag run is red. "
    else:
        body = (
            "A main-branch run is red. Fix or revert within 24 hours, and do not merge unrelated "
            "pull requests onto a red branch. "
        )
    body += f"This issue closes itself once `{workflow}` succeeds on `{branch}` again."
    return create(title_for(workflow, branch, kind=kind), f"{summary(workflow, run)}\n\n{body}")


def green_close(workflow: str, run: dict, number: int = 42) -> list[str]:
    return ["issue", "close", str(number), "--comment", f"Green again. {summary(workflow, run)}"]


def nightly_summary(workflow: str, run: dict | None) -> str:
    if run is None:
        return f"No scheduled run was found for `{workflow}` on `dev`."
    return f"The last scheduled run of `{workflow}` on `dev` was created at {run['created_at']}: {run['html_url']}"


def missing_create(workflow: str, run: dict | None) -> list[str]:
    return create(
        title_for(workflow, kind="missing"),
        f"No scheduled run was created in the last 36 hours. {nightly_summary(workflow, run)}\n\n"
        "The 36-hour backstop detects dropped daily schedules. "
        "This issue closes itself when a recent scheduled run exists.",
    )


class Tracker:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.responses: dict[str, dict] = {}
        self.tag_run_dates: dict[str, str] = {}
        self.answer(
            ["issue", "list", "--state", "open", "--label", "bug", "--limit", "1000", "--json", "number,title"],
            "[]",
        )
        self.answer(
            [
                "api",
                "--paginate",
                f"{BASE}/code-scanning/alerts?state=open&ref=refs/heads/dev&per_page=100",
                "--jq",
                ALERT_FILTER,
            ],
            "",
        )
        for workflow in WORKFLOWS:
            self.answer(["api", f"{BASE}/actions/workflows/{workflow}", "--jq", ".name"], workflow)
            for event in ("push", "schedule"):
                self.runs(branch_api(workflow, event=event), [])
        self.tags({})
        for workflow in DAILY:
            self.runs(nightly_api(workflow), [run_object(event="schedule")])

    def answer(self, args: list[str], stdout: str = "", *, code: int = 0) -> None:
        self.responses[json.dumps(args)] = {"stdout": stdout, "code": code}

    def runs(self, args: list[str], runs: list[dict]) -> None:
        self.answer(args, json.dumps({"workflow_runs": runs}))

    def branch_runs(self, workflow: str, runs: list[dict], *, branch: str = "dev", page: int = 1) -> None:
        for event in ("push", "schedule"):
            self.runs(branch_api(workflow, branch, page, event=event), [run for run in runs if run["event"] == event])

    def tags(self, dates: dict[str, str]) -> None:
        self.tag_run_dates.update(dates)
        for prefix in ("py-", "jl-", "rs-"):
            self.answer(refs_api(prefix), "\n".join(tag for tag in self.tag_run_dates if tag.startswith(prefix)))
        for tag, date in self.tag_run_dates.items():
            self.runs(recent_tag_api(tag), [run_object(branch=tag, created=date)] if date >= SINCE else [])
            for workflow in WORKFLOWS:
                self.responses.setdefault(
                    json.dumps(tags_api(workflow, tag)),
                    {"stdout": '{"workflow_runs": []}', "code": 0},
                )

    def tag_runs(self, workflow: str, runs: list[dict]) -> None:
        tag = runs[0]["head_branch"]
        self.tags({tag: RECENT})
        self.runs(tags_api(workflow, tag), runs)

    def calls(self, filename: str = "calls.bin") -> list[list[str]]:
        path = self.folder / filename
        if not path.exists():
            return []
        fields = path.read_bytes().decode().split("\0")
        assert fields.pop() == ""
        calls = []
        cursor = 0
        while cursor < len(fields):
            end = cursor + 1 + int(fields[cursor])
            assert end <= len(fields)
            calls.append(fields[cursor + 1 : end])
            cursor = end
        return calls

    def api_calls(self) -> list[list[str]]:
        return [args for args in self.calls() if args[:1] == ["api"]]

    def prepare(self) -> Path:
        responses = self.folder / "responses"
        responses.mkdir()
        patterns: dict[int, list[str]] = {}
        for index, (encoded_args, response) in enumerate(self.responses.items()):
            args = json.loads(encoded_args)
            # Bash uses LC_ALL=C to count bytes too; length prefixes make this
            # injective even for empty arguments, newlines, and punctuation.
            key = "".join(f"{len(arg.encode())}:{arg}" for arg in args)
            key_length = len(key.encode())
            # Escape the four characters special inside Bash double quotes.
            # The entire case pattern stays literal, including glob syntax.
            for special in ("\\", '"', "$", "`"):
                key = key.replace(special, "\\" + special)
            patterns.setdefault(key_length, []).append(f'  "{key}") index={index} ;;')
            (responses / str(index)).write_bytes(f"{response['code']}\n{response['stdout']}\0".encode())
        requests = self.folder / "requests"
        requests.mkdir()
        for length, cases in patterns.items():
            (requests / f"{length}.bash").write_text(
                '# shellcheck shell=bash\ncase "${key?}" in\n' + "\n".join(cases) + '\nesac\n: "${index?}"\n',
            )
        fake = self.folder / "gh"
        fake.write_bytes(FAKE_GH.read_bytes())
        fake.chmod(0o755)
        return fake

    def issues(self, *titles: str, body: str = "", comments: str = "") -> None:
        self.answer(
            ["issue", "list", "--state", "open", "--label", "bug", "--limit", "1000", "--json", "number,title"],
            json.dumps([{"number": number, "title": title} for number, title in enumerate(titles, 42)]),
        )
        for number in range(42, 42 + len(titles)):
            self.answer(["api", f"{BASE}/issues/{number}", "--jq", ".body"], body)
            self.answer(
                ["api", "--paginate", f"{BASE}/issues/{number}/comments?per_page=100", "--jq", ".[].body"],
                comments,
            )

    def check(self, expected: list[list[str]], *, code: int = 0) -> subprocess.CompletedProcess:
        for args in expected:
            # A test can inject a failure for an expected mutation too.
            self.responses.setdefault(json.dumps(args), {"stdout": "", "code": 0})
        self.prepare()
        done = subprocess.run(
            ["/bin/bash", "scripts/ci/trunk-ci-issues.sh"],
            cwd=ROOT,
            env={
                **os.environ,
                "PATH": f"{self.folder}{os.pathsep}{os.environ['PATH']}",
                "FAKE_GH_DIR": str(self.folder),
                "GH_REPO": "test/repo",
                "GH_TOKEN": "fake-token",
                "TRUNK_CI_NOW": NOW,
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        unexpected = self.folder / "unexpected.log"
        assert not unexpected.exists(), unexpected.read_text() if unexpected.exists() else ""
        assert done.returncode == code, done.stdout + done.stderr
        actual = self.calls("mutations.bin")
        assert actual == expected, done.stdout + done.stderr
        return done


@pytest.fixture
def tracker(tmp_path: Path) -> Tracker:
    return Tracker(tmp_path)


def test_push_green_schedule_red(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    red = run_object("failure", event="schedule")
    tracker.branch_runs(workflow, [run_object(number=2), red])
    tracker.check([red_create(workflow, red, kind="schedule")])


def test_push_red_schedule_green(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    red = run_object("failure")
    green = run_object(event="schedule", number=2)
    tracker.branch_runs(workflow, [red, green])
    tracker.issues(title_for(workflow, kind="schedule"))
    tracker.check([red_create(workflow, red), green_close(workflow, green)])


@pytest.mark.parametrize("push_conclusion", ["success", "failure"])
def test_dispatch_cannot_open_or_close_push_issue(tracker: Tracker, push_conclusion: str) -> None:
    workflow = "python-release.yml"
    push = run_object(push_conclusion)
    dispatch = run_object("failure" if push_conclusion == "success" else "success", event="workflow_dispatch", number=2)
    tracker.runs(branch_api(workflow), [dispatch, push])
    tracker.runs(branch_api(workflow, page=2), [])
    tracker.issues(title_for(workflow), body=push["html_url"])
    tracker.check([green_close(workflow, push)] if push_conclusion == "success" else [])


@pytest.mark.parametrize("workflow", ["python-release.yml", "julia-release.yml"])
def test_superseded_push_ignored_but_cancelled_tag_red(tracker: Tracker, workflow: str) -> None:
    cancelled = run_object("cancelled")
    tag = run_object("cancelled", branch="py-1.2.3", number=2)
    tracker.runs(branch_api(workflow), [cancelled])
    tracker.runs(branch_api(workflow, page=2), [])
    tracker.tag_runs(workflow, [tag])
    tracker.check([red_create(workflow, tag, kind="release")])


@pytest.mark.parametrize("workflow", DAILY)
@pytest.mark.parametrize("created", [OLD, None])
def test_stale_or_absent_nightly_opens(tracker: Tracker, workflow: str, created: str | None) -> None:
    run = run_object(event="schedule", created=created) if created else None
    tracker.runs(nightly_api(workflow), [run] if run else [])
    tracker.check([missing_create(workflow, run)])


@pytest.mark.parametrize("status", ["queued", "in_progress", "completed"])
@pytest.mark.parametrize("created", [RECENT, "2026-10-08T00:00:00Z", "2026-10-08T02:06:00Z"])
def test_recent_nightly_closes_missing(tracker: Tracker, status: str, created: str) -> None:
    workflow = "nightly.yml"
    run = run_object("failure" if status == "completed" else None, event="schedule", status=status, created=created)
    tracker.runs(nightly_api(workflow), [run])
    tracker.issues(title_for(workflow, kind="missing"))
    tracker.check(
        [["issue", "close", "42", "--comment", f"Nightly schedule resumed. {nightly_summary(workflow, run)}"]],
    )


def test_still_missing_does_not_repeat_comment(tracker: Tracker) -> None:
    workflow = "nightly.yml"
    tracker.runs(nightly_api(workflow), [run_object(event="schedule", created=OLD)])
    tracker.issues(title_for(workflow, kind="missing"))
    tracker.check([])


@pytest.mark.parametrize("tag", ["py-1.2.3", "jl-1.2.3", "rs-1.2.3"])
@pytest.mark.parametrize("conclusion", ["failure", "success"])
def test_release_tag_red_opens_green_closes(tracker: Tracker, tag: str, conclusion: str) -> None:
    workflow = "python-release.yml"
    run = run_object(conclusion, branch=tag)
    tracker.tag_runs(workflow, [run])
    if conclusion == "success":
        tracker.issues(title_for(workflow, tag, kind="release"))
    tracker.check(
        [red_create(workflow, run, kind="release")] if conclusion == "failure" else [green_close(workflow, run)],
    )


@pytest.mark.parametrize("kind", ["push", "schedule", "release"])
@pytest.mark.parametrize("location", ["body", "comments", "new"])
def test_failing_url_commented_only_once(tracker: Tracker, kind: str, location: str) -> None:
    workflow = "rust-test.yml"
    run = run_object(
        "failure",
        event="schedule" if kind == "schedule" else "push",
        branch="rs-1.2.3" if kind == "release" else "dev",
    )
    if kind == "release":
        tracker.tag_runs(workflow, [run])
    else:
        tracker.runs(branch_api(workflow, event=kind), [run])
        tracker.runs(branch_api(workflow, page=2), [])
    tracker.issues(
        title_for(workflow, run["head_branch"], kind=kind),
        body=run["html_url"] if location == "body" else "old failure",
        comments=run["html_url"] if location == "comments" else "old failure",
    )
    tracker.check(
        [["issue", "comment", "42", "--body", f"Still red. {summary(workflow, run)}"]] if location == "new" else [],
    )


def test_pages_past_ignored_events_and_conclusions(tracker: Tracker) -> None:
    workflow = "julia-release.yml"
    ignored = [
        run_object("skipped"),
        run_object("neutral"),
        run_object("cancelled"),
        run_object("failure", event="pull_request"),
        run_object("failure", event="pull_request_target"),
    ]
    tracker.runs(branch_api(workflow), ignored * 20)
    tracker.runs(branch_api(workflow, event="schedule"), [run_object("neutral", event="schedule")] * 100)
    red = run_object("timed_out")
    nightly = run_object("action_required", event="schedule", number=2)
    tracker.branch_runs(workflow, [red, nightly], page=2)
    tracker.check([red_create(workflow, red), red_create(workflow, nightly, kind="schedule")])


def test_tag_discovery_latest_completed_and_independent_tags(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    green = run_object(branch="rs-1.2.3")
    red = run_object("startup_failure", branch="jl-1.2.3", number=2)
    tracker.tag_runs(workflow, [green, run_object("failure", branch="rs-1.2.3")])
    tracker.tag_runs(workflow, [red, run_object(branch="jl-1.2.3")])
    # gh --paginate applies --jq to every page. A duplicate ref across pages
    # must still probe recent runs once and reconcile each workflow/tag once.
    tracker.answer(refs_api("rs-"), "rs-1.2.3\nrs-1.2.3\n")
    tracker.issues(title_for(workflow, "rs-1.2.3", kind="release"))
    tracker.check([red_create(workflow, red, kind="release"), green_close(workflow, green)])
    assert tracker.api_calls().count(recent_tag_api("rs-1.2.3")) == 1


@pytest.mark.parametrize(
    "failure",
    ["metadata", "runs", "nightly", "tags", "body", "comments", "create", "close", "comment"],
)
def test_api_failure_isolated(tracker: Tracker, failure: str) -> None:
    workflow = "dependency-integrity-check.yml"
    red = run_object("failure")
    nightly = run_object("failure", event="schedule", number=2)
    tracker.branch_runs(workflow, [red, nightly])
    expected = [red_create(workflow, red), red_create(workflow, nightly, kind="schedule")]
    if failure == "metadata":
        tracker.answer(["api", f"{BASE}/actions/workflows/{workflow}", "--jq", ".name"], code=1)
        expected = []
    elif failure == "runs":
        tracker.answer(branch_api(workflow), code=1)
        expected = expected[1:]
    elif failure == "nightly":
        tracker.answer(nightly_api(workflow), code=1)
    elif failure == "tags":
        tracker.tags({"py-1.2.3": RECENT})
        tracker.answer(tags_api(workflow), code=1)
    elif failure == "create":
        tracker.answer(expected[0], code=1)
    elif failure == "close":
        green = run_object()
        tracker.branch_runs(workflow, [green, nightly])
        tracker.issues(title_for(workflow))
        expected[0] = green_close(workflow, green)
        tracker.answer(expected[0], code=1)
    else:
        tracker.issues(title_for(workflow))
        expected = expected[1:]
        if failure == "body":
            tracker.answer(["api", f"{BASE}/issues/42", "--jq", ".body"], code=1)
        elif failure == "comments":
            tracker.answer(["api", "--paginate", f"{BASE}/issues/42/comments?per_page=100", "--jq", ".[].body"], code=1)
        else:
            comment = ["issue", "comment", "42", "--body", f"Still red. {summary(workflow, red)}"]
            expected.insert(0, comment)
            tracker.answer(comment, code=1)
    # Later tags, workflows, and code scanning still reconcile.
    tag = run_object("failure", branch="py-1.2.3", number=4)
    if failure not in ("metadata", "tags"):
        tracker.tag_runs(workflow, [tag])
        expected.append(red_create(workflow, tag, kind="release"))
    tracker.runs(branch_api("pre-commit.yml"), [red])
    tracker.runs(branch_api("pre-commit.yml", page=2), [])
    expected.append(red_create("pre-commit.yml", red))
    result = tracker.check(expected, code=1)
    assert "::error::" in result.stdout


def test_no_qualifying_runs_leave_issues_open(tracker: Tracker) -> None:
    tracker.issues(title_for("rust-test.yml"), title_for("rust-test.yml", kind="schedule"))
    tracker.check([])


def test_push_page_failure_still_reconciles_same_branch_nightly(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    nightly = run_object("failure", event="schedule")
    tracker.branch_runs(workflow, [run_object("skipped"), nightly])
    tracker.answer(branch_api(workflow, page=2), code=1)
    tracker.check([red_create(workflow, nightly, kind="schedule")], code=1)


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_latest_completed_tag_without_verdict_leaves_issue_untouched(tracker: Tracker, conclusion: str) -> None:
    workflow = "rust-test.yml"
    tag = "rs-1.2.3"
    tracker.tag_runs(workflow, [run_object(conclusion, branch=tag), run_object("failure", branch=tag)])
    tracker.issues(title_for(workflow, tag, kind="release"))
    tracker.check([])


@pytest.mark.parametrize("workflow", ["julia-release.yml", "python-release.yml"])
def test_cancelled_nightly_is_red_for_superseded_workflow(tracker: Tracker, workflow: str) -> None:
    cancelled = run_object("cancelled", event="schedule")
    tracker.runs(branch_api(workflow, event="schedule"), [cancelled, run_object(event="schedule", number=2)])
    tracker.check([red_create(workflow, cancelled, kind="schedule")])


@pytest.mark.parametrize("recent", [False, True])
def test_api_budget(tracker: Tracker, recent: bool) -> None:
    # Reserve capacity for 21 workflows (currently 20), with 25 old refs:
    # 21 * (1 metadata + 1 branch * 2 classes) + 3 refs + 25 recency probes
    # + 2 freshness reads + 1 code-scanning read = 94 REST API calls.
    # One additional recent tag costs 1 probe + 21 workflow reads: 116.
    # At the observed peak of 7 runs/hour: 7 * 94 = 658, or 7 * 116 = 812,
    # versus 7 * 136 = 952 before. GITHUB_TOKEN's 1,000/hour is shared:
    # allow room for issue inventory/mutations and other workflows too.
    # Exact unfiltered queries let the event= mutant traverse history, so
    # the numeric budget assertion detects the extra calls.
    capacity = 21
    assert len(WORKFLOWS) <= capacity
    tracker.tags({f"{('py-', 'jl-', 'rs-')[i % 3]}old-{i}": ANCIENT for i in range(25)})
    if recent:
        tracker.tags({"py-1.2.3": RECENT})
    for workflow in WORKFLOWS:
        tracker.runs(branch_api(workflow), [] if workflow == "nightly.yml" else [run_object()])
        if workflow in DAILY:
            tracker.runs(branch_api(workflow, event="schedule"), [run_object(event="schedule")])
        for page in range(1, 5):
            legacy = ["api", f"{BASE}/actions/workflows/{workflow}/runs?branch=dev&per_page=100&page={page}"]
            history_event = "schedule" if workflow == "nightly.yml" else "push"
            history = [run_object(event=history_event, number=page * 100 + i) for i in range(100)]
            if workflow == "dependency-integrity-check.yml" and page == 1:
                history[0] = run_object(event="schedule")
            tracker.runs(legacy, history if page <= 3 else [])
        if recent:
            tracker.runs(tags_api(workflow), [run_object(branch="py-1.2.3")])
    tracker.check([])
    count = len(tracker.api_calls())
    bound = capacity * 3 + 3 + 25 + 2 + 1 + (1 + capacity if recent else 0)
    assert count <= bound, f"API budget exceeded: {count} > {bound}"
    assert count == len(WORKFLOWS) * 3 + 31 + (1 + len(WORKFLOWS) if recent else 0)


@pytest.mark.parametrize("date", [ANCIENT, SINCE, RECENT, "2026-10-09T13:00:00Z"])
def test_tag_run_time_window(tracker: Tracker, date: str) -> None:
    workflow = "rust-test.yml"
    tag = "rs-1.2.3"
    tracker.tags({tag: date})
    red = run_object("failure", branch=tag)
    tracker.runs(tags_api(workflow, tag), [red])
    tracker.check([red_create(workflow, red, kind="release")] if date >= SINCE else [])
    assert tracker.api_calls().count(recent_tag_api(tag)) == 1
    assert (tags_api(workflow, tag) in tracker.api_calls()) == (date >= SINCE)


@pytest.mark.parametrize("failure", ["refs", "recency", "json", "runs"])
def test_tag_discovery_and_read_failure_isolated(tracker: Tracker, failure: str) -> None:
    workflow = "pre-commit.yml"
    tracker.tags({"py-1.2.3": RECENT, "jl-1.2.3": RECENT})
    if failure == "refs":
        tracker.answer(refs_api("py-"), code=1)
    elif failure == "recency":
        tracker.answer(recent_tag_api("py-1.2.3"), code=1)
    elif failure == "json":
        tracker.answer(recent_tag_api("py-1.2.3"), "invalid JSON")
    else:
        tracker.answer(tags_api(workflow), code=1)
    red = run_object("failure")
    tag = run_object("failure", branch="jl-1.2.3", number=2)
    tracker.runs(branch_api(workflow), [red])
    tracker.runs(tags_api(workflow, "jl-1.2.3"), [tag])
    tracker.check([red_create(workflow, red), red_create(workflow, tag, kind="release")], code=1)


def test_tag_name_encoded_for_recency_and_workflow_queries(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    tag = "py-release/1.2.3+build#1&x"
    red = run_object("failure", branch=tag)
    tracker.tag_runs(workflow, [red])
    tracker.check([red_create(workflow, red, kind="release")])


@pytest.mark.parametrize(
    ("args", "unknown"),
    [
        (["api", "ab", "c"], ["api", "a", "bc"]),
        (["api", "******"], ["api", "abcdef"]),
        (["api", "", "two\nlines"], ["api", "two\nlines"]),
        (["api", "π", "🦀"], ["api", "π🦀"]),
        (
            ["issue", "comment", "42", "--body", "'\"\\\r\n$(exit 99)`exit 99`"],
            ["issue", "comment", "42", "--body", "'\"\\\n$(exit 99)`exit 99`"],
        ),
        ([], [""]),
        (["api", "x" * 4096], ["api", "x" * 4095]),
    ],
)
def test_bash_fake_preserves_exact_argv_output_and_failure(
    tracker: Tracker,
    args: list[str],
    unknown: list[str],
) -> None:
    output = "\nline\r\nπ\n\n"
    tracker.answer(args, output, code=23)
    fake = tracker.prepare()
    for argv, code, stdout in ((args, 23, output.encode()), (unknown, 97, b"")):
        done = subprocess.run(
            [str(fake), *argv],
            env={**os.environ, "FAKE_GH_DIR": str(tracker.folder)},
            capture_output=True,
            check=False,
        )
        assert done.returncode == code
        assert done.stdout == stdout
        assert (b"Unexpected gh argv:" in done.stderr) == (code == 97)
    assert tracker.calls() == [args, unknown]
    assert tracker.api_calls() == [argv for argv in (args, unknown) if argv[:1] == ["api"]]
    assert tracker.calls("mutations.bin") == [
        argv for argv in (args, unknown) if argv[:2] in (["issue", "create"], ["issue", "close"], ["issue", "comment"])
    ]
    assert "Unexpected gh argv:" in (tracker.folder / "unexpected.log").read_text()


def script_array(name: str) -> list[str]:
    script = (ROOT / "scripts/ci/trunk-ci-issues.sh").read_text()
    match = re.search(rf"^{name}=\((.*?)\)", script, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"Missing {name} array"
    return shlex.split(match[1], comments=True)


def test_daily_workflows_match_watched_daily_crons() -> None:
    watched = script_array("workflows")
    assert tuple(watched) == WORKFLOWS
    expected = set()
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        triggers = workflow.get("on", workflow.get(True, {}))
        if path.name in watched and isinstance(triggers, dict):
            for schedule in triggers.get("schedule", []):
                fields = schedule["cron"].split()
                if len(fields) == 5 and fields[-3:] == ["*", "*", "*"]:
                    expected.add(path.name)
    daily = script_array("daily_workflows")
    assert len(daily) == len(set(daily))
    assert set(daily) == expected
    assert set(DAILY) == expected


@pytest.mark.parametrize("kind", ["push", "schedule", "release"])
@pytest.mark.parametrize("attempt", [1, 2])
@pytest.mark.parametrize("status", ["queued", "in_progress"])
@pytest.mark.parametrize("paged", [False, True])
def test_active_rerun_blocks_older_result_but_first_attempt_does_not(
    tracker: Tracker,
    kind: str,
    attempt: int,
    status: str,
    paged: bool,
) -> None:
    workflow = "python-release.yml"
    branch = "py-1.2.3" if kind == "release" else "dev"
    event = "schedule" if kind == "schedule" else "push"
    active = run_object(None, branch=branch, event=event, status=status, attempt=attempt, number=2)
    older = run_object(branch=branch, event=event)
    if kind == "release":
        tracker.tags({branch: RECENT})
    # An ignored first attempt on a previous page must not hide the rerun barrier.
    if paged:
        tracker.runs(
            branch_api(workflow, branch, event=event),
            [run_object(None, branch=branch, event=event, status="in_progress", number=3)] * 100,
        )
    tracker.runs(branch_api(workflow, branch, page=2 if paged else 1, event=event), [active, older])
    tracker.issues(title_for(workflow, branch, kind=kind), body=f"Previously red: {active['html_url']}")
    tracker.check([green_close(workflow, older)] if attempt == 1 else [])


def test_old_commit_with_recent_tag_run_is_checked(tracker: Tracker) -> None:
    workflow = "rust-test.yml"
    red = run_object("failure", branch="rs-old-commit")
    red["head_commit"] = {"id": red["head_sha"], "timestamp": ANCIENT}
    tracker.tag_runs(workflow, [red])
    tracker.runs(recent_tag_api(red["head_branch"]), [red])
    tracker.check([red_create(workflow, red, kind="release")])
    assert tracker.api_calls().count(recent_tag_api(red["head_branch"])) == 1


@pytest.mark.parametrize("ref_exists", [False, True])
def test_aged_out_tag_with_open_issue_still_closes(tracker: Tracker, ref_exists: bool) -> None:
    workflow = "rust-test.yml"
    tag = "rs-aged-out"
    green = run_object(branch=tag, created=ANCIENT)
    tracker.tags({tag: ANCIENT})
    if not ref_exists:
        tracker.answer(refs_api("rs-"), "")
    tracker.runs(tags_api(workflow, tag), [green])
    tracker.issues(title_for(workflow, tag, kind="release"))
    tracker.check([green_close(workflow, green)])


def test_release_issue_creation_failure_still_reconciles_other_workflows(tracker: Tracker) -> None:
    workflow = "julia-release.yml"
    tag = run_object("failure", branch="jl-1.2.3")
    tracker.tag_runs(workflow, [tag])
    failed = red_create(workflow, tag, kind="release")
    tracker.answer(failed, code=1)
    later = run_object("failure", number=2)
    tracker.runs(branch_api("pre-commit.yml"), [later])
    tracker.check([failed, red_create("pre-commit.yml", later)], code=1)
