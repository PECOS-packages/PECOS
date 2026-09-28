# Full Rust validation before merging

For platform-sensitive changes, add the `ci:full-rust` label to the pull
request. The **Rust test / linting** workflow then runs the full Rust suite
on Linux, macOS, and Windows as normal PR checks. It tests GitHub's PR merge
commit, including integration with the target branch.

The label persists: subsequent pushes rerun full validation. Removing it
cancels the current workflow and restores the default Linux smoke path.
Unrelated label changes do not cancel or restart validation. Existing branch
and changed-path filters still apply; documentation-only PRs do not trigger
this workflow solely because they carry the label.

Without the label, the default PR checks remain lightweight; the separate
core gate and targeted platform regressions still run under their own rules.
Pushes with matching paths to `dev`, `development`, `main`, or `master`
continue to run the full three-platform suite automatically. Manual dispatch
also runs the full suite, but use the PR label when the results should appear
in the PR checks list.

Wait for all three `rust-test` jobs on the latest PR revision before merging
a PR that needs full validation. The label does not change branch protection
or make these optional jobs required repository-wide.
