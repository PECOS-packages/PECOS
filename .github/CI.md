# Full Rust validation before merging

The **Rust test / linting** workflow runs the full Rust suite on Linux, macOS,
and Windows as normal PR checks when either:

- the pull request changes a platform-sensitive path, or
- the pull request carries the `ci:full-rust` label.

Platform-sensitive paths are listed in the `rust-test-matrix` job of
`.github/workflows/rust-test.yml`: crates whose build or tests differ by
operating system (build scripts, FFI and plugin ABIs, OS-specific `cfg` code,
native sources, the Selene runtime and plugin crates), the QIS/QIR fixtures
that Rust tests include, and workspace-wide files such as `Cargo.toml`,
`Cargo.lock`, and this workflow. The job prints which changed files selected
the full suite. Other pull requests run the lightweight Linux path. Extend the
list when a crate gains OS-specific behavior.

Use the label to force full validation for a change outside those paths. The
label persists: subsequent pushes rerun full validation. Removing it cancels the
current workflow and returns the PR to the path-based selection. Unrelated label
changes do not cancel or restart validation. Existing branch and changed-path
filters still apply; documentation-only PRs do not trigger this workflow solely
because they carry the label.

Full validation tests GitHub's PR merge commit, including integration with the
target branch. The separate core gate and targeted platform regressions still
run under their own rules. Pushes with matching paths to `dev`, `development`,
`main`, or `master` run the full three-platform suite automatically. Manual
dispatch also runs the full suite, but use the PR label when the results should
appear in the PR checks list.

Wait for all three `rust-test` jobs on the latest PR revision before merging a
PR that runs full validation. Neither the path rule nor the label changes branch
protection or makes these optional jobs required repository-wide.
