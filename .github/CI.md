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

## Validation tiers

Pull requests run fast checks, with labels such as `ci:full-rust`, `ci:julia`,
and `ci:selene-plugins` enabling more validation. Lane-specific path filters
and checks still apply.

Pushes to `dev` run per-commit validation under each workflow's existing path
filters. Release builds finish once started; newer pushes can replace pending
release builds.

Nightly validation runs every validation workflow with its full OS matrix.
Daily macOS crons are staggered for five macOS slots. Measured schedule delays
of 5-9 hours shift the 19:07-22:37 UTC stagger outside 13:00-23:00 UTC working
hours. Ubuntu-only crons remain in the early UTC hours. Each workflow tests `dev`'s
head at its own start time, so the nightly results do not represent one SHA.
Diff-based dependency review and the duplicate PR core gate stay in the PR tier.

Release tags (`py-*`, `jl-*`, `rs-*`) run the full validation matrix. Push release
tags one at a time: GitHub creates no tag push events when more than three tags
are pushed at once. See [GitHub's push-event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push).

For `py-*`, PyPI publication is manual. Run
`uv run --frozen python scripts/ci/release_tag_status.py <tag>` and require exit 0
before publishing. PyYAML comes from the dev group; fetch the tag locally before
running the check. Missing, unfinished, skipped, or failed workflows count as red.
The check expects every validation workflow in either the tag or this checkout.

For `jl-*`, GitHub-release publication is automatic on the tag push, as soon as
the Julia workflow's own jobs pass. Validate the commit before tagging: require
green post-merge runs for that SHA and a green nightly that tested it. Run the same status
command afterwards to check the full validation matrix.
