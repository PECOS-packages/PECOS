#!/usr/bin/env bash
# Keep one GitHub issue open per workflow that is red on a main branch.
#
# Post-merge and scheduled runs catch what pull-request CI does not, so a red
# run there must not go unnoticed. For each watched workflow and branch this
# reads the latest completed non-PR run: if it failed, open an issue (or
# comment on the open one when the failing run is new); if it succeeded, close
# the open issue.
#
# Usage: scripts/ci/trunk-ci-issues.sh
# Environment: GH_TOKEN and GH_REPO for the gh CLI.
set -euo pipefail

branches=(dev master)
workflows=(
  cargo-deny.yml
  codeql.yml
  cuda-build-check.yml
  dependency-integrity-check.yml
  dependency-review.yml
  github-actions-security.yml
  julia-release.yml
  julia-test.yml
  julia-version-consistency.yml
  osv-scanner.yml
  python-release.yml
  python-test.yml
  python-version-consistency.yml
  rust-test.yml
  rust-version-consistency.yml
  selene-general-noise-semantics.yml
  selene-plugins.yml
  test-docs-examples.yml
)
# These artifact workflows still cancel a superseded trunk run by design, so a
# cancelled run there says nothing about the branch.
superseded_ok=(julia-release.yml python-release.yml)

open_issues="$(gh issue list --state open --label github_actions --limit 500 --json number,title)"

for workflow in "${workflows[@]}"; do
  skip_cancelled=false
  [[ " ${superseded_ok[*]} " == *" $workflow "* ]] && skip_cancelled=true

  name="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}" --jq .name)"
  for branch in "${branches[@]}"; do
    # Newest first; page until a run that says something about the branch.
    run=""
    for ((page = 1; ; page++)); do
      runs="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=completed&per_page=100&page=${page}" \
        --jq "[.workflow_runs[] | {conclusion, event, head_sha, html_url}]")"
      [ "$(jq length <<<"$runs")" -gt 0 ] || break
      run="$(jq -c --argjson skip_cancelled "$skip_cancelled" '
          map(select(.event != "pull_request" and .event != "pull_request_target"
                     and .conclusion != "skipped" and .conclusion != "neutral"
                     and (($skip_cancelled | not) or .conclusion != "cancelled")))
          | first // empty' <<<"$runs")"
      [ -z "$run" ] || break
    done
    [ -n "$run" ] || continue

    conclusion="$(jq -r .conclusion <<<"$run")"
    url="$(jq -r .html_url <<<"$run")"
    sha="$(jq -r .head_sha <<<"$run")"
    title="Trunk CI red: ${name} on ${branch}"
    issue="$(jq -r --arg title "$title" 'map(select(.title == $title)) | first | .number // empty' <<<"$open_issues")"
    summary="\`${name}\` concluded **${conclusion}** on \`${branch}\` at ${sha}: ${url}"

    case "$conclusion" in
      success)
        if [ -n "$issue" ]; then
          gh issue close "$issue" --comment "Green again. ${summary}"
          echo "closed #${issue}: ${title}"
        fi
        ;;
      failure|timed_out|startup_failure|cancelled|action_required)
        if [ -z "$issue" ]; then
          gh issue create --title "$title" --label bug --label github_actions --label severity:high \
            --body "${summary}

A main-branch run is red. Fix or revert within 24 hours, and do not merge unrelated pull requests onto a red branch. This issue closes itself once \`${name}\` succeeds on \`${branch}\` again."
          echo "opened: ${title}"
        else
          # Read the whole issue before searching it: `grep -q` on a pipe can exit
          # early and SIGPIPE the producer, which pipefail reports as "not found".
          issue_text="$(gh issue view "$issue" --json body,comments --jq '[.body, .comments[].body] | join("\n")')"
          if ! grep -qF "$url" <<<"$issue_text"; then
            gh issue comment "$issue" --body "Still red. ${summary}"
            echo "commented #${issue}: ${title}"
          fi
        fi
        ;;
      *)
        echo "no action for ${name} on ${branch}: ${conclusion}" >&2
        ;;
    esac
  done
done
