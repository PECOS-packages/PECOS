#!/usr/bin/env bash
# Keep GitHub issues open for what is red on the main branches.
#
# Post-merge and scheduled runs catch what pull-request CI does not, so a red
# run there must not go unnoticed. For each watched workflow and branch this
# reads the latest completed non-PR run: if it failed, open an issue (or
# comment on the open one when the failing run is new); if it succeeded, close
# the open issue. A green CodeQL run can still leave open alerts, so one more
# issue tracks open high- and critical-severity code scanning alerts on dev.
#
# A watched workflow that cannot be read is reported and skipped, and the run
# fails at the end, so one renamed workflow does not stop the others.
#
# Usage: scripts/ci/trunk-ci-issues.sh
# Environment: GH_TOKEN and GH_REPO for the gh CLI.
set -euo pipefail
# Command substitutions (issue_text, gh calls) fail the script like any command.
shopt -s inherit_errexit

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
  nightly.yml
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

status=0
open_issues="$(gh issue list --state open --label bug --limit 1000 --json number,title)"

find_issue() {
  jq -r --arg title "$1" 'map(select(.title == $title)) | first | .number // empty' <<<"$open_issues"
}

# The issue body and every comment, read in full. `gh issue view` caps the
# comments it returns, and `grep -q` on a pipe can exit early and SIGPIPE the
# producer, which pipefail would report as "not found".
issue_text() {
  gh api "repos/{owner}/{repo}/issues/$1" --jq .body
  gh api --paginate "repos/{owner}/{repo}/issues/$1/comments?per_page=100" --jq '.[].body'
}

for workflow in "${workflows[@]}"; do
  skip_cancelled=false
  [[ " ${superseded_ok[*]} " == *" $workflow "* ]] && skip_cancelled=true

  if ! name="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}" --jq .name)"; then
    echo "::error::cannot read workflow ${workflow}; was it renamed or removed?"
    status=1
    continue
  fi
  for branch in "${branches[@]}"; do
    # Newest first; page until a run that says something about the branch.
    run=""
    for ((page = 1; ; page++)); do
      if ! runs="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=completed&per_page=100&page=${page}" \
        --jq "[.workflow_runs[] | {conclusion, event, head_sha, html_url}]")"; then
        echo "::error::cannot list runs of ${workflow} on ${branch}"
        status=1
        break
      fi
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
    issue="$(find_issue "$title")"
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
          text="$(issue_text "$issue")"
          if ! grep -qF "$url" <<<"$text"; then
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

# Open high- and critical-severity code scanning alerts on dev, oldest first.
# Alert numbers only grow, so an alert is new when its number is above every
# number the issue already mentions (a reopened alert keeps its old number and
# is not announced again); at most 100 are listed per message, which
# keeps each one under GitHub's body limit and posts a backlog in batches.
title="Trunk code scanning: open high-severity alerts on dev"
issue="$(find_issue "$title")"
# shellcheck disable=SC2016 # the backticks are literal Markdown in the jq output
alerts="$(gh api --paginate "repos/{owner}/{repo}/code-scanning/alerts?state=open&ref=refs/heads/dev&per_page=100" \
  --jq '.[] | select(.rule.security_severity_level == "high" or .rule.security_severity_level == "critical")
        | "\(.number)\t- \(.html_url) `\(.rule.id)` (\(.rule.security_severity_level)) in `\(.most_recent_instance.location.path)`"' |
  sort -n)"
alerts_url="https://github.com/${GH_REPO}/security/code-scanning?query=is%3Aopen+branch%3Adev"

if [ -z "$alerts" ]; then
  if [ -n "$issue" ]; then
    gh issue close "$issue" --comment "No open high- or critical-severity code scanning alerts on dev."
    echo "closed #${issue}: ${title}"
  fi
else
  last_reported=0
  if [ -n "$issue" ]; then
    text="$(issue_text "$issue")"
    last_reported="$({ grep -oE 'code-scanning/[0-9]+' <<<"$text" || true; } | cut -d/ -f2 | sort -n | tail -n 1)"
    last_reported="${last_reported:-0}"
  fi
  new_alerts="$(awk -F'\t' -v last="$last_reported" '$1 > last { print $2 }' <<<"$alerts")"
  if [ -n "$new_alerts" ]; then
    count="$(wc -l <<<"$new_alerts")"
    listed="$(head -n 100 <<<"$new_alerts")"
    if [ "$count" -gt 100 ]; then
      listed="${listed}
- ...and $((count - 100)) more, listed in later comments: ${alerts_url}"
    fi
    if [ -z "$issue" ]; then
      gh issue create --title "$title" --label bug --label severity:high \
        --body "Open high- and critical-severity code scanning alerts on \`dev\`:

${listed}

Fix each alert, or dismiss it with a reason in the Security tab if it is a false positive. This issue gets a comment when new alerts appear and closes itself when none are open."
      echo "opened: ${title}"
    else
      gh issue comment "$issue" --body "New open alerts on \`dev\`:

${listed}"
      echo "commented #${issue}: ${title}"
    fi
  fi
fi

exit "$status"
