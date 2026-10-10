#!/usr/bin/env bash
# Keep GitHub issues open for what is red on the main branches.
#
# Post-merge and scheduled runs catch what pull-request CI does not, so a red
# run there must not go unnoticed. For each watched workflow and branch this
# reads push and schedule runs separately, deferring active reruns: failures open
# issues (or comment when the failing run is new); successes close them. Daily
# schedules also get a missing-run backstop; a newly listed or changed schedule
# gets 36 hours before it can raise "Nightly missing". Recent release tags are tracked:
#   Trunk CI red: ${name} on ${branch} (nightly)
#   Nightly missing: ${name} on dev
#   Release tag CI red: ${name} on ${tag}
# A green CodeQL run can still leave open alerts, so one more
# issue tracks open high- and critical-severity code scanning alerts on dev.
#
# A watched workflow that cannot be read is reported and skipped, and the run
# fails at the end, so one renamed workflow does not stop the others.
#
# Usage: scripts/ci/trunk-ci-issues.sh
# Environment: GH_TOKEN and GH_REPO for the gh CLI; TRUNK_CI_NOW optionally
# supplies an ISO 8601 clock for deterministic staleness and release lookbacks.
set -euo pipefail
# Command substitutions (issue_text, gh calls) fail the script like any command.
shopt -s inherit_errexit

branches=(dev)
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
  pre-commit.yml
  python-release.yml
  python-test.yml
  python-version-consistency.yml
  rust-test.yml
  rust-version-consistency.yml
  selene-general-noise-semantics.yml
  selene-plugins.yml
  test-docs-examples.yml
)
# These artifact workflows finish running trunk builds, but a newer pending
# push still replaces an older pending one with conclusion cancelled. That
# says nothing about the branch, so pushes keep the exemption; scheduled
# cancellations are red.
superseded_ok=(julia-release.yml python-release.yml)

# Only workflows with a daily cron belong here; a missing run is independent
# of whether the latest completed nightly passed. Allow 36 hours because
# schedules can arrive 5-9 hours late without a dropped day. A newly listed or
# changed schedule gets 36 hours from its workflow file's last change on dev
# before it can raise "Nightly missing".
daily_workflows=(
  cargo-deny.yml
  codeql.yml
  cuda-build-check.yml
  dependency-integrity-check.yml
  github-actions-security.yml
  julia-release.yml
  julia-test.yml
  julia-version-consistency.yml
  nightly.yml
  osv-scanner.yml
  pre-commit.yml
  python-release.yml
  python-test.yml
  python-version-consistency.yml
  rust-test.yml
  rust-version-consistency.yml
  selene-general-noise-semantics.yml
  selene-plugins.yml
  test-docs-examples.yml
)
now="$(date -u -d "${TRUNK_CI_NOW:-now}" +%s)"
nightly_cutoff=$((now - 36 * 60 * 60))
release_cutoff=$((now - 14 * 24 * 60 * 60))
release_since="$(date -u -d "@${release_cutoff}" +%Y-%m-%dT%H:%M:%SZ)"
recent_tags=()
declare -A release_runs=()

status=0
open_issues="$(gh issue list --state open --label bug --limit 1000 --json number,title)"

find_issue() {
  jq -r --arg title "$1" 'map(select(.title == $title)) | first | .number // empty' <<<"$open_issues"
}

# The issue body and every comment, read in full. `gh issue view` caps the
# comments it returns, and `grep -q` on a pipe can exit early and SIGPIPE the
# producer, which pipefail would report as "not found".
issue_text() {
  gh api "repos/{owner}/{repo}/issues/$1" --jq .body || return 1
  gh api --paginate "repos/{owner}/{repo}/issues/$1/comments?per_page=100" --jq '.[].body'
}

# Explicit returns matter here: callers catch failures to reconcile the other
# classes, which disables errexit inside a called function.
reconcile_run() {
  local name="$1" branch="$2" title="$3" run="$4" kind="$5"
  local conclusion url sha issue summary body text
  conclusion="$(jq -r .conclusion <<<"$run")" || return 1
  url="$(jq -r .html_url <<<"$run")" || return 1
  sha="$(jq -r .head_sha <<<"$run")" || return 1
  issue="$(find_issue "$title")" || return 1
  summary="\`${name}\` concluded **${conclusion}** on \`${branch}\` at ${sha}: ${url}"

  case "$conclusion" in
    success)
      if [ -n "$issue" ]; then
        gh issue close "$issue" --comment "Green again. ${summary}" || return 1
        echo "closed #${issue}: ${title}"
      fi
      ;;
    failure|timed_out|startup_failure|cancelled|action_required)
      if [ -z "$issue" ]; then
        body="A main-branch run is red. Fix or revert within 24 hours, and do not merge unrelated pull requests onto a red branch. This issue closes itself once \`${name}\` succeeds on \`${branch}\` again."
        if [ "$kind" = release ]; then
          body="A release-tag run is red. This issue closes itself once \`${name}\` succeeds on \`${branch}\` again."
        fi
        gh issue create --title "$title" --label bug --label github_actions --label severity:high \
          --body "${summary}

${body}" || return 1
        echo "opened: ${title}"
      else
        text="$(issue_text "$issue")" || return 1
        if ! grep -qF "$url" <<<"$text"; then
          gh issue comment "$issue" --body "Still red. ${summary}" || return 1
          echo "commented #${issue}: ${title}"
        fi
      fi
      ;;
    *)
      echo "no action for ${name} on ${branch}: ${conclusion}" >&2
      ;;
  esac
}

check_class() {
  local workflow="$1" name="$2" branch="$3" event="$4"
  local skip_cancelled=false
  if [ "$event" = push ] && [[ " ${superseded_ok[*]} " == *" $workflow "* ]]; then
    skip_cancelled=true
  fi
  local page runs run title count run_status
  # Filter events at the API so absent nightlies do not scan all push history.
  # First attempts in flight and ignored conclusions may require another page.
  # A rerun in flight is a barrier: older results cannot resolve its issue.
  for ((page = 1; ; page++)); do
    runs="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}/runs?branch=${branch}&event=${event}&per_page=100&page=${page}")" || return 1
    count="$(jq '.workflow_runs | length' <<<"$runs")" || return 1
    # Freshness needs the newest scheduled run, even if this class scans on.
    if [ "$branch" = dev ] && [ "$event" = schedule ] && [ "$page" -eq 1 ]; then
      schedule_first_page="$runs"
    fi
    [ "$count" -gt 0 ] || return 0
    run="$(jq -c --arg event "$event" --argjson skip_cancelled "$skip_cancelled" '
      [.workflow_runs[] | select(.event == $event
        and ((.status != "completed" and (.run_attempt // 1) > 1)
          or (.status == "completed" and .conclusion != "skipped" and .conclusion != "neutral"
            and (($skip_cancelled | not) or .conclusion != "cancelled"))))]
      | first // empty' <<<"$runs")" || return 1
    if [ -z "$run" ]; then
      [ "$count" -eq 100 ] || return 0
      continue
    fi
    run_status="$(jq -r .status <<<"$run")" || return 1
    [ "$run_status" = completed ] || return 0
    title="Trunk CI red: ${name} on ${branch}"
    [ "$event" != schedule ] || title+=" (nightly)"
    reconcile_run "$name" "$branch" "$title" "$run" trunk || return 1
    return 0
  done
}

check_nightly() {
  local workflow="$1" name="$2" runs="$3" run created url issue title summary
  local commits changed changed_epoch
  # Reuse page 1, including queued/running runs that prove the schedule fired.
  run="$(jq -c '.workflow_runs | first // empty' <<<"$runs")" || return 1
  title="Nightly missing: ${name} on dev"
  issue="$(find_issue "$title")" || return 1
  created=""
  summary="No scheduled run was found for \`${name}\` on \`dev\`."
  if [ -n "$run" ]; then
    created="$(jq -r .created_at <<<"$run")" || return 1
    url="$(jq -r .html_url <<<"$run")" || return 1
    summary="The last scheduled run of \`${name}\` on \`dev\` was created at ${created}: ${url}"
    created="$(date -u -d "$created" +%s)" || return 1
  fi
  if [ -n "$created" ] && [ "$created" -ge "$nightly_cutoff" ]; then
    if [ -n "$issue" ]; then
      gh issue close "$issue" --comment "Nightly schedule resumed. ${summary}" || return 1
      echo "closed #${issue}: ${title}"
    fi
  elif [ -z "$issue" ]; then
    # Only a prospective missing issue needs the schedule's establishment time.
    commits="$(gh api "repos/{owner}/{repo}/commits?path=.github/workflows/${workflow}&sha=dev&per_page=1")" || return 1
    changed="$(jq -er '.[0].commit.committer.date | strings | select(length > 0)' <<<"$commits")" || return 1
    changed_epoch="$(date -u -d "$changed" +%s)" || return 1
    if [ "$changed_epoch" -ge "$nightly_cutoff" ]; then
      echo "nightly check held off for ${name}: schedule changed at ${changed}"
      return 0
    fi
    gh issue create --title "$title" --label bug --label github_actions --label severity:high \
      --body "No scheduled run was created in the last 36 hours. ${summary}

The 36-hour backstop detects dropped daily schedules. This issue closes itself when a recent scheduled run exists." || return 1
    echo "opened: ${title}"
  fi
}

read_tag_runs() {
  local tag="$1" encoded_tag page runs count pages=""
  encoded_tag="$(jq -rn --arg tag "$tag" '$tag | @uri')" || return 1
  # One repository listing serves recency and every workflow. Preserve page
  # order so an older rerun cannot hide a newer completed result.
  for ((page = 1; ; page++)); do
    runs="$(gh api "repos/{owner}/{repo}/actions/runs?branch=${encoded_tag}&event=push&per_page=100&page=${page}")" || return 1
    count="$(jq '.workflow_runs | length' <<<"$runs")" || return 1
    pages+="${runs}"$'\n'
    [ "$count" -eq 100 ] || break
  done
  jq -cs '[.[].workflow_runs[]]' <<<"$pages"
}

discover_tags() {
  local prefix tags tag runs recent
  local -a candidates=()
  local -A seen=() issue_tags=()
  for prefix in py- jl- rs-; do
    if ! tags="$(gh api --paginate "repos/{owner}/{repo}/git/matching-refs/tags/${prefix}" \
      --jq '.[].ref | ltrimstr("refs/tags/")')"; then
      echo "::error::cannot list release tags with prefix ${prefix}"
      status=1
      continue
    fi
    while IFS= read -r tag; do
      [ -n "$tag" ] || continue
      [ -z "${seen[$tag]:-}" ] || continue
      seen["$tag"]=1
      candidates+=("$tag")
    done <<<"$tags"
  done
  # Open issues outlive the discovery window, even if a tag ref was deleted.
  # Workflow names may contain " on "; tag names cannot contain spaces.
  tags="$(jq -r '[.[] | .title | select(startswith("Release tag CI red: "))
    | capture("^Release tag CI red: .+ on (?<tag>.+)$").tag] | unique[]' <<<"$open_issues")"
  while IFS= read -r tag; do
    [ -n "$tag" ] || continue
    issue_tags["$tag"]=1
    [ -z "${seen[$tag]:-}" ] || continue
    seen["$tag"]=1
    candidates+=("$tag")
  done <<<"$tags"
  for tag in "${candidates[@]}"; do
    if ! runs="$(read_tag_runs "$tag")"; then
      echo "::error::cannot list release tag runs for ${tag}"
      status=1
      continue
    fi
    # API created_at timestamps and release_since use the same UTC ISO format.
    if ! recent="$(jq --arg cutoff "$release_since" 'any(.[]; .created_at >= $cutoff)' <<<"$runs")"; then
      echo "::error::cannot determine recency for release tag ${tag}"
      status=1
      continue
    fi
    if [ "$recent" = true ] || [ -n "${issue_tags[$tag]:-}" ]; then
      recent_tags+=("$tag")
      release_runs["$tag"]="$runs"
    fi
  done
}

check_tags() {
  local workflow="$1" name="$2" run tag title run_status
  local result=0
  for tag in "${recent_tags[@]}"; do
    # Repository results mix workflows. Filter by file, then skip first attempts
    # in flight; an active rerun blocks older results only for this workflow.
    # Skipped/neutral leave the issue alone; cancellation is always red here.
    if ! run="$(jq -c --arg path ".github/workflows/${workflow}" '
      [.[] | select(.path == $path and .event == "push"
        and (.status == "completed" or (.run_attempt // 1) > 1))]
      | first // empty' <<<"${release_runs[$tag]}")"; then
      result=1
      continue
    fi
    [ -n "$run" ] || continue
    if ! run_status="$(jq -r .status <<<"$run")"; then
      result=1
      continue
    fi
    [ "$run_status" = completed ] || continue
    title="Release tag CI red: ${name} on ${tag}"
    reconcile_run "$name" "$tag" "$title" "$run" release || result=1
  done
  return "$result"
}

discover_tags

for workflow in "${workflows[@]}"; do
  if ! name="$(gh api "repos/{owner}/{repo}/actions/workflows/${workflow}" --jq .name)"; then
    echo "::error::cannot read workflow ${workflow}; was it renamed or removed?"
    status=1
    continue
  fi
  # An unread first page must not reuse another workflow's freshness data.
  schedule_first_page=""
  for branch in "${branches[@]}"; do
    for event in push schedule; do
      if ! check_class "$workflow" "$name" "$branch" "$event"; then
        echo "::error::cannot reconcile ${workflow} on ${branch} (${event})"
        status=1
      fi
    done
  done
  if [[ " ${daily_workflows[*]} " == *" $workflow "* ]] && [ -n "$schedule_first_page" ]; then
    if ! check_nightly "$workflow" "$name" "$schedule_first_page"; then
      echo "::error::cannot reconcile nightly staleness for ${workflow}"
      status=1
    fi
  fi
  if ! check_tags "$workflow" "$name"; then
    echo "::error::cannot reconcile release tags for ${workflow}"
    status=1
  fi
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
