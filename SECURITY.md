# Security Policy

## Reporting a Vulnerability

Please do not file public issues for suspected vulnerabilities.

Use GitHub's private vulnerability reporting for this repository when available. If private reporting is not available to you, open a minimal public issue asking for a private security contact and do not include exploit details, tokens, logs, or proof-of-concept code.

Include enough detail for maintainers to reproduce and assess the issue privately:

- affected package, crate, workflow, or component
- affected versions, commits, or release artifacts
- impact and prerequisites
- reproduction steps or a minimal proof of concept
- any suspected dependency or CI provenance issue

## Dependency and CI Incidents

For suspected dependency compromise, malicious package activity, leaked credentials, or CI tampering, include the exact package name, version, registry, lockfile entry, workflow run, or artifact involved. Maintainers should treat these reports as potentially active incidents until dependencies, generated artifacts, and GitHub Actions tokens have been checked.

## Supported Versions

Security fixes are prioritized for the default branch and currently maintained release lines. Older development snapshots may receive fixes only when the affected code is still supported or the risk carries forward into supported releases.
