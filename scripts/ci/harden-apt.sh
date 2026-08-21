#!/usr/bin/env bash
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
#
# Make apt survive a flaky mirror on hosted runners.
#
# GitHub's Ubuntu images point apt at azure.archive.ubuntu.com, which goes unreachable
# often enough that runner-images carries long-running issues about it. When it does, a
# fetch can stall rather than fail: on 2026-08-18 the CUDA install step sat for over four
# hours on apt with no output, and on 2026-08-19 it burned its 15-minute budget the same
# way. Neither is a slow download -- the step takes about 40 seconds when the mirror
# answers.
#
# Three settings, each aimed at that failure:
#   ForceIPv4  - apt prefers IPv6 when the runner advertises it, and a blackholed IPv6
#                route is the usual cause of a hang rather than an error.
#   Timeout    - bounds a stalled connection so it fails and can be retried, instead of
#                hanging until the step's own timeout kills it.
#   Retries    - apt 2.3.2+ retries 3 times by default; 5 covers a mirror that is
#                flapping rather than down.
set -euo pipefail

sudo tee /etc/apt/apt.conf.d/99-pecos-ci >/dev/null <<'CONF'
Acquire::ForceIPv4 "true";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
Acquire::Retries "5";
CONF

echo "apt hardened for CI:"
cat /etc/apt/apt.conf.d/99-pecos-ci
