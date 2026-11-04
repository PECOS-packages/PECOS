# PECOS ZIP 9 backport

This directory vendors the crates.io `ndarray-npy` 0.9.1 release:

- Upstream: <https://github.com/jturner314/ndarray-npy>
- Release source commit: `61701168bc364adb5882cd111485e001af4823e3`
- Crate SHA-256: `7b313788c468c49141a9d9b6131fc15f403e6ef4e8446a0b2e18f664ddb278a9`
- License: MIT OR Apache-2.0 (both license files are retained).

The source, tests, fixtures, examples, and upstream documentation are retained.
`Cargo.toml` uses the release's original, unnormalized manifest. Registry metadata,
the upstream lockfile, and upstream CI configuration are omitted.

## Local changes

Backport the applicable parts of upstream commit
[`0215776b072b80b5d5bfe11455f4e8584f14501e`](https://github.com/jturner314/ndarray-npy/commit/0215776b072b80b5d5bfe11455f4e8584f14501e):

- Require `zip = "9"` instead of `zip = "2"`, preserving its feature selection.
- Raise the declared Rust version to 1.88, as required by ZIP 9.
- Adapt `NpzReader::names` to ZIP 9's fallible filename API.

The ndarray 0.16 dependency, crate version, and decoder implementation are unchanged.
The workspace's existing ZIP 9.0.5 satisfies the new dependency, so the lockfile
removes ZIP 2.4.2 without updating unrelated packages.

## Why a backport is necessary

As checked on 2026-11-02, the latest published `relay-bp` is 0.2.2. It requires
ndarray-npy 0.9 and ndarray < 0.17; upstream relay still has those constraints.
Published ndarray-npy 0.10.0 requires ndarray 0.17 and ZIP 6, so upgrading to it
would neither preserve decoder types nor fix RUSTSEC-2026-0429. Upstream's
unreleased ZIP 9 update also requires ndarray 0.17. This backport upgrades only
the archive dependency while preserving PECOS's existing relay adapter.

[RUSTSEC-2026-0429](https://rustsec.org/advisories/RUSTSEC-2026-0429.html) affects
`ZipArchive::extract` and is fixed in ZIP >= 9.0.0. ndarray-npy reads entries
without calling `extract`, but retaining an affected ZIP package still fails the
workspace's dependency policy. No advisory suppression is needed with this patch.

## Validation and removal

The workspace tests in `crates/pecos-relay-bp/tests/npz_compatibility.rs` exercise
stored and deflated archives, array names and lookup, and relay's sparse-matrix
loader. Every Maturin package declares development-only path dependencies on
both root patch targets (this crate and the HiGHS shim), so its source archive
contains those crates and their local dependencies. The patch table alone is not
discovered by Maturin's source-package traversal. These declarations do not add
production dependency edges. Existing relay and native BP parity tests verify
decoder compatibility.

`scripts/ci/check_rust_sdists.py` builds all five Maturin source distributions
through their configured PEP 517 backends and extracts them outside the checkout.
It verifies both local patches are present, permits offline pruning of the copied
workspace lockfile without introducing new package pins, then requires locked
all-features metadata resolution with all local sources inside the archive.
The Python lint workflow runs this check. Selene plugins use Hatchling and are
not inputs to this Maturin check.
The retained upstream suite can also be run with
`cargo test --manifest-path vendor/ndarray-npy/Cargo.toml --all-features`.

This directory is excluded from workspace membership to preserve upstream's
formatting/lint conventions, and is excluded from first-party pre-commit hooks.
The workspace version checker lists it as an intentionally independent dependency.
Its source is still built through `[patch.crates-io]`.
Remove this patch and directory when relay-bp accepts a published ndarray-npy
version using ZIP >= 9; rerun the decoder tests and the locked all-features
cargo-deny check when doing so. Cargo patches apply at the workspace root; a
consumer building PECOS from a different root must carry an equivalent patch
until upstream releases a compatible fix.
