# Releasing PECOS (Python packages)

The final PyPI push is deliberately **manual**: artifact building and testing
are automated, but a human runs the upload and confirms each package. This
sequence was last exercised for `0.10.0.dev0`.

## 1. Version bump (its own PR)

The version is a literal in many coordinated places. One command moves them all:

```
just bump-python-version <new-version>
```

It rewrites `[project].version` in the root meta-package and in every distribution
(`python/pecos-rslib{,-llvm,-cuda,-exp}`, `python/quantum-pecos`, the
`python/selene-plugins/*` packages, and `exp/zluppy`, which rides the same train from
outside the uv workspace), rewrites the exact internal pins (the root's
`quantum-pecos[cuda12/13]==...` and quantum-pecos's `pecos-rslib==...` /
`pecos-rslib-llvm==...`), regenerates all three lockfiles with the pinned uv from
`.github/uv.toml`, and then runs `just python-workspace-check`. Nothing is written unless
every file is on the old version, so a bump cannot half-land.

Verify: the lockfile diffs are version lines only; `git grep <old-version>` returns
nothing outside this file's historical note; `uv lock --check` passes;
`just python-ci-sync-test` then `python -c "import pecos; print(pecos.__version__)"`
reports the new version.

Versioning convention: a user-visible default-behavior change gets a minor
bump (e.g. `0.8.x -> 0.9.0.dev0`); additive-only work increments the dev/patch
number.

### Version trains

Each language ships on its own train, and each train has a guard that fails CI when
a member drifts off it:

| Train | Source of truth | Guard |
| --- | --- | --- |
| Python | `[project].version` in the root `pyproject.toml` | `just python-workspace-check` -- every tracked `pyproject.toml` with a `[build-system]` must match, wherever it lives |
| Rust | `[workspace.package].version` in the root `Cargo.toml` | `just rust-workspace-check` -- every workspace member must use `version.workspace = true` |
| Julia | `version` in `julia/PECOS.jl/Project.toml` | `just julia-version-check` -- manifest, FFI crate, and `build_tarballs.jl` must agree, as must the Julia compat bound; `just rust-workspace-check` separately holds the FFI crate to it |
| Go | `version` in `go/pecos-go-ffi/Cargo.toml` | single declaration: `pecos_version()` derives the string from `CARGO_PKG_VERSION`, and Go modules are versioned by git tag |

A Python release moves only the Python train. The Rust, Julia, and Go trains are
bumped on their own schedules, and the guards keep them from being dragged along
by accident. Julia has its own bump command, matching the Python one:

```
just bump-julia-version <new-version>
```

It rewrites all three Julia files, refreshes `Cargo.lock` for the FFI crate, and runs both
guards. Rust and Go are a single literal each (`[workspace.package].version` and
`go/pecos-go-ffi/Cargo.toml`), so they need no tool.

## 2. Merge and wait for post-merge green

Merge the bump PR to `dev`, then wait for **all** post-merge workflows on the
merge commit to pass -- including the long ones (`python-core` is the full
suite and only runs post-merge; `Python Artifacts` is the full wheel matrix,
i.e. the release rehearsal). Optionally run `just check-all` locally on the
merge commit as a belt-and-braces gate.

## 3. Tag

```
git tag py-<version> <merge-sha>
git push origin py-<version>
```

The `py-*` tag triggers `python-release.yml` in full-release mode: all
platform wheels (pecos-rslib, pecos-rslib-llvm), quantum-pecos wheel + sdist,
abi3 wheel tests across Python versions, and a `collect_artifacts` bundle
pinned to the tagged commit. Wait for it to go green. Publish from the **tag
run's** bundle, not a branch run -- provenance stays tied to the immutable
ref.

## 4. Download the bundle and dry-run

```
gh run download <tag-run-id> -n pecos-distribution -D pecos-distribution-<version>
./scripts/publish-wheels.sh --dry-run -f pecos-distribution-<version>
```

(`-f` accepts the extracted directory `gh` produces, or the original zip.)

Download into a version-suffixed directory. A previous release's bundle left in
a plain `pecos-distribution/` would mix two versions in one directory, and the
preflight -- complete set, consistent version, only expected files -- is the
only thing standing between that and a stray upload.

The dry run must show a real `twine check ... PASSED` per file (twine must be
installed, e.g. `uv tool install twine`). Keep it current: build tools raise the
core-metadata version they emit over time, and a checker older than the metadata
rejects the artifacts even though PyPI accepts them -- `uv tool upgrade twine` if a
check fails naming a metadata version. The script prints which twine it is using, and
validates every package before uploading any of them, so a stale checker stops the run
rather than surfacing after the first packages are already public. It prints only a
`Distribution checks passed` summary and swallows `twine check` output unless it
fails, so confirm the per-file result directly:
`uvx twine check pecos-distribution-<version>/*/*.whl pecos-distribution-<version>/*/*.tar.gz`
and check the `PASSED` count equals the file count.

## 5. Publish (the manual step)

```
./scripts/publish-wheels.sh -f pecos-distribution
```

Confirm each package at its prompt. Upload order matters and the script
enforces it: all packages are preflighted (complete set, consistent version,
only expected files) before anything uploads, dependencies go first
(`pecos-rslib` -> `pecos-rslib-llvm` -> `quantum-pecos`, which pins both at
exact versions), and any failure or declined prompt aborts the remaining
uploads rather than continuing. Publishing `quantum-pecos` alone (`-p`) is
blocked unless its pinned dependencies already exist on PyPI. Checks run
`twine check --strict`.
Credentials come from `~/.pypirc`; a **new** package's first upload needs an
account-scoped token (project-scoped tokens cannot create projects).

## 6. GitHub release

Create a release on the tag (marked pre-release for dev versions), attach the
bundle's files, and include:

- the headline change and any escape hatch / migration note
- an install snippet -- uv users need `--prerelease=allow` for dev-version
  transitive pins

```
gh release create py-<version> --prerelease --title "py-<version>" \
  --notes-file <notes.md> --generate-notes <dist-files...>
```

## 7. Verify from PyPI

Prove the published set resolves and runs, from a clean environment:

```
uv venv /tmp/verify && uv pip install --python /tmp/verify \
  --no-cache --prerelease=allow quantum-pecos==<version>
/tmp/verify/bin/python -c "import pecos; print(pecos.__version__)"
```

(`--no-cache` matters: a resolver that looked while an upload was in flight
may have cached "not found".)
