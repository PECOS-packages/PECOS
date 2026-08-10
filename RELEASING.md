# Releasing PECOS (Python packages)

The final PyPI push is deliberately **manual**: artifact building and testing
are automated, but a human runs the upload and confirms each package. This
sequence was last exercised for `0.9.0.dev1`.

## 1. Version bump (its own PR)

The version is a literal in many coordinated places. Bump them all:

- `version =` in every workspace `pyproject.toml` (root meta-package,
  `python/pecos-rslib{,-llvm,-cuda,-exp}`, `python/quantum-pecos`, and the
  `python/selene-plugins/*` packages)
- Exact-version internal pins: the root's `quantum-pecos[cuda12/13]==...`
  entries and quantum-pecos's `pecos-rslib==...` / `pecos-rslib-llvm==...`
- Regenerate all three lockfiles: `just lock` (runs the pinned uv from
  `.github/uv.toml` at the root and in `exp/zlup/` and `exp/zluppy/`; verify
  the diffs are version-lines-only)

Verify: `git grep <old-version>` returns nothing outside this file's
historical note; `uv lock --check` passes;
`just python-ci-sync-test` then `python -c "import pecos; print(pecos.__version__)"`
reports the new version.

Versioning convention: a user-visible default-behavior change gets a minor
bump (e.g. `0.8.x -> 0.9.0.dev0`); additive-only work increments the dev/patch
number.

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
gh run download <tag-run-id> -n pecos-distribution -D pecos-distribution
./scripts/publish-wheels.sh --dry-run -f pecos-distribution
```

(`-f` accepts the extracted directory `gh` produces, or the original zip.)

The dry run must show a real `twine check ... PASSED` per file (twine must be
installed, e.g. `uv tool install twine`).

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
