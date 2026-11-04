# pecos-selene-stab-mps

PECOS StabMps (stabilizer tableau + MPS) simulator plugin for the Selene quantum emulator.

Stabilizer gates are O(n) on the tableau; non-Clifford rotations decompose in the stabilizer basis and apply to the MPS. Cost is polynomial when non-Clifford count is bounded.

## Usage

```python
from pecos_selene_stab_mps import StabMpsPlugin

sim = StabMpsPlugin()
```

The plugin inherits `StabMps::for_qec()`'s Exact measurement mode. The frozen
Stage-B repetition-code benchmark in
`exp/pecos-stab-tn/examples/measurement_mode_bench.rs` measured a 1.497x
geometric-mean slowdown and a 1.548x per-workload maximum versus Pragmatic on
that noisy repetition-code family,
inside the decided Exact-everywhere limits. Exact avoids Pragmatic mode's
known conditional-state bias on honest non-Clifford circuit families.

## Quantinuum SW releases

`.github/workflows/stab-mps-release.yml` builds only this plugin. It produces
`py3-none` wheels for Python 3.12+ on Linux x86-64 (manylinux 2.28), macOS 13+
(Intel and Apple Silicon), and Windows x86-64. CUDA and LLVM are not required.
Linux wheels are built in a manylinux container and repaired with auditwheel;
each platform installs its wheel in an isolated environment and loads the native
library using `scripts/ci/smoke_selene_wheel.py`.

Configure these Actions secrets in the fork before publishing (organization
secrets must explicitly grant the fork access):

- `QTNM_ARTIF_SW_WRITE_USER`
- `QTNM_ARTIF_SW_WRITE_TOKEN`

The destination is
`https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/`, matching the
guppyft-lib release workflow.

To release:

1. Set `[project].version` in this plugin's `pyproject.toml` to a unique PEP 440
   version, regenerate `uv.lock` with `uv lock`, and commit both files. For fork
   builds, a local version such as
   `0.11.0.dev0+qtnm.1` distinguishes the package from upstream. This Python
   distribution version is independent of the Rust workspace version; no Rust
   version bump is needed.
2. Optionally run **StabMps Release** manually on the branch to build and test
   without publishing. Manual runs and pull requests never receive upload
   credentials. GitHub enables manual dispatch once the workflow exists on the
   default branch; before then, opening a pull request runs the build checks.
3. Tag that commit with the exact package version and push the tag:

   ```bash
   git tag stab-mps-v0.11.0.dev0+qtnm.1
   git push origin stab-mps-v0.11.0.dev0+qtnm.1
   ```

The tag must match the Python package version. After all four builds and smoke
tests pass, the workflow uploads the wheels to SW Artifactory and creates a
GitHub release with the same wheels attached. Development and prerelease
versions are marked as GitHub prereleases. Use a new version for each release;
the workflow does not overwrite existing Artifactory uploads.

Install a published fork release using the authenticated SW index, for example:

```bash
uv pip install --index https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple \
  --index https://pypi.org/simple 'pecos-selene-stab-mps==0.11.0.dev0+qtnm.1'
```

Supply read credentials through your normal package-index authentication.
Consumers such as guppyft-lib must replace their Git source override with the
published version to use the wheels.
