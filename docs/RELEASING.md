# Releasing

A release is a `v*` tag. `.github/workflows/release.yml` then builds
`tsfm-benchmark` (repo root) and `tsfm-lens` (`tsfm_lens/`), publishes both to
PyPI with Trusted Publishing, and creates a GitHub Release with the sdists and
wheels attached. The workflow refuses to run past the build step if the tag
does not equal the `version` in both `pyproject.toml` files.

## One-time setup

### 1. PyPI trusted publishers (both projects)

Neither project name exists on PyPI yet (checked 2026-10-03: `tsfm-lens` and
`tsfm-benchmark` both returned 404), so register them as *pending publishers*
at <https://pypi.org/manage/account/publishing/>. Add one for each:

| Field | `tsfm-benchmark` | `tsfm-lens` |
|---|---|---|
| PyPI project name | `tsfm-benchmark` | `tsfm-lens` |
| Owner | `DanK10097110` | `DanK10097110` |
| Repository | `TSFM-Interp` | `TSFM-Interp` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment name | `pypi` | `pypi` |

Names can be claimed by anyone before the first upload, so register soon.

### 2. GitHub environment

Repository Settings -> Environments -> New environment named `pypi`. Optionally
add yourself as a required reviewer to get a manual approval gate before the
upload.

### 3. Zenodo

Log in at <https://zenodo.org> with GitHub, open GitHub settings in Zenodo
(<https://zenodo.org/account/settings/github/>) and switch the
`TSFM-Interp` repository on. Zenodo mints a DOI when the GitHub Release is
published, using `.zenodo.json`. Add the DOI badge to the README afterwards.

## Each release

1. Make sure `dev` has been merged to the branch you release from and that the
   versions agree: `pyproject.toml`, `tsfm_lens/pyproject.toml`,
   `CITATION.cff` (`version`) and the `CHANGELOG.md` heading.
2. Optional: add `date-released: "YYYY-MM-DD"` to `CITATION.cff` (it is
   commented out so the file never carries a wrong date). Zenodo and GitHub
   record the release date themselves, so skipping this is harmless.
3. Tag and push:

   ```bash
   git tag v1.0.0 && git push origin v1.0.0
   ```

4. Watch the Actions run. After `publish-pypi` succeeds, the packages are at
   <https://pypi.org/project/tsfm-benchmark/> and
   <https://pypi.org/project/tsfm-lens/>, and the GitHub Release appears.

A PyPI version can never be re-uploaded, even after deletion. If a release run
fails after the upload step, bump to `1.0.1` rather than retagging.

## Local dry run (no upload)

```bash
python -m venv .venv_rel && .venv_rel/bin/pip install build twine
.venv_rel/bin/python -m build --outdir dist .
.venv_rel/bin/python -m build --outdir dist tsfm_lens
.venv_rel/bin/twine check dist/*
```

Install the wheels into an empty venv and run `tsfm-lens --help` before
tagging. The wheels contain no configs: run the smoke preset from a repository
checkout (`cd tsfm_lens && tsfm-lens --config configs/smoke.yaml`).
