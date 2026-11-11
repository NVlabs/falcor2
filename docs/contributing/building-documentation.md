(falcor2-building-documentation)=
# Building documentation

Falcor2 uses Sphinx with the Furo theme, MyST Markdown sources, and SlangPy's shared native API generator. The
build includes the existing guides and design pages alongside reference pages extracted from real Python and
native bindings. It does not execute examples or create rendering devices.

## Prerequisites

Use Python 3.12 or newer for the documentation toolchain and an existing Falcor2 development environment. Initialize
the repository's pinned SlangPy submodule and build Falcor2 Release using the normal development workflow before
building documentation. Rebuild when native APIs change. The documentation command does not configure or compile
native code, and fails if it cannot import this checkout's bindings or shared generator.

Install the documentation dependencies into that same environment, from the repository root:

```bash
python -m pip install -r docs/requirements.txt
```

On Windows with a repository-local environment, substitute `.\.conda\python.exe` for `python`. Do not use a
different installed Falcor2 package to document this checkout.

## Build and preview

From the repository root:

```bash
python -m sphinx -M html docs docs/_build
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open [the local preview](http://127.0.0.1:8000/) and stop the server with Ctrl+C when finished. Use another port if
8000 is occupied. Warnings are reported without failing the build; actual build errors still fail.

To rebuild all pages and disregard Sphinx's cached document state:

```bash
python -m sphinx -M html docs docs/_build -E -a
```

For a separate strict validation, add `-W --keep-going` to treat warnings as errors.

## Uploading the result

Upload the entire `docs/_build/html` directory to a static web host, including its `_static` assets and search
index. The site uses relative navigation and works beneath a hosting subdirectory. No Python service is required
by the hosted site.

## GitLab Pages

The protected default-branch web or scheduled pipeline with `BUILD_WHEELS=true` builds documentation once, at the
end of the Linux Python 3.12 wheel job. It uses that job's current Release bindings and labels the site with the full
wheel version, including the pipeline suffix (for example, `0.1.0+rtr.123`). Documentation warnings remain visible in
the job log but do not block wheel publication. Actual documentation build errors still fail the job.
After `pip-publish` succeeds, the final `pages` job publishes the site.

Find the base URL under **Deploy > Pages** in GitLab. The root lists all available versions. Each release is at
`<base-url>/0.1.0+rtr.123/`, with a complete duplicate of the newest release at `<base-url>/latest/`. Older versions
remain available. An older pipeline that finishes late adds its version without moving `latest` backwards.

Enable Pages and the package registry for the project, and configure the desired Pages access control in GitLab.
The job uses `CI_JOB_TOKEN`; it does not need a separate publishing token. The pinned SlangPy generator commit
must be available from the submodule remote before a fresh CI checkout can run.

### Keeping version history

Every deployment includes the complete history because GitLab Pages replaces the previous site. The serialized
`pages` job restores and updates a cumulative archive in the project's generic package registry:
`falcor2-docs-history`, version `1`, asset `site.zip`. Allow duplicate assets for this package; downloading it
returns the newest saved archive. Other download errors or a corrupt archive stop publication; HTTP 404 initializes
the first release. The job saves the archive before handing the complete `public/` artifact to Pages.

Retain the newest archive permanently. A package cleanup policy may keep only the newest duplicate assets, since
each contains all versions. Do not delete the package or its final asset: the next publication would interpret a
missing archive as a first release. If history is lost, restore `site.zip` from a backup, or zip the contents of a
complete saved `public/` artifact and upload it to the same package location before publishing again. Job artifacts
expire after a week; they are useful for diagnosis and recovery but are not the history store. Version pruning is
manual and is not part of the publishing job.

Retry a failed Pages job after addressing its error. If the history upload succeeded before deployment failed,
the retry restores that saved history and replaces only its own version. The resource group prevents concurrent
jobs from overwriting each other's history.

### Reproducing the CI steps locally

After building the wheels, use their build environment to generate the versioned site:

```bash
python tools/docs.py build --wheelhouse .tmp/pip-wheels/wheelhouse
python tools/docs.py stage --pipeline-iid 123
python -m http.server 8000 --bind 127.0.0.1 --directory public
```

Use the pipeline number corresponding to the wheel version. Staging creates `public/` and
`.tmp/docs-pages/site.zip`. To test a subsequent release, save that zip, regenerate HTML for the new wheel, and pass
`--previous <saved-site.zip>` with its pipeline number. Use `--output <new-directory>` to keep outputs separate;
staging refuses to reuse an existing output directory. These commands prepare local files; CI performs the upload.

## Editing and generation

Edit narrative sources in `docs/`. Keep existing stable MyST targets when changing titles or links. The design
pages follow the [design-document authoring policy](design-documents.md).

Native reference text is rendered by `external/slangpy/docs/generate_api.py`, called with Falcor2's namespace
selection and `docs/api_order.json`. Ordinary Python APIs use Sphinx autodoc. Falcor2's adapter is
`docs/generate_api.py`; it explicitly selects public packages rather than traversing every implementation module.

Do not edit or commit `docs/generated/` or `docs/_build/`. API fragments regenerate on each build, and generation
errors stop the build even when older fragments exist. Update native comments or Python docstrings to improve
reference descriptions. Additional Sphinx output formats can be added separately; HTML is the validated output.
