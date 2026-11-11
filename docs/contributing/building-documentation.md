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


## Editing and generation

Edit narrative sources in `docs/`. Keep existing stable MyST targets when changing titles or links. The design
pages follow the [design-document authoring policy](design-documents.md).

Native reference text is rendered by `external/slangpy/docs/generate_api.py`, called with Falcor2's namespace
selection and `docs/api_order.json`. Ordinary Python APIs use Sphinx autodoc. Falcor2's adapter is
`docs/generate_api.py`; it explicitly selects public packages rather than traversing every implementation module.

Do not edit or commit `docs/generated/` or `docs/_build/`. API fragments regenerate on each build, and generation
errors stop the build even when older fragments exist. Update native comments or Python docstrings to improve
reference descriptions. Additional Sphinx output formats can be added separately; HTML is the validated output.
