---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "ac4dd82eaf561d495127c1f002d553f887b142a6"
---

(falcor2-design-documentation-generation)=
# Documentation generation and publication

Falcor2's documentation pipeline combines maintained narrative pages with API references generated from real
development bindings. Native declaration extraction supplies Python help; SlangPy's shared generator renders
selected native APIs, while Sphinx autodoc handles ordinary Python objects. Sphinx assembles these inputs into
static HTML. The wheel pipeline supplies a release label, and a serialized GitLab Pages job preserves complete
version directories in a cumulative archive. These ownership boundaries explain where a documentation change
belongs, why native builds must precede reference generation, and why successful local HTML generation does not
establish successful publication or durable retention.

## Purpose and scope

This page explains how source descriptions and runtime APIs become a versioned documentation site. Contributors
can use it to trace missing or stale reference content and understand publication ordering and recovery. Commands,
environment setup and hosting configuration belong in [Building documentation](../contributing/building-documentation.md).
This pipeline documents Python APIs, including native bindings; it does not extract a standalone C++ or Slang API
reference. HTML is the validated delivery format.

## Architecture

### Source help and reference rendering have different owners

Native API help reaches the site through the compiled extension. The `falcor2_pydoc` CMake target runs
`pybind11_mkdoc` over supported C++ headers and writes `src/falcor2_ext/py_doc.h`. Binding macros such as `D(...)`
and `DEF_*` attach those descriptions to native Python objects. Reflected properties use their existing
description-forwarding path. Python definitions supply their own docstrings.

The website build reads the resulting runtime help. It does not regenerate `py_doc.h`, compile the extension,
or repair stale binaries. Changes to extracted native descriptions therefore require extraction and a Release
rebuild before website generation. Reflection descriptions require the corresponding binding rebuild, without
the extraction step. The [binding documentation rules](../contributing/coding-style.md#binding-documentation)
govern these choices and their validation; an exported member can legitimately have no descriptive help.

`docs/generate_api.py` owns Falcor2's API selection and adaptation. It loads the shared renderer from the pinned
`external/slangpy/docs/generate_api.py` and supplies namespace filters, ordering, output location and formatting
hooks. Shared native signature, overload, property and enum rendering stays in SlangPy. Ordinary Python classes
and functions use Sphinx autodoc, avoiding the native renderer's assumptions about nanobind objects.

### Explicit exports define reference coverage

The native reference selects Falcor-owned members from named public namespaces. It does not use Python `__all__`
as its complete native catalog, because the extension injects additional symbols. Python selection examines
explicit public packages, honors their `__all__` when present, and otherwise considers nonprivate exports.
Ownership checks exclude imported dependency objects. Neither path walks every implementation module.

Python class and function aliases link to the first selected public entry for the same object. Root-package
exports take precedence through package ordering; constants are not deduplicated by object identity. Autodoc
includes public and undocumented class members, constructors and `__call__` where present. Selection policy
belongs in the Falcor adapter; rendering improvements shared by native consumers belong in SlangPy.

Sphinx owns Markdown/reStructuredText parsing, cross-references, navigation, search and HTML assembly. The
configuration uses MyST for Markdown and Furo for presentation. Small adapters normalize native docstrings and
prevent builtin type references from resolving to same-named enum members; they do not suppress warnings globally.

## Runtime flow

### From development bindings to static HTML

1. Sphinx loads `docs/conf.py`, which puts this checkout on the Python import path. Its `builder-inited` hook
   invokes API generation before Sphinx reads source pages.
2. The adapter imports Falcor2 and its selected namespaces, requires the repository's exact `falcor2/__init__.py`
   path, and checks the shared generator's interface. A working, current Release development build is a
   prerequisite. The path check identifies the source package; it does not verify native binary freshness.
3. The adapter prepares native reference text and Python autodoc directives, then writes
   `docs/generated/native-api.rst` and `python-api.rst` when their contents change. The native renderer first
   writes to a temporary destination. Both fragments are prepared before updating the final files.
4. The maintained `docs/api/native.rst` and `python.rst` wrappers include those fragments. Sphinx reads them
   alongside narrative pages and writes `docs/_build/html`. Generated fragments are excluded from independent
   page discovery. The complete HTML tree, including assets and search data, is the static delivery artifact.

The documentation build imports real objects without executing examples or creating rendering devices.
`docs/generated/` and `docs/_build/` are disposable outputs, not maintained source. Native declaration extraction
has its own generated artifact and lifecycle.

### Wheel labels and versioned publication

Local generation uses Falcor2's runtime version unless Sphinx receives an explicit release label. The wheel job
uses `tools/docs.py build` to read the full distribution version from exactly one Falcor2 wheel, preserving
suffixes such as `+rtr.123`. That metadata labels the site; API generation still imports the source development
bindings. The helper clears the previous version marker before invoking Sphinx and writes `docs/_build/version.txt`
only after Sphinx succeeds.

CI generates documentation once in the Linux x86_64 Python 3.12 wheel job. Generation can also occur in a web
wheel build on another branch. Publication has narrower rules: the `pages` job follows successful wheel
publication for eligible protected default-branch web or scheduled pipelines. It consumes that Python 3.12 job's
HTML and version marker.


The Pages job serializes through the `falcor2-docs-pages` resource group. It retrieves `site.zip` from generic
package `falcor2-docs-history`, version `1`, using `CI_JOB_TOKEN`. HTTP 404 initializes history; other download
failures stop the job. `tools/docs.py stage` restores prior versions into a temporary tree, adds the current
complete HTML directory, rebuilds the root index and `latest/`, and writes a new cumulative archive.

`versions.json` maps release strings to pipeline IIDs and identifies `latest`. The greatest numeric IID wins,
so an older pipeline finishing late adds history without moving `latest` backwards. Semantic version ordering
does not select the latest directory. A version cannot move to another IID, and an IID cannot acquire a second
version. Retrying the same pair replaces that entire version directory, removing pages absent from the retry.

The job uploads cumulative history before handing `public/` to Pages. If upload succeeds but deployment fails,
a retry restores the saved history and replaces its own version. Download and upload use bounded curl retries;
the resource-group lock prevents concurrent jobs from replacing one another's cumulative history.

## Invariants and rationale

### Generation failures remain failures

Missing bindings, an incompatible shared generator or required API-generation errors abort the documentation
build. The adapter also detects the shared renderer's property-processing failure diagnostic and treats it as
fatal. Old fragments may remain on disk after failure, but they are not an accepted fallback.

Ordinary Sphinx builds and the wheel documentation helper report warnings without treating them as errors.
Actual build errors still fail the producer job. Separate validation can use `-W --keep-going` to make warnings
fatal. Successful non-strict generation therefore does not imply warning-free documentation.

Stable ordering and write-if-changed behavior keep unchanged fragments and their modification times stable.
The adapter replaces process-local object addresses only in native signature defaults, preserving real
hexadecimal values and prose. This narrow reproducibility contract concerns reference fragments; it does not
promise identical HTML or ZIP bytes across runtimes and toolchain versions.

### Retention depends on the history store

Staging refuses an existing output directory and validates version names, manifest identities and nonempty
version indexes. Restored archive entries cannot use traversal paths, absolute paths, backslashes, drive-style
colons or symlinks. The stage helper prepares local files; CI performs publication.

Each deployment carries complete history because Pages replaces the previous site. Expiring job artifacts are
not the durable history store. The registry must permit the duplicate history assets used by publication and
retain its newest complete archive. Losing the last archive makes a later HTTP 404 indistinguishable from the
first release. There is no automatic version pruning. Recovery and retention settings are documented in the
build guide.

Source and local tests establish generation and assembly behavior. Live Pages availability, access settings,
registry configuration and retention require external verification; they cannot be inferred from a staged site.

## Source landmarks

- Native help: [extraction target](../../src/falcor2_ext/CMakeLists.txt),
  [binding macros](../../src/falcor2_ext/nanobind.h) and
  [reflection forwarding](../../src/falcor2_ext/nanobind_reflector.h).
- Reference selection and rendering: [Falcor adapter](../generate_api.py), [ordering](../api_order.json),
  [shared native generator](../../external/slangpy/docs/generate_api.py) and [API wrappers](../api/index.md).
- Website assembly: [Sphinx configuration](../conf.py) and [documentation dependencies](../requirements.txt).
- Versioned delivery: [build/stage helper](../../tools/docs.py).

- Regression evidence: [API-generation tests](../../tests/python/tools/test_docs_generation.py) and
  [history/publication-contract tests](../../tests/python/tools/test_docs.py) cover selection, failures,
  repeatability, archive safety, retries and ordering. They do not prove live deployment.

## Related design documents

- [Python and native integration](python-native-integration.md): runtime imports, bindings and reflection.
- [Build and runtime packaging](build-and-runtime-packaging.md): native artifact production and discovery.
