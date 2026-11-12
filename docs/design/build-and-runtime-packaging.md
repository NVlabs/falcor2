---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "409c5040161fef448f9c9ce1bf4b687f9a579c77"
---

(falcor2-design-build-and-runtime-packaging)=
# Build and runtime packaging

Falcor2's build joins native libraries, Python extensions and runtime resources into a package selected for one
Python ABI and build configuration. Its bundled OpenUSD uses private native and Python identities so another USD
provider can coexist in the same process. Falcor and the USD wrappers share a privately named oneTBB scheduler
and allocator. CMake owns dependency integration and deployment; Python package initialization discovers the
selected runtime and plugin resources. Development builds keep configuration-specific binaries outside the source
package, while wheels install them together. Generated editor stubs describe that package without selecting which
runtime gets loaded.

## Purpose and scope

This page explains how the build produces a coherent runtime and how imports find it. Contributors can use it to
distinguish Python ABI selection, native loader isolation, deployment layout and editor metadata when changing a
dependency or diagnosing a package mismatch. CI qualification, publication, website generation and binding-object
ownership are separate concerns. Build commands remain in the contributor guidance.

## Architecture

### Select the Python ABI before dependency resolution

The root CMake configuration includes `falcor2_python_bootstrap.cmake` before `project()` invokes vcpkg. The
bootstrap selects the interpreter and obtains its `sys.implementation.cache_tag`, then preserves
`FALCOR_PYTHON_EXECUTABLE` and `FALCOR_PYTHON_ABI` in the dependency-build environment. `tools/build.py` and
`setup.py` supply their active interpreter explicitly; direct CMake configuration uses the same bootstrap.

The USD triplet settings include the ABI tag in vcpkg's package identity but leave the interpreter's absolute
path untracked. Compatible environments can therefore reuse the dependency binary while a different ABI selects
a different package. `falcor2_usd.cmake` resolves Python development files against the selected interpreter,
including when a compatible cached USD package recorded another environment's path.

The `falcor2_usd` interface target presents the imported `usd_m` monolith, `TBB::tbb` and `Python3::Python` to
Falcor consumers. The explicit TBB dependency matters because USD public headers call TBB directly even though
the monolith's own dependencies are private. `FALCOR_BUILD_PYTHON=OFF` omits Falcor's extension and wrapper
deployment; the current USD dependency still builds Python support and requires the selected interpreter.

### Keep USD identities private and TBB shared

The USD overlay builds OpenUSD 26.08 as a core-only shared monolith, with imaging, applications and unrelated
plugins disabled. Its CMake target is `usd_m`; its runtime basename is `falcor2_usd_ms`. Isolation spans several
namespaces:

| Boundary | Falcor-owned identity |
| --- | --- |
| C++ symbols | Versioned internal namespace, currently `falcor2UsdInternal_v0_26_8` |
| Native loading | Prefixed USD monolith and private TBB scheduler/allocator names |
| Python modules | `falcor2.pxr`, including registered native module and exception names |
| Plugin discovery | `FALCOR2_USD_PLUGIN_PATH` |

The overlay rewrites absolute Python imports and native registration names together. Moving a `pxr` directory
or renaming a library alone would leave other collision paths intact. A `falcor2.pxr.Usd.Stage` also has a
different native type from an external provider's Stage; the stage-import bridge accepts the private type.

The oneTBB overlay forces shared linkage. Falcor, the monolith and the Python wrappers link the same scheduler
instead of embedding independent schedulers in each wrapper. Current scheduler and allocator basenames are
`falcor2_tbb_2021_13` and `falcor2_tbbmalloc_2021_13`, with platform extensions and Debug suffixes. Allocator and
optional `tbbbind` runtime lookups use matching private names. Linux exports also use the
`FALCOR2_TBB_2021_13` symbol-version node to avoid unversioned global-symbol interposition. These private identities
do not change oneTBB's reported upstream interface version.

Core USD plugin code resides in the monolith, but its registry and resources remain external. Deployment must
retain `usd/plugInfo.json` and the resource tree; the absence of separate plugin libraries does not eliminate
plugin discovery.

### Deploy one matching configuration

`falcor2_usd_deployment` is the logical CMake deployment target. It stages the monolith, scheduler, allocator,
plugin resources and, when enabled, Python wrappers. `FALCOR_OUTPUT_DIRECTORY` includes the configuration for
multi-configuration builds. Debug uses the dependency's Debug wrapper tree; Release, RelWithDebInfo and
MinSizeRel use its Release tree. Both wrapper variants use the selected regular CPython ABI, not a debug Python
interpreter.

| Development location, relative to `FALCOR_OUTPUT_DIRECTORY` | Contents |
| --- | --- |
| Root | Native Falcor runtime and private USD/TBB runtime libraries |
| `usd/` | USD plugin registry and resources |
| `python/falcor2/` | Falcor extension and another matching USD/TBB runtime set |
| `python/falcor2/pxr/` | Configuration-matching private USD Python modules |

The wrapper-side runtime copies support the wrappers' relative loader paths. On Unix, deployment preserves the
TBB names required by their SONAMEs, including the major ABI suffix, rather than copying only the physical
minor-version filename. Wrapper search paths include `$ORIGIN/../..` on Linux and `@loader_path/../..` on macOS
for the final package layout.

Plugin synchronization replaces its controlled output resource directory when its stamp needs rebuilding, so
obsolete metadata does not survive a refreshed resource set. Native-only builds still stage and install the
USD/TBB runtime and plugin resources. CMake also installs collected vcpkg copyright notices, including notices
for transitive dependencies, independently of Python wrapper deployment.

### Separate runtime wrappers from editor metadata

The USD port generates Python documentation and `.pyi` stubs from Doxygen metadata plus runtime Boost.Python
reflection. The generator retains overload signatures, documentation, classes and properties, using `Any` when
type or signature information is unavailable. If Doxygen is unavailable for the host, documentation and stub
generation are skipped; this does not remove the runtime wrappers.

Release generates the documentation and stubs, which are copied to Debug because their API metadata is
configuration-independent. The vcpkg package caches these outputs. Deployment mirrors Release `.pyi` files into
the ignored source `falcor2/pxr` tree for editors that do not execute Python's dynamic package-path setup. Native
USD modules remain in the selected build tree. Falcor's own nanobind stubs use separate extension-generation
targets; neither stub set determines runtime configuration.

## Runtime flow

1. A build prepares dependency binaries for the selected Python ABI, then compiles and stages the chosen
   configuration. The extension build writes source-package `.build_dir` with that configuration's runtime
   location. Switching configurations changes this development selection without copying native USD wrappers
   into the source package.
2. `falcor2/__init__.py` imports SlangPy first. On Windows, a deployed `falcor2.dll` in the package directory takes
   precedence; otherwise `.build_dir` selects the development runtime and its DLL-search handle is retained.
   Missing both produces an import error. On non-Windows platforms, an existing marker also selects the
   development runtime.
3. When the selected runtime contains `python/falcor2`, initialization prepends it to `falcor2.__path__`.
   Initialization also prepends the runtime's `usd` directory to `FALCOR2_USD_PLUGIN_PATH`, preserving existing
   entries, before loading `falcor2.falcor2_ext` and Python extensions. Private wrapper imports then follow the
   selected package path and native dependency identities.
4. Wheel construction uses `setup.py` to build Release and install into the wheel's `falcor2` directory, with
   binary, library and data install roots set to that directory. The wheel therefore places private `pxr`,
   runtime libraries, plugin resources and shader/material resources under one package. General native CMake
   installs retain their configurable install directories. `_package_paths.py` locates runtime resources inside
   an installed package when `falcor2/slang` exists, or at the checkout root for a source package.

## Invariants and rationale

Python wrappers, the extension and their native dependencies must agree on ABI and configuration. A private
filename cannot compensate for a dependency still linked under a public identity, and a source stub cannot
repair a mismatched binary. Shared TBB linkage avoids duplicated scheduler state while private names isolate
Falcor's runtime from an unrelated host provider.

Dependency integration, deployment and import discovery have distinct owners: `falcor2_usd` defines what
consumers link, `falcor2_usd_deployment` assembles runtime files, and package initialization exposes their
locations. This separation keeps source imports and wheel imports on the same runtime contract without making
source-tree contents a mixture of Debug and Release wrappers. Native runtime files and plugin metadata remain
necessary when Falcor's Python extension is disabled.

## Source landmarks

- ABI and build entry points: `CMakeLists.txt`, `cmake/falcor2_python_bootstrap.cmake`,
  `cmake/vcpkg-triplets/`, `tools/build.py` and `setup.py`.
- Dependency integration and deployment: `cmake/falcor2_usd.cmake`, `cmake/falcor2_usd_deployment.cmake`,
  `src/falcor2/CMakeLists.txt` and `src/falcor2_ext/CMakeLists.txt`.
- Private dependencies and stub generation: `external/vcpkg_overlays/usd/portfile.cmake`,
  `external/vcpkg_overlays/usd/016-private-python-package.patch`,
  `external/vcpkg_overlays/usd/015-private-python-runtime-rpath.patch`,
  `external/vcpkg_overlays/usd/stubgen_usd.py`, `external/vcpkg_overlays/tbb/portfile.cmake` and
  `external/vcpkg_overlays/tbb/private-runtime-names.patch`.
- Import and resource discovery: `falcor2/__init__.py` and `falcor2/_package_paths.py`.
- Behavioral checks: `tests/python/importers/test_usd_isolation.py`, `tests/python/tools/test_build.py`,
  `tests/python/tools/test_usd_deployment.py`, `tests/python/tools/test_stubgen_usd.py` and
  `tests/python/render/test_usd_stage.py`. The deployment tests model Unix library names with text artifacts;
  they verify staging and install rules without proving native Unix loading. Coexistence tests require an
  external `pxr` installation, and documentation checks can skip when generated metadata is absent.

## Related design documents


- [Python and native integration](python-native-integration.md): package initialization and language boundaries.
- [Scene loading and authoring](scene-loading-and-authoring.md): private USD stages and importer ownership.
- [MaterialX integration](materialx-integration.md): material libraries and resource consumers.
