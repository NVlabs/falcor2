# MTLX integration

`mtlx` is Falcor2's short name for its MaterialX integration. It follows the
standard `.mtlx` document extension and the naming used by integrations such
as OpenUSD's `UsdMtlx`.

The checked-in dependency currently implements MaterialX 1.39. The dependency
version is intentionally documented here instead of being encoded in C++
namespaces, Slang modules, CMake targets, or runtime paths. A future MaterialX
upgrade should update this note and compatibility tests without renaming the
integration.

The implementation is organized as follows:

- `mtlx_material.*` contains the renderer-facing `MaterialXMaterial` class.
- `codegen/` loads MaterialX documents and emits Falcor-compatible Slang.
- `lut_globals.*` and `dielectric_lut_globals.*` provide shared lookup tables.
- Runtime resources are copied to `mtlx/standard`, `mtlx/libraries`, and
  `mtlx/snippets` below the configured Falcor output directory.
