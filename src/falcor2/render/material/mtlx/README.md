# MTLX integration

`mtlx` is Falcor2's short name for its MaterialX integration. It follows the
standard `.mtlx` document extension and the naming used by integrations such
as OpenUSD's `UsdMtlx`.

The checked-in dependency currently implements MaterialX 1.39. The dependency
version is intentionally documented here instead of being encoded in C++
namespaces, Slang modules, CMake targets, or runtime paths. A future MaterialX
upgrade should update this note and compatibility tests without renaming the
integration.

## Default-first development and testing

Read the [MaterialX default-backend and validation policy](../../../../../docs/design/materialx-integration.md#default-backend-and-validation-policy)
before changing this integration. The primary consumer is the build's default backend. Make substantial work
function on the default first, then verify the other supported backends. Primary tests and examples leave the
layering mode at its default; explicit non-default modes are
compatibility checks. Keep codegen-only tests CPU-only, and avoid duplicating an entire feature matrix across modes.

The implementation is organized as follows:

- `mtlx_material.*` contains the renderer-facing `MaterialXMaterial` class.
- `codegen/` loads MaterialX documents and emits Falcor-compatible Slang.
- `lut_globals.*` and `dielectric_lut_globals.*` provide shared lookup tables.
- Runtime resources are copied to `mtlx/standard`, `mtlx/libraries`, and
  `mtlx/snippets` below the configured Falcor output directory.
