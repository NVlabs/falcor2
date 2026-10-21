---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-15
verified_against: "d4d9693f4cfcc31a65fc6504183f2dcb6630670b"
---

(falcor2-design-scene-loading-and-authoring)=
# Scene loading and authoring

Falcor2 scene loading is a staged conversion from files or Python authoring into a live, device-backed `Scene`.
Format readers decode glTF, USD, and USDZ into an `ImporterScene`, while an `Importer` records ordered edits and
post-load callbacks for PyScene and programmatic recipes. The live-scene boundary then creates materials, geometry,
entities, components, and animation from that CPU-side data. Explicit `load`, `from_importer`,
`from_importer_scene`, and `append` transitions keep those representations distinct, allow relative assets to be
resolved before GPU resources exist, and ensure callbacks run only after imported objects are available.

## Purpose and scope

This page explains ownership and flow from a scene path or PyScene recipe through format decoding, asset resolution,
`ImporterScene`, live-scene realization, append, and loaded callbacks. It also defines which operations are deferred
and how glTF, USD, and packaged USDZ assets reach the live scene.

The live [scene object model](scene-model.md), renderer composition in [Render pipeline](render-pipeline.md), and the
internal geometry, material, and lighting representations belong to their focused design pages. PyScene preview is
mentioned only where its direct-script behavior affects loading; preview rendering policy belongs to the render
pipeline rather than the importer contract.

## Architecture

### Three scene representations

`Importer` is a reusable recipe. It owns default `ImportOptions`, source and search paths, an ordered list of deferred
edits, and callbacks that require a live scene. Its camera, node, environment, and material collections are recording
interfaces rather than object stores. Camera and node creation returns importer-local selectors that later edits can
reference; exact-name material selectors resolve only when the edit stream is built.

`ImporterScene` is mutable CPU-side interchange data. It contains materials, encoded or path-backed textures and
assets, meshes, curves, indexed nodes, cameras, lights, prototypes, animation channels, bounds, and UV-origin
metadata. It owns neither an SGL device nor loaded callbacks. `build_importer_scene()` creates a fresh instance,
applies every recorded edit in order, and calculates its bounds.

`Scene` is the live, device-backed destination. `Scene.from_importer_scene()` realizes data only;
`Scene.from_importer()` first builds the recipe and then runs its callbacks. `Scene.load()` selects the complete file
workflow, while the `append()` overloads apply the corresponding data or recipe to an existing live scene. The
generic `import_scene()` function stops at `ImporterScene`, and format-specific `GltfImporter.load_scene()` and
`UsdImporter.load_scene()` expose their decoded data directly.

### Decoding and asset ownership

The generic dispatcher recognizes `.gltf` and `.glb` through TinyGLTF, and `.usd`, `.usda`, `.usdc`, and `.usdz`
through OpenUSD. It also owns import-option post-processing after format decoding: selected finite area lights can be
replaced by emissive meshes, and requested mesh normals and tangents can be recomputed. Direct format-specific
loader calls do not apply those generic passes.

TinyGLTF reads JSON or binary containers, resolves external image paths against the source file's directory, and
retains embedded image bytes in texture records. The USD reader opens an absolute path under an OpenUSD resolver
context, normalizes stage units and up-axis through a root transform, and records USD's lower-left UV convention.
For USDZ, package-relative texture and dome-light paths are registered once in `ImporterScene.assets`; OpenUSD
supplies the encoded member bytes without extracting the package. Live-scene realization decodes those bytes and
otherwise passes filesystem paths to the material or environment-light loading path.

An `Importer` resolves edit paths only while building. Its source file's directory has first priority, followed by
additional roots in recorded order; relative additional roots are interpreted from the source directory. The
process working directory is the final fallback. This resolver is used by explicit asset-import and environment-map
edits. It does not reinterpret arbitrary path-looking values inside `Properties`.

### USD hierarchy ownership

USD traversal owns node creation. Each visited transformable prim, native instance, or `UsdGeomScope` receives an
`ImporterNode`; a non-transformable instance or Scope uses an identity local transform. Scope groups therefore
preserve authored parentage and inherited transforms: a light at `/Lights/Key` has a `/Lights` parent node even when
that group has no renderable content.

Mesh, curve, light, camera, and native-instance import paths receive the node index from traversal and attach their
payload or prototype reference directly to that node. They do not create an extra node merely to hold the payload.
Point instancers create generated instance children beneath their own prim node. Material, shader, and geometry-subset
prims are handled separately and do not become ordinary hierarchy nodes through this rule.

Traversal uses the same node-ownership condition on entry and exit, including when a prim's payload is hidden or
its children are pruned. Keeping stack ownership in traversal prevents payload-specific early returns from changing
the parent of later siblings. Environment-map pole corrections likewise stay outside the hierarchy, as described in
[Lighting and sampling](lighting-and-sampling.md#environment-map-orientation).

### PyScene authoring

The `falcor2.pyscene` facade forwards each operation to the thread-local `Importer.get()`. A `.py` path passed to
`Scene.load()` or `append()` installs a fresh current importer, records the absolute source path, and executes the
file in fresh globals under the synthetic name `__falcor2_scene__`. The scene directory, plus the source checkout's
package root when available, is temporarily prepended to `sys.path`. Scoped objects restore both the previous
importer and the original Python search path after success or failure.

PyScene code can import assets, add search roots, create cameras and nodes, set an environment, replace all materials
with an exact name, and register `on_scene_loaded` callbacks. A guarded `if __name__ == "__main__"` block therefore
does not run during `Scene.load()`, but can call `pyscene.preview()` when the file is executed directly. Python helper
imports and Python-backed material constructors remain usable while the importer is built and callbacks run.

## Runtime flow

```text
 format path ----> Importer(import_asset edit) --+
                                                   |
 Python path ---> scoped execution ---> edits -----+---> build fresh ImporterScene
                                      callbacks    |       |
                                                   |       +-- resolve edit paths
                                                   |       +-- decode and post-process assets
                                                   |       +-- merge and remap arrays
                                                   |       +-- calculate bounds
                                                   v
                                      realize into new or existing Scene
                                                   |
                                                   +-- run Importer callbacks
```

1. `Scene.load()` converts the requested path to an absolute lexical path. A non-Python path becomes one deferred
   asset-import edit. A Python path executes first and may record any number of edits and callbacks.
2. Building the importer creates one resolver and applies edits in registration order. Each asset edit resolves its
   path, calls the generic format dispatcher, and appends the decoded arrays while remapping node, geometry, light,
   camera, prototype, and animation indices. Later edits can therefore select or modify data imported earlier.
3. Generic format post-processing runs for each imported asset before its data is merged. The completed importer
   scene then recalculates mesh and hierarchy bounds.
4. Live-scene realization creates materials and textures before geometry, then creates animation and recursively
   translates importer nodes into entities and components. The first imported camera becomes active only when the
   destination has no active camera. Mesh UVs are converted from their authored origin to the destination
   `SceneConfig` as they are realized.
5. Construction from an `Importer`, or append from one, runs callbacks in registration order after all its imported
   objects have been added. A callback can inspect and mutate the live scene. Construction or append from a bare
   `ImporterScene` has no callbacks to run.
6. Path-based `append()` uses the same normalization, execution, build, and callback path, but retains the target
   scene's existing objects, active-camera choice when present, device, and immutable configuration.

## Invariants and rationale

The stage names communicate ownership. `ImporterScene` data can be inspected or transformed without a GPU, while
callbacks stay on `Importer` because they require live scene objects. Converting a recipe to `ImporterScene` manually
and then calling `from_importer_scene()` intentionally drops callback behavior; use `from_importer()` when callbacks
are part of the recipe.

Edit ordering is observable. Selectors for newly created cameras, nodes, or parents must refer to earlier creation
edits. Material replacement is zero-to-many exact-name matching and no match is a valid no-op. Rebuilding an importer
replays decoding because the current edit stream has no asset-result cache.

Live append is additive but not transactional. Within a recipe that imports multiple assets, most indexed importer
arrays are offset during CPU-side merge, but texture indices embedded in material properties are not remapped, and
imported material references are resolved by name during live realization. The current implementation therefore
provides no complete isolation for multiple assets or same-named materials. If realization or a loaded callback
throws, prior changes to the destination may remain. Callers must not assume rollback or richer multi-asset conflict
semantics.

Source-relative resolution is deliberately limited to explicit path-bearing edits. Temporary Python search paths
are restored, but imported helper modules still participate in CPython's process-wide module cache. Scene authors
should not assume that same-named helper modules in different scene directories are isolated.

The destination scene owns its UV convention. New construction inherits `ImporterScene.uv_origin` unless the caller
supplies a `SceneConfig`; append never changes an existing scene's configuration and converts each mesh as needed.
This keeps one stable shader-facing convention even when assets from different formats are combined.

## Source landmarks

- `src/falcor2/importers/importer.h`, `src/falcor2/importers/importer.cpp`, and
  `src/falcor2/importers/importer_edits.cpp` - recipe ownership, ordered edits, path resolution, generic dispatch,
  merge remapping, and import-option passes.
- `src/falcor2/importers/importer_types.h` - CPU-side `ImporterScene` records, indices, bounds, and UV metadata.
- `src/falcor2/importers/gltf_importer/gltf_importer.cpp` - TinyGLTF decoding and external or embedded image data.
- `src/falcor2/importers/usd_importer/usd_importer.cpp`,
  `src/falcor2/importers/usd_importer/usd_importer_context.cpp`, and
  `src/falcor2/importers/usd_importer/usd_importer_utils.cpp` - OpenUSD stage decoding, hierarchy ownership, resolver
  context, and USDZ package-member retention.
- `src/falcor2/importers/importer_area_light_geometry.cpp` - generic conversion of selected imported area lights.
- `src/falcor2/render/scene_import.cpp` - path normalization, PyScene execution, live realization, append, callbacks,
  texture-byte resolution, and UV conversion.
- `src/falcor2/core/asset_resolver.cpp` and `src/falcor2/core/python_interpreter.cpp` - ordered asset roots and scoped
  Python globals and search paths.
- `src/falcor2_ext/importers/importer.cpp`, `src/falcor2_ext/importers/importer_types.cpp`, and
  `src/falcor2_ext/render/scene.cpp` - Python exposure of recipes, intermediate data, and explicit scene transitions.
- `falcor2/pyscene/authoring.py` and `falcor2/pyscene/preview.py` - concise authoring facade and direct-script flow.
- `tests/native/importers/test_importer_edit_stream.cpp`, `tests/native/render/test_scene.cpp`,
  `tests/python/pyscene/test_importer_edit_stream.py`, `tests/python/importers/test_usdimporter.py`, and
  `tests/python/core/test_asset_resolver.py` - deferred behavior, callback timing, source-relative loading, USDZ
  bytes, resolution priority, append, and failure restoration.
- `tests/native/usd_importer/test_usd_importer.cpp` - direct payload attachment, Scope grouping and transform
  inheritance, native instances, and traversal around hidden prims.
- `data/scenes/cornell-box-env.py` and `data/scenes/checker-material.py` - representative asset, camera, environment,
  material-replacement, helper-import, and preview authoring.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Render pipeline](render-pipeline.md)
- [Falcor2 design documentation](index.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [Lighting and sampling](lighting-and-sampling.md)
