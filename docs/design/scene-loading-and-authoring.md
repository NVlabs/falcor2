---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "ab87e5836e822a6dec45e0f4f9a2e70c9c67bc2f"
---

(falcor2-design-scene-loading-and-authoring)=
# Scene loading and authoring

Falcor2 scene loading converts files, USD stages or Python recipes into a live, device-backed `Scene` through
CPU-side `ImporterScene` data. An `Importer` records ordered edits and callbacks; format readers decode assets
and apply shared import options before realization. Importer geometry owns its storage while stream and
tessellator interfaces borrow views. Live realization creates materials, geometry, animation and placed objects,
converts UV conventions, and runs recipe callbacks after the imported objects exist. These boundaries separate
source interpretation and optional pose baking from GPU deformation, resource loading and scene updates.

## Purpose and scope

This page explains ownership and flow through decoding, asset resolution, geometry processing, PyScene,
realization and append. It records where source-format interpretation ends and the live scene begins.

[Scene model](scene-model.md) covers live object lifecycle. [Geometry and ray tracing](geometry-and-ray-tracing.md)
and [Geometry deformation](geometry-deformation.md) cover device storage and playback;
[MaterialX integration](materialx-integration.md) covers graph compilation after import. Detailed curve sampling
belongs to [Curve import and tessellation](curve-import-and-tessellation.md), and preview rendering belongs to
[Render pipeline](render-pipeline.md).

## Architecture

### Recipe, interchange data and live scene

`Importer` is a reusable recipe with default `ImportOptions`, source/search paths, ordered deferred edits and
loaded callbacks. Its camera, node, environment and material collections record operations. Camera and node
creation returns local selectors for subsequent edits; material selectors match exact names during the build.
Each `build_importer_scene()` creates fresh data, replays the edits and calculates bounds.

`ImporterScene` contains CPU-side materials, texture sources, assets, geometry, indexed nodes, cameras, lights,
skins, prototypes, animation channels and source metadata. It owns neither a device nor loaded callbacks.
`import_scene()` and format-specific importers return this representation. `Scene.from_importer_scene()` realizes
it; `Scene.from_importer()` additionally builds the recipe and runs its callbacks. `Scene.load()` selects the
path workflow. `Scene.from_stage()` imports an in-memory USD stage, and `append()` accepts a path, recipe or
interchange scene for an existing destination.

### Geometry storage and processing

`ImporterMesh` owns attribute descriptions, byte buffers and topology. `ImporterVertexBuffer` owns bytes and a
stride; `ImporterVertexStream<T>` borrows pointer, stride and count and need not be contiguous. Curves reuse these
mechanisms with separate position/radius sample arrays and static indices/UVs. Python arrays retain the owning
object, but storage-moving edits can still invalidate their views.

`TessellatorInputMesh` borrows topology and typed attribute spans until `tessellate()` returns. Producers supply
consistent, validated indices and cardinalities. Triangulation copies complete typed values; refined subdivision
also copies constant and uniform values exactly, including integers. During refinement, interpolated vertex,
varying and face-varying streams require float32. Positions are float3, and refined normals come from surface derivatives.
This neutral stream contract does not imply arbitrary integer USD primvar support.

Meshes, curves and lights carry processing properties initialized from `ImportOptions`. USD authored controls
can override refinement, normal recomputation and eligible light conversion. Refinement zero selects control
geometry; positive level N uses N+1 edge or cubic-span segments, with levels restricted to 0 through 256.
Polygon meshes and linear curves retain their ordinary path.

Shared option processing runs at format-import entry points, including direct glTF and USD file/stage calls.
It applies optional pose baking, area-light conversion, then requested normal/tangent recomputation.
Format-specific repair is separate: glTF repair replaces invalid authored normals and their associated tangent
frames while preserving unaffected values, including local vertex splits when needed. Missing-attribute
generation is distinct from this repair.

### Pose baking and sampled motion

Skin joints, inverse-bind transforms, node associations and influence streams remain in importer data until
consumed. Optional `ImporterScene.bake_pose()` evaluates mesh-local joint palettes, clones multiply referenced
geometry before per-instance baking, deforms positions and frames, and removes influence streams and skin associations.
Stored node placement and ordinary transform animation remain intact. Explicit time samples imported transform
channels; omitted time uses stored transforms. Automatic glTF baking uses the stored pose, while USD requests
zero seconds. Baking is disabled by default and does not install runtime skeletal deformation.

USD skin retention is enabled by `usd_import_skinning` or baking. Its supported path is linear-blend skinning
on control cages; combined skinning and mesh motion, blend shapes, dual-quaternion skinning and instanced USD
skinning are unsupported. glTF imports its first animation's transform channels and skips morph-weight targets.

Sampled geometry motion is separate. Multiple motion keys select dynamic mesh or curve geometry at realization;
vertex-cache factories install packed samples and matching topology. Transform channels instead become
`AnimationChannel` objects referenced by entity `TransformAnimation` components. Cache timing and history are
described in [Geometry deformation](geometry-deformation.md).

### Asset resolution and texture ownership

The dispatcher selects TinyGLTF for `.gltf`/`.glb` and OpenUSD for `.usd`/`.usda`/`.usdc`/`.usdz`. External glTF
images stay path-backed, resolved against the source directory; embedded images retain encoded bytes and
sampler metadata. USDZ package members are registered in `ImporterScene.assets` and read through OpenUSD without
unpacking the package.

Live texture resolution is consumer-specific. Native and MDL material routes can retain owned encoded `Blob`
data or filesystem paths for deferred loading. The MaterialX glTF route caches embedded images as
content-addressed files because its loader requires filenames. Environment-map realization can decode retained
bytes directly.

### Material translation

Material decoding records source properties and target selection; live realization performs
conversion. The historical glTF mode name `standard` selects dedicated native glTF materials; `standard_material`
is the approximate compatibility route to the `StandardMaterial`/`StandardSpecGlossMaterial` family.

USD MaterialX translation handles supported external XML references and already expanded UsdShade networks.
External references become small source-asset wrappers retaining XML material selection and source-layer
anchoring; they do not expose XML internals as USD prims. Expanded networks become MaterialX documents with
path-based node identity, concrete types and selected output ports preserved across nodegraph boundaries.
Conversion relies on available node definitions and diagnoses unknown multi-output ports. General XML graph
expansion and arbitrary USD overrides inside referenced graphs are not implemented.

### USD stage and hierarchy ownership

USD file loading targets the stage's session layer. Importing a caller-owned stage creates a private stage with
an anonymous session layer over the source session, retaining resolver context, population mask, muted layers,
load rules and interpolation. Generated MaterialX wrappers therefore leave the caller's layers unchanged.
Units and up-axis become an importer root transform. USD defaults to lower-left UVs, with the root-layer
`customLayerData` key `falcor:uvOriginOverride` accepting `upperLeft` or `lowerLeft`.

Traversal owns hierarchy entry and exit for transformable prims, native instances and Scope groups. Payload
adapters attach geometry, lights, cameras or prototype references to the node supplied by traversal. Pivot
transforms may create helper chains; payload and children use the final node. Reset transform stacks attach
under the import root. Point instancers create generated children. Material, shader and geometry-subset prims
do not become ordinary hierarchy nodes. Hidden payloads and pruned children preserve traversal's stack ownership.

### PyScene and path resolution

The `falcor2.pyscene` facade forwards operations to thread-local `Importer.get()`. Loading a `.py` file scopes a
fresh importer and executes fresh globals as `__falcor2_scene__`, so its `__main__` preview guard is skipped.
The scene directory and available source-checkout package root temporarily enter `sys.path`. Scopes restore the
previous importer and Python paths after success or failure. Search paths are also installed during building
and callbacks so Python material constructors can import helpers.

Explicit asset/environment edits resolve during the build: source directory first, additional roots in recorded
order, then working-directory fallback. Relative additional roots are anchored to the source directory.
Arbitrary path-looking values in `Properties` are not rewritten. Authoring can record cameras, nodes, environment
settings, exact-name material replacements and loaded callbacks; direct-script preview consumes that recipe.

## Runtime flow

1. Normalize a source path. Record an asset-import edit for a format file, or execute a Python scene to collect
   edits and callbacks. In-memory USD stages enter through stage import instead.
2. Replay recipe edits in order. Each asset edit resolves its source, decodes it and applies import options before
   CPU merge. Merge offsets geometry, material, skin, node, prototype and animation references; later edits can
   select earlier imported data. Recalculate bounds when the recipe is complete.
3. Realize materials before geometry. Material assignment uses the imported material index, so duplicate names
   do not collapse bindings. Convert each mesh/curve's own UV convention to the destination configuration and
   install static geometry or a vertex cache according to its motion keys.
4. Create transform animation before recursively creating entities and components, including prototype instances.
   Initial camera selection adopts the first authored camera while selection is pending. Explicit choices,
   including the built-in free camera, survive later imports; the free camera supplies the fallback.
5. Run recipe callbacks in registration order after realization. Bare `ImporterScene` construction has no
   callbacks. Append follows the same flow while preserving the destination's device, configuration and objects.

## Invariants and rationale

Callbacks belong to the recipe because they require live objects. Building a recipe manually and then passing
only its `ImporterScene` intentionally drops callback execution. Selector references must target earlier creation
edits; material replacement remains zero-to-many exact-name matching with no match allowed. Rebuilding replays
decoding: the edit stream has no whole-asset result cache, although import algorithms can use a blob cache.

Borrowed streams and tessellator inputs must not outlive or move their storage. Keeping a Python owner alive does
not preserve a particular buffer allocation across resizing or topology changes. Fixed-topology motion requires
matching vertex correspondence across samples; neither optional pose baking nor richer CPU influence data
implies runtime skinning.

Append is additive and not transactional. CPU merge remaps many indexed arrays but does not rewrite texture
indices embedded in material properties. Independent assets therefore lack complete texture-reference isolation.
Failure during realization or a callback can leave earlier changes in the destination. Exact-name replacement
may intentionally affect multiple materials, even though geometry assignment itself uses indices.

The destination keeps one UV convention. New construction adopts the interchange scene's default unless an
explicit configuration is supplied; each geometry retains its own source convention until realization. Append
converts new geometry without changing the destination. Restoring `sys.path` also does not isolate CPython's
process-wide helper-module cache: same-named modules in separate scene directories can interact.

## Source landmarks

- `src/falcor2/importers/importer.cpp`, `src/falcor2/importers/importer_edits.cpp` and
  `src/falcor2/importers/importer_types.cpp` - dispatch, options, ordered edits, merge and pose baking.
- `src/falcor2/importers/importer_types.h` and `src/falcor2/importers/mesh_tessellator/mesh_tessellator.h` -
  owned storage, borrowed streams and typed tessellation contracts.
- `src/falcor2/importers/gltf_importer/gltf_importer.cpp` - glTF decoding, skins, images and selective repair.
- `src/falcor2/importers/usd_importer/usd_importer.cpp`,
  `src/falcor2/importers/usd_importer/usd_importer_context.cpp`,
  `src/falcor2/importers/usd_importer/usd_importer_materialx.cpp` and
  `src/falcor2/importers/usd_importer/usdshade_to_mtlx.cpp` - stage isolation, hierarchy and graph translation.
- `src/falcor2/render/scene_import.cpp`, `src/falcor2/importers/mesh_import.cpp` and
  `src/falcor2/render/scene.cpp` - realization, texture sources, UV conversion and camera selection.
- `src/falcor2/core/asset_resolver.cpp`, `src/falcor2/core/python_interpreter.cpp` and
  `falcor2/pyscene/authoring.py` - path ownership and scoped recipe execution.
- `src/falcor2_ext/importers/importer_types.cpp` and `src/falcor2_ext/render/scene.cpp` - Python views and transitions.
- `tests/native/importers/test_importer_edit_stream.cpp`, `tests/native/importers/test_importer_vertex_stream.cpp`,
  `tests/native/importers/test_mesh_tessellator.cpp` and `tests/native/importers/test_timed_skin_baking.cpp` -
  deferred edits, storage, typed interpolation and pose semantics.
- `tests/native/usd_importer/test_usd_importer.cpp`, `tests/native/render/test_scene.cpp` and
  `tests/python/pyscene/test_importer_edit_stream.py` - hierarchy, stage conversion, append and restoration.
- `tests/python/importers/test_gltf_normal_repair.py`, `tests/python/render/test_gltf_texture_sources.py` and
  `tests/python/render/test_gltf_materialx_embedded.py` - repair and consumer-specific image ownership.
- `data/scenes/cornell-box-env.py` and `data/scenes/checker-material.py` - asset, camera, environment and material
  authoring with direct-script preview.

## Related design documents

- [Scene model](scene-model.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Geometry deformation](geometry-deformation.md)
- [Curve import and tessellation](curve-import-and-tessellation.md)
- [USD curve attribute counts and interpolation](usd-curve-attributes.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Render pipeline](render-pipeline.md)
