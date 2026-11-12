---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "e4bb4b384a6a6888211f4f2593404ec05a884139"
---

(falcor2-design-geometry-and-ray-tracing)=
# Geometry and ray tracing

Falcor2 separates scene-owned geometry and placement from the device representation used for ray traversal.
Static geometry supplies packed vertices; dynamic geometry owns fixed topology and a deformer that produces
current and previous vertices on the GPU. `RenderScene` owns pooled storage, a shared buffer-pointer table,
instance descriptors, transforms, and acceleration structures. Its update orders uploads and deformation before
bounds and traversal structures. Typed additional attributes remain separate from the hot packed vertices.
Renderers consume this synchronized representation through scene binding and a common hit-group policy, while
hit reconstruction distinguishes immediate native curve hits from reusable hit records.

## Purpose and scope

This page explains how live geometry, device storage, updates and hit reconstruction fit together. It defines
ownership and indexing boundaries that must remain consistent when changing geometry or a renderer. It covers
the deformation boundary; sample acquisition, clip timing and deformer algorithms are separate concerns.

[Scene loading and authoring](scene-loading-and-authoring.md) covers decoding and realization;
[Scene model](scene-model.md) covers object lifecycle. [Geometry deformation](geometry-deformation.md) explains
sample selection, update scheduling and current/previous vertex history. [Buffer pointers](buffer-pointers.md)
defines indirect access and resource lifetime. Material evaluation and analytic-light transport belong to their
respective pages.

## Architecture

### Geometry, deformation and placement

`Scene` owns static and dynamic mesh and curve geometry. Meshes contain sub-meshes of packed position, normal,
tangent, handedness and UV vertices with triangle indices. Curves use packed center/radius/UV vertices and
successive segment indices for linear swept spheres (LSS). Static authoring copies caller data into owned
storage. Mesh extraction preserves first-reference order and remaps additional attributes consistently; curve
list conversion duplicates or reorders endpoints and attributes when segment pairs are not adjacent.

`DynamicMeshGeometry` and `DynamicCurveGeometry` own prepared topology, static additional attributes and one
typed `GeometryDeformer`. Geometry owns the output ranges; the deformer chooses input samples and history work.
Attached deformers register with shared typed updaters owned by the corresponding render-geometry collection.
Updaters prepare participants and batch GPU work, while geometry supplies current/previous output pointers.
Replacing topology and deformer together recreates render state and resets history.

A `GeometryInstance` component combines geometry with an entity transform and per-part materials. A mesh creates
one low-level triangle entry per sub-mesh; a curve creates one LSS entry. Entries form a homogeneous
`RenderGeometryGroup`, shared by its placed instances. The public `GeometryGroup` class remains a placeholder;
`RenderGeometryGroup` is the working internal collection. Missing material slots use material ID zero.

### Pooled device storage

`RenderScene` maintains typed handle pools for geometries, groups, instances and transforms. Packed vertices
and indices occupy ranges in lazy `ManagedBufferPool` storage, rather than one device buffer per geometry.
Static and dynamic vertices use separate pools. Allocation selects a tuple width: static vertices need one
buffer; dynamic vertices need two synchronized buffers with the same offset for current and previous data.
Index storage is shared within the geometry type. Procedural LSS bounds have synchronized primitive ranges.

All geometry pools publish into one scene-wide `BufferHandleTable`. Despite its name, the table stores
`BufferPtr` values. A geometry descriptor carries table indices and element offsets; shader loaders resolve a
pointer, form the appropriate typed view and index the allocation. Tuple members occupy consecutive table
entries. Reallocation refreshes the table before table-based traversal and shading readers use it. Pointer
values do not own resources or establish barriers; the pools and update pass retain those responsibilities.

The pools use CPU-backed `ManagedVector` storage. Ordinary static updates upload dirty ranges. Dynamic output
pools suppress ordinary CPU uploads and are written by deformation kernels. Dynamic geometry creation,
destruction or data replacement can request collection-wide deformation initialization: the implementation
recomputes output and resets history rather than preserving GPU output across pool growth. The independent
`DeviceVector` utility supports content-preserving growth, but that behavior is not the geometry-pool contract.

### Typed additional attributes

Non-owning `GeometryVertexAttributeDesc` inputs are validated and copied into owned, tightly packed streams.
Scene attribute IDs occupy a 32-bit mask: UV0-7 use IDs 0-7, color zero uses 8, and named properties receive
monotonic scene-lifetime IDs 16-31. Material readers may reserve a name before geometry establishes its type and
width. Subsequent sources for the same named property must agree; presence remains per geometry.

The shader ABI supports Boolean, signed 32-bit integer, half and float values with one to four components.
Boolean values are normalized to zero-or-one bytes. Unsupported source types can remain in CPU geometry but
are omitted from shader lowering with a warning. Ordinary float32 float2 UV0 is already packed with the vertex;
wider UV0 and other properties use additional streams. Dynamic geometry keeps these additional streams static.

Triangle and LSS collections each own `RenderGeometryAdditionalAttributeStorage`. Every geometry has a row of
33 two-word records: a presence mask followed by present descriptors in attribute-ID order. Mask testing and
population count locate each stream's byte offset, type and width. Values occupy a separate aligned raw buffer,
so optional attributes do not enlarge traversal vertices. Empty rows remain valid; raw data is uploaded and
bound only when that geometry type has additional streams.

### Instances and acceleration structures

A geometry group owns a bottom-level acceleration structure (BLAS) over its triangle, LSS or analytic-light
primitives. A placed group contributes one native instance to the top-level acceleration structure (TLAS).
Instances share the BLAS while providing transforms and material assignments.

`RenderScene` emits one `GeometryInstanceDesc` for every entry in every placed group. The native instance ID is
the base descriptor index, so `InstanceID() + GeometryIndex2()` selects the exact sub-mesh or curve entry.
Descriptors identify geometry, material and transform and carry the buffer-range and motion information needed
by shaders. Current, inverse and previous transforms are stored separately. `Scene.bind()` exposes the TLAS,
pointer table, descriptors and resources under `g_scene`.

LSS uses native primitives when the device supports them, unless the implementation's force-procedural switch
is enabled. Otherwise a GPU pass computes segment bounding boxes and traversal uses a procedural intersection
shader. Both paths expose the same logical geometry and hit interface. Analytic-light geometry has its own type
and hit-group block; its transport semantics belong in [Lighting and sampling](lighting-and-sampling.md).

### Hit groups and reconstruction

The scene's `HitGroupPolicy` fixes three ray-type slots and three geometry types, with records shared per type.
The TLAS contribution is `geometry_type * ray_type_count`; the geometry contribution multiplier is zero.
The shader table orders triangle, LSS and analytic-light blocks, then ray types, independently of instance count.

`SceneRayTracingSetup` derives miss, closest-hit, any-hit and procedural-intersection entry points from renderer
ray descriptions. It pads omitted ray types, can omit real records for unused geometry types and enables the
hardware-LSS pipeline flag when needed. Native pipeline creation and SlangPy dispatch consume the same metadata.

`IDeferredHitInfo.create()` produces geometry-specific hit information. Packed hits retain geometry type,
instance identity, primitive index and local hit coordinates. LSS coordinates include caps. Immediate native
LSS shading may obtain endpoints from `GetLssPositionsAndRadii()` through the default-false
`HasTracingIntrinsics` generic. The reference tracer enables it only for native LSS closest-hit shading under
both its simple and SER schedulers. UVs and transforms still come from buffers.

Cached hits, procedural hits, query callers, opacity any-hit shading and previous-position reconstruction keep
buffer access. The helper admits the intrinsic only in supported hit stages, but stage eligibility alone does
not prove that a supplied hit is the current native hit. Persistent hits and payloads carry neither endpoints
nor an intrinsic-availability flag.

## Runtime flow

```text
 scene geometry + deformer       instance materials + entity transforms
             |                                  |
             +----------- Scene.update() -------+
                                |
                    uploads + GPU deformation
                                |
                   LSS bounds + pointer table
                                |
                   dirty group BLAS build/refit
                                |
                    instance records + TLAS
                                |
                 bound g_scene + hit-group table
                                |
                      renderer traversal
```

1. Scene updates synchronize geometry before entity/component render state. Static reauthoring replaces
   low-level entries and groups. Dynamic preparation updates existing fixed output ranges; deformation dirtiness
   reaches geometry consumers, including emissive geometry, without treating each frame as topology replacement.
2. `RenderScene` uploads descriptors, indices and static data, then prepares and dispatches geometry updaters.
   Dynamic buffers transition for writes and subsequent reads explicitly. LSS bounds consume the resulting
   current vertices through direct pointer work records. The shared table is then refreshed for traversal and
   shading readers.
3. Only dirty geometry groups receive BLAS work. Dynamic groups allow updates; a refit requires an existing
   BLAS, a refit request and no rebuild request. Rebuild takes precedence. Full builds reuse a sufficiently large
   allocation, and new or undersized allocations require a full build. Shared scratch storage covers build and
   update requirements; pending geometry work is submitted before BLAS execution.
4. Structural, group, material, culling or opacity changes regenerate instance records. Transform-only changes
   preserve their layout and use a compute pass to patch native transforms from scene buffers.
5. Clean or descriptor-payload-only instance changes skip TLAS work. Transform-only changes use in-place TLAS
   update by default; other changes and new allocations require a full build. A changed BLAS group currently
   causes a full TLAS build even when the BLAS itself was refitted. An empty scene binds a null TLAS.
6. The renderer resolves scene requirements and ray-tracing setup, binds synchronized resources and dispatches.
   Hit wrappers recover the logical instance and surface interaction. Motion reconstruction combines previous
   vertices, where available, with previous transforms; additional-property access uses the same hit identity.

## Invariants and rationale

### Identity, history and traversal policy

`GeometryInstanceID` identifies a synchronized descriptor snapshot, not a persistent scene object. A component
exposes its base ID and consecutive descriptor count; structural updates can repack them. Geometry groups must
be homogeneous because one TLAS contribution selects one geometry-type block.

Dynamic topology and vertex correspondence stay fixed while deforming. Current and previous vertex ranges share
indices. Deformer history distinguishes initialization, changed inputs and catching previous data up on a paused
frame. Catch-up alone does not mark current geometry deformed. Geometry replacement or collection initialization
resets history; callers must not assume continuity across such changes.

Opacity remains conservative: geometry build inputs are opaque, but any opacity-requiring material makes all
triangle TLAS instances non-opaque so renderers can evaluate any-hit opacity. Backface culling is a separate
contract. Each instance stores eligibility derived from its configured material list; material culling edits
update it without replacing the instance handle. Reference-tracer culling is an explicit, default-off renderer
specialization. Ineligible instances disable native culling, and picking remains two-sided.

### LSS intersection and endpoint reuse

The procedural scene intersector solves an endpoint quadratic for the tangent-cone body, including the cylinder
case, and tests enabled endpoint caps. It accepts the entry boundary only: starting inside a segment, clipping
its entry with `t_min`, or entering through a disabled cap does not substitute an exit hit. Successive-index
chains enable both caps on the first primitive of each chain and only the trailing cap thereafter. Acceptance
is per segment; whole-strand CSG merging would change that contract.

The generic `intersect_ray_linear_swept_sphere_han19` utility has different nearest-component semantics and can
return exits; scene traversal does not call it. Reconstruction uses the cap-inclusive coordinates of accepted
hits. Current-hit intrinsics replace only endpoint reads and cannot provide previous geometry or transform
history. Matching endpoint bits alone does not prove identical floating-point shading paths, so native-hit
render tests complement generic reconstruction coverage.

### Attribute and remapping contracts

Importer deduplication compares meaningful bytes of every declared stream, ignoring padding and preserving the
first occurrence. Hashes choose buckets; exact comparison resolves collisions. The remap applies to indices and
all attributes, preserving authored seams. Subsequent sub-mesh extraction is not another deduplication pass.

Floating-point property access converts half/float values, zero-fills trailing components and interpolates at
the hit. Integer and Boolean access selects the greatest-weight triangle vertex or nearer LSS endpoint, with
ties favoring the first vertex. Missing or incompatible queries fail with zero output; convenience getters apply
their fallback. Ordinary float2 UV0 comes from the surface interaction.

`SceneConfig.enable_custom_attributes` is immutable and defaults to enabled. Disabling it removes packed hit
identity from `SurfaceInteraction` and selects an invalid accessor. When enabled, MaterialX and MDL use the
production geometry accessor. Attribute registration changes runtime data without itself invalidating shader
requirements; geometry availability and other scene capabilities are tracked separately.

## Source landmarks

- `src/falcor2/render/geometry.h`, `src/falcor2/render/geometry/static_mesh_geometry.cpp`,
  `src/falcor2/render/geometry/static_curve_geometry.cpp`, and
  `src/falcor2/render/component/geometry_instance.cpp` - geometry ownership, packing and placement.
- `src/falcor2/render/geometry/dynamic_mesh_geometry.cpp`,
  `src/falcor2/render/geometry/dynamic_curve_geometry.cpp`,
  `src/falcor2/render/geometry/geometry_deformer.h`, and
  `src/falcor2/render/geometry/vertex_cache_updater.cpp` - output ownership and shared deformation preparation.
- `src/falcor2/render/render_scene.h`, `src/falcor2/render/render_scene.cpp`, and
  `src/falcor2/render/scene.cpp` - pools, update order, dirty propagation, BLAS/TLAS and binding.
- `src/falcor2/utils/managed_buffer_pool.h` and `src/falcor2/utils/buffer_handle_table.h` - lazy typed
  allocations, buffer tuples and the shared pointer table.
- `src/falcor2/render/geometry/geometry_attribute_registry.cpp`,
  `src/falcor2/render/geometry/geometry_vertex_attribute.cpp`, and
  `src/falcor2/importers/vertex_remap.cpp` - typed streams, IDs and exact remapping.
- `src/falcor2/render/ray_tracing_policy.h`, `src/falcor2/render/ray_tracing_setup.cpp`,
  `src/falcor2_ext/render/scene.cpp`, `src/falcor2_ext/render/mesh.cpp`, and
  `src/falcor2_ext/render/ray_tracing_setup.cpp` - shared policy and Python interfaces.
- `slang/falcor2/render/scene.slang`, `slang/falcor2/render/scene_ray_tracing.slangh`,
  `slang/falcor2/render/geomprop_accessor.slang`, and `slang/falcor2/render/geometry/` - device layout,
  hit wrappers, geometry loaders, property access and LSS intersection/reconstruction.
- `slang/falcor2/rendernodes/reference_pathtracer.slang` and
  `falcor2/rendernodes/reference_pathtracer_node.py` - renderer culling and native endpoint eligibility.
- `tests/native/render/test_scene.cpp`, `tests/native/render/test_dynamic_geometry.cpp`, and
  `tests/native/importers/test_vertex_deduplication.cpp` - BLAS rebuild/refit, deformation, IDs and remapping.
- `tests/python/render/test_geometry_properties.py`, `tests/python/render/test_motion_vectors.py`,
  `tests/python/render/test_mesh_motion.py`, `tests/python/render/test_gltf_backface_culling.py`, and
  `tests/python/pathtracer/test_lss_closest_hit.py` - typed access, deformation/history, culling and hits.
- `examples/pathtracer/simple.py` and `examples/pathtracer/headless.py` - consumers of the common setup.

## Related design documents

- [Geometry deformation](geometry-deformation.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Buffer pointers and indirect GPU access](buffer-pointers.md)
- [Render pipeline](render-pipeline.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Falcor2 design documentation](index.md)
