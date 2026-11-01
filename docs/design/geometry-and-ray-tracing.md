---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-10-27
verified_against: "eabf25d588bc365f6773906c930d9d63d6ed6356"
---

(falcor2-design-geometry-and-ray-tracing)=
# Geometry and ray tracing

Falcor2 separates scene-owned geometry from the compact device representation used for ray traversal. A `Geometry`
retains CPU-side mesh or curve data, including typed per-vertex properties, while an entity's `GeometryInstance`
supplies placement and per-part materials. During `Scene.update()`, these objects populate `RenderScene` pools,
shared device buffers, transform records, and two levels of acceleration structure. Optional attributes use separate
packed buffers and per-hit accessors so they do not enlarge the hot traversal vertex. A fixed hit-group policy and
`SceneRayTracingSetup` keep instance IDs, shader-table ordering, geometry types, and Slang hit reconstruction
consistent across renderers. Immediate native curve shading can reuse traversal endpoints, while saved hits and
other reconstruction contexts retain buffer access.

## Purpose and scope

This page explains live geometry ownership, mesh and curve representation, typed additional vertex attributes,
instancing, `RenderScene` storage, acceleration-structure updates, hit-group indexing, and renderer ray-tracing
setup, including the boundary between current-hit intrinsics and reusable hit records. It also records the exact
vertex deduplication contract applied while imported meshes are still CPU-side data.

File decoding and live-scene realization belong in [Scene loading and authoring](scene-loading-and-authoring.md).
General object lifecycle and update ordering belong in [Scene model](scene-model.md). Material evaluation, light
sampling, and path-transport policy are outside this page; see their focused pages and
[Render pipeline](render-pipeline.md).

## Architecture

### Scene geometry and placement

`Scene` owns `StaticMeshGeometry` and `StaticCurveGeometry` objects. Each mesh sub-mesh has packed position,
normal, tangent, handedness and UV vertices plus triangle indices. Loading extracts its referenced vertices in
first-reference order and applies the same remap to additional attributes. A static curve stores packed
center/radius/UV0 vertices and segment indices for linear swept spheres (LSS), plus optional per-control-point
attributes. List-form segment pairs are converted to the successive convention required by the device path,
duplicating or reordering endpoints and their attributes when pairs are not already adjacent.

A `GeometryInstance` component attaches geometry to an entity. The entity supplies the world transform, while the
component supplies one material slot per contained render geometry. A mesh creates one low-level triangle geometry
per sub-mesh and groups them; a curve creates one LSS geometry and a one-entry group. The component creates one
low-level group instance and exposes the base `GeometryInstanceID` plus the number of consecutive descriptors owned
by that instance after update. Missing material slots use material ID zero.

The public `GeometryGroup` is an unimplemented placeholder; internal `RenderGeometryGroup` is the working collection.

### Additional vertex attributes

Meshes and curves accept non-owning `GeometryVertexAttributeDesc` records describing semantic, index, type,
component count, byte layout and data pointer. Geometry validates and copies streams into owned, tightly packed
`GeometryVertexAttribute` storage. Python `MeshVertexAttribute` arrays and existing mesh-property convenience
arguments feed this same representation.

`Scene` assigns attribute IDs in a 32-bit namespace: UV0-7 occupy IDs 0-7, color zero uses ID 8, and named properties
use monotonic scene-lifetime IDs 16-31. Core packed properties need no additional ID. Material reads can reserve
names without establishing storage or presence. The first geometry source fixes type and width; later sources must
agree. Reads retain their requested type and fallback, while availability remains per geometry.

CPU geometry retains types outside the shader ABI, warning when lowering cannot expose one. The supported ABI is
Boolean, signed 32-bit integer, half, and float with one through four components; Boolean values become packed
zero-or-one bytes. Ordinary float32 float2 UV0 resides in `PackedTriangleVertex` or `PackedLSSVertex`; wider UV0
and other properties use additional streams. LSS surface reconstruction interpolates the packed endpoint UVs.

### RenderScene storage

`RenderScene` is the scene's private renderer-facing geometry layer. Typed integer handles address pools for triangle
geometry, LSS geometry, analytic area-light geometry, geometry groups, instances, and transforms. Triangle and LSS
pools allocate ranges in shared
managed arrays for packed vertices and indices; a handle-indexed descriptor records each range. Dirty ranges are
uploaded to device buffers rather than giving every geometry a separate vertex or index buffer.

Triangle and LSS pools each own `RenderGeometryAdditionalAttributeStorage`. Each geometry has 33 two-word records:
a presence mask followed by descriptors in attribute-ID order. Mask testing and population count locate byte offset,
type and width without searching. Values use a separate aligned raw buffer, updated and bound only when that
geometry type has additional streams. Zero-mask rows remain valid.

Each `RenderGeometryGroup` contains entries of one geometry type: triangles, LSS, or analytic area lights. It owns a
bottom-level acceleration
structure (BLAS): a device search structure over the primitives of one reusable geometry group. A corresponding
`RenderGeometryInstance` references the group, one lazily created entity transform, and its material IDs. For
shaders, `RenderScene` builds one `GeometryInstanceDesc` per geometry inside every group. The descriptors identify
geometry type, geometry handle, material, transform, and buffer ranges. Separate arrays hold current, inverse, and
previous transforms for shading and motion vectors.

One low-level group instance contributes one native instance to the top-level acceleration structure (TLAS): a
device search structure over placed BLASes and their transforms. Its native instance ID is the base index of that
group's shader descriptors. Within a hit shader, `InstanceID() + GeometryIndex2()` therefore selects the descriptor
for the exact sub-mesh or curve entry that was hit. `Scene.bind()` exposes the TLAS, geometry buffers, descriptors,
and transform arrays under `g_scene`.

LSS uses native hardware primitives when the device advertises support. Otherwise, Falcor2 computes segment
axis-aligned bounding boxes, builds procedural primitives, and supplies a custom intersection shader. Both paths
retain the same scene-facing LSS descriptors and hit-information interface. Analytic light intersection and
transport semantics belong in [Lighting and sampling](lighting-and-sampling.md).

### Procedural LSS mathematics and native policy

The fallback uses the tangent-cone geometry and clipping planes of
[Han et al. (2019)](https://doi.org/10.1111/cgf.13703), Section 3.2.2
([author copy](https://www.sci.utah.edu/~feng/assets/papers/tubes.pdf)). An endpoint quadratic avoids a distant apex
near equal radii and remains defined for cylinders.
[Embree's cone intersector](https://github.com/RenderKit/embree/blob/3d9cb89b9ea099c630e6272d37767e7dd4e78e74/kernels/geometry/roundline_intersector.h#L229-L332)
implements an equivalent quadratic under
[Apache-2.0](https://github.com/RenderKit/embree/blob/3d9cb89b9ea099c630e6272d37767e7dd4e78e74/LICENSE.txt);
this is a mathematical comparison, not a provenance claim.

Let `a = p1 - p0`, `dr = r1 - r0`, `q = x - p0`, and `g = dot(a, a) - dr*dr`. Minimizing
`|q - u*a|^2 - (r0 + u*dr)^2` over the interpolated spheres gives, for `g > 0`,

```text
u(q) = (dot(q, a) + r0*dr) / g
F(q) = g * (dot(q, q) - r0*r0) - (dot(q, a) + r0*dr)^2 = 0
```

Clipping `u` to the segment gives Han's tangent planes. Substituting the shifted ray gives
`A*t*t + 2*B*t + C = 0` (Embree's `B` is twice this coefficient). Cross-product identities reduce cancellation
near axis-parallel cylinders; `fma` preserves the small radius-difference contribution in `C`. For `g <= 0`,
containment eliminates the body. Nonunit directions support instance scaling. Centered coordinates, signed roots,
stable entry-root selection, a linear case and explicit zero-radius handling address numerical edge cases.

Hit acceptance follows the [native LSS intersection rules](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-intersection-candidate-determination):
only the entry boundary is accepted. Starting inside a segment, entering through its disabled cap, or clipping
its entry with `t_min` does not expose an exit as a replacement hit. `Scene.intersect_lss` implements
[successive-index chained caps](https://docs.vulkan.org/refpages/latest/refpages/source/VkRayTracingLssPrimitiveEndCapsModeNV.html):
both caps on the first primitive and only the trailing cap on subsequent primitives.

Rejection is per segment: another segment's enabled cap remains hittable from inside a strand. Whole-tube CSG
merging and neighboring-cone subtraction are excluded because they would change native chained-cap semantics.

The generic `intersect_ray_linear_swept_sphere_han19` utility retains the tangent-cone/unit-cone method and nearest
component-hit behavior, including possible exit hits; scene traversal no longer calls it. `curve_parameters_to_hit_pos`
uses Han's tangent geometry to reconstruct accepted hits from Falcor's cap-inclusive axial/azimuthal coordinates.

### Hit groups and renderer setup

The scene currently fixes `HitGroupPolicy` to three ray-type slots and three geometry types, with shader-table records
shared per geometry type. The TLAS instance contribution is therefore `geometry_type * ray_type_count`; the geometry
contribution multiplier is zero. The shader binding table is ordered by triangle, LSS, or area-light type, then
ray type, independent of instance count.

`SceneRayTracingSetup` is the renderer-facing bridge for that policy. Given ray descriptions, it derives standard
miss, closest-hit, any-hit, and procedural intersection entry-point names, pads omitted types with dummy records,
and can omit real records for unused geometry types. It also enables the hardware-LSS pipeline flag
when required. The helper exposes this metadata to Python and can link a native program, pipeline, and shader table;
the reference path tracer passes the same metadata to SlangPy's ray-tracing dispatch.

### Current-hit endpoint reuse

`IDeferredHitInfo.create()` produces a geometry-specific `IHitInfo`. Native LSS hit creation uses
`GetLssPositionsAndRadii()` to compute cap-inclusive axial/azimuthal coordinates. Packing retains geometry type,
instance identity, primitive index and those coordinates, without endpoints or an intrinsic-availability flag.

The reference tracer enables the default-false `HasTracingIntrinsics` generic only in its native LSS closest-hit
callback. It reaches `Scene.make_surface_interaction` at the ordinary shading point under both schedulers.
`detail::load_lss_segment` then selects native endpoints instead of buffer positions/radii. Buffer-backed UV
interpolation, curve reconstruction, coordinate conversions and transform access remain unchanged.

The helper's stage switch admits the intrinsic in closest-hit and any-hit stages and preserves compute/query
callability under Slang's capability checking. Other stages use buffers. The tracer's any-hit opacity path does
not enable the capability. Cached hits, procedural LSS, generic/query callers and previous-position reconstruction
also use buffers. Current-hit intrinsics cannot supply deformation or transform history.

## Runtime flow

```text
 CPU mesh or curve data
          |
          v
 Geometry ---- GeometryInstance ---- Entity transform
          \             |                  /
           +------ Scene.update() --------+
                         |
              packed shared buffers
                         |
              geometry group -> BLAS
                         |
              placed group(s) -> TLAS
                         |
          hit-group table + bound g_scene
                         |
                 renderer traversal
```

1. Import or programmatic authoring fills CPU-side geometry and marks its render state dirty. Supported additional
   streams register fixed or named scene IDs when geometry is attached or reauthored, reusing any names already
   reserved by material codegen. `Scene.update()` processes
   geometry before entity transforms and components. A static geometry update recreates its low-level entries and
   group; a component update recreates its low-level instance when geometry or materials change.
2. Geometry lowering preserves remapped vertex order and normalizes Boolean data. `RenderScene` uploads dirty core,
   additional-attribute, and transform ranges. Changed geometry dirties its groups and their instances; changed
   transforms mark only referencing instances.
3. Before BLAS construction, pending geometry uploads are submitted. The current implementation rebuilds every
   geometry-group BLAS whenever any group is dirty, allocating a new structure for each group. There is no BLAS
   refit path. A shared scratch buffer grows to the largest build requested and is reused.
4. Structural, group, material, culling, or opacity changes regenerate logical descriptors and native TLAS instance
   records. Transform-only changes preserve their layout; a compute kernel patches transforms from scene buffers.
5. TLAS work is skipped for descriptor-payload-only changes or clean instance state. Transform-only changes use
   in-place TLAS update (refit) by default; other changes perform a full build. New allocations require a full build.
   With no instances, the bound TLAS is null.
6. Renderer compilation obtains the scene's requirements and ray-tracing setup. At dispatch, the renderer binds
   `g_scene`; traversal selects the geometry-type/ray-type record, and the generated hit wrapper reconstructs the
   logical geometry instance and typed hit information used by shading. With custom attributes enabled, a
   `SurfaceInteraction` retains packed hit identity and creates a triangle, LSS, or invalid `IGeomPropAccessor` for
   typed values and object/world transforms.

## Invariants and rationale

Instances share geometry groups and BLASes while supplying different transforms and materials. A group's geometry
type must be uniform because its TLAS contribution selects one hit-group block.

`GeometryInstanceID` identifies a descriptor snapshot, not a persistent scene object. A component exposes its first
descriptor and `geometry_instance_count`. Structural updates can repack indices; resolve entity identity against
the synchronized scene. Geometry and transform edits reach device state through `Scene.update()`.

Opacity is conservative at the traversal boundary. Geometry build inputs are marked opaque, but when any current
material requires opacity evaluation, every triangle TLAS instance receives the native `no_opaque` flag and
renderers request the corresponding any-hit entry points. This is a scene-wide choice today; material-level opacity
semantics belong in the material-system design.

Imported mesh deduplication compares meaningful bytes of every declared attribute, ignoring padding and preserving
first occurrence. Hashes select buckets; byte comparison resolves collisions and preserves authored seams. The
dependency-free remapper returns both mapping directions for consistent index and attribute rewriting.
`StaticMeshGeometry` subsequently extracts self-contained sub-meshes; that operation is not another deduplication.

Additional attributes do not enlarge packed vertices or acceleration-structure inputs. The mask-and-rank layout
bounds geometry to 32 shader-visible properties and keeps lookup constant-time. Fixed IDs are authoring-order
independent; scene-wide named IDs avoid per-mesh name searches. Registration changes runtime data without itself
invalidating scene shader requirements.

Floating-point access converts half or float data to the requested width, zero-fills trailing components, and
interpolates over the hit. Integer and Boolean access chooses the greatest-weight triangle vertex or nearer LSS
endpoint; ties favor the first vertex. Missing or type-mismatched queries fail with zero output. Ordinary
float32 float2 UV0 for triangles and LSS comes from `SurfaceInteraction.uv`; wider UV0 is also an additional attribute.

Enabling `HasTracingIntrinsics` promises that the supplied hit is the current native LSS hit; shader stage alone
does not establish this. No endpoint state is added to packed hits or ray payloads. Access changes can affect
floating-point optimization, so matching endpoint bits alone does not establish identical rendered paths.
Native-hit image tests complement generic/query reconstruction tests.

`SceneConfig.enable_custom_attributes` is immutable and defaults to enabled. Disabling it specializes packed hit
identity out of `SurfaceInteraction`, leaving the invalid accessor fallback. MaterialX and MDL both use the
production per-hit accessor; the former static mock-provider interface has been removed.
MaterialX-specific name and semantic lowering is described in [MaterialX integration](materialx-integration.md).

## Source landmarks

- `src/falcor2/render/geometry.h`, `src/falcor2/render/geometry/geometry_vertex_attribute.cpp`,
  `src/falcor2/render/geometry/geometry_attribute_registry.cpp`,
  `src/falcor2/render/geometry/static_mesh_geometry.cpp`,
  `src/falcor2/render/geometry/static_curve_geometry.cpp`, and `src/falcor2/render/geometry/geometry_group.h` - CPU
  geometry, generic vertex streams, scene attribute IDs, packing, remapping, bounds, low-level group creation, and
  the public placeholder.
- `src/falcor2/render/scene.cpp`, `src/falcor2/render/component/geometry_instance.cpp`, and
  `src/falcor2/render/entity.cpp` - update order, placement, material slots, lazy render transforms, base IDs, and
  component synchronization.
- `src/falcor2/render/render_scene.h` and `src/falcor2/render/render_scene.cpp` - typed pools, shared device arrays,
  descriptor construction, dirty propagation, BLAS/TLAS builds, transform history, and shader binding.
- `src/falcor2/render/ray_tracing_policy.h` and `src/falcor2/render/ray_tracing_setup.cpp` - hit-group arithmetic,
  entry-point collection, dummy records, shader-table ordering, and LSS pipeline flags.
- `slang/falcor2/render/scene_types.slang`, `slang/falcor2/render/scene.slang`, and
  `slang/falcor2/render/scene_ray_tracing.slangh` - shared IDs, bound scene layout, and generated hit wrappers.
- `slang/falcor2/render/geomprop_accessor.slang`,
  `slang/falcor2/render/geometry/triangle_geometry_types.slang`,
  `slang/falcor2/render/geometry/triangle_geometry.slang`, and
  `slang/falcor2/render/geometry/lss_geometry.slang` - packed attribute ABI, typed geometry-polymorphic access,
  transform queries, hit reconstruction, surface interactions, and production per-hit property access.
- `slang/falcor2/render/geometry/lss_geometry_types.slang`, `slang/falcor2/render/hit_info.slang`, and
  `slang/falcor2/rendernodes/reference_pathtracer.slang` - packed curve vertices, persistent/deferred hit contracts,
  native endpoint eligibility, and reconstruction placement under simple and SER scheduling.
- `src/falcor2/importers/vertex_remap.h`, `src/falcor2/importers/vertex_remap.cpp`, and
  `src/falcor2/importers/importer_types.cpp` - exact, collision-safe importer vertex remapping and stream compaction.
- `src/falcor2_ext/render/scene.cpp`, `src/falcor2_ext/render/mesh.cpp`, and
  `src/falcor2_ext/render/ray_tracing_setup.cpp` - Python scene, mesh-editing, instance-ID, and setup interfaces.
- `tests/native/render/test_scene.cpp`, `tests/native/render/test_static_curve_geometry.cpp`, and
  `tests/native/importers/test_vertex_deduplication.cpp` - shader-table layout, placement, curve conversion, and
  remapping, attribute-registry, allocation-enablement, and deduplication invariants.
- `tests/python/render/test_static_mesh_geometry.py`, `tests/python/render/test_geometry_properties.py`,
  `tests/python/render/test_geometry_properties.slang`, `tests/python/render/test_object_ids.py`, and
  `tests/python/pathtracer/test_pathtracer.py` - Python authoring/readback, typed triangle and LSS attribute access,
  multi-geometry allocation and update, specialization, opacity, and end-to-end traversal.
- `tests/python/pathtracer/test_lss_closest_hit.py`, `tests/python/pathtracer/test_lss_closest_hit.slang`,
  `tests/python/render/test_usd_curves.py`, and `tests/python/render/test_motion_vectors.py` - native/procedural
  closest hits, UV interpolation, collapsed segments, cached reconstruction, and previous-position behavior.
- `examples/pathtracer/simple.py` and `examples/pathtracer/headless.py` - interactive and headless consumers of the
  same scene geometry and ray-tracing setup.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Render pipeline](render-pipeline.md)
- [Falcor2 design documentation](index.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
