---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-14
verified_against: "3c40a55d4725fc793996e58ac149edfe19257a48"
---

(falcor2-design-geometry-and-ray-tracing)=
# Geometry and ray tracing

Falcor2 separates scene-owned geometry from the compact device representation used for ray traversal. A `Geometry`
retains CPU-side mesh or curve data, including typed per-vertex properties, while an entity's `GeometryInstance`
supplies placement and per-part materials. During `Scene.update()`, these objects populate `RenderScene` pools,
shared device buffers, transform records, and two levels of acceleration structure. Optional attributes use separate
packed buffers and per-hit accessors so they do not enlarge the hot traversal vertex. A fixed hit-group policy and
`SceneRayTracingSetup` keep instance IDs, shader-table ordering, geometry types, and Slang hit reconstruction
consistent across renderers.

## Purpose and scope

This page explains live geometry ownership, mesh and curve representation, typed additional vertex attributes,
instancing, `RenderScene` storage, acceleration-structure updates, hit-group indexing, and renderer ray-tracing
setup. It also records the exact vertex deduplication contract applied while imported meshes are still CPU-side
data.

File decoding and live-scene realization belong in [Scene loading and authoring](scene-loading-and-authoring.md).
General object lifecycle and update ordering belong in [Scene model](scene-model.md). Material evaluation, light
sampling, and path-transport policy are outside this page; see their focused pages and
[Render pipeline](render-pipeline.md).

## Architecture

### Scene geometry and placement

`Scene` owns every `Geometry` in its geometry collection. The implemented renderable types are
`StaticMeshGeometry` and `StaticCurveGeometry`. A static mesh stores one or more self-contained sub-meshes. Each
sub-mesh has packed position, normal, tangent, handedness, and texture-coordinate vertices plus triangle indices.
Loading the mesh extracts only vertices referenced by that sub-mesh and remaps source indices in first-reference
order. The same remap is applied to every retained generic vertex attribute. A static curve stores packed
center/radius vertices and segment indices for linear swept spheres (LSS), plus optional per-control-point
attributes. List-form segment pairs are converted to the successive convention required by the device path,
duplicating or reordering endpoints and their attributes when pairs are not already adjacent.

A `GeometryInstance` component attaches geometry to an entity. The entity supplies the world transform, while the
component supplies one material slot per contained render geometry. A mesh creates one low-level triangle geometry
per sub-mesh and groups them; a curve creates one LSS geometry and a one-entry group. The component creates one
low-level group instance and exposes the base `GeometryInstanceID` plus the number of consecutive descriptors owned
by that instance after update. Missing material slots use material ID zero.

The public scene-object `GeometryGroup` is currently an unimplemented placeholder. It is distinct from the internal
`RenderGeometryGroup`, which is the working same-type collection used to build one acceleration structure.

### Additional vertex attributes

`GeometryVertexAttributeDesc` is the shared non-owning authoring record for static meshes and curves. It describes a
named semantic and index, scalar type, component count, byte layout, and data pointer. Geometry validates and copies
each stream into tightly packed owned `GeometryVertexAttribute` storage. The established static-mesh property helpers
use this same retained representation, and Python authors generic mesh or curve streams with `MeshVertexAttribute`
NumPy arrays. Existing `set_mesh_data()` arguments remain convenience inputs rather than separate storage.

The `Scene` assigns shader-visible attributes IDs in one 32-bit namespace. `TEXCOORD_0` through `TEXCOORD_7` use IDs
0 through 7 and `COLOR_0` uses ID 8. Core packed properties receive no additional ID. Other supported properties
use exact nonempty names and scene-lifetime IDs 16 through 31, regardless of their semantic enum. Named IDs are
monotonic. A material read can reserve a name before geometry provides it, without establishing a storage type or
claiming data is present. The first geometry source fixes the scalar type and component count; later sources of
that name must agree. Shader reads keep their own requested type and fallback, and availability remains per geometry.

CPU geometry retains types outside the shader ABI, warning when lowering cannot expose one. The supported ABI is
Boolean, signed 32-bit integer, half, and float with one through four components; Boolean values become packed
zero-or-one bytes. Ordinary triangle float2 UV0 remains in `PackedTriangleVertex`; wider UV0 and other properties
use additional streams. All LSS attributes use additional streams because its packed vertex contains only center
and radius.

### RenderScene storage

`RenderScene` is the scene's private renderer-facing geometry layer. Typed integer handles address pools for triangle
geometry, LSS geometry, geometry groups, instances, and transforms. Triangle and LSS pools allocate ranges in shared
managed arrays for packed vertices and indices; a handle-indexed descriptor records each range. Dirty ranges are
uploaded to device buffers rather than giving every geometry a separate vertex or index buffer.

Triangle and LSS pools each own `RenderGeometryAdditionalAttributeStorage`. Every geometry has a fixed row of 33
two-word records: a presence mask followed by present descriptors in attribute-ID order. Mask testing plus a
population count finds a descriptor without searching. Descriptors encode byte offset, type, and component count;
values occupy an aligned raw buffer independent of traversal data. Zero-mask rows remain valid, while raw values are
updated and bound only when some geometry of that type has additional streams.

Each `RenderGeometryGroup` contains only triangle entries or only LSS entries. It owns a bottom-level acceleration
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
retain the same scene-facing LSS descriptors and hit-information interface.

### Procedural LSS mathematics and native policy

The fallback uses the tangent-cone geometry and clipping planes described by
[Han et al. (2019), *Ray Tracing Generalized Tube Primitives: Method and Applications*](https://doi.org/10.1111/cgf.13703),
Section 3.2.2 ([author copy](https://www.sci.utah.edu/~feng/assets/papers/tubes.pdf)). It solves an endpoint quadratic
instead of transforming to a unit cone as in Section 4.1. This avoids constructing a distant apex for nearly equal
radii and remains defined at equal radii, where the body becomes a cylinder.

A public implementation of the equivalent quadratic is
[Embree's `ConeGeometryIntersector::intersectCone`](https://github.com/RenderKit/embree/blob/3d9cb89b9ea099c630e6272d37767e7dd4e78e74/kernels/geometry/roundline_intersector.h#L229-L332),
under [Apache-2.0](https://github.com/RenderKit/embree/blob/3d9cb89b9ea099c630e6272d37767e7dd4e78e74/LICENSE.txt).
This reference describes the mathematical correspondence; it does not establish code provenance.

Let `a = p1 - p0`, `dr = r1 - r0`, `q = x - p0`, and `g = dot(a, a) - dr*dr`. Minimizing
`|q - u*a|^2 - (r0 + u*dr)^2` over the interpolated spheres gives, for `g > 0`,

```text
u(q) = (dot(q, a) + r0*dr) / g
F(q) = g * (dot(q, q) - r0*r0) - (dot(q, a) + r0*dr)^2 = 0
```

Clipping to `0 <= u <= 1` gives Han's tangent planes. Substituting the shifted ray `q = O + t*D` yields
`A*t*t + 2*B*t + C = 0`; the coefficient called `B` in Embree is twice the one used here. The code evaluates
these coefficients using the identity `dot(cross(a,x), cross(a,y)) = dot(a,a)*dot(x,y) - dot(a,x)*dot(a,y)`
to reduce cancellation near axis-parallel cylinders. Its `fma` evaluation of `C` preserves the small
radius-difference contribution near a nearly cylindrical tangent. For `g <= 0`, one sphere contains the other
and the body is omitted.

The implementation accepts nonunit directions, as required after instance scaling. Centering the ray coordinates
improves distant-ray conditioning; roots remain signed until the original ray parameter is restored. Stable entry-root
selection, a linear-equation case, and explicit zero-radius/contained-sphere handling cover cases where a direct
application of the cone-apex construction is insufficient.

Hit acceptance follows the [native LSS intersection rules](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-intersection-candidate-determination):
only the entry boundary is accepted. Starting inside a segment, entering through its disabled cap, or clipping
its entry with `t_min` does not expose an exit as a replacement hit. `Scene.intersect_lss` implements
[successive-index chained caps](https://docs.vulkan.org/refpages/latest/refpages/source/VkRayTracingLssPrimitiveEndCapsModeNV.html):
both caps on the first primitive and only the trailing cap on subsequent primitives.

Rejection is per segment, so an enabled cap of another segment can still be hit from inside the strand. Han's
whole-tube CSG interval merging (Section 4.3) and Embree's neighboring-cone subtraction would impose different
semantics. The fallback implements neither, to match the native chained LSS contract.

The generic `intersect_ray_linear_swept_sphere_han19` utility retains the tangent-cone/unit-cone method and nearest
component-hit behavior, including possible exit hits; scene traversal no longer calls it. `curve_parameters_to_hit_pos`
uses Han's tangent geometry to reconstruct accepted hits from Falcor's cap-inclusive axial/azimuthal coordinates.

### Hit groups and renderer setup

The scene currently fixes `HitGroupPolicy` to three ray-type slots and two geometry types, with shader-table records
shared per geometry type. The TLAS instance contribution is therefore `geometry_type * ray_type_count`; the geometry
contribution multiplier is zero. The resulting shader binding table is ordered first by triangle or LSS type and
then by ray type, independent of the number of scene instances.

`SceneRayTracingSetup` is the renderer-facing bridge for that policy. Given ray descriptions, it derives standard
miss, closest-hit, any-hit, and LSS intersection entry-point names, pads omitted ray types or geometry types with
dummy records, and can omit real records for unused geometry types. It also enables the hardware-LSS pipeline flag
when required. The helper exposes this metadata to Python and can link a native program, pipeline, and shader table;
the reference path tracer passes the same metadata to SlangPy's ray-tracing dispatch.

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
4. Structural, group, material, or opacity changes regenerate logical geometry descriptors and native TLAS instance
   records. If transforms are the only change, those records keep their layout and a compute kernel patches native
   instance transforms from the transform buffers.
5. TLAS work is skipped when no instance state is dirty. Transform-only changes use in-place TLAS update (refit) by
   default; other changes perform a full build. Falcor2 allocates a larger TLAS when required and falls back to a
   full build for a new allocation. With no instances, the bound TLAS is null.
6. Renderer compilation obtains the scene's requirements and ray-tracing setup. At dispatch, the renderer binds
   `g_scene`; traversal selects the geometry-type/ray-type record, and the generated hit wrapper reconstructs the
   logical geometry instance and typed hit information used by shading. With custom attributes enabled, a
   `SurfaceInteraction` retains packed hit identity and creates a triangle, LSS, or invalid `IGeomPropAccessor` for
   typed values and object/world transforms.

## Invariants and rationale

Geometry data is reusable and placement is separate. Multiple entities may reference the same geometry group and
BLAS while supplying different transforms and materials. Conversely, the geometries within one render group must
all have the same geometry type because one TLAS instance contribution selects that type's hit-group block.

`GeometryInstanceID` is renderer-facing snapshot data, not persistent scene-object identity. One component's ID is
the first descriptor for its group, and `geometry_instance_count` is that group's geometry count. Structural updates
can repack descriptor indices; code that needs entity identity should resolve IDs against the synchronized scene and
must not retain them across arbitrary updates. Geometry and transform edits likewise take effect on device state only
after `Scene.update()`.

Opacity is conservative at the traversal boundary. Geometry build inputs are marked opaque, but when any current
material requires opacity evaluation, every triangle TLAS instance receives the native `no_opaque` flag and
renderers request the corresponding any-hit entry points. This is a scene-wide choice today; material-level opacity
semantics belong in the material-system design.

Imported mesh deduplication is exact and precedes live geometry creation. Vertex identity compares every declared
attribute's meaningful bytes across strided streams, ignores padding, and preserves the first source occurrence.
Hashes select candidate buckets only; byte comparison resolves collisions. This keeps position-equal vertices with
different normals, texture coordinates, colors, or other declared attributes distinct, preserving authored seams.
The local remapper has no production dependency and returns both source-to-compact and compact-to-source mappings so
indices and all buffers are rewritten consistently. Live `StaticMeshGeometry` then makes each sub-mesh
self-contained; that per-sub-mesh extraction is a separate operation, not another attribute-based deduplication pass.

Additional attributes do not enlarge packed vertices or acceleration-structure inputs. The mask-and-rank layout
bounds geometry to 32 shader-visible properties and keeps lookup constant-time. Fixed IDs are authoring-order
independent; scene-wide named IDs avoid per-mesh name searches. Registration changes runtime data without itself
invalidating scene shader requirements.

Floating-point access converts half or float data to the requested width, zero-fills trailing components, and
interpolates over the hit. Integer and Boolean access chooses the greatest-weight triangle vertex or nearer LSS
endpoint; ties favor the first vertex. Missing or type-mismatched queries fail with zero output. Ordinary triangle
float2 UV0 comes from `SurfaceInteraction.uv`; only wider UV0 is also an additional attribute.

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
