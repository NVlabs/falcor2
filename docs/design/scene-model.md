---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "88cadd20735e943369f61f5d03aadf55507d77be"
---

(falcor2-design-scene-model)=
# Scene model

The Falcor2 scene model connects editable native objects to synchronized renderer-facing state. A `Scene` owns
collections of materials, geometries, animations, entities and components, and applies pending changes at
`Scene.update()`. Entities supply transform hierarchy; components attach behavior; scene systems build device
records and acceleration structures. Animation edits participate in the same dirty-state protocol, including
when time is unchanged. The scene publishes immutable requirement snapshots, resolving material and consumer
capabilities independently of renderer-owned outputs. Renderers use the selected snapshot for shader linking
and ray-tracing setup, while object references, collection indices and generation counters have distinct
lifetime and invalidation roles.

## Purpose and scope

This page explains live-scene ownership, deferred lifecycle, animation bindings, update ordering, capability
resolution and shader binding. These contracts apply regardless of how a scene was populated.

[Scene loading and authoring](scene-loading-and-authoring.md) covers importer staging and callbacks.
[Geometry and ray tracing](geometry-and-ray-tracing.md) and [Geometry deformation](geometry-deformation.md) cover
GPU storage, acceleration structures and deformer history. [Material system](material-system.md),
[Named material outputs](material-aovs.md) and [Lighting and sampling](lighting-and-sampling.md) cover the
corresponding scene systems. Renderer composition and caching belong in [Render pipeline](render-pipeline.md).

## Architecture

### Objects, collections and the entity graph

`Scene` owns its device reference, immutable configuration, shared `falcor2.render` module and five typed
`SceneObjectCollection` instances. Collections hold references to objects; views expose iteration, indexing
and name/type searches. Registered factories create concrete materials, geometries and components. Creation
assigns scene membership and a collection index, supplies a default name when needed, and sets `added` before
calling the object's creation hook.

`SceneObject` supplies validity, membership and common lifecycle callbacks. Shared dirty bits cover resources,
addition and removal; each object family adds flags for its own consumers. Collections store per-object flags
and an aggregate atomic OR so systems can skip irrelevant work. Dirty bits coalesce edits rather than recording
an ordered history.

An `Entity` has a local transform and optional parent. World transforms are cached; edits or reparenting
invalidate descendants and notify attached components. Components belong to the scene-wide collection and are
also referenced by their entity. Geometry instances, cameras, lights and transform animation use this attachment
point. Low-level render transforms are allocated when geometry needs them rather than for every entity.

Each scene starts with a protected built-in free camera and selects it immediately. While initial selection
remains pending, an update with newly added components can select the first valid nonbuiltin camera. Explicit
camera selection, including selecting the free camera through `None`, ends that pending state. Removing the
active camera falls back to the free camera, not another authored camera. Raw collection iteration includes
built-ins; default search and filtered iteration omit them.

`TextureManager`, `MaterialSystem`, `LightSystem`, `RenderScene` and `EmissiveGeometrySystem` translate scene objects
into resources and device state. Central update ordering lets all of them consume an edit before its dirty flags
are cleared.

### Animation ownership and paused edits

`Animation` owns reference-counted `AnimationChannel` objects. Channels are not scene collection members and
attach to at most one animation. Replacing the channel list validates every incoming channel before changing
ownership, so null or foreign-owned entries cannot leave a partially replaced list. A retained channel preserves
its identity across list growth or replacement.

Channel property setters notify the owner's data flag; timestamp changes also reset the cached keyframe accessor.
The animation pass responds to time changes, animation collection dirtiness and component animation-state changes.
Consequently edits to keyframes, interpolation or bindings apply at the next update even in a paused scene.
Python array getters return lists; edits must be assigned back through a property to notify the owner.

`TransformAnimation` retains its `Animation` and selects translation, rotation and scale channels by index.
Attachment requires live objects in the same scene. The animation tracks non-owning backlinks to its live
bindings; removing a binding unregisters it. Scene reference cleanup clears bindings to removed animations.
Missing or empty channels leave their transform fields unchanged; disconnection preserves the last pose.
Removing an overriding binding can reevaluate remaining bindings at unchanged time.

Channel replacement and animation destruction detach channels. Removing an animation does not immediately
detach its channels if another reference retains it; the next update's removal pass clears its scene membership.
Detached channels remain independently editable. Retained channels or animations do not keep the scene alive,
and scene-dependent edits require the owning scene to remain alive.

Transform animation runs before transform upload. Geometry deformation also consumes `Scene.time`, through the
later `RenderScene` update, but owns a separate sample-selection and output-history contract. Animation-range
queries include valid animation key ranges and instantiated dynamic geometry; an unbound keyed animation can
contribute a range without making `has_animation()` report a live animation binding.

### Consumer requests and immutable requirements

`MaterialSystem` computes conservative capability demand from all valid materials, including unassigned ones.
Available geometry streams alone do not require their shader access paths. Capabilities currently describe
geometry-property access, material-output writes, opacity evaluation and solid transmission.

A mutable `SceneRenderRequest` adds consumer-required capabilities and names of material auxiliary outputs
(AOVs). It contains no output formats, targets or compiled resources. Requested names must already be declared
and synchronized by `Scene.update()`; resolution performs lookup, never registration. Any requested AOV currently
adds `material_aovs` and conservative `geometry_properties`. Duplicate names and ordering do not change the
meaning. Output allocation and writer selection remain renderer-owned.

### Capability selection and shader reuse

Scene capability policies select the shader features available to a renderer:

| Capability policy | Selection |
| --- | --- |
| `required` | Union of conservative material demand and consumer demand; the public default |
| `all` | Every supported capability, including output support without an output request |
| `specified` | An explicit supported mask that must cover material and consumer demand |

Unknown demand, unavailable outputs and insufficient explicit selections fail. `scene.requirements`, default
resolution, the shader helper and the reference path tracer use `required`; the scene retains an internal
all-capabilities template. The tracer derives interior tracking from selected `solid_transmission`, rather than
an independent override. Known opaque or thin-only materials can omit it, while unknown material implementations
remain conservative. Scene-specific selection can require more compiled variants across scenes while reducing
work in each specialization; `all` remains an explicit choice.

`SceneRequirements` is published as `ref<const SceneRequirements>`. It holds modules to link, type conformances,
selected capabilities and ray-tracing setup fields. Saved snapshots survive updates and scene destruction.
Native code constructs mutable fields before publication without retaining mutable aliases; Python exposes
read-only fields and copies mutable nested values. Resolution validates each request, then reuses selected
snapshots by effective capability mask. Different requests can therefore share one snapshot without sharing
output storage.

## Runtime flow

1. Creation enters a collection immediately, while add hooks and resource loading wait for update. `remove()`
   invalidates an object immediately and queues teardown; removing an entity recursively removes its descendants
   and components.
2. `Scene.update()` repairs invalid references and camera state, handles pending initial camera selection, then
   runs queued add hooks followed by remove hooks. Resource loading skips invalid removed objects even if they
   still carry resource or addition flags.
3. Textures update before animation, materials and lights. Geometry render state, entity transforms and component
   render state follow, carrying animation-driven changes into the same update's device representation.
4. The scene synchronizes requirement snapshots and conservative material demand, including the current AOV
   catalog generation. Material-driven culling is refreshed before `RenderScene` updates geometry/deformation and
   acceleration structures. Emissive geometry updates afterward with the required shader state available.
5. Collections tear down removed members, compact surviving objects and reset dirty flags. Teardown clears
   membership and releases collection references; external references can keep an invalid detached object alive.
   Unused named `SceneGlobals` are collected. The scene stores result flags, advances `update_generation` only
   for a nonempty result, and emits `updated` even for a no-op update. The outer call then submits any remaining
   command encoder; subsystems may already have submitted work for readbacks.

Renderers consume state after this boundary. `PathTracerPipeline` updates the scene before selecting its camera
and uses `update_generation` to invalidate accumulation. `SceneShaderHelper.prepare()` resolves consumer demand
and returns the linked module together with the exact snapshot used. Callers use that snapshot for conformances
and ray-tracing setup; failed preparation preserves the previous successful helper state.

`Scene.bind()` writes the selected material, light, emissive and render systems under `g_scene`, then binds
additional `SceneGlobals` at global scope. Requirement selection describes shader support; it does not replace
the update or binding operation.

## Invariants and rationale

A collection index is a temporary lookup position. Compaction preserves survivor order but rewrites indices;
retain an object reference rather than assuming its index remains stable. Validity changes at removal, before
collection ownership is released. Retaining an attached scene object does not extend its scene's lifetime.

The entity hierarchy must be an acyclic forest within one scene. `set_parent()` propagates invalidation but does
not validate cycles or cross-scene parenting. Callers preserve those preconditions. Dirty marking is equally
part of the interface: direct mutation that bypasses the relevant setter can leave device state stale.

`requirements_generation` signals a need to resolve again, not a requirement to recompile. Snapshot changes,
material-demand changes or synchronized AOV-catalog changes can advance it, even when an all-capabilities shader
remains unchanged. Generations distinguish scene states; consumers compare the resulting snapshots before
rebuilding resources.

`matches_shader_inputs()` compares capabilities, conformances and underlying Slang component identities.
Full `matches()` also compares hit-group policy, geometry-type mask, linear-swept-sphere intersection mode and
pipeline flags. A geometry-only change can require new ray-tracing setup while reusing a module. Independently,
acceleration-structure opacity follows actual material demand, not an arbitrarily broader shader selection.

## Source landmarks

- `src/falcor2/render/scene.h` and `src/falcor2/render/scene.cpp` - ownership, update ordering, cameras, requirements,
  generations, animation queries and binding.
- `src/falcor2/render/scene_object.h` and `src/falcor2/render/scene_object.cpp` - lifecycle, references, dirty
  aggregation, collection views and compaction.
- `src/falcor2/render/entity.cpp` and `src/falcor2/render/component.cpp` - hierarchy, attachment and invalidation.
- `src/falcor2/render/animation.h`, `src/falcor2/render/animation.cpp` and
  `src/falcor2/render/component/transform_animation.cpp` - channel ownership, notifications and binding lifetimes.
- `src/falcor2/render/scene_requirements.h`, `src/falcor2/render/scene_render_request.h` and
  `src/falcor2/render/scene_capabilities.h` - snapshot, request and capability contracts.
- `src/falcor2/render/material_system.cpp` and `slang/falcor2/rendernodes/reference_pathtracer.slang` - conservative
  demand and selected interior-tracking support.
- `src/falcor2_ext/render/scene.cpp` - collection/factory exposure, retained channels and read-only snapshots.
- `falcor2/editor/scene_shader.py` and `falcor2/rendernodes/pathtracer_pipeline_node.py` - preparation and the
  renderer's update boundary.
- `tests/native/render/test_scene.cpp`, `tests/native/render/test_scene_object.cpp` and
  `tests/python/render/test_scene.py` - lifecycle, camera policy, snapshot retention, requests and binding.
- `tests/native/render/test_animation.cpp` and `tests/python/render/test_animation.py` - ownership rejection,
  removed bindings, empty channels and an in-memory paused-edit example.
- `tests/python/render/test_scene_capabilities.py` - material demand, unused streams and transmission transitions.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Falcor2 design documentation](index.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Geometry deformation](geometry-deformation.md)
- [Material system](material-system.md)
- [Named material outputs](material-aovs.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Render pipeline](render-pipeline.md)
