---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-08
verified_against: "1bcd66bf773572a9dbb6ed68d7f315551f9bdc67"
---

(falcor2-design-scene-model)=
# Scene model

The Falcor2 scene model is the native, scene-owned representation that connects editable objects to renderer-facing
GPU state. A `Scene` owns typed collections of materials, geometries, animations, entities, and components; tracks
pending changes with per-object dirty flags; and applies them at the explicit `Scene.update()` boundary. Entities
form the transform hierarchy, components attach renderable behavior, and scene systems translate those objects into
material, light, emissive, and geometry state. The scene also publishes the Slang modules, type conformances, and
pipeline requirements that renderers must use when compiling and binding shaders.

## Purpose and scope

This page explains the live scene's ownership model, entity/component graph, deferred object lifecycle, dirty-state
protocol, update ordering, shader requirements, and binding boundary. It covers behavior shared by scenes regardless
of how they were populated.

Importer staging and callbacks belong in [Scene loading and authoring](scene-loading-and-authoring.md). Geometry
storage, instances,
acceleration structures, and hit groups belong in [Geometry and ray tracing](geometry-and-ray-tracing.md). Material
records and opacity logic belong in [Material system](material-system.md); light sampling belongs in
[Lighting and sampling](lighting-and-sampling.md); renderer and presentation composition belongs in
[Render pipeline](render-pipeline.md).

## Architecture

`Scene` is both owner and coordinator. It holds the SGL device, immutable live-scene configuration, the shared
`falcor2.render` Slang module, and five owning `SceneObjectCollection` instances. Read-oriented collection views
expose indexed access, iteration, and name/type searches without exposing ownership operations. Registered factories
create concrete materials, geometries, and components by C++ type or public type name; Python bindings expose the
same creation and collection model. Adding an object assigns its scene and current collection index, chooses a
default name when needed, and records the shared `added` dirty flag before its creation callback runs.

Every collection member derives from `SceneObject`. The base class supplies scene membership, a collection index,
a name, validity state, common lifecycle callbacks, and the shared `resources`, `added`, and `removed` dirty
bits. Concrete object families add lower dirty bits for their own synchronization needs. The collection stores flags
beside the objects and maintains an atomic OR of all pending flags, allowing an update pass to skip an entire
collection when no relevant work exists.

An `Entity` is a transform-bearing node in the scene graph. Its transform is local to an optional parent, while
`world_from_object_matrix()` composes and caches the parent chain. Changing or reparenting an entity invalidates
that cache for the entity and descendants, marks their transforms dirty, and notifies attached components. Components
are created through an entity, are referenced by that entity, and also appear in the scene-wide component collection.
Camera, geometry-instance, light, and transform-animation types use this attachment point to add behavior without
turning `Entity` into a union of renderer features.

The scene owns aggregate systems beside the object graph. `TextureManager` services pending texture resources;
`MaterialSystem` and `LightSystem` consume their relevant collections; `RenderScene` maintains the low-level
render representation; and `EmissiveGeometrySystem` derives emissive-light data from geometry, material, entity,
and component state. These systems are updated centrally rather than independently by renderers, so one scene update
produces a mutually consistent host/device snapshot.

## Runtime flow

Mutation and rendering are separated by an explicit synchronization boundary:

```text
 create, edit, reparent, remove, or change animation time
                       |
                       v
            collection dirty flags
                       |
                       v
                  Scene.update()
           lifecycle -> systems -> transforms
           -> requirements -> render state
                       |
                       v
         linked shaders and bound g_scene data
```

1. Creation makes an object visible in its typed collection immediately, but `on_add_to_scene()` and resource
   loading wait for the next update. `remove()` makes the object invalid immediately and marks it for removal;
   owning storage and collection compaction also wait for the update. Removing an entity recursively marks its child
   entities and components for removal.
2. The update first repairs references to invalid objects and chooses a replacement active camera when necessary.
   If no active camera exists, the first valid newly added camera is selected. It then calls add/remove lifecycle
   hooks and asks new or resource-dirty objects to load resources.
3. The texture manager runs before animation, material, and light updates. Animation channels may change entity
   transforms during this pass. Geometry render state, entity transforms, and component render state are then
   synchronized in that order, so animation-driven transform changes reach the same update's renderer-facing state.
4. The scene rebuilds a candidate `SceneRequirements` from its settings module, material modules and conformances,
   light conformances, enabled capabilities, and ray-tracing setup fields. Changes to that value or the scene's
   required capabilities advance `requirements_generation` and report the `requirements` update flag.
   The default `scene.requirements` enables all capabilities; `resolve_requirements()` selects capabilities
   for a consumer without changing that default.
5. `RenderScene` and emissive geometry update after requirements are known. Removed objects are destroyed, the
   typed collections are compacted, all dirty flags are reset, and unused ref-counted scene globals are collected.
   The scene stores the resulting `SceneUpdateFlags` and emits its `updated` signal; `update_generation`
   advances only when those result flags are non-empty. After the pass returns, `update()` submits the command
   encoder if any subsystem requested one.

Renderers must call `Scene.update()` before consuming scene state. `PathTracerPipeline` does this before resolving
the active camera and uses `update_generation` to invalidate accumulation. `SceneShaderHelper` caches a renderer
module linked with common utilities, `scene.render_module`, and the modules in the resolved requirements. It compares
shader inputs to reuse compatible modules. Ray-tracing dispatch setup also compares the full requirements,
including geometry and pipeline fields. Dispatch setup applies the current type conformances and uses a
write callback that calls `Scene.bind()`. Binding writes selected material, light, emissive-geometry, and render
scene systems under the shader's `g_scene` parameter block, then binds any additional `SceneGlobals` at global
scope.

## Invariants and rationale

Scene objects have one owning scene and one typed collection. Their raw collection index is an implementation handle,
not stable identity: compaction preserves surviving object order but rewrites indices. Code must not retain an index
across an update that can remove objects. A removed object is invalid as soon as `remove()` returns even though its
storage remains until update; callers and lifecycle hooks must check validity rather than treating collection presence
as liveness.

The entity hierarchy is expected to be an acyclic forest whose members belong to the same scene. `set_parent()`
updates raw parent/child relationships and recursively propagates transform invalidation; it does not validate cycles
or cross-scene parenting. Callers must preserve those preconditions. Removing a parent recursively removes its
descendants and components, while the update's reference-cleanup phase removes stale raw pointers before compaction.

Dirty flags are a coalescing change protocol, not a history. Multiple edits before one update combine their bits, and
the flags are cleared only after all consumers have run. New object types must mark the flag expected by the system
that owns their renderer-facing representation; direct state mutation that bypasses dirty marking can leave host and
device views inconsistent.

Update ordering is part of the contract. Invalid references are repaired before callbacks and systems inspect them;
animation precedes transform upload; scene requirements are settled before emissive shaders inspect them; and
compaction occurs only after every consumer has processed removal state. `requirements_generation` is narrower
than `update_generation`: the former indicates changes that may affect requirements, while the latter means the
update reported some scene change. Compare resolved requirements before rebuilding shaders or pipelines.

## Source landmarks

- `src/falcor2/render/scene.h` and `src/falcor2/render/scene.cpp` - scene ownership, update ordering,
  requirements, generations, active-camera policy, and shader binding.
- `src/falcor2/render/scene_object.h` and `src/falcor2/render/scene_object.cpp` - object lifecycle, typed
  collections, dirty aggregation, views, and compaction.
- `src/falcor2/render/entity.h`, `src/falcor2/render/entity.cpp`, and `src/falcor2/render/component.h` - entity
  hierarchy, transform propagation, component attachment, and recursive removal.
- `src/falcor2/render/scene_system.h`, `src/falcor2/render/scene_update.h`, and
  `src/falcor2/render/scene_requirements.h` - system interface, update context and result flags, and shader
  requirement record.
- `src/falcor2_ext/render/scene.cpp` - Python exposure of collections, factories, entities, components, updates,
  requirements, and binding.
- `falcor2/editor/scene_shader.py` and `falcor2/rendernodes/pathtracer_pipeline_node.py` - requirement-aware
  shader linking and the renderer's update boundary.
- `tests/native/render/test_scene.cpp`, `tests/native/render/test_scene_object.cpp`, and
  `tests/python/render/test_scene.py` - dirty-state, hierarchy, lifecycle, compaction, requirements, and binding
  behavior.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Falcor2 design documentation](index.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Render pipeline](render-pipeline.md)
