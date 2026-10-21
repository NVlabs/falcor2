---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-09
verified_against: "01b22e34e4715605fdd894a7dcbe736be8030736"
---

(falcor2-design-python-native-integration)=
# Python and native integration

Falcor2 presents one Python package assembled from three cooperating layers. The native `falcor2` shared library
owns rendering state and links against SGL; the nanobind module exposes that implementation by injecting symbols
directly into the existing `falcor2` package; and ordinary Python modules provide render-graph, editor, importer,
and workflow orchestration. SlangPy is imported first so its SGL object types and GPU binding runtime exist before
Falcor2 registers derived classes. SGL's intrusive object model bridges native references to Python wrapper
references, while Falcor-specific SlangPy marshallers translate selected scene objects and handles into concrete
Slang types for GPU calls.

## Purpose and scope

This page explains package initialization, binding registration, cross-language ownership, reflection, Python
subclassing, and SlangPy marshalling. It focuses on the architectural seams contributors encounter when a Python
workflow calls native scene code or a SlangPy kernel consumes a Falcor2 object. Detailed scene synchronization is in
[Scene model](scene-model.md), material dispatch is in [Material system](material-system.md), and render workflow
composition is in [Render pipeline](render-pipeline.md).

This is not an exhaustive Python API reference, packaging guide, or nanobind tutorial. Generated stubs and binary
installation mechanics are source landmarks only where they explain the public package shape.

## Architecture

### One public package, layered initialization

Importing `falcor2` first imports `slangpy`. On Windows, the initializer then distinguishes an installed package,
where `falcor2.dll` sits beside the package, from a development build, where `.build_dir` identifies the directory
added to the DLL search path. It imports `falcor2.falcor2_ext`, followed by `falcor2.extensions`, and finally exposes
selected Python orchestration classes and editor helpers.

The extension's `NB_MODULE` entry point deliberately ignores its own module namespace. It imports the already
created `falcor2` package and registers native classes, enums, functions, and selected submodules into that object.
`falcor::static_init()` runs before registration and `static_shutdown()` is attached to module cleanup. Binding
translation units export registration functions through `FALCOR_PY_DECLARE`, `FALCOR_PY_EXPORT`, and
`FALCOR_PY_IMPORT`; the central entry point calls them in dependency order. The result is one public namespace rather
than a user-facing native module parallel to the Python package.

### Ownership across nanobind and SGL

Falcor2's `Object` is SGL's reference-counted base, and `ref<T>` is its intrusive RAII pointer. Before an object has a
Python wrapper, references are counted in the native object and the final release deletes it. Nanobind registers the
base as an intrusive object. When a wrapper is associated, `set_self_py()` transfers outstanding native ownership
to the Python reference count; later `ref<T>` increments and decrements retain and release that same Python object.
The custom `ref<T>` type caster converts through the underlying pointer while preserving this lifetime bridge.

This means a scene can own a Python-derived native object without maintaining a second lifetime model. Return
policies such as `reference_internal` and `keep_alive` still express view and parent relationships for borrowed
objects and collection iterators. The importer path that stores a Python material class for later construction uses
an explicit `PythonObjectRef`; it acquires the global interpreter lock (GIL) for reference changes and later Python
execution.

### Reflection and Python-defined scene objects

Native reflected classes declare properties once through a templated `reflect()` method. The C++ type registry uses
that declaration for `Properties`, metadata, serialization, and UI introspection. `NanobindClassReflector` consumes
the same declaration to create Python attributes, filters out native-only metadata, and wraps setters so `on_change`
callbacks still mark native state dirty when assignment originates in Python.

`ReflectedObject` also exposes a mapping-style property surface and per-instance dynamic properties. In the other
direction, Python's `reflected_property` descriptors are collected into class metadata and adapted by
`PythonPropertyDescriptor`, allowing native property tooling to inspect and edit Python-defined values.

Scene factories accept Python subclasses of the bound `Material`, `Geometry`, and `Component` bases. They
distinguish registered native classes from Python-derived classes, run the Python class's `__new__` and `__init__`,
cast the embedded native base, add a `ref<T>` to the scene, and then apply construction properties. `Material` has
the full trampoline contract for lifecycle, update, cursor, and module hooks; the other accepted bases currently
expose less virtual behavior. The Python material sample demonstrates reflected dirty properties, texture loading,
a required Slang module, update, and cursor serialization.

### SlangPy marshalling and Python orchestration

Falcor2 passes SGL device objects, modules, tensors, shader cursors, and reflected layouts directly through the
SlangPy API rather than wrapping a second GPU abstraction. Importing `falcor2.extensions` registers marshallers only
after native types exist. `CameraMarshall` maps camera objects to `Camera`, `Ray`, or `RaySampler` dimensionality and
imports Falcor's render declarations. `MaterialMarshall` uses each value's concrete `slang_type_name`, contributes
that type to the SlangPy cache signature, and resolves an `IMaterial` parameter to the concrete generated or native
type. The base-type registration also covers Python material subclasses through Python method-resolution order.
`TextureHandle` has a generic cursor-writer registration that accepts its Slang specializations.

Ordinary Python remains the composition layer. Render nodes implement callable execution and nest through a
context-local stack; containers describe and convert tensor layouts; editor helpers create devices with Falcor and
SlangPy include paths. `SceneShaderHelper` links a caller module with the scene render module and current requirement
modules, caches it by source and requirement generation, then lets the native scene bind its globals. Heavy scene
updates, resource management, acceleration structures, and rendering kernels remain native or GPU work even when
Python determines their order.

## Runtime flow

```text
import falcor2
     |
     +--> import slangpy and initialize SGL Python types
     +--> locate native runtime on Windows
     +--> load falcor2_ext
     |       |
     |       +--> native static initialization
     |       +--> inject bindings into package namespace
     |
     +--> register Falcor SlangPy marshallers
     +--> expose Python orchestration modules
                         |
                         v
             create/edit native scene objects
                         |
             Scene.update() and requirements
                         |
             linked SlangPy module + scene.bind()
                         |
                         v
                    GPU execution
```

1. Package import establishes SlangPy/SGL before loading Falcor2's derived native bindings and Python adapters.
2. Python creates SGL devices and native scene objects, or subclasses supported scene bases. Intrusive references
   and Python references describe one coordinated object lifetime.
3. Reflected property writes and virtual hooks cross the binding boundary while preserving dirty callbacks and the
   scene lifecycle.
4. Python orchestration updates the scene and asks helpers or render nodes for a module specialized to current scene
   requirements.
5. SlangPy marshallers resolve Python values to concrete Slang types and cursor data; native scene binding supplies
   synchronized GPU resources before dispatch.

## Invariants and rationale

`falcor2.falcor2_ext` is an implementation-loading name, not a second public API hierarchy. Native binding code must
register into the package object after SlangPy is available, and Python adapters that refer to bound types must load
after the extension. Registration order is therefore part of initialization correctness.

An `Object` has one active ownership regime. Before Python association it uses native intrusive counts; after
association its `ref<T>` operations manipulate the wrapper's Python count. Code must not install a second Python
wrapper for the same object. Any native state that stores Python-only objects or invokes Python outside the immediate
call must hold an explicit reference and acquire the GIL at the boundary.

Reflection metadata is behavioral. A Python attribute generated from native reflection must trigger the same
`on_change` action as a C++ property write, or scene dirtiness diverges by language. Python-defined reflected
properties likewise need a stable descriptor view for native serialization and UI code. Construction properties are
applied only after a Python-derived scene object has entered native scene ownership.

SlangPy cache identity must include value-dependent concrete types. Two `Material` values can implement different
Slang structs even though Python sees the same base class, so the material marshaller's signature includes
`slang_type_name`. Scene requirement generations separately invalidate linked modules when the set of modules or
type conformances changes.

Python owns policy and composition; native and Slang code own synchronized data and parallel execution. The boundary
is intentionally permeable for research extensions, but extending it requires all applicable parts of the
contract: ownership, reflection or explicit bindings, dirty-state propagation, shader type identity, cursor writing,
and tests of the round trip.

## Source landmarks

- `falcor2/__init__.py` and `falcor2/_package_paths.py` - package import order, development DLL discovery, exported
  Python layer, runtime resources, and Slang include paths.
- `src/falcor2_ext/falcor2_ext.cpp`, `src/falcor2_ext/nanobind.h`, and
  `src/falcor2_ext/CMakeLists.txt` - package injection, registration order, extension linkage, output placement, and
  generated stubs.
- `external/slangpy/src/sgl/core/object.h`, `external/slangpy/src/sgl/core/object.cpp`, and
  `external/slangpy/src/slangpy_ext/core/object.cpp` - intrusive ownership, transfer to Python reference counting,
  and nanobind base registration.
- `external/slangpy/src/slangpy_ext/nanobind.h` and `src/falcor2/core/object.h` - `ref<T>` conversion and Falcor's
  shared SGL object aliases.
- `src/falcor2_ext/nanobind_reflector.h`, `src/falcor2/core/reflection.h`, and
  `src/falcor2/core/reflected_object.cpp` - shared native property declarations, runtime descriptors, serialization,
  dynamic properties, and dirty callbacks.
- `falcor2/reflection/properties.py` and `src/falcor2_ext/core/reflection/` - Python property descriptors and their
  native reflection adapters.
- `src/falcor2_ext/render/python_scene_object.h` and `src/falcor2_ext/render/scene.cpp` - Python subclass checks,
  trampolines, scene adoption, virtual callbacks, and collection lifetime policies.
- `falcor2/extensions/cameramarshall.py`, `falcor2/extensions/materialmarshall.py`, and
  `falcor2/extensions/cursormarshall.py` - Falcor-specific SlangPy type resolution and cursor writing.
- `falcor2/rendergraph/render_node.py`, `falcor2/editor/scene_shader.py`, and `falcor2/editor/utils.py` - Python
  orchestration, device creation, requirement-aware linking, and native scene binding.
- `tests/python/core/test_nanobind_reflector.py`, `tests/python/render/python_material_sample.py`,
  `tests/python/render/python_material_sample.slang`, and `tests/python/render/test_materials.py` - property callback,
  Python subclass, resource, cursor, type-conformance, and SlangPy dispatch behavior.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Render pipeline](render-pipeline.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Falcor2 design documentation](index.md)
