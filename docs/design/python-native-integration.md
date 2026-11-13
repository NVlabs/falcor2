---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "92340165f77e680093cae7efe73ae163c8a3d96e"
---

(falcor2-design-python-native-integration)=
# Python and native integration

Falcor2's Python package combines native rendering objects, Python orchestration, and SlangPy GPU calls.
Import initializes SlangPy/SGL before Falcor2 injects its native bindings into the public package and registers
shader adapters. Native intrusive references cooperate with Python wrapper references to retain objects across
the language boundary. Reflection preserves property-change callbacks, while shader marshallers translate selected
values into concrete Slang types and cursor data. Interned strings provide a separate form of identity for native
names and source references. These mechanisms let Python compose rendering workflows while native code owns
synchronized scene state and Slang performs GPU execution.

## Purpose and scope

For contributors familiar with Python and C++, this page explains how values cross the package, object, property,
and shader boundaries without losing their lifetime or type contracts. It should help a reader extend a Python
material or trace a native call without assuming that ownership, reflection, and shader specialization are the
same mechanism.

[Scene model](scene-model.md) explains synchronization and scene-object lifetime; [Render pipeline](render-pipeline.md)
explains renderer caches and frame history. This page covers initialization decisions that shape the public API;
binary deployment, exhaustive APIs, and material algorithms belong elsewhere.

## Architecture

### One public package, layered initialization

`falcor2/__init__.py` imports `slangpy` first, establishing SGL types before Falcor2 registers derived bindings.
On Windows, a deployed `falcor2.dll` in the package directory takes precedence; otherwise `.build_dir` selects the
development runtime directory, whose DLL-search handle is retained. Without either location, import fails.
On other platforms, an existing `.build_dir` also selects the runtime directory.

Before loading the extension, the initializer adds `<runtime>/python/falcor2` to the package search path when
present, making configuration-specific generated OpenUSD bindings discoverable. It also prepends an existing
`<runtime>/usd` directory to `FALCOR2_USD_PLUGIN_PATH`. These steps select runtime resources before native
initialization. Detailed packaging is separate from this import boundary.

The extension's `NB_MODULE` entry imports the already-created `falcor2` package and registers classes, functions,
and selected submodules there. `falcor2.falcor2_ext` is the loading name, not a parallel public API namespace.
`falcor::static_init()` initializes SGL before dependency-ordered registration through the `FALCOR_PY_*` functions.
Extension-module cleanup calls `static_shutdown()`, which runs registered callbacks before SGL shutdown.

Only after native registration does Python import `falcor2.extensions`, followed by selected orchestration and
editor helpers. `create_device()` supplies Falcor and SlangPy include paths; resource-path helpers distinguish
checkout and installed layouts. Falcor2 uses SGL devices and SlangPy modules directly.

### Ownership across nanobind and SGL

Falcor2's `Object` and `ref<T>` are aliases of SGL's intrusive reference-counted base and owning pointer.
Before Python association, references increment the native count and the final release destroys the object.
When nanobind associates a wrapper, `set_self_py()` transfers outstanding native references to that wrapper's
Python count. Subsequent `ref<T>` operations retain and release the same Python object through callbacks that
acquire the global interpreter lock (GIL). The `ref<T>` caster converts through the underlying native pointer.

Consequently, a scene retaining a Python-derived material also retains its Python implementation after a local
Python variable disappears. The bridge requires one wrapper association per object. It does not turn borrowed
scene or parent pointers into owning references: retaining an attached object alone does not retain its Scene.
View and iterator bindings use `reference_internal` and `keep_alive` where additional parent relationships apply.

Python-only state needs explicit retention too. Deferred importer material constructors capture a
`PythonObjectRef` to their Python class, and acquire the GIL when constructing it later. Reference changes also
acquire the GIL. These boundaries coordinate lifetime; they do not make arbitrary scene mutation thread-safe.

### Reflection and Python-defined scene objects

Native `reflect()` declarations feed runtime property descriptors, metadata, serialization, and UI inspection.
`NanobindClassReflector` uses the same declarations for Python attributes and preserves `on_change` callbacks in
generated setters. Python assignment therefore follows the notification contract of descriptor-mediated native
writes. Arbitrary direct C++ field assignment does not automatically notify.

Python `reflected_property` descriptors supply their own getters, setters, and callbacks. The `@reflected`
decorator collects class metadata; `PythonClassReflection` and `PythonPropertyDescriptor` adapt it for the
Python-facing `ui.properties_editor` and `PropertyEditor`. That adapter passes a Python-handle context to the
shared native editor.

The native scene inspector and `ReflectedObject` serialization instead use `class_descriptor()` and
`dynamic_properties()`. Python-added descriptors are not automatically merged into this native path by
subclassing Material or using a scene factory. A Python property can work through direct assignment and the Python
editor adapter without becoming a native Material-inspector or serialized property.

Scene factories accept Python subclasses of `Material`, `Geometry`, and `Component`. They invoke `__new__` and
`__init__`, cast the embedded native base, adopt it with `ref<T>`, and then apply supplied construction properties.
Initialization thus precedes scene adoption. Accepting a subclass does not imply full virtual forwarding:
Material forwards lifecycle, property-setting, update, cursor-writing, required-capability, and required-module
hooks; Geometry and Component do not provide comparable Python override coverage.

The Python material sample illustrates the supported path: property callbacks mark native dirty flags, resource
hooks load textures, update prepares the shader module, and cursor methods serialize parameters. Its test compares
direct and scene-dispatched shader evaluations after edits. That example does not establish native inspector
support for its Python-added fields.

### Interned strings and native identity

Native `ustring` is a pointer-sized value referring to canonical immutable bytes, used for names such as
`SourceAssetRef` fields. Equal contents share storage, so token equality compares pointers. Length and content
hash are cached; resetting a token changes its reference without modifying the shared record. Token ordering is
address-based, not lexical, and addresses are not persistent identifiers.

The append-only registry uses atomic lookups and serializes insertion within the selected shard. A miss is checked
again under the writer lock, ensuring competing insertions return one record. Release/acquire publication makes
fully initialized records visible. Records never move, and replaced lookup tables remain allocated until registry
destruction so existing readers can finish without reclamation bookkeeping.

Nanobind exposes these values as Python `str`, without a separate public token class. Shared native storage does
not imply Python object identity. Conversion interns the UTF-8
contents while the Python-owned view is valid; returning a string uses the stored length, preserving Unicode and
embedded NUL characters. Native null-pointer construction yields the canonical empty token, but Python `None`
is rejected. Invalid native UTF-8 cannot be returned as a Python string.

Registry lifetime differs from object ownership: tokens do not reference-count individual records. The registry
is a function-local static, not an explicit `falcor::static_shutdown()` callback. All users must stop before its
destruction; tokens and borrowed string views must not be used afterward. Concurrent interning does not make
concurrent mutation of the same token variable safe.

### SlangPy marshalling and Python orchestration

Importing `falcor2.extensions` registers adapters after their native Python types exist. Native cursor writers
serialize values, while Python adapters select compatible shader types, imports, and call dimensionality:

| Python value | Shader boundary |
| --- | --- |
| Camera or camera uniforms | `CameraMarshall` selects scalar `Camera` or two-dimensional `Ray`/`RaySampler` calls. |
| Material | `MaterialMarshall` resolves `IMaterial` to the value's concrete `slang_type_name`. |
| Reconstruction filter | `ReconstructionFilterMarshall` resolves `IReconstructionFilter` to its concrete filter type. |
| Texture handle | Cursor-writer registration accepts `TextureHandle` and its Slang specializations. |

SlangPy checks explicit Python registrations through the class method-resolution order before native metadata
fallback. Material registration therefore covers native derived classes and Python subclasses. Material and filter
signatures include the concrete Slang type, excluding object identity and parameter values. Parameter edits upload
new data without inherently requiring a new specialization. The shared cursor marshaller supports read-only scalar
inputs; it provides no shader-to-Python writeback.

Python orchestrates scene updates and GPU calls. After `Scene.update()`, `SceneShaderHelper.prepare()` returns a
linked module and its exact immutable requirements snapshot, defaulting to required capabilities. The helper links
the caller's module with utility, scene-render, and selected requirement modules, then binds the prepared scene's
globals on demand. Requirement-generation or consumer-demand changes trigger resolution; source-module and shader
input comparison decide whether to relink. Pipeline-only differences can preserve the module.

Failed helper preparation preserves its previous successful module, snapshot, keys, and bound scene. Consumers use
the returned snapshot for conformances and ray-tracing setup while retaining ownership of output storage and dispatch
specialization. See [renderer reuse boundaries](render-pipeline.md#requirements-modules-and-dispatch-reuse) for
the detailed cache contract.

## Runtime flow

1. Import establishes SlangPy/SGL, runtime search paths, native registration, and Python shader adapters in order.
2. Python creates native objects or supported subclasses. Scene adoption retains the object through the shared
   lifetime bridge; borrowed relationships keep their separate constraints.
3. Property edits invoke the applicable callbacks. Scene synchronization processes dirty resources and shader demand.
4. Python prepares a linked module and requirements snapshot. Marshallers resolve concrete shader types, and cursor
   writers upload values while scene binding supplies synchronized GPU resources.
5. SlangPy dispatches GPU execution; Python continues orchestration using the returned resources.

## Invariants and rationale

Object retention, string interning, reflection metadata, and shader signatures solve different problems. A retained
wrapper preserves Python behavior, an interned token shares immutable native text, a descriptor routes property
access, and a shader signature selects compatible generated code. None substitutes for the others.

Dirty-state propagation must accompany editable shader data. Updating a Python field without its required callback
can leave native state stale even when the wrapper is alive and marshalling succeeds. Likewise, a concrete Slang
type must be visible in the composed module; a Python base class or cursor writer alone cannot define it.

Python owns composition; native code and Slang own synchronized resources and parallel work. Extending that boundary
requires checking the applicable lifetime, callback, reflection-path, shader-type, and cursor contracts together.
The code and tests below demonstrate those crossings without implying that every native base supports every Python
extension mechanism.

## Source landmarks

- `falcor2/__init__.py`, `falcor2/_package_paths.py`, and `falcor2/editor/utils.py` - initialization and resource discovery.
- `src/falcor2_ext/falcor2_ext.cpp`, `src/falcor2_ext/nanobind.h`, and `src/falcor2/falcor2.cpp` - registration and cleanup.
- `src/falcor2/core/object.h`, `external/slangpy/src/sgl/core/object.cpp`,
  `external/slangpy/src/slangpy_ext/core/object.cpp`, and `external/slangpy/src/slangpy_ext/nanobind.h` - lifetime bridge and caster.
- `src/falcor2_ext/render/python_scene_object.h`, `src/falcor2_ext/render/scene.cpp`, and
  `src/falcor2_ext/importers/importer.cpp` - scene adoption, virtual forwarding, and retained Python constructors.
- `src/falcor2_ext/nanobind_reflector.h`, `src/falcor2/core/reflected_object.cpp`,
  `falcor2/reflection/properties.py`, and `src/falcor2_ext/core/reflection/python_class_reflection.cpp` - property paths.
- `src/falcor2_ext/ui/property_editor.cpp` and `src/falcor2/ui/property_editor.cpp` - Python and native instance adaptation.
- `src/falcor2/core/ustring.h`, `src/falcor2/core/ustring.cpp`, `src/falcor2_ext/core/ustring.h`, and
  `src/falcor2/render/source_asset_ref.h` - token semantics, registry publication, and Python conversion.
- `falcor2/extensions/cameramarshall.py`, `falcor2/extensions/materialmarshall.py`,
  `falcor2/extensions/reconstructionfiltermarshall.py`, `falcor2/extensions/cursormarshall.py`,
  `src/falcor2/core/cursor_writer.h`, `external/slangpy/slangpy/bindings/cursor.py`,
  `external/slangpy/slangpy/bindings/typeregistry.py`, and `falcor2/editor/scene_shader.py` - type selection and upload.
- `tests/python/core/test_nanobind_reflector.py`, `tests/python/core/test_python_properties.py`, and
  `tests/python/core/test_python_property_descriptor.py` - callbacks and Python descriptor behavior.
- `tests/python/render/python_material_sample.py`, `tests/python/render/python_material_sample.slang`,
  `tests/python/render/test_materials.py`, and `tests/python/render/test_scene.py` - subclass dispatch and helper reuse.
- `tests/native/core/test_ustring.cpp` and `tests/python/core/test_ustring.py` - canonical identity, concurrent growth, and conversion.

## Related design documents

- [Build and runtime packaging](build-and-runtime-packaging.md) - private dependencies and configuration-specific deployment.
- [Scene model](scene-model.md) - synchronization and attached-object lifetime.
- [Render pipeline](render-pipeline.md) - dispatch preparation, filters, and renderer-owned histories.
- [Scene loading and authoring](scene-loading-and-authoring.md) - importer-to-live-object boundaries.
- [Material system](material-system.md) - material resources and shader implementations.
- [Falcor2 system overview](overview.md) - package, native library, and shader relationships.
