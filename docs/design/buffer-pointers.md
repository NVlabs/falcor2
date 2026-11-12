---
status: draft
audience: Falcor2 practitioners, contributors and coding agents
last_verified: 2026-11-11
verified_against: "5dc69ce8590615db2434f1c592287c89d0c27c5f"
---

(falcor2-design-buffer-pointers)=
# Buffer pointers and indirect GPU access

Falcor2's `BufferPtr` and `RWBufferPtr` represent non-owning locations within GPU buffers. They connect host-side
resource allocation to shader-side byte access and typed indexing through one eight-byte representation. The
wrapper uses native pointers on Vulkan and CUDA, and a descriptor index plus byte offset on D3D12. Allocation
ownership, synchronization, and data layout remain the caller's responsibilities. This page explains those
boundaries, the distinction between element stride and explicit access alignment, and the constraints on casts and
comparisons. Geometry and material consumers use this pointer family; the former Falcor buffer-handle APIs have
been removed.

## Purpose and scope

This page explains how a buffer location crosses the host/shader boundary without becoming an owning resource or
a bounded view. `BufferPtr` is the read-only interface; `RWBufferPtr` permits reads and writes. Read-only describes
access through that value, not immutable storage: another writable pointer can change the data.

The C++ host types are concrete, untyped aliases named `BufferPtr` and `RWBufferPtr`; their access-mode template
is an implementation detail. Slang retains generic `BufferPtr<T>` and `RWBufferPtr<T>` types. Use the bare names
for the default untyped forms throughout host and shader code; supply generic arguments for typed pointers or
non-default access or load alignment. Slang constructor calls require specialization, so spell untyped
constructors as `BufferPtr<void>(bits)` or `RWBufferPtr<void>(bits)`. Shaders can form typed views after binding.

The API comments define individual operations and immediate usage restrictions. Tests and linked upstream issues
hold compiler reproductions. `BufferPtr` and `RWBufferPtr` replace Falcor's `BufferHandle`, `RWBufferHandle`,
`BufferOffsetPair`, and `BufferView`, including the `to_handle` and `to_rw_handle` host factories. SGL's
`BufferOffsetPair` resource ranges and tinygltf's `BufferView` asset records are unrelated and remain in use.

In Slang, import `falcor2.utils` to use `BufferPtr` and `RWBufferPtr`. Python exposes the pointer types and
the `to_buffer_ptr` and `to_rw_buffer_ptr` factories directly through `falcor2`. In shaders, use subscripts
for element access and `load`/`store` for byte-oriented access.

## Architecture

### Resource owner and pointer value

An SGL buffer owns the allocation and exposes its native address or descriptors. The host factories
`to_buffer_ptr` and `to_rw_buffer_ptr` resolve a location in that allocation. The resulting value contains only the
encoded location: it retains neither the buffer nor its device, and records no bounds. Replacing the underlying
GPU allocation invalidates pointers into the old allocation, even if a higher-level container remains alive.

The owner must retain resources through GPU completion and arrange resource states, barriers, and synchronization.
Passing a pointer value through a shader cursor does not establish those responsibilities for the referenced
allocation. Shader code must address live storage with appropriate access permissions.

### Backend representation

| Backend | Eight-byte payload | Meaning of access mode |
| --- | --- | --- |
| Vulkan / CUDA | Native pointer storage | Shader access qualifier |
| D3D12 | Low 32 bits: descriptor index; high 32 bits: byte offset | Selects an SRV or UAV descriptor |

A D3D12 shader resource view (SRV) supplies read-only access; an unordered access view (UAV) supplies read-write
access. Every raw load/store adds its byte displacement to the packed base byte offset. Arithmetic changes the
offset while retaining the descriptor. This lets a pointer address an interior allocation in a larger buffer.

Backend-specific bits are transport values, not portable addresses or serialized asset data. The all-ones value
is the shared null sentinel. Zero-filled pointer storage is not a portable null representation: descriptor zero
with byte offset zero is valid on D3D12. Default construction and shader `nullptr` conversion produce the sentinel.

Metal support is disabled. Emission tests demonstrate prospective MSL code generation, but Apple compilation,
resource residency integration, and GPU behavior have not been validated. See
[deferred Metal integration](#deferred-metal-integration); emission alone does not establish runtime support.

## Runtime flow

1. The owner allocates a buffer with the required access usage and fills it using the layout expected by its
   consumer backend.
2. A host factory validates the byte offset, requested initial alignment, and representability of the location.
   It creates an untyped read-only or writable pointer. An offset exactly at the buffer end is allowed for
   arithmetic, but not for memory access.
3. Cursor writing accepts only the corresponding `BufferPtr<void, access, 0>` shader specialization. Python
   exposes the same values and factories. Typed or aligned views are formed in the shader after binding.
4. Shader code uses `as<T>()` to reinterpret a location, indexes elements, or accesses a value at a byte offset.
   Casts do not transform data or verify its layout. The owner keeps the resources valid until GPU use completes.

The shader interface intentionally has no runtime bounds checks. Host validation establishes an initial location;
it cannot validate subsequent shader arithmetic, resource lifetime, or layout compatibility.

### Production consumers

Geometry tables and LSS bounds work records carry buffer pointers. Vertex and index readers form typed views with
`as<T>()` and use element indices. Table indices, pooled allocation offsets, ownership and resource-state handling
remain with the geometry systems. The host `BufferHandleTable` retains its name but stores BufferPtr payloads.

Deformation work records carry direct source `BufferPtr` values and current/previous `RWBufferPtr` destinations.
Preparation resolves these locations after pool updates; the updater batches the records while source owners
and output collections retain resources and declare states. [Geometry deformation](geometry-deformation.md)
describes this boundary.

Shared glTF, SpecGloss and spilled MaterialX data use allocation pointers in place of separate buffer/offset pairs.
MDL's read-only data, argument-block and texture-table offsets are relative to its allocation pointer, with
`ro_data_offset` zero. Material storage uploads and grows before material records are rewritten, so those records
receive pointers to the current allocation. The outer `MaterialData` slot remains 128 bytes.

Two local access forms preserve efficient generated code:

- [OpenPBR attribute loads](../../slang/falcor2/render/materials/openpbr_material.slang) reconstruct a D3D pointer
  with a constant zero byte offset while retaining its descriptor index. The host allocates a dedicated Attributes
  buffer and uploads at offset zero. Exposing that constant lets D3D combine small-field loads. If attributes move
  into pooled or interior storage, remove this shortcut or establish an equivalent zero-offset view first.
- [LSS bounds stores](../../slang/falcor2/render/kernels/lss_kernels.slang) reinterpret each AABB as `float2[3]`
  at the store. The AABB keeps its natural min/max fields and 24-byte layout; the temporary view gives CUDA three
  paired 64-bit stores. The host checks eight-byte base alignment, which the 24-byte stride preserves. This does
  not introduce packed AABB fields or a pointer-extent policy; general pointer arithmetic retains its full width.

## Invariants and rationale

### Element stride and access alignment are separate

Indexing and pointer arithmetic share `element_offset`: `p[i]` accesses the location selected by `p + i`, and
incrementing advances by one element stride. Stride includes required padding; it is not generally `sizeof(T)`.
CUDA delegates stepping to native typed `Ptr`, whose aggregate layout may contain backend-specific padding.
Vulkan and D3D12 round default-layout size up to its alignment and perform byte-address arithmetic explicitly.
The Vulkan implementation avoids the native negative-offset lowering problem tracked in
[Slang #14305](https://github.com/shader-slang/slang/issues/14305); this is not a prohibition on negative BufferPtr
indices that remain within the allocation.

Arithmetic requires a non-null pointer and a result within the same allocation or one-past-end, without address
overflow. On D3D12 the byte offset must remain representable in 32 bits. One-past-end values cannot be accessed;
shader arithmetic does not check these conditions.

Three alignment interfaces serve different purposes:

| Interface | Responsibility |
| --- | --- |
| Host factory `alignment` | Checks the initial address against 4, 8 or 16 bytes; stores no alignment metadata |
| Shader type `load_alignment` | Zero uses plain loads/stores; nonzero supplies an explicit operand for subsequent accesses |
| Explicit aligned operation | Supplies its own operand, overriding the type default for that operation |

Aligned operations support operands of 4, 8 or 16 bytes. The pointer type accepts only 0, 4, 8 or 16;
zero selects plain access. `load_alignment` applies to stores as well as loads.
It neither changes stride nor certifies an address. Arithmetic preserves it, so the caller must ensure each
effective address satisfies the promise, not just the allocation base. Every access requires a complete value
whose nonzero size is divisible by four, and an address aligned to at least four bytes and the type's natural
alignment. Plain access preserves these requirements.

### Layout compatibility must include fields

The producer and consumer must agree on field offsets as well as array stride. A successful cast, matching total
size, or aligned base cannot establish that agreement. On D3D12, BufferPtr uses raw `ByteAddressBuffer` access.
Slang can give nested records with implicit tail padding different field offsets in raw access, structured
access, and reflection. Even equal outer strides can hide the disagreement. Such interchange requires explicit
layout verification; unrestricted nested-record interoperability is not supported. The independent regressions
and [Slang #14311](https://github.com/shader-slang/slang/issues/14311) establish the affected cases.

D3D12 double stores are also unsupported pending
[Slang #14259](https://github.com/shader-slang/slang/issues/14259): constant double stores can silently corrupt
values. The API does not enforce this restriction recursively through aggregate types. Integer 64-bit values and
pointer payloads are covered independently; double deferral is not a blanket ban on eight-byte data.

### Value access and conversions

Subscripting returns or assigns values rather than exposing a native reference. Updating one field can read and
write the entire element, so concurrent writers to different fields are not independent. These are ordinary
non-atomic operations. This distinction matters when substituting BufferPtr for Slang's native `Ptr`.

Explicit native-backend casts may change access mode; a writable result still requires writable memory. D3D12
rejects access-mode changes because reinterpreting an SRV index cannot produce the required UAV, or vice versa.
Obtain the other access mode from the host factory instead.

Equality accepts matching pointees or typed/void pairs and ignores the alignment operand. D3D12 additionally
requires matching access modes and compares encoded view identity; different descriptors for the same allocation
are not canonicalized. Native backends allow equality across access modes. Ordering requires matching pointee
and access types and is meaningful only within the same allocation and, on D3D12, the same descriptor view.

Use `nullptr` for null comparisons and `bits()` for deliberate encoding comparisons. Numeric comparisons are
rejected. Boolean and custom bool-convertible operands can still trigger Slang's implicit-bool fallback rather
than pointer equality, as tracked in [Slang #14268](https://github.com/shader-slang/slang/issues/14268). The wrapper
retains boolean conversion for ordinary conditions; callers must restrict comparisons to compatible pointers or
`nullptr`. `is_valid()` is only a sentinel test, not a lifetime or bounds check.

### Deferred Metal integration

Both the host factory and production shader reject Metal. Guarded nightly tests stage a temporary shader copy
with native pointer storage enabled and emit Metal Shading Language (MSL) for typed and aligned access,
vector/matrix layouts, pointer tables, casts and nested records. These tests neither invoke Apple's compiler
nor execute a Metal GPU. Successful emission establishes neither a runtime ABI nor residency, lifetime or
synchronization for indirectly reached resources. The all-ones null contract is unchanged.

## Source landmarks

- [Shader API](../../slang/falcor2/utils/buffer_ptr.slang): storage, casts, loads/stores, subscripts and operators.
- [Host value and factories](../../src/falcor2/utils/buffer_ptr.h): encoding, validation and cursor contract.
- [Cursor registration](../../src/falcor2/utils/buffer_ptr.cpp) and
  [Python bindings](../../src/falcor2_ext/utils/buffer_ptr.cpp): host/shader integration.
- [Geometry table](../../src/falcor2/utils/buffer_handle_table.h),
  [deformation work](../../src/falcor2/render/geometry/vertex_cache_deformer.h) and
  [material storage updates](../../src/falcor2/render/material_system.cpp): production pointer publication and
  resource-state ownership.
- [Regular CI tests](../../tests/python/utils/test_buffer_ptr.py) and
  [shader cases](../../tests/python/utils/test_buffer_ptr.slang): host construction, cursor and functional
  bindings, basic reads/writes, byte offsets and nulls. These run on each selected backend.
- [Access regressions](../../tests/python/utils/test_buffer_ptr_access.py): divergent pointers and writable
  vector access.
- [Nightly tests](../../tests/python/utils/test_buffer_ptr_nightly.py) and
  [shader cases](../../tests/python/utils/test_buffer_ptr_nightly.slang): arithmetic and cast sweeps, alignment,
  aggregate layout, aliasing, pointer traversal and emitted code. The module-level `nightly` marker defers these
  tests unless `--nightly` (or `--all`) is enabled. Invalid-code checks and independent compiler reproductions
  compile isolated source strings. Strict expected failures link to the compiler issues. Metal emission uses
  guarded feasibility cases without a Metal runtime. Backend selection still applies in nightly runs.
- [Native tests](../../tests/native/utils/test_buffer_ptr.cpp): encoding, factory validation and cursor examples.

## Related design documents

- [Python and native integration](python-native-integration.md): binding and package boundaries.
- [Geometry and ray tracing](geometry-and-ray-tracing.md): existing buffer-backed geometry consumers.
- [Geometry deformation](geometry-deformation.md): direct source and destination pointers in batched updates.
- [Design index](index.md): architecture navigation for contributors and coding agents.
