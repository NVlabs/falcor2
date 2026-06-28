// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/shared_scene_types.h"
#include "falcor2/render/hit_group_policy.h"

#include "falcor2/core/object.h"
#include "falcor2/core/enum.h"

#include "falcor2/utils/managed_vector.h"
#include "falcor2/utils/offset_allocator.h"

#include <sgl/device/shader_cursor.h>
#include <sgl/device/raytracing.h>

#include <span>
#include <vector>
#include <utility>
#include <memory>

#include <cstdint>

namespace falcor {

// TODO(scene):
// This implements the low-level geometry handling of the scene.
// The design of this needs more discussion. It makes it harder to expose
// geometry handling to Python for example. But in my opinion we need a lower
// level system for handling geometries and not put all that logic in the higher
// level scene object classes.

/// Enable handle debug information.
/// This allows for debuggers to inspect the handle's underlying object.
#define FALCOR_RENDER_HANDLE_DEBUG 1

enum class RenderObjectType {
    triangle_geometry,
    lss_geometry,
    geometry_group,
    geometry_instance,
    transform,
};

class RenderHandle;

template<typename T, RenderObjectType Type>
class RenderObjectHandle {
public:
    static constexpr RenderObjectType TYPE = Type;
    using Index = uint32_t;
    static constexpr Index INVALID_INDEX = std::numeric_limits<Index>::max();
#if FALCOR_RENDER_HANDLE_DEBUG
    struct DebugInfo {
        T* self;
    };
#endif

    RenderObjectHandle() = default;
    explicit RenderObjectHandle(Index index)
        : m_index(index)
    {
    }
#if FALCOR_RENDER_HANDLE_DEBUG
    explicit RenderObjectHandle(Index index, const DebugInfo* debug_info)
        : m_index(index)
        , m_debug_info(debug_info)
    {
    }
#endif
    explicit RenderObjectHandle(RenderHandle handle);
    RenderObjectHandle(const RenderObjectHandle&) = default;
    RenderObjectHandle(RenderObjectHandle&&) = default;
    RenderObjectHandle& operator=(const RenderObjectHandle&) = default;
    RenderObjectHandle& operator=(RenderObjectHandle&&) = default;
    ~RenderObjectHandle() = default;

    bool is_valid() const { return m_index != INVALID_INDEX; }

    RenderObjectType type() const { return TYPE; }
    Index index() const { return m_index; }

    operator RenderHandle() const;

    operator bool() const { return is_valid(); }

    bool operator==(const RenderObjectHandle& other) const { return m_index == other.m_index; }
    bool operator!=(const RenderObjectHandle& other) const { return m_index != other.m_index; }

private:
    Index m_index{INVALID_INDEX};
#if FALCOR_RENDER_HANDLE_DEBUG
    const DebugInfo* m_debug_info{nullptr};
#endif
};

class RenderHandle {
public:
    using Index = uint32_t;
    static constexpr Index INVALID_INDEX = std::numeric_limits<Index>::max();

    template<typename T, RenderObjectType Type>
    RenderHandle(RenderObjectHandle<T, Type> handle)
        : m_type(Type)
        , m_index(handle.index())
    {
    }
    RenderHandle(const RenderHandle&) = default;
    RenderHandle(RenderHandle&&) = default;
    RenderHandle& operator=(const RenderHandle&) = default;
    RenderHandle& operator=(RenderHandle&&) = default;
    ~RenderHandle() = default;

    bool is_valid() const { return m_index != INVALID_INDEX; }

    RenderObjectType type() const { return m_type; }
    Index index() const { return m_index; }

    operator bool() const { return is_valid(); }

    bool operator==(const RenderHandle& other) const { return m_type == other.m_type && m_index == other.m_index; }
    bool operator!=(const RenderHandle& other) const { return m_type != other.m_type || m_index != other.m_index; }

private:
    RenderObjectType m_type;
    Index m_index;
};

template<typename T, RenderObjectType Type>
RenderObjectHandle<T, Type>::RenderObjectHandle(RenderHandle handle)
{
    FALCOR_CHECK(handle.type() == Type, "RenderHandle type does not match RenderObjectHandle type");
    m_index = handle.index();
}

template<typename T, RenderObjectType Type>
RenderObjectHandle<T, Type>::operator RenderHandle() const
{
    return RenderHandle(*this);
}

template<typename T, typename HandleT, typename DirtyFlagsT = uint32_t>
class RenderObjectPool {
public:
    using Handle = HandleT;
    using DirtyFlags = DirtyFlagsT;
    using Index = Handle::Index;
#if FALCOR_RENDER_HANDLE_DEBUG
    using DebugInfo = typename Handle::DebugInfo;
#endif

    RenderObjectPool() = default;
    ~RenderObjectPool() = default;

    size_t count() const { return m_next_index - m_free_indices.size(); }

    size_t capacity() const { return m_objects.size(); }

    Handle allocate()
    {
        Index index;
        if (m_free_indices.empty()) {
            index = m_next_index++;
        } else {
            index = m_free_indices.back();
            m_free_indices.pop_back();
        }
        if (index >= m_objects.size()) {
#if FALCOR_RENDER_HANDLE_DEBUG
            size_t old_capacity = m_objects.capacity();
#endif
            m_objects.resize(index + 1);
            m_allocated.resize(index + 1);
            m_dirty_flags.resize(index + 1);
#if FALCOR_RENDER_HANDLE_DEBUG
            m_debug_infos.resize(index + 1);
            m_debug_infos[index].reset(new DebugInfo());
            if (m_objects.capacity() != old_capacity) {
                for (size_t i = 0; i < m_objects.size(); ++i)
                    m_debug_infos[i]->self = &m_objects[i];
            }
#endif
        }
        m_objects[index] = {};
        m_allocated[index] = true;
        m_dirty_flags[index] = DirtyFlags{0};
#if FALCOR_RENDER_HANDLE_DEBUG
        return Handle{index, m_debug_infos[index].get()};
#else
        return Handle{index};
#endif
    }

    void free(Handle handle)
    {
        Index index = handle.index();
        m_allocated[index] = false;
        m_free_indices.push_back(index);
    }

    const T& operator[](Handle handle) const
    {
        FALCOR_CHECK(handle.is_valid(), "Invalid handle");
        return m_objects[handle.index()];
    }

    T& operator[](Handle handle)
    {
        FALCOR_CHECK(handle.is_valid(), "Invalid handle");
        return m_objects[handle.index()];
    }

    void mark_dirty(Handle handle, DirtyFlags flags)
    {
        FALCOR_CHECK(handle.is_valid(), "Invalid handle");
        if constexpr (std::is_same_v<DirtyFlags, bool>) {
            m_dirty_flags[handle.index()] = m_dirty_flags[handle.index()] | flags;
        } else {
            m_dirty_flags[handle.index()] |= flags;
        }
        m_combined_dirty_flags |= flags;
    }

    const std::vector<DirtyFlags>& dirty_flags() const { return m_dirty_flags; }

    DirtyFlags combined_dirty_flags() const { return m_combined_dirty_flags; }

    bool is_dirty() const { return m_combined_dirty_flags != DirtyFlags{0}; }

    bool is_dirty(Handle handle) const
    {
        FALCOR_CHECK(handle.is_valid(), "Invalid handle");
        return m_dirty_flags[handle.index()] != DirtyFlags{0};
    }

    void reset_dirty_flags()
    {
        if (is_dirty()) {
            std::fill(m_dirty_flags.begin(), m_dirty_flags.end(), DirtyFlags{0});
            m_combined_dirty_flags = DirtyFlags{0};
        }
    }

    // Iterators skipping free slots
    class Iterator {
    public:
        Iterator(RenderObjectPool& pool, Index index)
            : m_pool(pool)
            , m_index(index)
        {
            skip_free_indices();
        }
        Iterator& operator++()
        {
            ++m_index;
            skip_free_indices();
            return *this;
        }
        bool operator!=(const Iterator& other) const { return m_index != other.m_index; }
        std::pair<Handle, T&> operator*() { return {Handle{m_index}, m_pool.m_objects[m_index]}; }
        std::pair<Handle, const T&> operator*() const { return {Handle{m_index}, m_pool.m_objects[m_index]}; }

    private:
        void skip_free_indices()
        {
            while (m_index < m_pool.m_allocated.size() && !m_pool.m_allocated[m_index]) {
                ++m_index;
            }
        }

        RenderObjectPool& m_pool;
        Index m_index;
        const std::vector<bool>& m_allocated = m_pool.m_allocated;
    };

    Iterator begin() { return Iterator(*this, 0); }
    Iterator end() { return Iterator(*this, m_next_index); }

private:
    std::vector<T> m_objects;
    std::vector<bool> m_allocated;
    std::vector<DirtyFlags> m_dirty_flags;
    DirtyFlags m_combined_dirty_flags{DirtyFlags{0}};
    Index m_next_index{0};
    std::vector<Index> m_free_indices;
#if FALCOR_RENDER_HANDLE_DEBUG
    std::vector<std::unique_ptr<DebugInfo>> m_debug_infos;
#endif
};

// ----------------------------------------------------------------------------
// Handles
// ----------------------------------------------------------------------------

struct RenderTriangleGeometry;
using RenderTriangleGeometryHandle = RenderObjectHandle<RenderTriangleGeometry, RenderObjectType::triangle_geometry>;
struct RenderLSSGeometry;
using RenderLSSGeometryHandle = RenderObjectHandle<RenderLSSGeometry, RenderObjectType::lss_geometry>;
struct RenderGeometryGroup;
using RenderGeometryGroupHandle = RenderObjectHandle<RenderGeometryGroup, RenderObjectType::geometry_group>;
struct RenderGeometryInstance;
using RenderGeometryInstanceHandle = RenderObjectHandle<RenderGeometryInstance, RenderObjectType::geometry_instance>;
struct RenderTransform;
using RenderTransformHandle = RenderObjectHandle<RenderTransform, RenderObjectType::transform>;

// ----------------------------------------------------------------------------
// RenderTriangleGeometry
// ----------------------------------------------------------------------------

struct RenderTriangleGeometryDesc {
    uint32_t vertex_count;
    uint32_t index_count;
};

struct RenderTriangleGeometry {
    uint32_t vertex_count;
    uint32_t vertex_offset;
    uint32_t index_count;
    uint32_t index_offset;
    OffsetAllocator::Allocation vertex_allocation;
    OffsetAllocator::Allocation index_allocation;
    /// List of geometry groups referencing this geometry.
    std::vector<RenderGeometryGroupHandle> geometry_groups;
};

struct RenderTriangleGeometryData {
    std::span<const shared::PackedTriangleVertex> vertices;
    std::span<const uint32_t> indices;
};

struct RenderTriangleGeometries {
    RenderObjectPool<RenderTriangleGeometry, RenderTriangleGeometryHandle> m_pool;
    ManagedVector<shared::TriangleGeometryDesc> m_descs;
    ManagedVector<shared::PackedTriangleVertex> m_vertices;
    ManagedVector<uint32_t> m_indices;
    OffsetAllocator m_vertex_allocator;
    OffsetAllocator m_index_allocator;

    RenderTriangleGeometries(sgl::Device* device);

    RenderTriangleGeometryHandle create(const RenderTriangleGeometryDesc& desc);
    void destroy(RenderTriangleGeometryHandle handle);
    void update(RenderTriangleGeometryHandle handle, const RenderTriangleGeometryData& data);

    void write_geometry_instance_desc(RenderTriangleGeometryHandle handle, shared::GeometryInstanceDesc& desc) const;
    void write_acceleration_structure_build_input(
        RenderTriangleGeometryHandle handle,
        sgl::AccelerationStructureBuildInput& build_input
    ) const;

    void update_buffers(sgl::CommandEncoder* command_encoder);
    void bind(sgl::ShaderCursor cursor) const;
};

// ----------------------------------------------------------------------------
// RenderLSSGeometry
// ----------------------------------------------------------------------------

struct RenderLSSGeometryDesc {
    uint32_t vertex_count;
    uint32_t index_count;
};

struct RenderLSSGeometry {
    uint32_t vertex_count;
    uint32_t vertex_offset;
    uint32_t index_count;
    uint32_t index_offset;
    OffsetAllocator::Allocation vertex_allocation;
    OffsetAllocator::Allocation index_allocation;
    /// List of geometry groups referencing this geometry.
    std::vector<RenderGeometryGroupHandle> geometry_groups;
};

struct RenderLSSGeometryData {
    std::span<const shared::PackedLSSVertex> vertices;
    std::span<const uint32_t> indices;
};

/// Mode used for LSS (Linear Swept Sphere) intersection.
enum class RenderLSSMode {
    /// Use native hardware LSS acceleration structure primitives.
    hardware,
    /// Use procedural AABB acceleration structure primitives with custom intersection shader.
    procedural,
};

SGL_ENUM_INFO(
    RenderLSSMode,
    {
        {RenderLSSMode::hardware, "hardware"},
        {RenderLSSMode::procedural, "procedural"},
    }
);
SGL_ENUM_REGISTER(RenderLSSMode);

struct RenderLSSGeometries {
    RenderObjectPool<RenderLSSGeometry, RenderLSSGeometryHandle> m_pool;
    ManagedVector<shared::LSSGeometryDesc> m_descs;
    ManagedVector<shared::PackedLSSVertex> m_vertices;
    ManagedVector<uint32_t> m_indices;
    OffsetAllocator m_vertex_allocator;
    OffsetAllocator m_index_allocator;

    RenderLSSMode m_lss_mode{RenderLSSMode::hardware};
    ref<sgl::Buffer> m_aabb_buffer;
    ref<sgl::ComputeKernel> m_compute_aabbs_kernel;

    RenderLSSGeometries(sgl::Device* device, RenderLSSMode lss_mode);

    RenderLSSMode lss_mode() const { return m_lss_mode; }

    RenderLSSGeometryHandle create(const RenderLSSGeometryDesc& desc);
    void destroy(RenderLSSGeometryHandle handle);
    void update(RenderLSSGeometryHandle handle, const RenderLSSGeometryData& data);

    void write_geometry_instance_desc(RenderLSSGeometryHandle handle, shared::GeometryInstanceDesc& desc) const;
    void write_acceleration_structure_build_input(
        RenderLSSGeometryHandle handle,
        sgl::AccelerationStructureBuildInput& build_input
    ) const;

    void update_buffers(sgl::CommandEncoder* command_encoder);
    void bind(sgl::ShaderCursor cursor) const;
};

// ----------------------------------------------------------------------------
// RenderGeometryGroup
// ----------------------------------------------------------------------------

struct RenderGeometryGroupDesc {
    std::span<RenderHandle> geometries;
    // std::span<MaterialID> materials;
    // std::span<float4x4> transforms;
};

struct RenderGeometryGroup {
    std::vector<RenderHandle> geometries;
    shared::GeometryType geometry_type;
    ref<sgl::AccelerationStructure> blas;
};

// ----------------------------------------------------------------------------
// RenderGeometryInstance
// ----------------------------------------------------------------------------

struct RenderGeometryInstanceDesc {
    RenderGeometryGroupHandle geometry_group;
    RenderTransformHandle transform;
    std::span<shared::MaterialID> material_ids;
};

struct RenderGeometryInstance {
    RenderGeometryGroupHandle geometry_group;
    RenderTransformHandle transform;
    shared::GeometryInstanceID geometry_instance_id;
    uint32_t geometry_instance_count;
    std::vector<shared::MaterialID> material_ids;
};

enum class RenderGeometryInstanceDirtyFlags : uint8_t {
    none = 0,
    created = 1 << 0,
    destroyed = 1 << 1,
    geometry_group = 1 << 2,
    transform = 1 << 3,
};
FALCOR_ENUM_CLASS_OPERATORS(RenderGeometryInstanceDirtyFlags);

using RenderGeometryInstancePool
    = RenderObjectPool<RenderGeometryInstance, RenderGeometryInstanceHandle, RenderGeometryInstanceDirtyFlags>;

// ----------------------------------------------------------------------------
// RenderTransform
// ----------------------------------------------------------------------------

enum class RenderTransformDirtyFlags : uint8_t {
    none = 0,
    created = 1 << 0,
    updated = 1 << 1,
};
FALCOR_ENUM_CLASS_OPERATORS(RenderTransformDirtyFlags);

struct RenderTransformDesc {
    float4x4 world_from_object;
};

struct RenderTransform {
    float4x4 world_from_object;
    /// List of geometry instances referencing this transform.
    std::vector<RenderGeometryInstanceHandle> geometry_instances;
};

// ----------------------------------------------------------------------------
// RenderScene
// ----------------------------------------------------------------------------

class FALCOR_API RenderScene : public Object {
    FALCOR_OBJECT(RenderScene)
public:
    enum class DirtyFlags {
        none = 0,
        triangle_geometry = 1 << 0,
        lss_geometry = 1 << 1,
        geometry_group = 1 << 2,
        geometry_instance = 1 << 3,
        transform = 1 << 4,
    };

    RenderScene(sgl::Device* device);

    static ref<RenderScene> create(sgl::Device* device) { return make_ref<RenderScene>(device); }

    // RenderTriangleGeometry

    RenderTriangleGeometryHandle create_triangle_geometry(const RenderTriangleGeometryDesc& desc);
    void destroy_triangle_geometry(RenderTriangleGeometryHandle handle);
    const RenderTriangleGeometry& get_triangle_geometry(RenderTriangleGeometryHandle handle) const;
    void update_triangle_geometry(RenderTriangleGeometryHandle handle, const RenderTriangleGeometryData& data);

    // RenderLSSGeometry

    RenderLSSGeometryHandle create_lss_geometry(const RenderLSSGeometryDesc& desc);
    void destroy_lss_geometry(RenderLSSGeometryHandle handle);
    const RenderLSSGeometry& get_lss_geometry(RenderLSSGeometryHandle handle) const;
    void update_lss_geometry(RenderLSSGeometryHandle handle, const RenderLSSGeometryData& data);

    // RenderGeometryGroup

    RenderGeometryGroupHandle create_geometry_group(const RenderGeometryGroupDesc& desc);
    void destroy_geometry_group(RenderGeometryGroupHandle handle);
    const RenderGeometryGroup& get_geometry_group(RenderGeometryGroupHandle handle) const;

    // RenderGeometryInstance

    RenderGeometryInstanceHandle create_geometry_instance(const RenderGeometryInstanceDesc& desc);
    void destroy_geometry_instance(RenderGeometryInstanceHandle handle);
    const RenderGeometryInstance& get_geometry_instance(RenderGeometryInstanceHandle handle) const;
    size_t geometry_instance_desc_count() const;

    // RenderTransform

    RenderTransformHandle create_transform(const RenderTransformDesc& desc);
    void destroy_transform(RenderTransformHandle handle);
    const RenderTransform& get_transform(RenderTransformHandle handle) const;
    void update_transform(RenderTransformHandle handle, const float4x4& world_from_object);

    bool update(const HitGroupPolicy& hit_group_policy);
    void bind_to_scene(sgl::ShaderCursor cursor) const;

    bool has_triangle_geometry() const { return m_triangle_geometries.m_pool.count() > 0; }
    bool has_lss_geometry() const { return m_lss_geometries.m_pool.count() > 0; }
    RenderLSSMode lss_mode() const { return m_lss_geometries.lss_mode(); }
    bool has_geometry_type(shared::GeometryType type) const;

    /// If true, TLAS updates use refit when only transforms have changed (default: true).
    bool tlas_prefer_refit() const { return m_tlas_prefer_refit; }
    void set_tlas_prefer_refit(bool value) { m_tlas_prefer_refit = value; }

private:
    RenderTriangleGeometry& get_triangle_geometry(RenderTriangleGeometryHandle handle);
    RenderLSSGeometry& get_lss_geometry(RenderLSSGeometryHandle handle);
    RenderTransform& get_transform(RenderTransformHandle handle);

    sgl::Buffer* get_scratch_buffer(size_t required_size);

    struct UpdateContext;
    bool update(UpdateContext& ctx);
    void update_blases(UpdateContext& ctx);
    void build_tlas(UpdateContext& ctx);

    sgl::Device* m_device;

    ref<sgl::SlangModule> m_render_module;

    ref<sgl::ComputeKernel> m_compute_inverse_kernel;
    ref<sgl::ComputeKernel> m_update_tlas_instance_desc_transforms_kernel;

    // DirtyFlags m_dirty_flags{DirtyFlags::none};

    RenderTriangleGeometries m_triangle_geometries;
    RenderLSSGeometries m_lss_geometries;

    RenderObjectPool<RenderGeometryGroup, RenderGeometryGroupHandle> m_geometry_group_pool;
    RenderGeometryInstancePool m_geometry_instance_pool;
    RenderObjectPool<RenderTransform, RenderTransformHandle, RenderTransformDirtyFlags> m_transform_pool;

    ManagedVector<shared::GeometryInstanceDesc> m_geometry_instance_descs;

    ManagedVector<float4x4> m_world_from_object;
    ManagedVector<float4x4> m_object_from_world;
    ManagedVector<float4x4> m_prev_world_from_object;
    bool m_prev_transforms_need_update{false};

    ManagedVector<shared::InstanceData> m_instance_data;

    std::vector<sgl::AccelerationStructureInstanceDesc> m_tlas_instance_descs;
    std::vector<uint8_t> m_tlas_native_instance_descs;
    ref<sgl::Buffer> m_tlas_native_instance_descs_buffer;
    ref<sgl::AccelerationStructure> m_tlas;
    /// If true, TLAS updates use refit when only transforms have changed.
    bool m_tlas_prefer_refit{true};

    ref<sgl::Buffer> m_scratch_buffer;
};

FALCOR_ENUM_CLASS_OPERATORS(RenderScene::DirtyFlags);

} // namespace falcor
