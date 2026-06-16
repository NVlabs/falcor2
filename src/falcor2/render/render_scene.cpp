// SPDX-License-Identifier: Apache-2.0

#include "render_scene.h"

#include "falcor2/render/hit_group_policy.h"
#include "falcor2/render/utils/check_struct.h"

#include <sgl/device/kernel.h>

#include <slang-rhi/acceleration-structure-utils.h>

namespace falcor {

static constexpr bool COMPUTE_TRANSFORM_INVERSE_TRANSPOSE_ON_CPU = false;

static constexpr size_t MAX_BUFFER_SIZE = 2ull * 1024 * 1024 * 1024;

/// Force the use of procedural LSS geometry, even if hardware support is available.
/// This is used for testing purposes.
static constexpr bool FORCE_PROCEDURAL_LSS = false;

template<typename T>
void remove(std::vector<T>& vector, const T& item)
{
    auto it = std::find(vector.begin(), vector.end(), item);
    if (it != vector.end()) {
        vector.erase(it);
    }
}

// ----------------------------------------------------------------------------
// RenderTriangleGeometry
// ----------------------------------------------------------------------------

RenderTriangleGeometries::RenderTriangleGeometries(sgl::Device* device)
    : m_descs({
          .label = "TriangleGeometryData::descs",
      })
    , m_vertices({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access
              | sgl::BufferUsage::acceleration_structure_build_input,
          .label = "TriangleGeometryData::vertices",
      })
    , m_indices({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access
              | sgl::BufferUsage::acceleration_structure_build_input,
          .label = "TriangleGeometryData::indices",
      })
    , m_vertex_allocator(static_cast<uint32_t>(MAX_BUFFER_SIZE / sizeof(shared::PackedTriangleVertex)))
    , m_index_allocator(static_cast<uint32_t>(MAX_BUFFER_SIZE / sizeof(uint32_t)))
{
    FALCOR_UNUSED(device);
}

RenderTriangleGeometryHandle RenderTriangleGeometries::create(const RenderTriangleGeometryDesc& desc)
{
    FALCOR_CHECK(desc.vertex_count > 0, "Vertex count must be greater than zero");
    FALCOR_CHECK(desc.index_count > 0, "Index count must be greater than zero");

    RenderTriangleGeometryHandle handle = m_pool.allocate();
    RenderTriangleGeometry& geometry = m_pool[handle];

    geometry.vertex_count = desc.vertex_count;
    geometry.index_count = desc.index_count;

    // Allocate vertex data.
    geometry.vertex_allocation = m_vertex_allocator.allocate(desc.vertex_count);
    FALCOR_CHECK(geometry.vertex_allocation.is_valid(), "Failed to allocate vertex data");
    geometry.vertex_offset = geometry.vertex_allocation.offset;
    m_vertices.ensure_size(geometry.vertex_offset + geometry.vertex_count);
    m_vertices.mark_dirty(geometry.vertex_offset, geometry.vertex_offset + geometry.vertex_count);

    // Allocate index data.
    geometry.index_allocation = m_index_allocator.allocate(desc.index_count);
    FALCOR_CHECK(geometry.index_allocation.is_valid(), "Failed to allocate index data");
    geometry.index_offset = geometry.index_allocation.offset;
    m_indices.ensure_size(geometry.index_offset + geometry.index_count);
    m_indices.mark_dirty(geometry.index_offset, geometry.index_offset + geometry.index_count);

    // Allocate descriptor.
    m_descs.ensure_size(handle.index() + 1);
    m_descs[handle.index()] = {
        .vertex_offset = geometry.vertex_offset,
        .vertex_count = geometry.vertex_count,
        .index_offset = geometry.index_offset,
        .index_count = geometry.index_count,
    };
    m_descs.mark_dirty(handle.index(), handle.index() + 1);

    m_pool.mark_dirty(handle, true);

    return handle;
}

void RenderTriangleGeometries::destroy(RenderTriangleGeometryHandle handle)
{
    RenderTriangleGeometry& geometry = m_pool[handle];

    // Free vertex data.
    if (geometry.vertex_allocation.is_valid())
        m_vertex_allocator.free(geometry.vertex_allocation);

    // Free index data.
    if (geometry.index_allocation.is_valid())
        m_index_allocator.free(geometry.index_allocation);

    // Clear descriptor.
    m_descs[handle.index()] = {};
    m_descs.mark_dirty(handle.index(), handle.index() + 1);

    m_pool.free(handle);
}

void RenderTriangleGeometries::update(RenderTriangleGeometryHandle handle, const RenderTriangleGeometryData& data)
{
    RenderTriangleGeometry& geometry = m_pool[handle];

    // Copy vertex data.
    FALCOR_CHECK_EQ(data.vertices.size(), geometry.vertex_count);
    std::memcpy(m_vertices.data() + geometry.vertex_offset, data.vertices.data(), data.vertices.size_bytes());
    m_vertices.mark_dirty(geometry.vertex_offset, geometry.vertex_offset + geometry.vertex_count);

    // Copy index data.
    FALCOR_CHECK_EQ(data.indices.size(), geometry.index_count);
    std::memcpy(m_indices.data() + geometry.index_offset, data.indices.data(), data.indices.size_bytes());
    m_indices.mark_dirty(geometry.index_offset, geometry.index_offset + geometry.index_count);

    m_pool.mark_dirty(handle, true);
}

void RenderTriangleGeometries::write_geometry_instance_desc(
    RenderTriangleGeometryHandle handle,
    shared::GeometryInstanceDesc& desc
) const
{
    const RenderTriangleGeometry& geometry = m_pool[handle];
    shared::TriangleGeometryInstanceDesc& triangle_desc = reinterpret_cast<shared::TriangleGeometryInstanceDesc&>(desc);
    triangle_desc.header.type = shared::GeometryType::triangle;
    triangle_desc.header.geometry_id = shared::GeometryID{handle.index()};
    triangle_desc.vertex_offset = geometry.vertex_offset;
    triangle_desc.vertex_count = geometry.vertex_count;
    triangle_desc.index_offset = geometry.index_offset;
    triangle_desc.index_count = geometry.index_count;
}

void RenderTriangleGeometries::write_acceleration_structure_build_input(
    RenderTriangleGeometryHandle handle,
    sgl::AccelerationStructureBuildInput& build_input
) const
{
    const RenderTriangleGeometry& geometry = m_pool[handle];
    sgl::AccelerationStructureBuildInputTriangles triangles;
    triangles.vertex_buffers
        = {sgl::BufferOffsetPair(m_vertices.buffer(), geometry.vertex_offset * sizeof(shared::PackedTriangleVertex))};
    triangles.vertex_format = sgl::Format::rgb32_float;
    triangles.vertex_count = geometry.vertex_count;
    triangles.vertex_stride = sizeof(shared::PackedTriangleVertex);
    triangles.index_buffer = sgl::BufferOffsetPair(m_indices.buffer(), geometry.index_offset * sizeof(uint32_t));
    triangles.index_format = sgl::IndexFormat::uint32;
    triangles.index_count = geometry.index_count;
    triangles.flags = sgl::AccelerationStructureGeometryFlags::opaque;
    build_input = triangles;
}

void RenderTriangleGeometries::update_buffers(sgl::CommandEncoder* command_encoder)
{
    m_descs.update_buffer(command_encoder);
    m_vertices.update_buffer(command_encoder);
    m_indices.update_buffer(command_encoder);
}

void RenderTriangleGeometries::bind(sgl::ShaderCursor cursor) const
{
    if (m_descs.buffer())
        cursor["descs"] = m_descs.buffer();
    if (m_vertices.buffer())
        cursor["vertices"] = m_vertices.buffer();
    if (m_indices.buffer())
        cursor["indices"] = m_indices.buffer();
}

// ----------------------------------------------------------------------------
// RenderLSSGeometry
// ----------------------------------------------------------------------------

RenderLSSGeometries::RenderLSSGeometries(sgl::Device* device, RenderLSSMode lss_mode)
    : m_descs({
          .label = "LSSGeometryData::descs",
      })
    , m_vertices({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access
              | sgl::BufferUsage::acceleration_structure_build_input,
          .label = "LSSGeometryData::vertices",
      })
    , m_indices({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access
              | sgl::BufferUsage::acceleration_structure_build_input,
          .label = "LSSGeometryData::indices",
      })
    , m_vertex_allocator(static_cast<uint32_t>(MAX_BUFFER_SIZE / sizeof(shared::PackedLSSVertex)))
    , m_index_allocator(static_cast<uint32_t>(MAX_BUFFER_SIZE / sizeof(uint32_t)))
    , m_lss_mode(lss_mode)
{
    if (m_lss_mode == RenderLSSMode::procedural) {
        m_compute_aabbs_kernel = device->create_compute_kernel({
            .program = device->load_program("falcor2/render/kernels/lss_kernels.slang", {"compute_lss_aabbs"}),
        });
    }
}

RenderLSSGeometryHandle RenderLSSGeometries::create(const RenderLSSGeometryDesc& desc)
{
    FALCOR_CHECK(desc.vertex_count > 0, "Vertex count must be greater than zero");
    FALCOR_CHECK(desc.index_count > 0, "Index count must be greater than zero");

    RenderLSSGeometryHandle handle = m_pool.allocate();
    RenderLSSGeometry& geometry = m_pool[handle];

    geometry.vertex_count = desc.vertex_count;
    geometry.index_count = desc.index_count;

    // Allocate vertex data.
    geometry.vertex_allocation = m_vertex_allocator.allocate(desc.vertex_count);
    FALCOR_CHECK(geometry.vertex_allocation.is_valid(), "Failed to allocate vertex data");
    geometry.vertex_offset = geometry.vertex_allocation.offset;
    m_vertices.ensure_size(geometry.vertex_offset + geometry.vertex_count);
    m_vertices.mark_dirty(geometry.vertex_offset, geometry.vertex_offset + geometry.vertex_count);

    // Allocate index data.
    geometry.index_allocation = m_index_allocator.allocate(desc.index_count);
    FALCOR_CHECK(geometry.index_allocation.is_valid(), "Failed to allocate index data");
    geometry.index_offset = geometry.index_allocation.offset;
    m_indices.ensure_size(geometry.index_offset + geometry.index_count);
    m_indices.mark_dirty(geometry.index_offset, geometry.index_offset + geometry.index_count);

    // Allocate descriptor.
    m_descs.ensure_size(handle.index() + 1);
    m_descs[handle.index()] = {
        .vertex_offset = geometry.vertex_offset,
        .vertex_count = geometry.vertex_count,
        .index_offset = geometry.index_offset,
        .index_count = geometry.index_count,
    };
    m_descs.mark_dirty(handle.index(), handle.index() + 1);

    m_pool.mark_dirty(handle, true);

    return handle;
}

void RenderLSSGeometries::destroy(RenderLSSGeometryHandle handle)
{
    RenderLSSGeometry& geometry = m_pool[handle];

    // Free vertex data.
    if (geometry.vertex_allocation.is_valid())
        m_vertex_allocator.free(geometry.vertex_allocation);

    // Free index data.
    if (geometry.index_allocation.is_valid())
        m_index_allocator.free(geometry.index_allocation);

    // Clear descriptor.
    m_descs[handle.index()] = {};
    m_descs.mark_dirty(handle.index(), handle.index() + 1);

    m_pool.free(handle);
}

void RenderLSSGeometries::update(RenderLSSGeometryHandle handle, const RenderLSSGeometryData& data)
{
    RenderLSSGeometry& geometry = m_pool[handle];

    // Copy vertex data.
    FALCOR_CHECK_EQ(data.vertices.size(), geometry.vertex_count);
    std::memcpy(m_vertices.data() + geometry.vertex_offset, data.vertices.data(), data.vertices.size_bytes());
    m_vertices.mark_dirty(geometry.vertex_offset, geometry.vertex_offset + geometry.vertex_count);

    // Copy index data.
    FALCOR_CHECK_EQ(data.indices.size(), geometry.index_count);
    std::memcpy(m_indices.data() + geometry.index_offset, data.indices.data(), data.indices.size_bytes());
    m_indices.mark_dirty(geometry.index_offset, geometry.index_offset + geometry.index_count);

    m_pool.mark_dirty(handle, true);
}

void RenderLSSGeometries::write_geometry_instance_desc(
    RenderLSSGeometryHandle handle,
    shared::GeometryInstanceDesc& desc
) const
{
    const RenderLSSGeometry& geometry = m_pool[handle];
    shared::LSSGeometryInstanceDesc& lss_desc = reinterpret_cast<shared::LSSGeometryInstanceDesc&>(desc);
    lss_desc.header.type = shared::GeometryType::lss;
    lss_desc.header.geometry_id = shared::GeometryID{handle.index()};
    lss_desc.vertex_offset = geometry.vertex_offset;
    lss_desc.vertex_count = geometry.vertex_count;
    lss_desc.index_offset = geometry.index_offset;
    lss_desc.index_count = geometry.index_count;
}

void RenderLSSGeometries::write_acceleration_structure_build_input(
    RenderLSSGeometryHandle handle,
    sgl::AccelerationStructureBuildInput& build_input
) const
{
    const RenderLSSGeometry& geometry = m_pool[handle];
    if (m_lss_mode == RenderLSSMode::procedural) {
        sgl::AccelerationStructureBuildInputProceduralPrimitives procedural;
        procedural.aabb_buffers = {sgl::BufferOffsetPair(m_aabb_buffer, geometry.index_offset * sizeof(rhi::AABB))};
        procedural.aabb_stride = sizeof(rhi::AABB);
        procedural.primitive_count = geometry.index_count;
        procedural.flags = sgl::AccelerationStructureGeometryFlags::opaque;
        build_input = procedural;
    } else {
        sgl::AccelerationStructureBuildInputLinearSweptSpheres lss;
        lss.vertex_count = geometry.vertex_count;
        lss.primitive_count = geometry.index_count;
        lss.vertex_position_buffers
            = {sgl::BufferOffsetPair(m_vertices.buffer(), geometry.vertex_offset * sizeof(shared::PackedLSSVertex))};
        lss.vertex_position_format = sgl::Format::rgb32_float;
        lss.vertex_position_stride = sizeof(shared::PackedLSSVertex);
        lss.vertex_radius_buffers = {sgl::BufferOffsetPair(
            m_vertices.buffer(),
            geometry.vertex_offset * sizeof(shared::PackedLSSVertex) + sizeof(float3)
        )};
        lss.vertex_radius_format = sgl::Format::r32_float;
        lss.vertex_radius_stride = sizeof(shared::PackedLSSVertex);
        lss.index_buffer = sgl::BufferOffsetPair(m_indices.buffer(), geometry.index_offset * sizeof(uint32_t));
        lss.index_format = sgl::IndexFormat::uint32;
        lss.index_count = geometry.index_count;
        lss.indexing_mode = sgl::LinearSweptSpheresIndexingMode::successive;
        lss.end_caps_mode = sgl::LinearSweptSpheresEndCapsMode::chained;
        lss.flags = sgl::AccelerationStructureGeometryFlags::opaque;
        build_input = lss;
    }
}

void RenderLSSGeometries::update_buffers(sgl::CommandEncoder* command_encoder)
{
    m_descs.update_buffer(command_encoder);
    m_vertices.update_buffer(command_encoder);
    m_indices.update_buffer(command_encoder);

    if (m_lss_mode == RenderLSSMode::procedural && m_pool.is_dirty()) {
        // Compute total number of segments (indices) across all geometries.
        uint32_t total_segments = 0;
        for (const auto& [handle, geometry] : m_pool) {
            total_segments = std::max(total_segments, geometry.index_offset + geometry.index_count);
        }

        if (total_segments > 0) {
            // Ensure AABB buffer is large enough.
            size_t required_size = total_segments * sizeof(rhi::AABB);
            bool recompute_all = false;
            if (!m_aabb_buffer || m_aabb_buffer->size() < required_size) {
                m_aabb_buffer = command_encoder->device()->create_buffer({
                    .size = required_size,
                    .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::acceleration_structure_build_input,
                    .label = "LSSGeometryData::aabbs",
                });
                recompute_all = true;
            }

            // Dispatch AABB compute for each dirty geometry (or all geometries if we had to resize the AABB buffer).
            for (const auto& [handle, geometry] : m_pool) {
                if (m_pool.is_dirty(handle) || recompute_all) {
                    m_compute_aabbs_kernel->dispatch(
                        uint3(geometry.index_count, 1, 1),
                        [&](sgl::ShaderCursor cursor)
                        {
                            cursor = cursor.find_entry_point(0);
                            cursor["vertices"] = m_vertices.buffer();
                            cursor["indices"] = m_indices.buffer();
                            cursor["aabbs"] = m_aabb_buffer;
                            cursor["vertex_offset"] = geometry.vertex_offset;
                            cursor["index_offset"] = geometry.index_offset;
                            cursor["index_count"] = geometry.index_count;
                        },
                        command_encoder
                    );
                }
            }
        }
    }
}

void RenderLSSGeometries::bind(sgl::ShaderCursor cursor) const
{
    if (m_descs.buffer())
        cursor["descs"] = m_descs.buffer();
    if (m_vertices.buffer())
        cursor["vertices"] = m_vertices.buffer();
    if (m_indices.buffer())
        cursor["indices"] = m_indices.buffer();
}

// ----------------------------------------------------------------------------
// RenderScene
// ----------------------------------------------------------------------------

inline void check_shared_types(sgl::SlangModule* module)
{
    CHECK_STRUCT_BEGIN(module, "MaterialHeader", shared::MaterialHeader);
    CHECK_STRUCT_FIELD(type_id);
    CHECK_STRUCT_FIELD(flags);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "MaterialPayload", shared::MaterialPayload);
    CHECK_STRUCT_FIELD(data);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "MaterialData", shared::MaterialData);
    CHECK_STRUCT_FIELD(header);
    CHECK_STRUCT_FIELD(payload);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "LightHeader", shared::LightHeader);
    CHECK_STRUCT_FIELD(type);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "LightPayload", shared::LightPayload);
    CHECK_STRUCT_FIELD(data);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "LightData", shared::LightData);
    CHECK_STRUCT_FIELD(header);
    CHECK_STRUCT_FIELD(payload);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "GeometryInstanceDescHeader", shared::GeometryInstanceDescHeader);
    CHECK_STRUCT_FIELD(type);
    CHECK_STRUCT_FIELD(geometry_id);
    CHECK_STRUCT_FIELD(material_id);
    CHECK_STRUCT_FIELD(transform_id);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "GeometryInstanceDescPayload", shared::GeometryInstanceDescPayload);
    CHECK_STRUCT_FIELD(data);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "GeometryInstanceDesc", shared::GeometryInstanceDesc);
    CHECK_STRUCT_FIELD(header);
    CHECK_STRUCT_FIELD(payload);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "InstanceData", shared::InstanceData);
    CHECK_STRUCT_FIELD(transform_id);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "PackedTriangleVertex", shared::PackedTriangleVertex);
    CHECK_STRUCT_FIELD(position);
    CHECK_STRUCT_FIELD(normal_and_tangent);
    CHECK_STRUCT_FIELD(uv);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "TriangleGeometryDesc", shared::TriangleGeometryDesc);
    CHECK_STRUCT_FIELD(vertex_offset);
    CHECK_STRUCT_FIELD(vertex_count);
    CHECK_STRUCT_FIELD(index_offset);
    CHECK_STRUCT_FIELD(index_count);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "TriangleGeometryInstanceDesc", shared::TriangleGeometryInstanceDesc);
    CHECK_STRUCT_FIELD(header);
    CHECK_STRUCT_FIELD(vertex_offset);
    CHECK_STRUCT_FIELD(vertex_count);
    CHECK_STRUCT_FIELD(index_offset);
    CHECK_STRUCT_FIELD(index_count);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "PackedLSSVertex", shared::PackedLSSVertex);
    CHECK_STRUCT_FIELD(position);
    CHECK_STRUCT_FIELD(radius);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "LSSGeometryDesc", shared::LSSGeometryDesc);
    CHECK_STRUCT_FIELD(vertex_offset);
    CHECK_STRUCT_FIELD(vertex_count);
    CHECK_STRUCT_FIELD(index_offset);
    CHECK_STRUCT_FIELD(index_count);
    CHECK_STRUCT_END();

    CHECK_STRUCT_BEGIN(module, "LSSGeometryInstanceDesc", shared::LSSGeometryInstanceDesc);
    CHECK_STRUCT_FIELD(header);
    CHECK_STRUCT_FIELD(vertex_offset);
    CHECK_STRUCT_FIELD(vertex_count);
    CHECK_STRUCT_FIELD(index_offset);
    CHECK_STRUCT_FIELD(index_count);
    CHECK_STRUCT_END();
}

RenderScene::RenderScene(sgl::Device* device)
    : m_device(device)
    , m_triangle_geometries(device)
    , m_lss_geometries(
          device,
          (FORCE_PROCEDURAL_LSS || !device->has_feature(sgl::Feature::acceleration_structure_linear_swept_spheres))
              ? RenderLSSMode::procedural
              : RenderLSSMode::hardware
      )
    , m_geometry_instance_descs({
          .usage = sgl::BufferUsage::shader_resource,
          .label = "Scene::geometry_instance_descs",
      })
    , m_world_from_object({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
          .label = "Scene::world_from_object",
      })
    , m_object_from_world({
          .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
          .label = "Scene::object_from_world",
      })
    , m_prev_world_from_object({
          .usage = sgl::BufferUsage::shader_resource,
          .label = "Scene::prev_world_from_object",
      })
    , m_instance_data({
          .usage = sgl::BufferUsage::shader_resource,
          .label = "Scene::instance_data",
      })
{
    m_render_module = m_device->load_module("falcor2/render.slang");
    check_shared_types(m_render_module);

    m_compute_inverse_kernel = m_device->create_compute_kernel({
        .program = m_device->load_program("falcor2/render/kernels/transform_kernels.slang", {"compute_inverse"}),
    });
    m_update_tlas_instance_desc_transforms_kernel = m_device->create_compute_kernel({
        .program = m_device->load_program(
            "falcor2/render/kernels/transform_kernels.slang",
            {"update_tlas_instance_desc_transforms"}
        ),
    });
}

RenderTriangleGeometryHandle RenderScene::create_triangle_geometry(const RenderTriangleGeometryDesc& desc)
{
    RenderTriangleGeometryHandle handle = m_triangle_geometries.create(desc);
    return handle;
}

void RenderScene::destroy_triangle_geometry(RenderTriangleGeometryHandle handle)
{
    m_triangle_geometries.destroy(handle);
}

const RenderTriangleGeometry& RenderScene::get_triangle_geometry(RenderTriangleGeometryHandle handle) const
{
    return m_triangle_geometries.m_pool[handle];
}

RenderTriangleGeometry& RenderScene::get_triangle_geometry(RenderTriangleGeometryHandle handle)
{
    return m_triangle_geometries.m_pool[handle];
}

void RenderScene::update_triangle_geometry(RenderTriangleGeometryHandle handle, const RenderTriangleGeometryData& data)
{
    m_triangle_geometries.update(handle, data);
}

RenderLSSGeometryHandle RenderScene::create_lss_geometry(const RenderLSSGeometryDesc& desc)
{
    RenderLSSGeometryHandle handle = m_lss_geometries.create(desc);
    return handle;
}

void RenderScene::destroy_lss_geometry(RenderLSSGeometryHandle handle)
{
    m_lss_geometries.destroy(handle);
}

const RenderLSSGeometry& RenderScene::get_lss_geometry(RenderLSSGeometryHandle handle) const
{
    return m_lss_geometries.m_pool[handle];
}

RenderLSSGeometry& RenderScene::get_lss_geometry(RenderLSSGeometryHandle handle)
{
    return m_lss_geometries.m_pool[handle];
}

void RenderScene::update_lss_geometry(RenderLSSGeometryHandle handle, const RenderLSSGeometryData& data)
{
    m_lss_geometries.update(handle, data);
}

RenderGeometryGroupHandle RenderScene::create_geometry_group(const RenderGeometryGroupDesc& desc)
{
    FALCOR_CHECK(desc.geometries.size() > 0, "Geometry group must contain at least one geometry");

    RenderGeometryGroupHandle handle = m_geometry_group_pool.allocate();
    RenderGeometryGroup& geometry_group = m_geometry_group_pool[handle];
    FALCOR_ASSERT(!desc.geometries.empty());
    geometry_group.geometries.assign(desc.geometries.begin(), desc.geometries.end());
    shared::GeometryType geometry_type = shared::GeometryType::invalid;
    for (const RenderHandle& geometry_handle : geometry_group.geometries) {
        switch (geometry_handle.type()) {
        case RenderObjectType::triangle_geometry:
            get_triangle_geometry(RenderTriangleGeometryHandle{geometry_handle}).geometry_groups.push_back(handle);
            FALCOR_ASSERT(
                geometry_type == shared::GeometryType::invalid || geometry_type == shared::GeometryType::triangle
            );
            geometry_type = shared::GeometryType::triangle;
            break;
        case RenderObjectType::lss_geometry:
            get_lss_geometry(RenderLSSGeometryHandle{geometry_handle}).geometry_groups.push_back(handle);
            FALCOR_ASSERT(geometry_type == shared::GeometryType::invalid || geometry_type == shared::GeometryType::lss);
            geometry_type = shared::GeometryType::lss;
            break;
        default:
            FALCOR_THROW("Unsupported geometry type in geometry group");
        }
    }
    geometry_group.geometry_type = geometry_type;

    m_geometry_group_pool.mark_dirty(handle, true);

    return handle;
}

void RenderScene::destroy_geometry_group(RenderGeometryGroupHandle handle)
{
    RenderGeometryGroup& geometry_group = m_geometry_group_pool[handle];
    for (const RenderHandle& geometry_handle : geometry_group.geometries) {
        switch (geometry_handle.type()) {
        case RenderObjectType::triangle_geometry:
            remove(get_triangle_geometry(RenderTriangleGeometryHandle{geometry_handle}).geometry_groups, handle);
            break;
        case RenderObjectType::lss_geometry:
            remove(get_lss_geometry(RenderLSSGeometryHandle{geometry_handle}).geometry_groups, handle);
            break;
        default:
            FALCOR_THROW("Unsupported geometry type in geometry group");
        }
    }

    m_geometry_group_pool.free(handle);
}

const RenderGeometryGroup& RenderScene::get_geometry_group(RenderGeometryGroupHandle handle) const
{
    return m_geometry_group_pool[handle];
}

RenderGeometryInstanceHandle RenderScene::create_geometry_instance(const RenderGeometryInstanceDesc& desc)
{
    RenderGeometryInstanceHandle handle = m_geometry_instance_pool.allocate();
    m_geometry_instance_pool.mark_dirty(handle, RenderGeometryInstanceDirtyFlags::created);
    RenderGeometryInstance& geometry_instance = m_geometry_instance_pool[handle];
    geometry_instance.geometry_group = desc.geometry_group;
    geometry_instance.transform = desc.transform;
    geometry_instance.material_ids.assign(desc.material_ids.begin(), desc.material_ids.end());
    get_transform(desc.transform).geometry_instances.push_back(handle);
    return handle;
}

void RenderScene::destroy_geometry_instance(RenderGeometryInstanceHandle handle)
{
    RenderGeometryInstance& geometry_instance = m_geometry_instance_pool[handle];
    remove(get_transform(geometry_instance.transform).geometry_instances, handle);
    m_geometry_instance_pool.mark_dirty(handle, RenderGeometryInstanceDirtyFlags::destroyed);
    m_geometry_instance_pool.free(handle);
}

const RenderGeometryInstance& RenderScene::get_geometry_instance(RenderGeometryInstanceHandle handle) const
{
    return m_geometry_instance_pool[handle];
}

size_t RenderScene::geometry_instance_desc_count() const
{
    return m_geometry_instance_descs.size();
}

RenderTransformHandle RenderScene::create_transform(const RenderTransformDesc& desc)
{
    RenderTransformHandle handle = m_transform_pool.allocate();
    RenderTransform& transform = m_transform_pool[handle];
    transform.world_from_object = desc.world_from_object;
    m_transform_pool.mark_dirty(handle, RenderTransformDirtyFlags::created);
    return handle;
}

void RenderScene::destroy_transform(RenderTransformHandle handle)
{
    m_transform_pool.free(handle);
}

const RenderTransform& RenderScene::get_transform(RenderTransformHandle handle) const
{
    return m_transform_pool[handle];
}

RenderTransform& RenderScene::get_transform(RenderTransformHandle handle)
{
    return m_transform_pool[handle];
}

void RenderScene::update_transform(RenderTransformHandle handle, const float4x4& world_from_object)
{
    RenderTransform& transform = m_transform_pool[handle];
    transform.world_from_object = world_from_object;
    m_transform_pool.mark_dirty(handle, RenderTransformDirtyFlags::updated);
}

sgl::Buffer* RenderScene::get_scratch_buffer(size_t required_size)
{
    if (!m_scratch_buffer || m_scratch_buffer->size() < required_size) {
        m_scratch_buffer = m_device->create_buffer({
            .size = required_size,
            .usage = sgl::BufferUsage::unordered_access,
            .label = "m_scratch_buffer",
        });
    }
    return m_scratch_buffer;
}

struct RenderScene::UpdateContext {
    UpdateContext(sgl::Device* device, const HitGroupPolicy& hit_group_policy)
        : m_device(device)
        , m_hit_group_policy(hit_group_policy)
    {
    }

    const HitGroupPolicy& hit_group_policy() const { return m_hit_group_policy; }

    sgl::CommandEncoder* command_encoder()
    {
        if (!m_command_encoder)
            m_command_encoder = m_device->create_command_encoder();
        return m_command_encoder;
    }

    void submit_command_encoder()
    {
        if (m_command_encoder) {
            m_device->submit_command_buffer(m_command_encoder->finish());
            m_command_encoder = nullptr;
        }
    }

private:
    sgl::Device* m_device;
    const HitGroupPolicy& m_hit_group_policy;
    ref<sgl::CommandEncoder> m_command_encoder;
};

bool RenderScene::update(const HitGroupPolicy& hit_group_policy)
{
    UpdateContext ctx(m_device, hit_group_policy);
    bool changed = update(ctx);
    ctx.submit_command_encoder();
    return changed;
}

void RenderScene::bind_to_scene(sgl::ShaderCursor cursor) const
{
    cursor["tlas"] = m_tlas;
    m_triangle_geometries.bind(cursor["triangle_geometry"]);
    m_lss_geometries.bind(cursor["lss_geometry"]);
    if (m_geometry_instance_descs.buffer())
        cursor["geometry_instance_descs"] = m_geometry_instance_descs.buffer();
    if (m_world_from_object.buffer())
        cursor["world_from_object"] = m_world_from_object.buffer();
    if (m_object_from_world.buffer())
        cursor["object_from_world"] = m_object_from_world.buffer();
    if (m_prev_world_from_object.buffer())
        cursor["prev_world_from_object"] = m_prev_world_from_object.buffer();
}

bool RenderScene::has_geometry_type(shared::GeometryType type) const
{
    switch (type) {
    case shared::GeometryType::triangle:
        return has_triangle_geometry();
    case shared::GeometryType::lss:
        return has_lss_geometry();
    default:
        return false;
    }
}

bool RenderScene::update(UpdateContext& ctx)
{
    bool prev_world_from_object_needs_update = false;

    // Update previous frame transforms by just copying current -> prev for all transforms.
    // This is skipped if no transforms were dirty last frame, since in that case prev == current already.
    // We could optimize this by having m_prev_transforms_need_update be a bitmask of dirty transform indices.
    if (m_prev_transforms_need_update) {
        m_prev_world_from_object.resize(m_world_from_object.size());
        std::copy(
            m_world_from_object.data(),
            m_world_from_object.data() + m_world_from_object.size(),
            m_prev_world_from_object.data()
        );
        m_prev_world_from_object.mark_all_dirty();
        m_prev_transforms_need_update = false;
        prev_world_from_object_needs_update = true;
    }

    // Update all dirty transforms.
    // For updated (not created) transforms, we need to save the old current value to prev before overwriting current.
    // For created transforms, we can just copy the new current value to prev after writing current.
    if (m_transform_pool.is_dirty()) {
        m_world_from_object.resize(m_transform_pool.capacity());
        m_object_from_world.resize(m_transform_pool.capacity());
        m_prev_world_from_object.resize(m_transform_pool.capacity());

        const auto& dirty_flags = m_transform_pool.dirty_flags();

        // For updated (not created) transforms, save current -> prev before overwriting.
        for (const auto& [handle, transform] : m_transform_pool) {
            RenderTransformDirtyFlags flags = dirty_flags[handle.index()];
            if (flags == RenderTransformDirtyFlags::none)
                continue;
            if (is_set(flags, RenderTransformDirtyFlags::updated)) {
                m_prev_world_from_object.data()[handle.index()] = m_world_from_object.data()[handle.index()];
                m_prev_world_from_object.mark_dirty(handle.index(), handle.index() + 1);
            }
            m_world_from_object[handle.index()] = transform.world_from_object;
            m_world_from_object.mark_dirty(handle.index(), handle.index() + 1);
            if (is_set(flags, RenderTransformDirtyFlags::created)) {
                m_prev_world_from_object.data()[handle.index()] = transform.world_from_object;
                m_prev_world_from_object.mark_dirty(handle.index(), handle.index() + 1);
            }
            if (COMPUTE_TRANSFORM_INVERSE_TRANSPOSE_ON_CPU)
                m_object_from_world[handle.index()] = inverse(transform.world_from_object);
            for (const RenderGeometryInstanceHandle& geometry_instance : transform.geometry_instances) {
                m_geometry_instance_pool.mark_dirty(geometry_instance, RenderGeometryInstanceDirtyFlags::transform);
            }
        }

        // Upload all buffers.
        m_world_from_object.update_buffer(ctx.command_encoder());
        prev_world_from_object_needs_update = true;

        if (COMPUTE_TRANSFORM_INVERSE_TRANSPOSE_ON_CPU) {
            m_object_from_world.update_buffer(ctx.command_encoder());
        } else {
            m_object_from_world.update_buffer(ctx.command_encoder(), false);
            m_compute_inverse_kernel->dispatch(
                uint3(m_world_from_object.size(), 1, 1),
                [&](sgl::ShaderCursor cursor)
                {
                    cursor = cursor.find_entry_point(0);
                    cursor["transforms"] = m_world_from_object.buffer();
                    cursor["inverse_transforms"] = m_object_from_world.buffer();
                    cursor["first"] = uint32_t{0};
                    cursor["count"] = sgl::narrow_cast<uint32_t>(m_world_from_object.size());
                },
                ctx.command_encoder()
            );
        }

        // Schedule catch-up for next frame.
        m_prev_transforms_need_update = true;
    }

    if (prev_world_from_object_needs_update) {
        m_prev_world_from_object.update_buffer(ctx.command_encoder());
    }

    // Update geometries.
    // Update all device-side geometry buffers (vertices, indices, descriptors).
    m_triangle_geometries.update_buffers(ctx.command_encoder());
    m_lss_geometries.update_buffers(ctx.command_encoder());
    // TODO(scene): This is where we might call back to user code to run compute shaders on vertex data (displacement
    // shaders).
    // Check for dirty triangle geometry and mark geometry groups as dirty.
    if (m_triangle_geometries.m_pool.is_dirty()) {
        for (const auto& [handle, geometry] : m_triangle_geometries.m_pool) {
            if (m_triangle_geometries.m_pool.is_dirty(handle)) {
                for (const RenderGeometryGroupHandle& geometry_group : geometry.geometry_groups) {
                    m_geometry_group_pool.mark_dirty(geometry_group, true);
                }
            }
        }
    }
    // Check for dirty lss geometry and mark geometry groups as dirty.
    if (m_lss_geometries.m_pool.is_dirty()) {
        for (const auto& [handle, geometry] : m_lss_geometries.m_pool) {
            if (m_lss_geometries.m_pool.is_dirty(handle)) {
                for (const RenderGeometryGroupHandle& geometry_group : geometry.geometry_groups) {
                    m_geometry_group_pool.mark_dirty(geometry_group, true);
                }
            }
        }
    }

    // Update geometry groups.

    // TODO(scene): Collect all dirty geometry groups and update their BLASes only.
    // Mark geometry instances referencing dirty geometry groups as dirty.
    if (m_geometry_group_pool.is_dirty()) {
        for (const auto& [geometry_instance_handle, geometry_instance] : m_geometry_instance_pool) {
            if (m_geometry_group_pool.is_dirty(geometry_instance.geometry_group)) {
                m_geometry_instance_pool.mark_dirty(
                    geometry_instance_handle,
                    RenderGeometryInstanceDirtyFlags::geometry_group
                );
            }
        }
    }

    // Update BLASes.
    if (m_geometry_group_pool.is_dirty()) {
        // Make sure geometry buffers are up to date before building BLASes.
        ctx.submit_command_encoder();
        update_blases(ctx);
    }

    // Update geometry instance data.
    if (m_geometry_instance_pool.combined_dirty_flags() == RenderGeometryInstanceDirtyFlags::transform) {
        // Only transforms have changed.
    } else if (m_geometry_instance_pool.is_dirty()) {
        const HitGroupPolicy& hit_group_policy = ctx.hit_group_policy();
        uint32_t tlas_instance_desc_count = static_cast<uint32_t>(m_geometry_instance_pool.count());
        uint32_t tlas_instance_desc_index = 0;
        uint32_t geometry_instance_desc_count = 0;
        uint32_t geometry_instance_desc_index = 0;
        for (const auto& [handle, geometry_instance] : m_geometry_instance_pool) {
            const RenderGeometryGroup& geometry_group = m_geometry_group_pool[geometry_instance.geometry_group];
            geometry_instance_desc_count += static_cast<uint32_t>(geometry_group.geometries.size());
        }
        m_instance_data.resize(tlas_instance_desc_count);
        m_geometry_instance_descs.resize(geometry_instance_desc_count);
        m_tlas_instance_descs.resize(tlas_instance_desc_count);
        for (const auto& [handle, geometry_instance] : m_geometry_instance_pool) {
            const RenderGeometryGroup& geometry_group = m_geometry_group_pool[geometry_instance.geometry_group];
            shared::TransformID transform_id{geometry_instance.transform.index()};

            m_geometry_instance_pool[handle].geometry_instance_id
                = shared::GeometryInstanceID{geometry_instance_desc_index};
            m_geometry_instance_pool[handle].geometry_instance_count
                = static_cast<uint32_t>(geometry_group.geometries.size());

            m_instance_data[tlas_instance_desc_index].transform_id = transform_id;

            sgl::AccelerationStructureInstanceDesc instance_desc;
            instance_desc.transform = geometry_instance.transform.is_valid()
                ? float3x4(m_transform_pool[geometry_instance.transform].world_from_object)
                : float3x4::identity();
            instance_desc.instance_id = geometry_instance_desc_index;
            instance_desc.instance_mask = 0xff;
            instance_desc.instance_contribution_to_hit_group_index
                = hit_group_policy.instance_contribution_to_hit_group_index(
                    uint32_t(geometry_group.geometry_type),
                    geometry_instance_desc_index
                );
            instance_desc.flags = sgl::AccelerationStructureInstanceFlags::none;
            instance_desc.acceleration_structure = geometry_group.blas->handle();
            m_tlas_instance_descs[tlas_instance_desc_index] = instance_desc;

            for (size_t geometry_index = 0; geometry_index < geometry_group.geometries.size(); ++geometry_index) {
                const RenderHandle& geometry_handle = geometry_group.geometries[geometry_index];
                shared::GeometryInstanceDesc geometry_instance_desc = {};
                switch (geometry_handle.type()) {
                case RenderObjectType::triangle_geometry:
                    m_triangle_geometries.write_geometry_instance_desc(
                        RenderTriangleGeometryHandle(geometry_handle),
                        geometry_instance_desc
                    );
                    break;
                case RenderObjectType::lss_geometry:
                    m_lss_geometries.write_geometry_instance_desc(
                        RenderLSSGeometryHandle(geometry_handle),
                        geometry_instance_desc
                    );
                    break;
                default:
                    FALCOR_THROW("Unsupported geometry type in geometry instance");
                }
                geometry_instance_desc.header.transform_id = transform_id;
                geometry_instance_desc.header.material_id = geometry_index < geometry_instance.material_ids.size()
                    ? geometry_instance.material_ids[geometry_index]
                    : shared::MaterialID{0};
                m_geometry_instance_descs[geometry_instance_desc_index] = geometry_instance_desc;
                ++geometry_instance_desc_index;
            }
            ++tlas_instance_desc_index;
        }
        m_instance_data.mark_all_dirty();
        m_instance_data.update_buffer(ctx.command_encoder());
        m_geometry_instance_descs.mark_all_dirty();
        m_geometry_instance_descs.update_buffer(ctx.command_encoder());
    }

    // Build TLAS.
    build_tlas(ctx);

    bool changed = m_triangle_geometries.m_pool.is_dirty() || m_lss_geometries.m_pool.is_dirty()
        || m_geometry_group_pool.is_dirty() || m_geometry_instance_pool.is_dirty() || m_transform_pool.is_dirty();

    // Reset dirty flags.
    m_triangle_geometries.m_pool.reset_dirty_flags();
    m_lss_geometries.m_pool.reset_dirty_flags();
    m_geometry_group_pool.reset_dirty_flags();
    m_geometry_instance_pool.reset_dirty_flags();
    m_transform_pool.reset_dirty_flags();

    return changed;
}

void RenderScene::update_blases(UpdateContext& ctx)
{
    // Update acceleration structures.
    for (const auto& [handle, geometry_group] : m_geometry_group_pool) {
        // Build BLAS for the geometry group.
        std::vector<sgl::AccelerationStructureBuildInput> build_inputs;
        for (const RenderHandle& geometry_handle : geometry_group.geometries) {
            sgl::AccelerationStructureBuildInput input;
            switch (geometry_handle.type()) {
            case RenderObjectType::triangle_geometry:
                m_triangle_geometries.write_acceleration_structure_build_input(
                    RenderTriangleGeometryHandle{geometry_handle},
                    input
                );
                break;
            case RenderObjectType::lss_geometry:
                m_lss_geometries.write_acceleration_structure_build_input(
                    RenderLSSGeometryHandle{geometry_handle},
                    input
                );
                break;
            default:
                FALCOR_THROW("Unsupported geometry type in geometry group");
            }
            build_inputs.push_back(input);
        }

        sgl::AccelerationStructureBuildDesc build_desc;
        build_desc.inputs = build_inputs;

        sgl::AccelerationStructureSizes sizes = m_device->get_acceleration_structure_sizes(build_desc);

        geometry_group.blas = m_device->create_acceleration_structure({
            .size = sizes.acceleration_structure_size,
            .label = fmt::format("GeometryGroup{}", handle.index()),
        });

        sgl::Buffer* scratch_buffer = get_scratch_buffer(sizes.scratch_size);

        ctx.command_encoder()->build_acceleration_structure(build_desc, geometry_group.blas, nullptr, scratch_buffer);

        ctx.submit_command_encoder();
    }
}

void RenderScene::build_tlas(UpdateContext& ctx)
{
    bool dirty_transforms
        = (m_geometry_instance_pool.combined_dirty_flags() & RenderGeometryInstanceDirtyFlags::transform)
        == RenderGeometryInstanceDirtyFlags::transform;
    bool dirty_instance_data
        = (m_geometry_instance_pool.combined_dirty_flags() != RenderGeometryInstanceDirtyFlags::none);

    rhi::AccelerationStructureInstanceDescType native_instance_desc_type
        = rhi::getAccelerationStructureInstanceDescType(static_cast<rhi::DeviceType>(m_device->type()));
    size_t native_instance_desc_size = rhi::getAccelerationStructureInstanceDescSize(native_instance_desc_type);

    if (dirty_instance_data) {
        if (!m_tlas_native_instance_descs_buffer
            || m_tlas_native_instance_descs_buffer->size() < m_tlas_instance_descs.size() * native_instance_desc_size) {
            m_tlas_native_instance_descs_buffer = m_device->create_buffer({
                .size = m_tlas_instance_descs.size() * native_instance_desc_size,
                .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::acceleration_structure_build_input,
                .label = "m_tlas_native_instance_descs_buffer",
            });
        }

        if (m_tlas_native_instance_descs.size() < m_tlas_instance_descs.size() * native_instance_desc_size)
            m_tlas_native_instance_descs.resize(m_tlas_instance_descs.size() * native_instance_desc_size);

        rhi::convertAccelerationStructureInstanceDescs(
            m_tlas_instance_descs.size(),
            native_instance_desc_type,
            m_tlas_native_instance_descs.data(),
            native_instance_desc_size,
            reinterpret_cast<const rhi::AccelerationStructureInstanceDescGeneric*>(m_tlas_instance_descs.data()),
            sizeof(rhi::AccelerationStructureInstanceDescGeneric)
        );

        // Make sure any BLAS builds are finished.
        // TODO(scene): we should skip this if possible
        ctx.submit_command_encoder();
        ctx.command_encoder()->upload_buffer_data(
            m_tlas_native_instance_descs_buffer,
            0,
            m_tlas_native_instance_descs.size(),
            m_tlas_native_instance_descs.data()
        );
    }

    if (dirty_transforms) {
        m_update_tlas_instance_desc_transforms_kernel->dispatch(
            uint3(m_tlas_instance_descs.size(), 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                cursor = cursor.find_entry_point(0);
                cursor["instance_data"] = m_instance_data.buffer();
                cursor["world_from_object"] = m_world_from_object.buffer();
                cursor["as_instance_descs"] = m_tlas_native_instance_descs_buffer;
                cursor["first"] = uint32_t{0};
                cursor["count"] = sgl::narrow_cast<uint32_t>(m_tlas_instance_descs.size());
            },
            ctx.command_encoder()
        );
    }

    if (m_tlas_instance_descs.size() > 0) {
        sgl::AccelerationStructureBuildInputInstances build_input_instances;
        build_input_instances.instance_buffer = m_tlas_native_instance_descs_buffer;
        build_input_instances.instance_count = sgl::narrow_cast<uint32_t>(m_tlas_instance_descs.size());
        build_input_instances.instance_stride = sgl::narrow_cast<uint32_t>(native_instance_desc_size);
        sgl::AccelerationStructureBuildDesc build_desc;
        build_desc.inputs.push_back(build_input_instances);
        build_desc.flags = sgl::AccelerationStructureBuildFlags::allow_update
            | sgl::AccelerationStructureBuildFlags::prefer_fast_trace;

        // Use TLAS refit when only transforms changed and refit is preferred.
        bool transforms_only = dirty_transforms
            && (m_geometry_instance_pool.combined_dirty_flags() == RenderGeometryInstanceDirtyFlags::transform);
        bool refit = transforms_only && m_tlas_prefer_refit;
        if (refit)
            build_desc.mode = sgl::AccelerationStructureBuildMode::update;

        sgl::AccelerationStructureSizes sizes = m_device->get_acceleration_structure_sizes(build_desc);

        if (!m_tlas || m_tlas->desc().size < sizes.acceleration_structure_size) {
            m_tlas = m_device->create_acceleration_structure({
                .size = sizes.acceleration_structure_size,
                .label = "m_tlas",
            });
            // New TLAS needs a full build.
            refit = false;
            build_desc.mode = sgl::AccelerationStructureBuildMode::build;
        }

        size_t scratch_size = refit ? sizes.update_scratch_size : sizes.scratch_size;
        sgl::Buffer* scratch_buffer = get_scratch_buffer(scratch_size);

        ctx.command_encoder()
            ->build_acceleration_structure(build_desc, m_tlas, refit ? m_tlas.get() : nullptr, scratch_buffer);
    } else {
        m_tlas = nullptr;
    }
}

} // namespace falcor
