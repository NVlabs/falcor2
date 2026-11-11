// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_importer.h"

#include "falcor2/core/types.h"
#include "falcor2/importers/importer_types.h"

#include "falcor2/external/tiny_gltf.h"

#include <sgl/core/error.h>
#include <sgl/core/logger.h>
#include <sgl/core/short_vector.h>
#include <sgl/core/bitmap.h>
#include <sgl/core/memory_stream.h>
#include <sgl/core/thread.h>
#include <sgl/core/type_utils.h>

#include <sgl/math/matrix.h>

#include <stdexcept>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <sstream>
#include <optional>

namespace falcor {

//----------------------------------------------------------------------------
// Utility functions for reading values from tinygltf JSON extension objects
//----------------------------------------------------------------------------

static std::optional<float> read_float(const tinygltf::Value::Object& obj, const std::string& key)
{
    auto it = obj.find(key);
    if (it != obj.end() && it->second.IsNumber()) {
        return static_cast<float>(it->second.GetNumberAsDouble());
    }
    return std::nullopt;
}

static std::optional<float2> read_float2(const tinygltf::Value::Object& obj, const std::string& key)
{
    auto it = obj.find(key);
    if (it != obj.end() && it->second.IsArray()) {
        const auto& array = it->second.Get<tinygltf::Value::Array>();
        if (array.size() == 2 && array[0].IsNumber() && array[1].IsNumber()) {
            return float2(
                static_cast<float>(array[0].GetNumberAsDouble()),
                static_cast<float>(array[1].GetNumberAsDouble())
            );
        }
    }
    return std::nullopt;
}

static std::optional<float3> read_float3(const tinygltf::Value::Object& obj, const std::string& key)
{
    auto it = obj.find(key);
    if (it != obj.end() && it->second.IsArray()) {
        const auto& array = it->second.Get<tinygltf::Value::Array>();
        if (array.size() == 3 && array[0].IsNumber() && array[1].IsNumber() && array[2].IsNumber()) {
            return float3(
                static_cast<float>(array[0].GetNumberAsDouble()),
                static_cast<float>(array[1].GetNumberAsDouble()),
                static_cast<float>(array[2].GetNumberAsDouble())
            );
        }
    }
    return std::nullopt;
}

static std::optional<float4> read_float4(const tinygltf::Value::Object& obj, const std::string& key)
{
    auto it = obj.find(key);
    if (it != obj.end() && it->second.IsArray()) {
        const auto& array = it->second.Get<tinygltf::Value::Array>();
        if (array.size() == 4 && array[0].IsNumber() && array[1].IsNumber() && array[2].IsNumber()
            && array[3].IsNumber()) {
            return float4(
                static_cast<float>(array[0].GetNumberAsDouble()),
                static_cast<float>(array[1].GetNumberAsDouble()),
                static_cast<float>(array[2].GetNumberAsDouble()),
                static_cast<float>(array[3].GetNumberAsDouble())
            );
        }
    }
    return std::nullopt;
}

//----------------------------------------------------------------------------
// Utility functions for reading values from std::vector<double> (tinygltf arrays)
//----------------------------------------------------------------------------

static std::optional<float3> read_float3(const std::vector<double>& vec)
{
    if (vec.size() == 3) {
        return float3(static_cast<float>(vec[0]), static_cast<float>(vec[1]), static_cast<float>(vec[2]));
    }
    return std::nullopt;
}

static std::optional<float4> read_float4(const std::vector<double>& vec)
{
    if (vec.size() == 4) {
        return float4(
            static_cast<float>(vec[0]),
            static_cast<float>(vec[1]),
            static_cast<float>(vec[2]),
            static_cast<float>(vec[3])
        );
    }
    return std::nullopt;
}

//----------------------------------------------------------------------------
// Wrapper functions to read values from tinygltf JSON objects and store them
// as material parameters
//----------------------------------------------------------------------------

static void read_float_param(
    const tinygltf::Value::Object& obj,
    const std::string& key,
    Properties& params,
    const std::string& param_name,
    float default_value = 0.0f
)
{
    if (auto value = read_float(obj, key)) {
        params.set(param_name, *value);
    } else {
        params.set(param_name, default_value);
    }
}

static void read_float2_param(
    const tinygltf::Value::Object& obj,
    const std::string& key,
    Properties& params,
    const std::string& param_name,
    const float2& default_value = float2(0.0f)
)
{
    if (auto value = read_float2(obj, key)) {
        params.set(param_name, *value);
    } else {
        params.set(param_name, default_value);
    }
}

static void read_float3_param(
    const tinygltf::Value::Object& obj,
    const std::string& key,
    Properties& params,
    const std::string& param_name,
    const float3& default_value = float3(0.0f)
)
{
    if (auto value = read_float3(obj, key)) {
        params.set(param_name, *value);
    } else {
        params.set(param_name, default_value);
    }
}

static void read_float4_param(
    const tinygltf::Value::Object& obj,
    const std::string& key,
    Properties& params,
    const std::string& param_name,
    const float4& default_value = float4(0.0f)
)
{
    if (auto value = read_float4(obj, key)) {
        params.set(param_name, *value);
    } else {
        params.set(param_name, default_value);
    }
}

static void read_float4_param(
    const std::vector<double>& vec,
    Properties& params,
    const std::string& param_name,
    const float4& default_value = float4(0.0f)
)
{
    if (auto value = read_float4(vec)) {
        params.set(param_name, *value);
    } else {
        params.set(param_name, default_value);
    }
}


GltfImporter::GltfImporter() = default;

GltfImporter::~GltfImporter() = default;

// Helper structure to hold attribute accessor information
struct AttributeInfo {
    const tinygltf::Accessor* accessor = nullptr;
    const unsigned char* data_ptr = nullptr;
    size_t stride = 0;

    AttributeInfo() = default;

    AttributeInfo(const tinygltf::Model& model, const tinygltf::Accessor& acc)
        : accessor(&acc)
    {
        const auto& buffer_view = model.bufferViews[acc.bufferView];
        const auto& buffer = model.buffers[buffer_view.buffer];

        data_ptr = buffer.data.data() + buffer_view.byteOffset + acc.byteOffset;
        stride = buffer_view.byteStride;

        if (stride == 0) {
            // Tightly packed
            stride = tinygltf::GetComponentSizeInBytes(acc.componentType) * tinygltf::GetNumComponentsInType(acc.type);
        }
    }

    bool is_valid() const { return accessor != nullptr; }

    const float* get_float_data(size_t vertex_index) const
    {
        if (!is_valid() || accessor->componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
            return nullptr;
        }
        return reinterpret_cast<const float*>(data_ptr + vertex_index * stride);
    }

    const uint8_t* get_uint8_data(size_t index) const
    {
        if (!is_valid() || accessor->componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            return nullptr;
        }
        return reinterpret_cast<const uint8_t*>(data_ptr + index * stride);
    }

    const uint16_t* get_uint16_data(size_t index) const
    {
        if (!is_valid() || accessor->componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            return nullptr;
        }
        return reinterpret_cast<const uint16_t*>(data_ptr + index * stride);
    }

    const uint32_t* get_uint32_data(size_t index) const
    {
        if (!is_valid() || accessor->componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            return nullptr;
        }
        return reinterpret_cast<const uint32_t*>(data_ptr + index * stride);
    }

    ImporterJointIndices get_joint_data(size_t vertex_index) const
    {
        SGL_CHECK(
            is_valid() && accessor->type == TINYGLTF_TYPE_VEC4,
            "Joint accessor must contain four-component vectors"
        );

        ImporterJointIndices result;
        if (const uint8_t* uint8_data = get_uint8_data(vertex_index)) {
            for (size_t i = 0; i < 4; ++i)
                result.values[i] = uint8_data[i];
        } else if (const uint16_t* uint16_data = get_uint16_data(vertex_index)) {
            for (size_t i = 0; i < 4; ++i)
                result.values[i] = uint16_data[i];
        } else {
            SGL_THROW("Joint accessor must use unsigned byte or unsigned short components");
        }
        return result;
    }

    float4 get_weight_data(size_t vertex_index) const
    {
        SGL_CHECK(
            is_valid() && accessor->type == TINYGLTF_TYPE_VEC4,
            "Weight accessor must contain four-component vectors"
        );

        if (const float* data = get_float_data(vertex_index)) {
            return float4(data[0], data[1], data[2], data[3]);
        }
        if (const uint8_t* data = get_uint8_data(vertex_index)) {
            SGL_CHECK(accessor->normalized, "Unsigned byte weight accessors must be normalized");
            return float4(data[0], data[1], data[2], data[3]) * (1.f / 255.f);
        }
        if (const uint16_t* data = get_uint16_data(vertex_index)) {
            SGL_CHECK(accessor->normalized, "Unsigned short weight accessors must be normalized");
            return float4(data[0], data[1], data[2], data[3]) * (1.f / 65535.f);
        }
        SGL_THROW("Weight accessor must use float, normalized unsigned byte, or normalized unsigned short components");
    }

    // Helper function to load color data with proper type conversion
    float4 get_color_data(size_t vertex_index) const
    {
        if (!is_valid()) {
            return float4(1.0f); // Default to white
        }

        int num_components = tinygltf::GetNumComponentsInType(accessor->type);

        if (accessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
            const float* color_data = get_float_data(vertex_index);
            if (!color_data)
                return float4(1.0f);

            if (num_components == 3) {
                return float4(color_data[0], color_data[1], color_data[2], 1.0f);
            } else if (num_components == 4) {
                return float4(color_data[0], color_data[1], color_data[2], color_data[3]);
            }
        } else if (accessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            const uint8_t* color_data = get_uint8_data(vertex_index);
            if (!color_data)
                return float4(1.0f);

            // Convert from [0, 255] to [0, 1]
            if (num_components == 3) {
                return float4(color_data[0] / 255.0f, color_data[1] / 255.0f, color_data[2] / 255.0f, 1.0f);
            } else if (num_components == 4) {
                return float4(
                    color_data[0] / 255.0f,
                    color_data[1] / 255.0f,
                    color_data[2] / 255.0f,
                    color_data[3] / 255.0f
                );
            }
        } else if (accessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const uint16_t* color_data = get_uint16_data(vertex_index);
            if (!color_data)
                return float4(1.0f);

            // Convert from [0, 65535] to [0, 1]
            if (num_components == 3) {
                return float4(color_data[0] / 65535.0f, color_data[1] / 65535.0f, color_data[2] / 65535.0f, 1.0f);
            } else if (num_components == 4) {
                return float4(
                    color_data[0] / 65535.0f,
                    color_data[1] / 65535.0f,
                    color_data[2] / 65535.0f,
                    color_data[3] / 65535.0f
                );
            }
        }

        return float4(1.0f); // Default to white for unsupported formats
    }
};

static float4x4 load_matrix(const AttributeInfo& attribute, size_t index)
{
    SGL_CHECK(
        attribute.is_valid() && attribute.accessor->type == TINYGLTF_TYPE_MAT4,
        "Matrix accessor must contain 4-by-4 matrices"
    );
    const float* data = attribute.get_float_data(index);
    SGL_CHECK(data, "Matrix accessor must use float components");

    // glTF stores matrices column-major; SGL constructors take row-major coefficients.
    return float4x4{
        data[0],
        data[4],
        data[8],
        data[12],
        data[1],
        data[5],
        data[9],
        data[13],
        data[2],
        data[6],
        data[10],
        data[14],
        data[3],
        data[7],
        data[11],
        data[15]
    };
}

// Helper function for loading triangle indices
static uint3 load_triangle_indices(const AttributeInfo& index_attr, size_t triangle_index)
{
    size_t base_index = triangle_index * 3;

    if (const uint8_t* uint8_data = index_attr.get_uint8_data(base_index)) {
        return uint3(uint8_data[0], uint8_data[1], uint8_data[2]);
    } else if (const uint16_t* uint16_data = index_attr.get_uint16_data(base_index)) {
        return uint3(uint16_data[0], uint16_data[1], uint16_data[2]);
    } else if (const uint32_t* uint32_data = index_attr.get_uint32_data(base_index)) {
        return uint3(uint32_data[0], uint32_data[1], uint32_data[2]);
    } else {
        SGL_THROW("Unsupported index component type: " + std::to_string(index_attr.accessor->componentType));
    }
}

// Custom image loader just marks image 'as_is' and copies data, so it can be
// loaded later when needed.
static bool image_loader(
    tinygltf::Image* image,
    const int image_idx,
    std::string* err,
    std::string* warn,
    int req_width,
    int req_height,
    const unsigned char* bytes,
    int size,
    void* user_data
)
{
    SGL_UNUSED(image_idx);
    SGL_UNUSED(err);
    SGL_UNUSED(warn);
    SGL_UNUSED(req_width);
    SGL_UNUSED(req_height);
    SGL_UNUSED(user_data);
    image->as_is = true;
    image->image.resize(size);
    std::memcpy(image->image.data(), bytes, size);
    return true;
}

ref<ImporterScene> GltfImporter::load_scene(const std::filesystem::path& path)
{
    return load_scene(path, ImportOptions{});
}

ref<ImporterScene> GltfImporter::load_scene(const std::filesystem::path& path, const ImportOptions& import_options)
{
    m_path = path;
    m_source_asset_path.reset(std::filesystem::absolute(path).lexically_normal().string());
    m_scene = make_ref<ImporterScene>();
    m_model = std::make_unique<tinygltf::Model>();
    m_mesh_index_map.clear();

    if (!load_gltf_file(path)) {
        return nullptr;
    }

    extract_textures();
    extract_materials(import_options);
    extract_meshes(import_options);
    extract_lines(import_options);
    extract_cameras();
    extract_nodes();
    build_node_hierarchy();
    extract_skins();
    extract_animations();

    detail::apply_import_options(*m_scene, import_options);

    return m_scene;
}

// Extract cameras from GLTF and populate m_scene->cameras
void GltfImporter::extract_cameras()
{
    const tinygltf::Model& model = *m_model;
    m_scene->cameras.reserve(model.cameras.size());
    for (size_t cam_idx = 0; cam_idx < model.cameras.size(); ++cam_idx) {
        const auto& gltf_camera = model.cameras[cam_idx];
        ImporterCamera camera;
        camera.name = gltf_camera.name.empty() ? ("camera_" + std::to_string(cam_idx)) : gltf_camera.name;
        camera.source_asset_ref = source_asset_ref_for_object(fmt::format("/cameras/{}", cam_idx));
        // Defaults
        camera.focal_length = 50.0f;
        camera.fstop = 8.0f;
        camera.sensor_size_mm = 24.f;
        camera.enable_depth_of_field = false;
        camera.focus_distance = 1.0f;
        camera.depth_range = float2(0.01f, 1000.0f);
        // Projection
        if (gltf_camera.type == "perspective") {
            camera.projection = ImporterCamera::Projection::perspective;
            camera.fov_direction = ImporterCamera::FOVDirection::vertical;
            if (gltf_camera.perspective.aspectRatio > 0.0) {
                // Could use aspectRatio if needed
            }
            if (gltf_camera.perspective.yfov > 0.0) {
                camera.focal_length = ImporterCamera::focal_length_from_fov_degrees(
                    sgl::math::degrees(static_cast<float>(gltf_camera.perspective.yfov)),
                    camera.sensor_size_mm
                );
            }
            if (gltf_camera.perspective.znear > 0.0) {
                camera.depth_range.x = static_cast<float>(gltf_camera.perspective.znear);
            }
            if (gltf_camera.perspective.zfar > 0.0) {
                camera.depth_range.y = static_cast<float>(gltf_camera.perspective.zfar);
            }
        } else if (gltf_camera.type == "orthographic") {
            camera.projection = ImporterCamera::Projection::orthographic;
            camera.fov_direction = ImporterCamera::FOVDirection::vertical;
            camera.depth_range.x = static_cast<float>(gltf_camera.orthographic.znear);
            camera.depth_range.y = static_cast<float>(gltf_camera.orthographic.zfar);
        }
        m_scene->cameras.push_back(std::move(camera));
    }
}

bool GltfImporter::load_gltf_file(const std::filesystem::path& path)
{
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    std::string path_str = path.string();

    // Check file extension to determine loading method
    std::string ext = path.extension().string();
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(c));
    }

    // Defer texture loading until needed
    loader.SetImageLoader(image_loader, this);

    if (ext == ".glb") {
        ret = loader.LoadBinaryFromFile(m_model.get(), &err, &warn, path_str);
    } else if (ext == ".gltf") {
        ret = loader.LoadASCIIFromFile(m_model.get(), &err, &warn, path_str);
    } else {
        SGL_THROW("Unsupported GLTF file extension: " + ext);
    }

    if (!err.empty()) {
        SGL_THROW("GLTF loading error: " + err);
    }

    if (!ret) {
        SGL_THROW("Failed to parse GLTF file: " + path_str);
    }

    return true;
}

namespace {

using Double3 = sgl::math::vector<double, 3>;
constexpr double HEMISPHERE_TOLERANCE = 1e-4;

Double3 repair_face_normal(const ImporterMesh& mesh, const uint3& triangle)
{
    SGL_CHECK(
        triangle.x < mesh.vertex_count() && triangle.y < mesh.vertex_count() && triangle.z < mesh.vertex_count(),
        "Invalid triangle indices in glTF mesh '{}'",
        mesh.name
    );
    const auto positions = mesh.position_stream();
    const Double3 e0 = Double3(positions[triangle.y]) - Double3(positions[triangle.x]);
    const Double3 e1 = Double3(positions[triangle.z]) - Double3(positions[triangle.x]);
    const Double3 e2 = e1 - e0;
    const Double3 face = cross(e0, e1);
    // Ignore nearly collinear faces, independently of source scale.
    const double min_length = 1e-4 * std::max({dot(e0, e0), dot(e1, e1), dot(e2, e2)});
    return std::isfinite(dot(face, face)) && dot(face, face) > min_length * min_length ? face : Double3(0.0);
}

std::vector<uint8_t> invalid_vertex_normals(const ImporterMesh& mesh)
{
    const auto normals = mesh.normal_stream();
    std::vector<uint8_t> invalid(mesh.vertex_count(), false);
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const Double3 n(normals[i]);
        const double length_squared = sgl::math::dot(n, n);
        invalid[i] = !std::isfinite(length_squared) || length_squared == 0.0;
    }

    // Ignore roundoff around a tangent normal. Use double precision so that
    // source scale does not underflow/overflow the geometric comparison.
    for (const auto& subgeometry : mesh.subgeometries) {
        for (const auto& triangle : subgeometry.indices) {
            const Double3 face_normal = repair_face_normal(mesh, triangle);
            const double face_length_squared = sgl::math::dot(face_normal, face_normal);
            if (face_length_squared == 0.0)
                continue;
            for (uint32_t vertex_index : {triangle.x, triangle.y, triangle.z}) {
                if (invalid[vertex_index])
                    continue;
                const Double3 n(normals[vertex_index]);
                const double scale = std::sqrt(sgl::math::dot(n, n) * face_length_squared);
                if (sgl::math::dot(n, face_normal) < -HEMISPHERE_TOLERANCE * scale)
                    invalid[vertex_index] = true;
            }
        }
    }
    return invalid;
}

void repair_vertex_normals(ImporterMesh& mesh, std::vector<uint8_t>& repair)
{
    const size_t source_count = mesh.vertex_count();
    std::vector<Double3> sums(source_count, Double3(0.0));
    std::vector<std::vector<Double3>> incident_faces(source_count);
    for (const auto& subgeometry : mesh.subgeometries) {
        for (const auto& triangle : subgeometry.indices) {
            const Double3 face = repair_face_normal(mesh, triangle);
            if (dot(face, face) == 0.0)
                continue;
            for (uint32_t vertex : {triangle.x, triangle.y, triangle.z}) {
                if (repair[vertex]) {
                    sums[vertex] += face;
                    incident_faces[vertex].push_back(normalize(face));
                }
            }
        }
    }
    std::vector<uint8_t> split(source_count, false);
    auto normals = mesh.normal_stream();
    for (size_t vertex = 0; vertex < source_count; ++vertex) {
        if (!repair[vertex])
            continue;
        const auto& faces = incident_faces[vertex];
        // Preserve finite authored shading directions when only a neighboring
        // triangle's hemisphere is violated. Replacing them with a face average
        // can introduce a large discontinuity beside unrepaired vertices.
        Double3 n(normals[vertex]);
        const double length_squared = dot(n, n);
        if (!std::isfinite(length_squared) || length_squared == 0.0)
            n = sums[vertex];
        if (dot(n, n) == 0.0)
            n = faces.empty() ? Double3(0.0, 0.0, 1.0) : faces.front();
        n = normalize(n);
        // Project into the incident hemispheres before resorting to corner splits.
        for (int iteration = 0; iteration < 64; ++iteration) {
            bool valid = true;
            for (const Double3 face : faces) {
                const double alignment = dot(n, face);
                if (alignment < -HEMISPHERE_TOLERANCE * 0.5) {
                    n = normalize(n + (1e-6 - alignment) * face);
                    valid = false;
                }
            }
            if (valid)
                break;
        }
        normals[vertex] = float3(n);
        split[vertex] = std::any_of(
            faces.begin(),
            faces.end(),
            [&](Double3 face)
            {
                return dot(Double3(normals[vertex]), face) < -HEMISPHERE_TOLERANCE;
            }
        );
    }

    struct CornerRepair {
        uint32_t source;
        float3 normal;
    };
    std::vector<CornerRepair> corners;
    // Preserve existing attribute seams. Only vertices without a usable shared
    // normal need per-corner replacements; every other index remains unchanged.
    for (auto& subgeometry : mesh.subgeometries) {
        for (auto& triangle : subgeometry.indices) {
            const Double3 face = repair_face_normal(mesh, triangle);
            if (dot(face, face) == 0.0)
                continue;
            for (int corner = 0; corner < 3; ++corner) {
                const uint32_t source = triangle[corner];
                if (split[source]) {
                    triangle[corner] = sgl::narrow_cast<uint32_t>(source_count + corners.size());
                    corners.push_back({source, float3(normalize(face))});
                }
            }
        }
    }
    if (!corners.empty()) {
        mesh.allocate_vertices(corners.size());
        normals = mesh.normal_stream();
        for (size_t i = 0; i < corners.size(); ++i) {
            for (auto& buffer : mesh.buffers()) {
                std::memcpy(
                    buffer.data.data() + (source_count + i) * buffer.stride,
                    buffer.data.data() + corners[i].source * buffer.stride,
                    buffer.stride
                );
            }
            normals[source_count + i] = corners[i].normal;
        }
        repair.resize(mesh.vertex_count(), true);
    }
}

void repair_vertex_tangents(ImporterMesh& mesh, const std::vector<uint8_t>& repair, bool use_uvs)
{
    // MikkTSpace may split a vertex for mirrored UVs. Retain source corner IDs
    // to restore untouched authored frames after generation and before deduplication.
    const auto tangents = mesh.tangent_stream();
    const auto handedness = mesh.handedness_stream();
    std::vector<float4> authored(mesh.vertex_count());
    for (size_t i = 0; i < authored.size(); ++i)
        authored[i] = float4(tangents[i], handedness[i]);
    std::vector<uint32_t> source_corners;
    for (const auto& subgeometry : mesh.subgeometries)
        for (const auto& triangle : subgeometry.indices)
            source_corners.insert(source_corners.end(), {triangle.x, triangle.y, triangle.z});
    if (use_uvs)
        mesh.add_tangents_from_uvs();
    else
        mesh.add_tangents_from_normals();
    auto generated = mesh.tangent_stream();
    auto signs = mesh.handedness_stream();
    for (size_t i = 0; i < authored.size(); ++i) {
        if (!repair[i]) {
            generated[i] = authored[i].xyz();
            signs[i] = authored[i].w;
        }
    }
    size_t corner = 0;
    for (const auto& subgeometry : mesh.subgeometries) {
        for (const auto& triangle : subgeometry.indices) {
            for (uint32_t vertex : {triangle.x, triangle.y, triangle.z}) {
                const uint32_t source = source_corners[corner++];
                if (!repair[source]) {
                    generated[vertex] = authored[source].xyz();
                    signs[vertex] = authored[source].w;
                }
            }
        }
    }
}

void repair_invalid_vertex_normals(
    ImporterMesh& mesh,
    const std::filesystem::path& path,
    bool authored_tangents,
    bool use_uvs
)
{
    auto repair = invalid_vertex_normals(mesh);
    const size_t count = std::count(repair.begin(), repair.end(), uint8_t(1));
    if (count == 0)
        return;
    const size_t source_vertices = mesh.vertex_count();
    repair_vertex_normals(mesh, repair);
    sgl::log_warn(
        "glTF '{}', mesh '{}': repaired {} invalid vertex normals; {} normal corner splits.",
        path.string(),
        mesh.name,
        count,
        mesh.vertex_count() - source_vertices
    );
    if (authored_tangents)
        repair_vertex_tangents(mesh, repair, use_uvs);
}

} // namespace

void GltfImporter::extract_meshes(const ImportOptions& import_options)
{
    const tinygltf::Model& model = *m_model;

    std::vector<ImporterMesh> importer_meshes(model.meshes.size());

    // Process each mesh in the GLTF file
    sgl::thread::parallel_for(
        sgl::thread::blocked_range<size_t>(0, model.meshes.size()),
        [&model, &importer_meshes, &import_options, this](const auto& range)
        {
            size_t mesh_idx = range.begin();
            const auto& gltf_mesh = model.meshes[mesh_idx];
            ImporterMesh& importer_mesh = importer_meshes[mesh_idx];
            importer_mesh.properties = ImporterMesh::make_properties(import_options);

            sgl::short_vector<ImporterVertexStream<float2>> tex_coord_streams;
            sgl::short_vector<AttributeInfo> tex_coord_attrs;
            sgl::short_vector<ImporterVertexStream<ImporterJointIndices>> joint_streams;
            sgl::short_vector<ImporterVertexStream<float4>> weight_streams;
            sgl::short_vector<AttributeInfo> joint_attrs;
            sgl::short_vector<AttributeInfo> weight_attrs;

            importer_mesh.name = gltf_mesh.name.empty() ? "mesh_" + std::to_string(mesh_idx) : gltf_mesh.name;
            importer_mesh.source_asset_ref = source_asset_ref_for_object(fmt::format("/meshes/{}", mesh_idx));

            // Iterate GLTF mesh attributes for each primitive and create corresponding importer attributes
            uint32_t num_tex_coords = 0;
            uint32_t num_skinning_sets = 0;
            std::vector<ImporterMeshDefaultAttribute> default_attribs;
            for (size_t prim_idx = 0; prim_idx < gltf_mesh.primitives.size(); ++prim_idx) {
                for (const auto& attribute : gltf_mesh.primitives[prim_idx].attributes) {
                    if (attribute.first == "POSITION") {
                        default_attribs.push_back({ImporterSemantic::position});
                    } else if (attribute.first == "NORMAL") {
                        default_attribs.push_back({ImporterSemantic::normal});
                    } else if (attribute.first == "TANGENT") {
                        default_attribs.push_back({ImporterSemantic::tangent});
                        default_attribs.push_back({ImporterSemantic::handedness});
                    } else if (attribute.first.find("TEXCOORD_") == 0) {
                        uint32_t idx = std::stoul(attribute.first.substr(9));
                        num_tex_coords = std::max(num_tex_coords, idx + 1);
                        default_attribs.push_back(ImporterMeshDefaultAttribute{ImporterSemantic::tex_coord, idx});
                    } else if (attribute.first.find("COLOR_") == 0) {
                        uint32_t idx = std::stoul(attribute.first.substr(6));
                        default_attribs.push_back(ImporterMeshDefaultAttribute{ImporterSemantic::color, idx, 4});
                    } else if (attribute.first.find("JOINTS_") == 0) {
                        uint32_t idx = std::stoul(attribute.first.substr(7));
                        num_skinning_sets = std::max(num_skinning_sets, idx + 1);
                        default_attribs.push_back(ImporterMeshDefaultAttribute{ImporterSemantic::joints, idx});
                    } else if (attribute.first.find("WEIGHTS_") == 0) {
                        uint32_t idx = std::stoul(attribute.first.substr(8));
                        num_skinning_sets = std::max(num_skinning_sets, idx + 1);
                        default_attribs.push_back(ImporterMeshDefaultAttribute{ImporterSemantic::weights, idx});
                    }
                }
                importer_mesh.ensure_attributes(default_attribs);
            }

            // Process each primitive (submesh) in the mesh
            bool all_have_uvs = true;
            bool all_have_normals = true;
            bool all_have_tangents = true;
            for (size_t prim_idx = 0; prim_idx < gltf_mesh.primitives.size(); ++prim_idx) {
                const auto& primitive = gltf_mesh.primitives[prim_idx];

                // Only handle triangle primitives for now
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                    continue;
                }

                ImporterMesh::Subgeometry subgeom;
                subgeom.name = importer_mesh.name + "_primitive_" + std::to_string(prim_idx);

                // Assign material index.
                if (primitive.material >= 0 && primitive.material < static_cast<int>(model.materials.size()))
                    subgeom.material_index = primitive.material;

                // Get position data (required)
                auto pos_it = primitive.attributes.find("POSITION");
                if (pos_it == primitive.attributes.end()) {
                    SGL_THROW(
                        "Missing POSITION attribute in primitive " + std::to_string(prim_idx) + " of mesh '"
                        + importer_mesh.name + "'"
                    );
                }

                const auto& pos_accessor = model.accessors[pos_it->second];
                size_t vertex_count = pos_accessor.count;
                size_t vertex_offset = importer_mesh.allocate_vertices(vertex_count, true);

                // Get the streams we want.
                // Note: This must happen after allocate_vertices() as it may reallocate internal buffers
                auto positions = importer_mesh.position_stream();
                auto normals = importer_mesh.normal_stream();
                auto tangents = importer_mesh.tangent_stream();
                auto handedness = importer_mesh.handedness_stream();
                auto colors = importer_mesh.get_stream<float4>(importer_mesh.color_attribute());

                joint_streams.resize(num_skinning_sets);
                weight_streams.resize(num_skinning_sets);
                for (uint32_t i = 0; i < num_skinning_sets; ++i) {
                    auto* joint_attribute = importer_mesh.find_attribute(ImporterSemantic::joints, i);
                    auto* weight_attribute = importer_mesh.find_attribute(ImporterSemantic::weights, i);
                    SGL_CHECK(
                        joint_attribute && weight_attribute,
                        "Mesh '{}' must provide matching JOINTS_{} and WEIGHTS_{} attributes",
                        importer_mesh.name,
                        i,
                        i
                    );
                    joint_streams[i] = importer_mesh.get_stream<ImporterJointIndices>(*joint_attribute);
                    weight_streams[i] = importer_mesh.get_stream<float4>(*weight_attribute);
                }

                // Also load all texcoord sets
                tex_coord_streams.resize(num_tex_coords);
                for (uint32_t i = 0; i < num_tex_coords; ++i) {
                    auto attrib = importer_mesh.find_attribute(ImporterSemantic::tex_coord, i);
                    SGL_CHECK(
                        attrib,
                        "Failed to find TEXCOORD_" + std::to_string(i) + " attribute in mesh '" + importer_mesh.name
                            + "'"
                    );
                    tex_coord_streams[i] = importer_mesh.get_stream<float2>(*attrib);
                }

                // Prepare attribute accessors
                AttributeInfo pos_attr(model, pos_accessor);

                AttributeInfo normal_attr;
                auto norm_it = primitive.attributes.find("NORMAL");
                if (norm_it != primitive.attributes.end()) {
                    normal_attr = AttributeInfo(model, model.accessors[norm_it->second]);
                } else {
                    all_have_normals = false;
                }

                tex_coord_attrs.clear();
                tex_coord_attrs.resize(num_tex_coords);
                for (uint32_t i = 0; i < num_tex_coords; ++i) {
                    std::string attr_name = fmt::format("TEXCOORD_{}", i);
                    auto uv_it = primitive.attributes.find(attr_name);
                    if (uv_it != primitive.attributes.end()) {
                        tex_coord_attrs[i] = AttributeInfo(model, model.accessors[uv_it->second]);
                    }
                }
                if (num_tex_coords == 0 || !tex_coord_attrs[0].is_valid()) {
                    all_have_uvs = false;
                }

                AttributeInfo tangent_attr;
                auto tan_it = primitive.attributes.find("TANGENT");
                if (tan_it != primitive.attributes.end()) {
                    tangent_attr = AttributeInfo(model, model.accessors[tan_it->second]);
                } else {
                    all_have_tangents = false;
                }

                AttributeInfo color_attr;
                auto color_it = primitive.attributes.find("COLOR_0");
                if (color_it != primitive.attributes.end()) {
                    color_attr = AttributeInfo(model, model.accessors[color_it->second]);
                }

                joint_attrs.clear();
                joint_attrs.resize(num_skinning_sets);
                weight_attrs.clear();
                weight_attrs.resize(num_skinning_sets);
                for (uint32_t i = 0; i < num_skinning_sets; ++i) {
                    const std::string joints_name = fmt::format("JOINTS_{}", i);
                    const std::string weights_name = fmt::format("WEIGHTS_{}", i);
                    auto joints_it = primitive.attributes.find(joints_name);
                    auto weights_it = primitive.attributes.find(weights_name);
                    SGL_CHECK(
                        (joints_it == primitive.attributes.end()) == (weights_it == primitive.attributes.end()),
                        "Primitive {} of mesh '{}' must provide both {} and {}",
                        prim_idx,
                        importer_mesh.name,
                        joints_name,
                        weights_name
                    );
                    if (joints_it != primitive.attributes.end()) {
                        const auto& joint_accessor = model.accessors[joints_it->second];
                        const auto& weight_accessor = model.accessors[weights_it->second];
                        SGL_CHECK(
                            joint_accessor.count == vertex_count && weight_accessor.count == vertex_count,
                            "Skinning attribute count does not match POSITION count in primitive {} of mesh '{}'",
                            prim_idx,
                            importer_mesh.name
                        );
                        joint_attrs[i] = AttributeInfo(model, joint_accessor);
                        weight_attrs[i] = AttributeInfo(model, weight_accessor);
                    }
                }

                // Extract all vertex attributes in a single loop
                for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {

                    // Position (required)
                    if (const float* pos_data = pos_attr.get_float_data(vertex_idx)) {
                        positions[vertex_offset + vertex_idx] = float3(pos_data[0], pos_data[1], pos_data[2]);
                    } else {
                        SGL_THROW("Failed to read position data for vertex " + std::to_string(vertex_idx));
                    }

                    // Normal (optional)
                    if (normal_attr.is_valid()) {
                        if (const float* normal_data = normal_attr.get_float_data(vertex_idx)) {
                            normals[vertex_offset + vertex_idx]
                                = float3(normal_data[0], normal_data[1], normal_data[2]);
                        }
                    }

                    // Texture coordinates (optional)
                    for (uint32_t i = 0; i < num_tex_coords; ++i) {
                        if (tex_coord_attrs[i].is_valid()) {
                            if (const float* uv_data = tex_coord_attrs[i].get_float_data(vertex_idx)) {
                                tex_coord_streams[i][vertex_offset + vertex_idx] = float2(uv_data[0], uv_data[1]);
                            }
                        }
                    }

                    // Tangent (optional)
                    if (tangent_attr.is_valid()) {
                        if (const float* tangent_data = tangent_attr.get_float_data(vertex_idx)) {
                            tangents[vertex_offset + vertex_idx]
                                = float3(tangent_data[0], tangent_data[1], tangent_data[2]);
                            handedness[vertex_offset + vertex_idx] = (tangent_data[3] > 0.0f) ? 1.0f : -1.0f;
                        }
                    }

                    // Vertex color (optional)
                    if (colors.valid())
                        colors[vertex_offset + vertex_idx] = color_attr.get_color_data(vertex_idx);

                    // Joint indices and weights (optional, but always paired).
                    for (uint32_t i = 0; i < num_skinning_sets; ++i) {
                        if (joint_attrs[i].is_valid()) {
                            joint_streams[i][vertex_offset + vertex_idx] = joint_attrs[i].get_joint_data(vertex_idx);
                            weight_streams[i][vertex_offset + vertex_idx] = weight_attrs[i].get_weight_data(vertex_idx);
                        }
                    }
                }

                // Extract indices
                std::vector<uint3> indices;
                if (primitive.indices >= 0) {
                    const auto& index_accessor = model.accessors[primitive.indices];
                    size_t index_count = index_accessor.count;

                    // Ensure we have triangles
                    if (index_count % 3 != 0) {
                        SGL_THROW(
                            "Index count " + std::to_string(index_count) + " is not divisible by 3 in primitive "
                            + std::to_string(prim_idx) + " of mesh '" + importer_mesh.name + "'"
                        );
                    }

                    indices.resize(index_count / 3);

                    // Create AttributeInfo for index data
                    AttributeInfo index_attr(model, index_accessor);

                    // Load all triangle indices using the helper function
                    for (size_t i = 0; i < indices.size(); ++i) {
                        indices[i] = load_triangle_indices(index_attr, i);
                    }
                } else {
                    // Generate indices for non-indexed triangles
                    if (vertex_count % 3 != 0) {
                        SGL_THROW(
                            "Vertex count " + std::to_string(vertex_count)
                            + " is not divisible by 3 for non-indexed primitive " + std::to_string(prim_idx)
                            + " of mesh '" + importer_mesh.name + "'"
                        );
                    }

                    indices.resize(vertex_count / 3);
                    for (size_t i = 0; i < indices.size(); ++i) {
                        indices[i] = uint3(i * 3, i * 3 + 1, i * 3 + 2);
                    }
                }

                // Add to the submesh
                subgeom.indices = std::move(indices);

                // Adjust indices to account for vertex offset
                for (auto& triangle : subgeom.indices) {
                    triangle.x += static_cast<uint32_t>(vertex_offset);
                    triangle.y += static_cast<uint32_t>(vertex_offset);
                    triangle.z += static_cast<uint32_t>(vertex_offset);
                }

                importer_mesh.subgeometries.push_back(std::move(subgeom));
            }

            if (import_options.repair_broken_geometry && all_have_normals) {
                repair_invalid_vertex_normals(importer_mesh, m_path, all_have_tangents, all_have_uvs);
            }
            if (!all_have_normals) {
                importer_mesh.add_normals_from_faces();
                // Tangents authored for the discarded normal frame are no longer valid.
                all_have_tangents = false;
            }
            if (!all_have_tangents && all_have_uvs) {
                importer_mesh.add_tangents_from_uvs();
            }
            importer_mesh.deduplicate_vertices();
            if (!all_have_tangents && !all_have_uvs) {
                importer_mesh.add_tangents_from_normals();
            }
        }
    );

    m_mesh_index_map.assign(model.meshes.size(), -1);
    for (size_t mesh_index = 0; mesh_index < importer_meshes.size(); ++mesh_index) {
        auto& importer_mesh = importer_meshes[mesh_index];
        if (!importer_mesh.subgeometries.empty()) {
            m_mesh_index_map[mesh_index] = sgl::narrow_cast<int>(m_scene->meshes.size());
            m_scene->meshes.push_back(std::move(importer_mesh));
        }
    }
}

void GltfImporter::extract_lines(const ImportOptions& import_options)
{
    const auto& model = *m_model;
    m_curve_index_map.assign(model.meshes.size(), {});
    for (size_t mesh_index = 0; mesh_index < model.meshes.size(); ++mesh_index) {
        const auto& mesh = model.meshes[mesh_index];
        for (size_t primitive_index = 0; primitive_index < mesh.primitives.size(); ++primitive_index) {
            const auto& primitive = mesh.primitives[primitive_index];
            if (primitive.mode != TINYGLTF_MODE_LINE)
                continue;
            const std::string name = fmt::format("{}_primitive_{}", mesh.name, primitive_index);
            // The curve importer currently has no authored normal/color or deformation streams.
            // Diagnose these unsupported cases instead of silently rendering altered attributes.
            if (primitive.attributes.size() != 1 || !primitive.attributes.contains("POSITION")
                || !primitive.targets.empty()) {
                sgl::log_warn("glTF line '{}': only static POSITION-only lines are currently supported", name);
                continue;
            }
            ImporterMaterial material;
            if (primitive.material >= 0) {
                SGL_CHECK_LT(size_t(primitive.material), model.materials.size());
                material = m_scene->materials[primitive.material];
            }
            const auto& params = material.params;
            if (params.get("alphaMode", AlphaMode::opaque) != AlphaMode::opaque
                || params.has_property("baseColorTexture") || params.has_property("emissiveTexture")
                || params.get("_type", std::string("gltf_pbrMetallicRoughness")) == "gltf_pbrSpecularGlossiness") {
                sgl::log_warn("glTF line '{}': only opaque constant-color materials are currently supported", name);
                continue;
            }

            ImporterCurve curve;
            curve.properties = ImporterCurve::make_properties(import_options);
            curve.name = name;
            curve.source_asset_ref
                = source_asset_ref_for_object(fmt::format("/meshes/{}/primitives/{}", mesh_index, primitive_index));
            const auto& accessor = model.accessors.at(primitive.attributes.at("POSITION"));
            SGL_CHECK_EQ(accessor.type, TINYGLTF_TYPE_VEC3);
            AttributeInfo positions(model, accessor);
            curve.resize_vertices(accessor.count);
            const auto output_positions = curve.position_stream();
            for (size_t i = 0; i < accessor.count; ++i) {
                const float* p = positions.get_float_data(i);
                SGL_CHECK(p, "glTF line '{}' requires float positions", name);
                output_positions[i] = float3(p[0], p[1], p[2]);
            }
            // glTF 3.9.7 leaves line size/style to the implementation. Use round
            // segments with a 1 mm local-space diameter, independent of the camera.
            const auto radii = curve.radius_stream();
            std::fill(radii.begin(), radii.end(), 0.0005f);
            if (primitive.indices >= 0) {
                const auto& index_accessor = model.accessors.at(primitive.indices);
                SGL_CHECK_EQ(index_accessor.type, TINYGLTF_TYPE_SCALAR);
                AttributeInfo indices(model, index_accessor);
                for (size_t i = 0; i < index_accessor.count; ++i) {
                    if (const auto* data8 = indices.get_uint8_data(i))
                        curve.indices.push_back(*data8);
                    else if (const auto* data16 = indices.get_uint16_data(i))
                        curve.indices.push_back(*data16);
                    else if (const auto* data32 = indices.get_uint32_data(i))
                        curve.indices.push_back(*data32);
                    else
                        SGL_THROW("glTF line '{}' has an invalid index component type", name);
                }
            } else {
                for (size_t i = 0; i < accessor.count; ++i)
                    curve.indices.push_back(sgl::narrow_cast<uint32_t>(i));
            }
            SGL_CHECK(
                !curve.indices.empty() && curve.indices.size() % 2 == 0,
                "glTF line '{}' requires a nonempty sequence of index pairs",
                name
            );
            for (uint32_t index : curve.indices)
                SGL_CHECK_LT(size_t(index), curve.vertex_count());
            curve.calculate_local_aabb();

            // Follow 3.9.7's no-NORMAL recommendation without mutating materials
            // shared with triangles. KHR_materials_unlit already ignores emission.
            auto color = params.get("baseColorFactor", float4(1.f));
            if (params.get("_type", std::string()) != "gltf_unlit") {
                const auto emission = params.get("emissiveFactor", float3(0.f));
                color = float4(color.xyz() + emission, color.w);
            }
            material.name = name + "_unlit";
            material.params = Properties{};
            material.params.set("_type", "gltf_unlit");
            material.params.set("baseColorFactor", color);
            material.params.set("alphaMode", AlphaMode::opaque);
            material.params.set("doubleSided", true);
            curve.material_index = sgl::narrow_cast<int>(m_scene->materials.size());
            m_scene->materials.push_back(std::move(material));
            m_curve_index_map[mesh_index].push_back(sgl::narrow_cast<int>(m_scene->curves.size()));
            m_scene->curves.push_back(std::move(curve));
        }
    }
}

void GltfImporter::extract_nodes()
{
    const tinygltf::Model& model = *m_model;

    // Reserve space for nodes
    m_scene->nodes.reserve(model.nodes.size());

    // Process each node in the GLTF file
    for (size_t node_idx = 0; node_idx < model.nodes.size(); ++node_idx) {
        const auto& gltf_node = model.nodes[node_idx];

        ImporterNode importer_node;
        importer_node.name = gltf_node.name.empty() ? "node_" + std::to_string(node_idx) : gltf_node.name;
        importer_node.source_asset_ref = source_asset_ref_for_object(fmt::format("/nodes/{}", node_idx));

        // Extract transform matrix
        if (gltf_node.matrix.size() == 16) {
            // Matrix is provided directly (column-major order)
            importer_node.transform = float4x4{
                static_cast<float>(gltf_node.matrix[0]),
                static_cast<float>(gltf_node.matrix[4]),
                static_cast<float>(gltf_node.matrix[8]),
                static_cast<float>(gltf_node.matrix[12]),
                static_cast<float>(gltf_node.matrix[1]),
                static_cast<float>(gltf_node.matrix[5]),
                static_cast<float>(gltf_node.matrix[9]),
                static_cast<float>(gltf_node.matrix[13]),
                static_cast<float>(gltf_node.matrix[2]),
                static_cast<float>(gltf_node.matrix[6]),
                static_cast<float>(gltf_node.matrix[10]),
                static_cast<float>(gltf_node.matrix[14]),
                static_cast<float>(gltf_node.matrix[3]),
                static_cast<float>(gltf_node.matrix[7]),
                static_cast<float>(gltf_node.matrix[11]),
                static_cast<float>(gltf_node.matrix[15])
            };
        } else {
            // Build matrix from TRS components
            float4x4 translation_matrix = float4x4::identity();
            float4x4 rotation_matrix = float4x4::identity();
            float4x4 scale_matrix = float4x4::identity();

            // Translation
            if (gltf_node.translation.size() == 3) {
                translation_matrix = sgl::math::matrix_from_translation(float3(
                    static_cast<float>(gltf_node.translation[0]),
                    static_cast<float>(gltf_node.translation[1]),
                    static_cast<float>(gltf_node.translation[2])
                ));
            }

            // Rotation (quaternion: x, y, z, w)
            if (gltf_node.rotation.size() == 4) {
                quatf quat(
                    static_cast<float>(gltf_node.rotation[0]),
                    static_cast<float>(gltf_node.rotation[1]),
                    static_cast<float>(gltf_node.rotation[2]),
                    static_cast<float>(gltf_node.rotation[3])
                );
                rotation_matrix = sgl::math::matrix_from_quat(quat);
            }

            // Scale
            if (gltf_node.scale.size() == 3) {
                scale_matrix = sgl::math::matrix_from_scaling(float3(
                    static_cast<float>(gltf_node.scale[0]),
                    static_cast<float>(gltf_node.scale[1]),
                    static_cast<float>(gltf_node.scale[2])
                ));
            }

            // Combine: T * R * S
            importer_node.transform = sgl::math::mul(sgl::math::mul(translation_matrix, rotation_matrix), scale_matrix);
        }

        // Set mesh reference if this node references a mesh
        if (gltf_node.mesh >= 0 && gltf_node.mesh < static_cast<int>(model.meshes.size())) {
            SGL_CHECK_LT(static_cast<size_t>(gltf_node.mesh), m_mesh_index_map.size());
            importer_node.mesh_index = m_mesh_index_map[gltf_node.mesh];
        }
        if (gltf_node.skin >= 0) {
            SGL_CHECK_LT(static_cast<size_t>(gltf_node.skin), model.skins.size());
            importer_node.skin_index = gltf_node.skin;
        }

        // Set camera reference if this node references a camera
        if (gltf_node.camera >= 0 && gltf_node.camera < static_cast<int>(model.cameras.size())) {
            importer_node.camera_index = gltf_node.camera;
        }

        // Store children indices (will be processed in build_node_hierarchy)
        importer_node.children.reserve(gltf_node.children.size());
        for (int child_idx : gltf_node.children) {
            importer_node.children.push_back(child_idx);
        }

        m_scene->nodes.push_back(std::move(importer_node));
    }

    // Append identity children after all glTF nodes so source node indices used
    // by hierarchy, animation and skin channels remain unchanged.
    for (size_t node_index = 0; node_index < model.nodes.size(); ++node_index) {
        const auto& node = model.nodes[node_index];
        if (node.mesh < 0)
            continue;
        for (int curve_index : m_curve_index_map.at(node.mesh)) {
            if (node.skin >= 0) {
                sgl::log_warn("glTF node '{}': skinned lines are not currently supported", node.name);
                continue;
            }
            ImporterNode child;
            child.name = m_scene->curves[curve_index].name;
            child.curve_index = curve_index;
            child.parent = sgl::narrow_cast<int>(node_index);
            m_scene->nodes[node_index].children.push_back(sgl::narrow_cast<int>(m_scene->nodes.size()));
            m_scene->nodes.push_back(std::move(child));
        }
    }
}

void GltfImporter::build_node_hierarchy()
{
    const tinygltf::Model& model = *m_model;

    // Set parent relationships
    for (size_t node_idx = 0; node_idx < m_scene->nodes.size(); ++node_idx) {
        const auto& children = m_scene->nodes[node_idx].children;
        for (int child_idx : children) {
            if (child_idx >= 0 && child_idx < static_cast<int>(m_scene->nodes.size())) {
                m_scene->nodes[child_idx].parent = static_cast<int>(node_idx);
            }
        }
    }

    // Find root nodes (nodes without parents)
    for (size_t node_idx = 0; node_idx < m_scene->nodes.size(); ++node_idx) {
        if (m_scene->nodes[node_idx].parent == -1) {
            m_scene->root_nodes.push_back(static_cast<int>(node_idx));
        }
    }

    // If there are scenes defined in the GLTF file, use the default scene's nodes as root nodes
    if (!model.scenes.empty()) {
        int default_scene = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (default_scene < static_cast<int>(model.scenes.size())) {
            const auto& scene = model.scenes[default_scene];
            m_scene->root_nodes.clear();
            m_scene->root_nodes.reserve(scene.nodes.size());
            for (int node_idx : scene.nodes) {
                if (node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size())) {
                    m_scene->root_nodes.push_back(node_idx);
                }
            }
        }
    }
}

void GltfImporter::extract_skins()
{
    const tinygltf::Model& model = *m_model;
    m_scene->skins.reserve(model.skins.size());

    for (size_t skin_index = 0; skin_index < model.skins.size(); ++skin_index) {
        const tinygltf::Skin& gltf_skin = model.skins[skin_index];
        ImporterSkin skin;
        skin.name = gltf_skin.name.empty() ? "skin_" + std::to_string(skin_index) : gltf_skin.name;
        skin.skeleton_root = gltf_skin.skeleton;
        skin.joints = gltf_skin.joints;
        skin.joint_from_bind_mesh.assign(skin.joints.size(), float4x4::identity());

        if (gltf_skin.inverseBindMatrices >= 0) {
            SGL_CHECK_LT(static_cast<size_t>(gltf_skin.inverseBindMatrices), model.accessors.size());
            const tinygltf::Accessor& accessor = model.accessors[gltf_skin.inverseBindMatrices];
            SGL_CHECK(
                accessor.count == skin.joints.size(),
                "Skin '{}' must provide one inverse-bind matrix per joint",
                skin.name
            );
            const AttributeInfo inverse_bind_matrices(model, accessor);
            for (size_t joint_index = 0; joint_index < skin.joints.size(); ++joint_index)
                skin.joint_from_bind_mesh[joint_index] = load_matrix(inverse_bind_matrices, joint_index);
        }

        m_scene->skins.push_back(std::move(skin));
    }
}

void GltfImporter::extract_animations()
{
    const tinygltf::Model& model = *m_model;

    FALCOR_CHECK(
        std::find(model.extensionsRequired.begin(), model.extensionsRequired.end(), "KHR_interactivity")
            == model.extensionsRequired.end(),
        "Unsupported required extension KHR_interactivity"
    );
    if (model.animations.empty())
        return;

    // Use first animation only (single animation per scene).
    const auto& gltf_anim = model.animations[0];
    m_scene->animation.name = gltf_anim.name;
    m_scene->animation.source_asset_ref = source_asset_ref_for_object("/animations/0");

    for (const auto& gltf_channel : gltf_anim.channels) {
        // Skip channels without a valid target node.
        if (gltf_channel.target_node < 0 || gltf_channel.target_node >= static_cast<int>(m_scene->nodes.size()))
            continue;

        // Determine value type from target path.
        AnimationValueType value_type;
        if (gltf_channel.target_path == "translation") {
            value_type = AnimationValueType::float3;
        } else if (gltf_channel.target_path == "rotation") {
            value_type = AnimationValueType::quatf;
        } else if (gltf_channel.target_path == "scale") {
            value_type = AnimationValueType::float3;
        } else {
            // Skip unsupported target paths (e.g. "weights" for morph targets).
            continue;
        }

        // Validate sampler index.
        if (gltf_channel.sampler < 0 || gltf_channel.sampler >= static_cast<int>(gltf_anim.samplers.size()))
            continue;
        const auto& sampler = gltf_anim.samplers[gltf_channel.sampler];

        // Validate accessor indices.
        if (sampler.input < 0 || sampler.input >= static_cast<int>(model.accessors.size()))
            continue;
        if (sampler.output < 0 || sampler.output >= static_cast<int>(model.accessors.size()))
            continue;

        // Determine interpolation mode.
        AnimationInterpolationMode interp_mode;
        if (sampler.interpolation == "STEP") {
            interp_mode = AnimationInterpolationMode::step;
        } else if (sampler.interpolation == "CUBICSPLINE") {
            interp_mode = AnimationInterpolationMode::cubic_spline;
        } else {
            // Default to linear (covers "LINEAR" and any unknown string).
            interp_mode = AnimationInterpolationMode::linear;
        }

        // Read time keyframes from sampler input accessor.
        const auto& time_accessor = model.accessors[sampler.input];
        AttributeInfo time_attr(model, time_accessor);
        size_t keyframe_count = time_accessor.count;

        std::vector<float> times(keyframe_count);
        for (size_t i = 0; i < keyframe_count; ++i) {
            const float* t = time_attr.get_float_data(i);
            SGL_CHECK(t != nullptr, "Failed to read animation time keyframe");
            times[i] = *t;
        }

        // Read value keyframes from sampler output accessor.
        const auto& value_accessor = model.accessors[sampler.output];
        AttributeInfo value_attr(model, value_accessor);

        int components = (value_type == AnimationValueType::float_) ? 1
            : (value_type == AnimationValueType::float3)            ? 3
                                                                    : 4;

        // Validate value accessor count matches expected keyframe data.
        size_t expected_value_count = keyframe_count;
        if (interp_mode == AnimationInterpolationMode::cubic_spline)
            expected_value_count = keyframe_count * 3; // in-tangent, value, out-tangent per keyframe
        if (value_accessor.count < expected_value_count)
            continue;

        ImporterAnimationChannel channel;
        channel.value_type = value_type;
        channel.interpolation_mode = interp_mode;
        channel.times = std::move(times);
        SGL_ASSERT(channel.components_per_value() == components);

        if (interp_mode == AnimationInterpolationMode::cubic_spline) {
            // glTF cubic spline stores triplets per keyframe: [in-tangent, value, out-tangent].
            // The output accessor has 3 * keyframe_count elements.
            channel.values.resize(keyframe_count * components);
            channel.in_tangents.resize(keyframe_count * components);
            channel.out_tangents.resize(keyframe_count * components);

            for (size_t i = 0; i < keyframe_count; ++i) {
                size_t triplet_base = i * 3;
                const float* in_tan = value_attr.get_float_data(triplet_base + 0);
                const float* value = value_attr.get_float_data(triplet_base + 1);
                const float* out_tan = value_attr.get_float_data(triplet_base + 2);
                SGL_CHECK(in_tan && value && out_tan, "Failed to read cubic spline animation data");

                for (int c = 0; c < components; ++c) {
                    channel.in_tangents[i * components + c] = in_tan[c];
                    channel.values[i * components + c] = value[c];
                    channel.out_tangents[i * components + c] = out_tan[c];
                }
            }
        } else {
            // LINEAR or STEP: one value per keyframe.
            channel.values.resize(keyframe_count * components);
            for (size_t i = 0; i < keyframe_count; ++i) {
                const float* value = value_attr.get_float_data(i);
                SGL_CHECK(value != nullptr, "Failed to read animation value keyframe");
                for (int c = 0; c < components; ++c) {
                    channel.values[i * components + c] = value[c];
                }
            }
        }

        // Append channel and assign to node.
        int channel_index = static_cast<int>(m_scene->animation.channels.size());
        m_scene->animation.channels.push_back(std::move(channel));

        ImporterNode& node = m_scene->nodes[gltf_channel.target_node];
        if (gltf_channel.target_path == "translation") {
            node.animation_translation = channel_index;
        } else if (gltf_channel.target_path == "rotation") {
            node.animation_rotation = channel_index;
        } else if (gltf_channel.target_path == "scale") {
            node.animation_scale = channel_index;
        }
    }
}

void GltfImporter::extract_textures()
{
    const tinygltf::Model& model = *m_model;

    // Reserve space for textures
    m_scene->textures.reserve(model.textures.size());

    // Process each texture in the GLTF file
    for (size_t tex_idx = 0; tex_idx < model.textures.size(); ++tex_idx) {
        const auto& gltf_texture = model.textures[tex_idx];

        ImporterTexture importer_texture;
        importer_texture.source_name
            = gltf_texture.name.empty() ? "texture_" + std::to_string(tex_idx) : gltf_texture.name;

        // Handle texture source (image)
        if (gltf_texture.source >= 0 && gltf_texture.source < static_cast<int>(model.images.size())) {
            const auto& gltf_image = model.images[gltf_texture.source];

            if (!gltf_image.image.empty()) {
                // Embedded image data
                importer_texture.texture_data = gltf_image.image;
                // A container-local asset key shares decoded images across texture/sampler bindings.
                importer_texture.texture_path = m_path / "$images" / std::to_string(gltf_texture.source);
                importer_texture.mime_type = gltf_image.mimeType;
                importer_texture.source_name = gltf_image.name.empty() ? importer_texture.source_name : gltf_image.name;
            } else if (!gltf_image.uri.empty()) {
                // External image file
                importer_texture.texture_path = m_path.parent_path() / gltf_image.uri;
                importer_texture.source_name = gltf_image.name.empty() ? importer_texture.source_name : gltf_image.name;
            }
        }

        // Handle texture sampler settings
        if (gltf_texture.sampler >= 0 && gltf_texture.sampler < static_cast<int>(model.samplers.size())) {
            const auto& gltf_sampler = model.samplers[gltf_texture.sampler];

            // Convert magnification filter
            if (gltf_sampler.magFilter == 9728) { // NEAREST
                importer_texture.mag_filter = TextureFilterMode::nearest;
            } else if (gltf_sampler.magFilter == 9729) { // LINEAR
                importer_texture.mag_filter = TextureFilterMode::linear;
            }
            // If magFilter is not set, keep default (linear)

            // Preserve both within-level filtering and mip selection.
            if (gltf_sampler.minFilter == 9728 || gltf_sampler.minFilter == 9984 || gltf_sampler.minFilter == 9986) {
                // NEAREST, NEAREST_MIPMAP_NEAREST, NEAREST_MIPMAP_LINEAR
                importer_texture.min_filter = TextureFilterMode::nearest;
            } else if (gltf_sampler.minFilter == 9729 || gltf_sampler.minFilter == 9985
                       || gltf_sampler.minFilter == 9987) {
                // LINEAR, LINEAR_MIPMAP_NEAREST, LINEAR_MIPMAP_LINEAR
                importer_texture.min_filter = TextureFilterMode::linear;
            }
            importer_texture.use_mipmaps = gltf_sampler.minFilter != 9728 && gltf_sampler.minFilter != 9729;
            if (gltf_sampler.minFilter == 9984 || gltf_sampler.minFilter == 9985)
                importer_texture.mip_filter = TextureFilterMode::nearest;
            // If minFilter is not set, keep the existing trilinear default.

            // Convert S (U) wrapping mode
            if (gltf_sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) { // CLAMP_TO_EDGE
                importer_texture.wrap_s = TextureWrapMode::clamp_to_edge;
            } else if (gltf_sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) { // MIRRORED_REPEAT
                importer_texture.wrap_s = TextureWrapMode::mirror_repeat;
            } else if (gltf_sampler.wrapS == TINYGLTF_TEXTURE_WRAP_REPEAT) { // REPEAT
                importer_texture.wrap_s = TextureWrapMode::repeat;
            }
            // Default is REPEAT (10497)

            // Convert T (V) wrapping mode
            if (gltf_sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) { // CLAMP_TO_EDGE
                importer_texture.wrap_t = TextureWrapMode::clamp_to_edge;
            } else if (gltf_sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) { // MIRRORED_REPEAT
                importer_texture.wrap_t = TextureWrapMode::mirror_repeat;
            } else if (gltf_sampler.wrapT == TINYGLTF_TEXTURE_WRAP_REPEAT) { // REPEAT
                importer_texture.wrap_t = TextureWrapMode::repeat;
            }
            // Default is REPEAT (10497)
        }
        // If no sampler is specified, use defaults (Linear filters, Repeat wrapping)

        m_scene->textures.push_back(std::move(importer_texture));
    }
}


// Static helper functions for reading texture info properties and extensions
static void read_texture_transform_extension(
    const tinygltf::ExtensionMap& extensions,
    Properties& params,
    const std::string& param_name
)
{
    // Check if KHR_texture_transform extension is present
    auto transform_it = extensions.find("KHR_texture_transform");
    if (transform_it == extensions.end()) {
        return;
    }

    const tinygltf::Value& transform_value = transform_it->second;
    if (!transform_value.IsObject()) {
        return;
    }

    const tinygltf::Value::Object& transform_obj = transform_value.Get<tinygltf::Value::Object>();

    // Read offset (default: [0.0, 0.0])
    read_float2_param(transform_obj, "offset", params, param_name + ":offset", float2(0.0f, 0.0f));

    // Read rotation (default: 0.0)
    if (auto rotation = read_float(transform_obj, "rotation")) {
        params.set(param_name + ":rotation", *rotation);
    }

    // Read scale (default: [1.0, 1.0])
    read_float2_param(transform_obj, "scale", params, param_name + ":scale", float2(1.0f, 1.0f));

    // Identity transforms need neither MDL promotion nor a transform shader variant.
    const float2 offset = params.get(param_name + ":offset", float2(0.f));
    const float rotation = params.get(param_name + ":rotation", 0.f);
    const float2 scale = params.get(param_name + ":scale", float2(1.f));
    params.set(
        param_name + ":has_transform",
        offset.x != 0.f || offset.y != 0.f || rotation != 0.f || scale.x != 1.f || scale.y != 1.f
    );

    // Read texCoord override (overrides the textureInfo texCoord if supplied)
    auto texcoord_it = transform_obj.find("texCoord");
    if (texcoord_it != transform_obj.end() && texcoord_it->second.IsInt()) {
        int texcoord = texcoord_it->second.Get<int>();
        params.set(param_name + ":texcoord", texcoord);
    }
}

static void
read_texture_info(const tinygltf::TextureInfo& texture_info, Properties& params, const std::string& param_name)
{
    // Check if texture is valid before proceeding
    if (texture_info.index < 0) {
        return;
    }

    // Set the texture index
    params.set(param_name, texture_info.index);

    // Set the texture coordinate index
    params.set(param_name + ":texcoord", texture_info.texCoord);

    // Read texture transform extension
    read_texture_transform_extension(texture_info.extensions, params, param_name);
}

static void
read_texture_info(const tinygltf::NormalTextureInfo& texture_info, Properties& params, const std::string& param_name)
{
    // Check if texture is valid before proceeding
    if (texture_info.index < 0) {
        return;
    }

    // Set the texture index
    params.set(param_name, texture_info.index);

    // Set the texture coordinate index
    params.set(param_name + ":texcoord", texture_info.texCoord);

    // Read texture transform extension
    read_texture_transform_extension(texture_info.extensions, params, param_name);
}

static void
read_texture_info(const tinygltf::Value::Object& texture_obj, Properties& params, const std::string& param_name)
{
    // Read texture index
    auto index_it = texture_obj.find("index");
    if (index_it == texture_obj.end() || !index_it->second.IsInt()) {
        return; // No valid texture index found
    }

    int texture_index = index_it->second.Get<int>();
    if (texture_index < 0) {
        return; // Invalid texture index
    }

    // Set the texture index
    params.set(param_name, texture_index);

    // Read texture coordinate index (default: 0)
    int texcoord = 0;
    auto texcoord_it = texture_obj.find("texCoord");
    if (texcoord_it != texture_obj.end() && texcoord_it->second.IsInt()) {
        texcoord = texcoord_it->second.Get<int>();
    }
    params.set(param_name + ":texcoord", texcoord);

    // Read texture transform extension if present
    auto extensions_it = texture_obj.find("extensions");
    if (extensions_it != texture_obj.end() && extensions_it->second.IsObject()) {
        read_texture_transform_extension(extensions_it->second.Get<tinygltf::Value::Object>(), params, param_name);
    }
}

static void
read_texture_info_param(const tinygltf::Value::Object& parent_obj, Properties& params, const std::string& param_name)
{
    auto texture_it = parent_obj.find(param_name);
    if (texture_it != parent_obj.end() && texture_it->second.IsObject()) {
        const tinygltf::Value::Object& texture_obj = texture_it->second.Get<tinygltf::Value::Object>();
        read_texture_info(texture_obj, params, param_name);
    }
}

static bool gltf_material_supports_native(const tinygltf::Material& source, const Properties& params)
{
    const auto type = params.get<std::string>("_type");
    if (type != "gltf_pbrMetallicRoughness" && type != "gltf_pbrSpecularGlossiness")
        return false;
    for (const auto& [extension, value] : source.extensions) {
        if (extension == "KHR_materials_emissive_strength" || extension == "KHR_materials_pbrSpecularGlossiness")
            continue;
        if (type == "gltf_pbrMetallicRoughness"
            && (extension == "KHR_materials_ior" || extension == "KHR_materials_specular"
                || extension == "KHR_materials_transmission" || extension == "KHR_materials_volume"
                || extension == "KHR_materials_clearcoat" || extension == "KHR_materials_sheen"
                || extension == "KHR_materials_iridescence" || extension == "KHR_materials_anisotropy"
                || extension == "KHR_materials_diffuse_transmission"))
            continue;
        return false;
    }
    // Native MR supports specular strength/color and both workflows support texture
    // transforms. Presence of an unsupported lobe still requires MDL, even at zero strength.
    return true;
}

void GltfImporter::extract_materials(const ImportOptions& import_options)
{
    const tinygltf::Model& model = *m_model;

    // Reserve space for materials
    m_scene->materials.reserve(model.materials.size());

    // Process each material in the GLTF file
    for (size_t mat_idx = 0; mat_idx < model.materials.size(); ++mat_idx) {
        const auto& gltf_material = model.materials[mat_idx];

        ImporterMaterial importer_material;
        importer_material.name
            = gltf_material.name.empty() ? "material_" + std::to_string(mat_idx) : gltf_material.name;
        importer_material.source_asset_ref = source_asset_ref_for_object(fmt::format("/materials/{}", mat_idx));
        bool has_material_type_extension = false;

        for (const auto& [extension, value] : gltf_material.extensions) {
            if (extension.starts_with("KHR_materials_") && extension != "KHR_materials_unlit"
                && extension != "KHR_materials_pbrSpecularGlossiness" && extension != "KHR_materials_ior"
                && extension != "KHR_materials_emissive_strength" && extension != "KHR_materials_specular"
                && extension != "KHR_materials_clearcoat" && extension != "KHR_materials_transmission"
                && extension != "KHR_materials_volume" && extension != "KHR_materials_sheen"
                && extension != "KHR_materials_iridescence" && extension != "KHR_materials_anisotropy"
                && extension != "KHR_materials_diffuse_transmission")
                sgl::log_warn(
                    "glTF material '{}': extension '{}' is not handled by the material importer; "
                    "rendering its fallback without that feature.",
                    importer_material.name,
                    extension
                );
        }

        // KHR_materials_unlit selects a direction-independent material that uses only the
        // metallic-roughness base color inputs. It takes precedence over the fallback PBR model.
        auto unlit_ext_it = gltf_material.extensions.find("KHR_materials_unlit");
        if (unlit_ext_it != gltf_material.extensions.end()) {
            has_material_type_extension = true;
            importer_material.params.set("_type", "gltf_unlit");
            const auto& pbr = gltf_material.pbrMetallicRoughness;
            read_float4_param(
                pbr.baseColorFactor,
                importer_material.params,
                "baseColorFactor",
                float4(1.0f, 1.0f, 1.0f, 1.0f)
            );
            read_texture_info(pbr.baseColorTexture, importer_material.params, "baseColorTexture");
        }

        // Check for KHR_materials_pbrSpecularGlossiness extension
        auto pbr_sg_ext_it = gltf_material.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (!has_material_type_extension && pbr_sg_ext_it != gltf_material.extensions.end()) {
            const tinygltf::Value& pbr_sg_value = pbr_sg_ext_it->second;
            if (pbr_sg_value.IsObject()) {
                has_material_type_extension = true;
                importer_material.params.set("_type", "gltf_pbrSpecularGlossiness");
                const tinygltf::Value::Object& pbr_sg_obj = pbr_sg_value.Get<tinygltf::Value::Object>();
                read_float4_param(
                    pbr_sg_obj,
                    "diffuseFactor",
                    importer_material.params,
                    "diffuseFactor",
                    float4(1.0f, 1.0f, 1.0f, 1.0f)
                );
                read_float3_param(
                    pbr_sg_obj,
                    "specularFactor",
                    importer_material.params,
                    "specularFactor",
                    float3(1.0f, 1.0f, 1.0f)
                );
                read_float_param(pbr_sg_obj, "glossinessFactor", importer_material.params, "glossinessFactor", 1.0f);
                read_texture_info_param(pbr_sg_obj, importer_material.params, "diffuseTexture");
                read_texture_info_param(pbr_sg_obj, importer_material.params, "specularGlossinessTexture");
            }
        }


        // Extract metallic-roughness parameters if not using specular-glossiness extension
        if (!has_material_type_extension) {
            importer_material.params.set("_type", "gltf_pbrMetallicRoughness");
            const auto& pbr = gltf_material.pbrMetallicRoughness;
            read_float4_param(
                pbr.baseColorFactor,
                importer_material.params,
                "baseColorFactor",
                float4(1.0f, 1.0f, 1.0f, 1.0f)
            );
            importer_material.params.set("metallicFactor", pbr.metallicFactor);
            importer_material.params.set("roughnessFactor", pbr.roughnessFactor);
            read_texture_info(pbr.baseColorTexture, importer_material.params, "baseColorTexture");
            read_texture_info(pbr.metallicRoughnessTexture, importer_material.params, "metallicRoughnessTexture");

            auto sheen_it = gltf_material.extensions.find("KHR_materials_sheen");
            if (sheen_it != gltf_material.extensions.end() && sheen_it->second.IsObject()) {
                importer_material.params.set("hasSheen", true);
                const auto& obj = sheen_it->second.Get<tinygltf::Value::Object>();
                read_float3_param(obj, "sheenColorFactor", importer_material.params, "sheenColorFactor", float3(0.f));
                read_float_param(obj, "sheenRoughnessFactor", importer_material.params, "sheenRoughnessFactor", 0.f);
                read_texture_info_param(obj, importer_material.params, "sheenColorTexture");
                read_texture_info_param(obj, importer_material.params, "sheenRoughnessTexture");
            }
            auto iridescence_it = gltf_material.extensions.find("KHR_materials_iridescence");
            if (iridescence_it != gltf_material.extensions.end() && iridescence_it->second.IsObject()) {
                importer_material.params.set("hasIridescence", true);
                const auto& obj = iridescence_it->second.Get<tinygltf::Value::Object>();
                read_float_param(obj, "iridescenceFactor", importer_material.params, "iridescenceFactor", 0.f);
                read_float_param(obj, "iridescenceIor", importer_material.params, "iridescenceIor", 1.3f);
                read_float_param(
                    obj,
                    "iridescenceThicknessMinimum",
                    importer_material.params,
                    "iridescenceThicknessMinimum",
                    100.f
                );
                read_float_param(
                    obj,
                    "iridescenceThicknessMaximum",
                    importer_material.params,
                    "iridescenceThicknessMaximum",
                    400.f
                );
                read_texture_info_param(obj, importer_material.params, "iridescenceTexture");
                read_texture_info_param(obj, importer_material.params, "iridescenceThicknessTexture");
            }
            if (auto it = gltf_material.extensions.find("KHR_materials_anisotropy");
                it != gltf_material.extensions.end() && it->second.IsObject()) {
                importer_material.params.set("hasAnisotropy", true);
                const auto& obj = it->second.Get<tinygltf::Value::Object>();
                read_float_param(obj, "anisotropyStrength", importer_material.params, "anisotropyStrength", 0.f);
                read_float_param(obj, "anisotropyRotation", importer_material.params, "anisotropyRotation", 0.f);
                read_texture_info_param(obj, importer_material.params, "anisotropyTexture");
            }
            if (auto it = gltf_material.extensions.find("KHR_materials_diffuse_transmission");
                it != gltf_material.extensions.end() && it->second.IsObject()) {
                importer_material.params.set("hasDiffuseTransmission", true);
                const auto& obj = it->second.Get<tinygltf::Value::Object>();
                read_float_param(
                    obj,
                    "diffuseTransmissionFactor",
                    importer_material.params,
                    "diffuseTransmissionFactor",
                    0.f
                );
                read_float3_param(
                    obj,
                    "diffuseTransmissionColorFactor",
                    importer_material.params,
                    "diffuseTransmissionColorFactor",
                    float3(1.f)
                );
                read_texture_info_param(obj, importer_material.params, "diffuseTransmissionTexture");
                read_texture_info_param(obj, importer_material.params, "diffuseTransmissionColorTexture");
            }
            auto clearcoat_it = gltf_material.extensions.find("KHR_materials_clearcoat");
            if (clearcoat_it != gltf_material.extensions.end() && clearcoat_it->second.IsObject()) {
                const auto& clearcoat = clearcoat_it->second.Get<tinygltf::Value::Object>();
                importer_material.params.set("hasClearcoat", true);
                read_float_param(clearcoat, "clearcoatFactor", importer_material.params, "clearcoatFactor", 0.f);
                read_texture_info_param(clearcoat, importer_material.params, "clearcoatTexture");
                read_float_param(
                    clearcoat,
                    "clearcoatRoughnessFactor",
                    importer_material.params,
                    "clearcoatRoughnessFactor",
                    0.f
                );
                read_texture_info_param(clearcoat, importer_material.params, "clearcoatRoughnessTexture");
                read_texture_info_param(clearcoat, importer_material.params, "clearcoatNormalTexture");
                auto normal = clearcoat.find("clearcoatNormalTexture");
                if (normal != clearcoat.end() && normal->second.IsObject()) {
                    read_float_param(
                        normal->second.Get<tinygltf::Value::Object>(),
                        "scale",
                        importer_material.params,
                        "clearcoatNormalTexture:strength",
                        1.f
                    );
                }
            }

            auto specular_it = gltf_material.extensions.find("KHR_materials_specular");
            if (specular_it != gltf_material.extensions.end() && specular_it->second.IsObject()) {
                importer_material.params.set("hasSpecular", true);
                const auto& specular = specular_it->second.Get<tinygltf::Value::Object>();
                read_float_param(specular, "specularFactor", importer_material.params, "specularFactor", 1.f);
                read_texture_info_param(specular, importer_material.params, "specularTexture");
                read_float3_param(
                    specular,
                    "specularColorFactor",
                    importer_material.params,
                    "specularColorFactor",
                    float3(1.f)
                );
                read_texture_info_param(specular, importer_material.params, "specularColorTexture");
            }

            auto transmission_it = gltf_material.extensions.find("KHR_materials_transmission");
            if (transmission_it != gltf_material.extensions.end() && transmission_it->second.IsObject()) {
                const auto& transmission = transmission_it->second.Get<tinygltf::Value::Object>();
                importer_material.params.set("hasTransmission", true);
                read_float_param(
                    transmission,
                    "transmissionFactor",
                    importer_material.params,
                    "transmissionFactor",
                    0.f
                );
                read_texture_info_param(transmission, importer_material.params, "transmissionTexture");
            }

            auto ior_it = gltf_material.extensions.find("KHR_materials_ior");
            if (ior_it != gltf_material.extensions.end() && ior_it->second.IsObject()) {
                const auto& ior = ior_it->second.Get<tinygltf::Value::Object>();
                read_float_param(ior, "ior", importer_material.params, "ior", 1.5f);
            }

            auto volume_it = gltf_material.extensions.find("KHR_materials_volume");
            if (volume_it != gltf_material.extensions.end() && volume_it->second.IsObject()) {
                importer_material.params.set("hasVolume", true);
                const auto& volume = volume_it->second.Get<tinygltf::Value::Object>();
                read_float_param(volume, "thicknessFactor", importer_material.params, "thicknessFactor", 0.f);
                read_texture_info_param(volume, importer_material.params, "thicknessTexture");
                // Absence means infinite attenuation distance (no absorption).
                if (auto distance = read_float(volume, "attenuationDistance"))
                    importer_material.params.set("attenuationDistance", *distance);
                read_float3_param(
                    volume,
                    "attenuationColor",
                    importer_material.params,
                    "attenuationColor",
                    float3(1.f)
                );
            }
        }

        if (gltf_material.extensions.contains("KHR_materials_transmission"))
            importer_material.params.set("hasTransmission", true);

        // Always read normal map
        read_texture_info(gltf_material.normalTexture, importer_material.params, "normalTexture");
        if (gltf_material.normalTexture.index >= 0) {
            importer_material.params.set("normalTexture:strength", gltf_material.normalTexture.scale);
        }

        // Read emissive, handling emissive strength extension.
        auto emissive_factor_opt = read_float3(gltf_material.emissiveFactor);
        if (emissive_factor_opt) {
            float3 emissive_factor = *emissive_factor_opt;
            float emissive_strength = 1.0f;
            auto ext_it = gltf_material.extensions.find("KHR_materials_emissive_strength");
            if (ext_it != gltf_material.extensions.end() && ext_it->second.IsObject()) {
                const tinygltf::Value::Object& ext_obj = ext_it->second.Get<tinygltf::Value::Object>();
                if (auto strength = read_float(ext_obj, "emissiveStrength")) {
                    emissive_strength = *strength;
                }
            }
            emissive_factor = emissive_factor * emissive_strength;
            importer_material.params.set("emissiveFactor", emissive_factor);
        }
        read_texture_info(gltf_material.emissiveTexture, importer_material.params, "emissiveTexture");

        // Extract alpha mode and double-sided flag
        importer_material.params.set("doubleSided", gltf_material.doubleSided);
        if (gltf_material.alphaMode == "OPAQUE") {
            importer_material.params.set("alphaMode", AlphaMode::opaque);
        } else if (gltf_material.alphaMode == "MASK") {
            importer_material.params.set("alphaMode", AlphaMode::mask);
            importer_material.params.set("alphaCutoff", static_cast<float>(gltf_material.alphaCutoff));
        } else if (gltf_material.alphaMode == "BLEND") {
            importer_material.params.set("alphaMode", AlphaMode::blend);
        } else {
            importer_material.params.set("alphaMode", AlphaMode::opaque);
        }

        // Always set alpha cutoff with a default value for consistency
        if (!importer_material.params.has_property("alphaCutoff")) {
            importer_material.params.set("alphaCutoff", 0.5f);
        }

        // Retain the import-time choice with the material, including deferred imports and append.
        const auto mode = import_options.gltf_material_mode;
        if (importer_material.params.get<std::string>("_type") != "gltf_unlit") {
            if (mode == GLTFMaterialMode::standard_material)
                importer_material.params.set("_gltf_target", "standard_material");
            else if (mode == GLTFMaterialMode::materialx)
                importer_material.params.set("_gltf_target", "materialx");
            else if (mode == GLTFMaterialMode::openpbr)
                importer_material.params.set("_gltf_target", "openpbr");
        }
        if (importer_material.params.get<std::string>("_type") != "gltf_unlit"
            && (mode == GLTFMaterialMode::mdl
                || (mode == GLTFMaterialMode::automatic
                    && !gltf_material_supports_native(gltf_material, importer_material.params))))
            importer_material.params.set("_gltf_use_mdl", true);

        m_scene->materials.push_back(std::move(importer_material));
    }
}

SourceAssetRef GltfImporter::source_asset_ref_for_object(std::string_view object_path) const
{
    static const ustring format("gltf");
    return SourceAssetRef(format, m_source_asset_path, ustring(object_path));
}

} // namespace falcor
