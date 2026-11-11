// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include <nanobind/stl/vector.h>

#include <cstring>
#include <string>

#include "falcor2/render/geometry/static_curve_geometry.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/dynamic_mesh_geometry.h"
#include "falcor2/render/geometry/dynamic_curve_geometry.h"

namespace falcor {

struct PyMeshVertexAttribute {
    std::string name;
    MeshAttributeSemantic semantic{MeshAttributeSemantic::unknown};
    uint32_t index{0};
    nb::ndarray<nb::numpy> data;

    PyMeshVertexAttribute() = default;
    PyMeshVertexAttribute(
        MeshAttributeSemantic semantic,
        uint32_t index,
        nb::ndarray<nb::numpy> data,
        std::string name = ""
    )
        : name(std::move(name))
        , semantic(semantic)
        , index(index)
        , data(std::move(data))
    {
    }
};

MeshVertexAttributeDesc make_vertex_attribute_desc(const PyMeshVertexAttribute& attribute, size_t vertex_count)
{
    if (attribute.data.ndim() != 1 && attribute.data.ndim() != 2)
        throw nb::value_error("Vertex attribute data must be a one- or two-dimensional C-contiguous ndarray");
    if (attribute.data.stride(attribute.data.ndim() - 1) != 1
        || (attribute.data.ndim() == 2
            && attribute.data.stride(0) != sgl::narrow_cast<int64_t>(attribute.data.shape(1))))
        throw nb::value_error("Vertex attribute data must be C-contiguous");
    if (attribute.data.shape(0) != vertex_count)
        throw nb::value_error("Vertex attribute vertex count does not match positions");

    const uint32_t component_count
        = sgl::narrow_cast<uint32_t>(attribute.data.ndim() == 1 ? 1 : attribute.data.shape(1));
    return {
        .name = attribute.name,
        .semantic = attribute.semantic,
        .index = attribute.index,
        .type = sgl::dtype_to_data_type(attribute.data.dtype()),
        .component_count = component_count,
        .data = attribute.data.data(),
        .stride = sgl::narrow_cast<size_t>(attribute.data.stride(0)) * attribute.data.itemsize(),
        .data_size = attribute.data.nbytes(),
    };
}

inline void check_vertex_array_size(size_t actual, size_t expected, const char* attribute_name)
{
    if (actual != expected) {
        std::string message = fmt::format(
            "{} vertex count ({}) does not match sub-mesh vertex count ({})",
            attribute_name,
            actual,
            expected
        );
        throw nb::value_error(message.c_str());
    }
}

inline float3 read_float3(const float* values, size_t index)
{
    return float3(values[index * 3 + 0], values[index * 3 + 1], values[index * 3 + 2]);
}

template<typename T>
nb::ndarray<nb::numpy>
static_mesh_attribute_data(const MeshVertexAttribute& attribute, size_t vertex_count, uint32_t component_count = 0)
{
    if (component_count == 0)
        component_count = attribute.component_count;
    SGL_ASSERT(component_count <= attribute.component_count);

    auto typed_data = make_numpy_copy<T>(
        vertex_count * component_count,
        component_count == 1 ? std::initializer_list<size_t>{vertex_count}
                             : std::initializer_list<size_t>{vertex_count, component_count},
        [&](T* data)
        {
            for (size_t vertex_index = 0; vertex_index < vertex_count; ++vertex_index) {
                std::memcpy(
                    data + vertex_index * component_count,
                    attribute.data.data() + vertex_index * attribute.component_count * sizeof(T),
                    component_count * sizeof(T)
                );
            }
        }
    );
    return nb::cast<nb::ndarray<nb::numpy>>(nb::cast(typed_data));
}

nb::ndarray<nb::numpy>
static_mesh_attribute_data(const MeshVertexAttribute& attribute, size_t vertex_count, uint32_t component_count = 0)
{
    switch (attribute.type) {
    case sgl::DataType::bool_:
        return static_mesh_attribute_data<bool>(attribute, vertex_count, component_count);
    case sgl::DataType::int8:
        return static_mesh_attribute_data<int8_t>(attribute, vertex_count, component_count);
    case sgl::DataType::int16:
        return static_mesh_attribute_data<int16_t>(attribute, vertex_count, component_count);
    case sgl::DataType::int32:
        return static_mesh_attribute_data<int32_t>(attribute, vertex_count, component_count);
    case sgl::DataType::int64:
        return static_mesh_attribute_data<int64_t>(attribute, vertex_count, component_count);
    case sgl::DataType::uint8:
        return static_mesh_attribute_data<uint8_t>(attribute, vertex_count, component_count);
    case sgl::DataType::uint16:
        return static_mesh_attribute_data<uint16_t>(attribute, vertex_count, component_count);
    case sgl::DataType::uint32:
        return static_mesh_attribute_data<uint32_t>(attribute, vertex_count, component_count);
    case sgl::DataType::uint64:
        return static_mesh_attribute_data<uint64_t>(attribute, vertex_count, component_count);
    case sgl::DataType::float16:
        return static_mesh_attribute_data<sgl::math::float16_t>(attribute, vertex_count, component_count);
    case sgl::DataType::float32:
        return static_mesh_attribute_data<float>(attribute, vertex_count, component_count);
    case sgl::DataType::float64:
        return static_mesh_attribute_data<double>(attribute, vertex_count, component_count);
    default:
        SGL_THROW("Unsupported static mesh vertex attribute data type");
    }
}

nb::ndarray<nb::numpy> static_mesh_required_attribute_data(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    MeshAttributeSemantic semantic,
    uint32_t index,
    uint32_t component_count
)
{
    const MeshVertexAttribute* attribute = self.find_vertex_attribute(sub_mesh_index, semantic, index);
    SGL_ASSERT(attribute);
    return static_mesh_attribute_data(*attribute, self.vertex_count(sub_mesh_index), component_count);
}

nb::ndarray<nb::numpy> static_mesh_positions(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    return static_mesh_required_attribute_data(self, sub_mesh_index, MeshAttributeSemantic::position, 0, 3);
}

nb::ndarray<nb::numpy> static_mesh_normals(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    if (const auto* attribute = self.find_vertex_attribute(sub_mesh_index, MeshAttributeSemantic::normal, 0))
        return static_mesh_attribute_data(*attribute, self.vertex_count(sub_mesh_index), 3);

    // Optional source streams may be absent; the packed vertices still hold the
    // runtime values, including changes made through the legacy setters.
    auto vertices = self.vertices(sub_mesh_index);
    auto typed_data = make_numpy_copy<float>(
        vertices.size() * 3,
        {vertices.size(), 3},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i) {
                float3 normal = shared::detail::unpack_vertex_normal(vertices[i].normal_and_tangent);
                data[i * 3 + 0] = normal.x;
                data[i * 3 + 1] = normal.y;
                data[i * 3 + 2] = normal.z;
            }
        }
    );
    return nb::cast<nb::ndarray<nb::numpy>>(nb::cast(typed_data));
}

nb::ndarray<nb::numpy> static_mesh_tangents(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    if (const auto* attribute = self.find_vertex_attribute(sub_mesh_index, MeshAttributeSemantic::tangent, 0))
        return static_mesh_attribute_data(*attribute, self.vertex_count(sub_mesh_index), 3);

    auto vertices = self.vertices(sub_mesh_index);
    auto typed_data = make_numpy_copy<float>(
        vertices.size() * 3,
        {vertices.size(), 3},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i) {
                float3 tangent = shared::detail::unpack_vertex_tangent(vertices[i].normal_and_tangent);
                data[i * 3 + 0] = tangent.x;
                data[i * 3 + 1] = tangent.y;
                data[i * 3 + 2] = tangent.z;
            }
        }
    );
    return nb::cast<nb::ndarray<nb::numpy>>(nb::cast(typed_data));
}

nb::ndarray<nb::numpy> static_mesh_handedness(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    if (const auto* attribute = self.find_vertex_attribute(sub_mesh_index, MeshAttributeSemantic::handedness, 0))
        return static_mesh_attribute_data(*attribute, self.vertex_count(sub_mesh_index), 1);

    auto vertices = self.vertices(sub_mesh_index);
    auto typed_data = make_numpy_copy<float>(
        vertices.size(),
        {vertices.size()},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i)
                data[i] = shared::detail::unpack_vertex_handedness(vertices[i].normal_and_tangent);
        }
    );
    return nb::cast<nb::ndarray<nb::numpy>>(nb::cast(typed_data));
}

nb::ndarray<nb::numpy> static_mesh_texcoords(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    if (const auto* attribute = self.find_vertex_attribute(sub_mesh_index, MeshAttributeSemantic::tex_coord, 0))
        return static_mesh_attribute_data(*attribute, self.vertex_count(sub_mesh_index), 2);

    auto vertices = self.vertices(sub_mesh_index);
    auto typed_data = make_numpy_copy<float>(
        vertices.size() * 2,
        {vertices.size(), 2},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i) {
                data[i * 2 + 0] = vertices[i].uv[0];
                data[i * 2 + 1] = vertices[i].uv[1];
            }
        }
    );
    return nb::cast<nb::ndarray<nb::numpy>>(nb::cast(typed_data));
}

nb::ndarray<nb::numpy, uint32_t> static_mesh_indices(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto indices = self.indices(sub_mesh_index);
    if (indices.size() % 3 != 0)
        throw nb::value_error("Sub-mesh index count is not divisible by 3");

    return make_numpy_copy<uint32_t>(
        indices.size(),
        {indices.size() / 3, 3},
        [&](uint32_t* data)
        {
            if (!indices.empty())
                std::memcpy(data, indices.data(), indices.size() * sizeof(uint32_t));
        }
    );
}

void update_static_mesh_attribute_data(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    MeshAttributeSemantic semantic,
    uint32_t index,
    const void* data,
    size_t data_size
)
{
    MeshVertexAttribute* attribute = self.find_vertex_attribute(sub_mesh_index, semantic, index);
    if (!attribute)
        return;

    SGL_ASSERT(attribute->data.size() == data_size);
    std::memcpy(attribute->data.data(), data, data_size);
}

std::vector<PyMeshVertexAttribute> static_mesh_vertex_attributes(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    std::vector<PyMeshVertexAttribute> result;
    for (const MeshVertexAttribute& attribute : self.vertex_attributes(sub_mesh_index)) {
        PyMeshVertexAttribute& output = result.emplace_back();
        output.name = attribute.name;
        output.semantic = attribute.semantic;
        output.index = attribute.index;
        const size_t vertex_count = self.vertex_count(sub_mesh_index);
        output.data = static_mesh_attribute_data(attribute, vertex_count);
    }
    return result;
}

void static_mesh_set_positions(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "positions");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].position[0] = src[i * 3 + 0];
        vertices[i].position[1] = src[i * 3 + 1];
        vertices[i].position[2] = src[i * 3 + 2];
    }
    update_static_mesh_attribute_data(
        self,
        sub_mesh_index,
        MeshAttributeSemantic::position,
        0,
        src,
        values.size() * sizeof(float)
    );
    self.recompute_local_aabb();
    self.mark_vertices_dirty();
}

void static_mesh_set_normals(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "normals");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i)
        shared::detail::pack_vertex_normal(vertices[i].normal_and_tangent, read_float3(src, i));
    update_static_mesh_attribute_data(
        self,
        sub_mesh_index,
        MeshAttributeSemantic::normal,
        0,
        src,
        values.size() * sizeof(float)
    );
    self.mark_vertices_dirty();
}

void static_mesh_set_tangents(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "tangents");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i)
        shared::detail::pack_vertex_tangent(vertices[i].normal_and_tangent, read_float3(src, i));
    update_static_mesh_attribute_data(
        self,
        sub_mesh_index,
        MeshAttributeSemantic::tangent,
        0,
        src,
        values.size() * sizeof(float)
    );
    self.mark_vertices_dirty();
}

void static_mesh_set_handedness(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "handedness");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i)
        shared::detail::pack_vertex_handedness(vertices[i].normal_and_tangent, src[i]);
    update_static_mesh_attribute_data(
        self,
        sub_mesh_index,
        MeshAttributeSemantic::handedness,
        0,
        src,
        values.size() * sizeof(float)
    );
    self.mark_vertices_dirty();
}

void static_mesh_set_texcoords(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "texcoords");

    const size_t component_count = values.shape(1);
    if (component_count < 2 || component_count > 4)
        throw nb::value_error("texcoords must have between 2 and 4 components");

    const MeshVertexAttribute* attribute
        = self.find_vertex_attribute(sub_mesh_index, MeshAttributeSemantic::tex_coord, 0);
    if (attribute && attribute->component_count != component_count)
        throw nb::value_error("texcoords component count does not match TEXCOORD_0");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].uv[0] = src[i * component_count + 0];
        vertices[i].uv[1] = src[i * component_count + 1];
    }
    update_static_mesh_attribute_data(
        self,
        sub_mesh_index,
        MeshAttributeSemantic::tex_coord,
        0,
        src,
        values.size() * sizeof(float)
    );
    self.mark_vertices_dirty();
}

void static_mesh_set_indices(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> values
)
{
    auto indices = self.mutable_indices(sub_mesh_index);
    size_t index_count = values.shape(0) * 3;
    if (index_count != indices.size()) {
        std::string message
            = fmt::format("indices count ({}) does not match sub-mesh index count ({})", index_count, indices.size());
        throw nb::value_error(message.c_str());
    }

    if (!indices.empty())
        std::memcpy(indices.data(), values.data(), indices.size() * sizeof(uint32_t));
    self.mark_vertices_dirty();
}

} // namespace falcor

FALCOR_PY_EXPORT(render_static_mesh_geometry)
{
    using namespace falcor;

    nb::sgl_enum<MeshAttributeSemantic>(m, "MeshAttributeSemantic");
    nb::class_<PyMeshVertexAttribute>(m, "MeshVertexAttribute")
        .def(
            nb::init<MeshAttributeSemantic, uint32_t, nb::ndarray<nb::numpy>, std::string>(),
            "semantic"_a,
            "index"_a,
            "data"_a,
            "name"_a = ""
        )
        .def_rw("semantic", &PyMeshVertexAttribute::semantic)
        .def_rw("index", &PyMeshVertexAttribute::index)
        .def_rw("name", &PyMeshVertexAttribute::name)
        .def_prop_rw(
            "data",
            [](const PyMeshVertexAttribute& self)
            {
                return self.data;
            },
            [](PyMeshVertexAttribute& self, nb::ndarray<nb::numpy> data)
            {
                self.data = std::move(data);
            },
            nb::rv_policy::copy
        );

    nb::class_<StaticMeshGeometry, Geometry>(m, "StaticMeshGeometry", D(StaticMeshGeometry)) //
        .def_prop_ro("sub_mesh_count", &StaticMeshGeometry::sub_mesh_count, "Number of sub-meshes in this geometry.")
        .def(
            "vertex_count",
            &StaticMeshGeometry::vertex_count,
            "sub_mesh_index"_a,
            "Number of vertices in the given sub-mesh."
        )
        .def(
            "index_count",
            &StaticMeshGeometry::index_count,
            "sub_mesh_index"_a,
            "Number of indices in the given sub-mesh."
        )
        .def(
            "positions",
            &static_mesh_positions,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh vertex positions as a float32 ndarray of shape (N, 3)."
        )
        .def(
            "normals",
            &static_mesh_normals,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh vertex normals as a float32 ndarray of shape (N, 3)."
        )
        .def(
            "tangents",
            &static_mesh_tangents,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh vertex tangents as a float32 ndarray of shape (N, 3)."
        )
        .def(
            "handedness",
            &static_mesh_handedness,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh tangent handedness values as a float32 ndarray of shape (N,)."
        )
        .def(
            "texcoords",
            &static_mesh_texcoords,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh vertex texture coordinates as a float32 ndarray of shape (N, 2)."
        )
        .def(
            "indices",
            &static_mesh_indices,
            "sub_mesh_index"_a,
            "Return a copy of the sub-mesh triangle indices as a uint32 ndarray of shape (M, 3)."
        )
        .def(
            "vertex_attributes",
            &static_mesh_vertex_attributes,
            "sub_mesh_index"_a,
            "Return copies of the sub-mesh vertex attributes in remapped vertex order."
        )
        .def(
            "set_positions",
            &static_mesh_set_positions,
            "sub_mesh_index"_a,
            "positions"_a,
            "Set sub-mesh vertex positions from a float32 ndarray of shape (N, 3)."
        )
        .def(
            "set_normals",
            &static_mesh_set_normals,
            "sub_mesh_index"_a,
            "normals"_a,
            "Set sub-mesh vertex normals from a float32 ndarray of shape (N, 3)."
        )
        .def(
            "set_tangents",
            &static_mesh_set_tangents,
            "sub_mesh_index"_a,
            "tangents"_a,
            "Set sub-mesh vertex tangents from a float32 ndarray of shape (N, 3)."
        )
        .def(
            "set_handedness",
            &static_mesh_set_handedness,
            "sub_mesh_index"_a,
            "handedness"_a,
            "Set sub-mesh tangent handedness values from a float32 ndarray of shape (N,)."
        )
        .def(
            "set_texcoords",
            &static_mesh_set_texcoords,
            "sub_mesh_index"_a,
            "texcoords"_a,
            "Set TEXCOORD_0 from a float32 ndarray of shape (N, 2), (N, 3), or (N, 4)."
        )
        .def(
            "set_indices",
            &static_mesh_set_indices,
            "sub_mesh_index"_a,
            "indices"_a,
            "Set sub-mesh triangle indices from a uint32 ndarray of shape (M, 3)."
        )
        .def(
            "mark_vertices_dirty",
            &StaticMeshGeometry::mark_vertices_dirty,
            "Flag CPU-side vertex data dirty; Scene::update() will re-upload on the "
            "next call."
        )
        .def(
            "set_mesh_data",
            [](StaticMeshGeometry& self,
               nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> positions,
               std::vector<nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>> sub_mesh_indices,
               std::optional<nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>> normals,
               std::optional<nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>> tangents,
               std::optional<nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu>> handedness,
               std::optional<nb::ndarray<float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>> texcoords,
               std::string_view name,
               std::optional<std::vector<std::string>> sub_mesh_names,
               std::optional<std::vector<PyMeshVertexAttribute>> vertex_attributes)
            {
                size_t vertex_count = positions.shape(0);

                auto check_vertex_count = [&](const auto& arr, const char* label)
                {
                    if (arr.shape(0) != vertex_count) {
                        std::string message = fmt::format("{} vertex count does not match positions", label);
                        throw nb::value_error(message.c_str());
                    }
                };
                if (normals)
                    check_vertex_count(*normals, "normals");
                if (tangents)
                    check_vertex_count(*tangents, "tangents");
                if (handedness)
                    check_vertex_count(*handedness, "handedness");
                if (texcoords)
                    check_vertex_count(*texcoords, "texcoords");

                if (sub_mesh_names && sub_mesh_names->size() != sub_mesh_indices.size())
                    throw nb::value_error("sub_mesh_names length must match sub_mesh_indices length");

                StaticMeshGeometryDataDesc desc = {};
                std::vector<MeshVertexAttributeDesc> native_attributes;
                auto add_common_attribute
                    = [&](MeshAttributeSemantic semantic, uint32_t index, const void* data, uint32_t component_count)
                {
                    native_attributes.push_back({
                        .semantic = semantic,
                        .index = index,
                        .type = sgl::DataType::float32,
                        .component_count = component_count,
                        .data = data,
                        .stride = component_count * sizeof(float),
                        .data_size = vertex_count * component_count * sizeof(float),
                    });
                };
                add_common_attribute(MeshAttributeSemantic::position, 0, positions.data(), 3);
                if (normals)
                    add_common_attribute(MeshAttributeSemantic::normal, 0, normals->data(), 3);
                if (tangents)
                    add_common_attribute(MeshAttributeSemantic::tangent, 0, tangents->data(), 3);
                if (handedness)
                    add_common_attribute(MeshAttributeSemantic::handedness, 0, handedness->data(), 1);
                if (texcoords)
                    add_common_attribute(MeshAttributeSemantic::tex_coord, 0, texcoords->data(), 2);
                if (vertex_attributes) {
                    native_attributes.reserve(native_attributes.size() + vertex_attributes->size());
                    for (const PyMeshVertexAttribute& attribute : *vertex_attributes)
                        native_attributes.push_back(make_vertex_attribute_desc(attribute, vertex_count));
                }
                desc.name = name;
                desc.vertex_count = sgl::narrow_cast<uint32_t>(vertex_count);
                desc.vertex_attributes = native_attributes;

                desc.sub_meshes.reserve(sub_mesh_indices.size());
                for (size_t i = 0; i < sub_mesh_indices.size(); ++i) {
                    StaticMeshGeometryDataDesc::SubMesh sub_mesh = {};
                    if (sub_mesh_names)
                        sub_mesh.name = (*sub_mesh_names)[i];
                    sub_mesh.index_count = sgl::narrow_cast<uint32_t>(sub_mesh_indices[i].shape(0)) * 3;
                    sub_mesh.index_stream.data = sub_mesh_indices[i].data();
                    sub_mesh.index_stream.stride = sizeof(uint32_t);
                    desc.sub_meshes.push_back(sub_mesh);
                }

                self.set_mesh_data(desc);
            },
            "positions"_a,
            "sub_mesh_indices"_a,
            "normals"_a.none() = nb::none(),
            "tangents"_a.none() = nb::none(),
            "handedness"_a.none() = nb::none(),
            "texcoords"_a.none() = nb::none(),
            "name"_a = "",
            "sub_mesh_names"_a.none() = nb::none(),
            nb::kw_only(),
            "vertex_attributes"_a.none() = nb::none(),
            "Populate the mesh from numpy arrays. `positions` is (N, 3) float32 "
            "and required; `sub_mesh_indices` is a list of (M_i, 3) uint32 arrays, "
            "one per sub-mesh. Attribute arrays (normals, tangents, handedness, "
            "texcoords) are optional; missing attributes default to zero per-vertex. `vertex_attributes` is an "
            "optional sequence of MeshVertexAttribute values. "
            "Replaces any existing mesh data."
        );

    nb::class_<DynamicMeshGeometry, Geometry>(m, "DynamicMeshGeometry", D(DynamicMeshGeometry, 2));
    nb::class_<DynamicCurveGeometry, Geometry>(m, "DynamicCurveGeometry", D(DynamicCurveGeometry));

    nb::class_<StaticCurveGeometry, Geometry>(m, "StaticCurveGeometry", D(StaticCurveGeometry))
        .def(
            "set_curve_data",
            [](StaticCurveGeometry& self,
               nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> positions,
               nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu> radii,
               nb::ndarray<uint32_t, nb::shape<-1>, nb::c_contig, nb::device::cpu> indices,
               CurveIndexingMode indexing_mode,
               std::string_view name,
               std::optional<std::vector<PyMeshVertexAttribute>> vertex_attributes)
            {
                const size_t vertex_count = positions.shape(0);
                if (radii.shape(0) != vertex_count)
                    throw nb::value_error("radii vertex count does not match positions");

                std::vector<MeshVertexAttributeDesc> native_attributes;
                if (vertex_attributes) {
                    native_attributes.reserve(vertex_attributes->size());
                    for (const PyMeshVertexAttribute& attribute : *vertex_attributes)
                        native_attributes.push_back(make_vertex_attribute_desc(attribute, vertex_count));
                }

                StaticCurveGeometryDataDesc desc = {};
                desc.name = name;
                desc.vertex_count = sgl::narrow_cast<uint32_t>(vertex_count);
                desc.position_stream.data = reinterpret_cast<const float3*>(positions.data());
                desc.position_stream.stride = sizeof(float3);
                desc.radius_stream = {radii.data(), sizeof(float)};
                desc.index_count = sgl::narrow_cast<uint32_t>(indices.shape(0));
                desc.index_stream = {indices.data(), sizeof(uint32_t)};
                desc.indexing_mode = indexing_mode;
                desc.vertex_attributes = native_attributes;
                self.set_curve_data(desc);
            },
            "positions"_a,
            "radii"_a,
            "indices"_a,
            "indexing_mode"_a = CurveIndexingMode::list,
            "name"_a = "",
            nb::kw_only(),
            "vertex_attributes"_a.none() = nb::none(),
            D(StaticCurveGeometry, set_curve_data)
        );
}
