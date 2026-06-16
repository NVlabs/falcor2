// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include <nanobind/stl/vector.h>

#include <cstring>
#include <string>

#include "falcor2/render/geometry/static_mesh_geometry.h"

namespace falcor {

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

nb::ndarray<nb::numpy, float> static_mesh_positions(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto vertices = self.vertices(sub_mesh_index);
    return make_numpy_copy<float>(
        vertices.size() * 3,
        {vertices.size(), 3},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i) {
                data[i * 3 + 0] = vertices[i].position[0];
                data[i * 3 + 1] = vertices[i].position[1];
                data[i * 3 + 2] = vertices[i].position[2];
            }
        }
    );
}

nb::ndarray<nb::numpy, float> static_mesh_normals(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto vertices = self.vertices(sub_mesh_index);
    return make_numpy_copy<float>(
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
}

nb::ndarray<nb::numpy, float> static_mesh_tangents(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto vertices = self.vertices(sub_mesh_index);
    return make_numpy_copy<float>(
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
}

nb::ndarray<nb::numpy, float> static_mesh_handedness(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto vertices = self.vertices(sub_mesh_index);
    return make_numpy_copy<float>(
        vertices.size(),
        {vertices.size()},
        [&](float* data)
        {
            for (size_t i = 0; i < vertices.size(); ++i) {
                data[i] = shared::detail::unpack_vertex_handedness(vertices[i].normal_and_tangent);
            }
        }
    );
}

nb::ndarray<nb::numpy, float> static_mesh_texcoords(StaticMeshGeometry& self, size_t sub_mesh_index)
{
    auto vertices = self.vertices(sub_mesh_index);
    return make_numpy_copy<float>(
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
    self.mark_vertices_dirty();
}

void static_mesh_set_texcoords(
    StaticMeshGeometry& self,
    size_t sub_mesh_index,
    nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu> values
)
{
    auto vertices = self.mutable_vertices(sub_mesh_index);
    check_vertex_array_size(values.shape(0), vertices.size(), "texcoords");

    const float* src = values.data();
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].uv[0] = src[i * 2 + 0];
        vertices[i].uv[1] = src[i * 2 + 1];
    }
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
            "Set sub-mesh vertex texture coordinates from a float32 ndarray of shape (N, 2)."
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
               std::optional<std::vector<std::string>> sub_mesh_names)
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
                desc.name = name;
                desc.vertex_count = sgl::narrow_cast<uint32_t>(vertex_count);
                desc.position_stream.data = reinterpret_cast<const float3*>(positions.data());
                desc.position_stream.stride = sizeof(float) * 3;
                if (normals) {
                    desc.normal_stream.data = reinterpret_cast<const float3*>(normals->data());
                    desc.normal_stream.stride = sizeof(float) * 3;
                }
                if (tangents) {
                    desc.tangent_stream.data = reinterpret_cast<const float3*>(tangents->data());
                    desc.tangent_stream.stride = sizeof(float) * 3;
                }
                if (handedness) {
                    desc.handedness_stream.data = handedness->data();
                    desc.handedness_stream.stride = sizeof(float);
                }
                if (texcoords) {
                    desc.texcoord_stream.data = reinterpret_cast<const float2*>(texcoords->data());
                    desc.texcoord_stream.stride = sizeof(float) * 2;
                }

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
            "Populate the mesh from numpy arrays. `positions` is (N, 3) float32 "
            "and required; `sub_mesh_indices` is a list of (M_i, 3) uint32 arrays, "
            "one per sub-mesh. Attribute arrays (normals, tangents, handedness, "
            "texcoords) are optional; missing attributes default to zero per-vertex. "
            "Replaces any existing mesh data."
        );
}
