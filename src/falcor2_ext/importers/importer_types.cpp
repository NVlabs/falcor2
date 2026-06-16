// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/importers/importer_types.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace falcor {

inline nb::dlpack::dtype datatype_to_dtype(sgl::DataType data_type)
{
    switch (data_type) {
    case sgl::DataType::void_:
        return {};
    case sgl::DataType::bool_:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Bool, 8, 1};
    case sgl::DataType::int32:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 32, 1};
    case sgl::DataType::uint32:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 32, 1};
    case sgl::DataType::int64:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 64, 1};
    case sgl::DataType::uint64:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 64, 1};
    case sgl::DataType::float16:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 16, 1};
    case sgl::DataType::float32:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
    case sgl::DataType::float64:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 64, 1};
    case sgl::DataType::int8:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 8, 1};
    case sgl::DataType::uint8:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 8, 1};
    case sgl::DataType::int16:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 16, 1};
    case sgl::DataType::uint16:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 16, 1};
    default:
        SGL_THROW("Unsupported data type");
    }
}

std::optional<nb::ndarray<nb::numpy>> attrib_to_numpy(ImporterMesh& self, const ImporterMeshAttribute& attrib)
{
    if (!attrib.valid()) {
        return std::nullopt;
    }
    void* base_address;
    size_t stride;
    self.get_stream_info(attrib, base_address, stride);

    size_t n = self.vertex_count();
    size_t component_size = data_type_size(attrib.type);
    size_t stride_in_components = stride / component_size;
    auto dtype = datatype_to_dtype(attrib.type);

    if (attrib.num_components == 1) {
        return nb::ndarray<nb::numpy>(base_address, {n}, {}, {static_cast<int64_t>(stride_in_components)}, dtype);
    } else {
        return nb::ndarray<nb::numpy>(
            base_address,
            {n, attrib.num_components},
            {},
            {static_cast<int64_t>(stride_in_components), 1},
            dtype
        );
    }
}

void attribute_name_to_semantic_index(const std::string& name, ImporterSemantic& out_semantic, uint32_t& out_index)
{
    // Parse attribute name to determine semantic and index
    size_t underscore_pos = name.rfind('_');
    if (underscore_pos != std::string::npos) {
        std::string base_name = name.substr(0, underscore_pos);
        std::string index_str = name.substr(underscore_pos + 1);
        try {
            out_index = static_cast<uint32_t>(std::stoul(index_str));
            out_semantic = sgl::string_to_enum<ImporterSemantic>(base_name);
            return;
        } catch (...) {
            // Fall through to default parsing
        }
    }
    out_semantic = sgl::string_to_enum<ImporterSemantic>(name);
    out_index = 0;
}

using Indices = nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using MeshAttr = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;


ImporterMesh* mesh_create(std::vector<Indices> subgeometries, std::map<std::string, MeshAttr> attributes)
{
    auto mesh = new ImporterMesh();

    mesh->subgeometries.resize(subgeometries.size());

    for (size_t i = 0; i < subgeometries.size(); ++i) {
        const Indices& indices_nd = subgeometries[i];
        size_t triangle_count = indices_nd.shape(0);

        ImporterMesh::Subgeometry& subgeom = mesh->subgeometries[i];
        subgeom.indices.resize(triangle_count);
        std::memcpy(subgeom.indices.data(), indices_nd.data(), triangle_count * 3 * sizeof(uint32_t));
    }

    size_t vertex_count = 0;

    size_t offset = 0;
    for (auto it : attributes) {
        const std::string& name = it.first;
        const MeshAttr& attr_nd = it.second;

        SGL_CHECK(attr_nd.ndim() == 1 || attr_nd.ndim() == 2, "Attribute ndarray must be 1D or 2D");

        ImporterMeshAttribute new_attrib;
        new_attrib.name = name;
        new_attrib.num_components = static_cast<uint32_t>(attr_nd.ndim() == 1 ? 1 : attr_nd.shape(1));
        new_attrib.type = sgl::dtype_to_data_type(attr_nd.dtype());
        new_attrib.buffer = 0;

        // TODO: We should support passing in name, semantic, or semantic+index
        attribute_name_to_semantic_index(name, new_attrib.semantic, new_attrib.index);

        // Vertex count comes from positions
        if (new_attrib.semantic == ImporterSemantic::position) {
            vertex_count = attr_nd.shape(0);
        }

        // Align offset to attribute type size
        size_t dtype_size = sgl::data_type_size(new_attrib.type);
        if (offset % dtype_size != 0) {
            offset = (offset + dtype_size - 1) / dtype_size * dtype_size;
        }
        new_attrib.offset = static_cast<uint32_t>(offset);
        offset += dtype_size * new_attrib.num_components;

        mesh->add_attribute(new_attrib);
    }

    // Check we got positions
    if (!mesh->find_attribute(ImporterSemantic::position)) {
        SGL_THROW("Mesh must have position attribute");
    }

    // Allocate vertex buffers
    mesh->allocate_vertices(vertex_count);
    mesh->create_buffers_from_attributes();

    // Copy attribute data into mesh buffers
    for (auto it : attributes) {
        const std::string& name = it.first;
        const MeshAttr& attr_nd = it.second;

        const ImporterMeshAttribute* attrib = mesh->find_attribute(name);
        SGL_CHECK(attrib, "Attribute not found after creation: " + name);

        void* base_address;
        size_t stride;
        mesh->get_stream_info(*attrib, base_address, stride);

        size_t component_size = sgl::data_type_size(attrib->type);
        size_t stride_in_components = stride / component_size;

        size_t n = mesh->vertex_count();
        if (attrib->num_components == 1) {
            SGL_CHECK(attr_nd.shape(0) == n, "Attribute vertex count mismatch for attribute: " + name);
            for (size_t v = 0; v < n; ++v) {
                std::memcpy(
                    reinterpret_cast<uint8_t*>(base_address) + v * stride,
                    reinterpret_cast<const uint8_t*>(attr_nd.data()) + v * component_size,
                    component_size
                );
            }
        } else {
            SGL_CHECK(
                attr_nd.shape(0) == n && attr_nd.shape(1) == attrib->num_components,
                "Attribute vertex count or component count mismatch for attribute: " + name
            );
            for (size_t v = 0; v < n; ++v) {
                std::memcpy(
                    reinterpret_cast<uint8_t*>(base_address) + v * stride,
                    reinterpret_cast<const uint8_t*>(attr_nd.data()) + v * attrib->num_components * component_size,
                    attrib->num_components * component_size
                );
            }
        }
    }

    // Calculate local AABB
    mesh->calculate_local_aabb();

    return mesh;
}

} // namespace falcor

FALCOR_PY_EXPORT(importers_importer_types)
{
    using namespace falcor;

    nb::sgl_enum<ImporterSemantic> e(m, "ImporterSemantic", D(ImporterSemantic));

    nb::class_<ImporterMeshAttribute>(m, "ImporterMeshAttribute", D(ImporterMeshAttribute))
        .DEF_RW(ImporterMeshAttribute, name)
        .DEF_RW(ImporterMeshAttribute, semantic)
        .DEF_RW(ImporterMeshAttribute, index)
        .DEF_RW(ImporterMeshAttribute, type)
        .DEF_RW(ImporterMeshAttribute, num_components)
        .DEF_RW(ImporterMeshAttribute, buffer)
        .DEF_RW(ImporterMeshAttribute, offset)
        .DEF_PROP_RO(ImporterMeshAttribute, valid);

    nb::class_<ImporterMesh::Subgeometry>(m, "ImporterMeshSubgeometry", D(ImporterMesh, Subgeometry))
        .DEF_RW_2(ImporterMesh, Subgeometry, name)
        .DEF_RW_2(ImporterMesh, Subgeometry, indices, nb::rv_policy::reference_internal)
        .DEF_RW_2(ImporterMesh, Subgeometry, material_name)
        .def_prop_ro(
            "indices_numpy",
            [](ImporterMesh::Subgeometry& self)
            {
                size_t n = self.indices.size();
                return nb::ndarray<nb::numpy, const uint32_t>(reinterpret_cast<uint32_t*>(self.indices.data()), {n, 3});
            },
            nb::rv_policy::reference_internal,
            "Triangle indices as 2D ndarray of uint."
        );

    nb::class_<ImporterMesh>(m, "ImporterMesh", D(ImporterMesh))
        .def_static("create", &mesh_create, "subgeometries"_a, "attributes"_a, "Create a new ImporterMesh.")
        .DEF_RW(ImporterMesh, name)
        .DEF_RW(ImporterMesh, subgeometries, nb::rv_policy::reference_internal)
        .DEF_RO(ImporterMesh, local_aabb)
        .DEF_PROP_RO(ImporterMesh, vertex_count)
        .DEF_PROP_RO(ImporterMesh, attributes, nb::rv_policy::reference_internal)
        .def("calculate_local_aabb", &ImporterMesh::calculate_local_aabb, D(ImporterMesh, calculate_local_aabb))
        .def("add_normals_from_faces", &ImporterMesh::add_normals_from_faces, D(ImporterMesh, add_normals_from_faces))
        .def("add_tangents_from_uvs", &ImporterMesh::add_tangents_from_uvs, D(ImporterMesh, add_tangents_from_uvs))
        .def(
            "add_tangents_from_normals",
            &ImporterMesh::add_tangents_from_normals,
            D(ImporterMesh, add_tangents_from_normals)
        )
        .def("deduplicate_vertices", &ImporterMesh::deduplicate_vertices, D(ImporterMesh, deduplicate_vertices))
        .def(
            "find_attribute",
            nb::overload_cast<ImporterSemantic, uint32_t>(&ImporterMesh::find_attribute),
            "semantic"_a,
            "index"_a = 0,
            nb::rv_policy::reference_internal,
            D(ImporterMesh, find_attribute)
        )
        .def(
            "get_stream",
            [](ImporterMesh& self, const ImporterMeshAttribute& attrib)
            {
                return attrib_to_numpy(self, attrib);
            },
            "attrib"_a,
            nb::rv_policy::reference_internal,
            "Get numpy array for vertex attribute stream with proper type conversion."
        )
        .def(
            "find_stream",
            [](ImporterMesh& self, ImporterSemantic semantic, uint32_t index) -> std::optional<nb::ndarray<nb::numpy>>
            {
                const auto* attrib = self.find_attribute(semantic, index);
                if (!attrib) {
                    return std::nullopt;
                }
                return attrib_to_numpy(self, *attrib);
            },
            "semantic"_a,
            "index"_a = 0,
            nb::rv_policy::reference_internal,
            "Find and return numpy array for vertex attribute stream by semantic."
        )
        .def(
            "find_stream",
            [](ImporterMesh& self, std::string_view name) -> std::optional<nb::ndarray<nb::numpy>>
            {
                const auto* attrib = self.find_attribute(name);
                if (!attrib) {
                    return std::nullopt;
                }
                return attrib_to_numpy(self, *attrib);
            },
            "name"_a,
            nb::rv_policy::reference_internal,
            "Find and return numpy array for vertex attribute stream by name."
        )
        .def_prop_ro(
            "positions",
            [](ImporterMesh& self)
            {
                return attrib_to_numpy(self, self.position_attribute());
            },
            nb::rv_policy::reference_internal,
            "Vertex positions as ndarray (N, 3)."
        )
        .def_prop_ro(
            "normals",
            [](ImporterMesh& self)
            {
                return attrib_to_numpy(self, self.normal_attribute());
            },
            nb::rv_policy::reference_internal,
            "Vertex normals as ndarray (N, 3)."
        )
        .def_prop_ro(
            "tangents",
            [](ImporterMesh& self)
            {
                return attrib_to_numpy(self, self.tangent_attribute());
            },
            nb::rv_policy::reference_internal,
            "Vertex tangents as ndarray (N, 3)."
        )
        .def_prop_ro(
            "handedness",
            [](ImporterMesh& self)
            {
                return attrib_to_numpy(self, self.handedness_attribute());
            },
            nb::rv_policy::reference_internal,
            "Vertex handedness as ndarray (N)."
        )
        .def_prop_ro(
            "colors",
            [](ImporterMesh& self)
            {
                return attrib_to_numpy(self, self.color_attribute());
            },
            nb::rv_policy::reference_internal,
            "Vertex colors as ndarray (N, 3) or (N, 4)."
        )
        .def(
            "texcoords",
            [](ImporterMesh& self, int idx) -> std::optional<nb::ndarray<nb::numpy>>
            {
                if (idx == 0) {
                    // Use fast path for the common case of the first set of texcoords
                    return attrib_to_numpy(self, self.texcoord_attribute());
                } else {
                    auto texcoord_attrib = self.find_attribute(ImporterSemantic::tex_coord, idx);
                    if (!texcoord_attrib) {
                        return std::nullopt;
                    }
                    return attrib_to_numpy(self, *texcoord_attrib);
                }
            },
            "idx"_a = 0,
            nb::rv_policy::reference_internal,
            "Vertex texture coordinates as ndarray (N, 2)."
        );

    nb::sgl_enum<CurveIndexingMode>(m, "CurveIndexingMode", D(CurveIndexingMode));

    nb::class_<ImporterCurve>(m, "ImporterCurve", D(ImporterCurve))
        .DEF_RW(ImporterCurve, name)
        .DEF_RW(ImporterCurve, material_name)
        .DEF_RW(ImporterCurve, positions, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterCurve, radii, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterCurve, indices, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterCurve, indexing_mode)
        .DEF_RO(ImporterCurve, local_aabb)
        .def("calculate_local_aabb", &ImporterCurve::calculate_local_aabb, D(ImporterCurve, calculate_local_aabb));

    nb::sgl_enum<falcor::TextureFilterMode>(m, "TextureFilterMode", D(TextureFilterMode));
    nb::sgl_enum<falcor::TextureWrapMode>(m, "TextureWrapMode", D(TextureWrapMode));
    nb::sgl_enum<falcor::AlphaMode>(m, "AlphaMode", D(AlphaMode));

    nb::class_<falcor::ImporterTexture>(m, "ImporterTexture", D(ImporterTexture))
        .def(nb::init<>())
        .DEF_RW(ImporterTexture, source_name)
        .DEF_RW(ImporterTexture, texture_path)
        .DEF_RW(ImporterTexture, is_srgb)
        .DEF_RW(ImporterTexture, texture_transform)
        .DEF_RW(ImporterTexture, mime_type)
        .DEF_RW(ImporterTexture, mag_filter)
        .DEF_RW(ImporterTexture, min_filter)
        .DEF_RW(ImporterTexture, wrap_s)
        .DEF_RW(ImporterTexture, wrap_t)
        .def_prop_ro(
            "texture_data",
            [](falcor::ImporterTexture& self)
            {
                size_t n = self.texture_data.size();
                return nb::ndarray<nb::numpy, const uint8_t>(self.texture_data.data(), {n}, {}, {1});
            },
            nb::rv_policy::reference_internal,
            "Texture data as numpy array of uint8 (read-only)."
        );

    nb::class_<ImporterMaterial>(m, "ImporterMaterial", D(ImporterMaterial))
        .DEF_RW(ImporterMaterial, name)
        .DEF_RW(ImporterMaterial, params, nb::rv_policy::reference_internal);

    nb::class_<ImporterCamera> importer_camera(m, "ImporterCamera", D(ImporterCamera));
    importer_camera //
        .DEF_RW(ImporterCamera, name)
        .DEF_RW(ImporterCamera, focus_distance)
        .DEF_RW(ImporterCamera, focal_length)
        .DEF_RW(ImporterCamera, fstop)
        .DEF_RW(ImporterCamera, depth_range)
        .DEF_RW(ImporterCamera, projection)
        .DEF_RW(ImporterCamera, fov_direction)
        .DEF_RW(ImporterCamera, aperture);

    nb::sgl_enum<ImporterCamera::Projection>(importer_camera, "Projection", D(ImporterCamera, Projection));
    nb::sgl_enum<ImporterCamera::FOVDirection>(importer_camera, "FOVDirection", D(ImporterCamera, FOVDirection));

    nb::class_<ImporterLight> importer_light(m, "ImporterLight", D(ImporterLight));
    importer_light //
        .DEF_RW(ImporterLight, name)
        .DEF_RW(ImporterLight, type)
        .DEF_RW(ImporterLight, intensity)
        .DEF_RW(ImporterLight, degree_angular_diameter)
        .DEF_RW(ImporterLight, width)
        .DEF_RW(ImporterLight, height)
        .DEF_RW(ImporterLight, radius)
        .DEF_RW(ImporterLight, env_map_path);

    nb::sgl_enum<ImporterLight::Type>(importer_light, "Type", D(ImporterLight, Type));

    nb::class_<ImporterPrototype>(m, "ImporterPrototype", D(ImporterPrototype))
        .def(nb::init<>(), "Create a new ImporterPrototype")
        .DEF_RW(ImporterPrototype, name)
        .DEF_RW(ImporterPrototype, root_node);

    nb::sgl_enum<AnimationValueType>(m, "AnimationValueType", D(AnimationValueType));
    nb::sgl_enum<AnimationInterpolationMode>(m, "AnimationInterpolationMode", D(AnimationInterpolationMode));

    nb::class_<ImporterAnimationChannel>(m, "ImporterAnimationChannel", D(ImporterAnimationChannel))
        .def(nb::init<>(), "Create a new ImporterAnimationChannel")
        .DEF_RW(ImporterAnimationChannel, value_type)
        .DEF_RW(ImporterAnimationChannel, interpolation_mode)
        .DEF_RW(ImporterAnimationChannel, times, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterAnimationChannel, values, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterAnimationChannel, in_tangents, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterAnimationChannel, out_tangents, nb::rv_policy::reference_internal)
        .def("components_per_value", &ImporterAnimationChannel::components_per_value)
        .def("keyframe_count", &ImporterAnimationChannel::keyframe_count);

    nb::class_<ImporterAnimation>(m, "ImporterAnimation", D(ImporterAnimation))
        .def(nb::init<>(), "Create a new ImporterAnimation")
        .DEF_RW(ImporterAnimation, name)
        .DEF_RW(ImporterAnimation, channels, nb::rv_policy::reference_internal)
        .def("duration", &ImporterAnimation::duration);

    nb::class_<ImporterNode>(m, "ImporterNode", D(ImporterNode))
        .def(nb::init<>(), "Create a new ImporterNode")
        .DEF_RW(ImporterNode, name)
        .DEF_RW(ImporterNode, transform)
        .DEF_RW(ImporterNode, mesh_index)
        .DEF_RW(ImporterNode, light_index)
        .DEF_RW(ImporterNode, camera_index)
        .DEF_RW(ImporterNode, curve_index)
        .DEF_RW(ImporterNode, prototype_index)
        .DEF_RW(ImporterNode, children, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterNode, parent)
        .DEF_RW(ImporterNode, animation_translation)
        .DEF_RW(ImporterNode, animation_rotation)
        .DEF_RW(ImporterNode, animation_scale)
        .DEF_RW(ImporterNode, world_aabb)
        .def(
            "__repr__",
            [](const ImporterNode& node)
            {
                return fmt::format(
                    "<ImporterNode name='{}' mesh_index={} light_index={} camera_index={} curve_index={} parent={} "
                    "children={}>",
                    node.name,
                    node.mesh_index,
                    node.light_index,
                    node.camera_index,
                    node.curve_index,
                    node.parent,
                    node.children.size()
                );
            }
        );

    nb::class_<ImporterScene, Object>(m, "ImporterScene", D(ImporterScene))
        .DEF_RW(ImporterScene, materials, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, textures, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, meshes, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, nodes, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, cameras, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, lights, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, curves, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, prototypes, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, root_nodes, nb::rv_policy::reference_internal)
        .DEF_RW(ImporterScene, animation, nb::rv_policy::reference_internal)
        .def("calculate_aabbs", &ImporterScene::calculate_aabbs, D(ImporterScene, calculate_aabbs))
        .def("rescale_to_fit", &ImporterScene::rescale_to_fit, "box_size"_a = 1.f, D(ImporterScene, rescale_to_fit))
        .def("get_scene_aabb", &ImporterScene::get_scene_aabb, D(ImporterScene, get_scene_aabb))
        .def(
            "add_default_camera_robust_fit",
            &ImporterScene::add_default_camera_robust_fit,
            "fov_degrees"_a = 50.f,
            "aspect"_a = 16.f / 9.f,
            "coverage"_a = 0.9f,
            "outlier_percentile"_a = 0.05f,
            "Option 4: Add a robust-fit default camera and return its camera index."
        )
        .def(
            "add_default_camera_best_view",
            &ImporterScene::add_default_camera_best_view,
            "fov_degrees"_a = 50.f,
            "aspect"_a = 16.f / 9.f,
            "coverage"_a = 0.9f,
            "outlier_percentile"_a = 0.05f,
            "Option 5: Add a best-view default camera and return its camera index."
        )
        .def(
            "flatten_node_hierarchy",
            &ImporterScene::flatten_node_hierarchy,
            D(ImporterScene, flatten_node_hierarchy)
        );
}
