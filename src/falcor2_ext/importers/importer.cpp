// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include <nanobind/stl/function.h>

#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_collections.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/render/scene.h"

#include "render/python_scene_object.h"

#include <memory>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace falcor {

namespace {

ImporterMaterial::Constructor make_material_constructor(nb::type_object cls)
{
    if (!python_class_inherits_from<Material>(cls)) {
        throw nb::type_error("cls must inherit from Material");
    }

    if (nb::type_check(cls) && SceneObjectFactory<Material>::get().find_class_info(nb::type_info(cls)) != nullptr) {
        const std::type_info* type = &nb::type_info(cls);
        return [type](Scene& scene, const ImporterMaterial& material)
        {
            return scene.create_material(*type, material.params);
        };
    }

    auto cls_ref = std::make_shared<PythonObjectRef>(cls);
    return [cls_ref = std::move(cls_ref)](Scene& scene, const ImporterMaterial& material)
    {
        nb::gil_scoped_acquire gil;
        nb::type_object cls = nb::borrow<nb::type_object>(cls_ref->get());
        nb::object object = create_python_scene_object<Material>(&scene, cls, material.params);
        return nb::cast<Material*>(object);
    };
}

} // namespace

} // namespace falcor

FALCOR_PY_EXPORT(importers_importer)
{
    using namespace falcor;

    nb::class_<ImportOptions>(m, "ImportOptions", D(ImportOptions))
        .def(nb::init<>())
        .DEF_RW(ImportOptions, recompute_normals);

    nb::class_<ImporterSelector>(m, "ImporterSelector", D(ImporterSelector))
        .def(nb::init<>())
        .DEF_RW(ImporterSelector, id);

    nb::class_<ImporterCameraSelector, ImporterSelector>(m, "ImporterCameraSelector", D(ImporterCameraSelector))
        .def(nb::init<>());

    nb::class_<ImporterNodeSelector, ImporterSelector>(m, "ImporterNodeSelector", D(ImporterNodeSelector))
        .def(nb::init<>());

    nb::class_<ImporterMaterialSelector>(m, "ImporterMaterialSelector", D(ImporterMaterialSelector))
        .def(
            "replace",
            [](const ImporterMaterialSelector& self, std::string material_type, std::optional<Properties> props)
            {
                self.replace(std::move(material_type), std::move(props).value_or(Properties()));
            },
            "type"_a,
            "props"_a.none() = nb::none(),
            D(ImporterMaterialSelector, replace)
        )
        .def(
            "replace",
            [](const ImporterMaterialSelector& self, nb::type_object cls, std::optional<Properties> props)
            {
                self.replace(make_material_constructor(cls), std::move(props).value_or(Properties()));
            },
            "cls"_a,
            "props"_a.none() = nb::none(),
            nb::sig("def replace(self, cls: type[MaterialT], props: Properties | None = None) -> None"),
            D(ImporterMaterialSelector, replace_2)
        );

    nb::class_<ImporterMaterialCollection>(m, "ImporterMaterialCollection", D(ImporterMaterialCollection))
        .def(
            "find",
            &ImporterMaterialCollection::find,
            "name"_a,
            nb::keep_alive<0, 1>(),
            D(ImporterMaterialCollection, find)
        )
        .def(
            "__getitem__",
            &ImporterMaterialCollection::operator[],
            "name"_a,
            nb::keep_alive<0, 1>(),
            D(ImporterMaterialCollection, find)
        );

    nb::class_<ImporterCameraCollection>(m, "ImporterCameraCollection", D(ImporterCameraCollection))
        .def(
            "create",
            &ImporterCameraCollection::create,
            "name"_a,
            "focus_distance"_a = 1.f,
            "focal_length"_a = 50.f,
            "fstop"_a = 8.f,
            "depth_range"_a = float2(0.01f, 10000.f),
            "projection"_a = ImporterCamera::Projection::perspective,
            "fov_direction"_a = ImporterCamera::FOVDirection::vertical,
            "sensor_size_mm"_a = 24.f,
            D(ImporterCameraCollection, create)
        )
        .def(
            "create_fov",
            &ImporterCameraCollection::create_fov,
            "name"_a,
            "fov_degrees"_a = 70.f,
            "focus_distance"_a = 1.f,
            "fstop"_a = 8.f,
            "depth_range"_a = float2(0.01f, 10000.f),
            "projection"_a = ImporterCamera::Projection::perspective,
            "fov_direction"_a = ImporterCamera::FOVDirection::vertical,
            "sensor_size_mm"_a = 24.f,
            D(ImporterCameraCollection, create_fov)
        );

    nb::class_<ImporterEnvCollection>(m, "ImporterEnvCollection", D(ImporterEnvCollection))
        .def(
            "set",
            &ImporterEnvCollection::set,
            "path"_a,
            "exposure"_a = 0.f,
            "name"_a = "Environment Map",
            D(ImporterEnvCollection, set)
        );

    nb::class_<ImporterNodeCollection>(m, "ImporterNodeCollection", D(ImporterNodeCollection))
        .def(
            "create",
            &ImporterNodeCollection::create,
            "name"_a,
            "transform"_a = float4x4::identity(),
            "camera"_a.none() = nb::none(),
            "parent"_a.none() = nb::none(),
            D(ImporterNodeCollection, create)
        )
        .def(
            "create_camera",
            &ImporterNodeCollection::create_camera,
            "name"_a = "Camera",
            "transform"_a = float4x4::identity(),
            "focus_distance"_a = 1.f,
            "focal_length"_a = 50.f,
            "fstop"_a = 8.f,
            "depth_range"_a = float2(0.01f, 10000.f),
            "projection"_a = ImporterCamera::Projection::perspective,
            "fov_direction"_a = ImporterCamera::FOVDirection::vertical,
            "sensor_size_mm"_a = 24.f,
            D(ImporterNodeCollection, create_camera)
        )
        .def(
            "create_camera_fov",
            &ImporterNodeCollection::create_camera_fov,
            "name"_a = "Camera",
            "transform"_a = float4x4::identity(),
            "fov_degrees"_a = 70.f,
            "focus_distance"_a = 1.f,
            "fstop"_a = 8.f,
            "depth_range"_a = float2(0.01f, 10000.f),
            "projection"_a = ImporterCamera::Projection::perspective,
            "fov_direction"_a = ImporterCamera::FOVDirection::vertical,
            "sensor_size_mm"_a = 24.f,
            D(ImporterNodeCollection, create_camera_fov)
        );

    nb::class_<Importer, Object>(m, "Importer", D(Importer))
        .def(nb::init<ImportOptions>(), "default_import_options"_a = ImportOptions(), D(Importer, Importer))
        .def_static("create", &Importer::create, "default_import_options"_a = ImportOptions(), D(Importer, create))
        .def_static("get", &Importer::get, D(Importer, get))
        .def_static("set_current", &Importer::set_current, "importer"_a, D(Importer, set_current))
        .def_static("clear_current", &Importer::clear_current, D(Importer, clear_current))
        .DEF_PROP_RO(Importer, cameras, nb::rv_policy::reference_internal)
        .DEF_PROP_RO(Importer, nodes, nb::rv_policy::reference_internal)
        .DEF_PROP_RO(Importer, materials, nb::rv_policy::reference_internal)
        .DEF_PROP_RO(Importer, env, nb::rv_policy::reference_internal)
        .DEF_PROP_RW(Importer, source_path)
        .def("add_search_path", &Importer::add_search_path, "path"_a, D(Importer, add_search_path))
        .def("add_search_paths", &Importer::add_search_paths, "paths"_a, D(Importer, add_search_paths))
        .def("import_asset", &Importer::import_asset, "path"_a, D(Importer, import_asset))
        .def("on_scene_created", &Importer::on_scene_created, "callback"_a, D(Importer, on_scene_created))
        .def("build_importer_scene", &Importer::build_importer_scene, D(Importer, build_importer_scene));
}
