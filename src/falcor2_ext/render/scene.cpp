// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "nanobind_reflector.h"
#include "core/signal.h"
#include <nanobind/make_iterator.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <string>

#include "falcor2/render/scene.h"
#include "falcor2/render/scene_config.h"
#include "falcor2/render/scene_globals.h"
#include "falcor2/render/scene_requirements.h"
#include "falcor2/render/material_types.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/component/transform_animation.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/standard_specgloss_material.h"
#include "falcor2/render/material/unlit_material.h"
#include "falcor2/render/material/mdl/mdl_material.h"
#include "falcor2/render/material/mtlx/mtlx_material.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/light.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"

#include "render/python_scene_object.h"

#include <utility>

namespace falcor {

/// Helper to bind a SceneObjectCollectionView<T> as a Python class.
///
/// Note: find/find_all are reimplemented here rather than calling the C++ templated versions because
/// the C++ side uses dynamic_cast<T*> for "is-a" type checking, while the Python side needs
/// nb::isinstance() to correctly handle Python-derived classes. Both provide the same "is-a" semantics
/// but through their respective language's native type dispatch mechanism.
template<typename TObject>
void bind_collection_view(nb::module_& m, const char* name)
{
    using View = SceneObjectCollectionView<TObject>;
    nb::class_<View>(m, name, D(SceneObjectCollectionView))
        .def(
            "__len__",
            [](const View& self)
            {
                return self.size();
            },
            D(SceneObjectCollectionView, size)
        )
        .def(
            "__getitem__",
            [](View& self, int64_t index) -> TObject*
            {
                if (index < 0)
                    index += static_cast<int64_t>(self.size());
                if (index < 0 || static_cast<size_t>(index) >= self.size())
                    throw nb::index_error();
                return self[static_cast<size_t>(index)];
            },
            nb::rv_policy::reference_internal
        )
        .def(
            "__iter__",
            [](View& self)
            {
                return nb::make_iterator(nb::type<View>(), "Iterator", self.begin(), self.end());
            },
            nb::keep_alive<0, 1>()
        )
        .def(
            "find",
            [](View& self, std::optional<std::string_view> name, nb::object type) -> nb::object
            {
                for (size_t i = 0; i < self.size(); ++i) {
                    TObject* obj = self[i];
                    if (name.has_value() && obj->name() != name.value())
                        continue;
                    if (!type.is_none() && !nb::isinstance(nb::cast(obj), type))
                        continue;
                    return nb::cast(obj);
                }
                return nb::none();
            },
            "name"_a.none() = nb::none(),
            "type"_a.none() = nb::none(),
            nb::rv_policy::reference_internal,
            "Find the first object matching the given name and/or type. Returns None if not found."
        )
        .def(
            "find_all",
            [](View& self, std::optional<std::string_view> name, nb::object type) -> nb::list
            {
                nb::list result;
                for (size_t i = 0; i < self.size(); ++i) {
                    TObject* obj = self[i];
                    if (name.has_value() && obj->name() != name.value())
                        continue;
                    if (!type.is_none() && !nb::isinstance(nb::cast(obj), type))
                        continue;
                    result.append(nb::cast(obj));
                }
                return result;
            },
            "name"_a.none() = nb::none(),
            "type"_a.none() = nb::none(),
            nb::rv_policy::reference_internal,
            "Find all objects matching the given name and/or type."
        );
}

class PyMaterial : public Material {
public:
    NB_TRAMPOLINE(Material, 10);
    virtual void set_properties(const Properties& props) override { NB_OVERRIDE(set_properties, props); }
    virtual void on_create(std::optional<const Properties> props) override { NB_OVERRIDE(on_create, props); }
    virtual void on_destroy() override { NB_OVERRIDE(on_destroy); }
    virtual void on_add_to_scene() override { NB_OVERRIDE(on_add_to_scene); }
    virtual void on_remove_from_scene() override { NB_OVERRIDE(on_remove_from_scene); }
    virtual void on_load_resources() override { NB_OVERRIDE(on_load_resources); }
    virtual void update(SceneUpdateContext& ctx) override { _py_update(&ctx); }
    /// Required to bypass the copy forced by nanobind when SceneUpdateContext is passed by reference rather than
    /// pointer.
    void _py_update(SceneUpdateContext* ctx)
    {
        // Code adapted from NB_OVERRIDE_PURE_NAME, as we cannot deduce return type from NBBase.
        nanobind::detail::ticket nb_ticket(nb_trampoline, "update", false);
        if (nb_ticket.key.is_valid()) {
            return nanobind::cast<void>(nb_trampoline.base().attr(nb_ticket.key)(ctx));
        } else {
            Material::update(*ctx);
        }
    }
    virtual void write_to_cursor(sgl::BufferElementCursor cursor) const override
    {
        NB_OVERRIDE(write_to_cursor, cursor);
    }
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const override { NB_OVERRIDE(write_to_cursor, cursor); }
    virtual ref<sgl::SlangModule> required_module() const override { NB_OVERRIDE(required_module); }
    using Material::mark_dirty;
};


class PyGeometry : public Geometry {
    NB_TRAMPOLINE(Geometry, 1);
};

class PyComponent : public Component {
    NB_TRAMPOLINE(Component, 1);
};

class PySceneGlobals : public SceneGlobals {
    NB_TRAMPOLINE(SceneGlobals, 1);
    virtual void bind(sgl::ShaderCursor globals) const override { NB_OVERRIDE_PURE(bind, globals); }
};

} // namespace falcor

FALCOR_PY_EXPORT(render_scene)
{
    using namespace falcor;

    nb::sgl_enum_flags<SceneBindFlags>(m, "SceneBindFlags", D(SceneBindFlags));
    nb::sgl_enum_flags<SceneUpdateFlags>(m, "SceneUpdateFlags", D(SceneUpdateFlags));

    nb::enum_<shared::GeometryInstanceID>(
        m,
        "GeometryInstanceID",
        D_NA(GeometryInstanceID),
        nb::is_flag(),
        nb::is_arithmetic()
    )
        .value("invalid", shared::GeometryInstanceID::invalid);

    nb::enum_<shared::MaterialID>(m, "MaterialID", D_NA(MaterialID), nb::is_arithmetic(), nb::is_flag())
        .value("invalid", shared::MaterialID::invalid);

    nb::enum_<shared::MaterialFlags>(m, "MaterialFlags", D_NA(MaterialFlags), nb::is_arithmetic(), nb::is_flag())
        .value("none", shared::MaterialFlags::none)
        .value("two_sided", shared::MaterialFlags::two_sided)
        .value("thin_walled", shared::MaterialFlags::thin_walled)
        .value("unlit", shared::MaterialFlags::unlit);

    nb::enum_<shared::OpacityFlags>(m, "OpacityFlags", D_NA(OpacityFlags), nb::is_arithmetic(), nb::is_flag())
        .value("none", shared::OpacityFlags::none)
        .value("enabled", shared::OpacityFlags::enabled)
        .value("use_threshold", shared::OpacityFlags::use_threshold)
        .value("evaluate_material", shared::OpacityFlags::evaluate_material);

    nb::sgl_enum<AlphaMode>(m, "AlphaMode", D(AlphaMode));

    nb::enum_<shared::GeometryType>(m, "GeometryType", D_NA(GeometryType), nb::is_arithmetic())
        .value("triangle", shared::GeometryType::triangle)
        .value("lss", shared::GeometryType::lss);

    nb::class_<SceneUpdateContext>(m, "SceneUpdateContext", D(SceneUpdateContext))
        .def_prop_ro(
            "device",
            &SceneUpdateContext::device,
            nb::rv_policy::reference_internal,
            D(SceneUpdateContext, device)
        )
        .def_prop_ro(
            "command_encoder",
            &SceneUpdateContext::command_encoder,
            nb::rv_policy::reference_internal,
            D(SceneUpdateContext, command_encoder)
        )
        .def("submit", &SceneUpdateContext::submit, D(SceneUpdateContext, submit));

    nb::class_<SceneRequirements>(m, "SceneRequirements", D(SceneRequirements))
        .def_ro("modules", &SceneRequirements::modules, D(SceneRequirements, modules))
        .def_ro(
            "type_conformances",
            &SceneRequirements::type_conformances,
            nb::rv_policy::copy,
            D(SceneRequirements, type_conformances)
        )
        .def_ro(
            "ray_tracing_pipeline_flags",
            &SceneRequirements::ray_tracing_pipeline_flags,
            D(SceneRequirements, ray_tracing_pipeline_flags)
        )
        .def_ro(
            "requires_opacity_evaluation",
            &SceneRequirements::requires_opacity_evaluation,
            D(SceneRequirements, requires_opacity_evaluation)
        );

    nb::sgl_enum<SceneObjectKind>(m, "SceneObjectKind", D(SceneObjectKind));

    bind_collection_view<Material>(m, "MaterialCollectionView");
    bind_collection_view<Geometry>(m, "GeometryCollectionView");
    bind_collection_view<Animation>(m, "AnimationCollectionView");
    bind_collection_view<Entity>(m, "EntityCollectionView");
    bind_collection_view<Component>(m, "ComponentCollectionView");

    nb::class_<SceneObject, ReflectedObject>(m, "SceneObject", D(SceneObject))
        .DEF_PROP_RO(SceneObject, scene)
        .DEF_PROP_RO(SceneObject, kind)
        .DEF_PROP_RO(SceneObject, collection_index)
        .DEF_PROP_RW(SceneObject, name)
        .def("remove", &SceneObject::remove, D(SceneObject, remove))
        .DEF_PROP_RO(SceneObject, is_valid)
        .def("on_create", &SceneObject::on_create, "props"_a.none() = nb::none(), D(SceneObject, on_create))
        .def("on_destroy", &SceneObject::on_destroy, D(SceneObject, on_destroy))
        .def("on_add_to_scene", &SceneObject::on_add_to_scene, D(SceneObject, on_add_to_scene))
        .def("on_remove_from_scene", &SceneObject::on_remove_from_scene, D(SceneObject, on_remove_from_scene))
        .def("on_load_resources", &SceneObject::on_load_resources, D(SceneObject, on_load_resources));

    nb::class_<Material, PyMaterial, SceneObject> material(m, "Material", D(Material));
    nb::sgl_enum_flags<Material::DirtyFlags>(material, "DirtyFlags");
    reflection::bind<Material>(material);
    material
        .def(nb::init<>()) // __init__ is needed for Python-side inheritance
        .def_prop_rw(
            "slang_type_name",
            &Material::slang_type_name,
            &Material::set_slang_type_name,
            D(Material, slang_type_name)
        )
        .def_prop_ro(
            "material_id",
            [](const Material& self)
            {
                return static_cast<uint32_t>(self.material_id());
            },
            "The material's allocated ID. Equals MATERIAL_ID_INVALID until "
            "assigned by MaterialSystem::update."
        )
        .def_prop_ro("flags", &Material::flags, "Material flags used by the scene material header.")
        .def(
            "update",
            [](Material& self, SceneUpdateContext* ctx)
            {
                self.update(*ctx);
            },
            "ctx"_a,
            D(Material, update)
        )
        .def(
            "write_to_cursor",
            nb::overload_cast<sgl::BufferElementCursor>(&Material::write_to_cursor, nb::const_),
            "cursor"_a,
            "Write this material to a BufferElementCursor (Slang struct)."
        )
        .def(
            "write_to_cursor",
            nb::overload_cast<sgl::ShaderCursor>(&Material::write_to_cursor, nb::const_),
            "cursor"_a,
            "Write this material to a ShaderCursor (Slang struct)."
        )
        .def(
            "required_module",
            &Material::required_module,
            "Returns the additional Slang module required by this material, or None."
        )
        .def("mark_dirty", &PyMaterial::mark_dirty, "flags"_a, "Mark this material as dirty.");

    m.attr("MaterialT") = nb::type_var("MaterialT", "bound"_a = nb::type<Material>());

    nb::class_<Geometry, PyGeometry, SceneObject>(m, "Geometry", D(Geometry)) //
        .def(nb::init<>()) // __init__ is needed for Python-side inheritance
        .DEF_PROP_RO(Geometry, local_aabb);
    m.attr("GeometryT") = nb::type_var("GeometryT", "bound"_a = nb::type<Geometry>());

    nb::class_<Component, SceneObject>(m, "Component", D(Component)) //
        .def(nb::init<>())                                           // __init__ is needed for Python-side inheritance
        .DEF_PROP_RO(Component, entity);
    m.attr("ComponentT") = nb::type_var("ComponentT", "bound"_a = nb::type<Component>());

    nb::class_<SceneGlobals, PySceneGlobals, Object>(m, "SceneGlobals", D(SceneGlobals)).def(nb::init<>());

    nb::class_<AnimationChannel>(m, "AnimationChannel", D(AnimationChannel))
        .def(nb::init<>())
        .DEF_RW(AnimationChannel, value_type)
        .DEF_RW(AnimationChannel, interpolation_mode)
        .DEF_RW(AnimationChannel, times)
        .DEF_RW(AnimationChannel, values)
        .DEF_RW(AnimationChannel, in_tangents)
        .DEF_RW(AnimationChannel, out_tangents)
        .DEF_PROP_RO(AnimationChannel, components_per_value)
        .DEF_PROP_RO(AnimationChannel, keyframe_count);

    nb::class_<Animation, SceneObject>(m, "Animation", D(Animation))
        .DEF_RW(Animation, channels)
        .DEF_PROP_RO(Animation, duration);

    nb::class_<TransformAnimation, Component>(m, "TransformAnimation", D(TransformAnimation))
        .DEF_PROP_RW(TransformAnimation, animation)
        .DEF_PROP_RW(TransformAnimation, translation_channel)
        .DEF_PROP_RW(TransformAnimation, rotation_channel)
        .DEF_PROP_RW(TransformAnimation, scale_channel)
        .def("has_channels", &TransformAnimation::has_channels, D(TransformAnimation, has_channels));

    nb::class_<Entity, SceneObject>(m, "Entity", D(Entity))
        .DEF_PROP_RW(Entity, parent)
        .DEF_PROP_RO(Entity, children)
        .DEF_PROP_RO(Entity, components)
        .DEF_PROP_RW(Entity, transform)
        .def("set_world_transform", &Entity::set_world_transform, "world_transform"_a, D(Entity, set_world_transform))
        .DEF_PROP_RO(Entity, world_from_object_matrix)
        .DEF_PROP_RO(Entity, world_aabb)
        .def(
            "create_component",
            [](Entity* self, std::string_view type, std::optional<Properties> props)
            {
                return self->create_component(type, props);
            },
            "type"_a,
            "props"_a.none() = nb::none()
        )
        .def(
            "create_component",
            [](Entity* self, nb::type_object cls, std::optional<Properties> props) -> nb::object
            {
                if (SceneObjectFactory<Component>::get().find_class_info(nb::type_info(cls)) != nullptr) {
                    return nb::cast(self->create_component(nb::type_info(cls), props));
                } else {
                    if (!python_class_inherits_from<Component>(cls))
                        throw nb::type_error("cls must inherit from Component");
                    return create_python_scene_object<Component>(self->scene(), cls, props);
                }
            },
            "cls"_a,
            "props"_a.none() = nb::none(),
            nb::sig("def create_component(self, cls: type[ComponentT], props: Properties | None = None) -> ComponentT")
        );

    bind_signal<SceneUpdatedSignal>(m, "SceneUpdatedSignal");

    nb::class_<SceneConfig>(m, "SceneConfig", D(SceneConfig))
        .def(nb::init<UVOrigin>(), "uv_origin"_a = UVOrigin::upper_left)
        .DEF_RO(SceneConfig, uv_origin);

    nb::class_<Scene, Object>(m, "Scene", D(Scene))
        .def_static("create", &Scene::create, "device"_a, "config"_a = SceneConfig(), D(Scene, create))
        .def_static(
            "load",
            &Scene::load,
            "device"_a,
            "path"_a,
            "import_options"_a.none() = nb::none(),
            "config"_a.none() = nb::none(),
            D(Scene, load)
        )
        .def_static(
            "from_importer_scene",
            &Scene::from_importer_scene,
            "device"_a,
            "importer_scene"_a,
            "config"_a.none() = nb::none(),
            D(Scene, from_importer_scene)
        )
        .def_static(
            "from_importer",
            &Scene::from_importer,
            "device"_a,
            "importer"_a,
            "config"_a.none() = nb::none(),
            D(Scene, from_importer)
        )
        .def(
            "append",
            nb::overload_cast<const std::filesystem::path&, std::optional<ImportOptions>>(&Scene::append),
            "path"_a,
            "import_options"_a.none() = nb::none(),
            D(Scene, append)
        )
        .def("append", nb::overload_cast<const ImporterScene&>(&Scene::append), "importer_scene"_a, D(Scene, append_2))
        .def("append", nb::overload_cast<const Importer&>(&Scene::append), "importer"_a, D(Scene, append_3))
        .DEF_PROP_RO(Scene, device)
        .DEF_PROP_RO(Scene, config)
        .DEF_PROP_RO(Scene, texture_manager)
        .def("add_scene_globals", &Scene::add_scene_globals, "globals"_a, D(Scene, add_scene_globals))
        .def("remove_scene_globals", &Scene::remove_scene_globals, "globals"_a, D(Scene, remove_scene_globals))
        .def(
            "create_material",
            [](Scene* self, std::string_view type, std::optional<Properties> props)
            {
                return self->create_material(type, props);
            },
            "type"_a,
            "props"_a.none() = nb::none()
        )
        .def(
            "create_material",
            [](Scene* self, nb::type_object cls, std::optional<Properties> props) -> nb::object
            {
                if (SceneObjectFactory<Material>::get().find_class_info(nb::type_info(cls)) != nullptr) {
                    return nb::cast(self->create_material(nb::type_info(cls), props));
                } else {
                    if (!python_class_inherits_from<Material>(cls))
                        throw nb::type_error("cls must inherit from Material");
                    return create_python_scene_object<Material>(self, cls, props);
                }
            },
            "cls"_a,
            "props"_a.none() = nb::none(),
            nb::sig("def create_material(self, cls: type[MaterialT], props: Properties | None = None) -> MaterialT")
        )
        .def(
            "create_geometry",
            [](Scene* self, std::string_view type, std::optional<Properties> props)
            {
                return self->create_geometry(type, props);
            },
            "type"_a,
            "props"_a.none() = nb::none()
        )
        .def(
            "create_geometry",
            [](Scene* self, nb::type_object cls, std::optional<Properties> props) -> nb::object
            {
                if (SceneObjectFactory<Geometry>::get().find_class_info(nb::type_info(cls)) != nullptr) {
                    return nb::cast(self->create_geometry(nb::type_info(cls), props));
                } else {
                    if (!python_class_inherits_from<Geometry>(cls))
                        throw nb::type_error("cls must inherit from Geometry");
                    return create_python_scene_object<Geometry>(self, cls, props);
                }
            },
            "cls"_a,
            "props"_a.none() = nb::none(),
            nb::sig("def create_geometry(self, cls: type[GeometryT], props: Properties | None = None) -> GeometryT")
        )
        .def("create_animation", &Scene::create_animation, D(Scene, create_animation))
        .def("create_entity", &Scene::create_entity, "props"_a.none() = nb::none(), D(Scene, create_entity))
        .def("update", nb::overload_cast<>(&Scene::update), D(Scene, update))
        .def_prop_ro("update_flags", &Scene::update_flags, D(Scene, update_flags))
        .def_prop_ro("update_generation", &Scene::update_generation, D(Scene, update_generation))
        .def_prop_ro("updated", &Scene::updated, nb::rv_policy::reference_internal, D(Scene, updated))
        .def_prop_ro("render_module", &Scene::render_module, nb::rv_policy::reference_internal, D(Scene, render_module))
        .def_prop_ro("requirements", &Scene::requirements, nb::rv_policy::reference_internal, D(Scene, requirements))
        .def_prop_ro("requirements_generation", &Scene::requirements_generation, D(Scene, requirements_generation))
        .def("bind", &Scene::bind, "globals"_a, "flags"_a = SceneBindFlags::all, D(Scene, bind))
        .def_prop_ro(
            "materials",
            nb::overload_cast<>(&Scene::materials),
            nb::rv_policy::reference_internal,
            D(Scene, materials)
        )
        .def_prop_ro(
            "geometries",
            nb::overload_cast<>(&Scene::geometries),
            nb::rv_policy::reference_internal,
            D(Scene, geometries)
        )
        .def_prop_ro(
            "animations",
            nb::overload_cast<>(&Scene::animations),
            nb::rv_policy::reference_internal,
            D(Scene, animations)
        )
        .def_prop_ro(
            "entities",
            nb::overload_cast<>(&Scene::entities),
            nb::rv_policy::reference_internal,
            D(Scene, entities)
        )
        .def_prop_ro(
            "components",
            nb::overload_cast<>(&Scene::components),
            nb::rv_policy::reference_internal,
            D(Scene, components)
        )
        .def_prop_rw(
            "active_camera",
            &Scene::active_camera,
            &Scene::set_active_camera,
            nb::arg().none(),
            nb::rv_policy::reference_internal,
            D(Scene, active_camera)
        )
        .DEF_PROP_RW(Scene, time)
        .DEF_PROP_RO(Scene, animation_duration)
        .def("has_animation", &Scene::has_animation, D(Scene, has_animation));

    nb::class_<GeometryGroup, Geometry>(m, "GeometryGroup", D(GeometryGroup))
        .DEF_PROP_RW(GeometryGroup, geometries, nb::rv_policy::reference_internal);

    nb::class_<GeometryInstance, Component>(m, "GeometryInstance", D(GeometryInstance))
        .DEF_PROP_RW(GeometryInstance, geometry)
        .DEF_PROP_RW(GeometryInstance, materials, nb::rv_policy::reference_internal)
        .DEF_PROP_RO(GeometryInstance, geometry_instance_id)
        .DEF_PROP_RO(GeometryInstance, geometry_instance_count);

    nb::class_<EmissionOnlyMaterial, Material> emission_only_material(
        m,
        "EmissionOnlyMaterial",
        D(EmissionOnlyMaterial)
    );
    reflection::bind<EmissionOnlyMaterial>(emission_only_material);

    nb::class_<StandardMaterial, Material> standard_material(m, "StandardMaterial", D(StandardMaterial));
    reflection::bind<StandardMaterial>(standard_material);

    nb::class_<UnlitMaterial, Material> unlit_material(m, "UnlitMaterial", D(UnlitMaterial));
    reflection::bind<UnlitMaterial>(unlit_material);

    nb::class_<StandardSpecGlossMaterial, Material> standard_specgloss_material(
        m,
        "StandardSpecGlossMaterial",
        D(StandardSpecGlossMaterial)
    );
    reflection::bind<StandardSpecGlossMaterial>(standard_specgloss_material);

    nb::class_<MDLMaterial, Material> mdl_material(m, "MDLMaterial", D_NA(MDLMaterial));
    reflection::bind<MDLMaterial>(mdl_material);
    mdl_material.def(
        "build_texture_list",
        &MDLMaterial::build_texture_list,
        "List the TextureHandle instances used by this material. "
        "Valid after scene.update() / on_load_resources() has run."
    );

    nb::class_<MaterialXMaterial, Material> mtlx_material(m, "MaterialXMaterial", D(MaterialXMaterial));
    reflection::bind<MaterialXMaterial>(mtlx_material);
    mtlx_material.def(
        "build_texture_list",
        &MaterialXMaterial::build_texture_list,
        "List the TextureHandle instances used by this material. "
        "Valid after scene.update() / on_load_resources() has run."
    );

    nb::class_<CameraUniforms>(m, "CameraUniforms")
        .DEF_PROP_RO(CameraUniforms, width)
        .DEF_PROP_RO(CameraUniforms, height)
        .DEF_RO(CameraUniforms, dims)
        .DEF_RO(CameraUniforms, position)
        .DEF_RO(CameraUniforms, image_u)
        .DEF_RO(CameraUniforms, image_v)
        .DEF_RO(CameraUniforms, image_w)
        .DEF_RO(CameraUniforms, aperture_radius)
        .DEF_RO(CameraUniforms, focus_distance)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    nb::class_<Camera, Component> camera(m, "Camera", D(Camera));
    reflection::bind<Camera>(camera);
    camera.def("recompute", &Camera::recompute, D(Camera, recompute));
    camera.def("calc_uniforms", nb::overload_cast<>(&Camera::calc_uniforms, nb::const_), D(Camera, calc_uniforms));
    camera.def(
        "calc_uniforms",
        nb::overload_cast<int, int>(&Camera::calc_uniforms, nb::const_),
        "width"_a,
        "height"_a,
        D(Camera, calc_uniforms)
    );
    camera.def("calc_view_from_world", &Camera::calc_view_from_world, D(Camera, calc_view_from_world));
    camera.def(
        "calc_clip_from_view",
        nb::overload_cast<>(&Camera::calc_clip_from_view, nb::const_),
        D(Camera, calc_clip_from_view)
    );
    camera.def(
        "calc_clip_from_view",
        nb::overload_cast<int, int>(&Camera::calc_clip_from_view, nb::const_),
        "width"_a,
        "height"_a,
        D(Camera, calc_clip_from_view)
    );
    reflection::bind<Light>(m);
    reflection::bind<ConstantLight>(m);
    reflection::bind<DistantLight>(m);
    reflection::bind<PointLight>(m);
    reflection::bind<SphereLight>(m);
    reflection::bind<DiskLight>(m);
    reflection::bind<RectLight>(m);
    reflection::bind<EnvMapLight>(m);
}
