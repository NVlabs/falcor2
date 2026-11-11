// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "scene_object_templates.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/openpbr_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/standard_specgloss_material.h"
#include "falcor2/render/material/unlit_material.h"

#include <array>
#include <string>

namespace falcor::ui {
namespace {

SceneObject* create_empty_entity(Scene& scene)
{
    return scene.create_entity();
}

template<typename T>
void configure_noop(T*)
{
}

template<typename T, void (*configure)(T*) = configure_noop<T>>
SceneObject* create_entity_with_component(Scene& scene)
{
    Entity* entity = scene.create_entity();
    try {
        T* component = entity->create_component<T>();
        configure(component);
    } catch (...) {
        entity->remove();
        throw;
    }
    return entity;
}

void configure_spot_light(PointLight* light)
{
    light->set_enable_shaping(true);
    light->set_shaping_cone_angle(30.f);
    light->set_shaping_cone_softness(0.2f);
}

template<typename T>
SceneObject* create_material(Scene& scene)
{
    return scene.create_material<T>();
}

constexpr std::array<SceneObjectTemplate, 16> SCENE_OBJECT_TEMPLATES{{
    {"entity", "Empty Entity", "", create_empty_entity},
    {"camera", "Camera", "", create_entity_with_component<Camera>},
    {"point_light", "Point Light", "Light", create_entity_with_component<PointLight>},
    {"spot_light", "Spot Light", "Light", create_entity_with_component<PointLight, configure_spot_light>},
    {"distant_light", "Distant Light", "Light", create_entity_with_component<DistantLight>},
    {"rect_light", "Rect Light", "Light", create_entity_with_component<RectLight>},
    {"disk_light", "Disk Light", "Light", create_entity_with_component<DiskLight>},
    {"sphere_light", "Sphere Light", "Light", create_entity_with_component<SphereLight>},
    {"cylinder_light", "Cylinder Light", "Light", create_entity_with_component<CylinderLight>},
    {"constant_light", "Constant Light", "Light", create_entity_with_component<ConstantLight>},
    {"environment_light", "Environment Map", "Light", create_entity_with_component<EnvMapLight>},
    {"material.EmissionOnlyMaterial", "Emission Only Material", "Material", create_material<EmissionOnlyMaterial>},
    {"material.OpenPBRMaterial", "OpenPBR Material", "Material", create_material<OpenPBRMaterial>},
    {"material.StandardMaterial", "Standard Material", "Material", create_material<StandardMaterial>},
    {"material.StandardSpecGlossMaterial",
     "Standard Specular-Glossiness Material",
     "Material",
     create_material<StandardSpecGlossMaterial>},
    {"material.UnlitMaterial", "Unlit Material", "Material", create_material<UnlitMaterial>},
}};

} // namespace

CommandId SceneObjectTemplate::command_id() const
{
    return CommandId(std::string("scene_editor.create.").append(key));
}

std::span<const SceneObjectTemplate> scene_object_templates()
{
    return SCENE_OBJECT_TEMPLATES;
}

} // namespace falcor::ui
