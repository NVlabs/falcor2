// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light_import.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/scene.h"
#include "falcor2/utils/math/transform.h"
#include <cmath>

namespace falcor {
namespace {

/// Reuse a compatible component, otherwise remove it and create one owned by the entity.
template<typename T>
T* ensure_light(Entity* entity, Light* previous)
{
    if (auto* light = dynamic_cast<T*>(previous))
        return light;
    if (previous)
        previous->remove();
    return entity->create_component<T>();
}

/// Apply source radiometry and metadata shared by all runtime light types.
static void set_common_light_properties(Light* light, const ImporterLight& importer_light)
{
    light->_set_source_asset_ref(importer_light.source_asset_ref);
    light->set_color(importer_light.color);
    light->set_intensity(importer_light.intensity);
    light->set_exposure(importer_light.exposure);
    light->set_normalize(importer_light.normalize);
    light->set_enable_color_temperature(importer_light.enable_color_temperature);
    light->set_color_temperature(importer_light.color_temperature);
    light->set_geometry_visible(importer_light.geometry_visible);
}

/// Apply cone/focus parameters to a runtime type that supports spot shaping.
template<typename LightT>
static void set_light_shaping_properties(LightT* light, const ImporterLight& importer_light)
{
    light->set_enable_shaping(importer_light.enable_shaping);
    light->set_shaping_cone_angle(importer_light.shaping_cone_angle);
    light->set_shaping_cone_softness(importer_light.shaping_cone_softness);
    light->set_shaping_focus(importer_light.shaping_focus);
}

} // namespace
Light* update_imported_light(Entity* entity, Light* previous, const ImporterLight& importer_light)
{
    switch (importer_light.type) {
    case ImporterLight::Type::distant: {
        DistantLight* distant_light = ensure_light<DistantLight>(entity, previous);
        set_common_light_properties(distant_light, importer_light);

        // Scene distant light expects a half angle
        distant_light->set_cutoff_angle(0.5f * importer_light.degree_angular_diameter);
        return distant_light;
    }
    case ImporterLight::Type::rectangular: {
        RectLight* rect_light = ensure_light<RectLight>(entity, previous);
        set_common_light_properties(rect_light, importer_light);
        set_light_shaping_properties(rect_light, importer_light);
        rect_light->set_width(importer_light.width);
        rect_light->set_height(importer_light.height);
        return rect_light;
    }
    case ImporterLight::Type::sphere: {
        if (importer_light.treat_as_point) {
            PointLight* point_light = ensure_light<PointLight>(entity, previous);
            set_common_light_properties(point_light, importer_light);
            set_light_shaping_properties(point_light, importer_light);

            // Match the emitted power of the source sphere. A Lambertian sphere emits
            // pi * radiance * surface_area, while an isotropic point emits
            // 4 * pi * radiant_intensity.
            float point_intensity = 0.25f * importer_light.intensity;
            if (!importer_light.normalize) {
                const float radius_ws
                    = importer_light.radius * volume_equivalent_uniform_scale(entity->world_from_object_matrix());
                point_intensity *= 4.f * static_cast<float>(M_PI) * radius_ws * radius_ws;
            }
            point_light->set_intensity(point_intensity);
            point_light->set_normalize(false);
            return point_light;
        }
        SphereLight* sphere_light = ensure_light<SphereLight>(entity, previous);
        set_common_light_properties(sphere_light, importer_light);
        set_light_shaping_properties(sphere_light, importer_light);
        sphere_light->set_radius(importer_light.radius);
        sphere_light->set_enable_virtual_sphere_shrinking(importer_light.enable_virtual_sphere_shrinking);
        return sphere_light;
    }
    case ImporterLight::Type::point: {
        PointLight* point_light = ensure_light<PointLight>(entity, previous);
        set_common_light_properties(point_light, importer_light);
        set_light_shaping_properties(point_light, importer_light);
        point_light->set_normalize(false);
        return point_light;
    }
    case ImporterLight::Type::cylinder: {
        CylinderLight* cylinder_light = ensure_light<CylinderLight>(entity, previous);
        set_common_light_properties(cylinder_light, importer_light);
        set_light_shaping_properties(cylinder_light, importer_light);
        cylinder_light->set_radius(importer_light.radius);
        cylinder_light->set_length(importer_light.length);
        return cylinder_light;
    }
    case ImporterLight::Type::disk: {
        DiskLight* disk_light = ensure_light<DiskLight>(entity, previous);
        set_common_light_properties(disk_light, importer_light);
        set_light_shaping_properties(disk_light, importer_light);
        disk_light->set_radius(importer_light.radius);
        return disk_light;
    }
    case ImporterLight::Type::dome: {
        if (importer_light.env_map_path.empty()) {
            auto* light = ensure_light<ConstantLight>(entity, previous);
            set_common_light_properties(light, importer_light);
            return light;
        }
        auto* light = ensure_light<EnvMapLight>(entity, previous);
        light->set_pole_axis(importer_light.pole_axis);
        light->set_env_map_path(importer_light.env_map_path);
        set_common_light_properties(light, importer_light);
        return light;
    }
    case ImporterLight::Type::constant: {
        ConstantLight* constant_light = ensure_light<ConstantLight>(entity, previous);
        set_common_light_properties(constant_light, importer_light);
        return constant_light;
    }
    }
    FALCOR_THROW("Unsupported imported light type");
}
} // namespace falcor
