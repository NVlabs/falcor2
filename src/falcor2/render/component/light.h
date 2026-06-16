// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/component.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflection.h"

#include <sgl/device/buffer_cursor.h>

namespace sgl {
class SignatureBuffer;
} // namespace sgl

namespace falcor {

/// Base class for lights.
/// Each sub-class represents a different light type on the shader side.
class FALCOR_API Light : public Component {
    FALCOR_SCENE_OBJECT(Light, Component)
public:
    FALCOR_STATIC_WRITE_TO_CURSOR(Light);

    /// Constructor.
    Light();

    /// Destructor.
    ~Light() override;

    // Component interface

    virtual void on_entity_transform_changed() override;

    // Light interface

    /// Active flag. Light does not contribute to the scene lighting if not active.
    bool active() const { return m_active; }
    void set_active(bool active);

    /// Light exposure value in exposure stops.
    float exposure() const { return m_exposure; }
    void set_exposure(float exposure);

    /// Slang type name of the struct representing this light type.
    /// Specifies the type used for write_to_cursor().
    const std::string& slang_type_name() const { return m_slang_type_name; }
    void set_slang_type_name(std::string_view slang_type_name) { m_slang_type_name = slang_type_name; }

    /// Light type.
    shared::LightType light_type() const { return m_light_type; }
    void set_light_type(shared::LightType light_type) { m_light_type = light_type; }

    /// Light flags.
    shared::LightFlags light_flags() const { return m_light_flags; }
    void set_light_flags(shared::LightFlags light_flags) { m_light_flags = light_flags; }

    /// Light ID.
    shared::LightID light_id() const { return m_light_id; }
    void set_light_id(shared::LightID light_id) { m_light_id = light_id; }

    /// Called during Scene::update() to update this light.
    /// @param ctx Update context.
    virtual void update(SceneUpdateContext& ctx) { FALCOR_UNUSED(ctx); }

    /// Called during Scene::update() to write this light to the corresponding slang struct.
    virtual void write_to_cursor(sgl::BufferElementCursor cursor) const = 0;
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const = 0;

    /// Write the SlangPy cache signature for cursor-writer binding.
    static void write_slangpy_signature(sgl::SignatureBuffer& signature, const Light* value);

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "active",
                &Light::active,
                &Light::set_active,
                "Active flag.",
                reflection::default_value(true),
                reflection::ui_label("Active")
            )
            .def_prop_rw(
                "exposure",
                &Light::exposure,
                &Light::set_exposure,
                "Exposure value in stops.",
                reflection::default_value(0.f),
                reflection::ui_label("Exposure")
            );
    }

protected:
    bool m_active{true};
    float m_exposure{0.f};
    std::string m_slang_type_name;
    shared::LightType m_light_type{0};
    shared::LightFlags m_light_flags{shared::LightFlags::none};
    shared::LightID m_light_id{shared::LightID::invalid};
};

class FALCOR_API ConstantLight : public Light {
    FALCOR_SCENE_OBJECT(ConstantLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    ConstantLight();
    ~ConstantLight() override;

    float3 radiance() const { return m_radiance; }
    void set_radiance(float3 radiance);

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "radiance",
                &ConstantLight::radiance,
                &ConstantLight::set_radiance,
                "Constant radiance.",
                reflection::default_value(float3(1.f)),
                reflection::ui_label("Radiance"),
                reflection::UIFlags::display_as_color
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_radiance{1.f};
};

class FALCOR_API DistantLight : public Light {
    FALCOR_SCENE_OBJECT(DistantLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    DistantLight();
    ~DistantLight() override;

    float3 radiance() const { return m_radiance; }
    void set_radiance(float3 radiance);

    float cutoff_angle() const { return m_cutoff_angle; }
    void set_cutoff_angle(float cutoff_angle);

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "radiance",
                &DistantLight::radiance,
                &DistantLight::set_radiance,
                "Radiance.",
                reflection::default_value(float3(1.f)),
                reflection::ui_label("Radiance"),
                reflection::UIFlags::display_as_color
            )
            .def_prop_rw(
                "cutoff_angle",
                &DistantLight::cutoff_angle,
                &DistantLight::set_cutoff_angle,
                "Cutoff angle in degrees.",
                reflection::default_value(1.f),
                reflection::value_range(0.0, 180.0),
                reflection::ui_label("Cutoff Angle")
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_radiance{1.f};
    float m_cutoff_angle{1.f};
};

class FALCOR_API EnvMapLight : public Light {
    FALCOR_SCENE_OBJECT(EnvMapLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    EnvMapLight();
    ~EnvMapLight() override;

    const std::filesystem::path& env_map_path() const { return m_env_map_path; }
    void set_env_map_path(const std::filesystem::path& env_map_path);

    // SceneObject interface

    virtual void on_load_resources() override;

    // Light interface

    virtual void update(SceneUpdateContext& ctx) override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "env_map_path",
                &EnvMapLight::env_map_path,
                &EnvMapLight::set_env_map_path,
                "Environment map file path."
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    std::filesystem::path m_env_map_path;

    TextureHandle m_env_map_texture_handle;
    bool m_env_map_loaded{false};

    ref<sgl::Texture> m_importance_map_texture;
    bool m_importance_map_dirty{true};

    ref<sgl::ComputeKernel> m_build_importance_map_kernel;
};

class FALCOR_API PointLight : public Light {
    FALCOR_SCENE_OBJECT(PointLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    PointLight();
    ~PointLight() override;

    float3 intensity() const { return m_intensity; }
    void set_intensity(float3 intensity);

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "intensity",
                &PointLight::intensity,
                &PointLight::set_intensity,
                "Light intensity.",
                reflection::default_value(float3(1.f)),
                reflection::ui_label("Intensity"),
                reflection::UIFlags::display_as_color
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_intensity{1.f};
};

class FALCOR_API SpotLight : public Light {
    FALCOR_SCENE_OBJECT(SpotLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    SpotLight();
    ~SpotLight() override;

    float3 intensity() const { return m_intensity; }
    void set_intensity(float3 intensity);

    float cutoff_angle() const { return m_cutoff_angle; }
    void set_cutoff_angle(float cutoff_angle);

    float falloff_angle() const { return m_falloff_angle; }
    void set_falloff_angle(float falloff_angle);

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "intensity",
                &SpotLight::intensity,
                &SpotLight::set_intensity,
                "Light intensity.",
                reflection::default_value(float3(1.f)),
                reflection::ui_label("Intensity"),
                reflection::UIFlags::display_as_color
            )
            .def_prop_rw(
                "cutoff_angle",
                &SpotLight::cutoff_angle,
                &SpotLight::set_cutoff_angle,
                "Cutoff angle in degrees.",
                reflection::default_value(45.f),
                reflection::value_range(0.0, 180.0),
                reflection::ui_label("Cutoff Angle")
            )
            .def_prop_rw(
                "falloff_angle",
                &SpotLight::falloff_angle,
                &SpotLight::set_falloff_angle,
                "Falloff angle in degrees.",
                reflection::default_value(40.f),
                reflection::value_range(0.0, 180.0),
                reflection::ui_label("Falloff Angle")
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_intensity{1.f};
    float m_cutoff_angle{45.f};
    float m_falloff_angle{40.f};
};

} // namespace falcor
