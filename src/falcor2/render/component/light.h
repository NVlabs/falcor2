// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/component.h"
#include "falcor2/render/scene_types.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflection.h"
#include "falcor2/utils/color.h"

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

    /// Light color in the rendering color space.
    float3 color() const { return m_color; }
    void set_color(float3 color);

    /// Scalar light intensity.
    float intensity() const { return m_intensity; }
    void set_intensity(float intensity);

    /// Light exposure value in exposure stops.
    float exposure() const { return m_exposure; }
    void set_exposure(float exposure);

    /// Whether emission is divided by the light's world-space size factor.
    bool normalize() const { return m_normalize; }
    void set_normalize(bool normalize);

    /// Whether the color temperature multiplier is enabled.
    bool enable_color_temperature() const { return m_enable_color_temperature; }
    void set_enable_color_temperature(bool enable_color_temperature);

    /// Color temperature in Kelvin.
    float color_temperature() const { return m_color_temperature; }
    void set_color_temperature(float color_temperature);

    /// Whether finite area-light geometry is visible, opaque, and shadow-casting.
    bool geometry_visible() const { return m_geometry_visible; }
    void set_geometry_visible(bool visible);

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
                "color",
                &Light::color,
                &Light::set_color,
                "Light color in the rendering color space.",
                reflection::default_value(float3(1.f)),
                reflection::ui_label("Color"),
                reflection::UIFlags::display_as_color
            )
            .def_prop_rw(
                "intensity",
                &Light::intensity,
                &Light::set_intensity,
                "Scalar light intensity.",
                reflection::default_value(1.f),
                reflection::value_range_positive(),
                reflection::ui_label("Intensity")
            )
            .def_prop_rw(
                "exposure",
                &Light::exposure,
                &Light::set_exposure,
                "Exposure value in stops.",
                reflection::default_value(0.f),
                reflection::value_range(MIN_EXPOSURE, MAX_EXPOSURE),
                reflection::ui_label("Exposure")
            )
            .def_prop_rw(
                "normalize",
                &Light::normalize,
                &Light::set_normalize,
                "Divide emission by the light's world-space size factor.",
                reflection::default_value(false),
                reflection::ui_label("Normalize")
            )
            .def_prop_rw(
                "enable_color_temperature",
                &Light::enable_color_temperature,
                &Light::set_enable_color_temperature,
                "Enable the luminance-normalized color temperature multiplier.",
                reflection::default_value(false),
                reflection::ui_label("Enable Color Temperature")
            )
            .def_prop_rw(
                "color_temperature",
                &Light::color_temperature,
                &Light::set_color_temperature,
                "Blackbody color temperature in Kelvin. 6500 K is neutral white.",
                reflection::default_value(6500.f),
                reflection::value_range(MIN_COLOR_TEMPERATURE, MAX_COLOR_TEMPERATURE),
                reflection::ui_label("Color Temperature")
            )
            .def_prop_rw(
                "geometry_visible",
                &Light::geometry_visible,
                &Light::set_geometry_visible,
                "Whether finite area-light geometry is visible, opaque, and shadow-casting.",
                reflection::default_value(false),
                reflection::ui_label("Geometry Visible")
            );
    }

protected:
    static constexpr float MIN_LIGHT_ANGLE = 0.f;
    static constexpr float MAX_LIGHT_ANGLE = 180.f;
    static constexpr float MIN_EXPOSURE = -32.f;
    static constexpr float MAX_EXPOSURE = 32.f;

    float3 evaluate_emission(float size_factor = 1.f) const;

    bool m_active{true};
    float3 m_color{1.f};
    float m_intensity{1.f};
    float m_exposure{0.f};
    bool m_normalize{false};
    bool m_enable_color_temperature{false};
    float m_color_temperature{6500.f};
    bool m_geometry_visible{false};

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

    // Light interface

    template<reflection::ClassReflector R>
    static void reflect(R&)
    {
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;
};

class FALCOR_API DistantLight : public Light {
    FALCOR_SCENE_OBJECT(DistantLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    DistantLight();
    ~DistantLight() override;

    float cutoff_angle() const { return m_cutoff_angle; }
    void set_cutoff_angle(float cutoff_angle);

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "cutoff_angle",
                &DistantLight::cutoff_angle,
                &DistantLight::set_cutoff_angle,
                "Cutoff angle in degrees.",
                reflection::default_value(1.f),
                reflection::value_range(MIN_LIGHT_ANGLE, MAX_LIGHT_ANGLE),
                reflection::ui_label("Cutoff Angle")
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

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
    void set_env_map_texture(ref<sgl::Texture> env_map_texture);

    /// Alignment of the environment map's top pole in the owning entity's local space.
    EnvMapPoleAxis pole_axis() const { return m_pole_axis; }
    void set_pole_axis(EnvMapPoleAxis pole_axis);

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
            )
            .def_prop_rw(
                "pole_axis",
                &EnvMapLight::pole_axis,
                &EnvMapLight::set_pole_axis,
                "Environment map pole axis in the owning entity's local space.",
                reflection::default_value(EnvMapPoleAxis::y),
                reflection::ui_label("Pole Axis")
            );
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    std::filesystem::path m_env_map_path;
    EnvMapPoleAxis m_pole_axis{EnvMapPoleAxis::y};

    ref<sgl::Texture> m_env_map_texture;
    TextureHandle m_env_map_texture_handle;
    bool m_env_map_loaded{false};

    ref<sgl::Texture> m_importance_map_texture;
    bool m_importance_map_dirty{true};

    ref<sgl::ComputeKernel> m_build_importance_map_kernel;
};

/// Non-polymorphic state for an optional directional light emission profile.
///
/// Shaping follows USD semantics around the light's local negative-Z axis. It
/// modulates emitted radiance or intensity without changing the emitter shape.
class FALCOR_API LightShaping {
public:
    static constexpr float MIN_CONE_ANGLE = 0.f;
    static constexpr float MAX_CONE_ANGLE = 180.f;

    bool enabled() const { return m_enabled; }
    bool set_enabled(bool enabled);

    float cone_angle() const { return m_cone_angle; }
    bool set_cone_angle(float angle);

    float cone_softness() const { return m_cone_softness; }
    bool set_cone_softness(float softness);

    float focus() const { return m_focus; }
    bool set_focus(float focus);

    template<typename CursorT>
    void write_to_cursor(CursorT cursor) const;

    template<typename LightT, reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "enable_shaping",
                &LightT::enable_shaping,
                &LightT::set_enable_shaping,
                "Enable directional emission shaping.",
                reflection::default_value(false),
                reflection::ui_label("Enable Shaping")
            )
            .def_prop_rw(
                "shaping_cone_angle",
                &LightT::shaping_cone_angle,
                &LightT::set_shaping_cone_angle,
                "Angular cutoff from the local negative-Z axis, in degrees.",
                reflection::default_value(90.f),
                reflection::value_range(MIN_CONE_ANGLE, MAX_CONE_ANGLE),
                reflection::ui_label("Shaping Cone Angle")
            )
            .def_prop_rw(
                "shaping_cone_softness",
                &LightT::shaping_cone_softness,
                &LightT::set_shaping_cone_softness,
                "Fraction of the cone occupied by the smooth transition.",
                reflection::default_value(0.f),
                reflection::value_range(0.f, 1.f),
                reflection::ui_label("Shaping Cone Softness")
            )
            .def_prop_rw(
                "shaping_focus",
                &LightT::shaping_focus,
                &LightT::set_shaping_focus,
                "Off-axis cosine power exponent used to focus emission.",
                reflection::default_value(0.f),
                reflection::value_range_positive(),
                reflection::ui_label("Shaping Focus")
            );
    }

private:
    bool m_enabled{false};
    float m_cone_angle{90.f};
    float m_cone_softness{0.f};
    float m_focus{0.f};
};

#define FALCOR_LIGHT_SHAPING_MIXIN(member)                                                                             \
    bool enable_shaping() const                                                                                        \
    {                                                                                                                  \
        return (member).enabled();                                                                                     \
    }                                                                                                                  \
    void set_enable_shaping(bool enabled)                                                                              \
    {                                                                                                                  \
        if ((member).set_enabled(enabled))                                                                             \
            mark_dirty(DirtyFlags::render_state);                                                                      \
    }                                                                                                                  \
    float shaping_cone_angle() const                                                                                   \
    {                                                                                                                  \
        return (member).cone_angle();                                                                                  \
    }                                                                                                                  \
    void set_shaping_cone_angle(float angle)                                                                           \
    {                                                                                                                  \
        if ((member).set_cone_angle(angle))                                                                            \
            mark_dirty(DirtyFlags::render_state);                                                                      \
    }                                                                                                                  \
    float shaping_cone_softness() const                                                                                \
    {                                                                                                                  \
        return (member).cone_softness();                                                                               \
    }                                                                                                                  \
    void set_shaping_cone_softness(float softness)                                                                     \
    {                                                                                                                  \
        if ((member).set_cone_softness(softness))                                                                      \
            mark_dirty(DirtyFlags::render_state);                                                                      \
    }                                                                                                                  \
    float shaping_focus() const                                                                                        \
    {                                                                                                                  \
        return (member).focus();                                                                                       \
    }                                                                                                                  \
    void set_shaping_focus(float focus)                                                                                \
    {                                                                                                                  \
        if ((member).set_focus(focus))                                                                                 \
            mark_dirty(DirtyFlags::render_state);                                                                      \
    }

class FALCOR_API PointLight : public Light {
    FALCOR_SCENE_OBJECT(PointLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    PointLight();
    ~PointLight() override;

    FALCOR_LIGHT_SHAPING_MIXIN(m_shaping)

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        LightShaping::reflect<PointLight>(r);
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    LightShaping m_shaping;
};

class FALCOR_API SphereLight : public Light {
    FALCOR_SCENE_OBJECT(SphereLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    SphereLight();
    ~SphereLight() override;

    float radius() const { return m_radius; }
    void set_radius(float radius);

    bool enable_virtual_sphere_shrinking() const { return m_enable_virtual_sphere_shrinking; }
    void set_enable_virtual_sphere_shrinking(bool enable);

    FALCOR_LIGHT_SHAPING_MIXIN(m_shaping)

    /// World-space radius using the transform's volume-equivalent uniform scale.
    float world_radius() const;

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "radius",
                &SphereLight::radius,
                &SphereLight::set_radius,
                "Local-space radius.",
                reflection::default_value(0.5f),
                reflection::value_range_positive(),
                reflection::ui_label("Radius")
            )
            .def_prop_rw(
                "enable_virtual_sphere_shrinking",
                &SphereLight::enable_virtual_sphere_shrinking,
                &SphereLight::set_enable_virtual_sphere_shrinking,
                "Keep the sampled sphere smaller than the receiver distance while preserving emitted power.",
                reflection::default_value(false),
                reflection::ui_label("Enable Virtual Sphere Shrinking")
            );

        LightShaping::reflect<SphereLight>(r);
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float m_radius{0.5f};
    bool m_enable_virtual_sphere_shrinking{false};
    LightShaping m_shaping;
};

/// World-space circular approximation used to sample and trace a cylinder light.
struct CylinderLightWorldGeometry {
    float3 center_ws;
    float3 axis_ws;
    float3 radius_u_ws;
    float3 radius_v_ws;
    float half_length_ws;
    float radius_ws;
};

/// Finite cylinder light that emits from its open lateral surface.
class FALCOR_API CylinderLight : public Light {
    FALCOR_SCENE_OBJECT(CylinderLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    CylinderLight();
    ~CylinderLight() override;

    float radius() const { return m_radius; }
    void set_radius(float radius);

    float length() const { return m_length; }
    void set_length(float length);

    FALCOR_LIGHT_SHAPING_MIXIN(m_shaping)

    CylinderLightWorldGeometry world_geometry() const;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "radius",
                &CylinderLight::radius,
                &CylinderLight::set_radius,
                "Local-space radius.",
                reflection::default_value(0.5f),
                reflection::value_range_positive(),
                reflection::ui_label("Radius")
            )
            .def_prop_rw(
                "length",
                &CylinderLight::length,
                &CylinderLight::set_length,
                "Local-space length along the X axis.",
                reflection::default_value(1.f),
                reflection::value_range_positive(),
                reflection::ui_label("Length")
            );

        LightShaping::reflect<CylinderLight>(r);
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float m_radius{0.5f};
    float m_length{1.f};
    LightShaping m_shaping;
};

class FALCOR_API DiskLight : public Light {
    FALCOR_SCENE_OBJECT(DiskLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    DiskLight();
    ~DiskLight() override;

    float radius() const { return m_radius; }
    void set_radius(float radius);

    FALCOR_LIGHT_SHAPING_MIXIN(m_shaping)

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "radius",
                &DiskLight::radius,
                &DiskLight::set_radius,
                "Local-space radius.",
                reflection::default_value(0.5f),
                reflection::value_range_positive(),
                reflection::ui_label("Radius")
            );

        LightShaping::reflect<DiskLight>(r);
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float m_radius{0.5f};
    LightShaping m_shaping;
};

class FALCOR_API RectLight : public Light {
    FALCOR_SCENE_OBJECT(RectLight, Light)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    RectLight();
    ~RectLight() override;

    float width() const { return m_width; }
    void set_width(float width);

    float height() const { return m_height; }
    void set_height(float height);

    FALCOR_LIGHT_SHAPING_MIXIN(m_shaping)

    // Light interface

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "width",
                &RectLight::width,
                &RectLight::set_width,
                "Local-space width.",
                reflection::default_value(1.f),
                reflection::value_range_positive(),
                reflection::ui_label("Width")
            )
            .def_prop_rw(
                "height",
                &RectLight::height,
                &RectLight::set_height,
                "Local-space height.",
                reflection::default_value(1.f),
                reflection::value_range_positive(),
                reflection::ui_label("Height")
            );

        LightShaping::reflect<RectLight>(r);
    }

private:
    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float m_width{1.f};
    float m_height{1.f};
    LightShaping m_shaping;
};

#undef FALCOR_LIGHT_SHAPING_MIXIN

} // namespace falcor
