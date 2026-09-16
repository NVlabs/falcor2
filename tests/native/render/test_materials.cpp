// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/asset_resolver.h"
#include "falcor2/render/material/mdl/mdl_material.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/mtlx/codegen/codegen.h"
#include "falcor2/render/material/mtlx/mtlx_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/standard_specgloss_material.h"
#include "falcor2/render/material/unlit_material.h"
#include "falcor2/render/scene.h"

#include <sgl/device/device.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace falcor;

TEST_CASE("material header metadata codec")
{
    CHECK_EQ(sizeof(shared::MaterialHeader), 16);
    CHECK_EQ(sizeof(shared::MaterialData), 128);

    shared::MaterialHeader header = {};
    header.material_data = shared::detail::pack_material_data(
        shared::MaterialFlags::two_sided | shared::MaterialFlags::thin_walled,
        255
    );
    CHECK_EQ(
        shared::detail::get_material_flags(header),
        shared::MaterialFlags::two_sided | shared::MaterialFlags::thin_walled
    );
    CHECK_EQ(shared::detail::get_nested_priority(header), 255);
    CHECK_EQ(header.material_data & 0xffff0000u, 0u);
}

TEST_CASE_GPU("material nested priority properties")
{
    auto scene = Scene::create(ref(ctx.device));

    StandardMaterial* standard = scene->create_material<StandardMaterial>();
    CHECK_EQ(standard->nested_priority(), 0);
    standard->set_property<uint32_t>("nested_priority", 17);
    CHECK_EQ(standard->nested_priority(), 17);

    StandardSpecGlossMaterial* spec_gloss = scene->create_material<StandardSpecGlossMaterial>();
    spec_gloss->set_property<uint32_t>("nested_priority", 23);
    CHECK_EQ(spec_gloss->nested_priority(), 23);

    MaterialXMaterial* materialx = scene->create_material<MaterialXMaterial>();
    materialx->set_property<uint32_t>("nested_priority", 31);
    CHECK_EQ(materialx->nested_priority(), 31);
    CHECK_EQ(materialx->flags(), shared::MaterialFlags::two_sided);
}

namespace {

const char* LAYERING_TRAVERSAL_ORDER_PROPERTY = "materialx_layering_traversal_order";

class RequiredModuleTestMaterial final : public Material {
    FALCOR_SCENE_OBJECT(RequiredModuleTestMaterial, Material)
public:
    RequiredModuleTestMaterial() { set_slang_type_name("RequiredModuleTestMaterialData"); }

    void set_required_module(ref<sgl::SlangModule> module) { m_required_module = std::move(module); }
    ref<sgl::SlangModule> required_module() const override { return m_required_module; }

private:
    ref<sgl::SlangModule> m_required_module;
};

std::string upgrade_test_document_version(std::string source)
{
    const std::string old_version = "version=\"1.38\"";
    const std::string new_version = "version=\"1.39\"";
    size_t pos = 0;
    while ((pos = source.find(old_version, pos)) != std::string::npos) {
        source.replace(pos, old_version.size(), new_version);
        pos += new_version.size();
    }
    return source;
}

mtlx::CodeGenDesc make_materialx_codegen_desc(const std::string& shader)
{
    mtlx::CodeGenDesc desc;
    desc.document = upgrade_test_document_version(shader);
    desc.node_name = "Tiled_Brass";
    desc.positionfree_layering = false;
    desc.transmissive_bsdfs.clear();
    desc.make_editable = false;
    return desc;
}

std::vector<std::string> layering_traversal_order(const mtlx::CodeGenResult& result)
{
    return result.codegen_metadata.get_list<std::string>(LAYERING_TRAVERSAL_ORDER_PROPERTY);
}

std::string join_lines(const std::vector<std::string>& lines)
{
    std::string result;
    for (const std::string& line : lines) {
        result += line;
        result += "\n";
    }
    return result;
}

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool contains(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool contains(std::string_view source, std::string_view needle)
{
    return source.find(needle) != std::string_view::npos;
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

struct PreviewOutputCase {
    std::string type;
    std::string value;
};

std::vector<PreviewOutputCase> preview_output_cases()
{
    return {
        {"float", "0.5"},
        {"vector2", "0.2, 0.4"},
        {"vector3", "0.2, 0.4, 0.8"},
        {"vector4", "0.2, 0.4, 0.8, 0.5"},
        {"color3", "0.2, 0.4, 0.8"},
        {"color4", "0.2, 0.4, 0.8, 0.5"},
    };
}

std::string preview_output_node_graph_name(const PreviewOutputCase& preview_case)
{
    return "NG_" + preview_case.type;
}

std::string preview_output_name(const PreviewOutputCase& preview_case)
{
    return preview_output_node_graph_name(preview_case) + "/out";
}

std::string preview_output_node_graph(const PreviewOutputCase& preview_case)
{
    const std::string graph_name = preview_output_node_graph_name(preview_case);
    return "  <nodegraph name=\"" + graph_name + "\">\n" + "    <constant name=\"value\" type=\"" + preview_case.type
        + "\">\n" + "      <input name=\"value\" type=\"" + preview_case.type + "\" value=\"" + preview_case.value
        + "\" />\n" + "    </constant>\n" + "    <output name=\"out\" type=\"" + preview_case.type
        + "\" nodename=\"value\" />\n" + "  </nodegraph>\n";
}

std::string preview_output_shader(const PreviewOutputCase& preview_case)
{
    return "<?xml version=\"1.0\"?>\n"
           "<materialx version=\"1.38\" colorspace=\"lin_rec709\">\n"
        + preview_output_node_graph(preview_case) + "</materialx>\n";
}

std::string transformmatrix_vector4_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_transformmatrix_vector4">
    <transformmatrix name="transform1" type="vector4">
      <input name="in" type="vector4" value="0.2, 0.4, 0.8, 1.0" />
      <input name="mat" type="matrix44" value="1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0" />
    </transformmatrix>
    <output name="out" type="vector4" nodename="transform1" />
  </nodegraph>
</materialx>
)";
}

std::string image_vector2_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_image_vector2">
    <image name="image1" type="vector2">
      <input name="file" type="filename" value="resources/Images/grid.png" />
      <input name="default" type="vector2" value="0.25, 0.75" />
    </image>
    <output name="out" type="vector2" nodename="image1" />
  </nodegraph>
</materialx>
)";
}

std::string color4_to_float_channel_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_color4_to_float_g">
    <constant name="value" type="color4">
      <input name="value" type="color4" value="0.0, 0.5, 0.75, 1.0" />
    </constant>
    <output name="out" type="float" nodename="value" channels="g" />
  </nodegraph>
</materialx>
)";
}

std::string vector2_to_vector3_channel_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_vector2_to_vector3_xy0">
    <constant name="value" type="vector2">
      <input name="value" type="vector2" value="0.25, 0.75" />
    </constant>
    <output name="out" type="vector3" nodename="value" channels="xy0" />
  </nodegraph>
</materialx>
)";
}

std::string difference_color3_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_difference_color3">
    <difference name="difference1" type="color3">
      <input name="fg" type="color3" value="0.9, 0.2, 0.1" />
      <input name="bg" type="color3" value="0.1, 0.4, 0.8" />
      <input name="mix" type="float" value="0.5" />
    </difference>
    <output name="out" type="color3" nodename="difference1" />
  </nodegraph>
</materialx>
)";
}

std::string blur_vector3_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_blur_vector3">
    <blur name="blur1" type="vector3">
      <input name="in" type="vector3" value="0.2, 0.4, 0.8" />
      <input name="size" type="float" value="0.1" />
    </blur>
    <output name="out" type="vector3" nodename="blur1" />
  </nodegraph>
</materialx>
)";
}

std::string worleynoise3d_vector2_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_worleynoise3d_vector2">
    <worleynoise3d name="worley1" type="vector2">
      <input name="position" type="vector3" value="0.2, 0.4, 0.8" />
      <input name="jitter" type="float" value="0.5" />
    </worleynoise3d>
    <output name="out" type="vector2" nodename="worley1" />
  </nodegraph>
</materialx>
)";
}

std::string mix_edf_surface_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_mix_edf_surface">
    <uniform_edf name="edf1" type="EDF">
      <input name="color" type="color3" value="0.1, 0.8, 0.2" />
    </uniform_edf>
    <uniform_edf name="edf2" type="EDF">
      <input name="color" type="color3" value="0.8, 0.1, 0.4" />
    </uniform_edf>
    <mix name="mix1" type="EDF">
      <input name="fg" type="EDF" nodename="edf1" />
      <input name="bg" type="EDF" nodename="edf2" />
      <input name="mix" type="float" value="0.5" />
    </mix>
    <surface name="surface1" type="surfaceshader">
      <input name="edf" type="EDF" nodename="mix1" />
    </surface>
    <output name="out" type="surfaceshader" nodename="surface1" />
  </nodegraph>
</materialx>
)";
}

std::string mtlx_layered_property_update_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="base" type="BSDF">
    <input name="color" type="color3" value="0.1, 0.2, 0.3" />
  </oren_nayar_diffuse_bsdf>
  <oren_nayar_diffuse_bsdf name="top" type="BSDF">
    <input name="color" type="color3" value="0.7, 0.6, 0.5" />
  </oren_nayar_diffuse_bsdf>
  <layer name="layer1" type="BSDF">
    <input name="top" type="BSDF" nodename="top" />
    <input name="base" type="BSDF" nodename="base" />
  </layer>
  <surface name="surface1" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="layer1" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>)";
}

std::string rootless_srgb_material_shader()
{
    return R"(<materialx version="1.39">
  <image name="albedo" type="color3">
    <input name="file" type="filename" value="albedo.png" colorspace="srgb_texture" />
  </image>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
    <input name="color" type="color3" nodename="albedo" />
  </oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string retroreflective_microfacet_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <conductor_bsdf name="conductor" type="BSDF">
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="thinfilm_thickness" type="float" value="100.0" />
  </conductor_bsdf>
  <dielectric_bsdf name="dielectric" type="BSDF">
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="scatter_mode" type="string" value="RT" />
  </dielectric_bsdf>
  <generalized_schlick_bsdf name="generalized_schlick" type="BSDF">
    <input name="color82" type="color3" value="0.8, 0.7, 0.6" />
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="scatter_mode" type="string" value="RT" />
  </generalized_schlick_bsdf>
  <add name="microfacet_add" type="BSDF">
    <input name="in1" type="BSDF" nodename="conductor" />
    <input name="in2" type="BSDF" nodename="dielectric" />
  </add>
  <add name="all_bsdfs" type="BSDF">
    <input name="in1" type="BSDF" nodename="microfacet_add" />
    <input name="in2" type="BSDF" nodename="generalized_schlick" />
  </add>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="all_bsdfs" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

enum class MaterialXOpacityCase {
    opaque,
    constant,
    connected,
    mixed_out_of_range,
    luminance_custom_coefficients,
    luminance_nonuniform_color,
    luminance_overflow,
    custom_non_float_opacity,
    standard_surface_white,
    standard_surface_red,
    open_pbr_constant,
    open_pbr_connected,
};

std::string materialx_opacity_shader(MaterialXOpacityCase opacity_case)
{
    if (opacity_case == MaterialXOpacityCase::standard_surface_white
        || opacity_case == MaterialXOpacityCase::standard_surface_red) {
        const char* opacity = opacity_case == MaterialXOpacityCase::standard_surface_white ? "1, 1, 1" : "1, 0, 0";
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader">
    <input name="opacity" type="color3" value="{}" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
            opacity
        );
    }

    if (opacity_case == MaterialXOpacityCase::open_pbr_constant
        || opacity_case == MaterialXOpacityCase::open_pbr_connected) {
        const bool connected = opacity_case == MaterialXOpacityCase::open_pbr_connected;
        const std::string opacity_node = connected ? R"(  <constant name="opacity_value" type="float">
    <input name="value" type="float" value="0.25" />
  </constant>
)"
                                                   : "";
        const std::string opacity_input = connected
            ? R"(    <input name="geometry_opacity" type="float" nodename="opacity_value" />
)"
            : R"(    <input name="geometry_opacity" type="float" value="0.25" />
)";
        return R"(<materialx version="1.39" colorspace="lin_rec709">
)" + opacity_node
            + R"(  <open_pbr_surface name="surface" type="surfaceshader">
)" + opacity_input
            + R"(  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    }

    if (opacity_case == MaterialXOpacityCase::mixed_out_of_range) {
        return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="background" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" value="0.2" />
  </surface>
  <surface name="foreground" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" value="2.0" />
  </surface>
  <mix name="surface_mix" type="surfaceshader">
    <input name="bg" type="surfaceshader" nodename="background" />
    <input name="fg" type="surfaceshader" nodename="foreground" />
    <input name="mix" type="float" value="0.25" />
  </mix>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface_mix" />
  </surfacematerial>
</materialx>)";
    }

    if (opacity_case == MaterialXOpacityCase::luminance_custom_coefficients
        || opacity_case == MaterialXOpacityCase::luminance_nonuniform_color
        || opacity_case == MaterialXOpacityCase::luminance_overflow) {
        const char* color = "1.0, 1.0, 1.0";
        const char* coefficients = "0.0, 0.0, 0.0";
        if (opacity_case == MaterialXOpacityCase::luminance_nonuniform_color) {
            color = "1.0, 0.5, 0.25";
            coefficients = "0.2, 0.3, 0.5";
        } else if (opacity_case == MaterialXOpacityCase::luminance_overflow) {
            color = "3.0e38, 1.0, 1.0";
            coefficients = "2.0, 0.0, 0.0";
        }
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <constant name="opacity_color" type="color3">
    <input name="value" type="color3" value="{}" />
  </constant>
  <luminance name="opacity_luminance" type="color3">
    <input name="in" type="color3" nodename="opacity_color" />
    <input name="lumacoeffs" type="color3" value="{}" />
  </luminance>
  <extract name="opacity" type="float">
    <input name="in" type="color3" nodename="opacity_luminance" />
    <input name="index" type="integer" value="0" />
  </extract>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" nodename="opacity" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
            color,
            coefficients
        );
    }

    if (opacity_case == MaterialXOpacityCase::custom_non_float_opacity) {
        return R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_custom_surface" node="custom_surface">
    <input name="opacity" type="color3" value="1.0, 1.0, 1.0" />
    <output name="out" type="surfaceshader" />
  </nodedef>
  <implementation name="IM_custom_surface_genslangpt" nodedef="ND_custom_surface" sourcecode="{}" target="genslangpt" />
  <custom_surface name="surface" type="surfaceshader" />
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    }

    std::string opacity_node;
    std::string opacity_input;
    if (opacity_case == MaterialXOpacityCase::constant)
        opacity_input = R"(    <input name="opacity" type="float" value="0.25" />
)";
    else if (opacity_case == MaterialXOpacityCase::connected) {
        opacity_node = R"(  <constant name="opacity_value" type="float">
    <input name="value" type="float" value="0.25" />
  </constant>
)";
        opacity_input = R"(    <input name="opacity" type="float" nodename="opacity_value" />
)";
    }

    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
)" + opacity_node
        + R"(  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
)" + opacity_input
        + R"(  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::vector<std::string>
normalize_layering_traversal_order(const std::vector<std::string>& order, bool include_node_events)
{
    std::vector<std::string> result;
    for (const std::string& event : order) {
        if (event == "add:Tiled_Brass")
            continue;

        if (!include_node_events && starts_with(event, "add:"))
            continue;

        result.push_back(event);
    }
    return result;
}

} // namespace

TEST_SUITE_BEGIN("scene_materials");

TEST_CASE("material opacity data round-trips through packed header")
{
    const auto flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
    const float threshold = 0.5f;
    const float factor = 0.002f;
    shared::MaterialHeader header = {};
    header.opacity_data = shared::detail::pack_opacity_data(flags, 3, threshold, factor);

    CHECK_EQ(shared::detail::get_opacity_flags(header), flags);
    CHECK_EQ(shared::detail::get_opacity_texture_channel(header), 3);
    CHECK_EQ(shared::detail::get_opacity_threshold(header), 128.f / 255.f);
    CHECK(std::abs(shared::detail::get_opacity_factor(header) - factor) <= 0.5f / 65535.f);
}

TEST_CASE("scene materialx classifies root opacity and emits traversal evaluation")
{
    using OpacityKind = mtlx::CodeGenResult::OpacityResult::Kind;

    struct OpacityVariant {
        MaterialXOpacityCase opacity_case;
        bool editable;
        OpacityKind expected_kind;
        float expected_constant;
    };

    const std::array opacity_variants = {
        OpacityVariant{MaterialXOpacityCase::opaque, false, OpacityKind::opaque, 1.f},
        OpacityVariant{MaterialXOpacityCase::constant, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::connected, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::mixed_out_of_range, false, OpacityKind::constant, 0.4f},
        OpacityVariant{MaterialXOpacityCase::luminance_custom_coefficients, false, OpacityKind::constant, 0.f},
        OpacityVariant{MaterialXOpacityCase::luminance_nonuniform_color, false, OpacityKind::constant, 0.475f},
        OpacityVariant{MaterialXOpacityCase::luminance_overflow, false, OpacityKind::dynamic, 1.f},
        OpacityVariant{MaterialXOpacityCase::custom_non_float_opacity, false, OpacityKind::dynamic, 1.f},
        OpacityVariant{MaterialXOpacityCase::opaque, true, OpacityKind::dynamic, 1.f},
        OpacityVariant{MaterialXOpacityCase::standard_surface_white, false, OpacityKind::opaque, 1.f},
        OpacityVariant{MaterialXOpacityCase::standard_surface_red, false, OpacityKind::constant, 0.272229f},
        OpacityVariant{MaterialXOpacityCase::open_pbr_constant, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::open_pbr_connected, false, OpacityKind::constant, 0.25f},
    };
    const std::array layering_modes = {
        mtlx::LayeringMode::closure_tree,
        mtlx::LayeringMode::bsdf_mix,
    };

    for (mtlx::LayeringMode layering_mode : layering_modes) {
        for (const OpacityVariant& variant : opacity_variants) {
            CAPTURE(static_cast<int>(layering_mode));
            CAPTURE(static_cast<int>(variant.opacity_case));
            mtlx::CodeGenDesc desc;
            desc.document = materialx_opacity_shader(variant.opacity_case);
            desc.node_name = "M";
            desc.make_editable = variant.editable;
            desc.layering_mode = layering_mode;

            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->opacity.kind, variant.expected_kind);
            CHECK_EQ(result->opacity.constant, doctest::Approx(variant.expected_constant));
            const bool evaluates_opacity = variant.expected_kind == OpacityKind::dynamic;
            CHECK(contains(result->module_source, "eval_opacity<TLodSampler"));
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_EVALUATE_OPACITY ") + (evaluates_opacity ? "1" : "0")
            ));
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_OPACITY_IS_OPAQUE ")
                    + (variant.expected_kind == OpacityKind::opaque ? "1" : "0")
            ));
            const bool conservatively_emissive = variant.opacity_case == MaterialXOpacityCase::standard_surface_white
                || variant.opacity_case == MaterialXOpacityCase::standard_surface_red
                || variant.opacity_case == MaterialXOpacityCase::open_pbr_constant
                || variant.opacity_case == MaterialXOpacityCase::open_pbr_connected;
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_HAS_EMISSION ") + (conservatively_emissive ? "1" : "0")
            ));
            CHECK(contains(result->module_source, "out float3 opacity_weighted_emission"));
            CHECK(contains(result->module_source, "eval_emission<TLodSampler"));
            CHECK(contains(result->module_source, "data_buffer.load<"));
            CHECK_FALSE(contains(result->module_source, "data_buffer.load_aligned<"));
            CHECK_FALSE(contains(result->module_source, "MxSimpleBTDF()"));
            CHECK_FALSE(contains(result->module_source, "synthetic_opacity_mix"));
        }
    }
}

TEST_CASE("scene materialx selects editable parameters by generated name")
{
    auto make_desc = []
    {
        mtlx::CodeGenDesc desc;
        desc.document = materialx_opacity_shader(MaterialXOpacityCase::constant);
        desc.node_name = "M";
        return desc;
    };

    auto restricted = mtlx::CodeGen::generate(make_desc());
    REQUIRE(restricted);
    REQUIRE(restricted->all_material_params.m_params.size() >= 2);
    CHECK_EQ(restricted->all_material_params.m_editable_count, 0);

    CHECK_EQ(restricted->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);

    mtlx::CodeGenDesc unrelated_desc = make_desc();
    unrelated_desc.editable_param_names = {"diffuse_color"};
    auto unrelated = mtlx::CodeGen::generate(unrelated_desc);
    REQUIRE(unrelated);
    CHECK_EQ(unrelated->all_material_params.m_editable_count, 1);
    CHECK_EQ(unrelated->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);
    CHECK(unrelated->codegen_metadata.get<bool>("closure_pruning_effective_enabled"));

    mtlx::CodeGenDesc whitelist_desc = make_desc();
    whitelist_desc.editable_param_names = {"surface_opacity"};
    auto whitelist = mtlx::CodeGen::generate(whitelist_desc);
    REQUIRE(whitelist);
    CHECK_EQ(whitelist->all_material_params.m_editable_count, 1);
    for (const mtlx::MxParamInfo& param : whitelist->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.param_name == "surface_opacity");
    CHECK_EQ(whitelist->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::dynamic);
    CHECK(whitelist->codegen_metadata.get<bool>("closure_pruning_effective_enabled"));

    mtlx::CodeGenDesc complete_desc = make_desc();
    complete_desc.make_editable = true;
    auto complete = mtlx::CodeGen::generate(complete_desc);
    REQUIRE(complete);
    for (const mtlx::MxParamInfo& param : complete->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());
    CHECK_EQ(complete->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::dynamic);
}

TEST_CASE("scene materialx class compilation honors explicit editable parameter names")
{
    mtlx::CodeGenDesc desc;
    CHECK_FALSE(desc.is_editable_param("surface_base"));

    desc.class_compilation = true;
    CHECK(desc.is_editable_param("surface_base"));

    desc.editable_param_names = {"surface_base"};
    CHECK(desc.is_editable_param("surface_base"));
    CHECK_FALSE(desc.is_editable_param("surface_base_color"));

    desc.make_editable = true;
    CHECK(desc.is_editable_param("surface_base_color"));
}

TEST_CASE_GPU("scene material system deduplicates logical modules")
{
    const std::string module_name = "required_module_dedup_test";
    const std::string module_source = R"(
import falcor2.render;

public struct RequiredModuleTestMaterialData : IMaterial
{
    typedef InvalidMaterialInstance MaterialInstance;
    public MaterialHeader header;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        InvalidMaterial material = {};
        material.header = header;
        return material.setup_material_instance(si, lod_sampler, hints);
    }
};
)";
    ref<sgl::SlangModule> first_module = ctx.device->load_module_from_source(module_name, module_source);
    ref<sgl::SlangModule> second_module = ctx.device->load_module_from_source(module_name, module_source);
    REQUIRE(first_module);
    REQUIRE(second_module);
    REQUIRE_EQ(first_module->slang_module(), second_module->slang_module());
    REQUIRE_EQ(first_module->slang_component_type(), second_module->slang_component_type());
    REQUIRE_EQ(first_module->name(), second_module->name());

    auto scene = Scene::create(ref(ctx.device));
    auto* first = scene->create_material<RequiredModuleTestMaterial>();
    auto* second = scene->create_material<RequiredModuleTestMaterial>();
    first->set_required_module(first_module);
    second->set_required_module(second_module);
    scene->update();

    const auto requirements = scene->requirements();
    const size_t shared_module_count = std::count_if(
        requirements.modules.begin(),
        requirements.modules.end(),
        [&](const ref<sgl::SlangModule>& module)
        {
            return module->name() == module_name;
        }
    );
    CHECK_EQ(shared_module_count, 1);
}

TEST_CASE("scene materialx class compilation canonicalizes graph identity and instance data")
{
    auto make_document = [](std::string_view graph_name,
                            std::string_view image_name,
                            std::string_view inner_node_name,
                            std::string_view instance_name,
                            std::string_view surface_name,
                            std::string_view material_name,
                            std::string_view roughness,
                            std::string_view filename,
                            std::string_view tint,
                            bool reverse_node_declarations)
    {
        const std::string image = fmt::format(
            R"(    <image name="{}" type="color3">
      <input name="file" type="filename" interfacename="file" />
    </image>
)",
            image_name
        );
        const std::string multiply = fmt::format(
            R"(    <multiply name="{}" type="color3">
      <input name="in1" type="color3" nodename="{}" />
      <input name="in2" type="color3" interfacename="tint" />
    </multiply>
)",
            inner_node_name,
            image_name
        );
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_nested_color" node="nested_color">
    <input name="tint" type="color3" value="0.8, 0.8, 0.8" />
    <input name="file" type="filename" value="" />
    <output name="out" type="color3" />
  </nodedef>
  <nodegraph name="{}" nodedef="ND_nested_color">
{}{}    <output name="out" type="color3" nodename="{}" />
  </nodegraph>
  <nested_color name="{}" type="color3">
    <input name="tint" type="color3" value="{}" />
    <input name="file" type="filename" value="{}" />
  </nested_color>
  <standard_surface name="{}" type="surfaceshader">
    <input name="base_color" type="color3" nodename="{}" />
    <input name="specular_roughness" type="float" value="{}" />
  </standard_surface>
  <surfacematerial name="{}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="{}" />
  </surfacematerial>
</materialx>)",
            graph_name,
            reverse_node_declarations ? multiply : image,
            reverse_node_declarations ? image : multiply,
            inner_node_name,
            instance_name,
            tint,
            filename,
            surface_name,
            instance_name,
            roughness,
            material_name,
            surface_name
        );
    };

    auto generate_class = [&](std::string_view graph_name,
                              std::string_view image_name,
                              std::string_view inner_node_name,
                              std::string_view instance_name,
                              std::string_view surface_name,
                              std::string_view material_name,
                              std::string_view roughness,
                              std::string_view filename,
                              std::string_view tint,
                              bool reverse_node_declarations)
    {
        mtlx::CodeGenDesc desc;
        desc.document = make_document(
            graph_name,
            image_name,
            inner_node_name,
            instance_name,
            surface_name,
            material_name,
            roughness,
            filename,
            tint,
            reverse_node_declarations
        );
        desc.node_name = material_name;
        desc.class_compilation = true;
        return mtlx::CodeGen::generate(desc);
    };

    auto first = generate_class(
        "nested_graph_a",
        "base_image",
        "multiply_a",
        "instance_a",
        "surface_a",
        "material_a",
        "0.2",
        "first.png",
        "0.1, 0.2, 0.3",
        false
    );
    auto second = generate_class(
        "nested_graph_b",
        "renamed_texture",
        "multiply_b",
        "instance_b",
        "surface_b",
        "material_b",
        "0.8",
        "second.png",
        "0.7, 0.8, 0.9",
        true
    );
    REQUIRE(first);
    REQUIRE(second);
    CHECK_EQ(first->module_source, second->module_source);
    CHECK_EQ(first->module_name, second->module_name);
    CHECK_EQ(first->material_name, second->material_name);
    CHECK_EQ(first->material_data_name, second->material_data_name);
    CHECK_EQ(first->data_struct_size, second->data_struct_size);
    for (const mtlx::MxParamInfo& param : first->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());
    for (const mtlx::MxParamInfo& param : second->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());

    mtlx::CodeGenDesc different_topology_desc;
    different_topology_desc.document = std::string(R"(<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" value="0.5, 0.5, 0.5" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)");
    different_topology_desc.node_name = "M";
    different_topology_desc.class_compilation = true;
    auto different_topology = mtlx::CodeGen::generate(different_topology_desc);
    REQUIRE(different_topology);
    CHECK_NE(first->module_name, different_topology->module_name);

    std::vector<const mtlx::MxParamInfo*> first_params;
    std::vector<const mtlx::MxParamInfo*> second_params;
    for (const mtlx::MxParamInfo& param : first->all_material_params.m_params)
        first_params.push_back(&param);
    for (const mtlx::MxParamInfo& param : second->all_material_params.m_params)
        second_params.push_back(&param);
    const auto binding_less = [](const mtlx::MxParamInfo* lhs, const mtlx::MxParamInfo* rhs)
    {
        return lhs->binding_name < rhs->binding_name;
    };
    std::sort(first_params.begin(), first_params.end(), binding_less);
    std::sort(second_params.begin(), second_params.end(), binding_less);
    REQUIRE_EQ(first_params.size(), second_params.size());

    bool found_distinct_property_names = false;
    bool found_distinct_float_values = false;
    bool found_distinct_texture_paths = false;
    for (size_t i = 0; i < first_params.size(); ++i) {
        CHECK_EQ(first_params[i]->binding_name, second_params[i]->binding_name);
        found_distinct_property_names |= first_params[i]->param_name != second_params[i]->param_name;
        if (const float* first_value = first_params[i]->extract_value<float>()) {
            if (const float* second_value = second_params[i]->extract_value<float>())
                found_distinct_float_values |= *first_value != *second_value;
        }
        if (const auto* first_texture = first_params[i]->extract_value<mtlx::params::TextureInfo>()) {
            if (const auto* second_texture = second_params[i]->extract_value<mtlx::params::TextureInfo>())
                found_distinct_texture_paths |= first_texture->file_name != second_texture->file_name;
        }
    }
    CHECK(found_distinct_property_names);
    CHECK(found_distinct_float_values);
    CHECK(found_distinct_texture_paths);

    auto generate_editable = [&](std::string_view roughness)
    {
        mtlx::CodeGenDesc desc;
        desc.document = make_document(
            "nested_graph",
            "base_image",
            "multiply",
            "instance",
            "surface",
            "material",
            roughness,
            "texture.png",
            "0.5, 0.5, 0.5",
            false
        );
        desc.node_name = "material";
        desc.editable_param_names = {"surface_specular_roughness"};
        return mtlx::CodeGen::generate(desc);
    };
    auto first_editable = generate_editable("0.2");
    auto second_editable = generate_editable("0.8");
    REQUIRE(first_editable);
    REQUIRE(second_editable);
    CHECK_EQ(first_editable->module_source, second_editable->module_source);
    CHECK_EQ(first_editable->module_name, second_editable->module_name);
    for (const mtlx::MxParamInfo& param : first_editable->all_material_params.m_params)
        CHECK_EQ(param.interface.name, param.param_name);
}

TEST_CASE("scene materialx emits opacity and emission specialization defines")
{
    const std::string shader = R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <uniform_edf name="emission" type="EDF">
    <input name="color" type="color3" value="1, 0, 0" />
  </uniform_edf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="edf" type="EDF" nodename="emission" />
    <input name="opacity" type="float" value="0.25" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    const std::array layering_modes = {
        mtlx::LayeringMode::closure_tree,
        mtlx::LayeringMode::bsdf_mix,
    };

    for (mtlx::LayeringMode layering_mode : layering_modes) {
        mtlx::CodeGenDesc desc;
        desc.document = shader;
        desc.node_name = "M";
        desc.layering_mode = layering_mode;

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);
        CHECK(contains(result->module_source, "#define MATERIALX_EVALUATE_OPACITY 0"));
        CHECK(contains(result->module_source, "#define MATERIALX_OPACITY_IS_OPAQUE 0"));
        CHECK(contains(result->module_source, "#define MATERIALX_STATIC_OPACITY 0.250000"));
        CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
        CHECK(contains(result->module_source, "eval_opacity<TLodSampler"));
        CHECK(contains(result->module_source, "out float3 opacity_weighted_emission"));
        CHECK(contains(result->module_source, "let graph_opacity = MATERIALX_STATIC_OPACITY"));
        CHECK(contains(result->module_source, "eval_emission<TLodSampler"));
        CHECK(contains(result->module_source, "public override bool is_emissive()"));
    }
}

TEST_CASE_GPU("scene materialx publishes root opacity classification")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));
    auto classify = [&](MaterialXOpacityCase opacity_case, bool editable = false)
    {
        Properties props;
        props.set("mtlx_buffer", materialx_opacity_shader(opacity_case));
        props.set("mtlx_node_name", std::string("M"));
        if (editable)
            props.set("mtlx_editable_params", std::string("*"));
        MaterialXMaterial* material = scene->create_material<MaterialXMaterial>(props);
        return material->opacity_desc();
    };

    CHECK_EQ(classify(MaterialXOpacityCase::opaque).flags, shared::OpacityFlags::none);

    const Material::OpacityDesc constant = classify(MaterialXOpacityCase::constant);
    CHECK_EQ(constant.flags, shared::OpacityFlags::enabled);
    CHECK_EQ(constant.factor, doctest::Approx(0.25f));

    const auto evaluated_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
    const Material::OpacityDesc connected = classify(MaterialXOpacityCase::connected);
    CHECK_EQ(connected.flags, shared::OpacityFlags::enabled);
    CHECK_EQ(connected.factor, doctest::Approx(0.25f));
    CHECK_EQ(classify(MaterialXOpacityCase::opaque, true).flags, evaluated_flags);
    CHECK_EQ(classify(MaterialXOpacityCase::standard_surface_white).flags, shared::OpacityFlags::none);
    const Material::OpacityDesc standard_surface_red = classify(MaterialXOpacityCase::standard_surface_red);
    CHECK_EQ(standard_surface_red.flags, shared::OpacityFlags::enabled);
    CHECK_EQ(standard_surface_red.factor, doctest::Approx(0.272229f));
}

TEST_CASE("scene materialx 1.39 successor variants compile without 1.38")
{
    struct Variant {
        std::string name;
        std::string shader;
        bool expects_layering = false;
    };

    const std::vector<Variant> variants = {
        {
            "standard_surface",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <standard_surface name="SR_brass1" type="surfaceshader">
    <input name="base" type="float" value="1" />
    <input name="base_color" type="color3" value="0.8, 0.6, 0.25" />
    <input name="specular" type="float" value="0" />
    <input name="metalness" type="float" value="1" />
    <input name="coat" type="float" value="1" />
  </standard_surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_brass1" />
  </surfacematerial>
</materialx>
)",
            false,
        },
        {
            "multioutput_layer",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodedef name="ND_double_diffuse" node="double_diffuse">
    <input name="color1" type="color3" value="0.8, 0.8, 0.8" />
    <output name="out_bsdf1" type="BSDF"/>
    <output name="out_bsdf2" type="BSDF"/>
  </nodedef>
  <nodegraph name="IM_double_diffuse" nodedef="ND_double_diffuse">
    <oren_nayar_diffuse_bsdf name="diffuse1_bsdf" type="BSDF">
        <input name="color" type="color3" interfacename="color1"/>
    </oren_nayar_diffuse_bsdf>
    <oren_nayar_diffuse_bsdf name="diffuse2_bsdf" type="BSDF" />
    <output name="out_bsdf1" type="BSDF" nodename="diffuse1_bsdf" />
    <output name="out_bsdf2" type="BSDF" nodename="diffuse2_bsdf" />
  </nodegraph>
  <double_diffuse name="NG_diffuse1" type="multioutput" />
  <double_diffuse name="NG_diffuse2" type="multioutput" />
  <layer name="layer1" type="BSDF">
    <input name="top" type="BSDF" nodename="NG_diffuse1" output="out_bsdf1"/>
    <input name="base" type="BSDF" nodename="NG_diffuse2" output="out_bsdf2"/>
  </layer>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="layer1" />
  </surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>
)",
            true,
        },
        {
            "interface_layer",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodedef name="ND_interface_layer" node="interface_layer" nodegroup="pbr">
    <input name="top" type="BSDF" />
    <input name="base" type="BSDF" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="IM_interface_layer" nodedef="ND_interface_layer">
    <layer name="layer1" type="BSDF">
      <input name="top" type="BSDF" interfacename="top" />
      <input name="base" type="BSDF" interfacename="base" />
    </layer>
    <output name="out" type="BSDF" nodename="layer1" />
  </nodegraph>
  <oren_nayar_diffuse_bsdf name="diffuse1" type="BSDF" />
  <oren_nayar_diffuse_bsdf name="diffuse2" type="BSDF" />
  <interface_layer name="wrapped_layer" type="BSDF">
    <input name="top" type="BSDF" nodename="diffuse1" />
    <input name="base" type="BSDF" nodename="diffuse2" />
  </interface_layer>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="wrapped_layer" />
  </surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>
)",
            true,
        },
    };

    for (const Variant& variant : variants) {
        INFO("MaterialX 1.39 successor variant " << variant.name);
        auto result = mtlx::CodeGen::generate(make_materialx_codegen_desc(variant.shader));
        REQUIRE(result);
        CHECK(starts_with(result->module_name, "mtlx_"));
        CHECK(contains(result->module_source, "namespace mtlx"));
        CHECK_FALSE(contains(result->module_source, "MaterialX138"));
        if (variant.expects_layering)
            CHECK_FALSE(normalize_layering_traversal_order(layering_traversal_order(*result), true).empty());
    }
}

TEST_CASE("scene materialx targets Falcor linear Rec.709 for rootless documents")
{
    mtlx::CodeGenDesc desc;
    desc.document = rootless_srgb_material_shader();
    desc.node_name = "M";

    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(contains(result->module_source, "NG_srgb_texture_to_lin_rec709_color3(albedo_out, albedo_out_cm_out)"));
}

TEST_CASE_GPU("scene materialx retroreflective microfacet nodes compile")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("mtlx_buffer", retroreflective_microfacet_shader());
    props.set("mtlx_node_name", std::string("M"));

    scene->create_material("MaterialXMaterial", props);
    scene->update();
}

TEST_CASE_GPU("scene materialx defaults to 1.39 geomprop ID provider codegen")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));
    const std::filesystem::path shader_path = testing::get_case_temp_directory() / "materialx139_geomprop_ids.slang";

    Properties props;
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39">
  <geompropvalue name="custom_stream" type="vector3">
    <input name="geomprop" type="string" value="custom_stream" />
  </geompropvalue>
  <output name="custom_stream_out" type="vector3" nodename="custom_stream" />
</materialx>)"));
    props.set("mtlx_node_name", std::string("custom_stream_out"));
    props.set_list("mtlx_geomprop_names", std::vector<std::string>{"custom_stream"});
    props.set_list("mtlx_geomprop_ids", std::vector<int64_t>{77});
    props.set("debug_write_shader_path", shader_path.string());

    scene->create_material("MaterialXMaterial", props);
    scene->update();

    std::ifstream shader_file(shader_path);
    std::stringstream shader_buffer;
    shader_buffer << shader_file.rdbuf();
    const std::string shader_source = shader_buffer.str();
    CHECK(contains(shader_source, "static const uint mx_geomprop_id_custom_stream = 77u"));
    CHECK(contains(shader_source, "GeomPropProvider::get_float3(si, mx_geomprop_id_custom_stream)"));
}

TEST_CASE_GPU("scene materialx 1.39 OpenPBR compatibility property is removed")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));
    const std::filesystem::path shader_path = testing::get_case_temp_directory() / "materialx139_openpbr_compat.slang";

    Properties props;
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39" colorspace="lin_rec709">
  <open_pbr_surface name="openpbr1" type="surfaceshader">
    <input name="base_color" type="color3" value="0.1, 0.2, 0.3" />
    <input name="thin_film_weight" type="float" value="1.0" />
    <input name="geometry_thin_walled" type="boolean" value="true" />
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="openpbr1" />
  </surfacematerial>
</materialx>)"));
    props.set("mtlx_node_name", std::string("M"));
    props.set("debug_write_shader_path", shader_path.string());

    scene->create_material("MaterialXMaterial", props);
    scene->update();

    std::ifstream shader_file(shader_path);
    std::stringstream shader_buffer;
    shader_buffer << shader_file.rdbuf();
    const std::string shader_source = shader_buffer.str();
    CHECK_FALSE(contains(shader_source, "OpenPBRMaterialInstance openpbr_instance"));
}

TEST_CASE_GPU("scene materialx 1.39 static closure pruning property is removed")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39">
  <standard_surface name="surface1" type="surfaceshader" />
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>)"));
    props.set("mtlx_node_name", std::string("M"));
    props.set("mtlx_static_closure_pruning", std::string("on"));

    bool rejected = false;
    try {
        scene->create_material("MaterialXMaterial", props);
    } catch (const std::exception& e) {
        rejected = true;
        CHECK(contains(e.what(), "mtlx_static_closure_pruning"));
        CHECK(contains(e.what(), "mtlx_optimize_graph::closure_pruning"));
    }
    CHECK(rejected);
}

TEST_CASE_GPU("scene materialx 1.39 geomprop ID properties reject mismatched lists")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39">
  <geompropvalue name="custom_stream" type="vector3">
    <input name="geomprop" type="string" value="custom_stream" />
  </geompropvalue>
  <output name="custom_stream_out" type="vector3" nodename="custom_stream" />
</materialx>)"));
    props.set("mtlx_node_name", std::string("custom_stream_out"));
    props.set_list("mtlx_geomprop_names", std::vector<std::string>{"custom_stream"});
    props.set_list("mtlx_geomprop_ids", std::vector<int64_t>{});

    bool rejected = false;
    try {
        scene->create_material("MaterialXMaterial", props);
    } catch (const std::exception& e) {
        rejected = true;
        CHECK(contains(e.what(), "MaterialX geomprop name/ID lists must have the same length"));
    }
    CHECK(rejected);
}

TEST_CASE_GPU("scene materialx 1.39 reflected property changes regenerate shader")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));
    const std::filesystem::path initial_shader_path
        = testing::get_case_temp_directory() / "materialx139_property_initial.slang";
    const std::filesystem::path updated_shader_path
        = testing::get_case_temp_directory() / "materialx139_property_updated.slang";

    Properties props;
    props.set("mtlx_buffer", mtlx_layered_property_update_shader());
    props.set("mtlx_node_name", std::string("M"));
    props.set("mtlx_layering_mode", mtlx::LayeringMode::closure_tree);
    props.set("debug_write_shader_path", initial_shader_path.string());

    auto* material = checked_cast<MaterialXMaterial*>(scene->create_material("MaterialXMaterial", props));
    scene->update();

    const std::string initial_shader = read_text_file(initial_shader_path);
    REQUIRE(!initial_shader.empty());
    CHECK_FALSE(contains(initial_shader, "FlatRootBSDF"));

    const auto& desc = material->class_descriptor();
    const auto* debug_shader_path_property = desc.find_property("debug_write_shader_path");
    const auto* layering_property = desc.find_property("mtlx_layering_mode");
    REQUIRE(debug_shader_path_property);
    REQUIRE(layering_property);
    debug_shader_path_property->set<std::string>(material, updated_shader_path.string());
    layering_property->set<mtlx::LayeringMode>(material, mtlx::LayeringMode::bsdf_mix);
    scene->update();

    const std::string updated_shader = read_text_file(updated_shader_path);
    REQUIRE(!updated_shader.empty());
    CHECK(contains(updated_shader, "FlatRootBSDF"));
    CHECK_NE(initial_shader, updated_shader);
}

TEST_CASE_GPU("scene unlit material ABI and opacity")
{
    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("alpha_mode", AlphaMode::mask);
    props.set("alpha_factor", 0.6f);
    props.set("alpha_cutoff", 0.4f);
    props.set("base_color_factor", float3(1.5f, 2.f, 4.f));
    props.set("double_sided", true);

    auto* material = checked_cast<UnlitMaterial*>(scene->create_material("UnlitMaterial", props));
    scene->update();

    CHECK_EQ(material->flags(), shared::MaterialFlags::unlit | shared::MaterialFlags::two_sided);
    CHECK_EQ(material->_base_color_factor(), float3(1.5f, 2.f, 4.f));
    CHECK(material->_double_sided());

    const Material::OpacityDesc opacity = material->opacity_desc();
    CHECK_EQ(opacity.flags, shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold);
    CHECK_EQ(opacity.texture_channel, 3u);
    CHECK_EQ(opacity.factor, doctest::Approx(0.6f));
    CHECK_EQ(opacity.threshold, doctest::Approx(0.4f));

    auto requirements = scene->requirements();
    CHECK(!requirements.modules.empty());
}

TEST_CASE_GPU("scene emission-only material ABI")
{
    auto scene = Scene::create(ref(ctx.device));

    const ref<sgl::Texture> emission_texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm_srgb,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });

    Properties props;
    props.set("emission_color", float3(0.25f, 0.5f, 1.f));
    props.set("emission_color_texture", emission_texture);
    props.set("enable_color_temperature", true);
    props.set("color_temperature", 3000.f);
    props.set("emission_luminance", 4.f);

    auto* material = checked_cast<EmissionOnlyMaterial*>(scene->create_material("EmissionOnlyMaterial", props));
    scene->update();

    CHECK_EQ(material->flags(), shared::MaterialFlags::none);
    CHECK(material->enable_color_temperature());
    CHECK_EQ(material->color_temperature(), 3000.f);
    CHECK(!scene->requirements().modules.empty());
}

TEST_CASE_GPU("scene mdl")
{
    try {
        auto scene = Scene::create(ref(ctx.device));

        std::filesystem::path mdl_path = testing::project_directory() / "data/assets/mdl_sdk_examples";

        Properties props;
        props.set("mdl_library_path", mdl_path.string());
        props.set("mdl_material_name", "carbon_composite::carbon_composite");

        scene->create_material("MDLMaterial", props);
        scene->update();

        auto reqs = scene->requirements();
        CHECK(!reqs.modules.empty());
        CHECK(reqs.modules.front()->slang_module());
    } catch (const std::runtime_error& e) {
        std::string what = e.what();
        if (what.find("Failed to load the MDL SDK") != std::string::npos) {
            MESSAGE("Skipping scene mdl test due to MDL SDK initialization mismatch: " << what);
            return;
        }
        throw;
    }
}

TEST_CASE_GPU("scene mdl opacity classification and shader linkage")
{
    try {
        auto scene = Scene::create(ref(ctx.device));
        const std::filesystem::path mdl_path = testing::project_directory() / "data/assets/test_mdl_opacity";
        const std::filesystem::path shader_path = testing::get_case_temp_directory() / "mdl_opacity.slang";

        auto create_material = [&](std::string_view name, bool write_shader = false)
        {
            Properties props;
            props.set("mdl_library_path", mdl_path.string());
            props.set("mdl_material_name", fmt::format("opacity_test::{}", name));
            props.set("mdl_class_compilation", true);
            if (write_shader)
                props.set("debug_write_shader_path", shader_path.string());
            return scene->create_material<MDLMaterial>(props);
        };

        MDLMaterial* opaque = create_material("opaque");
        MDLMaterial* transmissive_opaque = create_material("transmissive_opaque");
        scene->update();
        CHECK(!scene->requirements().requires_opacity_evaluation);

        MDLMaterial* transparent = create_material("transparent");
        MDLMaterial* constant_half = create_material("constant_half");
        MDLMaterial* parameterized = create_material("parameterized", true);
        MDLMaterial* textured = create_material("textured");
        scene->update();

        CHECK_EQ(opaque->opacity_desc().flags, shared::OpacityFlags::none);
        CHECK_EQ(transmissive_opaque->opacity_desc().flags, shared::OpacityFlags::none);

        const Material::OpacityDesc transparent_desc = transparent->opacity_desc();
        CHECK_EQ(transparent_desc.flags, shared::OpacityFlags::enabled);
        CHECK_EQ(transparent_desc.factor, 0.f);

        const Material::OpacityDesc constant_half_desc = constant_half->opacity_desc();
        CHECK_EQ(constant_half_desc.flags, shared::OpacityFlags::enabled);
        CHECK_EQ(constant_half_desc.factor, 0.5f);

        const auto evaluated_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
        CHECK_EQ(parameterized->opacity_desc().flags, evaluated_flags);
        CHECK_EQ(textured->opacity_desc().flags, evaluated_flags);
        CHECK_EQ(textured->build_texture_list().size(), 1);
        CHECK(scene->requirements().requires_opacity_evaluation);

        const std::string shader = read_text_file(shader_path);
        CHECK(contains(shader, "override float eval_opacity"));
        CHECK(contains(shader, "geometry_cutout_opacity(state)"));
        CHECK(parameterized->required_module());
        CHECK(parameterized->required_module()->slang_module());
    } catch (const std::runtime_error& e) {
        std::string what = e.what();
        if (what.find("Failed to load the MDL SDK") != std::string::npos) {
            MESSAGE("Skipping scene mdl opacity test due to MDL SDK initialization mismatch: " << what);
            return;
        }
        throw;
    }
}

TEST_SUITE_END();
