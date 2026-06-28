// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/asset_resolver.h"
#include "falcor2/render/material/materialx/codegen/codegen.h"
#include "falcor2/render/material/materialx/materialx_material.h"
#include "falcor2/render/scene.h"

#include <sgl/device/device.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace falcor;

namespace {

const char* k_layering_traversal_order_property = "materialx_layering_traversal_order";

std::string materialx_139_source_from_138(std::string source)
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

materialx::CodeGenDesc make_materialx_codegen_desc(const std::string& shader)
{
    materialx::CodeGenDesc desc;
    desc.document = materialx_139_source_from_138(shader);
    desc.node_name = "Tiled_Brass";
    desc.positionfree_layering = false;
    desc.transmissive_bsdfs.clear();
    desc.auto_gamma_image = true;
    desc.make_editable = false;
    return desc;
}

std::vector<std::string> layering_traversal_order(const materialx::CodeGenResult& result)
{
    return result.codegen_metadata.get_list<std::string>(k_layering_traversal_order_property);
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

std::string mx139_layered_property_update_shader()
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
        auto result = materialx::CodeGen::generate(make_materialx_codegen_desc(variant.shader));
        REQUIRE(result);
        CHECK(starts_with(result->module_name, "mx139_"));
        CHECK(contains(result->module_source, "namespace mx139"));
        CHECK_FALSE(contains(result->module_source, "MaterialX138"));
        if (variant.expects_layering)
            CHECK_FALSE(normalize_layering_traversal_order(layering_traversal_order(*result), true).empty());
    }
}

TEST_CASE_GPU("scene materialx rejects CUDA by default")
{
    if (ctx.device->type() != sgl::DeviceType::cuda)
        return;

    auto scene = Scene::create(ref(ctx.device));
    bool rejected = false;
    try {
        scene->create_material("MaterialXMaterial", Properties{});
    } catch (const std::exception& e) {
        rejected = true;
        CHECK(contains(e.what(), "MaterialXMaterial is disabled on CUDA"));
        CHECK(contains(e.what(), "NVRTC 12.6"));
    }
    CHECK(rejected);
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

TEST_CASE_GPU("scene materialx 1.39 defaults to bsdf_mix layering")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));
    const std::filesystem::path shader_path
        = testing::get_case_temp_directory() / "materialx139_default_bsdf_mix_layering.slang";

    Properties props;
    props.set("mtlx_buffer", mx139_layered_property_update_shader());
    props.set("mtlx_node_name", std::string("M"));
    props.set("debug_write_shader_path", shader_path.string());

    scene->create_material("MaterialXMaterial", props);
    scene->update();

    const std::string shader_source = read_text_file(shader_path);
    REQUIRE(!shader_source.empty());
    CHECK(contains(shader_source, "FlatRootBSDF"));
    CHECK_FALSE(contains(shader_source, "MaterialInstance_Layered"));
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
    props.set("mtlx_buffer", mx139_layered_property_update_shader());
    props.set("mtlx_node_name", std::string("M"));
    props.set("mtlx_layering_mode", materialx::LayeringMode::closure_tree);
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
    layering_property->set<materialx::LayeringMode>(material, materialx::LayeringMode::bsdf_mix);
    scene->update();

    const std::string updated_shader = read_text_file(updated_shader_path);
    REQUIRE(!updated_shader.empty());
    CHECK(contains(updated_shader, "FlatRootBSDF"));
    CHECK_NE(initial_shader, updated_shader);
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

TEST_SUITE_END();
