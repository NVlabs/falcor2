// SPDX-License-Identifier: Apache-2.0


#include "testing.h"
#include "falcor2/render/material/materialx/codegen/codegen.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/material_conversions.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/importers/usd_importer/usd_importer_utils.h"
#include "falcor2/core/object.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/sdf/assetPath.h>
END_DISABLE_USD_WARNINGS

#include <sgl/core/platform.h>

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace falcor;

TEST_SUITE_BEGIN("usd_importer");

TEST_CASE("usd_value_property_conversion")
{
    SUBCASE("bool")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "flag", pxr::VtValue(true)));
        CHECK_EQ(props.type("flag"), PropertyType::bool_);
        CHECK_EQ(props.get<bool>("flag"), true);
    }

    SUBCASE("int")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "count", pxr::VtValue(uint32_t(7))));
        CHECK_EQ(props.type("count"), PropertyType::int_);
        CHECK_EQ(props.get<int64_t>("count"), 7);
    }

    SUBCASE("float")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "roughness", pxr::VtValue(3.5)));
        CHECK_EQ(props.type("roughness"), PropertyType::float_);
        CHECK(props.get<double>("roughness") == doctest::Approx(3.5));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "roughness", pxr::VtValue(3.5f)));
        CHECK_EQ(props.type("roughness"), PropertyType::float_);
        CHECK(props.get<double>("roughness") == doctest::Approx(3.5f));
    }

    SUBCASE("int2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "tile", pxr::VtValue(pxr::GfVec2i(1, 2))));
        CHECK_EQ(props.type("tile"), PropertyType::int2);
        CHECK_EQ(props.get<int2>("tile"), int2(1, 2));
    }

    SUBCASE("int3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "indices", pxr::VtValue(pxr::GfVec3i(1, 2, 3))));
        CHECK_EQ(props.type("indices"), PropertyType::int3);
        CHECK_EQ(props.get<int3>("indices"), int3(1, 2, 3));
    }

    SUBCASE("int4")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "plane", pxr::VtValue(pxr::GfVec4i(1, 2, 3, 4))));
        CHECK_EQ(props.type("plane"), PropertyType::int4);
        CHECK_EQ(props.get<int4>("plane"), int4(1, 2, 3, 4));
    }

    SUBCASE("uint2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "resolution", pxr::VtValue(pxr::GfSize2(640, 480))));
        CHECK_EQ(props.type("resolution"), PropertyType::uint2);
        CHECK_EQ(props.get<uint2>("resolution"), uint2(640, 480));
    }

    SUBCASE("uint3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "grid", pxr::VtValue(pxr::GfSize3(4, 5, 6))));
        CHECK_EQ(props.type("grid"), PropertyType::uint3);
        CHECK_EQ(props.get<uint3>("grid"), uint3(4, 5, 6));
    }

    SUBCASE("float2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "uv", pxr::VtValue(pxr::GfVec2d(0.25, 0.5))));
        CHECK_EQ(props.type("uv"), PropertyType::float2);
        CHECK_EQ(props.get<float2>("uv"), float2(0.25f, 0.5f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "uv", pxr::VtValue(pxr::GfVec2f(0.25f, 0.5f))));
        CHECK_EQ(props.type("uv"), PropertyType::float2);
        CHECK_EQ(props.get<float2>("uv"), float2(0.25f, 0.5f));
    }

    SUBCASE("float3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "color", pxr::VtValue(pxr::GfVec3d(0.25, 0.5, 0.75))));
        CHECK_EQ(props.type("color"), PropertyType::float3);
        CHECK_EQ(props.get<float3>("color"), float3(0.25f, 0.5f, 0.75f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "color", pxr::VtValue(pxr::GfVec3f(0.25f, 0.5f, 0.75f))));
        CHECK_EQ(props.type("color"), PropertyType::float3);
        CHECK_EQ(props.get<float3>("color"), float3(0.25f, 0.5f, 0.75f));
    }

    SUBCASE("float4")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "rgba", pxr::VtValue(pxr::GfVec4d(0.25, 0.5, 0.75, 1.0))));
        CHECK_EQ(props.type("rgba"), PropertyType::float4);
        CHECK_EQ(props.get<float4>("rgba"), float4(0.25f, 0.5f, 0.75f, 1.0f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "rgba", pxr::VtValue(pxr::GfVec4f(0.25f, 0.5f, 0.75f, 1.0f))));
        CHECK_EQ(props.type("rgba"), PropertyType::float4);
        CHECK_EQ(props.get<float4>("rgba"), float4(0.25f, 0.5f, 0.75f, 1.0f));
    }

    SUBCASE("string token")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "mode", pxr::VtValue(pxr::TfToken("repeat"))));
        CHECK_EQ(props.type("mode"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("mode"), std::string("repeat"));
    }

    SUBCASE("string asset path")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "path", pxr::VtValue(pxr::SdfAssetPath("textures/albedo.png"))));
        CHECK_EQ(props.type("path"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("path"), std::string("textures/albedo.png"));
    }

    SUBCASE("unsupported array stringify fallback")
    {
        Properties props;
        pxr::VtArray<pxr::VtArray<bool>> values;
        pxr::VtArray<bool> nested;
        nested.push_back(true);
        nested.push_back(false);
        values.push_back(nested);
        pxr::VtValue value(values);
        REQUIRE(usd_importer::set_from_value(props, "flags", value));
        CHECK_EQ(props.type("flags"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("flags"), pxr::TfStringify(value));
    }

    SUBCASE("bool array")
    {
        Properties props;
        pxr::VtArray<bool> values;
        values.push_back(true);
        values.push_back(false);
        values.push_back(true);
        REQUIRE(usd_importer::set_from_value(props, "flags", pxr::VtValue(values)));
        CHECK_EQ(props.type("flags"), PropertyType::list);
        CHECK_EQ(props.get_list<bool>("flags"), std::vector<bool>{true, false, true});
    }

    SUBCASE("int array")
    {
        Properties props;
        pxr::VtArray<uint32_t> values;
        values.push_back(7u);
        values.push_back(11u);
        REQUIRE(usd_importer::set_from_value(props, "counts", pxr::VtValue(values)));
        CHECK_EQ(props.type("counts"), PropertyType::list);
        CHECK_EQ(props.get_list<int64_t>("counts"), std::vector<int64_t>{7, 11});
    }

    SUBCASE("float array")
    {
        Properties props;
        pxr::VtArray<float> values;
        values.push_back(0.5f);
        values.push_back(1.25f);
        REQUIRE(usd_importer::set_from_value(props, "weights", pxr::VtValue(values)));
        CHECK_EQ(props.type("weights"), PropertyType::list);
        CHECK_EQ(props.get_list<double>("weights"), std::vector<double>{0.5, 1.25});
    }

    SUBCASE("float3 array")
    {
        Properties props;
        pxr::VtArray<pxr::GfVec3d> values;
        values.push_back(pxr::GfVec3d(0.25, 0.5, 0.75));
        values.push_back(pxr::GfVec3d(1.0, 2.0, 3.0));
        REQUIRE(usd_importer::set_from_value(props, "colors", pxr::VtValue(values)));
        CHECK_EQ(props.type("colors"), PropertyType::list);
        CHECK_EQ(
            props.get_list<float3>("colors"),
            std::vector<float3>{float3(0.25f, 0.5f, 0.75f), float3(1.0f, 2.0f, 3.0f)}
        );
    }

    SUBCASE("string token array")
    {
        Properties props;
        pxr::VtArray<pxr::TfToken> values;
        values.push_back(pxr::TfToken("repeat"));
        values.push_back(pxr::TfToken("clamp"));
        REQUIRE(usd_importer::set_from_value(props, "modes", pxr::VtValue(values)));
        CHECK_EQ(props.type("modes"), PropertyType::list);
        CHECK_EQ(
            props.get_list<std::string>("modes"),
            std::vector<std::string>{std::string("repeat"), std::string("clamp")}
        );
    }

    SUBCASE("string asset path array")
    {
        Properties props;
        pxr::VtArray<pxr::SdfAssetPath> values;
        values.push_back(pxr::SdfAssetPath("textures/albedo.png"));
        values.push_back(pxr::SdfAssetPath("textures/normal.png"));
        REQUIRE(usd_importer::set_from_value(props, "paths", pxr::VtValue(values)));
        CHECK_EQ(props.type("paths"), PropertyType::list);
        CHECK_EQ(
            props.get_list<std::string>("paths"),
            std::vector<std::string>{std::string("textures/albedo.png"), std::string("textures/normal.png")}
        );
    }
}

// Helper function to validate cornell-box geometry
static void validate_cornell_box_geometry(const ref<falcor::ImporterScene>& scene)
{
    CHECK_EQ(scene->meshes.size(), 16);

    auto find_mesh = [&](const std::string& name) -> const ImporterMesh*
    {
        auto it = std::find_if(
            scene->meshes.begin(),
            scene->meshes.end(),
            [&](const ImporterMesh& m)
            {
                return m.name == name;
            }
        );
        return it != scene->meshes.end() ? &(*it) : nullptr;
    };

    auto find_subgeometry = [](const ImporterMesh& mesh, const std::string& name) -> const ImporterMesh::Subgeometry*
    {
        auto it = std::find_if(
            mesh.subgeometries.begin(),
            mesh.subgeometries.end(),
            [&](const ImporterMesh::Subgeometry& s)
            {
                return s.name == name;
            }
        );
        return it != mesh.subgeometries.end() ? &(*it) : nullptr;
    };

    struct MeshRef {
        std::string name;
        size_t vertices_count;
    };

    std::vector<MeshRef> mesh_refs = {
        {"/cornell_box/tall_box_front/tall_box_front", 4},
        {"/cornell_box/tall_box_right/tall_box_right", 4},
        {"/cornell_box/tall_box_back/tall_box_back", 4},
        {"/cornell_box/tall_box_left/tall_box_left", 4},
        {"/cornell_box/tall_box_top/tall_box_top", 4},
        {"/cornell_box/short_box_back/short_box_back", 4},
        {"/cornell_box/short_box_right/short_box_right", 4},
        {"/cornell_box/short_box_front/short_box_front", 4},
        {"/cornell_box/short_box_left/short_box_left", 4},
        {"/cornell_box/short_box_top/short_box_top", 4},
        {"/cornell_box/left_wall/left_wall", 4},
        {"/cornell_box/right_wall/right_wall", 4},
        {"/cornell_box/back_wall/back_wall", 4},
        {"/cornell_box/light/light", 4},
        {"/cornell_box/ceiling/ceiling", 4},
        {"/cornell_box/floor/floor", 4},
    };

    for (const auto& ref : mesh_refs) {
        const ImporterMesh* mesh = find_mesh(ref.name);
        CHECK(mesh);
        if (mesh) {
            CHECK_EQ(mesh->vertex_count(), ref.vertices_count);
            CHECK_EQ(mesh->subgeometries.size(), 1);
            const ImporterMesh::Subgeometry* subgeo = find_subgeometry(*mesh, ref.name);
            CHECK(subgeo);
            if (subgeo) {
                CHECK_EQ(subgeo->indices.size(), 2);
            }
        }
    }

    CHECK_EQ(scene->cameras.size(), 1);
    if (scene->cameras.size() > 0) {
        CHECK_EQ(scene->cameras[0].name, "camera");
    }
}

// Helper function to print all materials in a scene
static void print_materials(const ref<falcor::ImporterScene>& scene, const std::string& test_name)
{
    MESSAGE("\n=== Materials for " << test_name << " ===");
    MESSAGE("Total materials: " << scene->materials.size());
    for (const auto& mat : scene->materials) {
        MESSAGE("\nMaterial: " << mat.name);
        auto converted = convert_material(mat);
        if (converted.has_value()) {
            MESSAGE(converted->to_string());
        } else {
            MESSAGE("<no conversion result>");
        }
    }
}

TEST_CASE("load_test_usdpreviewsurface")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "usdpreviewsurface" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        float3 base_color_factor;
        float3 emissive_factor;
        float metallic_factor;
        float roughness_factor;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", float3(0.725f, 0.71f, 0.68f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
        {"/cornell_box/materials/green", float3(0.14f, 0.45f, 0.091f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
        {"/cornell_box/materials/light", float3(0.18f, 0.18f, 0.18f), float3(17.f, 12.f, 4.f), 0.f, 0.5f},
        {"/cornell_box/materials/red", float3(0.63f, 0.065f, 0.05f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"UsdPreviewSurface"}, {"StandardMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "StandardMaterial");
        CHECK_EQ(converted->get<float3>("base_color_factor", float3(0.f)), expected.base_color_factor);
        CHECK_EQ(converted->get<float3>("emissive_factor", float3(0.f)), expected.emissive_factor);
        CHECK_EQ(converted->get<float>("metallic_factor", -1.f), expected.metallic_factor);
        CHECK_EQ(converted->get<float>("roughness_factor", -1.f), expected.roughness_factor);
    }
}

TEST_CASE("load_test_mtlx_ref")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-ref" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    const std::filesystem::path expected_mtlx_basepath
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-ref";

    struct ExpectedMaterial {
        std::string name;
        std::string node_name;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/MaterialX/Materials/white", "white"},
        {"/cornell_box/MaterialX/Materials/red", "red"},
        {"/cornell_box/MaterialX/Materials/light", "light"},
        {"/cornell_box/MaterialX/Materials/green", "green"},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it);
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
        CHECK_EQ(converted->get<std::string_view>("mtlx_node_name", ""), expected.node_name);
        CHECK_EQ(converted->get<std::string_view>("mtlx_path", ""), "cornell-box.mtlx");

        const std::filesystem::path actual_basepath = converted->get<std::filesystem::path>("mtlx_basepath", {});
        CHECK_EQ(actual_basepath.lexically_normal(), expected_mtlx_basepath.lexically_normal());
    }
}

TEST_CASE("load_test_mtlx_usdshade")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-usdshade" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        std::string node_name;
        std::string diffuse_color_value;
        std::string roughness_value;
        std::string emissive_color_value;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", "white", "0.725, 0.71, 0.68", "1", ""},
        {"/cornell_box/materials/red", "red", "0.63, 0.065, 0.05", "1", ""},
        {"/cornell_box/materials/light", "light", "", "", "17, 12, 4"},
        {"/cornell_box/materials/green", "green", "0.14, 0.45, 0.091", "1", ""},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"MaterialX"}, {"MaterialXMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
        CHECK_EQ(converted->get<std::string_view>("mtlx_node_name", ""), expected.node_name);

        const std::string mtlx_buffer = converted->get<std::string>("mtlx_buffer", "");
        CHECK_FALSE(mtlx_buffer.empty());
        CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
        CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
        CHECK(mtlx_buffer.find("<surfacematerial name=\"" + expected.node_name + "\"") != std::string::npos);

        if (!expected.diffuse_color_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"diffuseColor\" type=\"color3\" value=\"" + expected.diffuse_color_value + "\" />"
                )
                != std::string::npos
            );
        }
        if (!expected.roughness_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"roughness\" type=\"float\" value=\"" + expected.roughness_value + "\" />"
                )
                != std::string::npos
            );
        }

        if (!expected.emissive_color_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"emissiveColor\" type=\"color3\" value=\"" + expected.emissive_color_value + "\" />"
                )
                != std::string::npos
            );
        }
    }
}

TEST_CASE("load_test_mdl")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mdl" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        std::string mdl_name_suffix;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", "::white"},
        {"/cornell_box/materials/red", "::red"},
        {"/cornell_box/materials/light", "::light"},
        {"/cornell_box/materials/green", "::green"},
    };

    const std::string expected_mdl_library_path = testing::project_directory().root_path().string();

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"MDL"}, {"MDLMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MDLMaterial");
        CHECK_EQ(converted->get<bool>("mdl_class_compilation", false), true);
        CHECK_EQ(converted->get<std::string>("mdl_library_path", ""), expected_mdl_library_path);

        const std::string mdl_material_name = converted->get<std::string>("mdl_material_name", "");
        CHECK_FALSE(mdl_material_name.empty());
        CHECK(mdl_material_name.ends_with(expected.mdl_name_suffix));
    }
}

TEST_CASE("load_test_display_color")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "display-color" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    // display-color uses primitive display colors, not materials
    CHECK_EQ(scene->materials.size(), 0);

    validate_cornell_box_geometry(scene);

    // Print any display color information from the meshes
    std::cout << "\n=== Mesh Display Colors for display-color ===" << std::endl;
    std::cout << "Total meshes: " << scene->meshes.size() << std::endl;
    for (size_t i = 0; i < scene->meshes.size() && i < 5; ++i) {
        const auto& mesh = scene->meshes[i];
        std::cout << "Mesh " << i << ": " << mesh.name << std::endl;
    }
}

TEST_CASE("load_scene_usdpreviewsurface_material_input")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path.string());

    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    const ImporterMaterial& material = scene->materials[0];
    CHECK_EQ(material.name, "/Root/Looks/Mat");
    CHECK_EQ(material.params.get<std::string_view>("_type", ""), "usd_UsdPreviewSurface");
    CHECK_EQ(material.params.get<float3>("diffuseColor", float3(0)), float3(0.2f, 0.7f, 0.3f));
}

TEST_CASE("convert_material_usdpreviewsurface_to_standardmaterial")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    auto result = convert_material(scene->materials[0], {"UsdPreviewSurface"}, {"StandardMaterial"});
    REQUIRE(result.has_value());

    Properties expected;
    expected.set("_scene_material_type", std::string("StandardMaterial"));
    expected.set("base_color_factor", float3(0.2f, 0.7f, 0.3f));
    expected.set("emissive_factor", float3(0.0f, 0.0f, 0.0f));
    expected.set("metallic_factor", 0.0f);
    expected.set("roughness_factor", 0.4f);
    CHECK_EQ(*result, expected);
}

TEST_CASE("convert_material_materialx_to_materialxmaterial_mx139_reference")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "tests" / "native" / "usd_importer"
        / "test_scope_materials_mx139_openpbr.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    const std::filesystem::path expected_mtlx_basepath
        = testing::project_directory() / "external" / "MaterialX" / "resources" / "Materials" / "Examples" / "OpenPbr";

    const std::string path = "/Root/MaterialX/Materials/Default";
    auto it = std::find_if(
        scene->materials.begin(),
        scene->materials.end(),
        [&](const ImporterMaterial& m)
        {
            return m.name == path;
        }
    );
    REQUIRE(it != scene->materials.end());

    auto result = convert_material(*it, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
    CHECK_EQ(result->get<std::filesystem::path>("mtlx_path", {}), std::filesystem::path("open_pbr_default.mtlx"));
    CHECK_EQ(result->get<std::string_view>("mtlx_node_name", ""), "Default");

    const std::filesystem::path actual_basepath = result->get<std::filesystem::path>("mtlx_basepath", {});
    CHECK_EQ(actual_basepath.lexically_normal(), expected_mtlx_basepath.lexically_normal());

    materialx::CodeGenDesc desc;
    desc.document = actual_basepath / result->get<std::filesystem::path>("mtlx_path", {});
    desc.node_name = result->get<std::string>("mtlx_node_name", "");
    auto generated = materialx::CodeGen::generate(desc);
    REQUIRE(generated);
    CHECK_FALSE(generated->module_source.empty());
}

TEST_CASE("convert_material_materialx_id_network_to_materialxmaterial")
{
    ImporterMaterial material;
    material.name = "/Root/Looks/GeneratedMtlx";

    Properties terminal;
    terminal.set("_path", "/Root/Looks/GeneratedMtlx/Surface");
    terminal.set("_type", "Shader");
    terminal.set("info:id", "ND_standard_surface_surfaceshader");
    terminal.set("info:implementationSource", "id");
    terminal.set("base_color", float3(0.2f, 0.7f, 0.3f));
    terminal.set("base_color:typename", "color3f");
    terminal.set("roughness", 0.4f);
    terminal.set("roughness:typename", "float");

    material.output_to_material_network["_terminal:mtlx:surface"] = {terminal};

    auto result = convert_material(material, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type"), "MaterialXMaterial");
    CHECK_FALSE(result->get<std::string>("mtlx_node_name").empty());

    const std::string mtlx_buffer = result->get<std::string>("mtlx_buffer");
    CHECK_FALSE(mtlx_buffer.empty());
    // Verify that the buffer contains MaterialX XML structure
    CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
    CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
}

TEST_CASE("convert_material_mx139_openpbr_id_network_to_materialxmaterial")
{
    ImporterMaterial material;
    material.name = "/Root/Looks/GeneratedOpenPbr";

    Properties terminal;
    terminal.set("_path", "/Root/Looks/GeneratedOpenPbr/Surface");
    terminal.set("_type", "Shader");
    terminal.set("info:id", "ND_open_pbr_surface_surfaceshader");
    terminal.set("info:implementationSource", "id");
    terminal.set("base_weight", 1.0f);
    terminal.set("base_weight:typename", "float");
    terminal.set("base_color", float3(0.2f, 0.7f, 0.3f));
    terminal.set("base_color:typename", "color3f");
    terminal.set("specular_roughness", 0.4f);
    terminal.set("specular_roughness:typename", "float");

    material.output_to_material_network["_terminal:mtlx:surface"] = {terminal};

    auto result = convert_material(material, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type"), "MaterialXMaterial");
    CHECK_EQ(result->get<std::string_view>("mtlx_node_name", ""), "GeneratedOpenPbr");

    const std::string mtlx_buffer = result->get<std::string>("mtlx_buffer");
    CHECK_FALSE(mtlx_buffer.empty());
    CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
    CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
    CHECK(mtlx_buffer.find("<open_pbr_surface") != std::string::npos);

    materialx::CodeGenDesc desc;
    desc.document = mtlx_buffer;
    desc.node_name = result->get<std::string>("mtlx_node_name", "");
    auto generated = materialx::CodeGen::generate(desc);
    REQUIRE(generated);
    CHECK_FALSE(generated->module_source.empty());
}

TEST_CASE("convert_material_no_match_returns_empty")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    auto result = convert_material(scene->materials[0], {"MDL"}, {"MDLMaterial"});
    CHECK_FALSE(result.has_value());
}

TEST_CASE("load_linear_curves")
{
    const std::filesystem::path curves_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_curves.usda";
    REQUIRE(std::filesystem::exists(curves_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(curves_path);
    REQUIRE(scene);

    // We expect 6 curve prims (2 linear + 4 cubic).
    REQUIRE_EQ(scene->curves.size(), 6);

    // Find curves by exact trailing name match.
    auto find_curve = [&](const std::string& name) -> const ImporterCurve*
    {
        for (const auto& c : scene->curves) {
            if (c.name.size() >= name.size() && c.name.substr(c.name.size() - name.size()) == name)
                return &c;
        }
        return nullptr;
    };

    // First curve: 3 vertices, 2 segments.
    {
        const ImporterCurve* curve = find_curve("linear_curves");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 5);
        CHECK_EQ(curve->radii.size(), 5);
        // First curve has 3 verts (2 segments), second has 2 verts (1 segment) = 3 segments * 2 indices = 6.
        CHECK_EQ(curve->indices.size(), 6);

        // Check radii are half of widths.
        CHECK(curve->radii[0] == doctest::Approx(0.05f));
        CHECK(curve->radii[1] == doctest::Approx(0.04f));
        CHECK(curve->radii[2] == doctest::Approx(0.03f));

        // Check indices form line segments: (0,1), (1,2), (3,4).
        CHECK_EQ(curve->indices[0], 0);
        CHECK_EQ(curve->indices[1], 1);
        CHECK_EQ(curve->indices[2], 1);
        CHECK_EQ(curve->indices[3], 2);
        CHECK_EQ(curve->indices[4], 3);
        CHECK_EQ(curve->indices[5], 4);
    }

    // Second curve: no widths, should use default radius.
    {
        const ImporterCurve* curve = find_curve("linear_curves_no_widths");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 2);
        CHECK_EQ(curve->radii.size(), 2);
        CHECK_EQ(curve->indices.size(), 2);

        // Default radius should be 0.01.
        CHECK(curve->radii[0] == doctest::Approx(0.01f));
        CHECK(curve->radii[1] == doctest::Approx(0.01f));
    }

    // Check nodes reference curves.
    int curve_node_count = 0;
    for (const auto& node : scene->nodes) {
        if (node.curve_index >= 0)
            curve_node_count++;
    }
    CHECK_EQ(curve_node_count, 6);
}

TEST_CASE("load_cubic_bezier_curves")
{
    const std::filesystem::path curves_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_curves.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(curves_path);
    REQUIRE(scene);

    // Find curves by exact trailing name match.
    auto find_curve = [&](const std::string& name) -> const ImporterCurve*
    {
        for (const auto& c : scene->curves) {
            if (c.name.size() >= name.size() && c.name.substr(c.name.size() - name.size()) == name)
                return &c;
        }
        return nullptr;
    };

    // Bezier curve: 4 CVs = 1 span, 8 segments per span = 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bezier_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 9); // 8 segments + 1 = 9 vertices
        CHECK_EQ(curve->radii.size(), 9);
        CHECK_EQ(curve->indices.size(), 16); // 8 segments * 2 indices

        // Bezier curve endpoints should match first and last control points.
        CHECK(curve->positions.front().x == doctest::Approx(0.0f));
        CHECK(curve->positions.front().y == doctest::Approx(0.0f));
        CHECK(curve->positions.back().x == doctest::Approx(1.0f));
        CHECK(curve->positions.back().y == doctest::Approx(0.0f));

        // Radii at endpoints: widths are [0.2, 0.2, 0.1, 0.1], radii = width/2.
        // Bezier interpolates endpoints, so first radius = 0.1, last radius = 0.05.
        CHECK(curve->radii.front() == doctest::Approx(0.1f));
        CHECK(curve->radii.back() == doctest::Approx(0.05f));

        // Midpoint (t=0.5) of bezier with CVs (0,0,0),(0,1,0),(1,1,0),(1,0,0):
        // P(0.5) = 0.125*(0,0,0) + 0.375*(0,1,0) + 0.375*(1,1,0) + 0.125*(1,0,0)
        //        = (0.5, 0.75, 0)
        CHECK(curve->positions[4].x == doctest::Approx(0.5f));
        CHECK(curve->positions[4].y == doctest::Approx(0.75f));
    }

    // Multi-span Bezier: 7 CVs = 2 spans (stride 3), 2*8=16 segments, 17 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bezier_multi_span");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 17); // 2 spans * 8 + 1 = 17 vertices
        CHECK_EQ(curve->radii.size(), 17);
        CHECK_EQ(curve->indices.size(), 32); // 16 segments * 2 indices

        // First endpoint = first CV (0,0,0).
        CHECK(curve->positions.front().x == doctest::Approx(0.0f));
        CHECK(curve->positions.front().y == doctest::Approx(0.0f));
        // Last endpoint = last CV (3,1,0).
        CHECK(curve->positions.back().x == doctest::Approx(3.0f));
        CHECK(curve->positions.back().y == doctest::Approx(1.0f));

        // Junction between spans (vertex 8) should be CV[3] = (1,0,0).
        CHECK(curve->positions[8].x == doctest::Approx(1.0f));
        CHECK(curve->positions[8].y == doctest::Approx(0.0f));
    }

    // B-spline curve: 4 CVs = 1 span, 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bspline_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 9);
        CHECK_EQ(curve->indices.size(), 16);

        // B-spline does NOT interpolate endpoints.
        // P(0) with bspline basis and CVs (0,0,0),(0,1,0),(1,1,0),(1,0,0):
        // = (1/6)*(0,0,0) + (4/6)*(0,1,0) + (1/6)*(1,1,0) + 0*(1,0,0)
        // = (1/6, 5/6, 0)
        CHECK(curve->positions.front().x == doctest::Approx(1.f / 6.f));
        CHECK(curve->positions.front().y == doctest::Approx(5.f / 6.f));
    }

    // CatmullRom curve: 4 CVs = 1 span, 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("catmullrom_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->positions.size(), 9);
        CHECK_EQ(curve->indices.size(), 16);

        // CatmullRom interpolates through P1 and P2.
        // P(0) = P1 = (0,1,0)
        CHECK(curve->positions.front().x == doctest::Approx(0.0f));
        CHECK(curve->positions.front().y == doctest::Approx(1.0f));
        // P(1) = P2 = (1,1,0)
        CHECK(curve->positions.back().x == doctest::Approx(1.0f));
        CHECK(curve->positions.back().y == doctest::Approx(1.0f));
    }
}

// ---------------------------------------------------------------------------
// Animation import tests
// ---------------------------------------------------------------------------

/// Helper to find a node by name in an ImporterScene.
static const ImporterNode* find_node(const ref<ImporterScene>& scene, const std::string& name)
{
    for (const auto& node : scene->nodes) {
        if (node.name == name)
            return &node;
    }
    return nullptr;
}

/// Helper to get the index of a node by name.
static int find_node_index(const ref<ImporterScene>& scene, const std::string& name)
{
    for (size_t i = 0; i < scene->nodes.size(); ++i) {
        if (scene->nodes[i].name == name)
            return static_cast<int>(i);
    }
    return -1;
}

TEST_CASE("UsdImporter - Linear Translation Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file
        = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_linear_translation.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    // Should have animation data.
    CHECK_FALSE(scene->animation.channels.empty());
    CHECK_EQ(scene->animation.name, "UsdAnimation");

    // Find the animated node.
    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have a translation animation channel.
    REQUIRE_NE(cube_node->animation_translation, -1);
    CHECK_EQ(cube_node->animation_rotation, -1);
    CHECK_EQ(cube_node->animation_scale, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(ch.keyframe_count(), 3);

    // Time codes 0, 24, 48 at 24 fps => 0s, 1s, 2s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));
    CHECK_EQ(ch.times[2], doctest::Approx(2.f));

    // Values: (0,0,0), (2.5,0,0), (5,0,0).
    REQUIRE_EQ(ch.values.size(), 9);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[3], doctest::Approx(2.5f).epsilon(1e-4));
    CHECK_EQ(ch.values[4], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[5], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[6], doctest::Approx(5.f).epsilon(1e-4));
    CHECK_EQ(ch.values[7], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[8], doctest::Approx(0.f).epsilon(1e-4));

    // No tangent data for linear.
    CHECK(ch.in_tangents.empty());
    CHECK(ch.out_tangents.empty());

    CHECK_EQ(scene->animation.duration(), doctest::Approx(2.f));
}

TEST_CASE("UsdImporter - Rotation Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_rotation.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have a rotation animation channel (from matrix decomposition).
    REQUIRE_NE(cube_node->animation_rotation, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_rotation];
    CHECK_EQ(ch.value_type, AnimationValueType::quatf);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(ch.keyframe_count(), 2);

    // Time codes 0, 24 at 24 fps => 0s, 1s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));

    // First keyframe: identity rotation (0, 0, 0, 1) in xyzw order.
    REQUIRE_EQ(ch.values.size(), 8);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[3], doctest::Approx(1.f).epsilon(1e-3));

    // Second keyframe: 90 degrees around Y => (0, sin(45), 0, cos(45)).
    CHECK_EQ(ch.values[4], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[5], doctest::Approx(0.7071068f).epsilon(1e-3));
    CHECK_EQ(ch.values[6], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[7], doctest::Approx(0.7071068f).epsilon(1e-3));
}

TEST_CASE("UsdImporter - Multi Channel Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_multi_channel.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have all three animation channels.
    CHECK_NE(cube_node->animation_translation, -1);
    CHECK_NE(cube_node->animation_rotation, -1);
    CHECK_NE(cube_node->animation_scale, -1);

    // The union of time samples is {0, 48, 72} => {0s, 2s, 3s}.
    const auto& trans_ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(trans_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(trans_ch.keyframe_count(), 3);
    CHECK_EQ(trans_ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(trans_ch.times[1], doctest::Approx(2.f));
    CHECK_EQ(trans_ch.times[2], doctest::Approx(3.f));

    // Translation at t=2s should be (1,2,3), at t=3s held at (1,2,3).
    CHECK_EQ(trans_ch.values[3], doctest::Approx(1.f).epsilon(1e-3));
    CHECK_EQ(trans_ch.values[4], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(trans_ch.values[5], doctest::Approx(3.f).epsilon(1e-3));

    const auto& scale_ch = scene->animation.channels[cube_node->animation_scale];
    CHECK_EQ(scale_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(scale_ch.keyframe_count(), 3);

    // Scale at t=3s should be (2,2,2).
    CHECK_EQ(scale_ch.values[6], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(scale_ch.values[7], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(scale_ch.values[8], doctest::Approx(2.f).epsilon(1e-3));

    // Duration should be 3s (max time across channels).
    CHECK_EQ(scene->animation.duration(), doctest::Approx(3.f));
}

TEST_CASE("UsdImporter - Multi Node Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_multi_node.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    REQUIRE_GE(scene->animation.channels.size(), 2);

    const ImporterNode* parent_node = find_node(scene, "/Parent");
    REQUIRE(parent_node != nullptr);
    const ImporterNode* child_node = find_node(scene, "/Parent/Child");
    REQUIRE(child_node != nullptr);

    // Parent has translation animation.
    CHECK_NE(parent_node->animation_translation, -1);

    // Child has rotation animation.
    CHECK_NE(child_node->animation_rotation, -1);

    // Verify parent-child relationship.
    int parent_idx = find_node_index(scene, "/Parent");
    CHECK_EQ(child_node->parent, parent_idx);

    // Parent translation: (0,0,0) -> (10,0,0).
    const auto& parent_ch = scene->animation.channels[parent_node->animation_translation];
    CHECK_EQ(parent_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(parent_ch.keyframe_count(), 2);
    CHECK_EQ(parent_ch.values[3], doctest::Approx(10.f).epsilon(1e-3));

    // Child rotation channel.
    const auto& child_ch = scene->animation.channels[child_node->animation_rotation];
    CHECK_EQ(child_ch.value_type, AnimationValueType::quatf);
    REQUIRE_EQ(child_ch.keyframe_count(), 2);
}

TEST_CASE("UsdImporter - No Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_static.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    // Scene without animation should have empty animation data.
    CHECK(scene->animation.channels.empty());
    CHECK_EQ(scene->animation.duration(), 0.f);

    // No node should reference any animation channel.
    for (const auto& node : scene->nodes) {
        CHECK_EQ(node.animation_translation, -1);
        CHECK_EQ(node.animation_rotation, -1);
        CHECK_EQ(node.animation_scale, -1);
    }
}

TEST_CASE("UsdImporter - Time Code Normalization (30fps)")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_30fps.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);
    REQUIRE_NE(cube_node->animation_translation, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(ch.keyframe_count(), 3);

    // Time codes 0, 30, 60 at 30 fps => 0s, 1s, 2s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));
    CHECK_EQ(ch.times[2], doctest::Approx(2.f));

    // Values: (0,0,0), (3,0,0), (6,0,0).
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[3], doctest::Approx(3.f).epsilon(1e-4));
    CHECK_EQ(ch.values[6], doctest::Approx(6.f).epsilon(1e-4));

    CHECK_EQ(scene->animation.duration(), doctest::Approx(2.f));
}

TEST_SUITE_END();
