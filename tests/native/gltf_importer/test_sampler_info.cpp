// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"

#include <sgl/core/platform.h>

using namespace falcor;

TEST_CASE("GltfImporter - Sampler Information")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "TextureTransformTest" / "glTF"
        / "TextureTransformTest.gltf";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);
        REQUIRE_GT(scene->textures.size(), 0);

        // Check that sampler information is properly loaded
        for (const auto& texture : scene->textures) {
            // Verify that sampler properties have valid values
            CHECK(
                (texture.mag_filter == TextureFilterMode::nearest || texture.mag_filter == TextureFilterMode::linear)
            );
            CHECK(
                (texture.min_filter == TextureFilterMode::nearest || texture.min_filter == TextureFilterMode::linear)
            );
            CHECK(
                (texture.wrap_s == TextureWrapMode::repeat || texture.wrap_s == TextureWrapMode::mirror_repeat
                 || texture.wrap_s == TextureWrapMode::clamp_to_edge)
            );
            CHECK(
                (texture.wrap_t == TextureWrapMode::repeat || texture.wrap_t == TextureWrapMode::mirror_repeat
                 || texture.wrap_t == TextureWrapMode::clamp_to_edge)
            );
        }

        // The TextureTransformTest file should have textures with different wrap modes
        bool has_clamp = false;
        bool has_repeat = false;

        for (const auto& texture : scene->textures) {
            if (texture.wrap_s == TextureWrapMode::clamp_to_edge) {
                has_clamp = true;
            }
            if (texture.wrap_s == TextureWrapMode::repeat) {
                has_repeat = true;
            }
        }

        CHECK(has_clamp);  // Should have at least one texture with clamp
        CHECK(has_repeat); // Should have at least one texture with repeat
    }
}

TEST_CASE("GltfImporter - Default Sampler Values")
{
    // Test that ImporterTexture has sensible default values
    ImporterTexture texture;

    CHECK(texture.mag_filter == TextureFilterMode::linear);
    CHECK(texture.min_filter == TextureFilterMode::linear);
    CHECK(texture.wrap_s == TextureWrapMode::repeat);
    CHECK(texture.wrap_t == TextureWrapMode::repeat);
}
