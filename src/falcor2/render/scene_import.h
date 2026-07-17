// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fwd.h"

#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/render/scene_options.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>

#include <filesystem>
#include <optional>

namespace falcor {

/// Load an importer scene.
/// @param scene Scene to populate.
/// @param importer_scene Importer scene to load from.
FALCOR_API void load_importer_scene(Scene* scene, const ImporterScene& importer_scene);

/// Load a scene from disk.
/// @param scene Scene to populate.
/// @param path Path to scene file.
FALCOR_API void load_scene(Scene* scene, const std::filesystem::path& path, bool recompute_normals);

namespace detail {

// Internal implementation helpers for Scene::create(). These are declared here
// for use across render translation units, but are intentionally not exported.

/// Create a scene from disk.
/// @param device Device to use for rendering.
/// @param path Path to scene file.
/// @param recompute_normals If true, recompute normals for all meshes.
/// @param uv_origin Optional target scene texture coordinate origin. If unset, use imported scene convention.
ref<Scene> create_scene(
    ref<sgl::Device> device,
    const std::filesystem::path& path,
    bool recompute_normals = false,
    std::optional<UVOrigin> uv_origin = {}
);

/// Create a scene from an importer scene.
/// @param device Device to use for rendering.
/// @param importer_scene Importer scene to load from.
/// @param uv_origin Optional target scene texture coordinate origin. If unset, use importer_scene.uv_origin.
ref<Scene>
create_scene(ref<sgl::Device> device, const ImporterScene& importer_scene, std::optional<UVOrigin> uv_origin = {});

/// Create a scene from an importer and run its scene-created callbacks.
/// @param device Device to use for rendering.
/// @param importer Importer whose recorded edits and callbacks to apply.
/// @param uv_origin Optional target scene texture coordinate origin. If unset, use importer scene convention.
/// @param add_default_camera_best_view If true, add a best-view camera when the built importer scene has none.
/// @param camera_aspect Aspect ratio used when adding the best-view camera.
ref<Scene> create_scene(
    ref<sgl::Device> device,
    const Importer& importer,
    std::optional<UVOrigin> uv_origin = {},
    bool add_default_camera_best_view = false,
    float camera_aspect = 16.f / 9.f
);

} // namespace detail

} // namespace falcor
