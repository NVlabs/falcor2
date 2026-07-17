// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/falcor2.h"
#include "falcor2/core/config.h"

FALCOR_PY_DECLARE(core_platform);
FALCOR_PY_DECLARE(core_properties);
FALCOR_PY_DECLARE(core_reflected_object);
FALCOR_PY_DECLARE(core_reflection);
FALCOR_PY_DECLARE(core_signal);
FALCOR_PY_DECLARE(core_asset_resolver);

FALCOR_PY_DECLARE(render_static_mesh_geometry);
FALCOR_PY_DECLARE(render_scene);
FALCOR_PY_DECLARE(render_ray_tracing_setup);
FALCOR_PY_DECLARE(render_mdl_discovery);
FALCOR_PY_DECLARE(render_materialx_codegen);
FALCOR_PY_DECLARE(render_texture_manager);
FALCOR_PY_DECLARE(render_transform);

FALCOR_PY_DECLARE(utils_aabb);
FALCOR_PY_DECLARE(utils_managed_buffer);

FALCOR_PY_DECLARE(utils_algorithm_parallel_reduction);
FALCOR_PY_DECLARE(utils_algorithm_prefix_sum);

FALCOR_PY_DECLARE(utils_sampling_distribution_1d);

FALCOR_PY_DECLARE(importers_importer);
FALCOR_PY_DECLARE(importers_importer_types);
FALCOR_PY_DECLARE(importers_gltf_importer);
FALCOR_PY_DECLARE(importers_usd_importer);

FALCOR_PY_DECLARE(denoisers_optix_types);
FALCOR_PY_DECLARE(denoisers_optix_denoiser);
FALCOR_PY_DECLARE(denoisers_ngx_types);
FALCOR_PY_DECLARE(denoisers_ngx);
FALCOR_PY_DECLARE(denoisers_ngx_features);

FALCOR_PY_DECLARE(ui_camera_controller);
FALCOR_PY_DECLARE(ui_scene_editor);
FALCOR_PY_DECLARE(ui_scene_interaction_controller);
FALCOR_PY_DECLARE(ui_scene_picker);
FALCOR_PY_DECLARE(ui_selection_overlay);
FALCOR_PY_DECLARE(ui_widgets);

#if FALCOR_BUILD_TESTS
FALCOR_PY_DECLARE(testing_nanobind_reflector);
#endif

NB_MODULE(falcor2_ext, m_)
{
    FALCOR_UNUSED(m_);

    nb::module_ m = nb::module_::import_("falcor2");
    m.attr("__doc__") = "falcor2";

    falcor::static_init();

    m.attr("FALCOR_VERSION_MAJOR") = FALCOR_VERSION_MAJOR;
    m.attr("FALCOR_VERSION_MINOR") = FALCOR_VERSION_MINOR;
    m.attr("FALCOR_VERSION_PATCH") = FALCOR_VERSION_PATCH;
    m.attr("FALCOR_VERSION") = FALCOR_VERSION;
    m.attr("__version__") = FALCOR_VERSION;
    m.attr("FALCOR_GIT_VERSION") = FALCOR_GIT_VERSION;
    m.attr("FALCOR_BUILD_TYPE") = FALCOR_BUILD_TYPE;

    FALCOR_PY_IMPORT(utils_aabb);
    FALCOR_PY_IMPORT(utils_managed_buffer);
    FALCOR_PY_IMPORT(utils_algorithm_parallel_reduction);
    FALCOR_PY_IMPORT(utils_algorithm_prefix_sum);
    FALCOR_PY_IMPORT(utils_sampling_distribution_1d);

    FALCOR_PY_IMPORT(importers_importer_types);
    FALCOR_PY_IMPORT(importers_importer);
    FALCOR_PY_IMPORT(importers_gltf_importer);
    FALCOR_PY_IMPORT(importers_usd_importer);

    FALCOR_PY_IMPORT(core_asset_resolver);

    FALCOR_PY_IMPORT(denoisers_optix_types);
    FALCOR_PY_IMPORT(denoisers_optix_denoiser);
    FALCOR_PY_IMPORT(denoisers_ngx_types);
    FALCOR_PY_IMPORT(denoisers_ngx);
    FALCOR_PY_IMPORT(denoisers_ngx_features);

    FALCOR_PY_IMPORT(core_platform);
    FALCOR_PY_IMPORT(core_properties);
    FALCOR_PY_IMPORT(core_reflected_object);
    FALCOR_PY_IMPORT(core_reflection);
    FALCOR_PY_IMPORT(core_signal);

    FALCOR_PY_IMPORT(render_scene);
    FALCOR_PY_IMPORT(render_ray_tracing_setup);
    FALCOR_PY_IMPORT(render_mdl_discovery);
    FALCOR_PY_IMPORT(render_static_mesh_geometry);
    FALCOR_PY_IMPORT(render_materialx_codegen);
    FALCOR_PY_IMPORT(render_texture_manager);
    FALCOR_PY_IMPORT(render_transform);

    FALCOR_PY_IMPORT(ui_camera_controller);
    FALCOR_PY_IMPORT(ui_scene_editor);
    FALCOR_PY_IMPORT(ui_scene_interaction_controller);
    FALCOR_PY_IMPORT(ui_scene_picker);
    FALCOR_PY_IMPORT(ui_selection_overlay);
    FALCOR_PY_IMPORT(ui_widgets);

#if FALCOR_BUILD_TESTS
    FALCOR_PY_IMPORT(testing_nanobind_reflector);
#endif

    nanobind_falcor2_ext_module.m_free = [](void*)
    {
        falcor::static_shutdown();
    };
}
