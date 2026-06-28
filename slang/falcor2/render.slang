// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module render;

__include "render/bsdf.slang";
__include "render/camera.slang";
__include "render/emissive_geometry_sampler.slang";
__include "render/emissive_geometry_types.slang";
__include "render/emissive_geometry.slang";
__include "render/hit_info.slang";
__include "render/interaction.slang";
__include "render/geomprop_provider.slang";
__include "render/intersector.slang";
__include "render/light_sampler.slang";
__include "render/light_system.slang";
__include "render/light.slang";
__include "render/material_system.slang";
__include "render/material.slang";
__include "render/phase_function.slang";
__include "render/records.slang";
__include "render/scene_types.slang";
__include "render/scene.slang";
__include "render/slang_compat.slang";
__include "render/texture_manager_types.slang";
__include "render/texture_manager.slang";

__include "render/bsdfs/anisotropicggx.slang";
__include "render/bsdfs/beer_btdf.slang";
__include "render/bsdfs/dielectric_plate_bsdf.slang";
__include "render/bsdfs/diffuse_specular_brdf.slang";
__include "render/bsdfs/disney_diffuse_brdf.slang";
__include "render/bsdfs/fresnel.slang";
__include "render/bsdfs/generalized_schlick_bsdf.slang";
__include "render/bsdfs/lambert_diffuse_brdf.slang";
__include "render/bsdfs/lambert_diffuse_btdf.slang";
__include "render/bsdfs/oren_nayar.slang";
__include "render/bsdfs/pbrt_conductor_bsdf.slang";
__include "render/bsdfs/pbrt_dielectric_bsdf.slang";
__include "render/bsdfs/proceduralcloth_bsdf.slang";
__include "render/bsdfs/scratch_conductor_brdf.slang";
__include "render/bsdfs/sheen_bsdf.slang";
__include "render/bsdfs/simple_btdf.slang";
__include "render/bsdfs/specular_microfacet.slang";
__include "render/bsdfs/standard_bsdf.slang";

__include "render/geometry/lss_geometry_types.slang";
__include "render/geometry/lss_geometry.slang";
__include "render/geometry/triangle_geometry_types.slang";
__include "render/geometry/triangle_geometry.slang";

__include "render/lights/constant_light.slang";
__include "render/lights/distant_light.slang";
__include "render/lights/env_map_light.slang";
__include "render/lights/point_light.slang";
__include "render/lights/spot_light.slang";

__include "render/materials/invalid_material.slang";
__include "render/materials/openpbr_material.slang";
__include "render/materials/standard_material_common.slang";
__include "render/materials/standard_material_instance.slang";
__include "render/materials/standard_material.slang";
__include "render/materials/standard_specgloss_material.slang";

__include "render/phase_functions/dual_henyey_greenstein_phase_function.slang";
__include "render/phase_functions/henyey_greenstein_phase_function.slang";
__include "render/phase_functions/isotropic_phase_function.slang";
__include "render/phase_functions/null_phase_function.slang";
