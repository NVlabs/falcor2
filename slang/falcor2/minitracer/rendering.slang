// SPDX-License-Identifier: Apache-2.0

module rendering;
import slangpy;
import scene;
import falcor2.utils;

__include "rendering/records.slang";

__include "rendering/lighting/emissive_triangle_sampler_utils.slang";
__include "rendering/lighting/env_map_sampler.slang";
__include "rendering/lighting/scene_light_sampler.slang";
__include "rendering/lighting/uniform_emissive_triangle_sampler.slang";

__include "rendering/materials/diffuse_brdf.slang";
__include "rendering/materials/diffuse_specular_brdf.slang";
