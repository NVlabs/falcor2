// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module mx139;

__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.math;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.luts;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.bump_frame_validators;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.interface;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.mx_common.mx_bump_hemisphere;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.fresnel_airy;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.fresnel;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.microfacet_tools;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.compensation;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.scatter_mode;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.bsdf_wrapper;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.extra_bsdf_properties;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.oren_nayar_diffuse;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.burley_diffuse;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.translucent;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.subsurface_diffuse_fallback;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.conductor;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.dielectric;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.generalized_schlick;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.sheen;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.beer;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.chiang_hair;
__include falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.flat_layering;
