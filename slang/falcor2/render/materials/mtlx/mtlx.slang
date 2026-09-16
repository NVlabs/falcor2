// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module mtlx;

__include falcor2.render.materials.mtlx.math;
__include falcor2.render.materials.mtlx.luts;
__include falcor2.render.materials.mtlx.interface;
__include falcor2.render.materials.mtlx.fresnel_airy;
__include falcor2.render.materials.mtlx.fresnel;
__include falcor2.render.materials.mtlx.microfacet_tools;
__include falcor2.render.materials.mtlx.compensation;
__include falcor2.render.materials.mtlx.scatter_mode;
__include falcor2.render.materials.mtlx.bsdf_wrapper;
__include falcor2.render.materials.mtlx.extra_bsdf_properties;
__include falcor2.render.materials.mtlx.oren_nayar_diffuse;
__include falcor2.render.materials.mtlx.burley_diffuse;
__include falcor2.render.materials.mtlx.translucent;
__include falcor2.render.materials.mtlx.subsurface_diffuse_fallback;
__include falcor2.render.materials.mtlx.conductor;
__include falcor2.render.materials.mtlx.dielectric;
__include falcor2.render.materials.mtlx.generalized_schlick;
__include falcor2.render.materials.mtlx.sheen;
__include falcor2.render.materials.mtlx.beer;
__include falcor2.render.materials.mtlx.chiang_hair;
__include falcor2.render.materials.mtlx.flat_layering;
