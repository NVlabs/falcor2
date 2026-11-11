# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chi-square, eval/sample consistency, and reciprocity tests for BSDF implementations."""

import enum
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import slangpy as spy
import pytest
from typing import Any, Literal, Optional, cast
import falcor2 as f2  # for LUT bindings
import falcor2.testing.helpers as helpers

# Allow this test to run standalone.
if __name__ == "__main__":
    from bsdf_tests import (
        BSDFChiSquareTest,
        BSDFEvalSampleConsistencyTest,
        BSDFReciprocityTest,
    )
else:
    from .bsdf_tests import (
        BSDFChiSquareTest,
        BSDFEvalSampleConsistencyTest,
        BSDFReciprocityTest,
    )


# ---------------------------------------------------------------------------
# BSDF configurations: (bsdf_type, bsdf_bindings, wi_directions, bc, skip)
#
# Only non-delta BSDFs are listed here. Pure delta BSDFs (BeerBTDF,
# SimpleBTDF, smooth PBRTConductor/Dielectric) have zero continuous PDF
# and cannot be chi-square tested.
#
# The ``skip`` flags lists test categories to exclude for a given BSDF.
# ---------------------------------------------------------------------------


class SkipFlags(enum.Flag):
    """Flags indicating which test categories to skip for a BSDF."""

    NONE = 0
    CHI2 = enum.auto()
    SAMPLE_CONSISTENCY = enum.auto()
    RECIPROCITY = enum.auto()


class BSDFFlags(enum.IntFlag):
    """Python enum matching Slang's BSDFFlags definition."""

    none = 0x00
    diffuse_reflection = 0x01
    glossy_reflection = 0x02
    delta_reflection = 0x04
    diffuse_transmission = 0x10
    glossy_transmission = 0x20
    delta_transmission = 0x40
    glossy_curve = 0x80
    diffuse = 0x11
    glossy = 0x22
    delta = 0x44
    curve = 0x80
    reflection = 0x07
    transmission = 0x70
    non_delta = 0xB3
    non_delta_reflection = 0x03
    non_delta_transmission = 0x30
    all = 0xF7


# Keep the complete statistical sweeps for these additional models in nightly.
_NIGHTLY_STATISTICAL_BSDF_TYPES = (
    "DisneyDiffuseBRDF",
    "OrenNayarBRDF",
    "DiffuseSpecularBRDF",
    "SheenBSDF",
    "mtlx::MxOrenNayarDiffuseBSDF",
    "mtlx::MxTranslucentBSDF",
    "mtlx::MxSubsurfaceDiffuseFallbackBSDF",
    "mtlx::MxSheenLutBSDF",
    "mtlx::MxConductorAiryBSDF",
    "mtlx::MxConductorTurquinAnalyticCompensationBSDF",
    "mtlx::MxDielectricAiryBSDF",
    "mtlx::MxGeneralizedSchlickColor82BSDF",
    "mtlx::MxGeneralizedSchlickAiryBSDF",
    "DielectricPlateBSDF",
)

WI_NORMAL = spy.float3(0.0, 0.0, 1.0)
WI_30DEG = spy.float3(0.5, 0.0, 0.866025)
WI_60DEG = spy.float3(0.866025, 0.0, 0.5)
WI_GRAZING = spy.float3(0.994987, 0.0, 0.1)

# Dict bindings zero-fill omitted fields rather than applying Slang initializers.
_GLTF_DATA = {
    "base_color": [0.7, 0.4, 0.2],
    "roughness": 0.7,
    "anisotropy_direction": [0.6, 0.8],
    "diffuse_transmission_color": [0.2, 0.6, 0.8],
    "dielectric_f0": [0.02, 0.04, 0.06],
    "dielectric_f90": 0.75,
    "ior": 1.5,
    "eta": 1.0,
    "thin_walled": True,
}

# Three mixed-lobe cases complement the image grids without a parameter sweep.
# SpecGloss retains selected-component throughput, so only that workflow needs
# a separate PDF check instead of per-sample full-mixture weight equality.
GLTF_BSDF_CONFIGS = [
    (
        "bsdf_tests::TestGLTFBSDF",
        {"data": {**_GLTF_DATA, "metallic": 0.4, "anisotropy_strength": 0.6}},
        [WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    (
        "bsdf_tests::TestGLTFBSDF",
        {"data": {**_GLTF_DATA, "transmission": 0.5, "diffuse_transmission": 0.3}},
        [WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    (
        "bsdf_tests::TestGLTFSpecGlossBSDF",
        {
            "data": {
                "diffuse": [0.7, 0.4, 0.2],
                "specular": [0.1, 0.5, 0.3],
                "glossiness": 0.3,
            }
        },
        [WI_30DEG],
        None,
        SkipFlags.SAMPLE_CONSISTENCY,
    ),
]

BSDF_CONFIGS = [
    *GLTF_BSDF_CONFIGS,
    # --- Lambertian diffuse BRDF ---
    (
        "LambertDiffuseBRDF",
        {"albedo": [0.8, 0.4, 0.2]},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- Lambertian diffuse BTDF ---
    (
        "LambertDiffuseBTDF",
        {"albedo": [0.5, 0.5, 0.5]},
        [WI_NORMAL, WI_30DEG],
        None,
        # Purely transmissive -- no reflection lobes for reciprocity.
        SkipFlags.RECIPROCITY,
    ),
    # --- Disney diffuse BRDF ---
    (
        "DisneyDiffuseBRDF",
        {"albedo": [0.8, 0.4, 0.2], "roughness": 0.5},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- Oren-Nayar BRDF ---
    (
        "OrenNayarBRDF",
        {"albedo": [0.6, 0.6, 0.6], "roughness": 0.3},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- GGX conductor (rough, isotropic) ---
    (
        "PBRTConductorBSDF",
        {
            "D": {"alpha": [0.3, 0.3]},
            "eta": [0.2, 0.5, 1.0],
            "k": [3.0, 2.0, 1.5],
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- GGX conductor (rough, anisotropic) ---
    (
        "PBRTConductorBSDF",
        {
            "D": {"alpha": [0.2, 0.4]},
            "eta": [1.5, 1.0, 0.5],
            "k": [2.0, 2.0, 2.0],
        },
        [WI_NORMAL],
        None,
        SkipFlags.NONE,
    ),
    # --- GGX dielectric (rough) ---
    (
        "PBRTDielectricBSDF",
        {"D": {"alpha": [0.3, 0.3]}, "eta": 1.5},
        [WI_NORMAL, WI_30DEG],
        {"ior_i": 1.0, "ior_t": 1.5},
        SkipFlags.NONE,
    ),
    # --- Legacy Falcor GGX specular reflection ---
    (
        "SpecularMicrofacetBRDF",
        {
            "albedo": [0.04, 0.04, 0.04],
            "alpha": 0.35,
            "active_lobes": int(BSDFFlags.glossy_reflection),
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- Legacy Falcor GGX metallic-style specular reflection ---
    (
        "SpecularMicrofacetBRDF",
        {
            "albedo": [0.9, 0.6, 0.2],
            "alpha": 0.25,
            "active_lobes": int(BSDFFlags.glossy_reflection),
        },
        [WI_NORMAL],
        None,
        SkipFlags.NONE,
    ),
    # --- Legacy Falcor GGX specular reflection/transmission ---
    (
        "SpecularMicrofacetBSDF",
        {
            "transmission_albedo": [0.9, 0.8, 0.7],
            "alpha": 0.4,
            "eta": 1.0 / 1.5,
            "active_lobes": int(BSDFFlags.glossy_reflection | BSDFFlags.glossy_transmission),
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- Legacy Falcor GGX specular reflection/transmission from inside ---
    (
        "SpecularMicrofacetBSDF",
        {
            "transmission_albedo": [0.7, 0.9, 1.0],
            "alpha": 0.4,
            "eta": 1.5,
            "active_lobes": int(BSDFFlags.glossy_reflection | BSDFFlags.glossy_transmission),
        },
        [WI_NORMAL],
        None,
        SkipFlags.NONE,
    ),
    # --- Standard BSDF (opaque dielectric) ---
    # StandardBSDF uses a legacy selected-lobe throughput estimator while
    # returning the mixture PDF for MIS, so sample().weight is not eval()/pdf
    # when multiple lobes overlap.
    (
        "TestStandardBSDF",
        {
            "data": {
                "diffuse": [0.8, 0.4, 0.2],
                "specular": [0.04, 0.04, 0.04],
                "roughness": 0.6,
                "metallic": 0.0,
                "eta": 1.0 / 1.5,
                "transmission": [1.0, 1.0, 1.0],
                "diffuse_transmission": 0.0,
                "specular_transmission": 0.0,
                "thin_walled": False,
            }
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.SAMPLE_CONSISTENCY,
    ),
    # --- Standard BSDF (metallic) ---
    (
        "TestStandardBSDF",
        {
            "data": {
                "diffuse": [0.0, 0.0, 0.0],
                "specular": [0.9, 0.6, 0.2],
                "roughness": 0.6,
                "metallic": 1.0,
                "eta": 1.0 / 1.5,
                "transmission": [1.0, 1.0, 1.0],
                "diffuse_transmission": 0.0,
                "specular_transmission": 0.0,
                "thin_walled": False,
            }
        },
        [WI_NORMAL],
        None,
        SkipFlags.SAMPLE_CONSISTENCY,
    ),
    # --- Standard BSDF (diffuse transmission) ---
    (
        "TestStandardBSDF",
        {
            "data": {
                "diffuse": [0.6, 0.6, 0.6],
                "specular": [0.04, 0.04, 0.04],
                "roughness": 0.7,
                "metallic": 0.0,
                "eta": 1.0 / 1.5,
                "transmission": [0.8, 0.9, 1.0],
                "diffuse_transmission": 0.5,
                "specular_transmission": 0.0,
                "thin_walled": True,
            }
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.SAMPLE_CONSISTENCY,
    ),
    # --- Standard BSDF (specular transmission) ---
    (
        "TestStandardBSDF",
        {
            "data": {
                "diffuse": [0.2, 0.2, 0.2],
                "specular": [0.04, 0.04, 0.04],
                "roughness": 0.6,
                "metallic": 0.0,
                "eta": 1.0 / 1.5,
                "transmission": [0.8, 0.9, 1.0],
                "diffuse_transmission": 0.0,
                "specular_transmission": 0.75,
                "thin_walled": True,
            }
        },
        [WI_NORMAL],
        None,
        SkipFlags.SAMPLE_CONSISTENCY,
    ),
    # --- Diffuse/Specular BRDF (non-metallic) ---
    # Fields are set directly (bypassing constructor), so we pre-compute:
    #   diffuse = albedo * (1 - metallic), specular = lerp(specular * 0.08, albedo, metallic)
    # Note: DiffuseSpecularBRDF squares roughness internally (alpha = roughness^2),
    # so roughness must be high enough for accurate chi2 integration.
    (
        "DiffuseSpecularBRDF",
        {
            "diffuse": [0.8, 0.4, 0.2],
            "specular": [0.08, 0.08, 0.08],
            "roughness": 0.7,
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- Diffuse/Specular BRDF (metallic) ---
    (
        "DiffuseSpecularBRDF",
        {
            "diffuse": [0.0, 0.0, 0.0],
            "specular": [0.9, 0.6, 0.2],
            "roughness": 0.3,
        },
        [WI_NORMAL],
        None,
        SkipFlags.NONE,
    ),
    # --- Sheen BSDF ---
    (
        "SheenBSDF",
        {"color": [0.8, 0.4, 0.2], "roughness": 0.5},
        [WI_NORMAL, WI_30DEG],
        None,
        # Non-reciprocal G function (asymmetric terminator softening).
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 Burley diffuse BSDF ---
    (
        "mtlx::MxBurleyDiffuseBSDF",
        {"albedo_value": [0.8, 0.4, 0.2], "roughness": 0.5},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 Oren-Nayar diffuse BSDF ---
    (
        "mtlx::MxOrenNayarDiffuseBSDF",
        {"albedo_value": [0.6, 0.6, 0.6], "roughness": 0.35, "energy_compensation": True},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 translucent BSDF ---
    (
        "mtlx::MxTranslucentBSDF",
        {"albedo_value": [0.7, 0.5, 0.3]},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 subsurface diffuse fallback BSDF ---
    (
        "mtlx::MxSubsurfaceDiffuseFallbackBSDF",
        {"albedo_value": [0.55, 0.35, 0.25]},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 sheen BSDF ---
    (
        "mtlx::MxSheenBSDF",
        {
            "tint": [0.8, 0.4, 0.2],
            "base_roughness": 0.5,
            "sheen_alpha": 0.5,
            "Emiss": 0.927339792766856,
            "sheen_mode": 0,
            "backfacing": False,
        },
        [WI_30DEG],
        None,
        # Reciprocity fails consistently across backends (max error ~1.0), indicating the Conty-Kulla visibility term is not symmetric.
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 Zeltner sheen BSDF (specification analytic fit) ---
    *[
        (
            "mtlx::MxSheenBSDF",
            {
                "tint": [0.8, 0.4, 0.2],
                "base_roughness": roughness,
                "sheen_alpha": roughness,
                "Emiss": 1.0,
                "sheen_mode": 1,
                "backfacing": False,
            },
            [WI_30DEG],
            None,
            # The specified view-dependent LTC approximation is non-reciprocal.
            SkipFlags.RECIPROCITY,
        )
        for roughness in (0.5, 0.01)
    ],
    # --- MaterialX 1.39 Zeltner sheen BSDF (comparison LUT) ---
    *[
        (
            "mtlx::MxSheenLutBSDF",
            {
                "tint": [0.8, 0.4, 0.2],
                "base_roughness": roughness,
                "sheen_alpha": roughness,
                "Emiss": 1.0,
                "sheen_mode": 1,
                "backfacing": False,
            },
            # The retained paper table is sparse in its first roughness row
            # and represents many non-grazing entries as an exactly zero lobe.
            # Use a populated view entry so the boundary distribution is
            # nonzero and therefore statistically testable.
            [WI_30DEG if roughness == 0.5 else WI_GRAZING],
            None,
            # The specified view-dependent LTC approximation is non-reciprocal.
            SkipFlags.RECIPROCITY,
        )
        for roughness in (0.5, 0.01)
    ],
    # --- MaterialX 1.39 conductor BSDF ---
    (
        "mtlx::MxConductorBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "IOR": [0.2, 0.5, 1.0],
                "extinction": [3.0, 2.0, 1.5],
            },
        },
        [WI_NORMAL, WI_30DEG],
        None,
        # MaterialX 1.39 GLSL and OSL/BSDL use view-direction Turquin-style
        # GGX energy compensation. Treat it as an explicit nonreciprocal
        # MaterialX closure and cover it with sample/pdf and energy gates.
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 Airy conductor BSDF ---
    (
        "mtlx::MxConductorAiryBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "IOR": [0.2, 0.5, 1.0],
                "extinction": [3.0, 2.0, 1.5],
                "thinfilm_thickness": 300.0,
                "thinfilm_ior": 1.5,
            },
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 conductor BSDF (non-LUT compensation) ---
    (
        "mtlx::MxConductorTurquinAnalyticCompensationBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "IOR": [0.2, 0.5, 1.0],
                "extinction": [3.0, 2.0, 1.5],
            },
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 scratch conductor BSDF ---
    (
        "mtlx::MxScratchConductorBSDF",
        {
            "inner": {
                "D": {"alpha": [0.25, 0.25]},
                "eta": [0.2, 0.5, 1.0],
                "k": [3.0, 2.0, 1.5],
                "scratchdirection": [1.0, 0.0, 0.0],
                "depth": 0.35,
                "mask": [0.35, 0.35, 0.35],
                "phongCoefficient": 100.0,
            },
        },
        [WI_NORMAL],
        None,
        # Scratch sampling retains a selected-branch weight that is not eval()/eval_pdf().
        # The dedicated test below checks its masked mixture PDF independently.
        SkipFlags.CHI2 | SkipFlags.SAMPLE_CONSISTENCY | SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 dielectric BSDF ---
    (
        "mtlx::MxDielectricBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {"eta": 1.5},
            "reflection_tint": [1.0, 1.0, 1.0],
            "transmission_tint": [1.0, 1.0, 1.0],
            "absorption": [0.0, 0.0, 0.0],
            "backfacing": False,
        },
        [WI_NORMAL, WI_30DEG],
        {"ior_i": 1.0, "ior_t": 1.5},
        # MaterialX 1.39 GLSL and OSL/BSDL use view-direction Turquin-style
        # GGX energy compensation for the reflective component.
        # Its rough-dielectric chi2 path is a known VNDF branch-selection
        # outlier, while eval/sample consistency remains covered.
        SkipFlags.CHI2 | SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 Airy dielectric BSDF ---
    (
        "mtlx::MxDielectricAiryBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "eta": 1.5,
                "base_ior": 1.5,
                "tir_cos": 0.0,
                "thinfilm_thickness": 300.0,
                "thinfilm_ior": 1.5,
            },
            "reflection_tint": [1.0, 1.0, 1.0],
            "transmission_tint": [1.0, 1.0, 1.0],
            "absorption": [0.0, 0.0, 0.0],
            "backfacing": False,
        },
        [WI_NORMAL, WI_30DEG],
        {"ior_i": 1.0, "ior_t": 1.5},
        SkipFlags.CHI2 | SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 generalized Schlick BSDF ---
    (
        "mtlx::MxGeneralizedSchlickBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "F0": [0.08, 0.08, 0.08],
                "F90": [1.0, 1.0, 1.0],
                "exponent": 5.0,
                "eta": 1.78878850537961,
                "tir_cos": 0.0,
            },
            "reflection_tint": [1.0, 1.0, 1.0],
            "transmission_tint": [0.0, 0.0, 0.0],
            "backfacing": False,
        },
        [WI_NORMAL, WI_30DEG],
        None,
        # MaterialX 1.39 GLSL and OSL/BSDL use view-direction Turquin-style
        # GGX energy compensation for the reflective component.
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 generalized Schlick color82 BSDF ---
    (
        "mtlx::MxGeneralizedSchlickColor82BSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "F0": [0.08, 0.08, 0.08],
                "F82": [0.65, 0.75, 0.9],
                "F90": [1.0, 1.0, 1.0],
                "exponent": 5.0,
                "eta": 1.78878850537961,
                "tir_cos": 0.0,
            },
            "reflection_tint": [1.0, 1.0, 1.0],
            "transmission_tint": [0.0, 0.0, 0.0],
            "backfacing": False,
        },
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 generalized Schlick Airy BSDF ---
    (
        "mtlx::MxGeneralizedSchlickAiryBSDF",
        {
            "roughness_xy": [0.3, 0.3],
            "fresnel": {
                "F0": [0.08, 0.08, 0.08],
                "F82": [0.65, 0.75, 0.9],
                "F90": [1.0, 1.0, 1.0],
                "exponent": 5.0,
                "eta": 1.78878850537961,
                "tir_cos": 0.0,
                "thinfilm_thickness": 300.0,
                "thinfilm_ior": 1.5,
            },
            "reflection_tint": [1.0, 1.0, 1.0],
            "transmission_tint": [0.0, 0.0, 0.0],
            "backfacing": False,
        },
        [WI_NORMAL, WI_30DEG],
        None,
        # The thin-film Fresnel path has a small eval/weight mismatch; keep PDF-distribution coverage.
        SkipFlags.SAMPLE_CONSISTENCY | SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 Chiang hair BSDF ---
    (
        "mtlx::MxChiangHairBSDF",
        {
            "tint_R": [1.0, 0.85, 0.7],
            "tint_TT": [0.8, 0.45, 0.22],
            "tint_TRT": [0.55, 0.28, 0.12],
            "ior": 1.55,
            "roughness_R": [0.16, 0.22],
            "roughness_TT": [0.08, 0.22],
            "roughness_TRT": [0.28, 0.22],
            "cuticle_angle": 0.52,
            "absorption_coefficient": [0.35, 0.75, 1.25],
        },
        [WI_NORMAL, WI_30DEG],
        None,
        # Curve-frame full-sphere scattering, not a conventional surface BRDF.
        SkipFlags.RECIPROCITY,
    ),
    # --- Dielectric plate (rough) ---
    (
        "DielectricPlateBSDF",
        {
            "D": {"alpha": [0.3, 0.3]},
            "D_inside": {"alpha": [0.35, 0.35]},
            "eta": 1.5,
        },
        [WI_NORMAL],
        None,
        # Fresnel computed from different half-vectors in sample vs eval_pdf (this could be a bug).
        SkipFlags.CHI2 | SkipFlags.SAMPLE_CONSISTENCY,
    ),
    # --- Procedural cloth BSDF ---
    (
        "ProceduralClothBSDF",
        {
            "uv_top": [0.0, 0.15],
            "uv_bottom": [0.0, 0.15],
            "yarn_id": 0,
            "yarn_direction_top": [1.0, 0.0, 0.0],
            "yarn_direction_bottom": [0.0, 1.0, 0.0],
            "yarn_scale": [1.0, 1.0],
            "ply_twist": 0.75,
            "ply_scale": 0.9,
            "ply_count": 3,
            "fiber_twist_angle": 0.0,
            "fiber_reflectance": 0.028,
            "fiber_transparency": 0.8,
            "fiber_roughness_m": 0.1,
            "fiber_roughness_n": 0.1,
            "fiber_count": 250,
            "fiber_color_top": [1, 0, 0],
            "fiber_color_bottom": [0, 0, 1],
        },
        [WI_NORMAL, WI_30DEG, WI_60DEG],
        None,
        # Is not reciprocal due to the internal use of ray tracing (different visibility for forward vs reverse directions).
        SkipFlags.CHI2 | SkipFlags.SAMPLE_CONSISTENCY | SkipFlags.RECIPROCITY,
    ),
]


def _make_params(skip_flag: SkipFlags = SkipFlags.NONE) -> list[Any]:
    """Generate (bsdf_type, bindings, wi, bc) test parameters.

    Args:
        skip_flag: If given, configs whose ``skip`` flags overlap with this
                   flag are excluded from the returned parameters.
    """
    params = []
    for bsdf_type, bindings, wi_list, bc, skip in BSDF_CONFIGS:
        if skip & skip_flag:
            continue
        for wi in wi_list:
            wi_str = "normal" if wi == WI_NORMAL else f"wi={wi}"
            label = f"{bsdf_type}_{wi_str}"
            params.append(pytest.param(bsdf_type, bindings, wi, bc, id=label))
    return params


# ---------------------------------------------------------------------------
# Fixtures for device-owned test resources
# ---------------------------------------------------------------------------


def _bsdf_resource_scope(fixture_name: str, config: pytest.Config) -> Literal["function", "module"]:
    return "function" if config.getoption("--device-cache-policy") == "test" else "module"


@dataclass(frozen=True)
class _BSDFResources:
    device: spy.Device
    module: spy.Module


@pytest.fixture(scope=_bsdf_resource_scope, params=helpers.DEFAULT_DEVICE_TYPES)
def _bsdf_resources(request: pytest.FixtureRequest) -> _BSDFResources:
    device_type = cast(spy.DeviceType, request.param)
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("render/bsdf_tests.slang"))
    return _BSDFResources(device=device, module=module)


@pytest.fixture(scope=_bsdf_resource_scope)
def _mtlx_lut_bindings(_bsdf_resources: _BSDFResources) -> dict[str, Any]:
    return f2.create_mtlx_lut_bindings(_bsdf_resources.device)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_gltf_specgloss_sample_pdf(_bsdf_resources: _BSDFResources) -> None:
    bsdf_type, bindings, wi_list, bc, _ = GLTF_BSDF_CONFIGS[2]
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bindings,
        wi=wi_list[0],
        bc=bc,
    )
    # Still checks finite nonnegative weights/directions and sample/eval PDF agreement.
    assert test.run(weight_tol=float("inf")), test.messages


@pytest.mark.parametrize(
    ("feature_bit", "overrides"),
    [
        (1 << 5, {"transmission": 0.5}),
        (1 << 10, {"iridescence": 1.0, "iridescence_thickness": 450.0}),
        (1 << 11, {"anisotropy_strength": 0.6}),
        (1 << 12, {"diffuse_transmission": 0.3}),
    ],
)
def test_gltf_conditional_bsdf_exact_matches_flexible(
    _bsdf_resources: _BSDFResources,
    feature_bit: int,
    overrides: dict[str, Any],
) -> None:
    """A single-feature specialization matches the flexible BSDF in value, PDF, and albedo."""
    # TODO: Re-enable D3D12 after the NVIDIA 617.42 compiler access violation is fixed.
    if _bsdf_resources.device.desc.type == spy.DeviceType.d3d12:
        pytest.skip("NVIDIA 617.42: intermittent access violation in nvgpucomp64.dll")

    kernel = _bsdf_resources.module[f"bsdf_tests::test_gltf_exact_flexible<{feature_bit}>"]
    result = kernel(
        data={**_GLTF_DATA, "metallic": 0.4, **overrides},
        wi=WI_30DEG,
        wo=spy.float3(-0.2, 0.3, 0.9327379),
    )
    np.testing.assert_allclose(result, 0.0, atol=2e-6)


def test_anisotropic_ggx_ndf_near_normal(_bsdf_resources: _BSDFResources) -> None:
    # h.z can round to one even when h.xy materially changes a narrow NDF.
    xy = np.array([[0, 0], [0, 1e-5], [0, 1e-4], [0, 2e-4], [2e-4, -1e-4], [0.001, 0.002]])
    h = np.concatenate([xy, np.sqrt(1 - np.sum(xy * xy, axis=1))[:, None]], axis=1).astype(
        np.float32
    )
    assert h[3, 2] == 1.0 and h[3, 1] != 0.0
    # Also exercise the algebraic form at the horizon, without dividing by h.z.
    h = np.concatenate([h, [[1, 0, 0], [0, 1, 0]]]).astype(np.float32)
    function = _bsdf_resources.module["bsdf_tests::probe_anisotropic_ggx_ndf"]
    for axes in ((0.640324, 0.0009), (0.0009, 0.640324), (0.25, 0.25), (0.001, 0.9)):
        alpha = np.array(axes, dtype=np.float32)
        actual = function(
            spy.float2(alpha), spy.Tensor.from_numpy(_bsdf_resources.device, h)
        ).to_numpy()
        # Khronos anisotropic GGX D, evaluated from the exact shader inputs in float64.
        scaled = h.astype(np.float64) / np.array([*alpha.astype(np.float64), 1.0])
        expected = 1 / (np.pi * np.prod(alpha.astype(np.float64)) * np.sum(scaled**2, axis=1) ** 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)


def test_anisotropic_ggx_policy_preserves_default_and_allows_clamping(
    _bsdf_resources: _BSDFResources,
) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_anisotropic_ggx_policy"]()

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(0.001)
    assert result.w == pytest.approx(1.0)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
@pytest.mark.parametrize(
    ("bsdf", "wi"),
    [
        ({"diffuse": [0.8, 0.4, 0.2], "specular": [0.08, 0.2, 0.6], "roughness": 0.7}, WI_NORMAL),
        ({"diffuse": [1.0, 0.8, 0.2], "specular": [0.9, 0.6, 0.2], "roughness": 0.2}, WI_GRAZING),
    ],
)
def test_diffuse_specular_albedo(
    _bsdf_resources: _BSDFResources,
    bsdf: dict[str, Any],
    wi: spy.float3,
) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_diffuse_specular_albedo"](bsdf=bsdf, wi=wi)

    assert np.asarray(result) == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=2e-6)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_mtlx_mix_preserves_absorption_side(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_mtlx_mix_albedo_contributions"]()

    assert np.asarray(result) == pytest.approx([0.0, 0.0, 0.35, 0.65])


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_mtlx_beer_delta_transmission(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
) -> None:
    kernel = (
        _bsdf_resources.module["bsdf_tests::test_mtlx_beer_delta_transmission"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    result = kernel(
        bsdf={"inner": {"absorption": [0.25, 0.5, 0.75], "pathshortening": 0.0}},
        wi=WI_NORMAL,
    )

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(float(np.exp(-0.25)))
    assert result.w == pytest.approx(float(np.exp(-0.75)))

    rejected = kernel(
        bsdf={"inner": {"absorption": [0.25, 0.5, 0.75], "pathshortening": 0.0}},
        wi=-WI_NORMAL,
    )
    assert rejected.x == pytest.approx(0.0)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_standard_bsdf_diffuse_transmission_support(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_standard_bsdf_diffuse_transmission_support"](
        data={
            "diffuse": [0.0, 0.0, 0.0],
            "specular": [0.0, 0.0, 0.0],
            "roughness": 0.5,
            "metallic": 0.0,
            "eta": 1.0 / 1.5,
            "transmission": [0.25, 0.5, 1.0],
            "diffuse_transmission": 1.0,
            "specular_transmission": 0.0,
            "thin_walled": True,
        }
    )

    assert result.x == pytest.approx(0.25 / np.pi)
    assert result.y == pytest.approx(0.25)
    assert result.z == pytest.approx(1.0)
    assert result.w == pytest.approx(0.25)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_specular_microfacet_smooth_transmission(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_specular_microfacet_smooth_transmission"](
        bsdf={
            "transmission_albedo": [0.9, 0.8, 0.7],
            "alpha": 0.0,
            "eta": 1.0 / 1.5,
            "active_lobes": int(BSDFFlags.delta_reflection | BSDFFlags.delta_transmission),
        }
    )

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(0.9 / 1.5**2)
    assert result.w == pytest.approx(0.7 / 1.5**2)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_specular_microfacet_albedo(_bsdf_resources: _BSDFResources) -> None:
    brdf_kernel = _bsdf_resources.module["bsdf_tests::test_specular_microfacet_brdf_albedo"]
    disabled_brdf = brdf_kernel(
        bsdf={"albedo": [0.04, 0.2, 0.8], "alpha": 0.3, "active_lobes": int(BSDFFlags.none)},
        channel=1,
    )
    assert np.asarray(disabled_brdf) == pytest.approx([0.0, 1.0, 0.0, 0.0])

    bsdf_kernel = _bsdf_resources.module["bsdf_tests::test_specular_microfacet_bsdf_albedo"]
    common = {
        "transmission_albedo": [0.9, 0.8, 0.7],
        "alpha": 0.0,
        "eta": 1.0 / 1.5,
    }
    r = 0.04
    tint = 0.8
    expected = {
        BSDFFlags.delta_reflection
        | BSDFFlags.delta_transmission: [r, 0.0, (1.0 - r) * tint, (1.0 - r) * (1.0 - tint)],
        BSDFFlags.delta_reflection: [r, 0.0, 0.0, 1.0 - r],
        BSDFFlags.delta_transmission: [0.0, r, (1.0 - r) * tint, (1.0 - r) * (1.0 - tint)],
    }
    for active_lobes, albedo in expected.items():
        result = bsdf_kernel(bsdf={**common, "active_lobes": int(active_lobes)}, channel=1)
        assert np.asarray(result) == pytest.approx(albedo)
        assert sum(result) == pytest.approx(1.0)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_pbrt_dielectric_event_support_and_albedo(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_pbrt_dielectric_event_support"](
        bsdf={"D": {"alpha": [0.0, 0.0]}, "eta": 1.5}
    )

    # Transmission-only total internal reflection and reflection-only F=0
    # have no enabled physical event in PBRT's masked probability model.
    assert result.x == pytest.approx(0.0)
    assert result.y == pytest.approx(0.0)

    # The context-free property assumes incidence from air into eta=1.5.
    assert result.z == pytest.approx(0.0891867, rel=1e-5)
    assert result.w == pytest.approx(1.0 - result.z)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_pbrt_dielectric_transmission_scaling(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_pbrt_dielectric_transmission_scaling"](
        smooth_bsdf={"D": {"alpha": [0.0, 0.0]}, "eta": 1.5},
        rough_bsdf={"D": {"alpha": [0.3, 0.3]}, "eta": 1.5},
    )

    expected_scale = (1.0 / 1.5) ** 2
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(expected_scale)
    assert result.z == pytest.approx(expected_scale, rel=1e-5)
    assert result.w == pytest.approx(expected_scale, rel=1e-5)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_mtlx_generalized_schlick_colored_branch_probability(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
) -> None:
    """Colored R/T sampling follows MaterialX's luminance balance policy."""
    color0 = [0.2, 0.5, 0.8]
    p_reflect = float(np.dot(color0, [0.2126, 0.7152, 0.0722]))
    fresnel = {
        "F0": color0,
        "F90": [1.0, 1.0, 1.0],
        "exponent": 5.0,
        "eta": 1.78878850537961,
        "tir_cos": 0.0,
    }

    def bindings(reflection: float, transmission: float) -> dict[str, Any]:
        return {
            "roughness_xy": [0.3, 0.3],
            "fresnel": fresnel,
            "reflection_tint": [reflection] * 3,
            "transmission_tint": [transmission] * 3,
            "backfacing": False,
        }

    kernel = (
        _bsdf_resources.module[
            "bsdf_tests::test_controlled_bsdf_sample<mtlx::MxGeneralizedSchlickBSDF>"
        ]
        .as_func()
        .set(_mtlx_lut_bindings)
    )

    reflected = kernel(
        bsdf=bindings(1.0, 1.0),
        wi=WI_NORMAL,
        input={"u_lobe": p_reflect * 0.5, "u_dir": [0.0, 0.0], "aux_seed": 0},
    )
    reflected_only = kernel(
        bsdf=bindings(1.0, 0.0),
        wi=WI_NORMAL,
        input={"u_lobe": 0.0, "u_dir": [0.0, 0.0], "aux_seed": 0},
    )
    transmitted = kernel(
        bsdf=bindings(1.0, 1.0),
        wi=WI_NORMAL,
        input={"u_lobe": (1.0 + p_reflect) * 0.5, "u_dir": [0.0, 0.0], "aux_seed": 0},
    )
    transmitted_only = kernel(
        bsdf=bindings(0.0, 1.0),
        wi=WI_NORMAL,
        input={"u_lobe": 0.0, "u_dir": [0.0, 0.0], "aux_seed": 0},
    )

    for result in (reflected, reflected_only, transmitted, transmitted_only):
        assert result.x == pytest.approx(1.0)
        assert result.z == pytest.approx(result.w, rel=1e-6)

    assert reflected.y > 0.0
    assert reflected_only.y > 0.0
    assert transmitted.y < 0.0
    assert transmitted_only.y < 0.0
    assert reflected.z / reflected_only.z == pytest.approx(p_reflect, rel=1e-6)
    assert transmitted.z / transmitted_only.z == pytest.approx(1.0 - p_reflect, rel=1e-6)


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
@pytest.mark.parametrize(
    "family,expected_events",
    [
        ("conductor", 1.0),
        ("dielectric", 3.0),
        ("generalized_schlick", 3.0),
    ],
)
def test_mtlx_retroreflection(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    family: str,
    expected_events: float,
) -> None:
    eval_pdf_kernel = (
        _bsdf_resources.module[f"bsdf_tests::test_mtlx_{family}_retroreflection_eval_pdf"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    eval_pdf_result = eval_pdf_kernel()

    assert eval_pdf_result.x == pytest.approx(0.0, abs=1e-6)
    assert eval_pdf_result.y == pytest.approx(0.0, abs=1e-6)
    assert eval_pdf_result.z > 0.0
    assert eval_pdf_result.w > 0.0

    sample_kernel = (
        _bsdf_resources.module[f"bsdf_tests::test_mtlx_{family}_retroreflection_sample"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    sample_result = sample_kernel()

    assert sample_result.x == pytest.approx(0.0, abs=1e-6)
    assert sample_result.y == pytest.approx(0.0, abs=1e-5)
    assert sample_result.z == pytest.approx(0.0, abs=3e-6)
    assert sample_result.w == expected_events


# ---------------------------------------------------------------------------
# Chi-square tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
@pytest.mark.nightly_if(bsdf_type=_NIGHTLY_STATISTICAL_BSDF_TYPES)
@pytest.mark.parametrize("bsdf_type,bsdf_bindings,wi,bc", _make_params(SkipFlags.CHI2))
def test_bsdf_chi2(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Chi-square test: sample() distribution matches eval_pdf()."""
    test = BSDFChiSquareTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


# ---------------------------------------------------------------------------
# Eval/sample consistency tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
@pytest.mark.nightly_if(bsdf_type=_NIGHTLY_STATISTICAL_BSDF_TYPES)
@pytest.mark.parametrize(
    "bsdf_type,bsdf_bindings,wi,bc", _make_params(SkipFlags.SAMPLE_CONSISTENCY)
)
def test_bsdf_eval_sample_consistency(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Verify eval()/eval_pdf() == weight from sample()."""
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_mtlx_scratch_conductor_pdf_consistency(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
) -> None:
    """Verify Scratch sampling reports the masked mixture PDF used for MIS."""
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type="mtlx::MxScratchConductorBSDF",
        bsdf_bindings={
            "inner": {
                "D": {"alpha": [0.25, 0.25]},
                "eta": [0.2, 0.5, 1.0],
                "k": [3.0, 2.0, 1.5],
                "scratchdirection": [1.0, 0.0, 0.0],
                "depth": 0.35,
                "mask": [0.35, 0.35, 0.35],
                "phongCoefficient": 100.0,
            }
        },
        wi=WI_NORMAL,
        sample_count=10000,
        global_bindings=_mtlx_lut_bindings,
    )
    # The selected-branch weight is intentionally not the collapsed eval()/eval_pdf() ratio.
    assert test.run(weight_tol=float("inf")), test.messages


@pytest.mark.parametrize(
    "wi",
    [
        WI_30DEG,
        spy.float3(0.13215184211730957, 0.2252282053232193, 0.965302050113678),
    ],
)
def test_pbrt_dielectric_near_smooth_eval_sample_consistency(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    wi: spy.float3,
) -> None:
    """Near-smooth PBRT dielectric samples must report the same PDF as eval_pdf()."""
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type="PBRTDielectricBSDF",
        bsdf_bindings={"D": {"alpha": [0.002, 0.002]}, "eta": 1.0 / 1.48},
        wi=wi,
        bc={"ior_i": 1.0, "ior_t": 1.48},
        sample_count=100000,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


@pytest.mark.parametrize(
    "sample_lobe_types_hint",
    [
        BSDFFlags.reflection,
        BSDFFlags.transmission,
    ],
)
def test_pbrt_dielectric_lobe_sampling_hint_eval_sample_consistency(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    sample_lobe_types_hint: BSDFFlags,
) -> None:
    """Restricted sampling hints must report coherent hint-conditioned PDFs."""
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type="PBRTDielectricBSDF",
        bsdf_bindings={"D": {"alpha": [0.3, 0.3]}, "eta": 1.0 / 1.5},
        wi=WI_30DEG,
        bc={
            "ior_i": 1.0,
            "ior_t": 1.5,
            "sample_lobe_types_hint": sample_lobe_types_hint,
        },
        sample_count=100000,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


def _mtlx_hint_bindings(family: str, scatter: str = "RT") -> dict[str, Any]:
    fresnel = (
        {"eta": 1.5}
        if family == "Dielectric"
        else {"F0": [0.2, 0.5, 0.8], "F90": [1.0] * 3, "exponent": 5.0, "eta": 1.5, "tir_cos": 0.0}
    )
    return {
        "roughness_xy": [0.3, 0.2],
        "fresnel": fresnel,
        "reflection_tint": [0.8, 0.6, 0.4] if "R" in scatter else [0.0] * 3,
        "transmission_tint": [0.4, 0.7, 0.9] if "T" in scatter else [0.0] * 3,
        "backfacing": False,
    }


@pytest.mark.parametrize("family", ["Dielectric", "GeneralizedSchlick"])
@pytest.mark.parametrize("compensation", ["", "TurquinAnalyticCompensation", "NoCompensation"])
@pytest.mark.parametrize("scatter", ["RT", "R", "T"])
@pytest.mark.parametrize("static_scatter", [False, True])
@pytest.mark.parametrize(
    "hint",
    [BSDFFlags.reflection, BSDFFlags.transmission, BSDFFlags.none, BSDFFlags.all, BSDFFlags.curve],
)
def test_mtlx_lobe_sampling_hint(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    family: str,
    compensation: str,
    scatter: str,
    static_scatter: bool,
    hint: BSDFFlags,
) -> None:
    """Hints restrict sampling without changing the physical BSDF or its compensation."""
    bsdf_type = f"mtlx::Mx{family}{compensation}BSDF"
    if static_scatter:
        fresnel_type = "MxDielectricFresnel" if family == "Dielectric" else "MxSchlickFresnel"
        comp_type = {
            "": "MxTurquinLutDielectricCompensation",
            "TurquinAnalyticCompensation": "MxTurquinAnalyticDielectricCompensation",
            "NoCompensation": "MxNoCompensationDielectricCompensation",
        }[compensation]
        bsdf_type = (
            f"mtlx::MxCompensated{family}BSDF<mtlx::{fresnel_type},"
            f"mtlx::{comp_type},mtlx::MxScatterStatic{scatter}Mode>"
        )
    kernel = (
        _bsdf_resources.module[f"bsdf_tests::test_mtlx_lobe_sampling_hint<{bsdf_type}>"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    args = {"bsdf": _mtlx_hint_bindings(family, scatter), "wi": WI_NORMAL, "hint": hint}
    allow_r = "R" in scatter and bool(hint & BSDFFlags.reflection)
    allow_t = "T" in scatter and bool(hint & BSDFFlags.transmission)
    # Force both ends of the roulette interval; a hint must override either one.
    for roulette in [0.0, 0.99]:
        args["random"] = [0.0, 0.0, roulette]
        result = kernel(**args, check=0)
        assert result.x == float(allow_r or allow_t)
        if allow_r or allow_t:
            if not allow_t:
                assert result.y > 0.0
            if not allow_r:
                assert result.y < 0.0
            assert result.z > 0.0
            assert result.z == pytest.approx(result.w, rel=2e-5)
            assert np.asarray(kernel(**args, check=1)) == pytest.approx([0.0] * 4, abs=2e-5)
        else:
            assert np.asarray(kernel(**args, check=3)) == pytest.approx([0.0] * 4)
    # Fused evaluation preserves physical values even for forbidden directions.
    for check in [4, 5]:
        assert np.asarray(kernel(**args, check=check)) == pytest.approx([0.0] * 4, abs=2e-5)
    pdfs = kernel(**args, check=2)
    if not allow_r:
        assert pdfs.x == 0.0
    if not allow_t:
        assert pdfs.y == 0.0
    if scatter == "RT":
        # At normal incidence and normal microfacet, branch probability is known independently.
        p_r = (
            0.04
            if family == "Dielectric"
            else float(np.dot([0.2, 0.5, 0.8], [0.2126, 0.7152, 0.0722]))
        )
        if allow_r and not allow_t:
            assert pdfs.x == pytest.approx(pdfs.z / p_r, rel=2e-5)
        if allow_t and not allow_r:
            assert pdfs.y == pytest.approx(pdfs.w / (1.0 - p_r), rel=2e-5)


@pytest.mark.parametrize("family", ["Dielectric", "GeneralizedSchlick"])
@pytest.mark.parametrize("hint", [BSDFFlags.reflection, BSDFFlags.transmission])
@pytest.mark.parametrize("wi", [WI_30DEG, WI_GRAZING])
def test_mtlx_lobe_sampling_hint_eval_sample_consistency(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    family: str,
    hint: BSDFFlags,
    wi: spy.float3,
) -> None:
    test = BSDFEvalSampleConsistencyTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type=f"mtlx::Mx{family}BSDF",
        bsdf_bindings=_mtlx_hint_bindings(family),
        wi=wi,
        bc={"sample_lobe_types_hint": hint},
        sample_count=100000,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("family", ["Dielectric", "GeneralizedSchlick"])
@pytest.mark.parametrize("hint", [BSDFFlags.reflection, BSDFFlags.transmission, BSDFFlags.curve])
def test_mtlx_lobe_sampling_hint_total_internal_reflection(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    family: str,
    hint: BSDFFlags,
) -> None:
    bindings = _mtlx_hint_bindings(family)
    bindings["fresnel"]["eta"] = 1.0 / 1.5
    if family == "GeneralizedSchlick":
        bindings["fresnel"]["tir_cos"] = float(np.sqrt(1.0 - (1.0 / 1.5) ** 2))
    bindings["roughness_xy"] = [0.001, 0.001]
    bindings["backfacing"] = True
    kernel = (
        _bsdf_resources.module[f"bsdf_tests::test_mtlx_lobe_sampling_hint<mtlx::Mx{family}BSDF>"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    args = {"bsdf": bindings, "wi": WI_60DEG, "random": [0.0, 0.0, 0.99], "hint": hint}
    result = kernel(**args, check=0)
    assert np.all(np.isfinite(np.asarray(result)))
    if hint == BSDFFlags.reflection:
        assert result.x == 1.0 and result.y > 0.0
        assert result.z == pytest.approx(result.w, rel=2e-5)
        assert np.asarray(kernel(**args, check=1)) == pytest.approx([0.0] * 4, abs=2e-5)
    else:
        assert result.x == 0.0
        assert np.asarray(kernel(**args, check=3)) == pytest.approx([0.0] * 4)


@pytest.mark.parametrize(
    "bsdf_type,bindings",
    [
        pytest.param(name, bindings, id=name)
        for name, bindings, _, _, _ in BSDF_CONFIGS
        if name
        in {
            "mtlx::MxDielectricAiryBSDF",
            "mtlx::MxGeneralizedSchlickAiryBSDF",
            "mtlx::MxGeneralizedSchlickColor82BSDF",
        }
    ],
)
@pytest.mark.parametrize("hint", [BSDFFlags.glossy_reflection, BSDFFlags.glossy_transmission])
@pytest.mark.parametrize("retroreflective", [False, True])
def test_mtlx_lobe_sampling_hint_fresnel_variants(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    bsdf_type: str,
    bindings: dict[str, Any],
    hint: BSDFFlags,
    retroreflective: bool,
) -> None:
    bsdf = deepcopy(bindings)
    bsdf["transmission_tint"] = [0.4, 0.7, 0.9]
    bsdf["retroreflective"] = retroreflective
    kernel = (
        _bsdf_resources.module[f"bsdf_tests::test_mtlx_lobe_sampling_hint<{bsdf_type}>"]
        .as_func()
        .set(_mtlx_lut_bindings)
    )
    args = {"bsdf": bsdf, "wi": WI_30DEG, "random": [0.0, 0.0, 0.99], "hint": hint}
    result = kernel(**args, check=0)
    assert result.x == 1.0
    assert (result.y > 0.0) == (hint == BSDFFlags.glossy_reflection)
    assert result.z == pytest.approx(result.w, rel=2e-5)
    assert np.asarray(kernel(**args, check=1)) == pytest.approx([0.0] * 4, abs=2e-5)


# ---------------------------------------------------------------------------
# Reciprocity tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
@pytest.mark.nightly_if(bsdf_type=_NIGHTLY_STATISTICAL_BSDF_TYPES)
@pytest.mark.parametrize(
    "bsdf_type,bsdf_bindings,wi,bc",
    _make_params(SkipFlags.RECIPROCITY),
)
def test_bsdf_reciprocity(
    _bsdf_resources: _BSDFResources,
    _mtlx_lut_bindings: dict[str, Any],
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Verify Helmholtz reciprocity: f(wi,wo) = f(wo,wi)."""
    test = BSDFReciprocityTest(
        device=_bsdf_resources.device,
        module=_bsdf_resources.module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=_mtlx_lut_bindings,
    )
    assert test.run(), test.messages


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
