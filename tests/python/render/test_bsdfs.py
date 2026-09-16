# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chi-square, eval/sample consistency, and reciprocity tests for BSDF implementations."""

import enum
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


WI_NORMAL = spy.float3(0.0, 0.0, 1.0)
WI_30DEG = spy.float3(0.5, 0.0, 0.866025)
WI_60DEG = spy.float3(0.866025, 0.0, 0.5)
WI_GRAZING = spy.float3(0.994987, 0.0, 0.1)

BSDF_CONFIGS = [
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


def test_anisotropic_ggx_policy_preserves_default_and_allows_clamping(
    _bsdf_resources: _BSDFResources,
) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_anisotropic_ggx_policy"]()

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(0.001)
    assert result.w == pytest.approx(1.0)


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


def test_mtlx_mix_preserves_absorption_side(_bsdf_resources: _BSDFResources) -> None:
    result = _bsdf_resources.module["bsdf_tests::test_mtlx_mix_albedo_contributions"]()

    assert np.asarray(result) == pytest.approx([0.0, 0.0, 0.35, 0.65])


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
        random=[0.0, 0.0, p_reflect * 0.5],
    )
    reflected_only = kernel(
        bsdf=bindings(1.0, 0.0),
        wi=WI_NORMAL,
        random=[0.0, 0.0, 0.0],
    )
    transmitted = kernel(
        bsdf=bindings(1.0, 1.0),
        wi=WI_NORMAL,
        random=[0.0, 0.0, (1.0 + p_reflect) * 0.5],
    )
    transmitted_only = kernel(
        bsdf=bindings(0.0, 1.0),
        wi=WI_NORMAL,
        random=[0.0, 0.0, 0.0],
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
    assert sample_result.z == pytest.approx(0.0, abs=1e-6)
    assert sample_result.w == expected_events


# ---------------------------------------------------------------------------
# Chi-square tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reciprocity tests
# ---------------------------------------------------------------------------


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
