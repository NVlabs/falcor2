# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chi-square, eval/sample consistency, and reciprocity tests for BSDF implementations."""

import enum
import numpy as np
import slangpy as spy
import pytest
from typing import Any, Optional
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


class LobeTypes(enum.IntFlag):
    """Python enum matching Slang's LobeTypes definition."""

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
            "active_lobes": int(LobeTypes.glossy_reflection),
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
            "active_lobes": int(LobeTypes.glossy_reflection),
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
            "active_lobes": int(LobeTypes.glossy_reflection | LobeTypes.glossy_transmission),
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
            "active_lobes": int(LobeTypes.glossy_reflection | LobeTypes.glossy_transmission),
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
        "mx139::MxBurleyDiffuseBSDF",
        {"albedo_value": [0.8, 0.4, 0.2], "roughness": 0.5},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 Oren-Nayar diffuse BSDF ---
    (
        "mx139::MxOrenNayarDiffuseBSDF",
        {"albedo_value": [0.6, 0.6, 0.6], "roughness": 0.35, "energy_compensation": True},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 translucent BSDF ---
    (
        "mx139::MxTranslucentBSDF",
        {"albedo_value": [0.7, 0.5, 0.3]},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 subsurface diffuse fallback BSDF ---
    (
        "mx139::MxSubsurfaceDiffuseFallbackBSDF",
        {"albedo_value": [0.55, 0.35, 0.25]},
        [WI_NORMAL, WI_30DEG],
        None,
        SkipFlags.NONE,
    ),
    # --- MaterialX 1.39 sheen BSDF ---
    (
        "mx139::MxSheenBSDF",
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
    # --- MaterialX 1.39 conductor BSDF ---
    (
        "mx139::MxConductorBSDF",
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
        "mx139::MxConductorAiryBSDF",
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
        "mx139::MxConductorTurquinAnalyticCompensationBSDF",
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
        "mx139::MxScratchConductorBSDF",
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
        # Scratch sampling returns selected-branch pdf/weight values, while eval_pdf()
        # reports the masked mixture of scratch and unscratched conductor lobes.
        SkipFlags.CHI2 | SkipFlags.SAMPLE_CONSISTENCY | SkipFlags.RECIPROCITY,
    ),
    # --- MaterialX 1.39 dielectric BSDF ---
    (
        "mx139::MxDielectricBSDF",
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
        "mx139::MxDielectricAiryBSDF",
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
        "mx139::MxGeneralizedSchlickBSDF",
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
        "mx139::MxGeneralizedSchlickColor82BSDF",
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
        "mx139::MxGeneralizedSchlickAiryBSDF",
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
        "mx139::MxChiangHairBSDF",
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
# Helper to load the test module
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[spy.DeviceType, spy.Module] = {}
_MX139_LUT_BINDINGS_CACHE: dict[spy.DeviceType, dict[str, Any]] = {}


def _get_module(device: spy.Device, device_type: spy.DeviceType) -> spy.Module:
    if device_type not in _MODULE_CACHE:
        _MODULE_CACHE[device_type] = spy.Module(device.load_module("render/bsdf_tests.slang"))
    return _MODULE_CACHE[device_type]


def _get_mx139_lut_bindings(device: spy.Device, device_type: spy.DeviceType) -> dict[str, Any]:
    if device_type not in _MX139_LUT_BINDINGS_CACHE:
        _MX139_LUT_BINDINGS_CACHE[device_type] = f2.create_materialx_mx139_lut_bindings(device)
    return _MX139_LUT_BINDINGS_CACHE[device_type]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_anisotropic_ggx_policy_preserves_default_and_allows_clamping(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)

    result = module["bsdf_tests::test_anisotropic_ggx_policy"]()

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(0.001)
    assert result.w == pytest.approx(1.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mx139_beer_delta_transmission(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    kernel = module["bsdf_tests::test_mx139_beer_delta_transmission"].as_func().set(global_bindings)
    result = kernel(
        bsdf={"inner": {"absorption": [0.25, 0.5, 0.75], "pathshortening": 0.0}},
        wi=spy.float3(0.0, 0.0, -1.0),
    )

    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(float(np.exp(-0.25)))
    assert result.w == pytest.approx(float(np.exp(-0.75)))


# ---------------------------------------------------------------------------
# Chi-square tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("bsdf_type,bsdf_bindings,wi,bc", _make_params(SkipFlags.CHI2))
def test_bsdf_chi2(
    device_type: spy.DeviceType,
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Chi-square test: sample() distribution matches eval_pdf()."""
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    test = BSDFChiSquareTest(
        device=device,
        module=module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=global_bindings,
    )
    assert test.run(), test.messages


# ---------------------------------------------------------------------------
# Eval/sample consistency tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "bsdf_type,bsdf_bindings,wi,bc", _make_params(SkipFlags.SAMPLE_CONSISTENCY)
)
def test_bsdf_eval_sample_consistency(
    device_type: spy.DeviceType,
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Verify eval()/eval_pdf() == weight from sample()."""
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    test = BSDFEvalSampleConsistencyTest(
        device=device,
        module=module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=global_bindings,
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "wi",
    [
        WI_30DEG,
        spy.float3(0.13215184211730957, 0.2252282053232193, 0.965302050113678),
    ],
)
def test_pbrt_dielectric_near_smooth_eval_sample_consistency(
    device_type: spy.DeviceType,
    wi: spy.float3,
) -> None:
    """Near-smooth PBRT dielectric samples must report the same PDF as eval_pdf()."""
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    test = BSDFEvalSampleConsistencyTest(
        device=device,
        module=module,
        bsdf_type="PBRTDielectricBSDF",
        bsdf_bindings={"D": {"alpha": [0.002, 0.002]}, "eta": 1.0 / 1.48},
        wi=wi,
        bc={"ior_i": 1.0, "ior_t": 1.48},
        sample_count=100000,
        global_bindings=global_bindings,
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "sample_lobe_types_hint",
    [
        LobeTypes.reflection,
        LobeTypes.transmission,
    ],
)
def test_pbrt_dielectric_lobe_sampling_hint_eval_sample_consistency(
    device_type: spy.DeviceType,
    sample_lobe_types_hint: LobeTypes,
) -> None:
    """Restricted sampling hints must report coherent hint-conditioned PDFs."""
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    test = BSDFEvalSampleConsistencyTest(
        device=device,
        module=module,
        bsdf_type="PBRTDielectricBSDF",
        bsdf_bindings={"D": {"alpha": [0.3, 0.3]}, "eta": 1.0 / 1.5},
        wi=WI_30DEG,
        bc={
            "ior_i": 1.0,
            "ior_t": 1.5,
            "sample_lobe_types_hint": sample_lobe_types_hint,
        },
        sample_count=100000,
        global_bindings=global_bindings,
    )
    assert test.run(), test.messages


# ---------------------------------------------------------------------------
# Reciprocity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "bsdf_type,bsdf_bindings,wi,bc",
    _make_params(SkipFlags.RECIPROCITY),
)
def test_bsdf_reciprocity(
    device_type: spy.DeviceType,
    bsdf_type: str,
    bsdf_bindings: dict[str, Any],
    wi: spy.float3,
    bc: Optional[dict[str, float]],
) -> None:
    """Verify Helmholtz reciprocity: f(wi,wo) = f(wo,wi)."""
    device = helpers.get_device(device_type)
    module = _get_module(device, device_type)
    global_bindings = _get_mx139_lut_bindings(device, device_type)

    test = BSDFReciprocityTest(
        device=device,
        module=module,
        bsdf_type=bsdf_type,
        bsdf_bindings=bsdf_bindings,
        wi=wi,
        bc=bc,
        global_bindings=global_bindings,
    )
    assert test.run(), test.messages


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
