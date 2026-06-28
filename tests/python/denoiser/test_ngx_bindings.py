# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import Any

import falcor2 as f2
import falcor2.ngx as fngx
import falcor2.testing.helpers as helpers
import numpy as np
import pytest
import slangpy as spy


def _make_texture(
    device: spy.Device, width: int, height: int, format: spy.Format, label: str
) -> spy.Texture:
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    return device.create_texture(
        format=format, width=width, height=height, usage=usage, label=label
    )


def _assert_texture_field_retains_temporary_assignment(
    desc: Any,
    field_name: str,
    device: spy.Device,
    width: int,
    height: int,
    format: spy.Format,
) -> None:
    setattr(desc, field_name, _make_texture(device, width, height, format, field_name))
    gc.collect()

    texture = getattr(desc, field_name)
    assert texture is not None
    assert texture.device == device
    assert texture.width == width
    assert texture.height == height
    assert texture.format == format


def _clear_texture(
    command_encoder: spy.CommandEncoder, texture: spy.Texture, value: spy.float4
) -> None:
    command_encoder.clear_texture_float(texture, clear_value=value)


def _check_optimal_settings(settings: Any, target_width: int, target_height: int) -> None:
    assert settings.render_width > 0
    assert settings.render_height > 0
    assert settings.render_width <= target_width
    assert settings.render_height <= target_height
    assert settings.min_render_width <= settings.render_width
    assert settings.min_render_height <= settings.render_height
    assert settings.max_render_width >= settings.render_width
    assert settings.max_render_height >= settings.render_height


def _upload(texture: spy.Texture, data: np.ndarray[Any, Any]) -> None:
    texture.copy_from_numpy(np.ascontiguousarray(data))


def _constant_rgba(
    width: int, height: int, value: tuple[float, float, float, float]
) -> np.ndarray[Any, Any]:
    data = np.empty((height, width, 4), dtype=np.float16)
    data[...] = np.asarray(value, dtype=np.float16)
    return data


def _bilinear_sample(
    image: np.ndarray[Any, Any], x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    height, width = image.shape[:2]
    x = np.clip(x, 0.0, float(width - 1))
    y = np.clip(y, 0.0, float(height - 1))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0)[..., np.newaxis]
    wy = (y - y0)[..., np.newaxis]

    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _resize_linear(
    image: np.ndarray[Any, Any],
    width: int,
    height: int,
    jitter_x: float = 0.0,
    jitter_y: float = 0.0,
) -> np.ndarray[Any, Any]:
    src_height, src_width = image.shape[:2]
    dst_x = np.arange(width, dtype=np.float32)
    dst_y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(dst_x, dst_y)

    src_x = ((xx + 0.5 + jitter_x) * src_width / width) - 0.5
    src_y = ((yy + 0.5 + jitter_y) * src_height / height) - 0.5
    return _bilinear_sample(image, src_x, src_y)


def _make_reference_image(width: int, height: int) -> np.ndarray[Any, Any]:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    highlight = np.exp(-((xx - 0.68) ** 2 + (yy - 0.38) ** 2) / 0.018)
    shadow = np.exp(-((xx - 0.25) ** 2 + (yy - 0.72) ** 2) / 0.035)

    rgb = np.stack(
        (
            0.18 + 0.45 * xx + 0.18 * highlight,
            0.16 + 0.40 * yy + 0.12 * highlight,
            0.22 + 0.28 * (1.0 - xx) + 0.10 * shadow,
        ),
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _add_alpha(rgb: np.ndarray[Any, Any], alpha: float = 1.0) -> np.ndarray[Any, Any]:
    alpha_channel = np.full(rgb.shape[:2] + (1,), alpha, dtype=rgb.dtype)
    return np.concatenate((rgb, alpha_channel), axis=-1)


def _make_smoke_resources(
    device: spy.Device, desc: fngx.DLSSRRFeatureDesc
) -> dict[str, spy.Texture]:
    render_width = desc.render_width
    render_height = desc.render_height
    target_width = desc.target_width
    target_height = desc.target_height

    resources = {
        "diffuse_albedo": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_diffuse_albedo",
        ),
        "specular_albedo": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_specular_albedo",
        ),
        "normals": _make_texture(
            device, render_width, render_height, spy.Format.rgba16_float, "dlss_rr_py_normals"
        ),
        "roughness": _make_texture(
            device, render_width, render_height, spy.Format.r16_float, "dlss_rr_py_roughness"
        ),
        "color": _make_texture(
            device, render_width, render_height, spy.Format.rgba16_float, "dlss_rr_py_color"
        ),
        "depth": _make_texture(
            device, render_width, render_height, spy.Format.r32_float, "dlss_rr_py_depth"
        ),
        "motion_vectors": _make_texture(
            device, render_width, render_height, spy.Format.rg16_float, "dlss_rr_py_motion_vectors"
        ),
        "output": _make_texture(
            device, target_width, target_height, spy.Format.rgba16_float, "dlss_rr_py_output"
        ),
        "motion_vectors_reflections": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rg16_float,
            "dlss_rr_py_motion_vectors_reflections",
        ),
    }

    command_encoder = device.create_command_encoder()
    _clear_texture(command_encoder, resources["diffuse_albedo"], spy.float4(0.5, 0.5, 0.5, 1.0))
    _clear_texture(command_encoder, resources["specular_albedo"], spy.float4(0.04, 0.04, 0.04, 1.0))
    _clear_texture(command_encoder, resources["normals"], spy.float4(0.0, 0.0, 1.0, 0.0))
    _clear_texture(command_encoder, resources["roughness"], spy.float4(0.5))
    _clear_texture(command_encoder, resources["color"], spy.float4(0.1, 0.1, 0.1, 1.0))
    _clear_texture(command_encoder, resources["depth"], spy.float4(1.0))
    _clear_texture(command_encoder, resources["motion_vectors"], spy.float4(0.0))
    _clear_texture(command_encoder, resources["motion_vectors_reflections"], spy.float4(0.0))
    _clear_texture(command_encoder, resources["output"], spy.float4(0.0))
    device.submit_command_buffer(command_encoder.finish())
    device.wait()

    return resources


def _make_sr_smoke_resources(
    device: spy.Device, desc: fngx.DLSSSRFeatureDesc
) -> dict[str, spy.Texture]:
    render_width = desc.render_width
    render_height = desc.render_height
    target_width = desc.target_width
    target_height = desc.target_height

    color_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )
    motion_format = (
        spy.Format.rg32_float if device.desc.type == spy.DeviceType.cuda else spy.Format.rg16_float
    )
    resources = {
        "color": _make_texture(
            device, render_width, render_height, color_format, "dlss_sr_py_color"
        ),
        "depth": _make_texture(
            device, render_width, render_height, spy.Format.r32_float, "dlss_sr_py_depth"
        ),
        "motion_vectors": _make_texture(
            device, render_width, render_height, motion_format, "dlss_sr_py_motion_vectors"
        ),
        "output": _make_texture(
            device, target_width, target_height, color_format, "dlss_sr_py_output"
        ),
    }

    command_encoder = device.create_command_encoder()
    _clear_texture(command_encoder, resources["color"], spy.float4(0.1, 0.2, 0.3, 1.0))
    _clear_texture(command_encoder, resources["depth"], spy.float4(0.5))
    _clear_texture(command_encoder, resources["motion_vectors"], spy.float4(0.0))
    _clear_texture(command_encoder, resources["output"], spy.float4(0.0))
    device.submit_command_buffer(command_encoder.finish())
    device.wait()

    return resources


def _make_fg_smoke_resources(device: spy.Device, width: int, height: int) -> dict[str, spy.Texture]:
    color_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )
    motion_format = (
        spy.Format.rg32_float if device.desc.type == spy.DeviceType.cuda else spy.Format.rg16_float
    )
    resources = {
        "backbuffer": _make_texture(device, width, height, color_format, "dlss_g_py_backbuffer"),
        "depth": _make_texture(device, width, height, spy.Format.r32_float, "dlss_g_py_depth"),
        "motion_vectors": _make_texture(
            device, width, height, motion_format, "dlss_g_py_motion_vectors"
        ),
        "output_interpolated_frame": _make_texture(
            device, width, height, color_format, "dlss_g_py_output_interpolated_frame"
        ),
    }

    command_encoder = device.create_command_encoder()
    _clear_texture(command_encoder, resources["backbuffer"], spy.float4(0.1, 0.2, 0.3, 1.0))
    _clear_texture(command_encoder, resources["depth"], spy.float4(0.5))
    _clear_texture(command_encoder, resources["motion_vectors"], spy.float4(0.0))
    _clear_texture(command_encoder, resources["output_interpolated_frame"], spy.float4(0.0))
    device.submit_command_buffer(command_encoder.finish())
    device.wait()

    return resources


def _make_synthetic_resources(
    device: spy.Device, desc: fngx.DLSSRRFeatureDesc
) -> dict[str, spy.Texture]:
    render_width = desc.render_width
    render_height = desc.render_height
    target_width = desc.target_width
    target_height = desc.target_height

    resources = {
        "diffuse_albedo": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_synthetic_diffuse_albedo",
        ),
        "specular_albedo": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_synthetic_specular_albedo",
        ),
        "normals": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_synthetic_normals",
        ),
        "roughness": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.r16_float,
            "dlss_rr_py_synthetic_roughness",
        ),
        "color": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_synthetic_color",
        ),
        "depth": _make_texture(
            device, render_width, render_height, spy.Format.r32_float, "dlss_rr_py_synthetic_depth"
        ),
        "motion_vectors": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rg16_float,
            "dlss_rr_py_synthetic_motion_vectors",
        ),
        "output": _make_texture(
            device,
            target_width,
            target_height,
            spy.Format.rgba16_float,
            "dlss_rr_py_synthetic_output",
        ),
        "motion_vectors_reflections": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.rg16_float,
            "dlss_rr_py_synthetic_motion_vectors_reflections",
        ),
        "specular_hit_distance": _make_texture(
            device,
            render_width,
            render_height,
            spy.Format.r16_float,
            "dlss_rr_py_synthetic_specular_hit_distance",
        ),
    }

    x = np.linspace(0.0, 1.0, render_width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, render_height, dtype=np.float32)
    _xx, yy = np.meshgrid(x, y)
    roughness = np.clip(0.25 + 0.5 * yy, 0.0, 1.0)
    depth = 0.5 + 0.5 * yy
    zero_motion = np.zeros((render_height, render_width, 2), dtype=np.float16)
    guide_rgb = _resize_linear(
        _make_reference_image(target_width, target_height), render_width, render_height
    )

    _upload(resources["diffuse_albedo"], _add_alpha(guide_rgb).astype(np.float16))
    _upload(
        resources["specular_albedo"],
        _constant_rgba(render_width, render_height, (0.04, 0.04, 0.04, 1.0)),
    )
    _upload(resources["normals"], _constant_rgba(render_width, render_height, (0.0, 0.0, 1.0, 0.0)))
    _upload(resources["roughness"], roughness.astype(np.float16))
    _upload(resources["depth"], depth.astype(np.float32))
    _upload(resources["motion_vectors"], zero_motion)
    _upload(resources["motion_vectors_reflections"], zero_motion)
    _upload(
        resources["specular_hit_distance"],
        np.zeros((render_height, render_width), dtype=np.float16),
    )
    _upload(resources["output"], np.zeros((target_height, target_width, 4), dtype=np.float16))

    return resources


def _make_noisy_color_frame(
    reference: np.ndarray[Any, Any],
    render_width: int,
    render_height: int,
    rng: np.random.Generator,
    jitter_x: float,
    jitter_y: float,
) -> np.ndarray[Any, Any]:
    clean_rgb = _resize_linear(reference, render_width, render_height, jitter_x, jitter_y)
    noise = rng.normal(0.0, 0.08, clean_rgb.shape).astype(np.float32)
    noisy_rgb = np.clip(clean_rgb + noise, 0.0, 1.0)
    return _add_alpha(noisy_rgb).astype(np.float16)


def _make_evaluate_desc(
    resources: dict[str, spy.Texture],
    reset: bool = True,
    use_reflection_motion_vectors: bool = True,
) -> fngx.DLSSRREvaluateDesc:
    desc = fngx.DLSSRREvaluateDesc()
    desc.diffuse_albedo = resources["diffuse_albedo"]
    desc.specular_albedo = resources["specular_albedo"]
    desc.normals = resources["normals"]
    desc.roughness = resources["roughness"]
    desc.color = resources["color"]
    desc.depth = resources["depth"]
    desc.motion_vectors = resources["motion_vectors"]
    desc.output = resources["output"]
    if use_reflection_motion_vectors:
        desc.motion_vectors_reflections = resources["motion_vectors_reflections"]
    desc.reset = reset
    desc.motion_vector_scale_x = 1.0
    desc.motion_vector_scale_y = 1.0
    desc.pre_exposure = 1.0
    desc.exposure_scale = 1.0
    desc.frame_time_delta_ms = 16.6
    return desc


def _make_sr_evaluate_desc(
    resources: dict[str, spy.Texture],
    reset: bool = True,
) -> fngx.DLSSSREvaluateDesc:
    desc = fngx.DLSSSREvaluateDesc()
    desc.color = resources["color"]
    desc.depth = resources["depth"]
    desc.motion_vectors = resources["motion_vectors"]
    desc.output = resources["output"]
    desc.reset = reset
    desc.motion_vector_scale_x = 1.0
    desc.motion_vector_scale_y = 1.0
    desc.pre_exposure = 1.0
    desc.exposure_scale = 1.0
    return desc


def _make_fg_camera_desc(reset: bool = True) -> fngx.DLSSGCameraDesc:
    desc = fngx.DLSSGCameraDesc()
    identity = spy.float4x4.identity()
    desc.camera_view_to_clip = identity
    desc.clip_to_camera_view = identity
    desc.clip_to_lens_clip = identity
    desc.clip_to_prev_clip = identity
    desc.prev_clip_to_clip = identity
    desc.jitter_offset = spy.float2(0.0, 0.0)
    desc.motion_vector_scale = spy.float2(1.0, 1.0)
    desc.camera_pinhole_offset = spy.float2(0.0, 0.0)
    desc.camera_pos = spy.float3(0.0, 0.0, 0.0)
    desc.camera_up = spy.float3(0.0, 1.0, 0.0)
    desc.camera_right = spy.float3(1.0, 0.0, 0.0)
    desc.camera_fwd = spy.float3(0.0, 0.0, -1.0)
    desc.camera_near = 0.1
    desc.camera_far = 1000.0
    desc.camera_fov = 1.0
    desc.camera_aspect_ratio = 16.0 / 9.0
    desc.color_buffers_hdr = True
    desc.depth_inverted = False
    desc.camera_motion_included = True
    desc.reset = reset
    desc.multi_frame_count = 1
    desc.multi_frame_index = 1
    return desc


def _make_fg_evaluate_desc(
    resources: dict[str, spy.Texture],
    reset: bool = True,
) -> fngx.DLSSGEvaluateDesc:
    desc = fngx.DLSSGEvaluateDesc()
    desc.backbuffer = resources["backbuffer"]
    desc.depth = resources["depth"]
    desc.motion_vectors = resources["motion_vectors"]
    desc.output_interpolated_frame = resources["output_interpolated_frame"]
    desc.camera = _make_fg_camera_desc(reset=reset)
    return desc


def _mse(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ngx_evaluate_descs_retain_texture_fields(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    rr_desc = fngx.DLSSRREvaluateDesc()
    _assert_texture_field_retains_temporary_assignment(
        rr_desc, "color", device, 8, 4, spy.Format.rgba16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        rr_desc, "depth", device, 8, 4, spy.Format.r32_float
    )
    _assert_texture_field_retains_temporary_assignment(
        rr_desc, "motion_vectors", device, 8, 4, spy.Format.rg16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        rr_desc, "output", device, 16, 8, spy.Format.rgba16_float
    )

    sr_desc = fngx.DLSSSREvaluateDesc()
    _assert_texture_field_retains_temporary_assignment(
        sr_desc, "color", device, 8, 4, spy.Format.rgba16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        sr_desc, "depth", device, 8, 4, spy.Format.r32_float
    )
    _assert_texture_field_retains_temporary_assignment(
        sr_desc, "motion_vectors", device, 8, 4, spy.Format.rg16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        sr_desc, "output", device, 16, 8, spy.Format.rgba16_float
    )

    fg_desc = fngx.DLSSGEvaluateDesc()
    _assert_texture_field_retains_temporary_assignment(
        fg_desc, "backbuffer", device, 16, 8, spy.Format.rgba16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        fg_desc, "depth", device, 8, 4, spy.Format.r32_float
    )
    _assert_texture_field_retains_temporary_assignment(
        fg_desc, "motion_vectors", device, 8, 4, spy.Format.rg16_float
    )
    _assert_texture_field_retains_temporary_assignment(
        fg_desc, "output_interpolated_frame", device, 16, 8, spy.Format.rgba16_float
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_rr_api_smoke(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    ngx = helpers.get_ngx(device)
    same_ngx = fngx.NGX.get(device)

    assert ngx.device == device
    assert same_ngx.device == device
    assert ngx.info.dlssd_available
    assert ngx.info.dlssd_supported

    target_width = 1280
    target_height = 720
    settings = ngx.get_dlssd_optimal_settings(target_width, target_height, fngx.QualityMode.quality)
    _check_optimal_settings(settings, target_width, target_height)

    feature_desc = fngx.DLSSRRFeatureDesc()
    feature_desc.quality = fngx.QualityMode.quality
    feature_desc.render_width = settings.render_width
    feature_desc.render_height = settings.render_height
    feature_desc.target_width = target_width
    feature_desc.target_height = target_height

    feature = ngx.create_dlss_rr_feature(feature_desc)
    assert feature.device == device
    assert feature.ngx.device == device
    assert feature.desc.render_width == feature_desc.render_width
    assert feature.desc.target_width == feature_desc.target_width
    assert feature.last_evaluate_result == fngx.Result.none

    resources = _make_smoke_resources(device, feature_desc)
    evaluate_desc = _make_evaluate_desc(resources)

    feature.evaluate(evaluate_desc)
    device.wait()
    assert feature.last_evaluate_result == fngx.Result.success

    command_encoder = device.create_command_encoder()
    feature.evaluate(command_encoder, evaluate_desc)
    command_buffer = command_encoder.finish()
    del feature
    del evaluate_desc
    del resources
    device.submit_command_buffer(command_buffer)
    device.wait()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_sr_api_smoke(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    ngx = helpers.get_ngx(device)

    if not ngx.info.dlss_supported:
        pytest.skip("DLSS Super Resolution is not supported by this NGX runtime.")

    target_width = 1280
    target_height = 720
    settings = ngx.get_dlss_optimal_settings(target_width, target_height, fngx.QualityMode.quality)
    _check_optimal_settings(settings, target_width, target_height)

    feature_desc = fngx.DLSSSRFeatureDesc()
    feature_desc.quality = fngx.QualityMode.quality
    feature_desc.render_width = settings.render_width
    feature_desc.render_height = settings.render_height
    feature_desc.target_width = target_width
    feature_desc.target_height = target_height
    feature_desc.is_hdr = True
    feature_desc.auto_exposure = True

    feature = ngx.create_dlss_sr_feature(feature_desc)
    assert feature.device == device
    assert feature.ngx.device == device
    assert feature.desc.render_width == feature_desc.render_width
    assert feature.desc.target_width == feature_desc.target_width
    assert feature.last_evaluate_result == fngx.Result.none

    resources = _make_sr_smoke_resources(device, feature_desc)
    evaluate_desc = _make_sr_evaluate_desc(resources)

    feature.evaluate(evaluate_desc)
    device.wait()
    assert feature.last_evaluate_result == fngx.Result.success

    command_encoder = device.create_command_encoder()
    feature.evaluate(command_encoder, evaluate_desc)
    command_buffer = command_encoder.finish()
    del feature
    del evaluate_desc
    del resources
    device.submit_command_buffer(command_buffer)
    device.wait()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_g_api_smoke(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    ngx = helpers.get_ngx(device)

    if not ngx.info.frame_generation_supported:
        pytest.skip("DLSS Frame Generation is not supported by this NGX runtime.")

    width = 1280
    height = 720
    resources = _make_fg_smoke_resources(device, width, height)

    feature_desc = fngx.DLSSGFeatureDesc()
    feature_desc.width = width
    feature_desc.height = height
    feature_desc.render_width = width
    feature_desc.render_height = height
    feature_desc.backbuffer_format = resources["backbuffer"].format

    feature = ngx.create_dlss_g_feature(feature_desc)
    assert feature.device == device
    assert feature.ngx.device == device
    assert feature.desc.width == width
    assert feature.desc.render_width == width
    assert feature.last_evaluate_result == fngx.Result.none

    evaluate_desc = _make_fg_evaluate_desc(resources)

    feature.evaluate(evaluate_desc)
    device.wait()
    assert feature.last_evaluate_result == fngx.Result.success

    command_encoder = device.create_command_encoder()
    feature.evaluate(command_encoder, evaluate_desc)
    command_buffer = command_encoder.finish()
    del feature
    del evaluate_desc
    del resources
    device.submit_command_buffer(command_buffer)
    device.wait()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_rr_synthetic_denoises(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    ngx = helpers.get_ngx(device)

    target_width = 1280
    target_height = 720
    settings = ngx.get_dlssd_optimal_settings(target_width, target_height, fngx.QualityMode.quality)
    _check_optimal_settings(settings, target_width, target_height)

    feature_desc = fngx.DLSSRRFeatureDesc()
    feature_desc.quality = fngx.QualityMode.quality
    feature_desc.render_width = settings.render_width
    feature_desc.render_height = settings.render_height
    feature_desc.target_width = target_width
    feature_desc.target_height = target_height

    feature = ngx.create_dlss_rr_feature(feature_desc)
    resources = _make_synthetic_resources(device, feature_desc)

    reference = _make_reference_image(target_width, target_height)
    rng = np.random.default_rng(20260501)
    jitter_sequence = [
        (0.0, 0.0),
        (0.25, -0.25),
        (-0.25, 0.25),
        (0.375, 0.125),
    ]

    noisy_baseline = None
    for frame_index, (jitter_x, jitter_y) in enumerate(jitter_sequence):
        noisy_color = _make_noisy_color_frame(
            reference,
            feature_desc.render_width,
            feature_desc.render_height,
            rng,
            jitter_x,
            jitter_y,
        )
        _upload(resources["color"], noisy_color)

        evaluate_desc = _make_evaluate_desc(
            resources,
            reset=frame_index == 0,
            use_reflection_motion_vectors=False,
        )
        evaluate_desc.specular_hit_distance = resources["specular_hit_distance"]
        evaluate_desc.view_from_world = spy.float4x4.identity()
        evaluate_desc.clip_from_view = spy.math.perspective(
            spy.math.radians(70.0),
            float(feature_desc.render_width) / float(feature_desc.render_height),
            0.1,
            1000.0,
        )
        evaluate_desc.jitter_offset_x = jitter_x
        evaluate_desc.jitter_offset_y = jitter_y

        feature.evaluate(evaluate_desc)
        device.wait()

        assert feature.last_evaluate_result == fngx.Result.success
        noisy_baseline = _resize_linear(
            noisy_color[..., :3].astype(np.float32), target_width, target_height
        )

    assert noisy_baseline is not None
    output = np.asarray(resources["output"].to_numpy(), dtype=np.float32)
    output_rgb = np.clip(output[..., :3], 0.0, 1.0)
    noisy_mse = _mse(noisy_baseline, reference)
    output_mse = _mse(output_rgb, reference)

    assert output.shape == (target_height, target_width, 4)
    assert np.isfinite(output).all()
    assert np.max(np.abs(output[..., :3])) > 0.0
    assert output_mse < noisy_mse, f"output MSE {output_mse} >= noisy MSE {noisy_mse}"
