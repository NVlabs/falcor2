# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import slangpy as spy
from falcor2 import AssetResolver, Blob, TextureHandle, TextureLoadFlags, TextureManager
import numpy as np
import pytest
import falcor2.testing.helpers as helpers

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class CapturingLoggerOutput(spy.LoggerOutput):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def write(self, level: spy.LogLevel, name: str, msg: str) -> None:
        self.messages.append(msg)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_asset_resolver_priority_for_regular_and_udim_textures(
    device_type: spy.DeviceType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = DATA_DIR / "assets/textures/test_texture_manager"
    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    working = tmp_path / "working"

    (high_root / "regular").mkdir(parents=True)
    (working / "regular").mkdir(parents=True)
    (high_root / "tiles").mkdir(parents=True)
    (low_root / "tiles").mkdir(parents=True)
    shutil.copyfile(source_dir / "white_1k.png", high_root / "regular/shared.png")
    shutil.copyfile(source_dir / "regular_blue.png", working / "regular/shared.png")
    shutil.copyfile(source_dir / "udim_1001.png", high_root / "tiles/udim_1001.png")
    shutil.copyfile(source_dir / "udim_1003.png", low_root / "tiles/udim_1003.png")
    monkeypatch.chdir(working)

    device = helpers.get_device(device_type)
    texture_manager = TextureManager(device)
    resolver = AssetResolver([high_root, low_root])

    regular = texture_manager.load_texture("regular/shared.png", resolver=resolver)
    assert regular.is_valid()
    assert regular.path == high_root / "regular/shared.png"
    assert regular.texture is not None
    assert regular.texture.width == 1024

    udim = texture_manager.load_texture("tiles/udim_<UDIM>.png", resolver=resolver)
    assert udim.is_valid()
    assert udim.is_udim()
    assert len(udim.udim_tiles) == 1
    assert udim.udim_tiles[0].tile_index == 1001
    assert udim.udim_tiles[0].texture_handle.path == high_root / "tiles/udim_1001.png"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_invalid_handle(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    handle = TextureHandle()
    assert handle.is_valid() == False
    assert handle.is_finalized() == False
    assert handle.is_udim() == False
    assert handle.path == None
    assert handle.texture == None
    assert handle.sampler == None

    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")

    assert module.test_2d(
        handle=handle,
        uv=spy.float2(0, 0),
        default_value=spy.float4(1, 0, 1, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_3d(
        handle=handle,
        uvw=spy.float3(0, 0, 0),
        default_value=spy.float4(1, 0, 1, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_handle_flag_conversion(handle=handle) == handle.data


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("custom_sampler", [False, True])
@pytest.mark.parametrize("generate_mips", [False, True])
@pytest.mark.parametrize("srgb", [False, True])
@pytest.mark.parametrize("load_deferred", [False, True])
def test_load_texture(
    device_type: spy.DeviceType,
    custom_sampler: bool,
    generate_mips: bool,
    srgb: bool,
    load_deferred: bool,
):
    device = helpers.get_device(device_type)

    sampler: spy.Sampler | None = None
    if custom_sampler:
        sampler = device.create_sampler()

    texture_manager = TextureManager(device)

    expected_sampler = sampler if custom_sampler else texture_manager.default_sampler
    flags = TextureLoadFlags.generate_mips if generate_mips else TextureLoadFlags.raw
    if srgb:
        flags |= TextureLoadFlags.srgb
    expected_format = spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm

    white_1k_handle = texture_manager.load_texture(
        path=DATA_DIR / "assets/textures/test_texture_manager/white_1k.png",
        sampler=sampler,
        flags=flags,
        load_deferred=load_deferred,
    )

    assert white_1k_handle.is_valid()
    assert white_1k_handle.is_udim() == False
    assert white_1k_handle.path == DATA_DIR / "assets/textures/test_texture_manager/white_1k.png"
    assert white_1k_handle.sampler == expected_sampler

    regular_blue_handle = texture_manager.load_texture(
        path=DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png",
        sampler=sampler,
        flags=flags,
        load_deferred=load_deferred,
    )

    assert regular_blue_handle.is_valid()
    assert regular_blue_handle.is_udim() == False
    assert (
        regular_blue_handle.path
        == DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png"
    )
    assert regular_blue_handle.sampler == expected_sampler

    # Test cache.
    assert (
        texture_manager.load_texture(
            path=DATA_DIR / "assets/textures/test_texture_manager/white_1k.png",
            sampler=sampler,
            flags=flags,
            load_deferred=load_deferred,
        )
        == white_1k_handle
    )
    assert (
        texture_manager.load_texture(
            path=DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png",
            sampler=sampler,
            flags=flags,
            load_deferred=load_deferred,
        )
        == regular_blue_handle
    )

    if load_deferred:
        assert white_1k_handle.is_finalized() == False
        assert white_1k_handle.texture == None

        assert regular_blue_handle.is_finalized() == False
        assert regular_blue_handle.texture == None

        log_output = CapturingLoggerOutput()
        logger = spy.Logger.get()
        logger.add_output(log_output)
        try:
            texture_manager.update()
        finally:
            logger.remove_output(log_output)

        loaded_messages = [
            message for message in log_output.messages if message.startswith('Loaded texture "')
        ]
        assert len(loaded_messages) == 2

    assert white_1k_handle.is_finalized() == True
    assert white_1k_handle.texture != None
    assert white_1k_handle.texture.format == expected_format
    assert white_1k_handle.texture.width == 1024
    assert white_1k_handle.texture.height == 1024
    assert white_1k_handle.texture.mip_count == (11 if generate_mips else 1)

    assert regular_blue_handle.is_finalized() == True
    assert regular_blue_handle.texture != None
    assert regular_blue_handle.texture.format == expected_format
    assert regular_blue_handle.texture.width == 64
    assert regular_blue_handle.texture.height == 64
    assert regular_blue_handle.texture.mip_count == (7 if generate_mips else 1)

    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")

    assert module.test_2d(
        handle=white_1k_handle,
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 1, 1, 1)

    assert module.test_2d(
        handle=regular_blue_handle,
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(0, 0, 1, 1)

    texture_manager.update()

    stats = texture_manager.stats
    assert stats.total_handle_count == 2
    assert stats.regular_handle_count == 2
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 2
    assert stats.regular_texture_count == 2
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == (1 if not custom_sampler else 2)

    white_1k_handle = None
    regular_blue_handle = None

    texture_manager.update()

    assert stats.total_handle_count == 0
    assert stats.regular_handle_count == 0
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 0
    assert stats.regular_texture_count == 0
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_encoded_bytes_deferred_load_and_cache(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    manager = TextureManager(device)
    path = DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png"
    encoded = path.read_bytes()
    source = Blob(encoded, "embedded/blue.png")
    assert source.data == encoded
    assert source.name == "embedded/blue.png"

    color = manager.load_texture(source, load_deferred=True)
    same_color = manager.load_texture(source, load_deferred=True)
    linear = manager.load_texture(source, flags=TextureLoadFlags.data_rgb, load_deferred=True)
    limited = manager.load_texture(source, max_mip_count=3, load_deferred=True)
    other_source = manager.load_texture(Blob(encoded, "embedded/blue.png"), load_deferred=True)
    from_file = manager.load_texture(path, load_deferred=True)

    assert color == same_color
    assert color != linear
    assert color != limited
    assert color != other_source
    assert color.path is None
    assert color.texture is None
    assert from_file.path == path

    del encoded, source
    manager.update()

    assert color.is_finalized()
    assert color.texture is not None
    assert color.texture.format == spy.Format.rgba8_unorm_srgb
    assert linear.texture.format == spy.Format.rgba8_unorm
    assert limited.texture.width == limited.texture.height == 4
    assert limited.texture.mip_count == 3
    assert other_source.texture is not None
    assert from_file.texture is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_invalid_encoded_bytes_fail_during_update(device_type: spy.DeviceType) -> None:
    manager = TextureManager(helpers.get_device(device_type))
    with pytest.raises(RuntimeError, match="exactly one"):
        manager.load_texture("")

    handle = manager.load_texture(Blob(b"invalid image", "embedded/broken.png"), load_deferred=True)
    assert handle.texture is None
    with pytest.raises(RuntimeError, match="embedded/broken.png"):
        manager.update()

    with pytest.raises(RuntimeError, match="embedded/immediate.png"):
        manager.load_texture(Blob(b"invalid image", "embedded/immediate.png"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("load_deferred", [False, True])
@pytest.mark.parametrize("srgb_gamma", [False, True])
def test_grayscale_texture_interpretations_are_cached_separately(
    device_type: spy.DeviceType, load_deferred: bool, srgb_gamma: bool, tmp_path: Path
) -> None:
    assert spy.Bitmap.supports_png_metadata()
    path = tmp_path / "grayscale.png"
    values = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    spy.Bitmap(values, pixel_format=spy.Bitmap.PixelFormat.y, srgb_gamma=srgb_gamma).write(path)
    assert spy.Bitmap(path).srgb_gamma == srgb_gamma
    # Explicit color interpretation overrides metadata; data bindings retain encoded values.
    rgba = np.concatenate(
        [np.repeat(values[:, :, None], 3, axis=2), np.full((8, 8, 1), 255, dtype=np.uint8)],
        axis=2,
    )
    device = helpers.get_device(device_type)
    sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
        address_u=spy.TextureAddressingMode.mirror_repeat,
    )
    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")

    for generate_mips in (False, True):
        manager = TextureManager(device)
        handles: list[TextureHandle] = []
        # Encoding and RGB promotion are independent cache-key choices.
        for srgb, require_rgb in ((False, False), (False, True), (True, False), (True, True)):
            flags = TextureLoadFlags.generate_mips if generate_mips else TextureLoadFlags.raw
            if srgb:
                flags |= TextureLoadFlags.srgb
            if require_rgb:
                flags |= TextureLoadFlags.expand_luminance
            options = dict(
                sampler=sampler,
                flags=flags,
                load_deferred=load_deferred,
            )
            handle = manager.load_texture(path, **options)
            assert manager.load_texture(path, **options) == handle
            assert all(handle != previous for previous in handles)
            assert handle.is_finalized() == (not load_deferred)
            if load_deferred:
                assert handle.texture is None
            handles.append(handle)

        manager.update()
        assert manager.stats.total_texture_count == 4
        for index, handle in enumerate(handles):
            assert handle.is_finalized()
            assert handle.sampler == sampler
            texture = handle.texture
            assert texture is not None
            assert texture.mip_count == (4 if generate_mips else 1)
            assert texture.format == (
                spy.Format.r8_unorm
                if index in (0, 2)
                else (spy.Format.rgba8_unorm_srgb if index == 3 else spy.Format.rgba8_unorm)
            )
            np.testing.assert_array_equal(texture.to_numpy(), values if index in (0, 2) else rgba)
            sample = module.test_2d(
                handle=handle,
                uv=spy.float2(4.5 / 8, 4.5 / 8),
                default_value=spy.float4(0),
            )
            encoded = float(values[4, 4]) / 255.0
            value = ((encoded + 0.055) / 1.055) ** 2.4 if index == 3 else encoded
            if index in (0, 2):
                # Raw R8's missing alpha is backend-dependent and is not changed here.
                assert [sample.x, sample.y, sample.z] == pytest.approx([value, 0.0, 0.0], abs=5e-4)
            else:
                assert [sample.x, sample.y, sample.z, sample.w] == pytest.approx(
                    [value] * 3 + [1.0], abs=5e-4
                )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_deferred_load_filters_loaded_and_repeated_ids(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    texture_manager = TextureManager(device)
    white_path = DATA_DIR / "assets/textures/test_texture_manager/white_1k.png"
    blue_path = DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png"

    white_handle = texture_manager.load_texture(white_path, load_deferred=True)
    assert texture_manager.load_texture(white_path, load_deferred=False) == white_handle
    assert white_handle.is_finalized()

    blue_handle = texture_manager.load_texture(blue_path, load_deferred=True)
    assert texture_manager.load_texture(blue_path, load_deferred=True) == blue_handle

    log_output = CapturingLoggerOutput()
    logger = spy.Logger.get()
    logger.add_output(log_output)
    try:
        texture_manager.update()
    finally:
        logger.remove_output(log_output)

    loaded_messages = [
        message for message in log_output.messages if message.startswith('Loaded texture "')
    ]
    assert len(loaded_messages) == 1
    assert str(blue_path) in loaded_messages[0]
    assert blue_handle.is_finalized()
    assert blue_handle.texture is not None
    assert blue_handle.texture.width == 64


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("custom_sampler", [False, True])
@pytest.mark.parametrize("generate_mips", [False, True])
@pytest.mark.parametrize("load_deferred", [False, True])
def test_load_udim_texture(
    device_type: spy.DeviceType, custom_sampler: bool, generate_mips: bool, load_deferred: bool
):
    device = helpers.get_device(device_type)

    sampler: spy.Sampler | None = None
    if custom_sampler:
        sampler = device.create_sampler()

    texture_manager = TextureManager(device)

    expected_sampler = sampler if custom_sampler else texture_manager.default_sampler

    udim_handle = texture_manager.load_texture(
        path=DATA_DIR / "assets/textures/test_texture_manager/udim_<UDIM>.png",
        sampler=sampler,
        flags=TextureLoadFlags.color if generate_mips else TextureLoadFlags.environment,
        load_deferred=load_deferred,
    )

    assert udim_handle.is_valid()
    assert udim_handle.is_udim() == True
    assert udim_handle.path == DATA_DIR / "assets/textures/test_texture_manager/udim_<UDIM>.png"
    assert udim_handle.sampler == expected_sampler
    assert len(udim_handle.udim_tiles) == 2
    tile_0 = udim_handle.udim_tiles[0]
    assert tile_0.tile_index == 1001
    assert tile_0.tile_u == 0
    assert tile_0.tile_v == 0
    assert (
        tile_0.texture_handle.path
        == DATA_DIR / "assets/textures/test_texture_manager/udim_1001.png"
    )
    tile_1 = udim_handle.udim_tiles[1]
    assert tile_1.tile_index == 1003
    assert tile_1.tile_u == 2
    assert tile_1.tile_v == 0
    assert (
        tile_1.texture_handle.path
        == DATA_DIR / "assets/textures/test_texture_manager/udim_1003.png"
    )
    for tile in udim_handle.udim_tiles:
        assert tile.texture_handle.sampler == expected_sampler

    # Test cache.
    assert (
        texture_manager.load_texture(
            path=DATA_DIR / "assets/textures/test_texture_manager/udim_<UDIM>.png",
            sampler=sampler,
            flags=TextureLoadFlags.color if generate_mips else TextureLoadFlags.environment,
            load_deferred=load_deferred,
        )
        == udim_handle
    )

    if load_deferred:
        assert udim_handle.is_finalized() == False
        assert udim_handle.texture == None
        for tile in udim_handle.udim_tiles:
            assert tile.texture_handle.is_finalized() == False
            assert tile.texture_handle.texture == None

        texture_manager.update()

    assert udim_handle.is_finalized() == True
    for tile in udim_handle.udim_tiles:
        assert tile.texture_handle.is_finalized() == True
        assert tile.texture_handle.texture != None
        assert tile.texture_handle.texture.width == 64
        assert tile.texture_handle.texture.height == 64
        assert tile.texture_handle.texture.mip_count == (7 if generate_mips else 1)

    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")

    assert module.test_2d(
        handle={"data": udim_handle.data},
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(1, 0, 1, 1),
    ) == spy.float4(1, 0, 0, 1)

    assert module.test_2d(
        handle={"data": udim_handle.data},
        uv=spy.float2(1.5, 0.5),
        default_value=spy.float4(1, 0, 1, 1),
    ) == spy.float4(1, 0, 1, 1)

    assert module.test_2d(
        handle={"data": udim_handle.data},
        uv=spy.float2(2.5, 0.5),
        default_value=spy.float4(1, 0, 1, 1),
    ) == spy.float4(0, 1, 0, 1)

    texture_manager.update()

    stats = texture_manager.stats
    assert stats.total_handle_count == 3
    assert stats.regular_handle_count == 2
    assert stats.udim_handle_count == 1
    assert stats.total_texture_count == 3
    assert stats.regular_texture_count == 2
    assert stats.udim_texture_count == 1
    assert stats.total_sampler_count == 1 if not custom_sampler else 2

    udim_handle = None

    texture_manager.update()

    assert stats.total_handle_count == 2
    assert stats.regular_handle_count == 2
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 2
    assert stats.regular_texture_count == 2
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == 1 if not custom_sampler else 2

    tile = None
    tile_0 = None
    tile_1 = None

    texture_manager.update()

    assert stats.total_handle_count == 0
    assert stats.regular_handle_count == 0
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 0
    assert stats.regular_texture_count == 0
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_register_texture(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    texture_manager = TextureManager(device)

    texture_2d = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.float32),
    )

    texture_3d = device.create_texture(
        type=spy.TextureType.texture_3d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        depth=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=np.float32),
    )

    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")

    point_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point, mag_filter=spy.TextureFilteringMode.point
    )

    mirror_config = TextureManager.SamplerConfig()
    mirror_config.min_filter = spy.TextureFilteringMode.point
    mirror_config.mag_filter = spy.TextureFilteringMode.point
    mirror_config.address_u = spy.TextureAddressingMode.mirror_repeat

    handle_2d_default = texture_manager.register_texture(texture_2d)
    assert handle_2d_default.is_valid() == True
    assert handle_2d_default.is_finalized() == True
    assert handle_2d_default.is_udim() == False
    assert handle_2d_default.path == None
    assert handle_2d_default.texture == texture_2d
    assert handle_2d_default.sampler == texture_manager.default_sampler

    handle_2d_point = texture_manager.register_texture(texture_2d, point_sampler)
    assert handle_2d_point.is_valid() == True
    assert handle_2d_point.is_finalized() == True
    assert handle_2d_point.is_udim() == False
    assert handle_2d_point.path == None
    assert handle_2d_point.texture == texture_2d
    assert handle_2d_point.sampler == point_sampler

    handle_2d_mirror = texture_manager.register_texture(texture_2d, mirror_config)
    assert handle_2d_mirror.is_valid() == True
    assert handle_2d_mirror.is_finalized() == True
    assert handle_2d_mirror.is_udim() == False
    assert handle_2d_mirror.path == None
    assert handle_2d_mirror.texture == texture_2d
    assert handle_2d_mirror.sampler.desc.address_u == spy.TextureAddressingMode.mirror_repeat

    handle_3d_default = texture_manager.register_texture(texture_3d)
    assert handle_3d_default.is_valid() == True
    assert handle_3d_default.is_finalized() == True
    assert handle_3d_default.is_udim() == False
    assert handle_3d_default.path == None
    assert handle_3d_default.texture == texture_3d
    assert handle_3d_default.sampler == texture_manager.default_sampler

    handle_3d_point = texture_manager.register_texture(texture_3d, point_sampler)
    assert handle_3d_point.is_valid() == True
    assert handle_3d_point.is_finalized() == True
    assert handle_3d_point.is_udim() == False
    assert handle_3d_point.path == None
    assert handle_3d_point.texture == texture_3d
    assert handle_3d_point.sampler == point_sampler

    handle_3d_mirror = texture_manager.register_texture(texture_3d, mirror_config)
    assert handle_3d_mirror.is_valid() == True
    assert handle_3d_mirror.is_finalized() == True
    assert handle_3d_mirror.is_udim() == False
    assert handle_3d_mirror.path == None
    assert handle_3d_mirror.texture == texture_3d
    assert handle_3d_mirror.sampler.desc.address_u == spy.TextureAddressingMode.mirror_repeat

    # Test cache.
    assert texture_manager.register_texture(texture_2d) == handle_2d_default
    assert texture_manager.register_texture(texture_2d, point_sampler) == handle_2d_point
    assert texture_manager.register_texture(texture_2d, mirror_config) == handle_2d_mirror
    assert texture_manager.register_texture(texture_3d) == handle_3d_default
    assert texture_manager.register_texture(texture_3d, point_sampler) == handle_3d_point
    assert texture_manager.register_texture(texture_3d, mirror_config) == handle_3d_mirror

    texture_manager.update()

    assert module.test_2d(
        handle={"data": handle_2d_default.data},
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(0.5, 0, 0.5, 0)

    assert module.test_2d(
        handle={"data": handle_2d_point.data},
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_2d(
        handle={"data": handle_2d_mirror.data},
        uv=spy.float2(1.25, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_3d(
        handle={"data": handle_3d_default.data},
        uvw=spy.float3(0.5, 0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(0.5, 0.5, 0.5, 0)

    assert module.test_3d(
        handle={"data": handle_3d_point.data},
        uvw=spy.float3(0.5, 0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 1, 1, 0)

    assert module.test_3d(
        handle={"data": handle_3d_mirror.data},
        uvw=spy.float3(1.25, 0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 1, 1, 0)

    texture_manager.update()

    stats = texture_manager.stats
    assert stats.total_handle_count == 6
    assert stats.regular_handle_count == 6
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 2
    assert stats.regular_texture_count == 2
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == 3

    handle_2d_default = None
    handle_2d_point = None
    handle_2d_mirror = None
    handle_3d_default = None
    handle_3d_point = None
    handle_3d_mirror = None

    texture_manager.update()

    assert stats.total_handle_count == 0
    assert stats.regular_handle_count == 0
    assert stats.udim_handle_count == 0
    assert stats.total_texture_count == 0
    assert stats.regular_texture_count == 0
    assert stats.udim_texture_count == 0
    assert stats.total_sampler_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("load_deferred", [False, True])
def test_texture_load_presets_share_resources(
    device_type: spy.DeviceType, load_deferred: bool, tmp_path: Path
) -> None:
    path = tmp_path / "gray.png"
    values = np.full((4, 4), 128, dtype=np.uint8)
    spy.Bitmap(values, pixel_format=spy.Bitmap.PixelFormat.y).write(path)
    device = helpers.get_device(device_type)
    manager = TextureManager(device)
    groups = [
        (TextureLoadFlags.raw, TextureLoadFlags.none),
        (TextureLoadFlags.data, TextureLoadFlags.generate_mips),
        (
            TextureLoadFlags.data_rgb,
            TextureLoadFlags.normal,
            TextureLoadFlags.generate_mips | TextureLoadFlags.expand_luminance,
        ),
        (TextureLoadFlags.color, TextureLoadFlags.data_rgb | TextureLoadFlags.srgb),
        (
            TextureLoadFlags.environment,
            TextureLoadFlags.srgb | TextureLoadFlags.expand_luminance,
        ),
    ]
    handles = []
    for group in groups:
        handle = manager.load_texture(path, flags=group[0], load_deferred=load_deferred)
        for alias in group[1:]:
            assert manager.load_texture(path, flags=alias, load_deferred=load_deferred) == handle
        handles.append(handle)
    assert manager.load_texture(path, load_deferred=load_deferred) == handles[3]
    alternate_sampler = device.create_sampler(address_u=spy.TextureAddressingMode.clamp_to_edge)
    alternate = manager.load_texture(
        path, flags=TextureLoadFlags.normal, sampler=alternate_sampler, load_deferred=load_deferred
    )
    manager.update()
    assert alternate != handles[2]
    assert alternate.texture == handles[2].texture
    assert manager.stats.total_texture_count == 5
    assert [h.texture.mip_count for h in handles] == [1, 3, 3, 3, 1]
    assert [h.texture.format for h in handles] == [
        spy.Format.r8_unorm,
        spy.Format.r8_unorm,
        spy.Format.rgba8_unorm,
        spy.Format.rgba8_unorm_srgb,
        spy.Format.rgba8_unorm_srgb,
    ]
    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")
    sample = module.test_2d(handle=handles[4], uv=spy.float2(0.5), default_value=spy.float4(0))
    linear = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4
    assert [sample.x, sample.y, sample.z, sample.w] == pytest.approx([linear] * 3 + [1.0], abs=5e-4)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_load_color_preserves_hdr(device_type: spy.DeviceType, tmp_path: Path) -> None:
    path = tmp_path / "linear.exr"
    values = np.full((4, 4, 4), [0.25, 0.5, 2.0, 0.75], dtype=np.float32)
    spy.Bitmap(values, pixel_format=spy.Bitmap.PixelFormat.rgba, srgb_gamma=False).write(path)
    manager = TextureManager(helpers.get_device(device_type))
    handle = manager.load_texture(path, flags=TextureLoadFlags.environment)
    assert handle.texture.mip_count == 1
    assert handle.texture.format in (spy.Format.rgba16_float, spy.Format.rgba32_float)
    np.testing.assert_allclose(handle.texture.to_numpy(), values)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_raw_texture_retains_authored_mips(device_type: spy.DeviceType) -> None:
    path = DATA_DIR.parent / "external/slangpy/data/test_images/dds/bc1-unorm.dds"
    authored_mips = int.from_bytes(path.read_bytes()[28:32], "little")
    assert authored_mips > 1
    manager = TextureManager(helpers.get_device(device_type))
    raw = manager.load_texture(path, flags=TextureLoadFlags.raw)
    assert raw.texture.format == spy.Format.bc1_unorm
    assert raw.texture.mip_count == authored_mips


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimension", [2, 3])
def test_divergent_texture_and_sampler_handles(device_type: spy.DeviceType, dimension: int) -> None:
    device = helpers.get_device(device_type)
    manager = TextureManager(device)
    module = spy.Module.load_from_file(device, "render/test_texture_manager.slang")
    samplers = [
        device.create_sampler(min_filter=mode, mag_filter=mode, mip_filter=mode)
        for mode in (spy.TextureFilteringMode.linear, spy.TextureFilteringMode.point)
    ]
    handles = []
    expected = []
    for index in range(8):
        width = 2 if index % 2 == 0 else 4
        values = np.broadcast_to(
            np.arange(width, dtype=np.float32)[..., None] + index * 10,
            (1,) * (dimension - 1) + (width, 4),
        ).copy()
        texture = device.create_texture(
            type=spy.TextureType.texture_2d if dimension == 2 else spy.TextureType.texture_3d,
            format=spy.Format.rgba32_float,
            width=width,
            height=1,
            depth=1,
            usage=spy.TextureUsage.shader_resource,
            data=values,
        )
        for sampler_index, sampler in enumerate(samplers):
            handles.append(manager.register_texture(texture, sampler))
            sampled = 0.375 * width - 0.5 if sampler_index == 0 else int(0.375 * width)
            expected.append([index * 10 + sampled, index * 10])
    manager.update()
    # Adjacent lanes vary both texture and sampler, including repeated nonadjacent handles.
    indices = np.random.default_rng(42).integers(0, len(handles), size=1024)
    packed = np.array([handles[index].data for index in indices], dtype=np.uint32)
    function = module.test_divergent_2d if dimension == 2 else module.test_divergent_3d
    result = function(packed_handle=spy.Tensor.from_numpy(device, packed), _result="numpy")
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.float32)[indices])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
