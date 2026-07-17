# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import slangpy as spy
from falcor2 import AssetResolver, TextureHandle, TextureManager
import numpy as np
import pytest
import falcor2.testing.helpers as helpers

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
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
        handle={"data": handle.data},
        uv=spy.float2(0, 0),
        default_value=spy.float4(1, 0, 1, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_3d(
        handle={"data": handle.data},
        uvw=spy.float3(0, 0, 0),
        default_value=spy.float4(1, 0, 1, 0),
    ) == spy.float4(1, 0, 1, 0)

    assert module.test_handle_flag_conversion(handle={"data": handle.data}) == handle.data


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
    expected_format = spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm

    white_1k_handle = texture_manager.load_texture(
        path=DATA_DIR / "assets/textures/test_texture_manager/white_1k.png",
        sampler=sampler,
        generate_mips=generate_mips,
        srgb=srgb,
        load_deferred=load_deferred,
    )

    assert white_1k_handle.is_valid()
    assert white_1k_handle.is_udim() == False
    assert white_1k_handle.path == DATA_DIR / "assets/textures/test_texture_manager/white_1k.png"
    assert white_1k_handle.sampler == expected_sampler

    regular_blue_handle = texture_manager.load_texture(
        path=DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png",
        sampler=sampler,
        generate_mips=generate_mips,
        srgb=srgb,
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
            generate_mips=generate_mips,
            srgb=srgb,
            load_deferred=load_deferred,
        )
        == white_1k_handle
    )
    assert (
        texture_manager.load_texture(
            path=DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png",
            sampler=sampler,
            generate_mips=generate_mips,
            srgb=srgb,
            load_deferred=load_deferred,
        )
        == regular_blue_handle
    )

    if load_deferred:
        assert white_1k_handle.is_finalized() == False
        assert white_1k_handle.texture == None

        assert regular_blue_handle.is_finalized() == False
        assert regular_blue_handle.texture == None

        texture_manager.update()

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
        handle={"data": white_1k_handle.data},
        uv=spy.float2(0.5, 0.5),
        default_value=spy.float4(0, 0, 0, 0),
    ) == spy.float4(1, 1, 1, 1)

    assert module.test_2d(
        handle={"data": regular_blue_handle.data},
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
        generate_mips=generate_mips,
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
            generate_mips=generate_mips,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
