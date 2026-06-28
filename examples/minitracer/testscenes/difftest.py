# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from slangpy import Device, float3
from falcor2.minitracer.scene import Scene

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def diff_test_scene(device: Device) -> Scene:
    scene = Scene(device)

    zero_texture = scene.create_texture(1, 1, float3(0.0))
    one_texture = scene.create_texture(1, 1, float3(1.0))

    floor_albedo = scene.create_texture(1, 1, float3(0.3, 0.3, 0.3))
    floor_roughness = scene.create_texture(1, 1, float3(0.5))
    floor_metallic = scene.create_texture(1, 1, float3(0.0))

    floor_material = scene.create_material("floor")
    floor_material.albedo = floor_albedo
    floor_material.specular = zero_texture
    floor_material.metallic = floor_metallic
    floor_material.roughness = floor_roughness
    floor_material.emission = zero_texture

    box1_albedo = scene.create_texture(1, 1, float3(0.8, 0.2, 0.2))
    box1_roughness = scene.create_texture(1, 1, float3(0.5))
    box1_metallic = scene.create_texture(1, 1, float3(0.0))

    box1_material = scene.create_material("box1")
    box1_material.albedo = box1_albedo
    box1_material.specular = zero_texture
    box1_material.metallic = box1_metallic
    box1_material.roughness = box1_roughness
    box1_material.emission = zero_texture

    box2_albedo = scene.create_texture(1, 1, float3(0.2, 0.8, 0.2))
    box2_roughness = scene.create_texture(1, 1, float3(0.5))
    box2_metallic = scene.create_texture(1, 1, float3(0.0))

    box2_material = scene.create_material("box2")
    box2_material.albedo = box2_albedo
    box2_material.specular = zero_texture
    box2_material.metallic = box2_metallic
    box2_material.roughness = box2_roughness
    box2_material.emission = zero_texture

    box3_albedo = scene.create_texture(1, 1, float3(0.2, 0.2, 0.8))
    box3_roughness = scene.create_texture(1, 1, float3(0.5))
    box3_metallic = scene.create_texture(1, 1, float3(0.0))
    box3_emission = zero_texture

    box3_material = scene.create_material("box3")
    box3_material.albedo = box3_albedo
    box3_material.specular = zero_texture
    box3_material.metallic = box3_metallic
    box3_material.roughness = box3_roughness
    box3_material.emission = box3_emission

    emissive_material = scene.create_material("emissive")
    emissive_material.albedo = zero_texture
    emissive_material.specular = zero_texture
    emissive_material.metallic = zero_texture
    emissive_material.roughness = one_texture
    emissive_material.emission = scene.create_texture(1, 1, float3(50.0, 60.0, 70.0))

    floor = scene.create_box(float3(100, 1, 100))
    floor.transform.pos = float3(0, -0.5, 0)
    floor.material = floor_material

    box1 = scene.create_box(float3(1, 2, 1))
    box1.transform.pos = float3(-3, 1, 0)
    box1.material = box1_material

    box2 = scene.create_box(float3(1, 2, 1))
    box2.transform.pos = float3(0, 1, 0)
    box2.material = box2_material

    box3 = scene.create_box(float3(1, 2, 1))
    box3.transform.pos = float3(3, 1, 0)
    box3.material = box3_material

    light = scene.create_box(float3(8, 0.25, 0.25))
    light.transform.pos = float3(0, 4, 0)
    light.material = emissive_material

    camera = scene.create_camera(128, 128, 70)
    camera.transform.pos = float3(0, 2, 4)

    # env_map = scene.create_env_map(DATA_DIR / "assets/envmaps/furstenstein_2k.hdr")
    # env_map.scaling_factor = float3(50)
    env_map = scene.create_env_map(DATA_DIR / "assets/envmaps/brown_photostudio_02_2k.hdr")
    env_map.scaling_factor = float3(1)

    return scene
