# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2


@pytest.mark.parametrize("call", ["omitted", "none", "zero", "midpoint"])
def test_bake_pose_optional_time(call: str) -> None:
    scene = f2.ImporterScene()
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    scene.meshes = [
        f2.ImporterMesh.create(
            [np.array([[0, 1, 2]], dtype=np.uint32)],
            {
                "position": positions,
                "joints": np.zeros((3, 4), dtype=np.int32),
                "weights": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
            },
        )
    ]
    mesh_node = f2.ImporterNode()
    mesh_node.mesh_index = mesh_node.skin_index = 0
    joint_node = f2.ImporterNode()
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = 2
    joint_node.transform = spy.float4x4(matrix)
    joint_node.animation_translation = 0
    scene.nodes = [mesh_node, joint_node]
    scene.root_nodes = [0, 1]
    skin = f2.ImporterSkin()
    skin.joints = [1]
    skin.joint_from_bind = [spy.float4x4(np.eye(4, dtype=np.float32))]
    scene.skins = [skin]
    channel = f2.ImporterAnimationChannel()
    channel.value_type = f2.AnimationValueType.float3
    channel.times = [0, 2]
    channel.values = [4, 0, 0, 8, 0, 0]
    scene.animation.channels = [channel]

    if call == "omitted":
        scene.bake_pose()
    else:
        scene.bake_pose(time_seconds={"none": None, "zero": 0.0, "midpoint": 1.0}[call])
    expected_translation = {"omitted": 2, "none": 2, "zero": 4, "midpoint": 6}[call]
    np.testing.assert_allclose(scene.meshes[0].positions, positions + [expected_translation, 0, 0])
    assert not scene.skins
    assert scene.nodes[0].skin_index == -1
    assert scene.nodes[1].animation_translation == 0
    np.testing.assert_array_equal(scene.nodes[1].transform.to_numpy(), matrix)
