# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
import falcor2 as f2
from falcor2 import pyscene


@pytest.fixture(autouse=True)
def clear_current_importer() -> Iterator[None]:
    f2.Importer.clear_current()
    yield
    f2.Importer.clear_current()


def test_collections_follow_current_importer_after_clear() -> None:
    camera = pyscene.cameras.create(name="First")
    pyscene.nodes.create(name="First", camera=camera)

    scene = f2.Importer.get().build_importer_scene()
    assert [camera.name for camera in scene.cameras] == ["First"]
    assert [node.name for node in scene.nodes] == ["First"]

    f2.Importer.clear_current()
    pyscene.cameras.create(name="Second")

    scene = f2.Importer.get().build_importer_scene()
    assert [camera.name for camera in scene.cameras] == ["Second"]
