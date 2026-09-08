# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from falcor2.mcp.scene_inspection import scene_to_dict, write_scene_json


class Vector(list[float]):
    pass


class Camera(SimpleNamespace):
    pass


class EnvMapLight(SimpleNamespace):
    pass


class InspectionMode(Enum):
    preview = 1


class TextureFormat(Enum):
    rgba16_float = 1


class ArrayScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class NumericArray:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values
        self.size = sum(len(row) for row in values)

    def _columns(self) -> list[list[float]]:
        return [list(column) for column in zip(*self._values, strict=True)]

    def min(self, *, axis: int) -> list[float]:
        assert axis == 0
        return [min(column) for column in self._columns()]

    def max(self, *, axis: int) -> list[float]:
        assert axis == 0
        return [max(column) for column in self._columns()]

    def mean(self, *, axis: int) -> list[float]:
        assert axis == 0
        return [sum(column) / len(column) for column in self._columns()]


class StaticMeshGeometry(SimpleNamespace):
    @property
    def sub_mesh_count(self) -> int:
        return len(self.meshes)

    def vertex_count(self, sub_mesh_index: int) -> int:
        return len(self.meshes[sub_mesh_index]["positions"]._values)

    def index_count(self, sub_mesh_index: int) -> int:
        return self.meshes[sub_mesh_index]["index_count"]

    def positions(self, sub_mesh_index: int) -> NumericArray:
        return self.meshes[sub_mesh_index]["positions"]

    def normals(self, sub_mesh_index: int) -> NumericArray:
        return self.meshes[sub_mesh_index]["normals"]

    def texcoords(self, sub_mesh_index: int) -> NumericArray:
        return self.meshes[sub_mesh_index]["texcoords"]


def _scene() -> SimpleNamespace:
    material = SimpleNamespace(name="Paint", collection_index=0)
    geometry = SimpleNamespace(name="BodyMesh", collection_index=0)
    root = SimpleNamespace(name="Root")
    camera_node = SimpleNamespace(name="Main Camera")
    camera = Camera(
        name="Camera",
        collection_index=0,
        entity=camera_node,
        fov_y=45.0,
        focal_length=35.0,
        fstop=2.8,
        sensor_height=24.0,
        enable_depth_of_field=True,
        focus_distance=4.0,
    )
    instance = SimpleNamespace(
        name="Body",
        collection_index=1,
        geometry=geometry,
        materials=[material],
    )
    transform = SimpleNamespace(
        translation=Vector([1.0, 2.0, 3.0]),
        rotation=Vector([0.0, 0.0, 0.0, 1.0]),
        scale=Vector([1.0, 1.0, 1.0]),
        matrix=[Vector([1.0, 0.0]), Vector([0.0, 1.0])],
    )
    root.parent = None
    root.children = [camera_node]
    root.components = [instance]
    root.transform = transform
    camera_node.parent = root
    camera_node.children = []
    camera_node.components = [camera]
    camera_node.transform = transform
    camera_node.world_from_object_matrix = [
        Vector([1.0, 0.0, 0.0, 0.0]),
        Vector([0.0, 1.0, 0.0, 0.0]),
        Vector([0.0, 0.0, 1.0, 0.0]),
        Vector([1.0, 2.0, 3.0, 1.0]),
    ]
    return SimpleNamespace(
        entities=[root, camera_node],
        components=[camera, instance],
        materials=[material],
        geometries=[geometry],
        animations=[],
        active_camera=camera,
    )


def test_scene_to_dict_serializes_hierarchy_and_references() -> None:
    result = scene_to_dict(_scene())

    assert result["schema_version"] == 1
    assert result["counts"] == {
        "nodes": 2,
        "components": 2,
        "materials": 1,
        "geometries": 1,
        "animations": 0,
    }
    nodes = result["nodes"]
    assert nodes[0]["child_indices"] == [1]
    assert nodes[1]["parent_index"] == 0
    assert nodes[0]["components"][0]["geometry"] == {"index": 0, "name": "BodyMesh"}
    assert result["cameras"] == [
        {
            "index": 0,
            "name": "Camera",
            "camera_path": "Main Camera",
            "node": {"index": 1, "name": "Main Camera"},
            "optics": {
                "focal_length": 35.0,
                "fstop": 2.8,
                "sensor_height": 24.0,
                "fov_y": 45.0,
                "enable_depth_of_field": True,
                "focus_distance": 4.0,
            },
            "active": True,
            "local_transform": {
                "translation": [1.0, 2.0, 3.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "world_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 2.0, 3.0, 1.0],
            ],
        }
    ]
    assert result["active_camera"] == {"index": 0, "name": "Camera"}


def test_scene_to_dict_serializes_component_runtime_properties_deterministically() -> None:
    scene = _scene()
    scene.components[1].properties = {
        "weights": Vector([0.25, 0.75]),
        "owner": scene.materials[0],
        "mode": InspectionMode.preview,
        "enabled": True,
        "matrix": [Vector([1.0, 0.0]), Vector([0.0, 1.0])],
    }

    result = scene_to_dict(scene)
    properties = result["nodes"][0]["components"][0]["properties"]

    assert list(properties) == ["enabled", "matrix", "mode", "owner", "weights"]
    assert properties == {
        "enabled": True,
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
        "mode": {"name": "preview", "value": 1},
        "owner": {"name": "Paint", "type": "SimpleNamespace"},
        "weights": [0.25, 0.75],
    }
    json.dumps(result, allow_nan=False)


def test_scene_to_dict_adds_environment_map_runtime_details(tmp_path: Path) -> None:
    scene = _scene()
    env_path = tmp_path / "sky.EXR"
    env_path.write_bytes(b"")
    transform = SimpleNamespace(
        translation=Vector([4.0, 5.0, 6.0]),
        rotation=Vector([0.0, 0.0, 0.70710678, 0.70710678]),
        scale=Vector([1.0, 1.0, 1.0]),
    )
    env_node = SimpleNamespace(
        name="Environment",
        parent=None,
        children=[],
        components=[],
        transform=transform,
        world_from_object_matrix=[
            Vector([0.0, -0.01, 0.0, 0.0]),
            Vector([0.01, 0.0, 0.0, 0.0]),
            Vector([0.0, 0.0, 0.01, 0.0]),
            Vector([4.0, 5.0, 6.0, 1.0]),
        ],
    )
    env_light = EnvMapLight(
        name="component.99",
        entity=env_node,
        active=False,
        exposure=1.0,
        env_map_path=env_path,
        env_map_texture=SimpleNamespace(
            width=2048,
            height=1024,
            depth=1,
            array_length=1,
            mip_count=1,
            format=TextureFormat.rgba16_float,
        ),
        properties={
            "active": False,
            "env_map_path": env_path,
            "exposure": 1.0,
            "intensity": Vector([0.8, 0.7, 0.6]),
            "tint": Vector([1.0, 0.5, 0.25]),
        },
    )
    env_node.components = [env_light]
    scene.entities.append(env_node)
    scene.components.append(env_light)

    result = scene_to_dict(scene)
    component = next(
        component
        for node in result["nodes"]
        for component in node["components"]
        if component["type"] == "EnvMapLight"
    )
    environment = component["environment_map"]

    assert component["properties"]["env_map_path"] == str(env_path)
    assert environment["source_path"] == str(env_path)
    assert environment["resolved_path"] == str(env_path.resolve())
    assert environment["exists"] is True
    assert environment["source_format"] == "exr"
    assert environment["lighting"] == {
        "active": False,
        "exposure": 1.0,
        "intensity": [0.8, 0.7, 0.6],
        "tint": [1.0, 0.5, 0.25],
        "effective_intensity": [1.6, 0.7, 0.3],
    }
    assert environment["texture"] == {
        "width": 2048,
        "height": 1024,
        "dimensions": [2048, 1024],
        "depth": 1,
        "array_length": 1,
        "mip_count": 1,
        "format": "rgba16_float",
        "source_format": "exr",
    }
    assert environment["transform"]["local"]["rotation"] == [
        0.0,
        0.0,
        0.70710678,
        0.70710678,
    ]
    assert environment["transform"]["world_rotation_matrix"] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    json.dumps(result, allow_nan=False)


def test_scene_to_dict_filters_nodes_by_case_insensitive_wildcard() -> None:
    result = scene_to_dict(_scene(), node_name_pattern="*camera*")

    assert [node["name"] for node in result["nodes"]] == ["Main Camera"]
    assert result["materials"] == []
    assert result["geometries"] == []
    assert result["counts"]["nodes"] == 2
    assert result["node_filter"]["matched_count"] == 1


def test_scene_to_dict_filter_includes_referenced_details_and_geometry_statistics() -> None:
    scene = _scene()
    scene.materials[0].material_id = 7
    scene.materials[0].slang_type_name = "StandardMaterial"
    scene.materials[0].flags = InspectionMode.preview
    scene.materials[0].properties = {
        "base_color_factor": Vector([0.0, 0.0, 0.0]),
        "emissive_texture_path": Path("tree.exr"),
        "roughness_factor": 0.5,
        "scalar_value": ArrayScalar(0.25),
    }
    scene.materials[0].get_this = lambda: {
        "emissive_factor": Vector([1.0, 1.0, 1.0]),
    }
    scene.geometries[0] = StaticMeshGeometry(
        name="BodyMesh",
        meshes=[
            {
                "positions": NumericArray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
                "normals": NumericArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "texcoords": NumericArray([[0.25, 0.5], [0.75, 1.0]]),
                "index_count": 6,
            }
        ],
    )
    scene.entities[0].components[0].geometry = scene.geometries[0]

    result = scene_to_dict(scene, node_name_pattern="Root")

    assert result["node_filter"] == {
        "name_pattern": "Root",
        "case_sensitive": False,
        "matched_count": 1,
        "returned_material_count": 1,
        "returned_geometry_count": 1,
    }
    assert result["materials"] == [
        {
            "index": 0,
            "name": "Paint",
            "type": "SimpleNamespace",
            "properties": {
                "base_color_factor": [0.0, 0.0, 0.0],
                "emissive_texture_path": "tree.exr",
                "roughness_factor": 0.5,
                "scalar_value": 0.25,
            },
            "material_id": 7,
            "slang_type_name": "StandardMaterial",
            "flags": {"name": "preview", "value": 1},
            "runtime": {"emissive_factor": [1.0, 1.0, 1.0]},
        }
    ]
    assert result["geometries"] == [
        {
            "index": 0,
            "name": "BodyMesh",
            "type": "StaticMeshGeometry",
            "sub_meshes": [
                {
                    "index": 0,
                    "vertex_count": 2,
                    "index_count": 6,
                    "triangle_count": 2,
                    "attributes": {
                        "positions": {
                            "min": [0.0, 1.0, 2.0],
                            "max": [2.0, 3.0, 4.0],
                            "mean": [1.0, 2.0, 3.0],
                        },
                        "normals": {
                            "min": [0.0, 0.0, 0.0],
                            "max": [1.0, 1.0, 0.0],
                            "mean": [0.5, 0.5, 0.0],
                        },
                        "texcoords": {
                            "min": [0.25, 0.5],
                            "max": [0.75, 1.0],
                            "mean": [0.5, 0.75],
                        },
                    },
                }
            ],
        }
    ]


def test_scene_to_dict_sorts_cameras_by_node_path_and_marks_inactive() -> None:
    scene = _scene()
    second_node = SimpleNamespace(
        name="A Camera",
        parent=None,
        children=[],
        components=[],
        transform=scene.entities[0].transform,
    )
    second_camera = Camera(
        name="Second Camera",
        collection_index=2,
        entity=second_node,
        fov_y=60.0,
        focal_length=24.0,
        fstop=4.0,
        sensor_height=24.0,
        enable_depth_of_field=False,
        focus_distance=2.0,
    )
    second_node.components = [second_camera]
    scene.entities.append(second_node)
    scene.components.append(second_camera)

    result = scene_to_dict(scene)

    assert [camera["camera_path"] for camera in result["cameras"]] == [
        "A Camera",
        "Main Camera",
    ]
    assert [camera["active"] for camera in result["cameras"]] == [False, True]
    assert "world_matrix" not in result["cameras"][0]


def test_scene_to_dict_is_independent_of_collection_order_and_raw_indices() -> None:
    scene = _scene()
    expected = scene_to_dict(scene)

    scene.entities.reverse()
    scene.components.reverse()
    scene.materials[0].collection_index = 19
    scene.geometries[0].collection_index = 23

    assert scene_to_dict(scene) == expected


def test_write_scene_json_creates_parent_and_valid_json(tmp_path) -> None:
    path = write_scene_json(_scene(), tmp_path / "nested" / "scene.json")

    assert path == (tmp_path / "nested" / "scene.json").resolve()
    assert json.loads(path.read_text(encoding="utf-8"))["counts"]["nodes"] == 2
