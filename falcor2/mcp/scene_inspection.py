# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

JsonObject = dict[str, object]
_AUTO_COMPONENT_NAME = re.compile(r"component\.\d+", flags=re.ASCII)


def _type_name(value: object) -> str:
    return type(value).__name__


def _sequence(value: object) -> list[object]:
    """Convert Slang vector and quaternion values to JSON arrays."""
    try:
        length = len(value)  # type: ignore[arg-type]
    except TypeError:
        components = [name for name in ("x", "y", "z", "w") if hasattr(value, name)]
        return [getattr(value, name) for name in components]

    result: list[object] = []
    for index in range(length):
        item = value[index]  # type: ignore[index]
        if isinstance(item, (bool, int, float, str)):
            result.append(item)
        else:
            nested = _sequence(item)
            result.append(nested if nested else str(item))
    return result


def _matrix(value: object) -> list[list[float]]:
    """Convert a Slang 4x4 matrix or nested sequence to JSON rows."""
    matrix: Any = value
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _rotation_matrix(world_matrix: list[list[float]]) -> list[list[float]]:
    """Remove per-axis scale from a column-vector world transform basis."""
    result = [row[:3] for row in world_matrix[:3]]
    for column in range(3):
        length = math.sqrt(sum(result[row][column] ** 2 for row in range(3)))
        if length > 0.0:
            for row in range(3):
                result[row][column] /= length
    return result


def _enum_summary(value: object) -> JsonObject:
    name = getattr(value, "name", None)
    enum_value = getattr(value, "value", None)
    return {
        "name": str(name) if name is not None else str(value),
        "value": int(enum_value) if isinstance(enum_value, int) else str(enum_value),
    }


def _safe_attribute(value: object, name: str) -> object | None:
    try:
        return getattr(value, name)
    except (AttributeError, RuntimeError):
        return None


def _format_name(value: object) -> str:
    name = _safe_attribute(value, "name")
    if name is not None:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _texture_summary(value: object | None) -> JsonObject:
    """Describe an already available runtime texture without causing it to load."""
    if value is None:
        return {}

    texture = value
    width = _safe_attribute(texture, "width")
    height = _safe_attribute(texture, "height")
    if width is None or height is None:
        nested_texture = _safe_attribute(texture, "texture")
        if nested_texture is not None:
            texture = nested_texture
            width = _safe_attribute(texture, "width")
            height = _safe_attribute(texture, "height")

    result: JsonObject = {}
    if isinstance(width, int) and not isinstance(width, bool):
        result["width"] = width
    if isinstance(height, int) and not isinstance(height, bool):
        result["height"] = height
    if "width" in result and "height" in result:
        result["dimensions"] = [result["width"], result["height"]]

    for name in ("depth", "array_length", "mip_count"):
        field_value = _safe_attribute(texture, name)
        if isinstance(field_value, int) and not isinstance(field_value, bool):
            result[name] = field_value

    format_value = _safe_attribute(texture, "format")
    if format_value is not None:
        result["format"] = _format_name(format_value)

    texture_path = _safe_attribute(value, "path")
    if texture_path:
        result["path"] = str(texture_path)
    return result


def _json_value(value: object) -> object:
    """Convert a reflected runtime property to deterministic JSON data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _enum_summary(value)

    item = _safe_attribute(value, "item")
    if callable(item):
        try:
            scalar = item()
        except (RuntimeError, TypeError, ValueError):
            scalar = value
        if scalar is not value and isinstance(scalar, (bool, int, float, str)):
            return _json_value(scalar)

    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    texture = _texture_summary(value)
    if texture.get("dimensions") is not None:
        return {"type": _type_name(value), **texture}

    try:
        length = len(value)  # type: ignore[arg-type]
    except TypeError:
        length = None
    if length is not None:
        return [_json_value(value[index]) for index in range(length)]  # type: ignore[index]

    name = _safe_attribute(value, "name")
    if name is not None:
        return {"name": str(name), "type": _type_name(value)}
    return {"type": _type_name(value)}


def _runtime_properties(component: Any) -> JsonObject:
    properties = _safe_attribute(component, "properties")
    if properties is None:
        return {}

    if isinstance(properties, Mapping):
        items = properties.items()
    else:
        items_method = _safe_attribute(properties, "items")
        if not callable(items_method):
            return {}
        items = items_method()
    return {
        str(name): _json_value(value)
        for name, value in sorted(items, key=lambda item: str(item[0]))
    }


def _rgb_value(value: object, default: float) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * 3
    if isinstance(value, list) and len(value) >= 3:
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value[:3]):
            return [float(item) for item in value[:3]]
    return [default] * 3


def _environment_map_summary(component: Any, properties: JsonObject) -> JsonObject:
    result: JsonObject = {}

    path_value = _safe_attribute(component, "env_map_path")
    if path_value is None:
        path_value = properties.get("env_map_path")
    if isinstance(path_value, (str, Path)) and str(path_value):
        source_path = Path(path_value).expanduser()
        try:
            resolved_path = source_path.resolve(strict=False)
        except OSError:
            resolved_path = source_path.absolute()
        result["source_path"] = str(source_path)
        result["resolved_path"] = str(resolved_path)
        result["exists"] = resolved_path.is_file()
        if source_path.suffix:
            result["source_format"] = source_path.suffix[1:].lower()

    active_value = _safe_attribute(component, "active")
    if not isinstance(active_value, bool):
        active_value = properties.get("active", True)
    exposure_value = _safe_attribute(component, "exposure")
    if not isinstance(exposure_value, (int, float)) or isinstance(exposure_value, bool):
        exposure_value = properties.get("exposure", 0.0)
    exposure = float(exposure_value) if isinstance(exposure_value, (int, float)) else 0.0
    exposure_scale = 2.0**exposure
    tint = _rgb_value(properties.get("tint", properties.get("color")), 1.0)
    base_intensity = _rgb_value(properties.get("intensity"), 1.0)
    result["lighting"] = {
        "active": bool(active_value),
        "exposure": exposure,
        "intensity": base_intensity,
        "tint": tint,
        "effective_intensity": [
            base_intensity[index] * tint[index] * exposure_scale for index in range(3)
        ],
    }

    texture_value = None
    for name in ("env_map_texture_handle", "env_map_texture", "texture"):
        texture_value = _safe_attribute(component, name)
        if texture_value is not None:
            break
    texture = _texture_summary(texture_value)
    if source_path_suffix := result.get("source_format"):
        texture.setdefault("source_format", source_path_suffix)
    if texture:
        result["texture"] = texture

    entity = _safe_attribute(component, "entity")
    if entity is not None:
        transform_summary: JsonObject = {}
        local_transform = _safe_attribute(entity, "transform")
        if local_transform is not None:
            transform_summary["local"] = _transform_summary(local_transform)
        world_matrix_value = _safe_attribute(entity, "world_from_object_matrix")
        if world_matrix_value is not None:
            world_matrix = _matrix(world_matrix_value)
            transform_summary["world_matrix"] = world_matrix
            transform_summary["world_rotation_matrix"] = _rotation_matrix(world_matrix)
        if transform_summary:
            result["transform"] = transform_summary
    return result


def _object_summary(value: Any, index: int, *, name: str | None = None) -> JsonObject:
    return {
        "index": index,
        "name": str(value.name) if name is None else name,
        "type": _type_name(value),
    }


def _material_summary(value: Any, index: int) -> JsonObject:
    result = _object_summary(value, index)
    properties = _runtime_properties(value)
    if properties:
        result["properties"] = properties

    for name in ("material_id", "slang_type_name", "flags"):
        attribute = _safe_attribute(value, name)
        if attribute is not None:
            result[name] = _json_value(attribute)

    get_this = _safe_attribute(value, "get_this")
    if callable(get_this):
        try:
            runtime = get_this()
        except RuntimeError:
            runtime = None
        if runtime is not None:
            result["runtime"] = _json_value(runtime)
    return result


def _numeric_range(value: object) -> JsonObject:
    size = _safe_attribute(value, "size")
    if not isinstance(size, int) or isinstance(size, bool) or size == 0:
        return {}

    result: JsonObject = {}
    for output_name, method_name in (("min", "min"), ("max", "max"), ("mean", "mean")):
        method = _safe_attribute(value, method_name)
        if not callable(method):
            continue
        try:
            result[output_name] = _json_value(method(axis=0))
        except (RuntimeError, TypeError, ValueError):
            continue
    return result


def _geometry_summary(value: Any, index: int, *, include_attributes: bool) -> JsonObject:
    result = _object_summary(value, index)
    sub_mesh_count = _safe_attribute(value, "sub_mesh_count")
    if not isinstance(sub_mesh_count, int) or isinstance(sub_mesh_count, bool):
        return result

    sub_meshes: list[JsonObject] = []
    for sub_mesh_index in range(sub_mesh_count):
        sub_mesh: JsonObject = {"index": sub_mesh_index}
        for output_name, method_name in (
            ("vertex_count", "vertex_count"),
            ("index_count", "index_count"),
        ):
            method = _safe_attribute(value, method_name)
            if not callable(method):
                continue
            try:
                count = int(method(sub_mesh_index))
            except (IndexError, RuntimeError, TypeError, ValueError):
                continue
            sub_mesh[output_name] = count
            if output_name == "index_count":
                sub_mesh["triangle_count"] = count // 3

        if include_attributes:
            attributes: JsonObject = {}
            for name in ("positions", "normals", "texcoords"):
                method = _safe_attribute(value, name)
                if not callable(method):
                    continue
                try:
                    value_range = _numeric_range(method(sub_mesh_index))
                except (IndexError, RuntimeError, TypeError, ValueError):
                    continue
                if value_range:
                    attributes[name] = value_range
            if attributes:
                sub_mesh["attributes"] = attributes
        sub_meshes.append(sub_mesh)

    result["sub_meshes"] = sub_meshes
    return result


def _component_identity(component: Any, owner_name: str) -> tuple[object, ...]:
    name = str(component.name)
    stable_name = "" if _AUTO_COMPONENT_NAME.fullmatch(name) else name
    geometry = getattr(component, "geometry", None)
    geometry_name = str(geometry.name) if geometry is not None else ""
    material_names = tuple(
        sorted(
            str(material.name)
            for material in getattr(component, "materials", [])
            if material is not None
        )
    )
    runtime_properties = json.dumps(
        _runtime_properties(component),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        owner_name,
        _type_name(component),
        stable_name,
        geometry_name,
        material_names,
        runtime_properties,
    )


def _component_summary(
    component: Any,
    index: int,
    geometry_indices: dict[int, int],
    material_indices: dict[int, int],
) -> JsonObject:
    component_name = str(component.name)
    if _AUTO_COMPONENT_NAME.fullmatch(component_name):
        component_name = f"component.{index}"
    result = _object_summary(component, index, name=component_name)
    properties = _runtime_properties(component)
    result["properties"] = properties
    if hasattr(component, "geometry"):
        geometry = component.geometry
        if geometry is not None:
            result["geometry"] = {
                "index": geometry_indices[id(geometry)],
                "name": str(geometry.name),
            }
    if hasattr(component, "materials"):
        result["materials"] = [
            {"index": material_indices[id(material)], "name": str(material.name)}
            for material in component.materials
            if material is not None
        ]
    if "env_map_path" in properties or _safe_attribute(component, "env_map_path") is not None:
        result["environment_map"] = _environment_map_summary(component, properties)
    return result


def _transform_summary(transform: Any) -> JsonObject:
    return {
        "translation": _sequence(transform.translation),
        "rotation": _sequence(transform.rotation),
        "scale": _sequence(transform.scale),
    }


def _camera_summary(
    camera: Any,
    component_index: int,
    entity_indices: dict[int, int],
    active_camera: Any,
) -> JsonObject:
    entity = camera.entity
    camera_path = str(entity.name)
    camera_name = str(camera.name)
    if _AUTO_COMPONENT_NAME.fullmatch(camera_name):
        camera_name = f"component.{component_index}"
    result: JsonObject = {
        "index": component_index,
        "name": camera_name,
        "camera_path": camera_path,
        "node": {
            "index": entity_indices[id(entity)],
            "name": camera_path,
        },
        "optics": {
            "focal_length": float(camera.focal_length),
            "fstop": float(camera.fstop),
            "sensor_height": float(camera.sensor_height),
            "fov_y": float(camera.fov_y),
            "enable_depth_of_field": bool(camera.enable_depth_of_field),
            "focus_distance": float(camera.focus_distance),
        },
        "active": camera is active_camera,
    }
    transform = getattr(entity, "transform", None)
    if transform is not None:
        result["local_transform"] = _transform_summary(transform)
    world_matrix = getattr(entity, "world_from_object_matrix", None)
    if world_matrix is not None:
        result["world_matrix"] = _matrix(world_matrix)
    return result


def _matches_name(name: str, pattern: str | None, case_sensitive: bool) -> bool:
    if pattern is None:
        return True
    if case_sensitive:
        return fnmatch.fnmatchcase(name, pattern)
    return fnmatch.fnmatchcase(name.casefold(), pattern.casefold())


def scene_to_dict(
    scene: Any,
    *,
    node_name_pattern: str | None = None,
    case_sensitive: bool = False,
) -> JsonObject:
    """Return a JSON-compatible snapshot of a live Falcor2 scene."""
    if node_name_pattern is not None and not node_name_pattern:
        raise ValueError("node_name_pattern must be a non-empty string when provided")

    unordered_entities = list(scene.entities)
    component_owner_names = {
        id(component): str(entity.name)
        for entity in unordered_entities
        for component in entity.components
    }

    def entity_key(entity: Any) -> tuple[object, ...]:
        return (
            str(entity.name),
            tuple(
                sorted(
                    _component_identity(component, str(entity.name))
                    for component in entity.components
                )
            ),
            tuple(sorted(str(child.name) for child in entity.children)),
        )

    unordered_entity_ids = {id(entity) for entity in unordered_entities}
    entities: list[Any] = []
    visited_entity_ids: set[int] = set()

    def append_subtree(entity: Any) -> None:
        if id(entity) in visited_entity_ids or id(entity) not in unordered_entity_ids:
            return
        visited_entity_ids.add(id(entity))
        entities.append(entity)
        for child in sorted(entity.children, key=entity_key):
            append_subtree(child)

    roots = [
        entity
        for entity in unordered_entities
        if entity.parent is None or id(entity.parent) not in unordered_entity_ids
    ]
    for root in sorted(roots, key=entity_key):
        append_subtree(root)
    for entity in sorted(unordered_entities, key=entity_key):
        append_subtree(entity)
    entity_indices = {id(entity): index for index, entity in enumerate(entities)}
    materials = sorted(scene.materials, key=lambda value: (str(value.name), _type_name(value)))
    material_indices = {id(value): index for index, value in enumerate(materials)}
    geometries = sorted(scene.geometries, key=lambda value: (str(value.name), _type_name(value)))
    geometry_indices = {id(value): index for index, value in enumerate(geometries)}
    animations = sorted(scene.animations, key=lambda value: (str(value.name), _type_name(value)))
    components = sorted(
        scene.components,
        key=lambda component: _component_identity(
            component,
            component_owner_names.get(id(component), ""),
        ),
    )
    component_indices = {id(component): index for index, component in enumerate(components)}

    nodes: list[JsonObject] = []
    referenced_material_ids: set[int] = set()
    referenced_geometry_ids: set[int] = set()
    for index, entity in enumerate(entities):
        name = str(entity.name)
        if not _matches_name(name, node_name_pattern, case_sensitive):
            continue
        for component in entity.components:
            geometry = _safe_attribute(component, "geometry")
            if geometry is not None:
                referenced_geometry_ids.add(id(geometry))
            component_materials = _safe_attribute(component, "materials")
            if component_materials is not None:
                referenced_material_ids.update(
                    id(material) for material in component_materials if material is not None
                )
        parent = entity.parent
        nodes.append(
            {
                "index": index,
                "name": name,
                "parent_index": entity_indices.get(id(parent)) if parent is not None else None,
                "child_indices": sorted(entity_indices[id(child)] for child in entity.children),
                "transform": _transform_summary(entity.transform),
                "components": [
                    _component_summary(
                        component,
                        component_indices[id(component)],
                        geometry_indices,
                        material_indices,
                    )
                    for component in sorted(
                        entity.components,
                        key=lambda value: component_indices[id(value)],
                    )
                ],
            }
        )

    active_camera = scene.active_camera
    cameras = [
        _camera_summary(component, index, entity_indices, active_camera)
        for index, component in enumerate(components)
        if _type_name(component) == "Camera"
    ]
    cameras.sort(key=lambda camera: (str(camera["camera_path"]), str(camera["name"])))
    if node_name_pattern is None:
        returned_materials = materials
        returned_geometries = geometries
    else:
        returned_materials = [value for value in materials if id(value) in referenced_material_ids]
        returned_geometries = [
            value for value in geometries if id(value) in referenced_geometry_ids
        ]

    result: JsonObject = {
        "schema_version": 1,
        "counts": {
            "nodes": len(entities),
            "components": len(components),
            "materials": len(materials),
            "geometries": len(geometries),
            "animations": len(animations),
        },
        "nodes": nodes,
        "materials": [
            (
                _material_summary(value, material_indices[id(value)])
                if node_name_pattern is not None
                else _object_summary(value, material_indices[id(value)])
            )
            for value in returned_materials
        ],
        "geometries": [
            _geometry_summary(
                value,
                geometry_indices[id(value)],
                include_attributes=node_name_pattern is not None,
            )
            for value in returned_geometries
        ],
        "animations": [_object_summary(value, index) for index, value in enumerate(animations)],
        "cameras": cameras,
        "active_camera": None,
    }
    if active_camera is not None:
        active_camera_name = str(active_camera.name)
        if _AUTO_COMPONENT_NAME.fullmatch(active_camera_name):
            active_camera_name = f"component.{component_indices[id(active_camera)]}"
        result["active_camera"] = {
            "index": component_indices[id(active_camera)],
            "name": active_camera_name,
        }
    if node_name_pattern is not None:
        result["node_filter"] = {
            "name_pattern": node_name_pattern,
            "case_sensitive": case_sensitive,
            "matched_count": len(nodes),
            "returned_material_count": len(returned_materials),
            "returned_geometry_count": len(returned_geometries),
        }
    return result


def write_scene_json(
    scene: Any,
    path: Path,
    *,
    node_name_pattern: str | None = None,
    case_sensitive: bool = False,
) -> Path:
    """Write a live scene snapshot to a UTF-8 JSON file and return its absolute path."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scene_to_dict(
        scene,
        node_name_pattern=node_name_pattern,
        case_sensitive=case_sensitive,
    )
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
