# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small OBJ reader for render-material preview geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import falcor2 as f2


def _generate_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    texcoords: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    tangents = np.zeros_like(positions, dtype=np.float32)
    for i0, i1, i2 in triangles:
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        w0 = texcoords[i0]
        w1 = texcoords[i1]
        w2 = texcoords[i2]

        e1 = p1 - p0
        e2 = p2 - p0
        x1 = w1[0] - w0[0]
        x2 = w2[0] - w0[0]
        y1 = w1[1] - w0[1]
        y2 = w2[1] - w0[1]
        denom = x1 * y2 - x2 * y1
        r = 1.0 / denom if denom != 0.0 else 0.0
        tangent = (e1 * y2 - e2 * y1) * r

        tangents[i0] += tangent
        tangents[i1] += tangent
        tangents[i2] += tangent

    for index, (normal, tangent) in enumerate(zip(normals, tangents, strict=True)):
        if np.dot(tangent, tangent) > 0.0:
            tangent = tangent - normal * np.dot(normal, tangent)
            length = float(np.linalg.norm(tangent))
            tangents[index] = tangent / length if length > 0.0 else tangent
        else:
            sign = -1.0 if normal[2] < 0.0 else 1.0
            a = -1.0 / (sign + normal[2])
            b = normal[0] * normal[1] * a
            tangents[index] = np.array(
                [1.0 + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0]],
                dtype=np.float32,
            )

    return tangents.astype(np.float32)


class ObjReader:
    def read(
        self,
        path: Path | str,
        *,
        flip_v: bool = False,
        uv_origin: f2.UVOrigin = f2.UVOrigin.upper_left,
        material_name: str = "preview_material",
    ) -> f2.ImporterScene:
        path = Path(path)
        positions: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        vertices: list[tuple[int, int | None, int | None]] = []
        vertex_map: dict[tuple[int, int | None, int | None], int] = {}
        triangles: list[tuple[int, int, int]] = []
        current_name = path.stem or "obj"

        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            keyword = parts[0]
            values = parts[1:]
            if keyword == "v":
                if len(values) < 3:
                    raise ValueError(f"{path}:{line_number}: vertex position needs 3 values")
                positions.append((float(values[0]), float(values[1]), float(values[2])))
            elif keyword == "vt":
                if len(values) < 2:
                    raise ValueError(f"{path}:{line_number}: texture coordinate needs 2 values")
                v = float(values[1])
                texcoords.append((float(values[0]), 1.0 - v if flip_v else v))
            elif keyword == "vn":
                if len(values) < 3:
                    raise ValueError(f"{path}:{line_number}: normal needs 3 values")
                normals.append((float(values[0]), float(values[1]), float(values[2])))
            elif keyword in {"g", "o"} and values:
                current_name = values[0]
            elif keyword == "f":
                if len(values) < 3:
                    raise ValueError(f"{path}:{line_number}: face needs at least 3 vertices")
                face = [
                    self._parse_face_vertex(
                        token, len(positions), len(texcoords), len(normals), path, line_number
                    )
                    for token in values
                ]
                face_indices = []
                for key in face:
                    index = vertex_map.get(key)
                    if index is None:
                        index = len(vertices)
                        vertex_map[key] = index
                        vertices.append(key)
                    face_indices.append(index)
                for index in range(1, len(face_indices) - 1):
                    triangles.append(
                        (face_indices[0], face_indices[index], face_indices[index + 1])
                    )

        if not vertices or not triangles:
            raise ValueError(f"{path}: OBJ file did not contain any faces")

        has_texcoords = any(vertex[1] is not None for vertex in vertices)
        has_normals = any(vertex[2] is not None for vertex in vertices)
        if has_texcoords and any(vertex[1] is None for vertex in vertices):
            raise ValueError(
                f"{path}: mixed faces with and without texture coordinates are unsupported"
            )
        if has_normals and any(vertex[2] is None for vertex in vertices):
            raise ValueError(f"{path}: mixed faces with and without normals are unsupported")

        position_array = np.array([positions[vertex[0]] for vertex in vertices], dtype=np.float32)
        attributes: dict[str, np.ndarray] = {"position": position_array}
        if has_normals:
            normal_array = np.array([normals[vertex[2]] for vertex in vertices], dtype=np.float32)
            attributes["normal"] = normal_array
        if has_texcoords:
            texcoord_array = np.array(
                [texcoords[vertex[1]] for vertex in vertices], dtype=np.float32
            )
            attributes["tex_coord"] = texcoord_array

        triangle_array = np.array(triangles, dtype=np.uint32)
        if has_normals and has_texcoords:
            attributes["tangent"] = _generate_tangents(
                position_array,
                normal_array,
                texcoord_array,
                triangle_array,
            )
            # MaterialX OBJ loading derives bitangents as normal.cross(tangent), so
            # Falcor's reconstructed bitangent matches that convention with +1 handedness.
            attributes["handedness"] = np.ones((position_array.shape[0],), dtype=np.float32)

        mesh = f2.ImporterMesh.create(
            [triangle_array],
            attributes,
        )
        mesh.name = path.stem or "preview_mesh"
        mesh.uv_origin = uv_origin

        subgeometries = list(mesh.subgeometries)
        subgeometries[0].name = current_name
        subgeometries[0].material_name = material_name
        mesh.subgeometries = subgeometries

        material = f2.ImporterMaterial()
        material.name = material_name
        material.params = f2.Properties({"_scene_material_type": "StandardMaterial"})

        node = f2.ImporterNode()
        node.name = mesh.name
        node.mesh_index = 0

        scene = f2.ImporterScene()
        scene.uv_origin = uv_origin
        scene.materials = [material]
        scene.meshes = [mesh]
        scene.nodes = [node]
        scene.root_nodes = [0]
        scene.calculate_aabbs()
        return scene

    @staticmethod
    def _parse_face_vertex(
        token: str,
        position_count: int,
        texcoord_count: int,
        normal_count: int,
        path: Path,
        line_number: int,
    ) -> tuple[int, int | None, int | None]:
        values = token.split("/")
        if not values[0]:
            raise ValueError(f"{path}:{line_number}: face vertex is missing a position index")

        position = ObjReader._resolve_index(values[0], position_count, path, line_number)
        texcoord = None
        normal = None
        if len(values) > 1 and values[1]:
            texcoord = ObjReader._resolve_index(values[1], texcoord_count, path, line_number)
        if len(values) > 2 and values[2]:
            normal = ObjReader._resolve_index(values[2], normal_count, path, line_number)
        if len(values) > 3:
            raise ValueError(f"{path}:{line_number}: unsupported face vertex '{token}'")
        return position, texcoord, normal

    @staticmethod
    def _resolve_index(value: str, count: int, path: Path, line_number: int) -> int:
        index = int(value)
        if index == 0:
            raise ValueError(f"{path}:{line_number}: OBJ indices are 1-based")
        resolved = count + index if index < 0 else index - 1
        if resolved < 0 or resolved >= count:
            raise ValueError(f"{path}:{line_number}: OBJ index {value} is out of range")
        return resolved
