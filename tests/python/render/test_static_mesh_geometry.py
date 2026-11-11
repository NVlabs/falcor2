# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for StaticMeshGeometry numpy copy getters and setters."""

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


class _CapturingLoggerOutput(spy.LoggerOutput):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def write(self, level: spy.LogLevel, name: str, msg: str) -> None:
        self.messages.append(msg)


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    return helpers.get_device(device_type)


@pytest.fixture
def quad_mesh(device: spy.Device) -> tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]:
    scene = f2.Scene.create(device)
    geom = scene.create_geometry(f2.StaticMeshGeometry)

    data = {
        "positions": np.array(
            [
                [-0.5, 0.0, -0.5],
                [+0.5, 0.0, -0.5],
                [-0.5, 0.0, +0.5],
                [+0.5, 0.0, +0.5],
            ],
            dtype=np.float32,
        ),
        "normals": np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1)),
        "tangents": np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
        "handedness": np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        "texcoords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        # First-appearance order is 0, 1, 2, 3 so the deduplicated sub-mesh
        # vertex layout matches the input arrays.
        "indices": np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32),
    }

    geom.set_mesh_data(
        positions=data["positions"],
        sub_mesh_indices=[data["indices"]],
        normals=data["normals"],
        tangents=data["tangents"],
        handedness=data["handedness"],
        texcoords=data["texcoords"],
        name="quad",
    )

    return scene, geom, data


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestStaticMeshGeometryMutation:
    def test_sub_mesh_vertex_and_index_counts(
        self, quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]
    ):
        _, mesh, _ = quad_mesh
        assert mesh.sub_mesh_count >= 1
        assert mesh.vertex_count(0) == 4
        assert mesh.index_count(0) == 6

    def test_getters_return_expected_shapes_dtypes_and_values(
        self, quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]
    ):
        _, mesh, data = quad_mesh

        for name in ("positions", "normals", "tangents", "texcoords"):
            values = getattr(mesh, name)(0)
            assert values.shape == data[name].shape
            assert values.dtype == np.float32

        handedness = mesh.handedness(0)
        assert handedness.shape == data["handedness"].shape
        assert handedness.dtype == np.float32

        indices = mesh.indices(0)
        assert indices.shape == data["indices"].shape
        assert indices.dtype == np.uint32

        np.testing.assert_array_equal(mesh.positions(0), data["positions"])
        np.testing.assert_allclose(mesh.normals(0), data["normals"], rtol=0, atol=1e-3)
        np.testing.assert_allclose(mesh.tangents(0), data["tangents"], rtol=0, atol=1e-4)
        np.testing.assert_array_equal(mesh.handedness(0), data["handedness"])
        np.testing.assert_array_equal(mesh.texcoords(0), data["texcoords"])
        np.testing.assert_array_equal(mesh.indices(0), data["indices"])

    @pytest.mark.parametrize(
        "getter_name, replacement",
        [
            ("positions", np.full((4, 3), 7.0, dtype=np.float32)),
            ("normals", np.full((4, 3), 7.0, dtype=np.float32)),
            ("tangents", np.full((4, 3), 7.0, dtype=np.float32)),
            ("handedness", np.full((4,), 7.0, dtype=np.float32)),
            ("texcoords", np.full((4, 2), 7.0, dtype=np.float32)),
            ("indices", np.zeros((2, 3), dtype=np.uint32)),
        ],
    )
    def test_getters_return_copies(
        self,
        quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]],
        getter_name: str,
        replacement: np.ndarray,
    ):
        _, mesh, _ = quad_mesh
        getter = getattr(mesh, getter_name)

        original = getter(0).copy()
        returned = getter(0)
        returned[...] = replacement

        np.testing.assert_array_equal(getter(0), original)

    def test_setters_write_back_attributes(
        self, quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]
    ):
        _, mesh, data = quad_mesh

        positions = data["positions"] + np.array([2.0, -3.0, 0.5], dtype=np.float32)
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
        tangents = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1))
        handedness = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        texcoords = data["texcoords"] + np.array([0.25, -0.5], dtype=np.float32)
        indices = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.uint32)

        mesh.set_positions(0, positions)
        mesh.set_normals(0, normals)
        mesh.set_tangents(0, tangents)
        mesh.set_handedness(0, handedness)
        mesh.set_texcoords(0, texcoords)
        mesh.set_indices(0, indices)

        np.testing.assert_array_equal(mesh.positions(0), positions)
        np.testing.assert_allclose(mesh.normals(0), normals, rtol=0, atol=1e-3)
        np.testing.assert_allclose(mesh.tangents(0), tangents, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(mesh.handedness(0), handedness)
        np.testing.assert_array_equal(mesh.texcoords(0), texcoords)
        np.testing.assert_array_equal(mesh.indices(0), indices)

    def test_set_texcoords_updates_all_uv0_components(self, device: spy.Device):
        scene = f2.Scene.create(device)
        mesh = scene.create_geometry(f2.StaticMeshGeometry)
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        uv0 = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]], dtype=np.float32)
        updated_uv0 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=np.float32)

        mesh.set_mesh_data(
            positions,
            [indices],
            vertex_attributes=[f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 0, uv0)],
        )
        mesh.set_texcoords(0, updated_uv0)

        attribute = next(
            attribute
            for attribute in mesh.vertex_attributes(0)
            if attribute.semantic == f2.MeshAttributeSemantic.tex_coord and attribute.index == 0
        )
        np.testing.assert_array_equal(attribute.data, updated_uv0)
        np.testing.assert_array_equal(mesh.texcoords(0), updated_uv0[:, :2])

    def test_set_positions_updates_local_aabb(
        self, quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]
    ):
        _, mesh, _ = quad_mesh
        positions = np.array(
            [[2.0, 3.0, 4.0], [-1.0, 5.0, 9.0], [7.0, -2.0, 1.0], [0.0, 1.0, -3.0]],
            dtype=np.float32,
        )

        mesh.set_positions(0, positions)

        local_aabb = mesh.local_aabb
        np.testing.assert_array_equal(
            [local_aabb.min.x, local_aabb.min.y, local_aabb.min.z],
            positions.min(axis=0),
        )
        np.testing.assert_array_equal(
            [local_aabb.max.x, local_aabb.max.y, local_aabb.max.z],
            positions.max(axis=0),
        )

    @pytest.mark.parametrize(
        "setter_name, values",
        [
            ("set_positions", np.zeros((3, 3), dtype=np.float32)),
            ("set_normals", np.zeros((3, 3), dtype=np.float32)),
            ("set_tangents", np.zeros((3, 3), dtype=np.float32)),
            ("set_handedness", np.zeros((3,), dtype=np.float32)),
            ("set_texcoords", np.zeros((3, 2), dtype=np.float32)),
            ("set_indices", np.zeros((1, 3), dtype=np.uint32)),
        ],
    )
    def test_setters_reject_count_mismatches(
        self,
        quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]],
        setter_name: str,
        values: np.ndarray,
    ):
        _, mesh, _ = quad_mesh
        with pytest.raises(ValueError):
            getattr(mesh, setter_name)(0, values)

    def test_setter_dirty_state_survives_scene_update(
        self, quad_mesh: tuple[f2.Scene, f2.StaticMeshGeometry, dict[str, np.ndarray]]
    ):
        scene, mesh, data = quad_mesh
        mesh.set_texcoords(0, data["texcoords"] + np.array([1.0, 0.0], dtype=np.float32))
        scene.update()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestStaticMeshGeometryProceduralCreation:
    """Tests the set_mesh_data NumPy binding for procedural preview geometry."""

    def test_set_mesh_data_quad(self, device: spy.Device):
        scene = f2.Scene.create(device)
        geom = scene.create_geometry(f2.StaticMeshGeometry)

        positions = np.array(
            [
                [-0.5, 0.0, -0.5],
                [+0.5, 0.0, -0.5],
                [-0.5, 0.0, +0.5],
                [+0.5, 0.0, +0.5],
            ],
            dtype=np.float32,
        )
        normals = np.tile(np.array([0, 1, 0], dtype=np.float32), (4, 1))
        tangents = np.tile(np.array([1, 0, 0], dtype=np.float32), (4, 1))
        handedness = np.ones((4,), dtype=np.float32)
        texcoords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        # Indices listed so first-appearance order of unique vertices is 0,1,2,3;
        # set_mesh_data dedups/reorders by first appearance, and we want the
        # sub-mesh vertex layout to match the input arrays for easy round-trip.
        indices = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)

        geom.set_mesh_data(
            positions=positions,
            sub_mesh_indices=[indices],
            normals=normals,
            tangents=tangents,
            handedness=handedness,
            texcoords=texcoords,
            name="preview_quad",
        )

        assert geom.sub_mesh_count == 1
        assert geom.vertex_count(0) == 4
        # UVs should round-trip exactly.
        np.testing.assert_array_equal(geom.texcoords(0), texcoords)
        attributes = geom.vertex_attributes(0)
        uv0 = next(
            attribute
            for attribute in attributes
            if attribute.semantic == f2.MeshAttributeSemantic.tex_coord and attribute.index == 0
        )
        np.testing.assert_array_equal(uv0.data, texcoords)

    def test_set_mesh_data_positions_only(self, device: spy.Device):
        # Optional attributes should be accepted as None; the mesh must still
        # populate without raising.
        scene = f2.Scene.create(device)
        geom = scene.create_geometry(f2.StaticMeshGeometry)

        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)

        geom.set_mesh_data(positions=positions, sub_mesh_indices=[indices])

        assert geom.sub_mesh_count == 1
        assert geom.vertex_count(0) == 3

    def test_set_mesh_data_attribute_count_mismatch_raises(self, device: spy.Device):
        scene = f2.Scene.create(device)
        geom = scene.create_geometry(f2.StaticMeshGeometry)

        positions = np.zeros((4, 3), dtype=np.float32)
        normals_wrong = np.zeros((3, 3), dtype=np.float32)  # wrong vertex count
        indices = np.array([[0, 1, 2]], dtype=np.uint32)

        with pytest.raises(ValueError):
            geom.set_mesh_data(
                positions=positions,
                sub_mesh_indices=[indices],
                normals=normals_wrong,
            )

    @pytest.mark.parametrize(
        "name, semantic, values, default",
        [
            ("normals", f2.MeshAttributeSemantic.normal, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
            ("tangents", f2.MeshAttributeSemantic.tangent, [1.0, 0.0, 0.0], None),
            ("handedness", f2.MeshAttributeSemantic.handedness, -1.0, 0.0),
            ("texcoords", f2.MeshAttributeSemantic.tex_coord, [0.25, 0.5], [0.0, 0.0]),
        ],
    )
    def test_optional_getters_without_source_streams(
        self,
        device: spy.Device,
        name: str,
        semantic: f2.MeshAttributeSemantic,
        values: list[float] | float,
        default: list[float] | float | None,
    ):
        scene = f2.Scene.create(device)
        mesh = scene.create_geometry(f2.StaticMeshGeometry)
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        mesh.set_mesh_data(positions, [indices])

        assert all(attribute.semantic != semantic for attribute in mesh.vertex_attributes(0))
        getter = getattr(mesh, name)
        initial = getter(0)
        expected = np.array([values] * 3, dtype=np.float32)
        assert initial.shape == expected.shape
        assert initial.dtype == np.float32
        assert np.isfinite(initial).all()
        if default is not None:
            np.testing.assert_array_equal(initial, np.array([default] * 3, dtype=np.float32))

        # A source stream remains absent after legacy setters update the packed
        # vertices. Getters must return those runtime values, not invented zeros.
        getattr(mesh, f"set_{name}")(0, expected)
        assert all(attribute.semantic != semantic for attribute in mesh.vertex_attributes(0))
        actual = getter(0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        actual[...] = 7.0
        np.testing.assert_allclose(getter(0), expected, rtol=0, atol=1e-4)

        scene.update()
        np.testing.assert_allclose(getter(0), expected, rtol=0, atol=1e-4)

    def test_uv0_getter_defaults_with_only_uv1(self, device: spy.Device):
        scene = f2.Scene.create(device)
        mesh = scene.create_geometry(f2.StaticMeshGeometry)
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        uv1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        mesh.set_mesh_data(
            positions,
            [indices],
            vertex_attributes=[f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, uv1)],
        )

        np.testing.assert_array_equal(mesh.texcoords(0), np.zeros((3, 2), dtype=np.float32))
        attribute = next(
            attribute
            for attribute in mesh.vertex_attributes(0)
            if attribute.semantic == f2.MeshAttributeSemantic.tex_coord
        )
        assert attribute.index == 1
        np.testing.assert_array_equal(attribute.data, uv1)

    def test_named_vertex_attribute_identity_uses_name(self, device: spy.Device):
        scene = f2.Scene.create(device)
        geom = scene.create_geometry(f2.StaticMeshGeometry)
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        values = np.zeros((3, 2), dtype=np.float32)

        geom.set_mesh_data(
            positions,
            [indices],
            vertex_attributes=[
                f2.MeshVertexAttribute(f2.MeshAttributeSemantic.weights, 0, values, name="first"),
                f2.MeshVertexAttribute(f2.MeshAttributeSemantic.weights, 0, values, name="second"),
            ],
        )
        assert {attribute.name for attribute in geom.vertex_attributes(0)} >= {"first", "second"}

        with pytest.raises(
            RuntimeError, match="Duplicate static mesh named vertex attribute 'shared'"
        ):
            geom.set_mesh_data(
                positions,
                [indices],
                vertex_attributes=[
                    f2.MeshVertexAttribute(
                        f2.MeshAttributeSemantic.weights, 0, values, name="shared"
                    ),
                    f2.MeshVertexAttribute(
                        f2.MeshAttributeSemantic.joints, 0, values, name="shared"
                    ),
                ],
            )

    def test_set_mesh_data_retains_generic_vertex_attributes(self, device: spy.Device):
        scene = f2.Scene.create(device)
        geom = scene.create_geometry(f2.StaticMeshGeometry)
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        indices = np.array([[2, 0, 3], [3, 0, 1]], dtype=np.uint32)
        uv0 = np.array([[0, 0, 2], [1, 0, 3], [0, 1, 4], [1, 1, 5]], dtype=np.float32)
        uv6 = np.array(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            dtype=np.float32,
        )
        uv8 = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32)
        colors = np.array(
            [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [1, 1, 1, 0.4]], dtype=np.float32
        )
        colors1 = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=np.float32,
        )
        custom_values = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.uint16)
        joints = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.uint8,
        )
        attributes = [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 0, uv0),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 6, uv6),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 8, uv8, name="TEXCOORD_8"),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, colors),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 1, colors1, name="COLOR_1"),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                4,
                custom_values,
                name="matx_custom_index",
            ),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.joints,
                0,
                joints,
                name="JOINTS_0",
            ),
        ]

        log_output = _CapturingLoggerOutput()
        logger = spy.Logger.get()
        logger.add_output(log_output)
        try:
            geom.set_mesh_data(positions, [indices], vertex_attributes=attributes)
            scene.update()
        finally:
            logger.remove_output(log_output)

        unsupported_messages = [
            message
            for message in log_output.messages
            if "will not be available to shaders" in message
        ]
        assert len(unsupported_messages) == 2
        assert any("matx_custom_index" in message for message in unsupported_messages)
        assert any("JOINTS_0" in message for message in unsupported_messages)

        expected_order = np.array([2, 0, 3, 1])
        returned = {
            (attribute.semantic, attribute.index): attribute
            for attribute in geom.vertex_attributes(0)
        }
        np.testing.assert_array_equal(
            returned[(f2.MeshAttributeSemantic.tex_coord, 0)].data, uv0[expected_order]
        )
        np.testing.assert_array_equal(
            returned[(f2.MeshAttributeSemantic.tex_coord, 6)].data, uv6[expected_order]
        )
        np.testing.assert_array_equal(
            returned[(f2.MeshAttributeSemantic.tex_coord, 8)].data, uv8[expected_order]
        )
        np.testing.assert_array_equal(
            returned[(f2.MeshAttributeSemantic.color, 0)].data, colors[expected_order]
        )
        np.testing.assert_array_equal(
            returned[(f2.MeshAttributeSemantic.color, 1)].data, colors1[expected_order]
        )
        np.testing.assert_array_equal(geom.texcoords(0), uv0[expected_order, :2])
        custom = returned[(f2.MeshAttributeSemantic.unknown, 4)]
        assert custom.name == "matx_custom_index"
        assert custom.data.dtype == np.uint16
        np.testing.assert_array_equal(custom.data, custom_values[expected_order])
        joint_attribute = returned[(f2.MeshAttributeSemantic.joints, 0)]
        assert joint_attribute.name == "JOINTS_0"
        assert joint_attribute.data.dtype == np.uint8
        np.testing.assert_array_equal(joint_attribute.data, joints[expected_order])
