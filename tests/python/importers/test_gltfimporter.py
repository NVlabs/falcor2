# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pytest
import slangpy as spy
import falcor2 as f2

DATA = Path(__file__).parent.parent.parent.parent / "data"

# Common test files used across multiple tests
COMMON_TEST_FILES = [
    "assets/kronos/Box/glTF-Binary/Box.glb"  # Model with basic materials but no textures
    # Add more test files here to run the same tests on multiple models:
    # "assets/another_model.glb",
    # "assets/complex_scene.glb",
]

# Test files with materials and textures for advanced testing
MATERIAL_TEST_FILES = [
    "assets/kronos/Avocado/glTF-Binary/Avocado.glb",  # Model with materials and embedded textures
    "assets/kronos/AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf",  # Model with alpha mode testing
]

# Test files without normals for normal generation testing
NORMAL_GENERATION_TEST_FILES = [
    "assets/kronos/TextureTransformTest/glTF/TextureTransformTest.gltf"  # Model without normals that need to be generated
]


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_load_scene(test_file: str):
    """Test basic GLTF scene loading functionality."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)
    assert scene is not None

    # Check that we have the expected number of meshes
    assert len(scene.meshes) > 0

    # Verify the first mesh has the expected geometry counts
    first_mesh = scene.meshes[0]
    assert first_mesh.vertex_count == 24  # Expected vertex count

    # Count total triangles across all subgeometries
    total_triangles = sum(len(subgeom.indices) for subgeom in first_mesh.subgeometries)
    assert total_triangles == 12  # Expected triangle count

    # Test mesh data consistency
    for mesh in scene.meshes:
        assert mesh.positions is not None
        assert mesh.normals is not None
        assert mesh.tangents is not None
        assert mesh.handedness is not None
        uv0 = mesh.texcoords(0)
        assert mesh.positions.shape[0] == mesh.vertex_count
        assert mesh.normals.shape[0] == mesh.vertex_count
        assert mesh.tangents.shape[0] == mesh.vertex_count
        assert mesh.handedness.shape[0] == mesh.vertex_count
        if uv0 is not None:
            assert uv0.shape[0] == mesh.vertex_count
            assert uv0.shape[1] == 2  # UV coordinates (u, v)

        # Note: Individual vertex access is no longer available in the new stream-based API
        # The arrays (positions, normals, etc.) are now the primary way to access vertex data

        # Verify subgeometry data
        for subgeo in mesh.subgeometries:
            # Test indices_numpy consistency
            indices1D = subgeo.indices_numpy.flatten()
            for i, tri in enumerate(subgeo.indices):
                assert (subgeo.indices_numpy[i] == tri).all()
                assert spy.math.all(spy.bool3([indices1D[i * 3 + j] for j in range(3)] == tri))


def test_invalid_file():
    """Test error handling for invalid files."""
    gltf_importer = f2.GltfImporter()

    # Test with non-existent file - should raise an exception
    with pytest.raises(RuntimeError):
        gltf_importer.load_scene(Path("non_existent_file.glb"))


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_mesh_properties(test_file: str):
    """Test specific mesh properties and data validation."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    if scene is not None and len(scene.meshes) > 0:
        mesh = scene.meshes[0]

        # Test that mesh has a name
        assert mesh.name != ""

        # Test that vertex attributes have reasonable values
        # Test positions
        assert mesh.positions is not None
        for i in range(mesh.vertex_count):
            pos = mesh.positions[i]
            assert all(abs(pos_component) < 1000.0 for pos_component in pos)

        # Test normals
        assert mesh.normals is not None
        for i in range(mesh.vertex_count):
            normal = mesh.normals[i]
            normal_length = spy.math.length(spy.float3(normal[0], normal[1], normal[2]))
            assert 0.5 <= normal_length <= 1.5

        # Test UV coordinates
        uv0 = mesh.texcoords(0)
        if uv0 is not None:
            for i in range(mesh.vertex_count):
                uv = uv0[i]
                assert -10.0 <= uv[0] <= 10.0
                assert -10.0 <= uv[1] <= 10.0

        # Test handedness should be +/-1
        assert mesh.handedness is not None
        for i in range(mesh.vertex_count):
            handedness = mesh.handedness[i]
            assert handedness == 1.0 or handedness == -1.0

        # Test subgeometries
        assert len(mesh.subgeometries) > 0
        for subgeom in mesh.subgeometries:
            assert subgeom.name != ""
            assert len(subgeom.indices) > 0

            # All triangle indices should form valid triangles (no degenerate triangles)
            for triangle in subgeom.indices:
                assert triangle.x != triangle.y
                assert triangle.y != triangle.z
                assert triangle.z != triangle.x


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_scene_structure(test_file: str):
    """Test the overall scene structure."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None

    # Test scene components
    assert hasattr(scene, "meshes")
    assert hasattr(scene, "materials")
    assert hasattr(scene, "textures")
    assert hasattr(scene, "cameras")
    assert hasattr(scene, "lights")
    assert hasattr(scene, "nodes")
    assert hasattr(scene, "root_nodes")

    # Geometry and materials loading is now implemented
    assert len(scene.meshes) > 0
    # Materials and textures may or may not be present depending on the model
    assert len(scene.materials) >= 0
    assert len(scene.textures) >= 0
    # Cameras and lights are not yet implemented, so they should be empty
    assert len(scene.cameras) == 0
    assert len(scene.lights) == 0


@pytest.mark.parametrize("test_file", MATERIAL_TEST_FILES)
def test_material_loading(test_file: str):
    """Test material loading functionality."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None
    assert len(scene.materials) > 0, "Expected model to have materials"

    # Test each material
    for i, material in enumerate(scene.materials):
        assert material.name != "", f"Material {i} should have a non-empty name"
        assert hasattr(material.params, "get"), f"Material {i} params should be a Properties object"

        # Test material parameters - Note: Properties doesn't have direct iteration,
        # so we'll test specific known parameters instead
        assert material.params.has_property("_type"), f"Material {i} should have _type parameter"

    print(f"Loaded {len(scene.materials)} materials with proper structure")


@pytest.mark.parametrize("test_file", MATERIAL_TEST_FILES)
def test_texture_loading(test_file: str):
    """Test texture loading functionality."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None
    assert len(scene.textures) > 0, "Expected model to have textures"

    # Test each texture
    for i, texture in enumerate(scene.textures):
        assert texture.source_name != "", f"Texture {i} should have a non-empty source name"
        assert texture.texture_path != "", f"Texture {i} should have a texture path"

        # Test texture data - should have either embedded data or file path
        has_embedded_data = hasattr(texture, "texture_data") and len(texture.texture_data) > 0
        has_file_path = texture.texture_path != ""

        assert (
            has_embedded_data or has_file_path
        ), f"Texture {i} should have either embedded data or file path"

        # If embedded data exists, test its properties
        if has_embedded_data:
            assert len(texture.texture_data) > 0, f"Texture {i} embedded data should not be empty"

            # Test MIME type for embedded textures
            if hasattr(texture, "mime_type"):
                valid_mime_types = [
                    "image/jpeg",
                    "image/png",
                    "image/bmp",
                    "image/gif",
                    "image/webp",
                ]
                assert (
                    texture.mime_type in valid_mime_types
                ), f"Texture {i} has invalid MIME type: {texture.mime_type}"

        # Test texture transform (should be identity or valid transformation)
        if hasattr(texture, "texture_transform"):
            transform = texture.texture_transform
            # Basic sanity check - transform should be finite
            for row in range(4):
                for col in range(4):
                    assert spy.math.isfinite(
                        transform[row][col]
                    ), f"Texture {i} transform contains non-finite values"

    print(f"Loaded {len(scene.textures)} textures with proper structure")


@pytest.mark.parametrize("test_file", MATERIAL_TEST_FILES)
def test_material_texture_consistency(test_file: str):
    """Test consistency between materials and textures."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None

    # Collect all texture indices referenced by materials
    referenced_texture_indices = set()

    for material in scene.materials:
        for param_name in material.params.keys():
            param_value = material.params.get(param_name)
            if isinstance(param_value, int):
                referenced_texture_indices.add(param_value)

    # All referenced texture indices should be valid
    for texture_idx in referenced_texture_indices:
        assert (
            0 <= texture_idx < len(scene.textures)
        ), f"Material references invalid texture index {texture_idx} (only {len(scene.textures)} textures available)"

    # Test that referenced textures are accessible
    for texture_idx in referenced_texture_indices:
        texture = scene.textures[texture_idx]
        assert texture is not None
        assert texture.source_name != ""

    print(
        f"Material-texture consistency verified: {len(referenced_texture_indices)} texture references"
    )


@pytest.mark.parametrize("test_file", MATERIAL_TEST_FILES)
def test_pbr_material_structure(test_file: str):
    """Test that PBR material parameters are loaded correctly."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None
    assert len(scene.materials) > 0

    # Expected PBR parameters that should be present in GLTF materials
    expected_pbr_params = {
        "baseColorFactor",  # Base color factor
        "metallicFactor",  # Metallic factor
        "roughnessFactor",  # Roughness factor
        "emissiveFactor",  # Emissive factor
        "_type",  # Material type identifier
    }

    # Optional parameters that might be present
    optional_pbr_params = {
        "baseColorTexture",  # Base color texture
        "baseColorTexture_texcoord",  # Base color texture coordinates
        "metallicRoughnessTexture",  # Combined metallic-roughness texture
        "metallicRoughnessTexture_texcoord",  # Metallic-roughness texture coordinates
        "normalTexture",  # Normal map
        "normalTexture_texcoord",  # Normal texture coordinates
        "normalTexture_scale",  # Normal texture scale
        "occlusionTexture",  # Occlusion texture
        "occlusionTexture_texcoord",  # Occlusion texture coordinates
        "emissiveTexture",  # Emissive texture
        "emissiveTexture_texcoord",  # Emissive texture coordinates
        "alphaMode",  # Alpha mode (OPAQUE, MASK, BLEND)
        "alphaCutoff",  # Alpha cutoff for MASK mode
        "doubleSided",  # Double-sided flag
    }

    all_pbr_params = expected_pbr_params | optional_pbr_params

    for i, material in enumerate(scene.materials):
        # Check that we have some PBR parameters
        pbr_params_found = set(material.params.keys()) & all_pbr_params
        assert (
            len(pbr_params_found) > 0
        ), f"Material {i} '{material.name}' should have at least some PBR parameters"

        # Test specific parameter types and values
        for param_name in material.params.keys():
            param_value = material.params.get(param_name)
            if param_name == "alphaMode":
                assert isinstance(param_value, f2.AlphaMode) and param_value in [
                    f2.AlphaMode.opaque,
                    f2.AlphaMode.mask,
                    f2.AlphaMode.blend,
                ], f"Material {i} alphaMode should be an AlphaMode value (opaque, mask, blend), got '{param_value}'"

            elif param_name == "doubleSided":
                assert isinstance(
                    param_value, bool
                ), f"Material {i} doubleSided should be a boolean, got '{param_value}' (type: {type(param_value)})"

            elif param_name in [
                "metallicFactor",
                "roughnessFactor",
                "alphaCutoff",
                "normalTexture_scale",
            ]:
                # These should be numeric string values or texture indices
                if isinstance(param_value, str):
                    try:
                        float(param_value)  # Should be convertible to float
                    except ValueError:
                        pytest.fail(
                            f"Material {i} {param_name} should be numeric string, got '{param_value}'"
                        )
                elif isinstance(param_value, int):
                    assert (
                        0 <= param_value < len(scene.textures)
                    ), f"Material {i} {param_name} texture index out of range"

            elif param_name == "_type":
                # Material type should be a string identifying the material type
                assert isinstance(
                    param_value, str
                ), f"Material {i} _type should be a string, got {type(param_value)}"

            elif param_name in ["baseColorFactor", "emissiveFactor"]:
                # These can be color tuples like "(r, g, b, a)" or "(r, g, b)"
                if isinstance(param_value, str):
                    # Should be a color tuple format
                    assert param_value.startswith("(") and param_value.endswith(
                        ")"
                    ), f"Material {i} {param_name} color should be in (r,g,b) format, got '{param_value}'"

    print(f"PBR material structure validated for {len(scene.materials)} materials")


@pytest.mark.parametrize("test_file", MATERIAL_TEST_FILES)
def test_mesh_material_assignment(test_file: str):
    """Test that meshes correctly reference materials."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None

    # Collect all material names from materials list
    material_names = {material.name for material in scene.materials}

    # Check that mesh subgeometries reference valid materials
    for mesh_idx, mesh in enumerate(scene.meshes):
        for subgeom_idx, subgeom in enumerate(mesh.subgeometries):
            if subgeom.material_name != "":  # Empty string means no material assigned
                assert (
                    subgeom.material_name in material_names
                ), f"Mesh {mesh_idx} subgeometry {subgeom_idx} references unknown material '{subgeom.material_name}'"

    print(f"Mesh-material assignment verified for {len(scene.meshes)} meshes")


@pytest.mark.parametrize("test_file", NORMAL_GENERATION_TEST_FILES)
def test_normal_generation(test_file: str):
    """Test that normals are properly generated for models that don't have them."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None
    assert len(scene.meshes) > 0, "Expected model to have meshes"

    # Test each mesh to verify normals are generated
    for mesh_idx, mesh in enumerate(scene.meshes):
        assert mesh.vertex_count > 0, f"Mesh {mesh_idx} should have vertices"

        # Test that all vertices have normals
        assert mesh.normals is not None, f"Mesh {mesh_idx} should have normals"
        assert (
            mesh.normals.shape[0] == mesh.vertex_count
        ), f"Mesh {mesh_idx} should have same number of normals as vertices"

        # Test that normals are not zero vectors (indicating they were properly generated)
        non_zero_normals = 0
        for vertex_idx in range(mesh.vertex_count):
            normal = mesh.normals[vertex_idx]
            normal_length = spy.math.length(spy.float3(normal[0], normal[1], normal[2]))

            # Normal should not be a zero vector
            if normal_length > 0.01:  # Small tolerance for floating point precision
                non_zero_normals += 1

                # Normal should be approximately unit length (allowing some tolerance)
                assert (
                    0.8 <= normal_length <= 1.2
                ), f"Mesh {mesh_idx} vertex {vertex_idx} normal length {normal_length} should be close to 1.0"

                # Check that normal components are finite
                assert spy.math.isfinite(
                    normal[0]
                ), f"Mesh {mesh_idx} vertex {vertex_idx} normal.x is not finite"
                assert spy.math.isfinite(
                    normal[1]
                ), f"Mesh {mesh_idx} vertex {vertex_idx} normal.y is not finite"
                assert spy.math.isfinite(
                    normal[2]
                ), f"Mesh {mesh_idx} vertex {vertex_idx} normal.z is not finite"

        # At least some normals should be non-zero (most should be, given it's a proper mesh)
        assert (
            non_zero_normals > 0
        ), f"Mesh {mesh_idx} should have at least some non-zero normals (found {non_zero_normals}/{mesh.vertex_count})"

        # For this particular test model (TextureTransformTest), which has simple quad geometry,
        # we expect most vertices to have valid normals
        if "TextureTransformTest" in test_file:
            expected_min_valid_normals = (
                mesh.vertex_count * 0.8
            )  # At least 80% should have valid normals
            assert (
                non_zero_normals >= expected_min_valid_normals
            ), f"Mesh {mesh_idx} should have at least {expected_min_valid_normals} valid normals, got {non_zero_normals}"

    print(f"Normal generation verified for {len(scene.meshes)} meshes")


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_flattened_node_hierarchy(test_file: str):
    """Test that the node hierarchy is properly flattened after loading."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None

    # Call flatten_node_hierarchy to flatten the hierarchy
    scene.flatten_node_hierarchy()

    # Basic structure tests
    assert len(scene.nodes) > 0
    assert len(scene.root_nodes) > 0

    # Flattened hierarchy properties:
    # 1. All nodes should be root nodes (no hierarchy)
    # 2. Only nodes with meshes should be present
    # 3. Number of nodes should equal number of root nodes
    assert len(scene.nodes) == len(scene.root_nodes)

    # Test each node in the flattened hierarchy
    for i, node in enumerate(scene.nodes):
        # All nodes should be root nodes (no parents, no children)
        assert node.parent == -1, f"Node {i} should be a root node (parent == -1)"
        assert len(node.children) == 0, f"Node {i} should have no children in flattened hierarchy"

        # All nodes should have valid mesh references
        assert node.mesh_index >= 0, f"Node {i} should reference a mesh"
        assert node.mesh_index < len(
            scene.meshes
        ), f"Node {i} mesh index {node.mesh_index} is out of bounds"

        # Node should have a non-empty name
        assert node.name != "", f"Node {i} should have a non-empty name"

        # Transform should be a valid 4x4 matrix with finite values
        transform = node.transform
        for row in range(4):
            for col in range(4):
                value = transform[row][col]
                assert spy.math.isfinite(
                    value
                ), f"Node {i} transform[{row}][{col}] = {value} is not finite"

    # Test root node indices
    for i, root_idx in enumerate(scene.root_nodes):
        assert 0 <= root_idx < len(scene.nodes), f"Root node index {root_idx} is out of bounds"
        assert (
            root_idx == i
        ), f"Root node indices should be sequential (expected {i}, got {root_idx})"


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_node_mesh_consistency(test_file: str):
    """Test consistency between nodes and meshes."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None

    # Collect all mesh indices referenced by nodes
    referenced_mesh_indices = set()
    for node in scene.nodes:
        if node.mesh_index >= 0:
            referenced_mesh_indices.add(node.mesh_index)

    # All referenced mesh indices should be valid
    for mesh_idx in referenced_mesh_indices:
        assert (
            0 <= mesh_idx < len(scene.meshes)
        ), f"Referenced mesh index {mesh_idx} is out of bounds"

    # There should be at least one mesh reference
    assert len(referenced_mesh_indices) > 0, "At least one mesh should be referenced by nodes"

    # Test that referenced meshes are accessible and valid
    for node in scene.nodes:
        if node.mesh_index >= 0:
            mesh = scene.meshes[node.mesh_index]
            assert mesh is not None
            assert isinstance(mesh.name, str)
            assert mesh.vertex_count > 0


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES)
def test_transform_matrices(test_file: str):
    """Test node transform matrices in detail."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)

    assert scene is not None
    assert len(scene.nodes) > 0

    for i, node in enumerate(scene.nodes):
        transform = node.transform

        # Test that all matrix elements are finite
        for row in range(4):
            for col in range(4):
                value = transform[row][col]
                assert spy.math.isfinite(
                    value
                ), f"Node {i} transform[{row}][{col}] is not finite: {value}"
                assert not spy.math.isnan(value), f"Node {i} transform[{row}][{col}] is NaN"

        # Test that the matrix has reasonable values (not too large)
        for row in range(4):
            for col in range(4):
                value = abs(transform[row][col])
                assert (
                    value < 1e6
                ), f"Node {i} transform[{row}][{col}] has very large value: {transform[row][col]}"

        # The last row should typically be [0, 0, 0, 1] for a proper affine transform
        # But we'll just check that w component is non-zero for a valid homogeneous matrix
        assert transform[3][3] != 0, f"Node {i} transform has zero w component"


@pytest.mark.parametrize("test_file", COMMON_TEST_FILES + MATERIAL_TEST_FILES)
def test_scene_structure_matches_json(test_file: str):
    """
    Test that imported scene structure matches expected JSON structure.

    This test compares the scene structure loaded by the importer with a reference
    JSON file generated by the inspect_scene.py tool. This ensures that the importer
    consistently produces the expected scene structure.

    To add more test models:
    1. Place the model file in data/assets/
    2. Generate reference JSON using: python tools/inspect_scene.py <model_path> --export-json <json_path>
    3. Add the model and JSON paths to the test_models list below
    """
    import json
    import falcor2.importers as importers
    from deepdiff import DeepDiff

    model_path = DATA / test_file
    json_path = model_path.with_suffix(".json")

    # Skip if files don't exist
    if not model_path.exists():
        pytest.skip(f"Test model not found: {model_path}")
    if not json_path.exists():
        pytest.skip(f"JSON reference not found: {json_path}")

    # Load the scene using the importer
    scene = importers.import_scene(model_path)
    assert scene is not None, f"Failed to load scene: {model_path}"

    # Load the expected JSON structure
    with open(json_path, "r") as f:
        expected_structure = json.load(f)
    del expected_structure["file_path"]  # Remove file path for comparison

    # Generate the actual scene structure using the same analysis functions
    actual_structure = {
        "components": importers.get_scene_component_counts(scene),
        "nodes": importers.analyze_nodes(scene),
        "meshes": importers.analyze_meshes(scene),
    }

    # Add materials and textures if they exist
    if actual_structure["components"]["materials"] > 0:
        actual_structure["materials"] = importers.analyze_materials(scene)

    if actual_structure["components"]["textures"] > 0:
        actual_structure["textures"] = importers.analyze_textures(scene)

    # Compare the structures using deepdiff
    # We ignore the file_path since it may have different absolute paths
    diff = DeepDiff(
        expected_structure,
        actual_structure,
        ignore_order=False,
        exclude_paths=["root['file_path']"],  # Ignore file path differences
    )

    # Assert that there are no differences
    assert not diff, f"Scene structure mismatch for {model_path}:\n{diff.pretty()}"


def test_emissive_strength_extension():
    """Test KHR_materials_emissive_strength extension loading."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(
        DATA / "assets/kronos/EmissiveStrengthTest/glTF/EmissiveStrengthTest.gltf"
    )

    assert scene is not None
    assert len(scene.materials) > 0, "Expected EmissiveStrengthTest to have materials"

    # Find materials by name and test their emissive factors
    materials_by_name = {material.name: material for material in scene.materials}

    # Test material with emissiveStrength = 4
    assert "Emit4" in materials_by_name, "Should find Emit4 material"
    emit4_material = materials_by_name["Emit4"]
    assert emit4_material.params.has_property("emissiveFactor"), "Emit4 should have emissiveFactor"
    emit4_factor = emit4_material.params.get_float3("emissiveFactor")
    # Expected: [0.1, 0.5, 0.9] * 4 = [0.4, 2.0, 3.6]
    assert abs(emit4_factor.x - 0.4) < 1e-6, f"Expected Emit4 red to be 0.4, got {emit4_factor.x}"
    assert abs(emit4_factor.y - 2.0) < 1e-6, f"Expected Emit4 green to be 2.0, got {emit4_factor.y}"
    assert abs(emit4_factor.z - 3.6) < 1e-6, f"Expected Emit4 blue to be 3.6, got {emit4_factor.z}"

    # Test material with emissiveStrength = 2
    assert "Emit2" in materials_by_name, "Should find Emit2 material"
    emit2_material = materials_by_name["Emit2"]
    assert emit2_material.params.has_property("emissiveFactor"), "Emit2 should have emissiveFactor"
    emit2_factor = emit2_material.params.get_float3("emissiveFactor")
    # Expected: [0.1, 0.5, 0.9] * 2 = [0.2, 1.0, 1.8]
    assert abs(emit2_factor.x - 0.2) < 1e-6, f"Expected Emit2 red to be 0.2, got {emit2_factor.x}"
    assert abs(emit2_factor.y - 1.0) < 1e-6, f"Expected Emit2 green to be 1.0, got {emit2_factor.y}"
    assert abs(emit2_factor.z - 1.8) < 1e-6, f"Expected Emit2 blue to be 1.8, got {emit2_factor.z}"

    # Test material with NO emissiveStrength (should default to 1.0)
    assert "Emit1" in materials_by_name, "Should find Emit1 material"
    emit1_material = materials_by_name["Emit1"]
    assert emit1_material.params.has_property("emissiveFactor"), "Emit1 should have emissiveFactor"
    emit1_factor = emit1_material.params.get_float3("emissiveFactor")
    # Expected: [0.1, 0.5, 0.9] * 1.0 = [0.1, 0.5, 0.9]
    assert abs(emit1_factor.x - 0.1) < 1e-6, f"Expected Emit1 red to be 0.1, got {emit1_factor.x}"
    assert abs(emit1_factor.y - 0.5) < 1e-6, f"Expected Emit1 green to be 0.5, got {emit1_factor.y}"
    assert abs(emit1_factor.z - 0.9) < 1e-6, f"Expected Emit1 blue to be 0.9, got {emit1_factor.z}"

    # Test material with emissiveStrength = 8
    assert "Emit8" in materials_by_name, "Should find Emit8 material"
    emit8_material = materials_by_name["Emit8"]
    assert emit8_material.params.has_property("emissiveFactor"), "Emit8 should have emissiveFactor"
    emit8_factor = emit8_material.params.get_float3("emissiveFactor")
    # Expected: [0.1, 0.5, 0.9] * 8 = [0.8, 4.0, 7.2]
    assert abs(emit8_factor.x - 0.8) < 1e-6, f"Expected Emit8 red to be 0.8, got {emit8_factor.x}"
    assert abs(emit8_factor.y - 4.0) < 1e-6, f"Expected Emit8 green to be 4.0, got {emit8_factor.y}"
    assert abs(emit8_factor.z - 7.2) < 1e-6, f"Expected Emit8 blue to be 7.2, got {emit8_factor.z}"

    # Test material with emissiveStrength = 16
    assert "Emit16" in materials_by_name, "Should find Emit16 material"
    emit16_material = materials_by_name["Emit16"]
    assert emit16_material.params.has_property(
        "emissiveFactor"
    ), "Emit16 should have emissiveFactor"
    emit16_factor = emit16_material.params.get_float3("emissiveFactor")
    # Expected: [0.1, 0.5, 0.9] * 16 = [1.6, 8.0, 14.4]
    assert (
        abs(emit16_factor.x - 1.6) < 1e-6
    ), f"Expected Emit16 red to be 1.6, got {emit16_factor.x}"
    assert (
        abs(emit16_factor.y - 8.0) < 1e-6
    ), f"Expected Emit16 green to be 8.0, got {emit16_factor.y}"
    assert (
        abs(emit16_factor.z - 14.4) < 1e-6
    ), f"Expected Emit16 blue to be 14.4, got {emit16_factor.z}"

    print("KHR_materials_emissive_strength extension test passed")


def test_gltf_material_properties_available():
    """Test that new glTF material properties are available in ImporterMaterial."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    test_file = (
        data_dir / "assets" / "kronos" / "TextureSettingsTest" / "glTF" / "TextureSettingsTest.gltf"
    )

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    importer = f2.GltfImporter()
    scene = importer.load_scene(test_file)

    assert len(scene.materials) > 0, "Scene should have materials"

    # Test that at least some materials have the new properties
    found_properties = {"doubleSided": False, "alphaMode": False, "alphaCutoff": False}

    for material in scene.materials:
        param_keys = material.params.keys()

        if "doubleSided" in param_keys:
            found_properties["doubleSided"] = True
            double_sided = material.params.get_bool("doubleSided")
            assert isinstance(double_sided, bool), "doubleSided should be a boolean"

        if "alphaMode" in param_keys:
            found_properties["alphaMode"] = True
            alpha_mode = material.params.get_enum("alphaMode", f2.AlphaMode)
            assert isinstance(
                alpha_mode, f2.AlphaMode
            ), "alphaMode should be an AlphaMode enum value"
            assert alpha_mode in [
                f2.AlphaMode.opaque,
                f2.AlphaMode.mask,
                f2.AlphaMode.blend,
            ], f"Invalid alphaMode: {alpha_mode}"

        if "alphaCutoff" in param_keys:
            found_properties["alphaCutoff"] = True
            alpha_cutoff = material.params.get_float("alphaCutoff")
            assert isinstance(alpha_cutoff, float), "alphaCutoff should be a float"
            assert 0.0 <= alpha_cutoff <= 1.0, f"alphaCutoff should be 0.0-1.0, got {alpha_cutoff}"

    # All properties should be found in the TextureSettingsTest
    for prop, found in found_properties.items():
        assert found, f"Property {prop} should be present in materials"


def test_gltf_double_sided_materials():
    """Test that double sided materials are correctly identified."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    test_file = (
        data_dir / "assets" / "kronos" / "TextureSettingsTest" / "glTF" / "TextureSettingsTest.gltf"
    )

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    importer = f2.GltfImporter()
    scene = importer.load_scene(test_file)

    # Find the DoubleSidedMaterial which should have doubleSided=True
    double_sided_material = None
    single_sided_material = None

    for material in scene.materials:
        if material.name == "DoubleSidedMaterial":
            double_sided_material = material
        elif material.name == "SingleSidedMaterial":
            single_sided_material = material

    assert double_sided_material is not None, "DoubleSidedMaterial should exist"
    assert single_sided_material is not None, "SingleSidedMaterial should exist"

    # Test the double sided values
    assert (
        double_sided_material.params.get_bool("doubleSided") is True
    ), "DoubleSidedMaterial should have doubleSided=True"
    assert (
        single_sided_material.params.get_bool("doubleSided") is False
    ), "SingleSidedMaterial should have doubleSided=False"


def test_alpha_mode_enum_values():
    """Test that AlphaMode enum is properly exposed."""

    alpha_mode = f2.AlphaMode
    assert hasattr(alpha_mode, "opaque"), "AlphaMode should have opaque"
    assert hasattr(alpha_mode, "mask"), "AlphaMode should have mask"
    assert hasattr(alpha_mode, "blend"), "AlphaMode should have blend"

    # Test enum values
    assert alpha_mode.opaque.value == 0, "AlphaMode.opaque should be 0"
    assert alpha_mode.mask.value == 1, "AlphaMode.mask should be 1"
    assert alpha_mode.blend.value == 2, "AlphaMode.blend should be 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
