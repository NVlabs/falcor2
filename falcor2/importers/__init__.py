#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Importer utilities for Falcor2

Provides high-level functions for importing scenes from various formats
and printing scene structure information.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import falcor2 as f2
    import slangpy as spy
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules: {e}. "
        "Make sure Falcor2 is built and the Python environment is properly set up."
    )


def import_scene(file_path: Path) -> Optional[f2.ImporterScene]:
    """
    Import a scene from the given file path, automatically selecting the appropriate importer.

    Args:
        file_path: Path to the scene file

    Returns:
        ImporterScene if successful, None if failed

    Raises:
        ValueError: If the file format is not supported
        RuntimeError: If the import fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Scene file not found: {file_path}")

    # Get file extension
    extension = file_path.suffix.lower()

    # Choose the appropriate importer based on file extension
    if extension in [".gltf", ".glb"]:
        importer = f2.GltfImporter()
    elif extension in [".usd", ".usda", ".usdc", ".usdz"]:
        importer = f2.UsdImporter()
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: .gltf, .glb, .usd, .usda, .usdc, .usdz"
        )

    try:
        scene = importer.load_scene(file_path)
        return scene
    except Exception as e:
        raise RuntimeError(f"Failed to load scene from {file_path}: {e}")


def format_transform_matrix(transform: spy.math.float4x4, indent: int = 4) -> str:
    """Generate more human readable string version of transform matrix for printing."""
    spaces = " " * indent
    lines = []
    for row in range(4):
        row_values = [f"{transform[row][col]:8.5f}" for col in range(4)]
        lines.append(f"{spaces}[{', '.join(row_values)}]")
    return "[\n" + ",\n".join(lines) + "\n" + " " * (indent - 2) + "]"


def transform_to_list(transform: spy.math.float4x4) -> List[List[float]]:
    """Convert transform matrix to a JSON serializable format."""
    return [[float(transform[row][col]) for col in range(4)] for row in range(4)]


def get_scene_component_counts(scene: f2.ImporterScene) -> Dict[str, int]:
    """Get basic counts of scene components."""
    return {
        "meshes": len(scene.meshes),
        "materials": len(scene.materials),
        "textures": len(scene.textures),
        "cameras": len(scene.cameras),
        "lights": len(scene.lights),
        "nodes": len(scene.nodes),
        "root_nodes": len(scene.root_nodes),
    }


def analyze_meshes(scene: f2.ImporterScene, verbose: bool = False) -> Dict[str, Any]:
    """Analyze and return mesh information."""
    mesh_info = {"count": len(scene.meshes), "meshes": []}

    for i, mesh in enumerate(scene.meshes):
        mesh_data = {
            "index": i,
            "name": mesh.name,
            "vertex_count": mesh.vertex_count,
            "subgeometry_count": len(mesh.subgeometries),
            "subgeometries": [],
        }

        total_triangles = 0
        for j, subgeom in enumerate(mesh.subgeometries):
            triangle_count = len(subgeom.indices)
            total_triangles += triangle_count

            subgeom_data = {
                "index": j,
                "name": subgeom.name,
                "material_name": subgeom.material_name,
                "triangle_count": triangle_count,
            }
            mesh_data["subgeometries"].append(subgeom_data)

        mesh_data["total_triangles"] = total_triangles

        if verbose:
            # Add vertex data statistics
            if mesh.vertex_count > 0:
                positions = mesh.positions
                assert positions is not None
                mesh_data["bounds"] = {
                    "min": [float(positions[:, i].min()) for i in range(3)],
                    "max": [float(positions[:, i].max()) for i in range(3)],
                }

                # Add UV information - mesh.uvs has shape (N, 2)
                uv_attrs: list[f2.ImporterMeshAttribute] = []
                for attr in mesh.attributes:
                    if attr.semantic == f2.ImporterSemantic.tex_coord:
                        uv_attrs.append(attr)
                uv_info = {"total_uv_sets": len(uv_attrs)}
                active_uv_sets = []

                # Check if UV data contains actual values (non-zero values)
                for attr in uv_attrs:
                    uvs = mesh.get_stream(attr)
                    active_uv_sets.append(
                        {
                            "uv_set": attr.index,
                            "min": [float(uvs[:, 0].min()), float(uvs[:, 1].min())],
                            "max": [float(uvs[:, 0].max()), float(uvs[:, 1].max())],
                        }
                    )

                uv_info["active_uv_sets"] = active_uv_sets
                uv_info["active_count"] = len(active_uv_sets)
                mesh_data["uv_info"] = uv_info

                # Add vertex color information - use new API if available
                color_info = {}
                colors = mesh.colors

                if colors is not None:
                    colors_array = np.array(colors)
                    color_info["has_vertex_colors"] = True
                    color_info["min"] = [
                        float(colors_array[:, i].min()) for i in range(colors_array.shape[1])
                    ]
                    color_info["max"] = [
                        float(colors_array[:, i].max()) for i in range(colors_array.shape[1])
                    ]
                else:
                    color_info["has_vertex_colors"] = False

                mesh_data["color_info"] = color_info

        mesh_info["meshes"].append(mesh_data)

    return mesh_info


def analyze_materials(scene: f2.ImporterScene, verbose: bool = False) -> Dict[str, Any]:
    """Analyze and return material information."""
    material_info = {"count": len(scene.materials), "materials": []}

    for i, material in enumerate(scene.materials):
        material_data = {
            "index": i,
            "name": material.name,
            "parameter_count": 0,  # Will be set after counting parameters
            "parameters": {},
        }

        # Iterate over all properties using the keys() function
        param_count = 0
        for param_name in material.params.keys():
            param_count += 1
            param_value = material.params.get(param_name)
            # Convert all parameter values to strings for JSON serialization
            material_data["parameters"][param_name] = {
                "type": type(param_value).__name__,
                "value": str(param_value),
            }

        material_data["parameter_count"] = param_count

        material_info["materials"].append(material_data)

    return material_info


def analyze_textures(scene: f2.ImporterScene, verbose: bool = False) -> Dict[str, Any]:
    """Analyze and return texture information."""
    texture_info = {"count": len(scene.textures), "textures": []}

    for i, texture in enumerate(scene.textures):
        # Use only the filename to avoid absolute path differences across machines
        texture_path = Path(texture.texture_path).name if texture.texture_path != "." else "."

        texture_data = {
            "index": i,
            "source_name": texture.source_name,
            "texture_path": texture_path,
            "is_srgb": texture.is_srgb,
        }

        # Check for embedded data
        has_embedded_data = hasattr(texture, "texture_data") and len(texture.texture_data) > 0
        texture_data["has_embedded_data"] = has_embedded_data

        if has_embedded_data:
            texture_data["embedded_data_size"] = len(texture.texture_data)
            if hasattr(texture, "mime_type"):
                texture_data["mime_type"] = texture.mime_type

        if verbose:
            # Add texture transform if available
            if hasattr(texture, "texture_transform"):
                texture_data["texture_transform"] = transform_to_list(texture.texture_transform)

        texture_info["textures"].append(texture_data)

    return texture_info


def analyze_nodes(scene: f2.ImporterScene, verbose: bool = False) -> Dict[str, Any]:
    """Analyze and return node hierarchy information."""
    node_info = {
        "count": len(scene.nodes),
        "root_count": len(scene.root_nodes),
        "root_indices": list(scene.root_nodes),
        "nodes": [],
    }

    for i, node in enumerate(scene.nodes):
        node_data = {
            "index": i,
            "name": node.name,
            "mesh_index": node.mesh_index,
            "parent": node.parent,
            "children": list(node.children),
            "is_root": node.parent == -1,
            "has_mesh": node.mesh_index >= 0,
        }

        if verbose:
            node_data["transform"] = transform_to_list(node.transform)

        node_info["nodes"].append(node_data)

    # Analyze hierarchy properties
    all_are_roots = all(node.parent == -1 for node in scene.nodes)
    all_have_no_children = all(len(node.children) == 0 for node in scene.nodes)
    all_have_meshes = all(node.mesh_index >= 0 for node in scene.nodes)
    nodes_equal_roots = len(scene.nodes) == len(scene.root_nodes)

    node_info["hierarchy_analysis"] = {
        "is_flattened": all_are_roots and all_have_no_children and nodes_equal_roots,
        "all_nodes_are_roots": all_are_roots,
        "all_nodes_have_no_children": all_have_no_children,
        "all_nodes_have_meshes": all_have_meshes,
        "node_count_equals_root_count": nodes_equal_roots,
    }

    return node_info


def print_scene(
    scene: f2.ImporterScene,
    verbose: bool = False,
    nodes_only: bool = False,
    meshes_only: bool = False,
):
    """
    Print a human-readable summary of the scene structure.

    Args:
        scene: The ImporterScene to analyze
        verbose: Show detailed information including transforms and bounds
        nodes_only: Only show node hierarchy information
        meshes_only: Only show mesh information
    """
    components = get_scene_component_counts(scene)

    print("=" * 60)
    print("SCENE STRUCTURE SUMMARY")
    print("=" * 60)

    print(f"Scene Components:")
    print(f"   Meshes: {components['meshes']}")
    print(f"   Materials: {components['materials']}")
    print(f"   Textures: {components['textures']}")
    print(f"   Cameras: {components['cameras']}")
    print(f"   Lights: {components['lights']}")
    print(f"   Nodes: {components['nodes']}")
    print(f"   Root Nodes: {components['root_nodes']}")

    if not meshes_only:
        node_info = analyze_nodes(scene, verbose)
        analysis = node_info["hierarchy_analysis"]

        print(f"\nNode Hierarchy:")
        print(f"   Total Nodes: {node_info['count']}")
        print(f"   Root Nodes: {node_info['root_count']} {node_info['root_indices']}")

        print(f"\nHierarchy Analysis:")
        print(f"   Flattened: {'YES' if analysis['is_flattened'] else 'NO'}")
        print(f"   All nodes are roots: {'YES' if analysis['all_nodes_are_roots'] else 'NO'}")
        print(
            f"   All nodes have no children: {'YES' if analysis['all_nodes_have_no_children'] else 'NO'}"
        )
        print(f"   All nodes have meshes: {'YES' if analysis['all_nodes_have_meshes'] else 'NO'}")

        print(f"\nNode Details:")
        for node in node_info["nodes"]:
            print(f"   Node {node['index']}: '{node['name']}'")
            print(f"     Mesh: {node['mesh_index'] if node['has_mesh'] else 'None'}")
            print(f"     Parent: {node['parent'] if node['parent'] >= 0 else 'None'}")
            print(
                f"     Children: {len(node['children'])} {node['children'] if node['children'] else ''}"
            )

            if verbose and "transform" in node:
                print(f"     Transform:")
                for row_idx, row in enumerate(node["transform"]):
                    values = [f"{val:8.5f}" for val in row]
                    print(f"       [{', '.join(values)}]")

    # Show materials information (unless nodes-only is specified)
    if not nodes_only and components["materials"] > 0:
        material_info = analyze_materials(scene, verbose)
        print(f"\nMaterial Details:")
        for material in material_info["materials"]:
            print(f"   Material {material['index']}: '{material['name']}'")
            print(f"     Parameters: {material['parameter_count']}")

            if verbose:
                for name, param in material["parameters"].items():
                    print(f"       {name}: {param['value']}")

    # Show textures information (unless nodes-only is specified)
    if not nodes_only and components["textures"] > 0:
        texture_info = analyze_textures(scene, verbose)
        print(f"\nTexture Details:")
        for texture in texture_info["textures"]:
            print(f"   Texture {texture['index']}: '{texture['source_name']}'")
            print(f"     Path: {texture['texture_path']}")
            print(f"     sRGB: {texture['is_srgb']}")
            if texture["has_embedded_data"]:
                print(f"     Embedded: {texture['embedded_data_size']} bytes")
                if "mime_type" in texture:
                    print(f"     MIME Type: {texture['mime_type']}")

    if not nodes_only:
        mesh_info = analyze_meshes(scene, verbose)
        print(f"\nMesh Details:")
        for mesh in mesh_info["meshes"]:
            print(f"   Mesh {mesh['index']}: '{mesh['name']}'")
            print(f"     Vertices: {mesh['vertex_count']}")
            print(f"     Triangles: {mesh['total_triangles']}")
            print(f"     Subgeometries: {mesh['subgeometry_count']}")

            if verbose and "bounds" in mesh:
                bounds = mesh["bounds"]
                print(
                    f"     Bounds: [{bounds['min'][0]:.3f}, {bounds['min'][1]:.3f}, {bounds['min'][2]:.3f}] to "
                    f"[{bounds['max'][0]:.3f}, {bounds['max'][1]:.3f}, {bounds['max'][2]:.3f}]"
                )

            if verbose and "uv_info" in mesh:
                uv_info = mesh["uv_info"]
                print(f"     UV Sets: {uv_info['active_count']}/{uv_info['total_uv_sets']} active")
                for uv_set in uv_info["active_uv_sets"]:
                    print(
                        f"       UV Set {uv_set['uv_set']}: "
                        f"[{uv_set['min'][0]:.3f}, {uv_set['min'][1]:.3f}] to "
                        f"[{uv_set['max'][0]:.3f}, {uv_set['max'][1]:.3f}]"
                    )

            if verbose and "color_info" in mesh:
                color_info = mesh["color_info"]
                if color_info["has_vertex_colors"]:
                    print(
                        f"     Vertex Colors: YES ({color_info['non_white_vertex_count']}/{mesh['vertex_count']} vertices, {color_info['non_white_percentage']:.1f}%)"
                    )
                    print(
                        f"       RGBA Range: [{color_info['min'][0]:.3f}, {color_info['min'][1]:.3f}, {color_info['min'][2]:.3f}, {color_info['min'][3]:.3f}] to "
                        f"[{color_info['max'][0]:.3f}, {color_info['max'][1]:.3f}, {color_info['max'][2]:.3f}, {color_info['max'][3]:.3f}]"
                    )
                else:
                    print(f"     Vertex Colors: NO (all white/default)")
            elif verbose:
                print(f"     Vertex Colors: NO (all white/default)")

    print("=" * 60)


def get_supported_formats() -> List[str]:
    """Get a list of supported scene file formats."""
    return [".gltf", ".glb", ".usd", ".usda", ".usdc", ".usdz"]


def detect_format(file_path: Path) -> str:
    """
    Detect the format of a scene file.

    Args:
        file_path: Path to the scene file

    Returns:
        String description of the detected format

    Raises:
        ValueError: If the format is not supported
    """
    extension = Path(file_path).suffix.lower()

    format_map = {
        ".gltf": "GLTF (Text)",
        ".glb": "GLTF (Binary)",
        ".usd": "USD (Binary)",
        ".usda": "USD (ASCII)",
        ".usdc": "USD (Crate)",
        ".usdz": "USD (Archive)",
    }

    if extension in format_map:
        return format_map[extension]
    else:
        raise ValueError(f"Unsupported format: {extension}")
