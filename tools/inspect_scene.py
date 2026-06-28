#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scene Inspector Tool

A utility for inspecting and logging the structure of scenes loaded through Falcor2.
Useful for debugging, testing, and understanding scene hierarchies.

Usage:
    python tools/inspect_scene.py <path_to_scene_file>
    python tools/inspect_scene.py data/assets/objaverse/00e4a92210b04f4a9707cc89ddac0e63.glb

Options:
    --verbose, -v    Show detailed information including transform matrices
    --nodes-only     Only show node hierarchy information
    --meshes-only    Only show mesh information
    --export-json    Export scene structure to JSON file
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import falcor2.importers as importers


def main():
    parser = argparse.ArgumentParser(
        description="Inspect scene structure loaded through Falcor2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/inspect_scene.py data/assets/objaverse/00e4a92210b04f4a9707cc89ddac0e63.glb
  python tools/inspect_scene.py scene.glb --verbose
  python tools/inspect_scene.py scene.glb --export-json scene_info.json
  python tools/inspect_scene.py scene.glb --nodes-only
        """,
    )

    parser.add_argument("file_path", help="Path to the scene file to inspect")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information including transforms and bounds",
    )
    parser.add_argument(
        "--nodes-only", action="store_true", help="Only show node hierarchy information"
    )
    parser.add_argument("--meshes-only", action="store_true", help="Only show mesh information")
    parser.add_argument("--export-json", metavar="FILE", help="Export scene structure to JSON file")

    args = parser.parse_args()

    # Validate file path
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    file_path = file_path.absolute()

    # Load the scene
    print(f"Loading scene: {file_path}")
    try:
        scene = importers.import_scene(file_path)

        if scene is None:
            print("Error: Failed to load scene.")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading scene: {e}")
        sys.exit(1)

    # Output results
    if args.export_json:
        # Gather scene information for JSON export
        scene_info = {
            "file_path": str(file_path),
            "components": importers.get_scene_component_counts(scene),
        }

        if not args.meshes_only:
            scene_info["nodes"] = importers.analyze_nodes(scene, args.verbose)

        if not args.nodes_only:
            scene_info["meshes"] = importers.analyze_meshes(scene, args.verbose)

        # Always include materials and textures if they exist
        if scene_info["components"]["materials"] > 0:
            scene_info["materials"] = importers.analyze_materials(scene, args.verbose)

        if scene_info["components"]["textures"] > 0:
            scene_info["textures"] = importers.analyze_textures(scene, args.verbose)

        json_path = Path(args.export_json)
        with open(json_path, "w") as f:
            json.dump(scene_info, f, indent=2)
        print(f"Scene structure exported to: {json_path}")

    # Always print summary unless exporting to JSON only
    if not args.export_json or True:  # Always show summary for now
        importers.print_scene(scene, args.verbose, args.nodes_only, args.meshes_only)


if __name__ == "__main__":
    main()
