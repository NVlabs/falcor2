# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from typing import Optional

ET.register_namespace("", "http://schemas.microsoft.com/vstudio/debugger/natvis/2010")


def merge_natvis_files(natvis_paths: list[Path], output_path: Optional[Path]):
    seen_types = set()
    namespace = "http://schemas.microsoft.com/vstudio/debugger/natvis/2010"
    auto_visualizer = ET.Element("AutoVisualizer")
    namespaces = {"vis": namespace}

    for natvis_path in natvis_paths:
        print(f"Processing {natvis_path}")
        tree = ET.parse(natvis_path)
        root = tree.getroot()
        for type_element in root.iterfind("vis:Type", namespaces=namespaces):
            type_name = type_element.get("Name")
            if type_name not in seen_types:
                seen_types.add(type_name)
                auto_visualizer.append(type_element)

    tree = ET.ElementTree(auto_visualizer)
    ET.indent(tree, space="\t", level=0)
    if output_path:
        tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")
    else:
        tree.write(sys.stdout, encoding="unicode", xml_declaration=True, method="xml")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Merge multiple natvis files into one.")
    parse.add_argument("natvis_files", nargs="+", help="Input .natvis files to merge")
    parse.add_argument("-o", "--output_file", required=False, help="Output .natvis file path")

    args = parse.parse_args()
    natvis_paths = [Path(p) for p in args.natvis_files]
    output_path = Path(args.output_file) if args.output_file else None
    merge_natvis_files(natvis_paths, output_path)
