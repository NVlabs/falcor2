# SPDX-License-Identifier: Apache-2.0
"""Generate minimal animated glTF test files for unit tests."""

import struct
import json
import os
import math


def _pack_buffer(floats: list[float]) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


def _write_glb(path: str, gltf: dict, bin_data: bytes) -> None:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4-byte alignment
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # Pad binary to 4-byte alignment
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    with open(path, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN chunk
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)


def generate_linear_translation(output_dir: str) -> None:
    """A single triangle translating linearly along X from 0 to 5 over 2 seconds."""
    # Vertex data: 3 vertices of a triangle
    # fmt: off
    positions = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
    ]
    # fmt: on
    indices = [0, 1, 2]

    # Animation: translation from (0,0,0) to (5,0,0) over 2s with 3 keyframes
    times = [0.0, 1.0, 2.0]
    # fmt: off
    translations = [
        0.0, 0.0, 0.0,
        2.5, 0.0, 0.0,
        5.0, 0.0, 0.0,
    ]
    # fmt: on

    # Pack binary data
    pos_bytes = _pack_buffer(positions)
    idx_bytes = struct.pack("<3H", *indices)
    # Pad indices to 4 bytes
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    idx_bytes += b"\x00" * idx_pad
    time_bytes = _pack_buffer(times)
    trans_bytes = _pack_buffer(translations)

    bin_data = pos_bytes + idx_bytes + time_bytes + trans_bytes

    pos_offset = 0
    idx_offset = len(pos_bytes)
    time_offset = idx_offset + len(idx_bytes)
    trans_offset = time_offset + len(time_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "falcor2-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "animated_node"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {  # 0: positions
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {  # 1: indices
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
            {  # 2: animation times
                "bufferView": 2,
                "componentType": 5126,
                "count": 3,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            {  # 3: animation translations
                "bufferView": 3,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_bytes) - idx_pad},
            {"buffer": 0, "byteOffset": time_offset, "byteLength": len(time_bytes)},
            {"buffer": 0, "byteOffset": trans_offset, "byteLength": len(trans_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "TranslateX",
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 0, "path": "translation"},
                    }
                ],
                "samplers": [
                    {
                        "input": 2,
                        "output": 3,
                        "interpolation": "LINEAR",
                    }
                ],
            }
        ],
    }

    _write_glb(os.path.join(output_dir, "anim_linear_translation.glb"), gltf, bin_data)


def generate_step_rotation(output_dir: str) -> None:
    """A single triangle with STEP rotation animation (2 keyframes)."""
    # fmt: off
    positions = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
    ]
    # fmt: on
    indices = [0, 1, 2]

    # STEP rotation: identity at t=0, 90-degree Y rotation at t=1
    times = [0.0, 1.0]
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    # fmt: off
    rotations = [
        0.0, 0.0, 0.0, 1.0,  # identity quaternion (x,y,z,w)
        0.0, s, 0.0, c,  # 90 degrees around Y
    ]
    # fmt: on

    pos_bytes = _pack_buffer(positions)
    idx_bytes = struct.pack("<3H", *indices)
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    idx_bytes += b"\x00" * idx_pad
    time_bytes = _pack_buffer(times)
    rot_bytes = _pack_buffer(rotations)

    bin_data = pos_bytes + idx_bytes + time_bytes + rot_bytes

    pos_offset = 0
    idx_offset = len(pos_bytes)
    time_offset = idx_offset + len(idx_bytes)
    rot_offset = time_offset + len(time_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "falcor2-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "step_rotation_node"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [1.0],
            },
            {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_bytes) - idx_pad},
            {"buffer": 0, "byteOffset": time_offset, "byteLength": len(time_bytes)},
            {"buffer": 0, "byteOffset": rot_offset, "byteLength": len(rot_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "StepRotation",
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 0, "path": "rotation"},
                    }
                ],
                "samplers": [
                    {
                        "input": 2,
                        "output": 3,
                        "interpolation": "STEP",
                    }
                ],
            }
        ],
    }

    _write_glb(os.path.join(output_dir, "anim_step_rotation.glb"), gltf, bin_data)


def generate_multi_channel(output_dir: str) -> None:
    """A single triangle with translation, rotation, and scale animated on the same node."""
    # fmt: off
    positions = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
    ]
    # fmt: on
    indices = [0, 1, 2]

    # Translation: (0,0,0) -> (1,2,3) over 2s
    trans_times = [0.0, 2.0]
    translations = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]

    # Rotation: identity -> 90 around Y over 1s
    rot_times = [0.0, 1.0]
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    rotations = [0.0, 0.0, 0.0, 1.0, 0.0, s, 0.0, c]

    # Scale: (1,1,1) -> (2,2,2) over 3s with 3 keyframes
    scale_times = [0.0, 1.5, 3.0]
    scales = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0]

    pos_bytes = _pack_buffer(positions)
    idx_bytes = struct.pack("<3H", *indices)
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    idx_bytes += b"\x00" * idx_pad

    trans_time_bytes = _pack_buffer(trans_times)
    trans_bytes = _pack_buffer(translations)
    rot_time_bytes = _pack_buffer(rot_times)
    rot_bytes = _pack_buffer(rotations)
    scale_time_bytes = _pack_buffer(scale_times)
    scale_bytes = _pack_buffer(scales)

    bin_data = (
        pos_bytes
        + idx_bytes
        + trans_time_bytes
        + trans_bytes
        + rot_time_bytes
        + rot_bytes
        + scale_time_bytes
        + scale_bytes
    )

    offsets = []
    off = 0
    for b in [
        pos_bytes,
        idx_bytes,
        trans_time_bytes,
        trans_bytes,
        rot_time_bytes,
        rot_bytes,
        scale_time_bytes,
        scale_bytes,
    ]:
        offsets.append(off)
        off += len(b)

    gltf = {
        "asset": {"version": "2.0", "generator": "falcor2-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "multi_channel_node"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            # 0: positions
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            # 1: indices
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            # 2: translation times
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            # 3: translation values
            {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"},
            # 4: rotation times
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [1.0],
            },
            # 5: rotation values
            {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC4"},
            # 6: scale times
            {
                "bufferView": 6,
                "componentType": 5126,
                "count": 3,
                "type": "SCALAR",
                "min": [0.0],
                "max": [3.0],
            },
            # 7: scale values
            {"bufferView": 7, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(idx_bytes) - idx_pad},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(trans_time_bytes)},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(trans_bytes)},
            {"buffer": 0, "byteOffset": offsets[4], "byteLength": len(rot_time_bytes)},
            {"buffer": 0, "byteOffset": offsets[5], "byteLength": len(rot_bytes)},
            {"buffer": 0, "byteOffset": offsets[6], "byteLength": len(scale_time_bytes)},
            {"buffer": 0, "byteOffset": offsets[7], "byteLength": len(scale_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "MultiChannel",
                "channels": [
                    {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
                    {"sampler": 2, "target": {"node": 0, "path": "scale"}},
                ],
                "samplers": [
                    {"input": 2, "output": 3, "interpolation": "LINEAR"},
                    {"input": 4, "output": 5, "interpolation": "LINEAR"},
                    {"input": 6, "output": 7, "interpolation": "LINEAR"},
                ],
            }
        ],
    }

    _write_glb(os.path.join(output_dir, "anim_multi_channel.glb"), gltf, bin_data)


def generate_cubicspline(output_dir: str) -> None:
    """A single triangle with CUBICSPLINE translation animation."""
    # fmt: off
    positions = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
    ]
    # fmt: on
    indices = [0, 1, 2]

    times = [0.0, 1.0, 2.0]
    # CUBICSPLINE: each keyframe has 3 VEC3 values: [in_tangent, value, out_tangent]
    # Total: 3 keyframes * 3 triplets * 3 components = 27 floats
    # fmt: off
    cubic_data = [
        # Keyframe 0: in_tangent, value, out_tangent
        0.0, 0.0, 0.0,  # in-tangent (unused for first keyframe typically)
        0.0, 0.0, 0.0,  # value
        1.0, 0.0, 0.0,  # out-tangent
        # Keyframe 1: in_tangent, value, out_tangent
        0.0, 0.0, 0.0,  # in-tangent
        3.0, 2.0, 0.0,  # value
        1.0, 0.0, 0.0,  # out-tangent
        # Keyframe 2: in_tangent, value, out_tangent
        0.0, 0.0, 0.0,  # in-tangent
        5.0, 0.0, 0.0,  # value
        0.0, 0.0, 0.0,  # out-tangent (unused for last keyframe typically)
    ]
    # fmt: on

    pos_bytes = _pack_buffer(positions)
    idx_bytes = struct.pack("<3H", *indices)
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    idx_bytes += b"\x00" * idx_pad
    time_bytes = _pack_buffer(times)
    cubic_bytes = _pack_buffer(cubic_data)

    bin_data = pos_bytes + idx_bytes + time_bytes + cubic_bytes

    pos_offset = 0
    idx_offset = len(pos_bytes)
    time_offset = idx_offset + len(idx_bytes)
    cubic_offset = time_offset + len(time_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "falcor2-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "cubicspline_node"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 3,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            # 9 VEC3 values (3 keyframes * 3 triplet elements)
            {"bufferView": 3, "componentType": 5126, "count": 9, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_bytes) - idx_pad},
            {"buffer": 0, "byteOffset": time_offset, "byteLength": len(time_bytes)},
            {"buffer": 0, "byteOffset": cubic_offset, "byteLength": len(cubic_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "CubicSplineTranslation",
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 0, "path": "translation"},
                    }
                ],
                "samplers": [
                    {
                        "input": 2,
                        "output": 3,
                        "interpolation": "CUBICSPLINE",
                    }
                ],
            }
        ],
    }

    _write_glb(os.path.join(output_dir, "anim_cubicspline.glb"), gltf, bin_data)


def generate_multi_node(output_dir: str) -> None:
    """Two nodes: parent with translation, child with rotation. Tests hierarchy + multiple animated nodes."""
    # fmt: off
    positions = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
    ]
    # fmt: on
    indices = [0, 1, 2]

    # Parent translation: (0,0,0) -> (10,0,0)
    parent_times = [0.0, 1.0]
    parent_trans = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]

    # Child rotation: identity -> 90 degrees around Z
    child_times = [0.0, 1.0]
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    child_rot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, s, c]

    pos_bytes = _pack_buffer(positions)
    idx_bytes = struct.pack("<3H", *indices)
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    idx_bytes += b"\x00" * idx_pad
    pt_bytes = _pack_buffer(parent_times)
    pv_bytes = _pack_buffer(parent_trans)
    ct_bytes = _pack_buffer(child_times)
    cv_bytes = _pack_buffer(child_rot)

    bin_data = pos_bytes + idx_bytes + pt_bytes + pv_bytes + ct_bytes + cv_bytes

    offsets = []
    off = 0
    for b in [pos_bytes, idx_bytes, pt_bytes, pv_bytes, ct_bytes, cv_bytes]:
        offsets.append(off)
        off += len(b)

    gltf = {
        "asset": {"version": "2.0", "generator": "falcor2-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "parent_node", "children": [1]},
            {"mesh": 0, "name": "child_node"},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [1.0],
            },
            {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"},
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [1.0],
            },
            {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(idx_bytes) - idx_pad},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(pt_bytes)},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(pv_bytes)},
            {"buffer": 0, "byteOffset": offsets[4], "byteLength": len(ct_bytes)},
            {"buffer": 0, "byteOffset": offsets[5], "byteLength": len(cv_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "MultiNode",
                "channels": [
                    {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                ],
                "samplers": [
                    {"input": 2, "output": 3, "interpolation": "LINEAR"},
                    {"input": 4, "output": 5, "interpolation": "LINEAR"},
                ],
            }
        ],
    }

    _write_glb(os.path.join(output_dir, "anim_multi_node.glb"), gltf, bin_data)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "assets", "animation_test")
    os.makedirs(output_dir, exist_ok=True)
    generate_linear_translation(output_dir)
    generate_step_rotation(output_dir)
    generate_multi_channel(output_dir)
    generate_cubicspline(output_dir)
    generate_multi_node(output_dir)
    print(f"Generated test files in {output_dir}")
