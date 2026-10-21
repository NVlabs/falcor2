# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU evaluation for importer verification; this is not a scene playback system."""

import numpy as np
import falcor2 as f2


def sample_track(
    times: list[float] | np.ndarray,
    values: np.ndarray,
    mode: f2.AnimationInterpolationMode,
    time: float,
) -> np.ndarray:
    if not np.isfinite(time):
        raise ValueError("Sample time must be finite")
    times = np.asarray(times)
    right = int(np.searchsorted(times, time, side="right"))
    if right == 0:
        return values[0].copy()
    left = right - 1
    if right == len(times) or mode == f2.AnimationInterpolationMode.step:
        return values[left].copy()
    dt = times[right] - times[left]
    u = np.float32((time - times[left]) / dt)
    if mode == f2.AnimationInterpolationMode.linear:
        return (1 - u) * values[left] + u * values[right]
    raise ValueError("Unsupported vertex-preview interpolation")


def _normalize(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(lengths > 1e-8, values / np.maximum(lengths, 1e-8), fallback)


class MeshSampler:
    """Own snapshots of a mesh's imported samples, leaving the original data unchanged."""

    def __init__(self, mesh: f2.ImporterMesh) -> None:
        self.base = {}
        self.samples = {}
        self.times = mesh.motion_times
        self.mode = mesh.motion_interpolation_mode
        self.indices = [part.indices_numpy.copy() for part in mesh.subgeometries]
        for name, semantic in (
            ("positions", f2.ImporterSemantic.position),
            ("normals", f2.ImporterSemantic.normal),
            ("tangents", f2.ImporterSemantic.tangent),
            ("handedness", f2.ImporterSemantic.handedness),
        ):
            stream = mesh.find_stream(semantic)
            if stream is not None:
                self.base[name] = np.array(stream, copy=True)
                if name == "handedness":
                    self.base[name] = self.base[name].ravel()
            if self.times and stream is not None and semantic != f2.ImporterSemantic.handedness:
                values = [
                    np.array(mesh.find_stream(semantic, i), copy=True)
                    for i in range(len(self.times))
                ]
                self.samples[name] = np.asarray(values)

    def sample(self, time: float) -> dict[str, np.ndarray]:
        result = {name: values.copy() for name, values in self.base.items()}
        for name, values in self.samples.items():
            result[name] = sample_track(self.times, values, self.mode, time)
        if "normals" in result:
            n = _normalize(result["normals"], self.base["normals"])
            result["normals"] = n
            if "tangents" in result:
                t = result["tangents"] - n * np.sum(n * result["tangents"], axis=-1, keepdims=True)
                axis = np.where((np.abs(n[:, 2]) < 0.9)[:, None], [0, 0, 1], [0, 1, 0]).astype(
                    np.float32
                )
                fallback = _normalize(np.cross(axis, n), np.array([1, 0, 0], dtype=np.float32))
                result["tangents"] = _normalize(t, fallback)
        return result
