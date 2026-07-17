# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike

import falcor2 as f2
import slangpy as spy


_AssetPath = str | PathLike[str]


class _CameraCollection:
    def create(
        self,
        name: str,
        focus_distance: float = 1.0,
        focal_length: float = 50.0,
        fstop: float = 8.0,
        depth_range: spy.float2 = spy.float2(0.01, 10000.0),
        projection: f2.ImporterCamera.Projection = f2.ImporterCamera.Projection.perspective,
        fov_direction: f2.ImporterCamera.FOVDirection = f2.ImporterCamera.FOVDirection.vertical,
        sensor_size_mm: float = 24.0,
    ) -> f2.ImporterCameraSelector:
        return f2.Importer.get().cameras.create(
            name=name,
            focus_distance=focus_distance,
            focal_length=focal_length,
            fstop=fstop,
            depth_range=depth_range,
            projection=projection,
            fov_direction=fov_direction,
            sensor_size_mm=sensor_size_mm,
        )

    def create_fov(
        self,
        name: str,
        fov_degrees: float = 70.0,
        focus_distance: float = 1.0,
        fstop: float = 8.0,
        depth_range: spy.float2 = spy.float2(0.01, 10000.0),
        projection: f2.ImporterCamera.Projection = f2.ImporterCamera.Projection.perspective,
        fov_direction: f2.ImporterCamera.FOVDirection = f2.ImporterCamera.FOVDirection.vertical,
        sensor_size_mm: float = 24.0,
    ) -> f2.ImporterCameraSelector:
        return f2.Importer.get().cameras.create_fov(
            name=name,
            fov_degrees=fov_degrees,
            focus_distance=focus_distance,
            fstop=fstop,
            depth_range=depth_range,
            projection=projection,
            fov_direction=fov_direction,
            sensor_size_mm=sensor_size_mm,
        )


class _NodeCollection:
    def create(
        self,
        name: str,
        transform: spy.float4x4 = spy.float4x4.identity(),
        camera: f2.ImporterCameraSelector | None = None,
        parent: f2.ImporterNodeSelector | None = None,
    ) -> f2.ImporterNodeSelector:
        return f2.Importer.get().nodes.create(
            name=name,
            transform=transform,
            camera=camera,
            parent=parent,
        )

    def create_camera(
        self,
        name: str = "Camera",
        transform: spy.float4x4 = spy.float4x4.identity(),
        focus_distance: float = 1.0,
        focal_length: float = 50.0,
        fstop: float = 8.0,
        depth_range: spy.float2 = spy.float2(0.01, 10000.0),
        projection: f2.ImporterCamera.Projection = f2.ImporterCamera.Projection.perspective,
        fov_direction: f2.ImporterCamera.FOVDirection = f2.ImporterCamera.FOVDirection.vertical,
        sensor_size_mm: float = 24.0,
    ) -> f2.ImporterNodeSelector:
        return f2.Importer.get().nodes.create_camera(
            name=name,
            transform=transform,
            focus_distance=focus_distance,
            focal_length=focal_length,
            fstop=fstop,
            depth_range=depth_range,
            projection=projection,
            fov_direction=fov_direction,
            sensor_size_mm=sensor_size_mm,
        )

    def create_camera_fov(
        self,
        name: str = "Camera",
        transform: spy.float4x4 = spy.float4x4.identity(),
        fov_degrees: float = 70.0,
        focus_distance: float = 1.0,
        fstop: float = 8.0,
        depth_range: spy.float2 = spy.float2(0.01, 10000.0),
        projection: f2.ImporterCamera.Projection = f2.ImporterCamera.Projection.perspective,
        fov_direction: f2.ImporterCamera.FOVDirection = f2.ImporterCamera.FOVDirection.vertical,
        sensor_size_mm: float = 24.0,
    ) -> f2.ImporterNodeSelector:
        return f2.Importer.get().nodes.create_camera_fov(
            name=name,
            transform=transform,
            fov_degrees=fov_degrees,
            focus_distance=focus_distance,
            fstop=fstop,
            depth_range=depth_range,
            projection=projection,
            fov_direction=fov_direction,
            sensor_size_mm=sensor_size_mm,
        )


class _MaterialCollection:
    def find(self, name: str) -> f2.ImporterMaterialSelector:
        return f2.Importer.get().materials.find(name)

    def __getitem__(self, name: str) -> f2.ImporterMaterialSelector:
        return f2.Importer.get().materials[name]


class _EnvironmentCollection:
    def set(
        self,
        path: _AssetPath,
        exposure: float = 0.0,
        name: str = "Environment Map",
    ) -> None:
        f2.Importer.get().env.set(path=path, exposure=exposure, name=name)


cameras = _CameraCollection()
nodes = _NodeCollection()
materials = _MaterialCollection()
env = _EnvironmentCollection()


def load_asset(path: _AssetPath) -> None:
    f2.Importer.get().import_asset(path)


def add_search_path(path: _AssetPath) -> None:
    f2.Importer.get().add_search_path(path)


def add_search_paths(paths: Sequence[_AssetPath]) -> None:
    f2.Importer.get().add_search_paths(paths)


def on_scene_created(callback: Callable[[f2.Scene], None]) -> None:
    f2.Importer.get().on_scene_created(callback)
