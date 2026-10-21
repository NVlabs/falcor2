// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace falcor {

// vertex_remap.h
struct VertexRemap;

// importer_types.h

struct ImporterMeshAttribute;
struct ImporterMeshDefaultAttribute;
struct ImporterVertexBuffer;
class ImporterMesh;
class ImporterCurve;
struct ImporterTexture;
struct ImporterNode;
struct ImporterMaterial;
struct ImporterCamera;
struct ImporterLight;
struct ImporterAnimationChannel;
struct ImporterAnimation;
struct ImporterSkin;
class ImporterScene;

// importer.h
struct ImportOptions;
struct ImporterSelector;
struct ImporterCameraSelector;
struct ImporterNodeSelector;
class ImporterMaterialSelector;
class ImporterBuildContext;
class ImporterEdit;
class ImportAssetEdit;
class ReplaceMaterialEdit;
class SetEnvironmentEdit;
class ImporterCameraCollection;
class ImporterNodeCollection;
class ImporterEnvCollection;
class ImporterMaterialCollection;
class Importer;
class ScopedCurrentImporter;

// gltf_importer.h
class GltfImporter;

// usda_importer.h
class UsdImporter;

}; // namespace falcor
