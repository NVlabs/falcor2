// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace falcor {

// animation.h

class Animation;
class AnimationChannel;

// component.h

class Component;

// emissive_geometry_system.h

class EmissiveGeometrySystem;

// light_sampler.h

class LightSampler;
class UniformLightSampler;
class PowerLightSampler;
class HierarchicalLightSampler;

// emissive_triangle_tree.h

class EmissiveTriangleTree;

// entity.h

class Entity;

// geometry.h

class Geometry;

// light_system.h

class LightSystem;

// material_system.h

class MaterialSystem;

// material.h

class Material;

// scene_globals.h

class SceneGlobals;

// scene_object.h

class SceneObject;

template<typename TObject>
class SceneObjectCollection;

template<typename TObject>
class SceneObjectCollectionView;

// scene_system.h

class SceneSystem;

// scene_update.h

class SceneUpdateContext;

// scene.h

class Scene;
struct SceneConfig;

using MaterialCollection = SceneObjectCollection<Material>;
using MaterialCollectionView = SceneObjectCollectionView<Material>;
using GeometryCollection = SceneObjectCollection<Geometry>;
using GeometryCollectionView = SceneObjectCollectionView<Geometry>;
using AnimationCollection = SceneObjectCollection<Animation>;
using AnimationCollectionView = SceneObjectCollectionView<Animation>;
using EntityCollection = SceneObjectCollection<Entity>;
using EntityCollectionView = SceneObjectCollectionView<Entity>;
using ComponentCollection = SceneObjectCollection<Component>;
using ComponentCollectionView = SceneObjectCollectionView<Component>;

// texture_manager.h

class TextureManager;

// transform.h

class Transform;

// component/xxx

class Camera;
class GeometryInstance;
class Light;
class TransformAnimation;

// geometry/xxx

class GeometryGroup;
class StaticCurveGeometry;
class StaticMeshGeometry;

// material/xxx

class EmissionOnlyMaterial;
class StandardMaterial;
class StandardSpecGlossMaterial;
class UnlitMaterial;

} // namespace falcor
