// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// Shared render link-time options. Subsystem-specific options stay with their implementations.

// Fixed scene configuration
// Supplied by scene settings. Consumers must not override these independently of the scene.

/// Apply texture coordinate corrections for a lower-left UV origin, as specified by SceneConfig.uv_origin.
public extern static const bool UV_ORIGIN_LOWER_LEFT = false;

// Scene capabilities
// Supplied by scene settings from resolved requirements. Consumers must use the scene capability policy.

/// Retain hit information for scene-backed geometry property reads.
public extern static const bool ENABLE_GEOMETRY_PROPERTIES = true;

/// Scene shader capability; output writers still select which offered AOVs are consumed.
public extern static const bool ENABLE_MATERIAL_AOVS = true;

/// Allow renderers to track interiors required by solid-surface transmission.
public extern static const bool ENABLE_SOLID_TRANSMISSION = true;

/// Scene settings select material opacity evaluation from the resolved opacity capability.
public extern struct OpacityEvaluator : IOpacityEvaluator = AcceptAllHitsEvaluator;

// Renderer overrides
// Renderers may specialize these options when composing shaders.

/// Renderers can specialize this constant to enable conservative triangle ray offsets.
public extern static const bool ENABLE_PRECISE_RAY_OFFSETS = false;

/// Caller opt-in for hardware triangle backface culling on eligible instances.
public extern static const bool ENABLE_TRIANGLE_BACKFACE_CULLING = false;
