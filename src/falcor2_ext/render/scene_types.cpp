// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/render/scene_types.h"

FALCOR_PY_EXPORT(render_scene_types)
{
    using namespace falcor;

    nb::sgl_enum<UVOrigin>(m, "UVOrigin", D(UVOrigin));
    nb::sgl_enum<CurveIndexingMode>(m, "CurveIndexingMode", D(CurveIndexingMode));
    nb::sgl_enum<AnimationValueType>(m, "AnimationValueType", D(AnimationValueType));
    nb::sgl_enum<AnimationInterpolationMode>(m, "AnimationInterpolationMode", D(AnimationInterpolationMode));
    nb::sgl_enum<AlphaMode>(m, "AlphaMode", D(AlphaMode));
    nb::sgl_enum<EnvMapPoleAxis>(m, "EnvMapPoleAxis", "Environment map pole axis in the owning entity's local space.");
}
