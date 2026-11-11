// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace falcor::mtlx::structured {

class ILayeringPolicy;

/// Internal backend factories used by the common LayeringMode selector.
std::unique_ptr<ILayeringPolicy> create_closure_tree_policy();
std::unique_ptr<ILayeringPolicy> create_bsdf_mix_policy();
std::unique_ptr<ILayeringPolicy> create_hair_policy();

} // namespace falcor::mtlx::structured
