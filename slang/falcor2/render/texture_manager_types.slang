// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
import falcor2.utils;
#endif

namespace detail {

struct TextureHandleFieldsNonCUDA
{
    typedef BitField<uint, 0, 20> texture;
    typedef BitField<uint, 20, 10> sampler;
    typedef BitField<uint, 31, 1> udim;
};

struct TextureHandleFieldsCUDA
{
    typedef BitField<uint, 0, 30> texture;
    typedef BitField<uint, 0, 0> sampler;
    typedef BitField<uint, 31, 1> udim;
};

#ifndef FALCOR_SHARED_HOST
#ifdef __TARGET_CUDA__
typedef TextureHandleFieldsCUDA TextureHandleFields;
#else
typedef TextureHandleFieldsNonCUDA TextureHandleFields;
#endif
#endif

static constexpr uint INVALID_TEXTURE_HANDLE = 0xffffffff;

} // namespace detail
