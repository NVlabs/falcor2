// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sgl/core/macros.h"

SGL_DIAGNOSTIC_PUSH
SGL_DISABLE_MSVC_WARNING(4100)
SGL_DISABLE_CLANG_WARNING("-Wunused-parameter")
SGL_DISABLE_CLANG_WARNING("-Wdeprecated-literal-operator")
SGL_DISABLE_GCC_WARNING("-Wunused-parameter")
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tinygltf/tiny_gltf.h"
SGL_DIAGNOSTIC_POP
