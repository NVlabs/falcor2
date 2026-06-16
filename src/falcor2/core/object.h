// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/fwd.h"

#include <sgl/core/object.h>

namespace falcor {

using Object = sgl::Object;

#define FALCOR_OBJECT(name) SGL_OBJECT(name)

template<typename T>
using ref = sgl::ref<T>;
using sgl::make_ref;
using sgl::static_ref_cast;
using sgl::dynamic_ref_cast;

template<typename To, typename From>
To checked_cast(From from)
{
    FALCOR_CHECK(!from || dynamic_cast<To>(from), "Invalid cast");
    return static_cast<To>(from);
}

} // namespace falcor
