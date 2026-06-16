// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/fwd.h"

#include "falcor2/core/object.h"

#include <sgl/device/shader_cursor.h>

namespace falcor {

class FALCOR_API SceneGlobals : public Object {
    FALCOR_OBJECT(SceneGlobals)
public:
    virtual ~SceneGlobals() { }

    /// Bind this scene globals object to the shader.
    /// @param globals The shader cursor pointing to the global parameters.
    virtual void bind(sgl::ShaderCursor globals) const = 0;
};

} // namespace falcor
