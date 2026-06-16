// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/denoisers/ngx_types.h"

#include <sgl/device/fwd.h>

namespace falcor::ngx {

class NGX;

class FALCOR_API DLSSSRFeature : public Object {
    FALCOR_OBJECT(DLSSSRFeature)

public:
    class Impl;

    /// Create a DLSS Super Resolution feature from an initialized NGX context.
    DLSSSRFeature(ref<NGX> ngx, const DLSSSRFeatureDesc& desc);

    /// Release the backend-specific feature implementation.
    ~DLSSSRFeature();

    /// Return the parent NGX context kept alive by this feature.
    NGX* ngx() const { return m_ngx; }

    /// Return the SGL device owned by the parent NGX context.
    sgl::Device* device() const;

    /// Return the feature creation descriptor.
    const DLSSSRFeatureDesc& desc() const { return m_desc; }

    /// Submit DLSS Super Resolution evaluation on a short-lived command buffer.
    void evaluate(const DLSSSREvaluateDesc& desc);

    /// Record a DLSS Super Resolution evaluation into an existing command encoder.
    void evaluate(sgl::CommandEncoder* command_encoder, const DLSSSREvaluateDesc& desc);

    /// Return and clear the NGX result from the most recently executed evaluation callback, or Result::none if none has
    /// executed.
    Result last_evaluate_result();

private:
    void throw_if_last_evaluate_failed() const;

    ref<NGX> m_ngx;
    DLSSSRFeatureDesc m_desc;
    Impl* m_impl = nullptr;
};

} // namespace falcor::ngx
