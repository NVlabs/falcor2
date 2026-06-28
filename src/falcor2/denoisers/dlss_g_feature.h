// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/denoisers/ngx_types.h"

#include <sgl/device/fwd.h>

namespace falcor::ngx {

class NGX;

class FALCOR_API DLSSGFeature : public Object {
    FALCOR_OBJECT(DLSSGFeature)

public:
    class Impl;

    /// Create a DLSS Frame Generation feature from an initialized NGX context.
    DLSSGFeature(ref<NGX> ngx, const DLSSGFeatureDesc& desc);

    /// Release the backend-specific feature implementation.
    ~DLSSGFeature();

    /// Return the parent NGX context kept alive by this feature.
    NGX* ngx() const { return m_ngx; }

    /// Return the SGL device owned by the parent NGX context.
    sgl::Device* device() const;

    /// Return the feature creation descriptor.
    const DLSSGFeatureDesc& desc() const { return m_desc; }

    /// Submit DLSS Frame Generation evaluation on a short-lived command buffer.
    void evaluate(const DLSSGEvaluateDesc& desc);

    /// Record a DLSS Frame Generation evaluation into an existing command encoder.
    void evaluate(sgl::CommandEncoder* command_encoder, const DLSSGEvaluateDesc& desc);

    /// Return and clear the NGX result from the most recently executed evaluation callback, or Result::none if none has
    /// executed.
    Result last_evaluate_result();

private:
    void throw_if_last_evaluate_failed() const;

    ref<NGX> m_ngx;
    DLSSGFeatureDesc m_desc;
    Impl* m_impl = nullptr;
};

} // namespace falcor::ngx
