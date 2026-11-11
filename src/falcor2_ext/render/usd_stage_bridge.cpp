// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/usd_importer/usd_importer_macros.h"

#include "usd_stage_bridge.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/stage.h>
#include <pxr/external/boost/python/extract.hpp>
END_DISABLE_USD_WARNINGS

namespace falcor::usd_stage_bridge {

const pxr::UsdStage* try_extract(PyObject* object)
{
    PXR_BOOST_PYTHON_NAMESPACE::extract<pxr::UsdStage*> extract_stage(object);
    return extract_stage.check() ? extract_stage() : nullptr;
}

} // namespace falcor::usd_stage_bridge
