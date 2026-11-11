// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/usd_importer/usd_importer.h"

// On Windows, Python.h in debug mode tries to link against python3XX_d.lib
// which is not available in standard Python distributions.
#if defined(_DEBUG) && defined(_MSC_VER)
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

namespace falcor::usd_stage_bridge {

/// Try to extract a Falcor-owned USD stage from a Python object.
/// Returns nullptr when the object does not wrap a compatible stage.
const pxr::UsdStage* try_extract(PyObject* object);

} // namespace falcor::usd_stage_bridge
