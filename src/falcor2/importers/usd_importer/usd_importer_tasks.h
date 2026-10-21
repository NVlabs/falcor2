// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sgl/core/thread.h"

#include <exception>

namespace falcor::usd_importer {

/// Run traversal and drain its workers before returning or propagating an error.
/// Workers capture the importer context, so they must finish before traversal's
/// caller can unwind that context. If both fail, preserve the traversal error;
/// TaskGroup::wait() drains all workers before reporting a worker error.
template<typename Traverse>
void run_traversal_and_wait(sgl::thread::TaskGroup& tasks, Traverse&& traverse)
{
    std::exception_ptr traversal_error;
    try {
        traverse();
    } catch (...) {
        traversal_error = std::current_exception();
    }
    try {
        tasks.wait();
    } catch (...) {
        if (!traversal_error)
            throw;
    }
    if (traversal_error)
        std::rethrow_exception(traversal_error);
}

} // namespace falcor::usd_importer
