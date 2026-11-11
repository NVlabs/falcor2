// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/work/dispatcher.h>
END_DISABLE_USD_WARNINGS

#include <exception>
#include <mutex>
#include <utility>

namespace falcor::usd_importer {

/// Importer tasks use USD's diagnostic transport, but retain C++ exceptions for the caller.
/// Submit and wait from the traversal thread; wait before destroying any captured context.
class USDTaskGroup {
public:
    template<typename Task>
    void do_async(Task&& task)
    {
        m_dispatcher.Run(
            [this, task = std::forward<Task>(task)]
            {
                run_task(task);
            }
        );
    }

    void wait()
    {
        m_dispatcher.Wait();
        if (auto exception = std::exchange(m_exception, {}))
            std::rethrow_exception(exception);
    }

private:
    template<typename Task>
    void run_task(const Task& task)
    {
        try {
            task();
        } catch (...) {
            // Let USD finish transporting diagnostics and reset its dispatcher normally.
            // Escaping through Wait() would leave an exception for its noexcept destructor.
            {
                std::lock_guard lock(m_mutex);
                if (!m_exception)
                    m_exception = std::current_exception();
            }
            m_dispatcher.Cancel();
        }
    }

    std::mutex m_mutex;
    std::exception_ptr m_exception;
    // Destroy first so its final wait can still access the exception state.
    pxr::WorkDispatcher m_dispatcher;
};

/// Run traversal and drain its workers before returning or propagating an error.
/// Workers capture the importer context, so they must finish before traversal's
/// caller can unwind that context. If both fail, preserve the traversal error;
/// USDTaskGroup::wait() finishes running workers and cancels pending work on a worker error.
template<typename Traverse>
void run_traversal_and_wait(USDTaskGroup& tasks, Traverse&& traverse)
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
