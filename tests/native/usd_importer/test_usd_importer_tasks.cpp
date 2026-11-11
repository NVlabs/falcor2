// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/usd_importer/usd_importer_tasks.h"

#include <atomic>
#include <stdexcept>
#include <string>

TEST_SUITE_BEGIN("usd_importer");

TEST_CASE("USD traversal drains workers and preserves the primary error")
{
    bool traversal_fails = false;
    bool worker_fails = false;
    SUBCASE("success") { }
    SUBCASE("traversal failure")
    {
        traversal_fails = true;
    }
    SUBCASE("worker failure")
    {
        worker_fails = true;
    }
    SUBCASE("both fail")
    {
        traversal_fails = true;
        worker_fails = true;
    }

    sgl::thread::TaskGroup tasks;
    std::atomic<bool> worker_finished{false};
    std::atomic<bool> other_worker_finished{false};
    std::string error;
    try {
        // Exercise the same lifetime boundary as UsdImporterContext::run, with
        // real task failures rather than exceptions thrown by logger outputs.
        falcor::usd_importer::run_traversal_and_wait(
            tasks,
            [&]
            {
                tasks.do_async(
                    [&]
                    {
                        worker_finished = true;
                        if (worker_fails)
                            throw std::runtime_error("worker failed");
                    }
                );
                tasks.do_async(
                    [&]
                    {
                        other_worker_finished = true;
                    }
                );
                if (traversal_fails)
                    throw std::runtime_error("traversal failed");
            }
        );
    } catch (const std::runtime_error& exception) {
        error = exception.what();
    }

    CHECK(error == (traversal_fails ? "traversal failed" : worker_fails ? "worker failed" : ""));
    CHECK(worker_finished);
    CHECK(other_worker_finished);
    // The group must already be drained, including its stored exceptions.
    // This also keeps a broken implementation from leaving tasks at destruction.
    CHECK_NOTHROW(tasks.wait());
}

TEST_SUITE_END();
