// SPDX-License-Identifier: Apache-2.0

#include "falcor2.h"

#include "falcor2/core/format.h"
#include "falcor2/core/logger.h"

#include "git_version_falcor2.h"

#include <sgl/sgl.h>

#include <string>
#include <mutex>
#include <vector>

static inline const char* git_version()
{
    static std::string str{
        fmt::format("commit: {} / branch: {}", GIT_VERSION_COMMIT, GIT_VERSION_BRANCH)
        + (GIT_VERSION_DIRTY ? " (local changes)" : "")
    };
    return str.c_str();
}

const char* FALCOR_GIT_VERSION = git_version();

namespace falcor {

static std::mutex s_shutdown_callbacks_mutex;
static std::vector<ShutdownCallbackFunc> s_shutdown_callbacks;

void static_init()
{
    sgl::static_init();
    sgl::log_info("falcor2 init");
}

void static_shutdown()
{
    sgl::log_info("falcor2 shutdown");
    {
        std::lock_guard lock(s_shutdown_callbacks_mutex);
        for (auto& callback : s_shutdown_callbacks)
            callback();
        s_shutdown_callbacks.clear();
    }
    sgl::static_shutdown();
}

void register_shutdown_callback(ShutdownCallbackFunc callback)
{
    std::lock_guard lock(s_shutdown_callbacks_mutex);
    s_shutdown_callbacks.push_back(std::move(callback));
}

} // namespace falcor
