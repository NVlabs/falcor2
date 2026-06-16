// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/macros.h"

#include <functional>

#define FALCOR_VERSION_MAJOR 0
#define FALCOR_VERSION_MINOR 1
#define FALCOR_VERSION_PATCH 0

#define FALCOR_VERSION                                                                                                 \
    SGL_TO_STRING(FALCOR_VERSION_MAJOR)                                                                                \
    "." SGL_TO_STRING(FALCOR_VERSION_MINOR) "." SGL_TO_STRING(FALCOR_VERSION_PATCH)

extern FALCOR_API const char* FALCOR_GIT_VERSION;

namespace falcor {

using ShutdownCallbackFunc = std::function<void(void)>;

FALCOR_API void static_init();
FALCOR_API void static_shutdown();
FALCOR_API void register_shutdown_callback(ShutdownCallbackFunc callback);

} // namespace falcor
