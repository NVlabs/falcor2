// SPDX-License-Identifier: Apache-2.0

#include "platform.h"
#include "error.h"

#if SGL_LINUX

#include <dlfcn.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <thread>

namespace falcor::platform {

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path(
        []()
        {
            Dl_info info;
            if (dladdr((void*)&runtime_directory, &info) == 0)
                FALCOR_THROW("Failed to get the runtime directory. dladdr() failed.");
            return std::filesystem::path(info.dli_fname).parent_path();
        }()
    );
    return path;
}

const std::filesystem::path& app_data_directory()
{
    static std::filesystem::path path(
        []()
        {
            const char* home = std::getenv("HOME");
            if (!home) {
                struct passwd* pw = getpwuid(getuid());
                if (pw)
                    home = pw->pw_dir;
            }
            if (!home)
                FALCOR_THROW("Failed to determine home directory.");
            std::filesystem::path dir = std::filesystem::path(home) / ".falcor2";
            std::filesystem::create_directories(dir);
            return dir;
        }()
    );
    return path;
}

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

SharedLibraryHandle load_shared_library(const std::filesystem::path& path)
{
    return dlopen(path.c_str(), RTLD_LAZY);
}

void release_shared_library(SharedLibraryHandle library)
{
    dlclose(library);
}

void* get_proc_address(SharedLibraryHandle library, const char* proc_name)
{
    return dlsym(library, proc_name);
}

// -------------------------------------------------------------------------------------------------
// File locking
// -------------------------------------------------------------------------------------------------

FileLock::FileLock(const std::filesystem::path& lock_file)
{
    int fd = open(lock_file.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0)
        FALCOR_THROW("Failed to open lock file '{}'. errno: {}", lock_file.string(), errno);
    m_handle = static_cast<uintptr_t>(fd);
}

FileLock::~FileLock()
{
    if (m_locked)
        unlock();
    int fd = static_cast<int>(m_handle);
    if (fd >= 0)
        close(fd);
}

bool FileLock::try_lock(std::chrono::milliseconds timeout)
{
    if (m_locked)
        return true;

    int fd = static_cast<int>(m_handle);
    auto start = std::chrono::steady_clock::now();
    while (true) {
        int ret = flock(fd, LOCK_EX | LOCK_NB);
        if (ret == 0) {
            m_locked = true;
            return true;
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout)
            return false;

        // Brief sleep before retry
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void FileLock::unlock()
{
    if (!m_locked)
        return;

    int fd = static_cast<int>(m_handle);
    flock(fd, LOCK_UN);
    m_locked = false;
}

} // namespace falcor::platform

#endif // SGL_LINUX
