// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <sgl/core/macros.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <typeinfo>

namespace falcor {
/// Shared library handle.
using SharedLibraryHandle = void*;
} // namespace falcor

namespace falcor::platform {

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

/// The project source directory. Note that this is only valid during development.
[[nodiscard]] FALCOR_API const std::filesystem::path& project_directory();

/// The runtime directory. This is the path where the falcor2 runtime library
/// (falcor2.dll or libfalcor2.so) resides.
[[nodiscard]] FALCOR_API const std::filesystem::path& runtime_directory();

/// The application data directory.
/// Windows: %LOCALAPPDATA%/falcor2
/// Linux/macOS: ~/.falcor2
/// Creates the directory if it doesn't exist.
[[nodiscard]] FALCOR_API const std::filesystem::path& app_data_directory();

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

/// Load a shared library.
[[nodiscard]] FALCOR_API SharedLibraryHandle load_shared_library(const std::filesystem::path& path);

/// Release a shared library.
FALCOR_API void release_shared_library(SharedLibraryHandle library);

/// Get a function pointer from a library.
FALCOR_API void* get_proc_address(SharedLibraryHandle library, const char* proc_name);

// -------------------------------------------------------------------------------------------------
// File locking
// -------------------------------------------------------------------------------------------------

/// RAII file lock for cross-process synchronization.
/// Uses LockFileEx on Windows and flock on Linux/macOS.
class FALCOR_API FileLock {
public:
    explicit FileLock(const std::filesystem::path& lock_file);
    ~FileLock();

    /// Try to acquire the lock within the given timeout.
    /// Returns true if the lock was acquired, false on timeout.
    bool try_lock(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    /// Release the lock. Safe to call even if not locked.
    void unlock();

    /// Returns true if the lock is currently held.
    bool is_locked() const { return m_locked; }

private:
    uintptr_t m_handle{0};
    bool m_locked{false};

    FALCOR_NON_COPYABLE(FileLock);
    FALCOR_NON_MOVABLE(FileLock);
};

// -------------------------------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------------------------------

/// Return a readable string representation of a C++ type.
FALCOR_API std::string demangle_type_name(const std::type_info& t);

} // namespace falcor::platform
