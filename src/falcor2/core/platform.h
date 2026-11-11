// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// GPU scheduling
// -------------------------------------------------------------------------------------------------

/// Temporarily request process-wide GPU REALTIME scheduling priority on Windows.
/// Applies to CUDA, Vulkan and D3D12 under WDDM; it does not guarantee uninterrupted execution.
/// Construction requests the priority immediately and throws if reading or setting it fails. Other platforms silently
/// leave scheduling unchanged. Keep the guard's lifetime short: realtime priority can make
/// the desktop unresponsive.
///
/// This guard is single-use. Nested guards must be closed in reverse order; do not overlap guards
/// across threads or change the process GPU priority while a guard is alive.
/// Restoration failures are logged without throwing.
/// In Python, use with falcor2.platform.set_gpu_realtime_priority() for deterministic restoration.
class FALCOR_API ScopedGPURealtimePriority {
public:
    ScopedGPURealtimePriority();
    /// Restore the previous priority. Restoration failures are logged without throwing.
    ~ScopedGPURealtimePriority() noexcept;

    /// Restore the previous priority, logging failures without throwing. Repeated calls after success do nothing.
    void close() noexcept;

private:
    int m_previous_priority{-1};

    FALCOR_NON_COPYABLE(ScopedGPURealtimePriority);
    FALCOR_NON_MOVABLE(ScopedGPURealtimePriority);
};

// -------------------------------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------------------------------

/// Return a readable string representation of a C++ type.
FALCOR_API std::string demangle_type_name(const std::type_info& t);

} // namespace falcor::platform
