// SPDX-License-Identifier: Apache-2.0

#include "platform.h"
#include "error.h"

#if SGL_WINDOWS

#ifndef WINDOWS_LEAN_AND_MEAN
#define WINDOWS_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <ShlObj.h>

namespace falcor::platform {

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path(
        []()
        {
            HMODULE hm = NULL;
            if (GetModuleHandleExA(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    (LPCSTR)&runtime_directory,
                    &hm
                )
                == 0)
                FALCOR_THROW("Failed to get the runtime directory. GetModuleHandle failed: {}", GetLastError());
            CHAR path_str[1024];
            if (GetModuleFileNameA(hm, path_str, sizeof(path_str)) == 0)
                FALCOR_THROW("Failed to get the runtime directory. GetModuleFileNameA failed: {}", GetLastError());
            return std::filesystem::path(path_str).parent_path();
        }()
    );
    return path;
}

const std::filesystem::path& app_data_directory()
{
    static std::filesystem::path path(
        []()
        {
            PWSTR local_app_data = nullptr;
            HRESULT hr = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &local_app_data);
            if (FAILED(hr))
                FALCOR_THROW("Failed to get local app data directory. SHGetKnownFolderPath failed: {}", hr);
            std::wstring local_app_data_str(local_app_data);
            CoTaskMemFree(local_app_data);
            std::filesystem::path dir = std::filesystem::path(local_app_data_str) / "falcor2";
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
    /// Comments taken from slang-platform
    // We try to search the DLL in two different attempts.
    // First attempt - LoadLibraryExW()
    // If it failed to find one, we will use LoadLibraryW() to search over all PATH.
    // Search order: 1) The directory that contains the DLL (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR).
    //                  This directory is searched only for dependencies of the DLL being loaded.
    //               2) Application directory
    //               3) User directories (AddDllDirectory/SetDllDirectory)
    //               4) System32
    //               5) PATH environment variable (by the 2nd attempt with LoadLibraryW())
    // https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexw
    // https://docs.microsoft.com/en-us/windows/desktop/api/libloaderapi/nf-libloaderapi-loadlibraryw
    SharedLibraryHandle handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!handle)
        handle = LoadLibraryW(path.c_str());

    return handle;
}

void release_shared_library(SharedLibraryHandle library)
{
    FreeLibrary(static_cast<HMODULE>(library));
}

void* get_proc_address(SharedLibraryHandle library, const char* proc_name)
{
    return GetProcAddress(static_cast<HMODULE>(library), proc_name);
}

// -------------------------------------------------------------------------------------------------
// File locking
// -------------------------------------------------------------------------------------------------

FileLock::FileLock(const std::filesystem::path& lock_file)
{
    HANDLE h = CreateFileW(
        lock_file.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (h == INVALID_HANDLE_VALUE)
        FALCOR_THROW("Failed to open lock file '{}'. Error: {}", lock_file.string(), GetLastError());
    m_handle = reinterpret_cast<uintptr_t>(h);
}

FileLock::~FileLock()
{
    if (m_locked)
        unlock();
    HANDLE h = reinterpret_cast<HANDLE>(m_handle);
    if (h != INVALID_HANDLE_VALUE)
        CloseHandle(h);
}

bool FileLock::try_lock(std::chrono::milliseconds timeout)
{
    if (m_locked)
        return true;

    HANDLE h = reinterpret_cast<HANDLE>(m_handle);
    OVERLAPPED overlapped = {};
    DWORD flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        if (LockFileEx(h, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
            m_locked = true;
            return true;
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout)
            return false;

        // Brief sleep before retry
        Sleep(1);
    }
}

void FileLock::unlock()
{
    if (!m_locked)
        return;

    HANDLE h = reinterpret_cast<HANDLE>(m_handle);
    OVERLAPPED overlapped = {};
    UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &overlapped);
    m_locked = false;
}

} // namespace falcor::platform

#endif // SGL_WINDOWS
