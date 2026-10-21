# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own a CI command's descendants after the command itself has exited."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from collections.abc import Sequence
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _JobBasicLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_int64),
        ("kernel_time", ctypes.c_int64),
        ("period_user_time", ctypes.c_int64),
        ("period_kernel_time", ctypes.c_int64),
        ("page_faults", wintypes.DWORD),
        ("total_processes", wintypes.DWORD),
        ("active_processes", wintypes.DWORD),
        ("terminated_processes", wintypes.DWORD),
    ]


class _WindowsJob:
    def __init__(self) -> None:
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        functions = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (argtypes, restype) in functions.items():
            function = getattr(self._api, name)
            function.argtypes = argtypes
            function.restype = restype
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._api.SetInformationJobObject(
            self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self._api.CloseHandle(self._handle)
            raise error

    def assign(self, pid: int) -> None:
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE, required by AssignProcessToJobObject.
        handle = self._api.OpenProcess(0x0101, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self._api.AssignProcessToJobObject(self._handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._api.CloseHandle(handle)

    def _active_processes(self) -> int:
        accounting = _JobAccounting()
        if not self._api.QueryInformationJobObject(
            self._handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return accounting.active_processes

    def close(self, grace: float) -> None:
        try:
            deadline = time.monotonic() + grace
            while self._active_processes() and time.monotonic() < deadline:
                time.sleep(0.02)
            if self._active_processes():
                if not self._api.TerminateJobObject(self._handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                deadline = time.monotonic() + 5
                while self._active_processes():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Windows test job did not exit after termination")
                    time.sleep(0.02)
        finally:
            self._api.CloseHandle(self._handle)


class ProcessTree:
    """A retained process group or Job Object, independent of live-parent ancestry."""

    def __init__(self, command: str | Sequence[str], **kwargs: Any) -> None:
        self._job: _WindowsJob | None = None
        self._closed = False
        if os.name != "nt":
            self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
            self._pgid = self.process.pid
            return

        self._job = _WindowsJob()
        process: subprocess.Popen[str] | None = None
        try:
            payload = json.dumps({"command": command, "shell": kwargs.pop("shell", False)})
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), payload],
                shell=False,
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                **kwargs,
            )
            self.process = process
            # The launcher cannot create descendants until assignment succeeds.
            self._job.assign(process.pid)
            assert process.stdin is not None
            process.stdin.write("\n")
            process.stdin.close()
        except BaseException:
            try:
                if process is not None:
                    process.kill()
                    process.wait(timeout=5)
                    if process.stdin is not None:
                        process.stdin.close()
                    if process.stdout is not None:
                        process.stdout.close()
            finally:
                self._job.close(0)
            raise

    def close(self, grace: float = 0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._job is not None:
                self._job.close(grace)
            else:
                deadline = time.monotonic() + grace
                while True:
                    try:
                        os.killpg(self._pgid, 0)
                    except ProcessLookupError:
                        break
                    if time.monotonic() >= deadline:
                        try:
                            os.killpg(self._pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        break
                    time.sleep(0.02)
        finally:
            self.process.wait(timeout=5)


def _launch() -> int:
    if sys.stdin.readline() != "\n":
        return 1
    payload = json.loads(sys.argv[1])
    try:
        process = subprocess.Popen(
            payload["command"], shell=payload["shell"], stdin=subprocess.DEVNULL
        )
    except OSError as exc:
        print(f"Cannot launch test command: {exc}", file=sys.stderr)
        return 127
    return process.wait()


if __name__ == "__main__":
    sys.exit(_launch())
