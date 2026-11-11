# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only process sampling for the local soak harness (requires psutil)."""

import argparse
import json
import subprocess
import sys
from typing import TypedDict

import psutil


class ProcessSample(TypedDict):
    rss_bytes: int
    threads: int
    os_resources: int
    os_resource_kind: str
    cpu_seconds: float
    gpu_memory_mib: int | None


def sample(pid: int) -> ProcessSample:
    process = psutil.Process(pid)
    with process.oneshot():
        result: ProcessSample = {
            "rss_bytes": process.memory_info().rss,
            "threads": process.num_threads(),
            "os_resources": process.num_handles() if sys.platform == "win32" else process.num_fds(),
            "os_resource_kind": "handles" if sys.platform == "win32" else "file_descriptors",
            "cpu_seconds": sum(process.cpu_times()[:2]),
            "gpu_memory_mib": None,
        }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        ).stdout
        memory = []
        for line in output.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
                memory.append(int(fields[1]))
        if memory:
            result["gpu_memory_mib"] = sum(memory)
    except (OSError, subprocess.SubprocessError):
        pass  # Unsupported WDDM/container accounting is reported as unavailable.
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    print(json.dumps(sample(args.pid)))
