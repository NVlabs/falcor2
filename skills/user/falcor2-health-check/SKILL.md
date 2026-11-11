---
name: falcor2-health-check
description: Check whether Falcor2 is correctly installed in the selected Python environment, native imports work, and a GPU device can be created with the installed driver. Use for installation verification or diagnosing import, GPU, and driver setup failures.
---

# Falcor2 health check

Identify the project's selected Python environment from its configuration and
dependency records. Use that interpreter, not the agent's bundled Python. Preserve
the existing environment and package manager, whether using wheels or a checkout.

Run that interpreter with [scripts/verify.py](scripts/verify.py). It reports Python,
Falcor2 and SlangPy versions and import locations, optional NVIDIA driver-query
diagnostics, and actual GPU device creation. Use a bounded execution timeout and
report a crash or timeout as a failed check. `--imports-only` skips GPU and driver
checks; explicitly report those as unverified, even when the command succeeds.

Report the failed stage and original diagnostic. A successful GPU check means the
installed driver supports device creation for the reported adapter/API; it does
not certify every GPU, rendering correctness, or optional CUDA/PyTorch, DLSS, or
OptiX features. A missing `nvidia-smi` alone does not prove a broken driver. Do not
infer a driver failure solely from a missing package or native-library import.

Suggest the smallest next step based on evidence. Do not install packages or
drivers, change environments, rebuild source, or generate an application as part
of this diagnostic skill unless the user separately requests a repair.
