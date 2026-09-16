# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import slangpy as spy


_CUDA_CAPABILITY_PREFIX = "_cuda_sm_"
_CUDA_ARCHITECTURE_ARGUMENTS = ("--gpu-architecture", "-arch")
_CUDA_ARCHITECTURE_ARGUMENT_PREFIXES = (
    "--gpu-architecture=",
    "-arch=",
)


def configure_cuda_architecture(
    compiler_options: spy.SlangCompilerOptions,
    device: spy.Device | None,
    architecture: int | None = None,
) -> int:
    """Configure the CUDA architecture passed to Slang's downstream compiler.

    The architecture uses CUDA's ``major * 10 + minor`` encoding. For example,
    ``75`` denotes compute capability 7.5 and produces ``compute_75``, while
    ``120`` denotes compute capability 12.0 and produces ``compute_120``.
    Passing ``None`` selects the highest compute capability reported by the
    CUDA device. An explicit value is used exactly and is not clamped.

    Args:
        compiler_options: Slang compiler options to update.
        device: CUDA device whose capabilities are used for automatic selection,
            or ``None`` when specifying an explicit architecture.
        architecture: Explicit architecture, or ``None`` to select automatically.

    Returns:
        The selected CUDA architecture using ``major * 10 + minor`` encoding.

    Raises:
        ValueError: If an explicit architecture is not positive, or automatic
            selection is requested without a CUDA device.
        RuntimeError: If automatic selection is requested and the device reports
            no CUDA compute capability.
    """
    if architecture is not None and architecture <= 0:
        raise ValueError("Explicit CUDA architecture must be greater than zero.")

    device_architectures: list[int] = []
    if device is not None and device.info.type == spy.DeviceType.cuda:
        for capability in device.capabilities:
            if not capability.startswith(_CUDA_CAPABILITY_PREFIX):
                continue

            version = capability[len(_CUDA_CAPABILITY_PREFIX) :].split("_")
            if len(version) != 2:
                continue
            try:
                major, minor = (int(part) for part in version)
            except ValueError:
                continue
            device_architectures.append(major * 10 + minor)

    device_architecture = max(device_architectures) if device_architectures else None
    if architecture is None:
        if device is None or device.info.type != spy.DeviceType.cuda:
            raise ValueError("CUDA architecture configuration requires a CUDA device.")
        if device_architecture is None:
            raise RuntimeError("CUDA device capabilities do not report a CUDA compute capability.")
        selected_architecture = device_architecture
    else:
        selected_architecture = architecture

    downstream_args = list(compiler_options.downstream_args)
    filtered_args: list[str] = []
    skip_next = False
    for argument in downstream_args:
        if skip_next:
            skip_next = False
            continue

        # Split key/value form: "--gpu-architecture" "compute_120" or "-arch" "compute_120".
        if argument in _CUDA_ARCHITECTURE_ARGUMENTS:
            skip_next = True
            continue

        # Joined key=value form: "--gpu-architecture=compute_120" or "-arch=compute_120".
        if argument.startswith(_CUDA_ARCHITECTURE_ARGUMENT_PREFIXES):
            continue

        filtered_args.append(argument)

    architecture_argument = f"--gpu-architecture=compute_{selected_architecture}"
    filtered_args.append(architecture_argument)
    compiler_options.downstream_args = filtered_args

    device_architecture_name = (
        "unknown" if device_architecture is None else f"compute_{device_architecture}"
    )
    spy.log_info(
        f"CUDA architecture: device={device_architecture_name}, "
        f"selected=compute_{selected_architecture} ({architecture_argument})"
    )
    return selected_architecture
