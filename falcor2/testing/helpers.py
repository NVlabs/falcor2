# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

import boto3
import pytest
import subprocess

from slangpy import (
    Device,
    DeviceType,
    SlangCompilerOptions,
    SlangDebugInfoLevel,
    SlangFloatingPointMode,
    Module,
    Logger,
    LogLevel,
    Function,
    NativeHandle,
)
import slangpy

if TYPE_CHECKING:
    import falcor2 as f2


if os.environ.get("SLANGPY_DEVICE", None) is not None:
    DEFAULT_DEVICE_TYPES = [DeviceType[os.environ["SLANGPY_DEVICE"]]]
elif sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "darwin":
    # TODO: we don't run any slangpy tests on metal due to slang bugs for now
    DEFAULT_DEVICE_TYPES = []  # [DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE: dict[
    tuple[
        DeviceType,
        bool,
        SlangFloatingPointMode,
        tuple[NativeHandle, ...],
        bool,
        Path | None,
        Path | None,
    ],
    Device,
] = {}

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODULE_AND_SHADER_CACHE_DIR = PROJECT_ROOT / ".shader-cache" / "tests"
MODULE_CACHE_ENABLED = False
SHADER_CACHE_ENABLED = False
MODULE_AND_SHADER_CACHE_ROOT = DEFAULT_MODULE_AND_SHADER_CACHE_DIR.resolve()
CONFIG: dict[str, Any] = {}
DRIVER_VERSION: Optional[Sequence[int]] = None
USED_TORCH_DEVICES: bool = False

# Enable this to make tests just run on d3d12 for faster testing
# DEFAULT_DEVICE_TYPES = [DeviceType.d3d12]

# Always dump stuff when testing
slangpy.set_dump_generated_shaders(True)


def configure_module_and_shader_cache(
    *,
    module_cache_enabled: bool = False,
    shader_cache_enabled: bool = False,
    cache_dir: str | Path | None = None,
) -> None:
    """Configure persistent module and shader caches used by test devices."""
    global MODULE_CACHE_ENABLED
    global SHADER_CACHE_ENABLED
    global MODULE_AND_SHADER_CACHE_ROOT

    root = Path(cache_dir) if cache_dir is not None else DEFAULT_MODULE_AND_SHADER_CACHE_DIR
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    MODULE_CACHE_ENABLED = module_cache_enabled
    SHADER_CACHE_ENABLED = shader_cache_enabled
    MODULE_AND_SHADER_CACHE_ROOT = root.resolve()


def get_module_and_shader_cache_path() -> tuple[Path | None, Path | None]:
    root = MODULE_AND_SHADER_CACHE_ROOT / "python"
    module_cache_path = root / "module" if MODULE_CACHE_ENABLED else None
    shader_cache_path = root / "shader" if SHADER_CACHE_ENABLED else None
    return module_cache_path, shader_cache_path


def get_ngx_vulkan_pre_device_info() -> Optional[Any]:
    try:
        import falcor2.ngx

        return falcor2.ngx.get_vulkan_pre_device_info()
    except (ImportError, RuntimeError):
        return None


# Returns a unique random 16 character string for every variant of every test.


@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


def create_test_camera(
    scene: f2.Scene,
    width: int = 64,
    height: int = 64,
    fov_y: float = 70.0,
    position: slangpy.float3 | None = None,
    rotation: Any = None,
) -> f2.Camera:
    """Create a camera component on a new entity in the given scene."""
    import falcor2 as f2

    entity = scene.create_entity()
    camera = entity.create_component(f2.Camera)
    camera.width = width
    camera.height = height
    camera.fov_y = fov_y
    if position is not None or rotation is not None:
        transform = f2.Transform()
        if position is not None:
            transform.translation = position
        if rotation is not None:
            transform.rotation = rotation
        entity.transform = transform
    return camera


def close_all_devices():
    # After all tests finish, close remaining devices before shared GPU contexts
    # such as torch start shutting down.
    global USED_TORCH_DEVICES
    if USED_TORCH_DEVICES:
        import torch

        torch.cuda.synchronize()

    for device in Device.get_created_devices():
        label = device.desc.label or "<unlabeled>"
        print(f"Closing device on shutdown {label}")
        device.close()
    DEVICE_CACHE.clear()


def close_leaked_devices():
    # Keep helper-managed cached devices alive between tests, but clean up
    # devices created directly by tests or public helper wrappers.
    cached_device_ids = {id(device) for device in DEVICE_CACHE.values()}
    for device in Device.get_created_devices():
        if id(device) in cached_device_ids or device.desc.label.startswith("cached-"):
            continue
        label = device.desc.label or "<unlabeled>"
        print(f"Closing leaked device {label}")
        device.close()


# Helper to get the driver version, as a list of version components
def get_driver_version():
    """Get the driver version string for the current device."""
    global DRIVER_VERSION
    if DRIVER_VERSION is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        DRIVER_VERSION = [int(x) for x in result.stdout.strip().split(".")]
    return DRIVER_VERSION


# Helper to get device of a given type
def get_device(
    type: DeviceType,
    use_cache: bool = True,
    cuda_interop: bool = False,
    floating_point_mode: SlangFloatingPointMode = SlangFloatingPointMode.default,
    existing_device_handles: Optional[Sequence[NativeHandle]] = None,
    enable_print: bool = False,
) -> Device:
    module_cache_path, shader_cache_path = get_module_and_shader_cache_path()
    cache_key = (
        type,
        cuda_interop,
        floating_point_mode,
        tuple(existing_device_handles) if existing_device_handles else (),
        enable_print,
        module_cache_path,
        shader_cache_path,
    )
    if use_cache and cache_key in DEVICE_CACHE:
        return DEVICE_CACHE[cache_key]

    additional_vulkan_instance_extensions: Optional[Sequence[str]] = None
    additional_vulkan_device_extensions: Optional[Sequence[str]] = None
    if type == DeviceType.vulkan:
        ngx_info = get_ngx_vulkan_pre_device_info()
        if ngx_info is not None:
            additional_vulkan_instance_extensions = ngx_info.required_vulkan_instance_extensions
            additional_vulkan_device_extensions = ngx_info.required_vulkan_device_extensions

    label = "cached-falcor2" if use_cache else "uncached-falcor2"
    label += f"-{type.name}"
    if cuda_interop:
        label += "-cuda-interop"
    if floating_point_mode != SlangFloatingPointMode.default:
        label += f"-{floating_point_mode.name}"
    if enable_print:
        label += "-print"

    device = Device(
        type=type,
        enable_debug_layers=True,
        enable_print=enable_print,
        compiler_options=SlangCompilerOptions(
            {
                "include_paths": [
                    slangpy.SHADER_PATH,
                    PROJECT_ROOT / "slang",
                    PROJECT_ROOT / "tests" / "python",
                    Path(__file__).parent,
                ],
                # This is a WAR for https://github.com/shader-slang/slang/issues/8886, to be fixed by https://github.com/shader-slang/slang/pull/8978
                "debug_info": (
                    SlangDebugInfoLevel.standard
                    if type != DeviceType.vulkan
                    else SlangDebugInfoLevel.none
                ),
                "floating_point_mode": floating_point_mode,
            }
        ),
        enable_cuda_interop=cuda_interop,
        existing_device_handles=existing_device_handles,
        module_cache_path=module_cache_path,
        shader_cache_path=shader_cache_path,
        additional_vulkan_instance_extensions=additional_vulkan_instance_extensions,
        additional_vulkan_device_extensions=additional_vulkan_device_extensions,
        label=label,
    )
    if use_cache:
        DEVICE_CACHE[cache_key] = device
    return device


def get_ngx(device: Device) -> Any:
    import falcor2.ngx

    try:
        ngx = falcor2.ngx.NGX.get(device)
    except RuntimeError as exc:
        pytest.skip(f"NGX is not available for {device.desc.type}: {exc}")

    if not ngx.info.dlssd_supported:
        pytest.skip(f"NGX DLSSD is not supported for {device.desc.type}")

    return ngx


# Helper that gets a device that wraps the current torch cuda context.
# This is useful for testing the torch integration.
def get_torch_device(type: DeviceType, use_cache: bool = True) -> Device:
    import torch

    global USED_TORCH_DEVICES
    USED_TORCH_DEVICES = True

    # For testing, comment this in to disable backwards passes running on other threads
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()

    handles = slangpy.get_cuda_current_context_native_handles()
    return get_device(
        type,
        use_cache=use_cache,
        existing_device_handles=handles,
        cuda_interop=True,
    )


def create_module(
    device: Device,
    module_source: str,
    link: list[Any] = [],
    options: dict[str, Any] = {},
) -> slangpy.Module:
    """
    Helper that creates a module from source (if not already loaded) and find / returns
    a kernel function for it. This helper supports nested functions and structs, e.g.
    create_function_from_module(device, "MyStruct.add_numbers", <src>).
    """

    if not 'import "slangpy";' in module_source:
        module_source = 'import "slangpy";\n' + module_source

    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    spy_module = slangpy.Module(module, link=link, options=options)
    spy_module.logger = Logger(level=LogLevel.warn)
    return spy_module


def create_function_from_module(
    device: Device,
    func_name: str,
    module_source: str,
    link: list[Any] = [],
    options: dict[str, Any] = {},
) -> slangpy.Function:
    """
    Helper that creates a module from source (if not already loaded) and find / returns
    a kernel function for it. This helper supports nested functions and structs, e.g.
    create_function_from_module(device, "MyStruct.add_numbers", <src>).
    """

    if not 'import "slangpy";' in module_source:
        module_source = 'import "slangpy";\n' + module_source

    slang_module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    module = Module(slang_module, link=link, options=options)
    module.logger = Logger(level=LogLevel.warn)

    names = func_name.split(".")

    if len(names) == 1:
        function = module.find_function(names[0])
    else:
        type_name = "::".join(names[:-1])
        function = module.find_function_in_struct(type_name, names[-1])
    if function is None:
        raise ValueError(f"Could not find function {func_name}")
    return cast(Function, function)


def assert_float3_equal(a: slangpy.float3, b: slangpy.float3, epsilon: float = 0.001):
    assert abs(a.x - b.x) < epsilon, f"{a.x} != {b.x}"
    assert abs(a.y - b.y) < epsilon, f"{a.y} != {b.y}"
    assert abs(a.z - b.z) < epsilon, f"{a.z} != {b.z}"


def get_ci_config_path():
    """Get the path to CI machine config file."""
    if os.name == "nt":
        return Path("c:\\ci\\config.json")
    elif os.name == "posix":
        return Path("/opt/ci/config.json")
    else:
        return None


def get_user_home_path():
    """Get the path to the user's home directory."""
    if os.name == "nt":
        return Path(os.environ.get("USERPROFILE", "C:\\"))
    elif os.name == "posix":
        return Path("~/").expanduser()
    else:
        return None


def get_config():
    global CONFIG
    if CONFIG:
        return CONFIG

    paths = []

    # Falcor2 config in user home directory or project root
    home_path = get_user_home_path()
    if home_path:
        paths.append(home_path / ".falcor2/secrets.json")
        paths.append(home_path / ".falcor2/config.json")

    # Falcor2 config in project root
    paths.append(PROJECT_ROOT / ".falcor2/secrets.json")
    paths.append(PROJECT_ROOT / ".falcor2/config.json")

    # CI config path
    cfpath = get_ci_config_path()
    if cfpath:
        paths.append(cfpath)

    # Merge configs
    config = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                c = json.load(f)
                config.update(c)
        except (json.JSONDecodeError, IOError) as e:
            pass
    CONFIG = config
    return CONFIG


def get_config_value(key: str) -> Optional[str]:
    """Get a configuration value from a config file."""
    return get_config().get(key, None)


def create_boto3_client():
    """Create an S3 client using environment variables for configuration."""
    ENDPOINT = "https://pdx.s8k.io"
    ACCESS_KEY = "team-nvr-ci"

    SECRET_KEY = None
    if SECRET_KEY is None:
        SECRET_KEY = get_config_value("s3key")
    if SECRET_KEY is None:
        SECRET_KEY = os.getenv("CSS_S3_KEY")  # TODO(ccummings): Move to NV Vault

    return boto3.client(
        "s3", endpoint_url=ENDPOINT, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )
