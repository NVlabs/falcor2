# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from falcor2.testing import tier_selection

pytest_plugins = ["pytester"]


def _make_selection_plugin(pytester: pytest.Pytester) -> None:
    # Import only the selection module in the child suite: no renderer or GPU is
    # needed to establish fixture exclusion and parameter/marker semantics.
    module_path = str(Path(tier_selection.__file__).resolve())
    pytester.makeconftest(
        f"""
import importlib.util
import pytest
spec = importlib.util.spec_from_file_location("tier_selection", {module_path!r})
tiers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiers)

def pytest_addoption(parser):
    parser.addoption("--nightly", action="store_true")

def pytest_configure(config):
    config.addinivalue_line("markers", "nightly: deferred test")
    config.addinivalue_line("markers", "nightly_if: deferred parameters")
    config.addinivalue_line("markers", "nightly_if_not_vk: deferred non-Vulkan backends")

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    tiers.deselect_nightly_items(config, items, nightly=config.getoption("--nightly"))
"""
    )


def _make_suite(pytester: pytest.Pytester) -> None:
    _make_selection_plugin(pytester)
    pytester.makepyfile(
        """
import pytest
from pathlib import Path

@pytest.fixture
def expensive_fixture():
    Path("deferred-fixture-ran").write_text("yes")

@pytest.mark.nightly
def test_whole_case(expensive_fixture):
    pass

@pytest.mark.parametrize("api", ["primary", "secondary", "new-api"])
@pytest.mark.parametrize("variant", ["ordinary", "costly", "new-variant"])
@pytest.mark.nightly_if(api=["secondary"], variant=["costly"])
def test_partial_matrix(api, variant):
    pass

@pytest.mark.parametrize("value", [1, 2, 3, 4])
@pytest.mark.nightly_if(value=[2])
@pytest.mark.nightly_if(value=[3])
def test_union(value):
    pass

@pytest.mark.parametrize("value", [pytest.param(1, marks=pytest.mark.nightly), 2])
def test_parameter_mark(value):
    pass

@pytest.mark.skip(reason="capability unavailable")
def test_capability():
    assert False
"""
    )


@pytest.mark.parametrize("nightly", [False, True])
def test_tier_selection_excludes_fixture_work_and_restores_full_matrix(
    pytester: pytest.Pytester, nightly: bool
) -> None:
    _make_suite(pytester)
    result = pytester.runpytest("-q", *(["--nightly"] if nightly else []))
    result.assert_outcomes(passed=16 if nightly else 11, skipped=1, deselected=0 if nightly else 5)
    assert (pytester.path / "deferred-fixture-ran").exists() == nightly


@pytest.mark.parametrize(
    "marker, diagnostic",
    [
        ("pytest.mark.nightly_if(1)", "requires named parameter value lists"),
        ("pytest.mark.nightly_if(missing=[1])", "unknown parameter 'missing'"),
        ("pytest.mark.nightly_if(value=1)", "requires a list or tuple"),
    ],
)
def test_tier_selection_rejects_invalid_source_annotations(
    pytester: pytest.Pytester, marker: str, diagnostic: str
) -> None:
    _make_suite(pytester)
    pytester.makepyfile(
        test_invalid=f"""
import pytest
@pytest.mark.parametrize("value", [1])
@{marker}
def test_invalid(value):
    pass
"""
    )
    result = pytester.runpytest("--collect-only", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert diagnostic in result.stderr.str()


@pytest.mark.parametrize("nightly", [False, True])
def test_non_vulkan_selection_preserves_conditions_and_excludes_fixture_work(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, nightly: bool
) -> None:
    _make_selection_plugin(pytester)
    # A restricted backend selection must not promote CUDA to a regular backend.
    monkeypatch.setenv("SLANGPY_DEVICE", "cuda")
    pytester.makepyfile(
        """
from pathlib import Path
import pytest
from slangpy import DeviceType

DEVICE_TYPES = [DeviceType.vulkan, DeviceType.d3d12, DeviceType.cuda, DeviceType.metal]

@pytest.fixture
def device_witness(device_type: DeviceType) -> DeviceType:
    Path(f"direct-{DEVICE_TYPES.index(device_type)}").write_text("yes")
    return device_type

@pytest.fixture
def _bsdf_resources(request: pytest.FixtureRequest) -> DeviceType:
    Path(f"indirect-{DEVICE_TYPES.index(request.param)}").write_text("yes")
    return request.param

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.nightly_if_not_vk
def test_backends(device_type: DeviceType, device_witness: DeviceType) -> None:
    assert device_witness == device_type

@pytest.mark.parametrize("_bsdf_resources", DEVICE_TYPES, indirect=True)
@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_indirect(_bsdf_resources: DeviceType) -> None:
    assert isinstance(_bsdf_resources, DeviceType)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("variant", ["ordinary", "costly"])
@pytest.mark.nightly_if_not_vk(variant=("costly",))
def test_conditional(device_type: DeviceType, variant: str) -> None:
    pass

@pytest.mark.parametrize("device_type", [DeviceType.vulkan, DeviceType.cuda])
@pytest.mark.nightly_if_not_vk
@pytest.mark.nightly_if(device_type=[DeviceType.vulkan])
def test_union(device_type: DeviceType) -> None:
    pass

@pytest.mark.parametrize("device_type", [DeviceType.vulkan, DeviceType.metal])
@pytest.mark.nightly_if_not_vk
@pytest.mark.nightly
def test_whole_case(device_type: DeviceType) -> None:
    Path("whole-case-ran").write_text("yes")

@pytest.mark.parametrize("device_type", [DeviceType.cuda])
@pytest.mark.nightly_if_not_vk
def test_only_cuda(device_type: DeviceType, device_witness: DeviceType) -> None:
    assert device_witness == DeviceType.cuda

@pytest.mark.parametrize("device_type", [])
@pytest.mark.nightly_if_not_vk
def test_no_configured_backend(device_witness: DeviceType) -> None:
    pytest.fail("An empty backend matrix must remain skipped")

@pytest.mark.parametrize("_bsdf_resources", [], indirect=True)
@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")
def test_no_indirect_backend(_bsdf_resources: DeviceType) -> None:
    pytest.fail("An empty indirect backend matrix must remain skipped")
"""
    )
    result = pytester.runpytest("-q", *(["--nightly"] if nightly else []))
    result.assert_outcomes(passed=21 if nightly else 7, skipped=2, deselected=0 if nightly else 14)
    for prefix in ("direct", "indirect"):
        for index in range(4):
            assert (pytester.path / f"{prefix}-{index}").exists() == (nightly or index == 0)
    assert (pytester.path / "whole-case-ran").exists() == nightly


@pytest.mark.parametrize(
    "marker, device_value, diagnostic",
    [
        ("pytest.mark.nightly_if_not_vk(1)", "DeviceType.vulkan", "only keyword arguments"),
        (
            "pytest.mark.nightly_if_not_vk(device_param=1)",
            "DeviceType.vulkan",
            "device_param requires a parameter name",
        ),
        (
            "pytest.mark.nightly_if_not_vk(device_param='missing')",
            "DeviceType.vulkan",
            "unknown parameter 'missing'",
        ),
        ("pytest.mark.nightly_if_not_vk", "'vulkan'", "requires a DeviceType value"),
        (
            "pytest.mark.nightly_if_not_vk(missing=[1])",
            "DeviceType.vulkan",
            "unknown parameter 'missing'",
        ),
        (
            "pytest.mark.nightly_if_not_vk(value=1)",
            "DeviceType.vulkan",
            "requires a list or tuple",
        ),
    ],
)
def test_non_vulkan_selection_rejects_invalid_source_annotations(
    pytester: pytest.Pytester, marker: str, device_value: str, diagnostic: str
) -> None:
    _make_selection_plugin(pytester)
    pytester.makepyfile(
        f"""
import pytest
from slangpy import DeviceType

@pytest.mark.parametrize("device_type", [{device_value}])
@pytest.mark.parametrize("value", [1])
@{marker}
def test_invalid(device_type: DeviceType, value: int) -> None:
    pass
"""
    )
    result = pytester.runpytest("--collect-only", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert diagnostic in result.stderr.str()
