# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import slangpy as spy

from falcor2.testing.image_test_plugin import ImageTest
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("generate", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_image_never_reaches_reference_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate: bool, value: float
) -> None:
    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    plugin = Mock()
    check = ImageTest("invalid", "fixture", tmp_path, generate, plugin)
    image = np.ones((2, 2, 4), dtype=np.float32)
    image[0, 0, 1] = value
    with pytest.raises(AssertionError, match="NaN/Inf"):
        check(image)
    assert plugin.mock_calls == []
    assert not list(tmp_path.rglob("*.exr"))


@pytest.mark.parametrize("invalid", ["nan", "shape"])
def test_invalid_reference_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    plugin = Mock()
    plugin.get_metadata.return_value = {"reference": {"reference": "stored"}}
    check = ImageTest("reference", "fixture", tmp_path, False, plugin)
    stored = np.ones((3 if invalid == "shape" else 2, 2, 4), dtype=np.float32)
    if invalid == "nan":
        stored[0, 0, 0] = np.nan
    spy.Bitmap(stored).write(str(check.references_dir / "stored.exr"))
    result = check._compare_with_reference(
        np.ones((2, 2, 4), dtype=np.float32), "different", None, 1
    )
    assert result.status == "failed"
    assert "Invalid reference" in result.message


def test_hdr_error_does_not_overflow_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    plugin = Mock()
    plugin.get_metadata.return_value = {"hdr": {"reference": "stored"}}
    check = ImageTest("hdr", "fixture", tmp_path, False, plugin)
    spy.Bitmap(np.zeros((2, 2, 4), dtype=np.float32)).write(
        str(check.references_dir / "stored.exr")
    )
    monkeypatch.setattr(check, "_save_error_images", Mock())
    result = check._compare_with_reference(
        np.full((2, 2, 4), 1e20, dtype=np.float32), "different", None, 1
    )
    assert result.status == "failed"
    assert np.isfinite(result.difference)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf])
def test_invalid_tolerance_cannot_accept_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: float
) -> None:
    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    plugin = Mock()
    check = ImageTest("tolerance", "fixture", tmp_path, False, plugin)
    with pytest.raises(ValueError, match="tolerance"):
        check(np.ones((2, 2, 3), dtype=np.float32), tolerance=tolerance)
    assert plugin.mock_calls == []
