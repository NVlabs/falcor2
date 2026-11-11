# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from falcor2.testing import _native


@pytest.mark.parametrize("value", ["", "ascii", "\u00e9/\u4e16\u754c/\U0001f642", "a\0b"])
def test_roundtrip(value: str) -> None:
    assert _native._ustring_roundtrip(value) == value
    assert _native._ustring_shares_storage(value, (value + "!")[:-1])


@pytest.mark.parametrize("value", [None, 42, b"usd", ["usd"], "\ud800"])
def test_rejects_invalid_input(value: object) -> None:
    with pytest.raises(TypeError):
        _native._ustring_roundtrip(value)


def test_rejects_invalid_utf8_output() -> None:
    with pytest.raises(UnicodeDecodeError):
        _native._ustring_invalid_utf8()


def test_signature_uses_python_string() -> None:
    assert "value: str" in _native._ustring_roundtrip.__doc__
