# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from falcor2.testing import _native


def test_python_assignment_dispatches_cpp_on_change() -> None:
    obj = _native.OnChangeTest()
    assert obj.value == 0
    assert obj.on_change_count == 0

    obj.value = 42
    assert obj.value == 42
    assert obj.on_change_count == 1

    obj.value = 99
    assert obj.value == 99
    assert obj.on_change_count == 2
