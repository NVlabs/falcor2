// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public void fill_color<T, let N : int>(vector<T, N> color, out vector<T, N> output)
{
    output = color;
}

public void fill_scalar<T>(T val, out T output)
{
    output = val;
}
