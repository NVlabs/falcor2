# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pyscene_native_loading_support import configure


configure()
raise RuntimeError("intentional native PyScene loading failure")
