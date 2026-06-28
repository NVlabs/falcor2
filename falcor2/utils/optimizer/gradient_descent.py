# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .optimizer import Optimizer
from slangpy import Device, Module
from slangpy.reflection import SlangType


class GradientDescentOptimizer(Optimizer):
    def __init__(self, device: Device, learning_rate: float):
        super().__init__(Module.load_from_file(device, "falcor2/utils.slang"))
        self.learning_rate = learning_rate

    def get_type_name(self, dtype: SlangType) -> str:
        return f"GradientDescentOptimizer<{dtype.full_name}>"

    def get_this(self):
        return {"learning_rate": self.learning_rate}
