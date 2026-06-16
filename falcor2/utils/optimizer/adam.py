# SPDX-License-Identifier: Apache-2.0
from .optimizer import Optimizer
from slangpy import Device, Module
from slangpy.reflection import SlangType


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        device: Device,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-6,
    ):
        super().__init__(Module.load_from_file(device, "falcor2/utils.slang"))

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_type_name(self, dtype: SlangType) -> str:
        return f"AdamOptimizer<{dtype.full_name}>"

    def get_this(self):
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }
