# SPDX-License-Identifier: Apache-2.0
import numpy as np

from slangpy import InstanceList, Module, Tensor, CommandEncoder, pack, Function
from slangpy.reflection import SlangType

from typing import Optional, cast


class OptimizerPool:
    def __init__(self, module: Module, optim_type_name: str):
        super().__init__()

        optim_type = module.find_struct(optim_type_name)
        if optim_type is None:
            raise ValueError(
                f"Could not find optimizer type '{optim_type_name}' in slang module '{module.name}'. "
                "This could be due to a missing import or a type error. Make sure "
                "this is a valid type in the module, e.g. by pasting in the type above "
                "and checking for compile errors"
            )

        batch_type = module.find_struct(f"{optim_type_name}::Batch")
        if batch_type is None:
            raise ValueError(
                f"Could not find optimizer batch type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        state_type = module.find_struct(f"{optim_type_name}::State")
        if state_type is None:
            raise ValueError(
                f"Could not find optimizer state type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        step_func = module.find_function_in_struct(optim_type, "step")
        if step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::step()' in slang module '{module.name}'. "
                f"Make sure the type {optim_type_name} implements IOptimizer"
            )

        batch_step_func = module.find_function_in_struct(optim_type, "batch_step")
        if batch_step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::batch_step()' in slang module '{module.name}'. "
                f"Make sure the type {optim_type_name} implements IOptimizer"
            )

        batch_zero_grads_func = module.find_function_in_struct(optim_type, "batch_zero_grads")
        if batch_zero_grads_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::batch_zero_grads()' in slang module '{module.name}'. "
                f"Make sure the type {optim_type_name} implements IOptimizer"
            )

        self.optim_type = optim_type
        self.state_type = state_type
        self.batch_type = batch_type
        self.step_func = step_func
        self.batch_step_func = cast(Function, batch_step_func)
        self.batch_zero_grads = cast(Function, batch_zero_grads_func)
        self.params: list[Tensor] = []
        self.mapping = np.ndarray((0, 2), dtype=np.int32)
        self.batches: list[InstanceList] = []

    def finalize(self):
        """
        Finalizes the optimizer pool, preparing it for use.
        This is called after all parameters have been added.
        """
        # Convert the mapping to a packed array
        self.mapping_buffer = Tensor.empty(
            self.optim_type.module.device, shape=(self.mapping.shape[0],), dtype="int2"
        )
        self.mapping_buffer.copy_from_numpy(self.mapping)

        # Create the packed batch data
        self.batches_packed = pack(self.optim_type.module, self.batches)

        # Workaround for Slang reflection issue - explicitly specialize batch_step and batch_zero_grads
        # for the number of batches we have, so that Slang can find the correct function.
        specialized_batch_step_func = self.optim_type.module.find_function_in_struct(
            self.optim_type, f"batch_step<{len(self.batches)}>"
        )
        assert specialized_batch_step_func is not None
        self.batch_step_func = cast(Function, specialized_batch_step_func)
        specialized_batch_zero_grads_func = self.optim_type.module.find_function_in_struct(
            self.optim_type, f"batch_zero_grads<{len(self.batches)}>"
        )
        assert specialized_batch_zero_grads_func is not None
        self.batch_zero_grads = cast(Function, specialized_batch_zero_grads_func)

    def add_parameter(self, param: Tensor, grad: Optional[Tensor] = None):
        """
        Adds a parameter to the optimizer pool. Optionally specify a separate gradient tensor,
        which will be used instead of the parameter's own gradient tensor.
        """

        param_idx = len(self.params)
        self.params.append(param)

        # If grad not specified, use the parameter's own gradient tensor.
        if grad is None:
            if param.grad is None:
                raise ValueError(
                    f"Parameterhas no gradient tensor. "
                    "Make sure to set the gradient tensor before adding it to the optimizer."
                )
            grad = cast(Tensor, param.grad)

        self.batches.append(
            InstanceList(
                self.batch_type,
                {
                    "params": param.storage,
                    "grads": grad.storage,
                    "states": self.state_type(param).storage,
                },
            )
        )

        # Append to the mapping array 1 entry for each element of the tensor,
        # where the entry is [param_idx, element_idx]
        new_mapping = np.column_stack(
            (
                np.full(param.element_count, param_idx, dtype=np.int32),
                np.arange(param.element_count, dtype=np.int32),
            )
        )
        self.mapping = np.vstack([self.mapping, new_mapping])


class Optimizer:
    """
    This is the base class of all optimizers.

    Creating an optimizer is done in two phases: First, by calling the constructor
    of the optimizer (e.g. AdamOptimizer) and setting its parameters. This is light-weight
    and does not do much work yet.
    Second, by calling .initialize(module, parameters), passing in the slang module containing
    the required slang types and a list of network parameters to optimize. This may perform
    allocation and reflection work.

    .step() performs one optimization step and resets the network gradients.

    For implementers of new optimizers, the following methods should be overridden:
    - get_type_name(dtype) returning the name of a slang type implementing IOptimizer<dtype>
    - get_this(), returning a python type that may be passed to slang (e.g. a dict)
    """

    def __init__(self, module: Module):
        super().__init__()
        self._initialized = False
        self.module = module

    def initialize(self, parameters: list[Tensor]):
        """
        Initializes the optimizer from a list of trainable parameters.

        The optimizer must be initialized before it can be used.

        module is a loaded slang module containing the required slang types.

        Parameter tensors don't all have to have the same precision, and it is allowed to use networks
        with e.g. mixed float and half precision parameters.
        """
        self._initialized = True
        self.parameters = parameters
        self.pools: dict[str, OptimizerPool] = {}

        for i, param in enumerate(parameters):
            type_name = self.get_type_name(param.dtype)

            pool = self._get_or_create_optimizer_pool(type_name)
            pool.add_parameter(param)

        for pool in self.pools.values():
            pool.finalize()

    def _get_or_create_optimizer_pool(self, optim_type_name: str) -> OptimizerPool:
        """
        Returns an existing optimizer pool for the given type, or creates a new one if it does not exist.
        """
        if optim_type_name not in self.pools:
            self.pools[optim_type_name] = OptimizerPool(self.module, optim_type_name)
        return self.pools[optim_type_name]

    def step(self, cmd: Optional[CommandEncoder] = None):
        """
        Performs one step of the optimizer and resets network gradients.

        If cmd is provided, the slang calls are appended to the given command buffer.
        """
        self.check_initialized()

        this = self.get_this()
        for pool in self.pools.values():
            if cmd is None:
                pool.batch_step_func(this, pool.batches_packed, pool.mapping_buffer)
            else:
                pool.batch_step_func.append_to(cmd, this, pool.batches_packed, pool.mapping_buffer)

    def zero_grads(self, cmd: Optional[CommandEncoder] = None):
        """
        Resets the gradients of all parameters in the optimizer.

        If cmd is provided, the slang calls are appended to the given command buffer.
        """
        self.check_initialized()

        this = self.get_this()
        for pool in self.pools.values():
            if cmd is None:
                pool.batch_zero_grads(this, pool.batches_packed, pool.mapping_buffer)
            else:
                pool.batch_zero_grads.append_to(cmd, this, pool.batches_packed, pool.mapping_buffer)

    def get_type_name(self, dtype: SlangType) -> str:
        """Returns the name of a slang type implementing IOptimizer<dtype>"""
        raise NotImplementedError()

    def get_this(self):
        """
        Returning a python type that may be passed to slang (e.g. a dict)

        Currently, this type has to be compatible with any optimizer precision.
        This may change in the future.
        """
        raise NotImplementedError()

    def check_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "Optimizer is uninitialized. Make sure to "
                "call .initialize() before using the optimizer"
            )
