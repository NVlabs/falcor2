# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pytest
import numpy as np
from falcor2.utils import AdamOptimizer, GradientDescentOptimizer
import falcor2.testing.helpers as helpers


# Slang code for polynomial evaluation with automatic differentiation
POLYNOMIAL_TEST_CODE = """
import slangpy;

// Evaluate a quadratic polynomial: ax^2 + bx + c
[Differentiable]
float polynomial(float x, float a, float b, float c) {
    return a * x * x + b * x + c;
}

// Mean squared error loss function
[Differentiable]
float mse_loss(float predicted, float target) {
    float diff = predicted - target;
    return 0.5 * diff * diff;
}

// Training function that computes loss given parameters and data point
[Differentiable]
float train_step(float x, float target, float a, float b, float c) {
    // Forward pass
    float predicted = polynomial(x, a, b, c);

    // Compute loss
    float loss = mse_loss(predicted, target);

    return loss;
}
"""

CREATE_OPTIMIZER = {}
CREATE_OPTIMIZER["AdamOptimizer"] = lambda device: AdamOptimizer(device, learning_rate=1.0)
CREATE_OPTIMIZER["GradientDescentOptimizer"] = lambda device: GradientDescentOptimizer(
    device, learning_rate=0.0001
)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("optimizer_name", CREATE_OPTIMIZER.keys())
def test_optimizer_polynomial_fitting(device_type: spy.DeviceType, optimizer_name: str):
    """Test Adam optimizer by fitting a quadratic polynomial to data points"""
    device = helpers.get_device(device_type)

    # Get the differentiable train_step function
    train_step_func = helpers.create_function_from_module(
        device, "train_step", POLYNOMIAL_TEST_CODE
    ).return_type(spy.Tensor)

    # True polynomial coefficients: 2x^2 + 3x + 1
    true_a, true_b, true_c = 2.0, 3.0, 1.0

    # Generate training data
    np.random.seed(42)
    x_data = np.random.uniform(-2.0, 2.0, 1000).astype(np.float32)
    y_data = np.array([true_a * x**2 + true_b * x + true_c for x in x_data], dtype=np.float32)

    # Initialize parameters with random values as tensors
    x = spy.Tensor.from_numpy(device, x_data).with_grads()
    y = spy.Tensor.from_numpy(device, y_data).with_grads()
    a = spy.Tensor.from_numpy(device, np.array([0.1], dtype=np.float32)).with_grads()
    b = spy.Tensor.from_numpy(device, np.array([0.1], dtype=np.float32)).with_grads()
    c = spy.Tensor.from_numpy(device, np.array([0.1], dtype=np.float32)).with_grads()
    loss = spy.Tensor.zeros_like(x).with_grads()
    loss.grad.copy_from_numpy(np.ones_like(x.to_numpy()))

    # Create Adam optimizer
    optimizer = CREATE_OPTIMIZER[optimizer_name](device)
    optimizer.initialize([a, b, c])

    # Training loop
    num_epochs = 100
    encoder = device.create_command_encoder()
    for epoch in range(num_epochs):
        train_step_func.bwds.append_to(encoder, x, y, a, b, c, loss)
        optimizer.step(encoder)

    device.submit_command_buffer(encoder.finish())

    # Check that parameters converged to true values (within tolerance)
    final_a = a.to_numpy()[0]
    final_b = b.to_numpy()[0]
    final_c = c.to_numpy()[0]

    print(f"True params: a={true_a}, b={true_b}, c={true_c}")
    print(f"Learned params: a={final_a:.3f}, b={final_b:.3f}, c={final_c:.3f}")

    # Allow some tolerance for convergence
    assert abs(final_a - true_a) < 0.1, f"Parameter 'a' not converged: {final_a} vs {true_a}"
    assert abs(final_b - true_b) < 0.1, f"Parameter 'b' not converged: {final_b} vs {true_b}"
    assert abs(final_c - true_c) < 0.1, f"Parameter 'c' not converged: {final_c} vs {true_c}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
