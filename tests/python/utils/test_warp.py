# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from typing import Any
from math import pi, sqrt
import pytest
import falcor2.testing.helpers as helpers
from falcor2.utils.chi2 import ChiSquareTest


def check_inverse(
    device: spy.Device,
    func: spy.Function | Any,
    inverse: spy.Function | Any,
    input_domain: Any = ((1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)),
    atol: float = 1e-5,
):
    for x in np.linspace(input_domain[0][0], input_domain[0][1], 10):
        for y in np.linspace(input_domain[1][0], input_domain[1][1], 10):
            p1 = spy.float2(x, y)
            p2 = func(p1)
            p3 = inverse(p2)
            assert np.allclose([p1.x, p1.y], p3, atol=atol)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_spherical_to_cartesian(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = module.spherical_to_cartesian, module.cartesian_to_spherical
    assert np.allclose(func(spy.float2(0, 0)), [0, 0, 1])
    assert np.allclose(func(spy.float2(0.5 * pi, 0)), [1, 0, 0], atol=1e-7)
    assert np.allclose(func(spy.float2(0.5 * pi, 0.5 * pi)), [0, 1, 0], atol=1e-7)
    check_inverse(
        device,
        func,
        inverse,
        input_domain=((1e-6, pi - 1e-6), (1e-6, 2 * pi - 1e-6)),
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_latlong_map_to_world(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = module.latlong_map_to_world, module.world_to_latlong_map
    assert np.allclose(func(spy.float2(0, 0)), [0, 1, 0])
    assert np.allclose(func(spy.float2(0, 0.5)), [0, 0, 1], atol=1e-7)
    assert np.allclose(func(spy.float2(0.5, 0.5)), [0, 0, -1], atol=1e-7)
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_disk_polar(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = module.square_to_uniform_disk_polar, module.uniform_disk_to_square_polar
    assert np.allclose(func(spy.float2(0.5, 0)), [0, 0])
    assert np.allclose(func(spy.float2(0, 1)), [1, 0])
    assert np.allclose(func(spy.float2(0.5, 1)), [-1, 0], atol=1e-7)
    assert np.allclose(func(spy.float2(1, 1)), [1, 0], atol=1e-6)
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_disk_concentric(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_uniform_disk_concentric,
        module.uniform_disk_to_square_concentric,
    )
    assert np.allclose(func(spy.float2(0, 0)), ([-1 / sqrt(2), -1 / sqrt(2)]))
    assert np.allclose(func(spy.float2(0.5, 0.5)), [0, 0])
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_sphere(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = module.square_to_uniform_sphere, module.uniform_sphere_to_square
    assert np.allclose(func(spy.float2(0, 0)), [0, 0, 1])
    assert np.allclose(func(spy.float2(0, 1)), [0, 0, -1])
    assert np.allclose(func(spy.float2(0.5, 0.5)), [-1, 0, 0], atol=1e-7)
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_hemisphere_polar(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_uniform_hemisphere_polar,
        module.uniform_hemisphere_to_square_polar,
    )
    assert np.allclose(func(spy.float2(0, 0)), [1, 0, 0])
    assert np.allclose(func(spy.float2(0.5, 0)), [-1, 0, 0], atol=1e-7)
    assert np.allclose(func(spy.float2(0.5, 0.5)), [-sqrt(0.75), 0, 0.5], atol=1e-7)
    assert np.allclose(func(spy.float2(0, 0.5)), [sqrt(0.75), 0, 0.5], atol=1e-7)
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_hemisphere_concentric(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_uniform_hemisphere_concentric,
        module.uniform_hemisphere_to_square_concentric,
    )
    assert np.allclose(func(spy.float2(0.5, 0.5)), [0, 0, 1])
    assert np.allclose(func(spy.float2(0, 0.5)), [-1, 0, 0])
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_cosine_hemisphere_polar(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_cosine_hemisphere_polar,
        module.cosine_hemisphere_to_square_polar,
    )
    assert np.allclose(func(spy.float2(0.5, 0.5)), [-1 / sqrt(2), 0, 1 / sqrt(2)], atol=1e-7)
    assert np.allclose(func(spy.float2(0.5, 0)), [0, 0, 1])
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_cosine_hemisphere_polar_dist(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_warp.slang"))

    test = ChiSquareTest(
        device=device,
        module=module,
        target="SquareToCosineHemispherePolarTarget",
        res=spy.uint2(201, 101),
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_cosine_hemisphere_concentric(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_cosine_hemisphere_concentric,
        module.cosine_hemisphere_to_square_concentric,
    )
    assert np.allclose(func(spy.float2(0.5, 0.5)), [0, 0, 1])
    assert np.allclose(func(spy.float2(0.5, 0)), [0, -1, 0], atol=1e-7)
    check_inverse(device, func, inverse)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_cosine_hemisphere_concentric_dist(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_warp.slang"))

    test = ChiSquareTest(
        device=device,
        module=module,
        target="SquareToCosineHemisphereConcentricTarget",
        res=spy.uint2(201, 101),
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_square_to_uniform_cone(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    func, inverse = (
        module.square_to_uniform_cone_concentric,
        module.uniform_cone_to_square_concentric,
    )
    assert np.allclose(func(spy.float2(0.5, 0.5), 1), [0, 0, 1])
    assert np.allclose(func(spy.float2(0.5, 0), 1), [0, 0, 1], atol=1e-7)
    assert np.allclose(func(spy.float2(0.5, 0), 0), [0, -1, 0], atol=1e-7)
    func_wrapper = lambda v: func(v, 0.3)
    inverse_wrapper = lambda v: inverse(v, 0.3)
    check_inverse(device, func_wrapper, inverse_wrapper)


def visualize_warp_2d_to_2d(
    device: spy.Device,
    module: spy.Module,
    mapping_func: str,
    inv_mapping_func: str,
    extra_args: Any = {},
    input_domain: Any = ((1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)),
    N: int = 16,
):
    float_type = module.layout.require_type_by_name("float")

    X, Y = np.meshgrid((np.linspace(0, N, N) / N), (np.linspace(0, N, N) / N))
    unit_square = np.ascontiguousarray(np.vstack([X.ravel(), Y.ravel()]).T, dtype=np.float32)
    colors = np.ascontiguousarray(
        np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())]).T, dtype=np.float32
    )

    # map unit square to input domain
    inputs_np = np.copy(unit_square)
    inputs_np[:, 0] = (
        inputs_np[:, 0] * (input_domain[0][1] - input_domain[0][0]) + input_domain[0][0]
    )
    inputs_np[:, 1] = (
        inputs_np[:, 1] * (input_domain[1][1] - input_domain[1][0]) + input_domain[1][0]
    )

    inputs = spy.Tensor.from_numpy(device, inputs_np)
    points = spy.Tensor.zeros(device, (N * N, 2), dtype=float_type)
    inv_points = spy.Tensor.zeros(device, (N * N, 2), dtype=float_type)
    module[mapping_func](inputs, _result=points, **extra_args)
    module[inv_mapping_func](points, _result=inv_points, **extra_args)
    points_np = points.to_numpy().view(np.float32)
    inv_points_np = inv_points.to_numpy().view(np.float32)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax0 = fig.add_subplot(131, aspect="equal")
    ax0.set_xlim(input_domain[0])
    ax0.set_ylim(input_domain[1])
    ax1 = fig.add_subplot(132, aspect="equal")
    ax2 = fig.add_subplot(133, aspect="equal")
    ax2.set_xlim(input_domain[0])
    ax2.set_ylim(input_domain[1])

    ax0.scatter(inputs_np[:, 0], inputs_np[:, 1], c=colors)
    ax1.scatter(points_np[:, 0], points_np[:, 1], c=colors)
    ax2.scatter(inv_points_np[:, 0], inv_points_np[:, 1], c=colors)

    plt.show()


def visualize_warp_2d_to_3d(
    device: spy.Device,
    module: spy.Module,
    mapping_func: str,
    inv_mapping_func: str,
    extra_args: Any = {},
    input_domain: Any = ((1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)),
    N: int = 16,
):
    float_type = module.layout.require_type_by_name("float")

    X, Y = np.meshgrid((np.linspace(0, N, N) / N), (np.linspace(0, N, N) / N))
    unit_square = np.ascontiguousarray(np.vstack([X.ravel(), Y.ravel()]).T, dtype=np.float32)
    colors = np.ascontiguousarray(
        np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())]).T, dtype=np.float32
    )

    # map unit square to input domain
    inputs_np = np.copy(unit_square)
    inputs_np[:, 0] = (
        inputs_np[:, 0] * (input_domain[0][1] - input_domain[0][0]) + input_domain[0][0]
    )
    inputs_np[:, 1] = (
        inputs_np[:, 1] * (input_domain[1][1] - input_domain[1][0]) + input_domain[1][0]
    )

    inputs = spy.Tensor.from_numpy(device, inputs_np)
    points = spy.Tensor.zeros(device, (N * N, 3), dtype=float_type)
    inv_points = spy.Tensor.zeros(device, (N * N, 2), dtype=float_type)
    module[mapping_func](inputs, _result=points, **extra_args)
    module[inv_mapping_func](points, _result=inv_points, **extra_args)
    points_np = points.to_numpy().view(np.float32).reshape(-1, 3)
    inv_points_np = inv_points.to_numpy().view(np.float32).reshape(-1, 2)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax0 = fig.add_subplot(131, aspect="equal")
    ax0.set_xlim(input_domain[0])
    ax0.set_ylim(input_domain[1])
    ax1 = fig.add_subplot(132, aspect="equal", projection="3d")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)  # type: ignore
    ax2 = fig.add_subplot(133, aspect="equal")
    ax2.set_xlim(input_domain[0])
    ax2.set_ylim(input_domain[1])

    ax0.scatter(inputs_np[:, 0], inputs_np[:, 1], c=colors)
    ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors)
    ax2.scatter(inv_points_np[:, 0], inv_points_np[:, 1], c=colors)

    plt.show()


def visualize_cartesian_to_spherical(device_type: spy.DeviceType = spy.DeviceType.automatic):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device,
        module,
        "spherical_to_cartesian",
        "cartesian_to_spherical",
        input_domain=((0, pi), (0, 2 * pi)),
    )


def visualize_latlong_map_to_world(device_type: spy.DeviceType = spy.DeviceType.automatic):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(device, module, "latlong_map_to_world", "world_to_latlong_map")


def visualize_square_to_uniform_disk_polar(device_type: spy.DeviceType = spy.DeviceType.automatic):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_2d(
        device, module, "square_to_uniform_disk_polar", "uniform_disk_to_square_polar"
    )


def visualize_square_to_uniform_disk_concentric(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_2d(
        device, module, "square_to_uniform_disk_concentric", "uniform_disk_to_square_concentric"
    )


def visualize_square_to_uniform_sphere(device_type: spy.DeviceType = spy.DeviceType.automatic):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(device, module, "square_to_uniform_sphere", "uniform_sphere_to_square")


def visualize_square_to_uniform_hemisphere_polar(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device, module, "square_to_uniform_hemisphere_polar", "uniform_hemisphere_to_square_polar"
    )


def visualize_square_to_uniform_hemisphere_concentric(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device,
        module,
        "square_to_uniform_hemisphere_concentric",
        "uniform_hemisphere_to_square_concentric",
    )


def visualize_square_to_cosine_hemisphere_polar(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device, module, "square_to_cosine_hemisphere_polar", "cosine_hemisphere_to_square_polar"
    )


def visualize_square_to_cosine_hemisphere_concentric(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device,
        module,
        "square_to_cosine_hemisphere_concentric",
        "cosine_hemisphere_to_square_concentric",
    )


def visualize_square_to_cone_concentric(device_type: spy.DeviceType = spy.DeviceType.automatic):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    visualize_warp_2d_to_3d(
        device,
        module,
        "square_to_cone_concentric",
        "cone_to_square_concentric",
        {"cos_cutoff": 0.25},
    )


# visualize_cartesian_to_spherical()
# visualize_latlong_map_to_world()
# visualize_square_to_uniform_disk_polar()
# visualize_square_to_uniform_disk_concentric()
# visualize_square_to_uniform_sphere()
# visualize_square_to_uniform_hemisphere_polar()
# visualize_square_to_uniform_hemisphere_concentric()
# visualize_square_to_cosine_hemisphere_polar()
# visualize_square_to_cosine_hemisphere_concentric()
# visualize_square_to_cone_concentric()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
