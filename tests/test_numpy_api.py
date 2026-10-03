# SPDX-License-Identifier: MIT
"""Array facade tests; provider kernels are exercised, never substituted."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sensitivity.covariance import first_order_covariance
from sensitivity.jacobian import finite_difference_jacobian
from sensitivity.models import DifferentiableModel
from sensitivity.numpy_api import jacobian, linearize, propagate_covariance


@pytest.mark.parametrize("method", ["central", "forward"])
def test_numerical_methods_reuse_provider_kernel(method):
    f = lambda x: np.array([x[0] ** 2 + x[1], np.sin(x[1])])
    x = np.array([2.0, 0.3])
    model = DifferentiableModel("fixture", f, 2, 2)
    expected = finite_difference_jacobian(model, x, method=method, relative_step=1e-5)
    actual = linearize(f, x, method=method, relative_step=1e-5)
    assert_array_equal(actual.jacobian, expected.matrix)
    assert_array_equal(actual.step, expected.step)
    assert actual.source == method


def test_affine_and_rectangular_output():
    a = np.array([[2.0, -1.0], [0.0, 3.0], [1.0, 2.0]])
    j = jacobian(lambda x: a @ x + 7.0, [2, 3])
    assert_allclose(j, a, atol=1e-9)
    assert j.shape == (3, 2) and j.dtype == np.float64 and j.flags.writeable


def test_scalar_output_and_input_are_length_one():
    j = jacobian(lambda x: x[0] ** 2, 3.0)
    assert j.shape == (1, 1)
    assert_allclose(j, [[6.0]], atol=1e-9)


def test_nonlinear_value_is_not_jacobian_times_point():
    result = linearize(lambda x: x**2 + 1, [2.0], derivative=lambda x: [[2 * x[0]]])
    assert_array_equal(result.value, [5.0])
    assert_array_equal(result.jacobian, [[4.0]])
    assert result.source == "analytical" and result.step is None
    assert_array_equal(result.propagate_covariance([[0.25]]), [[4.0]])


def test_snapshots_do_not_alias_input_or_each_other():
    x = np.array([2.0, 3.0])
    result = linearize(lambda y: y, x)
    x[:] = 99
    assert_array_equal(result.point, [2.0, 3.0])
    for array in (result.point, result.value, result.jacobian, result.step):
        assert not array.flags.writeable
        assert not np.shares_memory(array, x)
    assert not np.shares_memory(result.point, result.value)


def test_callback_cannot_mutate_caller_input():
    x = np.array([2.0])
    def bad(v):
        v[0] = 7
        return v
    with pytest.raises(ValueError):
        linearize(bad, x)
    assert_array_equal(x, [2.0])


@pytest.mark.parametrize("x", [[], [[1.0]], [np.nan], [np.inf], [1j], [True], ["1"]])
def test_invalid_input_refused(x):
    with pytest.raises(ValueError):
        jacobian(lambda y: y, x)


@pytest.mark.parametrize("value", [[], [[1.0]], [np.inf], [1j], [True]])
def test_invalid_function_output_refused(value):
    with pytest.raises(ValueError):
        jacobian(lambda x: value, [1.0])


def test_nonfinite_probe_refused():
    with pytest.raises(ValueError, match="finite"):
        jacobian(lambda x: [0.0] if x[0] == 1 else [np.nan], [1.0])


def test_changing_output_dimension_refused():
    with pytest.raises(ValueError, match="output dim"):
        jacobian(lambda x: [1.0] if x[0] == 1 else [1.0, 2.0], [1.0])


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf, True, [1e-6]])
def test_invalid_steps_refused(step):
    with pytest.raises(ValueError):
        jacobian(lambda x: x, [1.0], relative_step=step)


def test_unrepresentable_step_refused():
    with pytest.raises(ValueError, match="representable"):
        jacobian(lambda x: x, [1.0], relative_step=1e-30)


@pytest.mark.parametrize("kwargs", [
    {"method": "complex"}, {"method": "analytical"},
    {"method": "central", "derivative": lambda x: [[1.0]]},
    {"derivative": lambda x: [[1.0]], "relative_step": 1e-6},
])
def test_ambiguous_or_unsupported_methods_refused(kwargs):
    with pytest.raises(ValueError):
        jacobian(lambda x: x, [1.0], **kwargs)


def test_analytical_shape_and_values_checked():
    for derivative in (lambda x: [1.0], lambda x: [[1.0, 2.0]], lambda x: [[np.nan]]):
        with pytest.raises(ValueError):
            jacobian(lambda x: x, [1.0], derivative=derivative)


def test_noncallables_refused():
    with pytest.raises(TypeError):
        linearize(3, [1.0])
    with pytest.raises(TypeError):
        linearize(lambda x: x, [1.0], derivative=3)


def test_covariance_parity_and_cross_correlation():
    p = np.ones((3, 3)) + 0.09 * np.eye(3)
    j = np.ones((1, 3)) / 3
    assert_allclose(propagate_covariance(j, p), [[1.03]])
    assert_array_equal(propagate_covariance(j, p), first_order_covariance(j, p))
    assert_allclose(propagate_covariance([[1, -1, 0]], p), [[0.18]])


def test_singular_and_exact_zero_covariance_supported():
    assert_array_equal(propagate_covariance([[1, -1]], np.ones((2, 2))), [[0.0]])
    assert_array_equal(propagate_covariance(np.eye(2), [[0, 0], [0, 3]]), [[0, 0], [0, 3]])


@pytest.mark.parametrize("p", [
    [[1, 2], [2, 1]], [[0, 1e-30], [1e-30, 1]],
    [[1, 0.1], [0.3, 1]], [[-1, 0], [0, 1]],
    [[np.nan, 0], [0, 1]], [[1, 0]], [], [1, 2],
])
def test_invalid_covariance_refused_without_repair(p):
    with pytest.raises(ValueError):
        propagate_covariance(np.eye(2), p)


def test_coordinate_scaling_respects_full_covariance():
    p = np.array([[4.0, 0.5], [0.5, 1.0]])
    j = np.array([[2.0, -3.0]])
    t = np.diag([1000.0, 0.01])
    converted = propagate_covariance(j @ np.linalg.inv(t), t @ p @ t.T)
    assert_allclose(converted, propagate_covariance(j, p), rtol=1e-14)


@pytest.mark.parametrize("j,p", [([[np.nan]], [[1]]), ([[1j]], [[1]]), ([[1e308]], [[1e308]]), ([[1, 2]], [[1]])])
def test_jacobian_and_overflow_refusals(j, p):
    with pytest.raises(ValueError):
        propagate_covariance(j, p)
