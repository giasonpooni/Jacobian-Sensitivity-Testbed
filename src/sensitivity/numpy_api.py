# SPDX-License-Identifier: MIT
"""Array/callable facade over JSPT's existing local numerical kernels.

No runtime, evidence ledger, engine dependency, or automatic noise model.
This real-valued, unbatched API is additive; existing model APIs are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import _require_psd, first_order_covariance
from .jacobian import finite_difference_jacobian, jacobian_at
from .models import DifferentiableModel

Array = NDArray[np.float64]
Function = Callable[[Array], ArrayLike]
Method = Literal["auto", "analytical", "central", "forward"]
__all__ = ["Linearization", "jacobian", "linearize", "propagate_covariance"]


def _array(value: ArrayLike, name: str, ndim: int) -> Array:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not {raw.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    if ndim == 1 and result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _readonly(value: Array) -> Array:
    result = value.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Linearization:
    """Local value and derivative; arrays are independent, read-only snapshots.

    ``value`` is f(point), not J @ point. ``source`` describes how the derivative
    was obtained, not whether it has been independently verified.
    """

    point: Array
    value: Array
    jacobian: Array
    source: str
    step: Array | None

    def propagate_covariance(self, covariance: ArrayLike) -> Array:
        """Push a joint input covariance through this local Jacobian."""
        return propagate_covariance(self.jacobian, covariance)


def linearize(
    function: Function,
    x: ArrayLike,
    *,
    derivative: Function | None = None,
    method: Method = "auto",
    relative_step: float | None = None,
) -> Linearization:
    """Evaluate a pure f: R^n -> R^m and its local Jacobian at x.

    Scalars are normalized to length-one vectors. Callbacks receive read-only
    copies of 1-D float64 inputs. No batching, complex step, or implicit JAX
    tracing is attempted. A supplied derivative is caller-declared, not checked.
    Finite differences inherit JSPT's steps and local approximation limits.
    """
    if not callable(function) or (derivative is not None and not callable(derivative)):
        raise TypeError("function and any derivative must be callable")
    if method not in {"auto", "analytical", "central", "forward"}:
        raise ValueError(f"unsupported method {method!r}")
    selected = ("analytical" if derivative is not None else "central") if method == "auto" else method
    if selected == "analytical" and derivative is None:
        raise ValueError("analytical method requires derivative")
    if selected != "analytical" and derivative is not None:
        raise ValueError("derivative would be ignored by the selected numerical method")
    if relative_step is not None:
        step_arg = np.asarray(relative_step)
        if step_arg.ndim != 0 or step_arg.dtype.kind not in "iuf":
            raise ValueError("relative_step must be a positive finite real scalar")
        relative_step = float(step_arg)
        if not np.isfinite(relative_step) or relative_step <= 0:
            raise ValueError("relative_step must be a positive finite real scalar")
        if selected == "analytical":
            raise ValueError("relative_step does not apply to an analytical derivative")
    point = _array(x, "x", 1)

    def forward(probe: Array) -> Array:
        return _array(function(_readonly(probe)), "function output", 1)

    value = forward(point)

    def analytical(probe: Array) -> Array:
        assert derivative is not None
        return _array(derivative(_readonly(probe)), "derivative output", 2)

    model = DifferentiableModel(
        name=getattr(function, "__name__", "callable"),
        forward=forward,
        input_dim=point.size,
        output_dim=value.size,
        jacobian=analytical if derivative is not None else None,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if selected == "analytical":
            estimate = jacobian_at(model, point, source="analytical")
        else:
            estimate = finite_difference_jacobian(
                model, point, method=selected, relative_step=relative_step,
            )
        matrix = _array(estimate.matrix, "Jacobian", 2)
        if estimate.step is not None:
            step = _array(estimate.step, "finite-difference step", 1)
            probes = [point + step]
            if selected == "central":
                probes.append(point - step)
            if np.any(step <= 0) or any(
                not np.all(np.isfinite(probe)) or np.any(probe == point)
                for probe in probes
            ):
                raise ValueError("finite-difference step is not representable at x")
        else:
            step = None
    return Linearization(
        _readonly(point), _readonly(value), _readonly(matrix), estimate.source,
        None if step is None else _readonly(step),
    )


def jacobian(
    function: Function,
    x: ArrayLike,
    *,
    derivative: Function | None = None,
    method: Method = "auto",
    relative_step: float | None = None,
) -> Array:
    """Return an ordinary writable (m, n) array; use linearize for diagnostics."""
    return linearize(
        function, x, derivative=derivative, method=method, relative_step=relative_step,
    ).jacobian.copy()


def propagate_covariance(jacobian: ArrayLike, covariance: ArrayLike) -> Array:
    """Return J P J.T using JSPT's existing correlation-scaled PSD checks.

    P is the full joint input covariance; cross-correlations are never dropped.
    No noise term, independence assumption, clipping, or nearest-PSD repair is
    introduced. The result is local/first-order unless the map is affine.
    """
    matrix = _array(jacobian, "Jacobian", 2)
    cov = _array(covariance, "covariance", 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = first_order_covariance(matrix, cov)
        # Validate the output too: the older low-level API can overflow even
        # when every input is finite. Reuse its validator rather than repair.
        return _require_psd(result, "propagated covariance").copy()
