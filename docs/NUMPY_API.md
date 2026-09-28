# NumPy microtools: callable and array API

This additive facade lets a notebook or application use JSPT without creating
experiment, evidence, or workbench objects. It calls the existing provider
kernels; it does not copy them into NET or create another numerical package.
The repository's visibility and MIT licence are unchanged. This is not a PyPI
release or a claim of inclusion in NumPy/SciPy.

```python
import numpy as np
from sensitivity.numpy_api import jacobian, linearize, propagate_covariance

f = lambda x: np.array([x[0] ** 2 + x[1]])
x = np.array([2.0, 3.0])
J = jacobian(f, x)  # shape (1, 2), approximately [[4, 1]]
P_y = propagate_covariance(J, [[0.04, 0.01], [0.01, 0.09]])
np.testing.assert_allclose(P_y, [[0.81]], rtol=1e-8)

local = linearize(f, x, derivative=lambda x: [[2 * x[0], 1.0]])
assert local.source == "analytical"  # caller-declared, not verified
np.testing.assert_array_equal(local.value, [7.0])  # f(x), not J @ x
```

Install from an authorized checkout with `python -m pip install -e .`.
Run the new tests with `python -m pytest tests/test_numpy_api.py -q`.
Run the complete existing suite before merging with `python -m pytest -q`.

## Interface

| Function | Output | Responsibility |
| --- | --- | --- |
| `jacobian(f, x, ...)` | writable float64 `(m, n)` ndarray | Local derivative through the existing kernel. |
| `linearize(f, x, ...)` | `Linearization` | Independent read-only point/value/Jacobian snapshots, derivative source and numerical step. |
| `propagate_covariance(J, P)` | float64 `(m, m)` ndarray | Full correlated covariance using existing PSD validation. |

Scalars normalize to length-one vectors. Inputs are unbatched finite real
numeric arrays; complex, boolean, string, empty, and higher-dimensional inputs
are refused rather than silently reinterpreted. Callbacks must be pure,
deterministic functions of a read-only 1-D input. They can be evaluated more
than once. Shape-changing outputs and invalid probe values are refused.

`method="auto"` selects a supplied derivative or the existing central-difference
method. Explicit `central`, `forward`, and `analytical` are supported. Ignored
settings are refused: a numerical method cannot silently ignore `derivative`,
and an analytical method cannot silently ignore `relative_step`. Complex-step,
JAX, declared-model workflows, and their existing APIs remain unchanged; this
small facade does not extend them. Numerical steps follow the existing kernel;
nonpositive, nonfinite, or unrepresentable steps are refused.

## Scientific limits and authority

`P_y = J P_x J.T` is exact for an affine map and first-order/local otherwise.
The full joint input covariance is required. Shared uncertainties and negative
weights remain meaningful; no diagonalization or independent-noise assumption
is inserted. Invalid covariance is refused with existing correlation-scaled
checks, including exact-zero-variance cross-covariance rules. No eigenvalue
clipping or nearest-PSD repair is added. Outputs are checked for overflow and
PSD as well as inputs.

A supplied derivative is not independently verified. A small Jacobian does
not establish trajectory observability, identifiability, stability, or global
sensitivity. Units, frames, and clocks must already be consistent. This array
API does not invent scientific evidence or operation/execution/verification
IDs. Use the existing `sensitivity.ciw_adapter` and versioned CIW contracts for
a retained NET operation; that endpoint and all provider pins are unchanged.

Engine consumers should receive explicit results or buffers. They must not
embed or publish private provider source implicitly. Blender/Godot/Bevy
bindings and hard-real-time qualification are outside this increment.

## Validation in this increment

49 new tests passed locally on Python 3.13.5 / NumPy 2.3.5. The local run used
a partial source checkout containing this facade and the exact existing
`models.py`, `jacobian.py`, and `covariance.py` blobs, verified with Git blob
hashes. The repository's full `__init__`, packaging, and complete legacy suite
were not exercised by that partial-checkout run. The existing pull-request CI
is the full-checkout regression gate; inspect its result separately rather
than treating the 49 tests as full-suite evidence.

Upstream blob identities used for the local run:

```text
models.py      f51a139c3722af701e8b9c52720b9170cad47959
jacobian.py    2e98e7f862ba267f4dd1ac159df27bdff30a74ef
covariance.py  2211fc7f1633b23b68515e51b4abd1969b7a6bd1
```
