# Notations Sensitivity

**Measure how local perturbations propagate through composed models and changes of coordinates.**

[Run](#install-and-run) · [API](docs/KERNEL.md) · [Covariance provider](#workbench-covariance-operation) · [Research profile](#research-profile)

## Notation Systems

**Frontier Tooling and Instrumentation for Digital Futures.** We develop computational instruments and operational tooling connecting scientific methods, specialized computation and human expertise.

[Notations Systems Terminal](https://github.com/giasonpooni/Notations-Systems-Terminal) composes supported investigations; this provider retains its derivative and covariance mathematics. Governed evidence and Cartesian Graphics' interactive worlds, simulation technology and digital IP keep separate state and approval. [Organization profile](https://github.com/giasonpooni/Notations-Systems-Terminal/blob/b41b84922d4963a9206202029afd1e78b9451f9c/PUBLIC_POSITIONING.md).

| Identity | Scope |
| --- | --- |
| Current repository | `Notations-Sensitivity-Testbed` |
| Provider / import | Jacobian and Sensitivity Propagation Testbed / JSPT; `sensitivity` |
| Friendly operation-family target | `math.sensitivity` |
| Existing endpoint | `jspt.covariance-propagate.v1` via `sensitivity.ciw_adapter` |
| Boundary | Local first-order derivatives, explicit compositions, coordinate consistency and covariance propagation |

The friendly name does not register a new alias. The covariance endpoint accepts a supplied Jacobian; accepting it is not derivative verification. NET / `net` / `ciw`, imports, contracts and historical pins remain unchanged.

## What the instrument demonstrates

| Operation | Meaning |
| --- | --- |
| Derivative checks | Compare a declared Jacobian with central, forward or complex-step estimates |
| Composition | Compare chain-rule propagation with the composed map |
| Coordinate consistency | Transform both model and perturbation before comparing physical predictions |
| Local validity | Test where `dy ~ J dx` ceases to describe `f(x+dx)-f(x)` |
| Correlated uncertainty | Compare `Sigma_y ~ J Sigma_x J^T` with Monte Carlo under stated assumptions |

For invertible linear charts `x' = T x` and `y' = S y`,

```text
J' = S J T^{-1}
J' dx' = S (J dx)
```

Raw Jacobian entries and unscaled singular values are not invariant under changes of units. The perturbation and output coordinates must transform too. Norm-based sensitivity scores must declare their scaling. Invalid charts are refused rather than silently repaired.

## Install and run

Use Python 3.12/3.13 and NumPy. The supported uv route is:

```sh
git clone https://github.com/giasonpooni/Notations-Sensitivity-Testbed.git
cd Notations-Sensitivity-Testbed
uv run --python 3.13 python examples/quickstart.py
uv run --python 3.13 --with pytest pytest -q
```

A plain virtual environment also works:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e . pytest
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src pytest -q
```

The example writes `results/quickstart.md`.

## Using the library

Applications can use `jacobian_at`, `first_order_covariance` and `push_covariance` without adopting the experiment suite or replacing their domain types. Additional interfaces include:

```python
from sensitivity import (
    compose, jacobian_at, jvp, check_composition,
    check_coordinate_consistency, check_derivative,
    sweep_perturbation_scale,
)
```

[API and numerical constraints](docs/KERNEL.md).

## Workbench covariance operation

```sh
PYTHONPATH=src python -m sensitivity.ciw_adapter < examples/covariance_request.json
```

The endpoint consumes a content-addressed `covariance-artifact.v1` and an explicit ordered Jacobian. It returns full covariance with source, basis, units, frame and linearization-point declarations. Weighted aggregation retains cross-covariance; invertible charts retain conditioning and round-trip checks.

The shared-offset fixture returns mean variance `1.03`: shared variance `1` remains while independent variance `0.09` is divided over three readings. Negative weights may legitimately cancel a shared component. The endpoint reuses the existing kernels; it neither differentiates the supplied matrix nor runs a nonlinear Monte Carlo check.

NET can bind a clean source revision as a pinned subprocess. Evidence, specifications, executions, results and verification remain distinct. See [transport, refusal meanings and validation evidence](docs/CIW_ADAPTER.md).

## Research profile

**Question:** which predictions survive composition and representation changes, and how far can local approximation be trusted?

Use derivative references, chart changes, perturbation sweeps and correlated covariance as bounded specimens. A locally zero derivative does not prove global irrelevance. A successful ablation on one input is not a theorem of minimal representation. Compare physical increments and task outcomes, not unscaled matrix entries.

Measure error and refusal behavior before runtime or context savings. Independent implementations and shared-kernel language bindings provide different evidence. Python/Julia/Rust/C++ and CUDA providers require separate implementation and qualification. [Shared research protocol](https://github.com/giasonpooni/Notations-Systems-Terminal/blob/b41b84922d4963a9206202029afd1e78b9451f9c/RESEARCH_PROGRAMME.md).

## Scope and limits

First-order, local, explicit compositions only. Global sensitivity, discontinuous mode changes and trajectory sensitivities are unsupported. Covariance propagation is exact for affine maps under its stated model; for nonlinear maps, the experiment reports the Monte Carlo discrepancy rather than assuming adequacy.

[Scope](docs/SCOPE.md) · [Methods](docs/METHODS.md) · [Stack role](docs/STACK_ROLE.md)

Former repository names remain compatibility context. This documentation changes no numerical source, tests, dependencies, licence, permissions or release state. No new runtime or GPU qualification is claimed.

## License

[MIT](LICENSE). Existing contributor and third-party notices remain in force. Organization positioning does not transfer rights or establish nonprofit status.
