# SDMLab Architecture

This document describes how the `sdmlab` package is organized, why it's organized that way, and
how it's meant to grow — from a neural-network-only library today into something that can also
host classic ML and other paradigms later without restructuring existing code.

For how the public API is versioned (the `sdmlab.v1` namespace, and what happens when a breaking
change needs a `v2`), see [VERSIONING.md](VERSIONING.md).

## Folder Structure

```
sdmlab/                                     # repo root (github.com/me-lier/sdmlab)
├── pyproject.toml                          # src-layout; numpy required, cupy under "gpu" extra
├── README.md
├── LICENSE
├── CHANGELOG.md                            # one entry per version bump (v0.1.0, v0.2.0, ...)
├── .gitignore
├── .github/workflows/ci.yml                # optional: pytest on push — add whenever ready
│
├── src/
│   └── sdmlab/
│       ├── __init__.py                     # __version__; `from sdmlab.v1 import *` (current-
│       │                                   #   version pointer, see VERSIONING.md); also
│       │                                   #   re-exports Tensor, set_backend directly (unversioned)
│       ├── py.typed                        # PEP 561 marker
│       │
│       │                                   # ---- flat-import shims (matches the example API) ----
│       │                                   #   forward through the CURRENT version namespace,
│       │                                   #   not straight at nn/optim — see VERSIONING.md
│       ├── layers.py                       # `from sdmlab.v1.layers import *`      -> v0.1.0
│       ├── activations.py                  # `from sdmlab.v1.activations import *` -> v0.2.0
│       ├── losses.py                       # `from sdmlab.v1.losses import *`      -> v0.3.0
│       ├── optimizers.py                   # `from sdmlab.v1.optimizers import *`  -> v0.4.0
│       │
│       ├── v1/                             # === the "v1" public API snapshot ===
│       │   ├── __init__.py                 # re-exports layers/activations/losses/optimizers
│       │   │                               #   + Sequential — this is what "v1" means, concretely
│       │   ├── layers.py                   # -> sdmlab.nn.layers (until v1 needs a frozen override)
│       │   ├── activations.py              # -> sdmlab.nn.activations
│       │   ├── losses.py                   # -> sdmlab.nn.losses
│       │   └── optimizers.py               # -> sdmlab.optim
│       │                                   #   (a future v2/ is a sibling of this, created only
│       │                                   #    on an intentional breaking change — VERSIONING.md)
│       │
│       ├── backend/                        # === FOUNDATION 1 — array-op abstraction ===
│       │   ├── __init__.py                 # set_backend("numpy"|"cupy"), get_backend()
│       │   ├── base.py                     # Backend protocol every backend must satisfy
│       │   ├── numpy_backend.py            # v0.1.0 — default backend
│       │   └── cupy_backend.py             # # future — same protocol, GPU array ops
│       │
│       ├── tensor/                         # === FOUNDATION 2 — the Tensor object ===
│       │   ├── __init__.py                 # exports Tensor
│       │   ├── tensor.py                   # shape/dtype, requires_grad/grad, .numpy(), dunders
│       │   ├── dtype.py                    # dtype registry (float32, float64, int64, bool, ...)
│       │   └── creation.py                 # zeros, ones, randn, arange, eye, from_numpy
│       │
│       ├── autograd/                       # === FOUNDATION 3 — comp. graph + backprop ===
│       │   ├── __init__.py                 # backward(), no_grad(), enable_grad()
│       │   ├── function.py                 # Function base: forward()/backward()/.apply()
│       │   ├── ops.py                      # Add, Mul, MatMul, Sum, Exp, Log, ... as Functions
│       │   ├── graph.py                    # Node/edge bookkeeping, topological backward traversal
│       │   └── grad_mode.py                # no_grad()/enable_grad() context managers
│       │
│       ├── params/                         # === FOUNDATION 4 — paradigm-neutral trainability ===
│       │   ├── __init__.py                 # exports Parameter, Module
│       │   ├── parameter.py                # Parameter(Tensor) — auto-registering learnable leaf
│       │   └── module.py                   # Module: .parameters(), zero_grad(), train()/eval()
│       │                                   #   -> the ONLY contract optim/ depends on
│       │
│       ├── nn/                             # === CONSUMER — neural-network building blocks ===
│       │   ├── __init__.py                 # flat re-exports: Linear, ReLU, Sigmoid, Softmax, ...
│       │   ├── init.py                     # xavier_uniform_, kaiming_uniform_, ...
│       │   ├── containers.py               # Sequential                            -> v0.8.0
│       │   ├── layers/
│       │   │   ├── base.py                 # Layer(Module): forward()/__call__ contract
│       │   │   └── linear.py               # Linear / Dense                        -> v0.1.0
│       │   ├── activations/
│       │   │   ├── relu.py
│       │   │   ├── sigmoid.py
│       │   │   ├── tanh.py
│       │   │   └── softmax.py                                                     -> v0.2.0
│       │   └── losses/
│       │       ├── mse.py
│       │       └── cross_entropy.py                                               -> v0.3.0
│       │
│       ├── optim/                          # === CONSUMER — generic optimizers, not nn-specific ===
│       │   ├── optimizer.py                # base: param_groups, step(), zero_grad(), weight_decay
│       │   ├── sgd.py               # + Momentum                                  -> v0.4.0
│       │   ├── adagrad.py                                                         -> v0.5.0
│       │   ├── rmsprop.py                                                         -> v0.6.0
│       │   ├── adam.py                                                            -> v0.7.0
│       │   └── lr_scheduler.py      # StepLR, ExponentialLR, ...                  -> v0.8.0
│       │
│       ├── training/                       # === generic fit-loop, paradigm-agnostic ===
│       │   └── trainer.py                  # batches, forward/backward/step, epochs -> v0.8.0
│       │
│       ├── datasets/                       # dataset.py, dataloader.py, transforms.py -> v0.8.0
│       ├── metrics/                        # classification.py, regression.py         -> v0.8.0
│       ├── utils/                          # seed.py, serialization.py
│       │
│       ├── classic_ml/                     # # FUTURE — not created yet, see "Extending Later"
│       │   ├── base.py                     # Estimator: fit()/predict()/transform()
│       │   ├── linear_model/               # LinearRegression, LogisticRegression
│       │   ├── svm/
│       │   ├── tree/                       # decision trees, ensembles
│       │   ├── cluster/                    # k-means
│       │   └── decomposition/              # PCA
│       │
│       └── <paradigm>/                     # # FUTURE — e.g. rl/, probabilistic/
│
├── tests/                                  # mirrors src/sdmlab/ 1:1
├── examples/                               # one runnable script per milestone
└── docs/
    ├── ARCHITECTURE.md                     # this file
    ├── VERSIONING.md                       # the v1/v2 API-namespace mechanism, in full
    └── notes/                              # one write-up per concept as it's built
        ├── tensor.md
        ├── autograd.md
        ├── backend-abstraction.md
        └── optimizers.md
```

## Design Principles

**One-way dependency rule.** `backend → tensor → autograd → params`, in that order, and nothing in
that chain ever imports `nn`, `optim`, `training`, `datasets`, `metrics`, `classic_ml`, or any
future paradigm package. Each layer adds exactly one idea:

1. `backend` — how to run an array op on the active device.
2. `tensor` — how to hold an array plus enough bookkeeping to know if it needs a gradient.
3. `autograd` — how to record which ops produced a tensor, and replay that in reverse.
4. `params` — how to mark a tensor as learnable, and how a collection of such tensors gets
   discovered by something that wants to update them.

None of these four packages knows what a "layer," "activation," "loss," or "estimator" is. `nn/`
is the first *consumer* that composes `params.Module` + `autograd` ops into "a network built from
stacked `forward()` calls." A future `classic_ml` estimator is a second, structurally different
consumer of the same four packages — composing them into "an object with `fit()`/`predict()`"
instead — optionally reusing concrete `nn` building blocks (e.g. `nn.Linear` + `nn.Sigmoid` for
logistic regression) or skipping `autograd` entirely (k-means, decision trees, PCA via SVD need
only `tensor`/`backend`).

**`params/` is paradigm-neutral, not part of `nn/`.** A `Dense` layer's weight matrix and a future
`LogisticRegression`'s weight vector are both just `Parameter` objects wrapping `Tensor`s. `optim/`
depends only on "exposes `.parameters()`" — never on `nn` — so any future paradigm gets working
optimizers for free just by subclassing `Module`.

**Backend abstraction.** `backend/base.py` defines the array-op surface (`matmul`, `add`, `exp`,
`sum`, `random.normal`, dtype casts, ...) as a `Backend` protocol. `numpy_backend.py` implements it
with NumPy; `cupy_backend.py` implements the identical protocol with CuPy. `backend/__init__.py`
holds one "active backend" reference, swapped by `sdmlab.set_backend("numpy" | "cupy")`. Every
`Tensor` method and every `autograd` op calls `backend.get_backend().op(...)` instead of
`numpy.op(...)` directly, so `tensor/`, `autograd/`, `params/`, `nn/`, `optim/`, `training/` are
written once against the protocol and never touched again when a second backend is added.

**Root import shims.** `layers.py`, `activations.py`, `losses.py`, `optimizers.py` at the package
root are one-line re-exports so calls like `from sdmlab.layers import Dense` work directly, while
the real code stays organized under `nn/`/`optim/`. They forward through `sdmlab.v1` rather than
straight at `nn`/`optim` — see [VERSIONING.md](VERSIONING.md) for why that indirection exists and
what it costs (nothing, until a version needs to diverge).

## Roadmap → File → Version

| Item | Location | Version |
|---|---|---|
| Autograd engine / backprop | `autograd/graph.py`, `autograd/function.py` | v0.1.0 |
| CPU backend | `backend/numpy_backend.py` | v0.1.0 |
| Parameter / Module | `params/` | v0.1.0 |
| Dense / Linear layer | `nn/layers/linear.py` | v0.1.0 |
| Activations (ReLU, Sigmoid, Tanh, Softmax) | `nn/activations/*.py` | v0.2.0 |
| Losses (MSE, CrossEntropy) | `nn/losses/*.py` | v0.3.0 |
| SGD (+ Momentum) | `optim/sgd.py` | v0.4.0 |
| AdaGrad | `optim/adagrad.py` | v0.5.0 |
| RMSProp | `optim/rmsprop.py` | v0.6.0 |
| Adam | `optim/adam.py` | v0.7.0 |
| Weight decay | `optim/optimizer.py` (param_group option) | v0.4.0+, formalized v0.8.0 |
| LR scheduling | `optim/lr_scheduler.py` | v0.8.0 |
| Batch / mini-batch training | `training/trainer.py` + `datasets/dataloader.py` | v0.8.0 |
| Model abstractions (Sequential) | `nn/containers.py` | v0.8.0 |
| Training loops | `training/trainer.py` | v0.8.0 |
| Metrics | `metrics/classification.py`, `metrics/regression.py` | v0.8.0 |
| Dataset utilities | `datasets/dataset.py`, `datasets/transforms.py` | v0.8.0 |
| GPU / CuPy backend | `backend/cupy_backend.py` | first post-v0.8.0 milestone |

## Extending Later

**Rule of thumb:** promote something into `backend/`, `tensor/`, `autograd/`, or `params/` only
when two or more paradigms would otherwise duplicate it. Don't pre-build it speculatively.

**Adding classic ML** (linear/logistic regression, SVMs, decision trees, clustering, ...):
1. New top-level package `src/sdmlab/classic_ml/` (plus `tests/classic_ml/`, `examples/classic_ml/`)
   — nothing existing moves.
2. `classic_ml/base.py` defines an `Estimator`/`RegressorMixin`/`ClassifierMixin` contract with
   `fit(X, y)` / `predict(X)`.
3. Gradient-based estimators (e.g. `LogisticRegression`) subclass `params.Module` directly, compose
   `nn.Linear` + `nn.Sigmoid` + a loss from `nn/losses/`, and train with an unmodified
   `optim.Adam` — because `optim/` never required `nn` in the first place. Non-gradient estimators
   (`tree/decision_tree.py`, `cluster/kmeans.py`, `decomposition/pca.py`) skip `params`/`autograd`/
   `nn` entirely and call `tensor`/`backend` ops directly.
4. Zero files under `backend/`, `tensor/`, `autograd/`, `params/`, `nn/`, or `optim/` change.

**Adding an unnamed future paradigm** (reinforcement learning, probabilistic models, ...):
1. New top-level package `src/sdmlab/<paradigm>/`, free to organize itself however it needs
   (e.g. `agents/`, `environments/`, `policies/` for RL).
2. It depends downward on `backend`/`tensor`/`autograd`/`params`/`optim` wherever useful, and may
   optionally reuse `nn` building blocks — but `nn` and `classic_ml` never depend on it.
3. If a second paradigm later wants the same new abstraction (e.g. both a future `rl/` and
   `probabilistic/` want a generic `Distribution`), that's the trigger to promote it into the
   foundation as a new sibling package — not before.

## Packaging

- **`pyproject.toml`** — `src/`-layout, package name `sdmlab`, dynamic version sourced from
  `sdmlab.__version__`. Hard dependency: `numpy`. Optional extras: `gpu = ["cupy"]`,
  `dev = ["pytest", "ruff", "mypy"]`.
- **`tests/`** — mirrors `src/sdmlab/` 1:1 so it's always obvious which test file exercises which
  milestone's folder.
- **`examples/`** — one runnable script per milestone.
- **`docs/notes/`** — one write-up per concept (`tensor.md`, `autograd.md`,
  `backend-abstraction.md`, `optimizers.md`, ...), giving the "document" step of the
  learn → implement → test → document → release cycle a concrete home as each version ships.
- **`CHANGELOG.md`** — one entry per version bump.
