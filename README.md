# SDMLab

**S**rinivas **D**eekonda **M**achine Learning **Lab** — a from-scratch machine learning /
neural-network library, built incrementally as a long-term learning and engineering project.

## What this is

SDMLab implements the core building blocks of ML/deep-learning frameworks — tensors, autograd,
layers, optimizers, and eventually classic ML algorithms — from first principles in Python,
instead of wrapping PyTorch, TensorFlow, or JAX. The goal is to understand how these frameworks
actually work internally by building them, one concept at a time, rather than to compete with any
of them.

Priority order: understanding and correctness first, API convenience second.

## Philosophy

Every component follows the same loop before it's considered done:

**learn → implement → test → document → release**

Development happens in small, regular iterations. Each concept — a layer, an activation, an
optimizer — gets implemented, tested, documented, and shipped as its own version bump rather than
bundled into one large release. See [Versioning](#versioning) below for how that's tracked.

## Current scope: neural networks, from scratch

- Tensors with automatic differentiation — computational graphs, backpropagation
- Dense/Linear layers, activation functions, loss functions
- Optimizers: SGD, Momentum, AdaGrad, RMSProp, Adam, learning-rate scheduling, weight decay
- Batch / mini-batch training, a `Sequential` model abstraction, training loops, metrics
- Dataset utilities
- A backend abstraction over NumPy (CPU) today, CuPy (GPU) later, switched with
  `sdmlab.set_backend("numpy" | "cupy")`

Classic ML (linear/logistic regression, SVMs, decision trees, clustering, ...) and other paradigms
are explicitly planned for later — the architecture is designed so they slot in as new modules
without requiring a rewrite of anything above. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full folder structure and the reasoning
behind it, and [docs/nn/CODING_GUIDE.md](docs/nn/CODING_GUIDE.md) for the coding conventions used
while building the neural-network module.

## Example (target API)

```python
from sdmlab.layers import Dense
from sdmlab.activations import ReLU
from sdmlab.losses import CrossEntropy
from sdmlab.optimizers import Adam

model = sdmlab.Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
])
optimizer = Adam(learning_rate=0.001)
```

This is the API being built toward — check the version history below for what's actually
implemented at any given point.

## Installation

```bash
pip install sdmlab
```

The package name is registered on PyPI; the library itself is early-stage and under active
development, so check the installed version against [Versioning](#versioning) before relying on
any particular API surface.

## Versioning

SDMLab follows [semantic versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), tracked in
[CHANGELOG.md](CHANGELOG.md) and `sdmlab.__version__`. A `MINOR` bump generally corresponds to one
new concept landing (a layer, an optimizer, ...); a `MAJOR` bump is reserved for a breaking change
to the public API once the library is stable enough for that to matter.

To depend on a specific release, pin it the normal `pip` way:

```bash
pip install sdmlab==0.1.0
```

## Project layout

```
src/sdmlab/       the package
tests/            mirrors src/sdmlab/ 1:1
examples/         one runnable script per milestone
docs/             architecture notes and per-module coding guides
```

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

[GNU General Public License v3.0](LICENSE)

## Author

Srinivas Deekonda
