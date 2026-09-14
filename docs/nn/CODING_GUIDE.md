# `nn/` Coding Guide

Scope: conventions for everything under `src/sdmlab/nn/` (layers, activations, losses,
containers) — specifically how `forward`/`backward` must be written so autograd works correctly.
This doc will grow, and similar guides will be added for other modules (`optim/`, `autograd/`,
...) once those are actively being built.

## 1. Two different levels of "backward" — don't confuse them

There are two different places gradients get written in this codebase, and they are NOT the same
job:

- **`autograd/ops.py`** — the primitive differentiable operations (`Add`, `Mul`, `MatMul`, `Sum`,
  `Exp`, ...). Each one is a `Function` subclass that manually defines both `forward()` *and*
  `backward()` — this is the only place a derivative is ever hand-derived and hand-written.
- **`nn/` (layers, activations, losses)** — these are built entirely out of the primitives above.
  Because every primitive already knows its own `backward()`, the graph that autograd builds while
  `forward()` runs already knows how to backpropagate through it automatically.

**Rule: nothing under `nn/` ever defines a `backward()` method.** If you find yourself wanting to
write one inside `nn/layers/*.py`, `nn/activations/*.py`, or `nn/losses/*.py`, that's a sign the
operation you need doesn't exist yet as a primitive and belongs in `autograd/ops.py` instead —
build it there once, then use it from as many `nn/` classes as you want, for free, forever.

## 2. Required shape for every `nn/` class (Layer / Activation / Loss)

```python
class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(...)   # shape: (in_features, out_features)
        self.bias = Parameter(...)     # shape: (out_features,)

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor of shape (batch, in_features)
        return x @ self.weight + self.bias
```

- Only `forward()` is implemented. Never `backward()`.
- Signature is always `forward(self, *inputs: Tensor) -> Tensor` (or a tuple of `Tensor`s for a
  multi-output layer). Inputs and outputs are always `Tensor`, never a raw NumPy/CuPy array —
  that's what keeps the op differentiable and backend-agnostic.
- Callers never call `.forward(x)` directly — always call the instance itself, `layer(x)`, which
  `Module.__call__` routes to `forward()`. This is what leaves room for hooks (e.g. train/eval
  mode switches) later without changing call sites.
- `forward()` must be pure with respect to parameters and inputs — no mutating `self` state beyond
  documented mode flags (e.g. a dropout layer's train/eval flag). The same inputs must always
  produce the same outputs and the same graph.
- Every learnable tensor is assigned as `self.<name> = Parameter(...)` inside `__init__`, never
  created ad hoc inside `forward()`. This is what makes `Module.parameters()` discovery work,
  which `optim/` depends on.
- Stateless classes (activations, losses with no learnable weights) follow the exact same
  contract — statelessness doesn't change anything about the forward-only rule.

## 3. Required shape for a primitive op, for context (lives in `autograd/`, not `nn/`)

This is what makes rule 1 possible — shown here so it's clear what "the primitive already knows
its own backward" actually means:

```python
class MatMul(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return Tensor(backend.matmul(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b
```

- `backward()` returns exactly one gradient per positional argument `forward()` took, in the same
  order (`None` for any input that doesn't require grad).
- `forward()` computes using `backend` ops only — never call another `Function`'s `.apply()` from
  inside a `forward()`/`backward()` pair, or graph tracking breaks.

## 4. File & naming conventions inside `nn/`

- One class per file; file name is the snake_case of the class name (`Dense` → `dense.py`, `ReLU`
  → `relu.py`).
- Each subpackage's `__init__.py` re-exports its public classes flatly
  (`from .linear import Dense`) so `sdmlab.nn.Dense` / `from sdmlab.layers import Dense` both work
  without deep imports.

## 5. Testing convention

Every new `forward()` ships with a numeric gradient check (finite-difference comparison against
autograd's computed gradient) in the mirrored `tests/nn/...` file before it's considered done.
This is how a wrong `backward()` in an underlying primitive op gets caught — `nn/` itself has no
`backward()` to test directly, so the check has to run through the whole graph.

## 6. Shape comments

Document the expected `Tensor` shape for every `forward()` argument and return value as an inline
comment (see the `Dense` example above) — shape mismatches are the most common bug in this kind of
code, and they're silent until something downstream crashes or trains wrong.
