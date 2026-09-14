# API Versioning

SDMLab has two independent versioning concepts. Don't conflate them:

1. **Release version** (`0.1.0`, `0.2.0`, ...) — every published PyPI release, tracked in
   `pyproject.toml`, `CHANGELOG.md`, and `sdmlab.__version__`. This bumps constantly — roughly
   one bump per concept shipped (see the table in [ARCHITECTURE.md](ARCHITECTURE.md)).
2. **API namespace version** (`sdmlab.v1`, `sdmlab.v2`, ...) — a much coarser, much rarer thing.
   It's a promise: "code written against `sdmlab.v1` keeps working, forever, no matter how many
   release versions ship after it." A new namespace (`v2`) is only created when you deliberately
   decide to break that promise on purpose.

This document is about #2.

## What a consumer actually gets

```python
import sdmlab
sdmlab.Sequential(...)              # follows whatever version is CURRENT — your call, not theirs

import sdmlab.v1
sdmlab.v1.layers.Dense(...)         # pinned to v1's API, forever, even after v2 exists

from sdmlab.layers import Dense     # same as the bare `import sdmlab` case: follows CURRENT
```

Nobody has to type `v1` to get a working import. It only matters to someone who deliberately wants
to freeze against a specific API shape.

## The mechanism

```
src/sdmlab/
├── __init__.py          # __version__ = "0.1.0"; from sdmlab.v1 import *   <- the ONE line
│                         #   that decides what "CURRENT" means. You edit this by hand when
│                         #   you decide to move it, never automatically.
├── layers.py             # from sdmlab.v1.layers import *
├── activations.py         # from sdmlab.v1.activations import *
├── losses.py               # from sdmlab.v1.losses import *
├── optimizers.py            # from sdmlab.v1.optimizers import *
│
├── v1/                        # the v1 API snapshot
│   ├── __init__.py             # from .layers import *; from .activations import *; ...
│   ├── layers.py                # from sdmlab.nn.layers import *
│   ├── activations.py            # from sdmlab.nn.activations import *
│   ├── losses.py                  # from sdmlab.nn.losses import *
│   └── optimizers.py               # from sdmlab.optim import *
│
├── nn/  optim/  training/  datasets/  metrics/  utils/    # the live implementation —
│                                                            #   evolves continuously, unversioned
└── backend/  tensor/  autograd/  params/                    # the engine — never versioned at all
```

Four tiers, each answering a different question:

- **`backend/` `tensor/` `autograd/` `params/`** — the engine. Internal implementation detail,
  never part of the public-facing version promise, never duplicated across versions.
- **`nn/` `optim/` `training/` `datasets/` `metrics/` `utils/`** — the live implementation. This is
  where you actually write and evolve code day to day. It has no version number of its own; it's
  just "the current source."
- **`v1/` (and later `v2/`, ...)** — a namespace of pure re-exports. On the day `v1` is created, and
  for as long as nothing under it needs to diverge, every file in `v1/` is a one-line forward to
  the live implementation. Zero duplicated logic.
- **Root shims + `sdmlab/__init__.py`** — the single point of indirection that decides which
  version namespace bare `import sdmlab` resolves to.

## The rule that makes this actually work

**A version namespace is a promise, not a folder you casually edit.** As long as changes to
`nn/`/`optim/`/etc. are backward compatible with what `v1` has always re-exported, nothing about
`v1/` needs to change — it keeps forwarding to live code for free. That's the common case, and it's
why this costs nothing today: `v1/` is 100% pass-through right now because there's nothing yet to
be incompatible with.

The moment a change to the live implementation would actually break something `v1` already
exports (a renamed class, a changed constructor signature, a different return type):

1. **Before** making the breaking change, copy the *specific* piece being broken into
   `v1/_frozen/<name>.py` — just that one class/function, exactly as it behaved under v1.
2. Repoint `v1/layers.py` (or whichever file) to import that one symbol from `v1/_frozen/` instead
   of from the live module. Everything else in `v1/` keeps forwarding to live code as before.
3. Now make the breaking change in `nn/`/`optim`/etc. freely.

This means **only the pieces that actually changed incompatibly ever get frozen** — never the
whole library, and never speculatively. `v1/_frozen/` doesn't exist until the first breaking change
happens to demand it.

## When to cut a `v2/`

Not on every breaking change — that's what step-by-step freezing inside `v1/` already handles.
Cut a brand new `v2/` namespace only when you deliberately decide the API has accumulated enough
intentional change to call it a new major version (this is the same moment a `MAJOR` semver bump
happens, per [README.md](../README.md#versioning)). At that point:

1. Create `src/sdmlab/v2/`, structured the same way `v1/` is, re-exporting the *new* live behavior
   directly (no freezing needed yet — `v2` starts as pure pass-through too).
2. `v1/` is untouched and keeps working exactly as it always has, `_frozen/` pieces included.
3. When you're ready to move everyone's default import forward, change the one line in
   `sdmlab/__init__.py` (and the root shims) from `sdmlab.v1` to `sdmlab.v2`. That's the entire
   "cutover" — existing code doing `import sdmlab.v1` explicitly is completely unaffected.

## Right now

Only `v1/` exists, and every file in it is a plain re-export of the live implementation — there is
nothing frozen yet because nothing has shipped to be incompatible with. Don't add a `v1/_frozen/`
directory speculatively; add it the day a real breaking change actually needs it.
