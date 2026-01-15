# Project Overview

This repository provides `knnp`, an implementation of the Nearest Neighbors with
Momentum (kNN+p) algorithm from Nibauer et al. (2022) for ordering phase-space
observations in stellar streams.

- **Language**: Python 3.12+
- **Main API**: `nearest_neighbors_with_momentum` function for ordering
  phase-space data
- **Design goals**: Maximum performance with JAX, dict-based phase-space
  representation for flexibility and JAX tree compatibility

## Coding Style

### Module Structure: `__all__` Before Imports

**CRITICAL**: In all Python modules, `__all__` must be defined **before** any
imports (except `__future__` imports). This makes the public API immediately
visible at the top of every file.

```python
"""Module docstring."""

__all__: tuple[str, ...] = (
    "PublicClass",
    "public_function",
)

from collections.abc import Mapping

# ... rest of imports
```

**Why**: Seeing the public API first helps readers understand what a module
exports without scrolling through imports.

### Other Conventions

- Always use type hints
- `__all__` should be a tuple (not list) for immutability
- **NEVER use `from __future__ import annotations`** - causes issues with plum
  dispatch and runtime type introspection
- Use `jax.tree.map` and `jax.tree.leaves` for operations on component dicts
- Phase-space data is two dicts: `position: dict[str, Array]` and
  `velocity: dict[str, Array]` with matching keys
- Follow phase-space notation: `q` for position, `p` for momentum/velocity, `w`
  for full phase-space point

## Tooling

- This repo uses `uv` for dependency and environment management
- Run tests with: `uv run pytest tests/ -v`

## Testing

- Use `pytest` for all test suites
- Add unit tests for every new function
- Test JAX compatibility (`jit`, `vmap`, `grad`)

## Final Notes

Preserve JAX compatibility. Functions should work with `jax.jit` and `jax.vmap`.
Use dict-based APIs for maximum flexibility with different coordinate systems.
