---
sd_hide_title: true
---

<h1> <code> knnp </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quickstart
guides/algorithm
guides/autoencoder
guides/examples
guides/api-design
guides/jax-integration
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🔌 API Reference

api/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: More

contributing
```

# 🚀 Get Started

**knnp** is a Python library implementing the **Nearest Neighbors with Momentum** (kNN+p) algorithm from [Nibauer et al. (2022)](https://arxiv.org/abs/2201.12042) for ordering phase-space observations in stellar streams.

The algorithm combines:
- **Spatial proximity**: Finding nearby points in position space
- **Velocity momentum**: Preferring points that lie in the direction of the current velocity

This is particularly useful for tracing stellar streams where stars follow coherent trajectories through phase-space.

---

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install knnp
```

:::

:::{tab-item} uv

```bash
uv add knnp
```

:::

::::

## Quick Example

```python
import jax.numpy as jnp
from knnp import nearest_neighbors_with_momentum

# Define phase-space data as dictionaries
position = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5]),
}
velocity = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5]),
}

# Run the algorithm
result = nearest_neighbors_with_momentum(
    position, velocity, start_idx=0, lam=1.0  # momentum weight
)

print(result["ordered_indices"])
# (0, 1, 2, 3)
```

## Features

- ✅ **JAX-native**: Full support for JIT compilation, vectorization, and auto-differentiation
- ✅ **High performance**: Optimized with `jax.lax.while_loop` for speed
- ✅ **Gap filling**: Autoencoder neural network interpolates skipped tracers
- ✅ **Flexible**: Works in any number of dimensions
- ✅ **Type-safe**: Full type annotations with `jaxtyping`
- ✅ **Well-tested**: Comprehensive test suite with property-based testing

## Algorithm

The kNN+p algorithm uses the following distance metric:

$$d = d_0 + \lambda \cdot (1 - \cos\theta)$$

where:
- $d_0$ is the Euclidean distance in position space
- $\cos\theta$ is the cosine similarity between the velocity vector and the direction to the candidate point
- $\lambda$ controls the weight of the momentum term

When $\cos\theta = 1$ (candidate is in the velocity direction), the momentum term is 0.
When $\cos\theta = -1$ (candidate is opposite to velocity), the momentum term is $2\lambda$.

## Key Parameters

- **`lam` (λ)**: Momentum weight. Higher values favor points aligned with velocity.
  - `lam=0`: Pure nearest neighbor (spatial only)
  - `lam>0`: Balances spatial proximity and momentum alignment

- **`max_dist`**: Maximum distance to next point. Stops at gaps in the data.

- **`terminate_indices`**: Stop when reaching specific points.

- **`n_max`**: Maximum number of points to include.

## Data Format

Phase-space data uses **raw Python dictionaries** for maximum performance and JAX compatibility:

```python
# Position dictionary: coordinate names → arrays
position = {"x": array, "y": array, "z": array}

# Velocity dictionary: same keys → velocity components
velocity = {"x": array, "y": array, "z": array}
```

This dict-based API is designed for:
- Efficient JAX tree operations via `jax.tree.map`
- Seamless integration with JAX transformations (`jit`, `vmap`, `grad`)
- Minimal overhead in hot loops

## Citation

If you use knnp in your research, please cite the original paper:

```bibtex
@article{nibauer2022charting,
  title={Charting Galactic Accelerations with Stellar Streams and Machine Learning},
  author={Nibauer, Jacob and others},
  journal={arXiv preprint arXiv:2201.12042},
  year={2022}
}
```

## Next Steps

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {material-regular}`rocket_launch;2em` Quickstart
:link: guides/quickstart
:link-type: doc

Get started with knnp in 5 minutes
:::

:::{grid-item-card} {material-regular}`science;2em` Algorithm Details
:link: guides/algorithm
:link-type: doc

Understand the math and implementation
:::

:::{grid-item-card} {material-regular}`psychology;2em` Autoencoder
:link: guides/autoencoder
:link-type: doc

Fill gaps with neural network interpolation
:::

:::{grid-item-card} {material-regular}`code;2em` API Reference
:link: api/index
:link-type: doc

Full API documentation
:::

::::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
