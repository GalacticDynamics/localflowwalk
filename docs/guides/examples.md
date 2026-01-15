# Examples Gallery

This page showcases various use cases of the kNN+p algorithm.

For a complete interactive demonstration, see the [Jupyter notebook](https://github.com/nstarman/nearest-neighbours-with-momentum/blob/main/examples/demo.ipynb) in the repository.

## Example 1: Simple Linear Stream

The simplest case: points along a straight line with aligned velocities.

```python
import jax.numpy as jnp
from knnp import nearest_neighbors_with_momentum

# Create a linear stream
n_points = 10
t = jnp.linspace(0, 5, n_points)

position = {
    "x": t,
    "y": 0.5 * t,  # Slope of 0.5
}

velocity = {
    "x": jnp.ones(n_points),
    "y": 0.5 * jnp.ones(n_points),
}

result = nearest_neighbors_with_momentum(position, velocity, start_idx=0, lam=1.0)

print(result["ordered_indices"])
# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
```

**Key point**: With aligned velocities, the algorithm naturally follows the stream from start to end.

## Example 2: Curved Stream (Arc)

A more challenging case: points along a curved arc where velocity is tangent to the curve.

```python
import jax
import jax.numpy as jnp

# Create points along a semi-circular arc
n_points = 20
theta = jnp.linspace(0, jnp.pi, n_points)

# Shuffle to make it harder
key = jax.random.PRNGKey(42)
shuffle_idx = jax.random.permutation(key, n_points)

position = {
    "x": jnp.cos(theta)[shuffle_idx],
    "y": jnp.sin(theta)[shuffle_idx],
}

# Tangent velocity (perpendicular to radial)
velocity = {
    "x": -jnp.sin(theta)[shuffle_idx],
    "y": jnp.cos(theta)[shuffle_idx],
}

# Start from one end (max x)
start_idx = int(jnp.argmax(position["x"]))

result = nearest_neighbors_with_momentum(
    position, velocity, start_idx=start_idx, lam=1.0
)
```

**Key point**: Even with shuffled points, the algorithm reconstructs the original arc by following the velocity field.

## Example 3: Effect of Lambda (λ)

Compare different momentum weights on the same data:

```python
# Same arc data as Example 2

results = {}
for lam in [0.0, 1.0, 5.0]:
    results[lam] = nearest_neighbors_with_momentum(
        position, velocity, start_idx=start_idx, lam=lam
    )

print("λ=0 (spatial only):", results[0.0]["ordered_indices"][:5])
print("λ=1 (balanced):", results[1.0]["ordered_indices"][:5])
print("λ=5 (strong momentum):", results[5.0]["ordered_indices"][:5])
```

**Key point**:
- **λ=0**: May take shortcuts through the interior of the arc
- **λ>0**: Follows the arc more faithfully
- **λ>>1**: Strongly enforces following the velocity direction

## Example 4: Noisy Stream

Real data has noise. The algorithm can handle it:

```python
# Create a noisy stream
n_points = 30
t = jnp.linspace(0, jnp.pi, n_points)

# Add noise
key = jax.random.PRNGKey(7)
keys = jax.random.split(key, 3)
noise_scale = 0.08

position = {
    "x": t * 4 + jax.random.normal(keys[0], (n_points,)) * noise_scale,
    "y": jnp.sin(t) * 1.5 + jax.random.normal(keys[1], (n_points,)) * noise_scale,
}

# Velocity follows the clean curve
velocity = {
    "x": jnp.ones(n_points) * 4,
    "y": jnp.cos(t) * 1.5,
}

# Normalize velocity
v_norm = jnp.sqrt(velocity["x"] ** 2 + velocity["y"] ** 2)
velocity = {k: v / v_norm for k, v in velocity.items()}

# Shuffle
shuffle_idx = jax.random.permutation(keys[2], n_points)
position = {k: v[shuffle_idx] for k, v in position.items()}
velocity = {k: v[shuffle_idx] for k, v in velocity.items()}

start_idx = int(jnp.argmin(position["x"]))
result = nearest_neighbors_with_momentum(
    position, velocity, start_idx=start_idx, lam=5.0
)
```

**Key point**: The momentum term helps "filter" noise by following the velocity field rather than chasing random fluctuations.

## Example 5: Gap Detection with max_dist

Detect and stop at gaps in the data:

```python
# Create two disconnected segments
t1 = jnp.linspace(0, 2, 10)
t2 = jnp.linspace(5, 7, 10)  # Gap from 2 to 5

position = {
    "x": jnp.concat([t1, t2]),
    "y": jnp.concat([jnp.sin(t1), jnp.sin(t2)]),
}

velocity = {
    "x": jnp.ones(20),
    "y": jnp.concat([jnp.cos(t1), jnp.cos(t2)]),
}

# Without max_dist: crosses the gap
result_no_limit = nearest_neighbors_with_momentum(
    position, velocity, start_idx=0, lam=1.0
)
print(f"No limit: {len(result_no_limit['ordered_indices'])} points")

# With max_dist: stops at the gap
result_with_limit = nearest_neighbors_with_momentum(
    position, velocity, start_idx=0, lam=1.0, max_dist=2.0
)
print(f"With max_dist=2.0: {len(result_with_limit['ordered_indices'])} points")
print(f"Skipped: {len(result_with_limit['skipped_indices'])} points")
```

**Key point**: `max_dist` enables automatic segmentation of disconnected stream components.

## Example 6: 3D Helix

The algorithm works in any number of dimensions:

```python
# Create a 3D helix
n_points = 40
t = jnp.linspace(0, 4 * jnp.pi, n_points)

# Shuffle
key = jax.random.PRNGKey(456)
shuffle_idx = jax.random.permutation(key, n_points)

position = {
    "x": jnp.cos(t)[shuffle_idx],
    "y": jnp.sin(t)[shuffle_idx],
    "z": (t / (4 * jnp.pi))[shuffle_idx],
}

# Tangent velocity
velocity = {
    "x": -jnp.sin(t)[shuffle_idx],
    "y": jnp.cos(t)[shuffle_idx],
    "z": (jnp.ones(n_points) / (4 * jnp.pi))[shuffle_idx],
}

# Start from the bottom
start_idx = int(jnp.argmin(position["z"]))

result = nearest_neighbors_with_momentum(
    position, velocity, start_idx=start_idx, lam=3.0
)
```

**Key point**: The algorithm generalizes naturally to any dimension. Just provide coordinate components as dictionary keys.

## Example 7: Limiting the Number of Points

Use `n_max` to limit the search:

```python
# Only find the first 5 points
result = nearest_neighbors_with_momentum(
    position,
    velocity,
    start_idx=0,
    lam=1.0,
    n_max=5,
)

print(len(result["ordered_indices"]))  # 5
print(len(result["skipped_indices"]))  # N - 5
```

**Key point**: Useful for:
- Previewing results quickly
- Finding local neighborhoods
- Reducing computation for very large datasets

## Example 8: Using Terminate Indices

Stop when reaching specific points:

```python
# Stop if we reach indices 7 or 8
result = nearest_neighbors_with_momentum(
    position,
    velocity,
    start_idx=0,
    lam=1.0,
    terminate_indices={7, 8},
)

# Will stop when first encountering index 7 or 8
last_idx = result["ordered_indices"][-1]
assert last_idx in {7, 8}
```

**Key point**: Useful for:
- Connecting streams between known endpoints
- Avoiding problematic regions
- Implementing bidirectional searches

## Example 9: Accessing Low-Level Functions

For custom distance metrics or analysis:

```python
from knnp.phasespace import (
    euclidean_distance,
    unit_direction,
    unit_velocity,
    cosine_similarity,
)

# Current point
current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
current_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

# Candidate point
candidate_pos = {"x": jnp.array(1.0), "y": jnp.array(1.0)}

# Compute components
d0 = euclidean_distance(current_pos, candidate_pos)
u_dir = unit_direction(current_pos, candidate_pos)
u_vel = unit_velocity(current_vel)
cos_sim = cosine_similarity(u_dir, u_vel)

# Manual distance calculation
lam = 1.0
total_distance = d0 + lam * (1 - cos_sim)

print(f"Distance: {float(d0):.2f}")
print(f"Cosine similarity: {float(cos_sim):.2f}")
print(f"Total (with λ={lam}): {float(total_distance):.2f}")
```

**Key point**: All distance/direction computations are exposed as standalone functions for debugging and analysis.

## Example 10: JAX Transformations

Use with JIT, vmap, and grad:

```python
from jax import jit, vmap, grad


# JIT compilation
@jit
def fast_order(pos, vel):
    return nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)


result = fast_order(position, velocity)

# Note: vmap over dict structures requires custom handling
# See the JAX Integration guide for details
```

**Key point**: The algorithm is fully compatible with JAX transformations for maximum performance.

## Next Steps

- See the [demo notebook](https://github.com/nstarman/nearest-neighbours-with-momentum/blob/main/examples/demo.ipynb) for interactive visualizations
- Read the [Algorithm Details](algorithm.md) for mathematical background
- Check the [API Reference](../api/index.md) for all available functions
