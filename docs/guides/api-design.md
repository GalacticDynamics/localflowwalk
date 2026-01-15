# API Design Philosophy

This page explains the design choices behind knnp's API, particularly the use of **raw dictionaries** for phase-space data.

## Dictionary-Based API

### Core Data Structures

knnp uses simple Python dictionaries for phase-space data:

% skip: next

```python
# Position: coordinate names → JAX arrays
position: dict[str, Array] = {
    "x": jnp.array([...]),
    "y": jnp.array([...]),
    "z": jnp.array([...]),
}

# Velocity: same keys → velocity components
velocity: dict[str, Array] = {
    "x": jnp.array([...]),
    "y": jnp.array([...]),
    "z": jnp.array([...]),
}
```

### Why Dictionaries?

#### 1. Maximum Performance

Dictionaries are JAX PyTrees by default, enabling efficient tree operations:

% skip: next

```python
# No custom PyTree registration needed
jax.tree.map(lambda x: x * 2, position)
jax.tree.map(lambda x: x[0], velocity)
```

The internal implementation uses `jax.tree.map` for operations like `get_w_at`:

% skip: next

```python
# Efficient extraction via tree operations
q_out = jax.tree.map(lambda v: v[idx], q)
p_out = jax.tree.map(lambda v: v[idx], p)
```

This is faster than dict comprehensions in JAX's tracing/compilation context.

#### 2. Flexibility

Dictionaries support arbitrary coordinate systems:

% skip: next

```python
# Cartesian
position = {"x": ..., "y": ..., "z": ...}

# Spherical
position = {"r": ..., "theta": ..., "phi": ...}

# Cylindrical
position = {"rho": ..., "phi": ..., "z": ...}

# Custom
position = {"custom_coord_1": ..., "custom_coord_2": ...}
```

The algorithm doesn't care about coordinate names—only that position and velocity have matching keys.

#### 3. Seamless JAX Integration

JAX transformations work naturally:

% skip: next

```python
from jax import jit, vmap


@jit
def order_stream(pos, vel):
    # Dicts are PyTrees—no special handling needed
    return nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)


result = order_stream(position, velocity)
```

#### 4. Minimal Overhead

No class instantiation, no validation, no conversions:

% skip: next

```python
# Direct from data
position = {"x": x_array, "y": y_array}

# No need for:
# position = Position(x=x_array, y=y_array)  # Extra object creation
# position = jnp.stack([x_array, y_array])   # Loses coordinate names
```

In the hot loop of `nearest_neighbors_with_momentum`, every operation counts. Raw dicts minimize overhead.

### Type Aliases for Clarity

We provide type aliases to distinguish between single phase-space points
and arrays of points:

```python
from collections.abc import Mapping
from typing import TypeAlias

from jaxtyping import Array, Float

# Scalar (0-D) and 1D array types
FloatScalar = Float[Array, " "]
FloatArray = Float[Array, " N"]

# Component dictionary types
ScalarComponents: TypeAlias = Mapping[str, FloatScalar]
VectorComponents: TypeAlias = Mapping[str, FloatArray]
```

This makes type signatures clear and precise:

% skip: next

```python
def nearest_neighbors_with_momentum(
    position: VectorComponents,
    velocity: VectorComponents,
    start_idx: int,
    lam: float,
) -> KNNPResult:
    """Run kNN+p algorithm on arrays of points."""


def euclidean_distance(
    q_a: ScalarComponents,
    q_b: ScalarComponents,
) -> FloatScalar:
    """Compute distance between two single points."""
```

## Phase-Space Notation

We use standard physics notation for phase-space:

- **q** (position): Generalized coordinates
- **p** (momentum/velocity): Generalized momenta
- **w = (q, p)**: Full phase-space point

This is reflected in function names:

```python
def get_w_at(
    q: VectorComponents,
    p: VectorComponents,
    idx: int,
    /,
) -> tuple[ScalarComponents, ScalarComponents]:
    """Extract a single phase-space point at the given index."""
    ...
```

### Benefits of Standard Notation

1. **Familiar to physicists**: Standard Hamiltonian mechanics notation
2. **Concise**: Short variable names in implementation
3. **Clear intent**: Distinguishes position (`q`) from velocity (`p`)

In user-facing APIs, we use full names (`position`, `velocity`) for clarity.

## Return Types: TypedDict

Results are returned as `TypedDict` for both structure and flexibility:

```python
from typing import TypedDict


class KNNPResult(TypedDict):
    ordered_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]
    position: dict[str, Array]
    velocity: dict[str, Array]
```

### Why TypedDict?

#### 1. Type Safety

IDEs and type checkers know the structure:

% skip: next

```python
result = nearest_neighbors_with_momentum(...)
# IDE autocomplete knows these keys exist
result["ordered_indices"]
result["skipped_indices"]
```

#### 2. Dict Convenience

Access by key, unpack, iterate:

% skip: next

```python
# Direct access
indices = result["ordered_indices"]

# Unpacking
ordered_indices, skipped_indices, pos, vel = (
    result["ordered_indices"],
    result["skipped_indices"],
    result["position"],
    result["velocity"],
)

# Iteration
for key, value in result.items():
    ...
```

#### 3. JAX Compatibility

TypedDict is a regular dict, so it's a PyTree:

% skip: next

```python
@jit
def process_result(result: KNNPResult):
    # Works seamlessly with JAX
    return jax.tree.map(some_function, result)
```

#### 4. No Class Overhead

No need for methods, properties, or inheritance. Just data.

## Design Principles

The API design follows these principles:

### 1. Scalar-First Design

Functions should work on scalar objects (individual points with scalar components) and let users apply `vmap` for batching:

% skip: next

```python
# Works on scalars
q = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
p = {"x": jnp.array(0.1), "y": jnp.array(0.2)}

# Users batch via vmap
get_many = jax.vmap(lambda idx: get_w_at(position, velocity, idx))
results = get_many(jnp.arange(10))
```

This composes better with JAX transformations than requiring specific array shapes.

### 2. Explicit Over Implicit

Prefer explicit parameters over implicit configuration:

% skip: next

```python
# Good: explicit parameters
result = nearest_neighbors_with_momentum(
    position,
    velocity,
    start_idx=0,
    lam=1.0,
    max_dist=2.0,
)

# Avoid: implicit configuration
# algorithm.set_lambda(1.0)  # Global state
# algorithm.set_max_dist(2.0)
# result = algorithm.run(position, velocity)
```

### 3. Immutability

All operations return new objects; nothing is mutated:

% skip: next

```python
# Returns new dicts, doesn't modify input
ordered_pos, ordered_vel = get_ordered_w(result)

# Original unchanged
assert result["position"] is not ordered_pos
```

This is essential for JAX compatibility and functional programming.

### 4. JAX-First

Design for JAX from the ground up:
- Use JAX arrays, not NumPy arrays
- Ensure all operations are traceable
- Avoid Python control flow in computational paths
- Support JIT, vmap, grad out of the box

### 5. Minimal Abstractions

Use the simplest abstraction that works:
- Dicts instead of custom classes
- TypedDict instead of dataclasses
- Functions instead of methods
- Type aliases instead of inheritance

## Comparison with Alternatives

### Why Not Named Tuples?

```python
# Named tuple approach
from typing import NamedTuple


class PhaseSpacePoint(NamedTuple):
    x: Array
    y: Array
    z: Array
```

**Problems:**
- Tied to specific coordinates (x, y, z)
- Can't add/remove dimensions without new class
- Less flexible for different coordinate systems

### Why Not Stacked Arrays?

% skip: next

```python
# Stacked array approach
position = jnp.stack([x, y, z], axis=-1)  # Shape: (N, 3)
```

**Problems:**
- Loses coordinate names
- Assumes fixed number of dimensions
- Requires knowing the stacking convention
- Harder to work with individual components

**When we DO use stacking:**
Internally, the optimized algorithm DOES stack arrays for performance:

% skip: next

```python
# Internal optimization in the hot loop
keys = tuple(sorted(position.keys()))
pos_stacked = jnp.stack([position[k] for k in keys], axis=-1)
```

But this is hidden from users—the public API uses dicts.

### Why Not Custom Classes?

```python
# Custom class approach
class Position:
    def __init__(self, **coords):
        self.coords = coords
```

**Problems:**
- Additional PyTree registration needed
- Overhead from class machinery
- Less idiomatic for JAX users
- Harder to integrate with existing JAX code

## Best Practices

When using knnp:

1. **Keep dicts consistent**: Ensure `position` and `velocity` have the same keys
2. **Use JAX arrays**: Convert NumPy arrays to JAX before passing to knnp
3. **Leverage tree operations**: Use `jax.tree.map` for transformations
4. **Minimize conversions**: Work with dicts directly rather than converting to/from other formats
5. **Trust the types**: Let type checkers help you catch errors

## Evolution and Future

This API design is intentional and stable. We prioritize:

- **Performance**: Dictionaries are fast and JAX-native
- **Simplicity**: No complex class hierarchies
- **Compatibility**: Works with existing JAX ecosystems
- **Flexibility**: Supports any coordinate system

Future additions will maintain these principles.

## See Also

- [JAX Integration Guide](jax-integration.md) - Advanced JAX usage
- [Algorithm Details](algorithm.md) - Implementation specifics
- [API Reference](../api/index.md) - Complete API documentation
