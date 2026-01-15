# JAX Integration

This guide covers advanced JAX usage with knnp, including JIT compilation, vectorization, differentiation, and performance optimization.

## Basic JAX Compatibility

All knnp functions work seamlessly with JAX:

```python
import jax
import jax.numpy as jnp
from knnp import nearest_neighbors_with_momentum

# Phase-space data
position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}

# Direct call works
result = nearest_neighbors_with_momentum(position, velocity, start_idx=0, lam=1.0)
```

The dict-based API is JAX PyTree compatible by default—no special setup needed.

## JIT Compilation

### Basic JIT

Compile for maximum performance:

```python
from jax import jit


@jit
def order_stream(position, velocity, start_idx, lam):
    return nearest_neighbors_with_momentum(
        position,
        velocity,
        start_idx=start_idx,
        lam=lam,
    )


# First call: compilation + execution
result = order_stream(position, velocity, start_idx=0, lam=1.0)

# Subsequent calls: fast cached execution
result2 = order_stream(position, velocity, start_idx=0, lam=1.5)
```

### Static Arguments

Use `static_argnums` for arguments that should be treated as compile-time constants:

```python
order_stream_jit = jit(
    nearest_neighbors_with_momentum,
    static_argnums=(2,),  # start_idx is static
)

# Each unique start_idx triggers recompilation
result = order_stream_jit(position, velocity, 0, 1.0)  # Compile for start_idx=0
result = order_stream_jit(position, velocity, 1, 1.0)  # Recompile for start_idx=1
```

### Partial Application

Combine JIT with `functools.partial` for cleaner code:

```python
from functools import partial

# Fix some parameters
order_with_defaults = jit(
    partial(
        nearest_neighbors_with_momentum,
        start_idx=0,
        lam=1.0,
    )
)

# Call with just position and velocity
result = order_with_defaults(position, velocity)
```

## Vectorization (vmap)

### Batch Processing

Process multiple streams in parallel:

```python
from jax import vmap

# Multiple streams (batched)
positions = {
    "x": jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]]),  # Stream 1  # Stream 2
}
velocities = {
    "x": jnp.array([[1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.1, 1.2]]),
}

# vmap over batch dimension
order_batch = vmap(
    nearest_neighbors_with_momentum,
    in_axes=(0, 0, None, None),  # batch pos/vel, broadcast start_idx/lam
)

results = order_batch(positions, velocities, 0, 1.0)
# results["ordered_indices"] has shape (2, N) - one ordering per stream
```

### Parameter Sweeps

Explore parameter space efficiently:

```python
# Sweep over lambda values
lambdas = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])

sweep_lambda = vmap(
    lambda lam: nearest_neighbors_with_momentum(position, velocity, 0, lam),
)

results = sweep_lambda(lambdas)
# results["ordered_indices"] has shape (5, N) - one ordering per lambda
```

### Combining JIT and vmap

Maximum performance with both:

```python
# Order matters: vmap outside, jit inside
order_batch_fast = jit(
    vmap(
        nearest_neighbors_with_momentum,
        in_axes=(0, 0, None, None),
    )
)

# Or: jit outside, vmap inside (often equivalent)
order_batch_fast2 = jit(
    vmap(
        nearest_neighbors_with_momentum,
        in_axes=(0, 0, None, None),
    )
)

results = order_batch_fast(positions, velocities, 0, 1.0)
```

## Differentiation

### Gradients

Compute gradients with respect to inputs:

```python
from jax import grad


def loss_function(position, velocity, lam):
    """Example: minimize total distance traveled."""
    result = nearest_neighbors_with_momentum(position, velocity, 0, lam)

    # Extract ordered positions
    ordered_pos = result["position"]

    # Compute sequential distances
    pos_array = jnp.stack([ordered_pos[k] for k in sorted(ordered_pos.keys())])
    diffs = jnp.diff(pos_array, axis=0)
    distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))

    return jnp.sum(distances)


# Gradient w.r.t. lambda
grad_loss_wrt_lambda = grad(loss_function, argnums=2)
gradient = grad_loss_wrt_lambda(position, velocity, 1.0)
```

**Note**: Not all operations are differentiable. The ordering operation itself involves discrete choices (argmin), which have zero or undefined gradients. But you can differentiate through the *results* of the ordering.

### Value and Gradient

Compute both simultaneously:

```python
from jax import value_and_grad

loss_and_grad = value_and_grad(loss_function, argnums=2)
loss_val, grad_val = loss_and_grad(position, velocity, 1.0)
```

## Advanced Patterns

### Conditional Execution

Use `jax.lax.cond` for conditional logic:

```python
from jax.lax import cond


def adaptive_ordering(position, velocity, use_momentum):
    """Order with or without momentum based on flag."""

    def with_momentum(unused):
        return nearest_neighbors_with_momentum(position, velocity, 0, lam=1.0)

    def without_momentum(unused):
        return nearest_neighbors_with_momentum(position, velocity, 0, lam=0.0)

    return cond(use_momentum, with_momentum, without_momentum, None)


result = jit(adaptive_ordering)(position, velocity, True)
```

### Scan for Sequential Processing

Process data in sequence:

```python
from jax.lax import scan


def process_frames(frames_position, frames_velocity):
    """Process time series of phase-space snapshots."""

    def step(carry, frame):
        pos, vel = frame
        result = nearest_neighbors_with_momentum(pos, vel, 0, 1.0)
        # Could use previous ordering as prior...
        return carry, result

    _, results = scan(step, None, (frames_position, frames_velocity))
    return results
```

### Custom VJP

For advanced users: define custom gradients:

```python
from jax import custom_vjp


@custom_vjp
def ordering_with_custom_grad(position, velocity, lam):
    # Forward pass
    result = nearest_neighbors_with_momentum(position, velocity, 0, lam)
    # Return something differentiable
    ordered_pos_array = jnp.stack(
        [result["position"][k] for k in sorted(result["position"].keys())]
    )
    return ordered_pos_array


def ordering_fwd(position, velocity, lam):
    # Forward: compute result and save residuals
    result = ordering_with_custom_grad(position, velocity, lam)
    return result, (position, velocity, lam)


def ordering_bwd(residuals, g):
    # Backward: define custom gradient
    position, velocity, lam = residuals
    # Custom gradient logic...
    return (g, g, 0.0)  # Simplified example


ordering_with_custom_grad.defvjp(ordering_fwd, ordering_bwd)
```

## Performance Optimization

### Internal Optimizations

knnp uses several JAX performance techniques internally:

#### 1. while_loop Instead of Python Loops

The core algorithm uses `jax.lax.while_loop`:

```python
# Not this (Python loop, can't JIT):
for i in range(n):
    ...


# But this (JAX loop, JIT-compatible):
def body_fn(state):
    ...
    return new_state


def cond_fn(state):
    return state[3] < n  # Continue while i < n


final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
```

This enables:
- Full JIT compilation
- No Python overhead
- Potential loop unrolling/optimization

#### 2. Stacked Arrays for Vectorization

Internally, dicts are stacked for efficient computation:

```python
# Public API: dicts
position = {"x": x_arr, "y": y_arr, "z": z_arr}

# Internal: stack for vectorization
keys = tuple(sorted(position.keys()))
pos_stacked = jnp.stack([position[k] for k in keys], axis=-1)  # Shape: (N, 3)

# Now: vectorized distance computation
distances = jnp.sqrt(jnp.sum((pos_stacked - current_pos) ** 2, axis=-1))
```

#### 3. Masking with Infinity

Avoid boolean indexing (not JAX-friendly):

```python
# Not this:
unvisited = ~visited
candidate_distances = distances[unvisited]

# But this:
masked_distances = jnp.where(visited, jnp.inf, distances)
nearest_idx = jnp.argmin(masked_distances)
```

#### 4. Tree Operations

Use `jax.tree.map` for dict operations:

```python
# Efficient
q_out = jax.tree.map(lambda v: v[idx], q)

# Less efficient (still works)
q_out = {k: v[idx] for k, v in q.items()}
```

### User Optimizations

To get maximum performance:

#### 1. Use JAX Arrays

Convert NumPy arrays before calling:

```python
import numpy as np

# NumPy data
pos_np = {"x": np.array([...])}

# Convert to JAX
pos_jax = jax.tree.map(jnp.asarray, pos_np)

# Now use JAX version
result = nearest_neighbors_with_momentum(pos_jax, vel_jax, 0, 1.0)
```

#### 2. Preallocate When Possible

JAX benefits from fixed shapes:

```python
# Good: fixed size
position = {"x": jnp.zeros(1000)}

# Less efficient: dynamic resize
# position = {"x": jnp.concat([position["x"], new_data])}
```

#### 3. Batch Operations

Use vmap instead of loops:

```python
# Slow: Python loop
results = []
for i in range(n_streams):
    result = nearest_neighbors_with_momentum(positions[i], velocities[i], 0, 1.0)
    results.append(result)

# Fast: vmap
results = vmap(
    nearest_neighbors_with_momentum,
    in_axes=(0, 0, None, None),
)(positions, velocities, 0, 1.0)
```

#### 4. Profile and Monitor

Use JAX profiling tools:

```python
# Enable XLA compilation logs
jax.config.update("jax_log_compiles", True)

# Or use jax.profiler for detailed analysis
# (see JAX documentation)
```

## Hardware Acceleration

### GPU/TPU

knnp works on GPU/TPU with no code changes:

```python
import jax

# Check available devices
print(jax.devices())  # [cuda(id=0), ...] or [tpu(id=0), ...]

# Explicitly place on GPU
with jax.default_device(jax.devices("gpu")[0]):
    result = nearest_neighbors_with_momentum(position, velocity, 0, 1.0)
```

**Note**: For small problems (< 1000 points), CPU may be faster due to overhead. Profile to verify.

### Multi-Device Parallelism

For very large problems, use `pmap` for multi-GPU:

```python
from jax import pmap

# Shard data across devices
n_devices = jax.device_count()
positions_sharded = jax.tree.map(
    lambda x: x.reshape(n_devices, -1, *x.shape[1:]), positions
)

# Parallel computation
order_parallel = pmap(
    nearest_neighbors_with_momentum,
    in_axes=(0, 0, None, None),
)

results = order_parallel(positions_sharded, velocities_sharded, 0, 1.0)
```

## Debugging JAX Code

### Common Issues

#### 1. Tracer Errors

**Error**: `leaked tracer` or `tracer bool`

**Cause**: Using Python control flow with traced values:

```python
# Bad: Python if with JAX array
if jnp.sum(distances) > 10:  # Error!
    ...

# Good: jax.lax.cond
jax.lax.cond(
    jnp.sum(distances) > 10,
    true_fun,
    false_fun,
    operand,
)
```

#### 2. Shape Errors

**Error**: `got shape X, expected Y`

**Cause**: Inconsistent shapes in batched operations.

**Solution**: Check vmap axes carefully:

```python
# Make sure batch dimensions align
print(jax.tree.map(lambda x: x.shape, positions))
```

#### 3. Compilation Slowness

**Cause**: Large functions, dynamic shapes, or frequent recompilation.

**Solutions**:
- Use `static_argnums` for constants
- Avoid creating new JIT functions in loops
- Consider `jax.checkpoint` for memory-intensive code

### Debug Mode

Disable JIT for debugging:

```python
with jax.disable_jit():
    # Runs in Python mode, easier to debug
    result = nearest_neighbors_with_momentum(position, velocity, 0, 1.0)
    # Can use print(), pdb, etc.
```

### Print Debugging

Use `jax.debug.print`:

```python
from jax.debug import print as jax_print


@jit
def debug_order(position, velocity):
    result = nearest_neighbors_with_momentum(position, velocity, 0, 1.0)
    # This works inside JIT!
    jax_print("Ordered indices: {}", result["ordered_indices"])
    return result
```

## Best Practices

1. **JIT everything**: Wrap computational functions in `jit`
2. **Batch with vmap**: Use vmap instead of loops for parallel operations
3. **Avoid Python control flow**: Use `jax.lax.cond`, `jax.lax.scan`, etc.
4. **Profile first**: Not every function benefits from GPU acceleration
5. **Use tree operations**: Leverage `jax.tree.map` for dict manipulation
6. **Static arguments**: Use `static_argnums` for compile-time constants
7. **Monitor compilation**: Set `jax_log_compiles=True` during development

## See Also

- [API Design](api-design.md) - Understanding the dict-based API
- [Algorithm Details](algorithm.md) - Implementation specifics
- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX guide
- [API Reference](../api/index.md) - Complete API documentation
