# Autoencoder for Gap Filling

The walk algorithm skips some tracers due to the momentum condition. This guide explains how to use an autoencoder to assign ordering values ($\gamma$) to these skipped tracers.

## Problem and Solution

**Problem**: walk inevitably skips tracers that don't align with the velocity direction.

**Solution**: An autoencoder with two networks:
- **Encoder**: $(x, v) \rightarrow (\gamma, p)$ — predicts ordering and membership probability
- **Decoder**: $\gamma \rightarrow x$ — reconstructs position from ordering

The encoder learns from the walk-ordered tracers and generalizes to predict $\gamma$ for skipped tracers.

## Quick Start

```python
import jax
import jax.numpy as jnp
import localflowwalk as lfw

# Get initial ordering from walk
pos = {"x": jnp.linspace(0, 5, 50), "y": jnp.sin(jnp.linspace(0, jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, jnp.pi, 50))}
result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

# Create normalizer and autoencoder
key = jax.random.key(0)
normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
autoencoder = lfw.nn.PathAutoencoder.make(normalizer, key=key)

# Train autoencoder
config = lfw.nn.TrainingConfig(n_epochs_phase2=500)
trained, _, losses = lfw.nn.train_autoencoder(
    autoencoder, result, config=config, key=key
)

# Fill gaps
ae_result = lfw.nn.fill_ordering_gaps(trained, result)

gamma = ae_result.gamma
ordered_all = ae_result.indices
```

## How It Works

1. **Initialization**: Walk assigns $\gamma \in [-1, 1]$ to ordered tracers
2. **Phase 1**: Encoder learns to predict $\gamma$ from phase-space coordinates
3. **Phase 2**: Both networks train together with momentum constraint — ensures velocity alignment
4. **Membership**: Network outputs probability $p$ to distinguish stream from background

## Customizing Training

```python
config = lfw.nn.TrainingConfig(
    n_epochs_phase1=200,  # Phase 1 epochs (OrderingNet)
    n_epochs_phase2=500,  # Phase 2 epochs (TrackNet)
    batch_size=32,  # Batch size for training
    lambda_prob=1.0,  # Probability loss weight (Phase 1)
    lambda_q=1.0,  # Spatial reconstruction loss weight (Phase 2)
    lambda_p=(1.0, 150.0),  # Velocity alignment loss weight range (Phase 2)
    show_pbar=False,
)

trained, _, losses = lfw.nn.train_autoencoder(
    autoencoder, result, config=config, key=key
)
```

**Key parameters**:
- `lambda_p`: Higher maximum (100-150) enforces stronger velocity alignment in Phase 2
- `n_epochs_phase1`: Should be ~200-500 for good initial interpolation
- `batch_size`: Larger batches are more stable but require more memory
- `lambda_q`: Weight for spatial reconstruction loss in Phase 2
