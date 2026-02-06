r"""Autoencoder Neural Network for interpolating skipped tracers.

This module implements the autoencoder neural network from Appendix A.2 of
Nibauer et al. (2022) for assigning $\gamma$ values to stream tracers that were
skipped by the phase-flow walk algorithm.

The autoencoder consists of two parts:

1. **Interpolation Network**: Maps phase-space coordinates $(x, v) \to (\gamma,
   p)$ where $\gamma \in [-1, 1]$ is the ordering parameter and $p \in [0, 1]$
   is the membership probability.
2. **Param-Net (Decoder)**: Maps $\gamma \to x$, reconstructing the position
   from the ordering parameter.

Training follows a two-step process:

1. Train the interpolation network on ordered tracers from phase-flow walk.
2. Jointly train both networks with a momentum condition to refine $\gamma$
   values.

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning." Appendix A.2.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import localflowwalk as lfw

Create phase-space data and run phase-flow walk:

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.sin(jnp.linspace(0, jnp.pi, 20))}
>>> vel = {"x": jnp.ones(20), "y": jnp.cos(jnp.linspace(0, jnp.pi, 20))}
>>> walkresult = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

Train autoencoder to interpolate skipped tracers:

>>> keys = jax.random.split(jax.random.key(0), 2)
>>> normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
>>> ae = lfw.nn.PathAutoencoder.make(normalizer, key=keys[0])
>>> cfg = lfw.nn.TrainingConfig(n_epochs_phase2=100)
>>> ae, *_ = lfw.nn.train_autoencoder(ae, walkresult, key=keys[1])

Predict $\gamma$ for all tracers (including skipped ones):

>>> result = lfw.nn.fill_ordering_gaps(ae, walkresult)
>>> result.gamma
Array([-0.882473  , -0.7614391 , -0.65105796, -0.5597597 , -0.4720139 ,
       -0.38593802, -0.30389455, -0.22263375, -0.13833937, -0.05297466,
       0.03270309,  0.11717375,  0.19892035,  0.2784833 ,  0.35768527,
       0.43787432,  0.51947576,  0.60757154,  0.71056205,  0.8206106 ],
       dtype=float32)

"""

__all__: tuple[str, ...] = (
    # Autoencoder
    "PathAutoencoder",
    "train_autoencoder",
    "TrainingConfig",
    # Encoder-You-Decoder
    "EncoderYouDecoder",
    "RunningMeanDecoder",
    "train_simple_autoencoder",
    # Encoder
    "OrderingNet",
    "train_ordering_net",
    "OrderingTrainingConfig",
    "encoder_loss",
    # Decoder
    "TrackNet",
    "decoder_loss",
    "TrackTrainingConfig",
    # Training functions
    "AutoencoderResult",
    "fill_ordering_gaps",
    # Normalizers
    "AbstractNormalizer",
    "StandardScalerNormalizer",
)

from typing import NamedTuple

import jax.numpy as jnp

from .autoencoder import PathAutoencoder, TrainingConfig, train_autoencoder
from .normalize import AbstractNormalizer, StandardScalerNormalizer
from .order_net import (
    OrderingNet,
    OrderingTrainingConfig,
    encoder_loss,
    train_ordering_net,
)
from .track_net import TrackNet, TrackTrainingConfig, decoder_loss
from .youdecoder import (
    EncoderYouDecoder,
    RunningMeanDecoder,
    train_simple_autoencoder,
)
from localflowwalk._src.algorithm import LocalFlowWalkResult
from localflowwalk._src.custom_types import FSzN, ISzN, VectorComponents

# ===================================================================


class AutoencoderResult(NamedTuple):
    """Result of autoencoder training and prediction.

    Attributes
    ----------
    gamma : Array
        Ordering parameters for all tracers.
    membership_prob : Array
        Membership probabilities for all tracers.
    positions : VectorComponents
        Original position data (dict with 1D arrays).
    velocities : VectorComponents
        Original velocity data (dict with 1D arrays).
    indices : Array
        Indices sorted by gamma value.

    """

    gamma: FSzN
    membership_prob: FSzN
    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN


# ===================================================================


def fill_ordering_gaps(
    model: PathAutoencoder,
    lfw_result: LocalFlowWalkResult,
    /,
    prob_threshold: float = 0.5,
) -> AutoencoderResult:
    r"""Use trained autoencoder to fill gaps in phase-flow walk ordering.

    This function predicts $\gamma$ values for all tracers (including those
    skipped by phase-flow walk) and returns a complete ordering.

    Parameters
    ----------
    model : PathAutoencoder
        Trained autoencoder model.
    lfw_result : LocalFlowWalkResult
        Result from walk_local_flow.
    prob_threshold : float, optional
        Minimum membership probability to include. Default: 0.5.

    Returns
    -------
    result : AutoencoderResult
        Complete ordering including previously skipped tracers.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
    >>> keys = jax.random.split(jax.random.key(0), 2)
    >>> ae = lfw.nn.PathAutoencoder.make(normalizer, key=keys[0])
    >>> ae, *_ = lfw.nn.train_autoencoder(ae, result, key=keys[1])
    >>> full_ordering = lfw.nn.fill_ordering_gaps(ae, result)

    """
    q, p = lfw_result.positions, lfw_result.velocities

    # Predict gamma and probability for all tracers
    gamma, prob = model.encode(q, p)
    # Sort by gamma to get ordering
    sorted_indices = jnp.argsort(gamma)

    # Filter by probability threshold
    high_prob_mask = prob[sorted_indices] >= prob_threshold
    filtered_indices = sorted_indices[high_prob_mask]

    return AutoencoderResult(
        gamma=gamma,
        membership_prob=prob,
        positions=q,
        velocities=p,
        indices=filtered_indices,
    )
