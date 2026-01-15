r"""Autoencoder Neural Network for interpolating skipped tracers.

This module implements the autoencoder neural network from Appendix A.2 of
Nibauer et al. (2022) for assigning $\gamma$ values to stream tracers that were
skipped by the kNN+p algorithm.

The autoencoder consists of two parts:
1. **Interpolation Network**: Maps phase-space coordinates $(x, v) \to (\gamma, p)$
   where $\gamma \in [-1, 1]$ is the ordering parameter and $p \in [0, 1]$ is the
   membership probability.
2. **Param-Net (Decoder)**: Maps $\gamma \to x$, reconstructing the position from
   the ordering parameter.

Training follows a two-step process:
1. Train the interpolation network on ordered tracers from kNN+p
2. Jointly train both networks with a momentum condition to refine $\gamma$ values

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning." Appendix A.2.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> from knnp import nearest_neighbors_with_momentum
>>> from knnp.autoencoder import (
...     InterpolationNetwork,
...     ParamNet,
...     Autoencoder,
...     train_autoencoder,
... )

Create phase-space data and run kNN+p:

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.sin(jnp.linspace(0, jnp.pi, 20))}
>>> vel = {"x": jnp.ones(20), "y": jnp.cos(jnp.linspace(0, jnp.pi, 20))}
>>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)

Train autoencoder to interpolate skipped tracers:

>>> rngs = jax.random.PRNGKey(0)
>>> autoencoder = Autoencoder(rngs=rngs, n_dims=2)
>>> trained, losses = train_autoencoder(autoencoder, result, n_epochs=100)

Predict $\gamma$ for all tracers (including skipped ones):

>>> gamma, prob = trained.predict(pos, vel)

"""

__all__: tuple[str, ...] = (
    # Network components
    "InterpolationNetwork",
    "ParamNet",
    "Autoencoder",
    # Training functions
    "train_autoencoder",
    "AutoencoderResult",
    "TrainingConfig",
    # Convenience functions
    "fill_ordering_gaps",
)

from dataclasses import dataclass
from typing import TypedDict

import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm.auto import tqdm

from .algorithm import KNNPResult
from .custom_types import VectorComponents


class InterpolationNetwork(nnx.Module):
    r"""Interpolation network: maps phase-space $(x, v) \to (\gamma, p)$.

    This network takes 6D (or 2D * n_dims) phase-space coordinates and outputs:
    - $\gamma \in [-1, 1]$: The ordering parameter along the stream
    - $p \in [0, 1]$: The membership probability (1 = likely stream member)

    The architecture follows Appendix B.3 of Nibauer et al. (2022):
    - Fully connected MLP with 3 hidden layers of 100 nodes each
    - tanh activation functions
    - Output $\gamma$ uses tanh to constrain to [-1, 1]
    - Output p uses sigmoid to constrain to [0, 1]

    The $\gamma$ output is scaled by a temperature parameter to prevent
    tanh saturation during training. Without this, the network tends to
    output saturated values near ±1 for all points, losing fine-grained
    ordering information.

    Parameters
    ----------
    rngs : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    def __init__(
        self,
        rngs: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims
        input_size = 2 * n_dims  # Position + velocity

        # Build layers
        layers = []
        in_features = input_size
        for _ in range(n_hidden):
            layers.append(nnx.Linear(in_features, hidden_size, rngs=nnx.Rngs(rngs)))
            in_features = hidden_size
        self.layers = nnx.List(layers)

        # Output heads
        self.gamma_head = nnx.Linear(hidden_size, 1, rngs=nnx.Rngs(rngs))
        self.prob_head = nnx.Linear(hidden_size, 1, rngs=nnx.Rngs(rngs))

    def __call__(
        self, phase_space: Float[Array, "... D"]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Forward pass through the interpolation network.

        Parameters
        ----------
        phase_space : Array
            Phase-space coordinates of shape (..., 2*n_dims).
            For each point: [x, y, z, vx, vy, vz] (or 2D equivalent).

        Returns
        -------
        gamma : Array
            Ordering parameter in [-1, 1], shape (...).
        prob : Array
            Membership probability in [0, 1], shape (...).

        """
        x = phase_space
        for layer in self.layers:
            x = nnx.tanh(layer(x))

        # Output heads with appropriate activations
        # Scale gamma head by temperature to prevent tanh saturation
        # Without scaling, the network learns to saturate at ±1 for all points
        gamma_temperature = 5.0  # Reduces pre-activation magnitude
        gamma = nnx.tanh(self.gamma_head(x) / gamma_temperature).squeeze(-1)
        prob = nnx.sigmoid(self.prob_head(x)).squeeze(-1)

        return gamma, prob


class ParamNet(nnx.Module):
    r"""Param-Net (decoder): maps $\gamma \to$ position (x, y, z).

    This network reconstructs the stream track position from the ordering
    parameter $\gamma$. It serves as the second half of the autoencoder.

    The architecture follows Appendix B.1 of Nibauer et al. (2022):
    - Fully connected MLP with 3 hidden layers of 100 nodes each
    - tanh activation functions
    - Linear output layer

    Parameters
    ----------
    rngs : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    def __init__(
        self,
        rngs: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims

        # Build layers (use nnx.List for Flax 0.12+ compatibility)
        layers = []
        in_features = 1  # γ is a scalar
        for _ in range(n_hidden):
            layers.append(nnx.Linear(in_features, hidden_size, rngs=nnx.Rngs(rngs)))
            in_features = hidden_size
        self.layers = nnx.List(layers)

        # Output layer for position
        self.output_layer = nnx.Linear(hidden_size, n_dims, rngs=nnx.Rngs(rngs))

    def __call__(self, gamma: Float[Array, "..."]) -> Float[Array, "... D"]:
        """Forward pass through Param-Net.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [-1, 1], shape (...).

        Returns
        -------
        position : Array
            Reconstructed position of shape (..., n_dims).

        """
        # Add feature dimension for linear layers
        x = gamma[..., None]

        for layer in self.layers:
            x = nnx.tanh(layer(x))

        return self.output_layer(x)


class Autoencoder(nnx.Module):
    r"""Autoencoder for stream tracer interpolation.

    Combines the InterpolationNetwork (encoder) and ParamNet (decoder)
    into a single autoencoder architecture that can:
    1. Predict $\gamma$ values for arbitrary phase-space points
    2. Reconstruct positions from $\gamma$ values
    3. Provide membership probabilities

    The autoencoder is trained in two phases:
    1. Train interpolation network on ordered tracers from kNN+p
    2. Joint training with momentum condition to refine $\gamma$ values

    Parameters
    ----------
    rngs : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int, optional
        Number of spatial dimensions. Default: 3.
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    Attributes
    ----------
    encoder : InterpolationNetwork
        The interpolation network mapping phase-space to $(\gamma, p)$.
    decoder : ParamNet
        The Param-Net mapping $\gamma$ to position.
    pos_mean : Array or None
        Mean of position data for standardization.
    pos_std : Array or None
        Std of position data for standardization.

    """

    def __init__(
        self,
        rngs: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims
        keys = jr.split(rngs, 2)

        self.encoder = InterpolationNetwork(keys[0], n_dims, hidden_size, n_hidden)
        self.decoder = ParamNet(keys[1], n_dims, hidden_size, n_hidden)

        # Standardization parameters (set during training)
        # Use nnx.Variable to mark as data fields for Flax 0.12+ compatibility
        self.pos_mean: nnx.Variable[Array | None] = nnx.Variable(None)
        self.pos_std: nnx.Variable[Array | None] = nnx.Variable(None)
        self.vel_mean: nnx.Variable[Array | None] = nnx.Variable(None)
        self.vel_std: nnx.Variable[Array | None] = nnx.Variable(None)

    def encode(
        self, position: VectorComponents, velocity: VectorComponents
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        r"""Encode phase-space coordinates to $(\gamma, p)$.

        Parameters
        ----------
        position : VectorComponents
            Position dictionary with 1D array values.
        velocity : VectorComponents
            Velocity dictionary with 1D array values.

        Returns
        -------
        gamma : Array
            Ordering parameters in [-1, 1].
        prob : Array
            Membership probabilities in [0, 1].

        """
        # Stack to array
        keys = tuple(sorted(position.keys()))
        pos_arr = jnp.stack([position[k] for k in keys], axis=-1)
        vel_arr = jnp.stack([velocity[k] for k in keys], axis=-1)

        # Standardize if parameters are set
        pos_mean = self.pos_mean.get_value()
        pos_std = self.pos_std.get_value()
        vel_mean = self.vel_mean.get_value()
        vel_std = self.vel_std.get_value()

        if pos_mean is not None and pos_std is not None:
            pos_arr = (pos_arr - pos_mean) / (pos_std + 1e-8)
        if vel_mean is not None and vel_std is not None:
            vel_arr = (vel_arr - vel_mean) / (vel_std + 1e-8)

        # Concatenate position and velocity
        phase_space = jnp.concat([pos_arr, vel_arr], axis=-1)

        return self.encoder(phase_space)

    def decode(self, gamma: Float[Array, " N"]) -> Float[Array, "N D"]:
        r"""Decode $\gamma$ to reconstructed position.

        Parameters
        ----------
        gamma : Array
            Ordering parameters in [-1, 1].

        Returns
        -------
        position : Array
            Reconstructed positions of shape (N, n_dims).
            Note: Returns standardized positions. Use decode_position()
            for unstandardized output.

        """
        return self.decoder(gamma)

    def decode_position(self, gamma: Float[Array, " N"]) -> VectorComponents:
        r"""Decode $\gamma$ to reconstructed position dictionary.

        This method handles unstandardization automatically.

        Parameters
        ----------
        gamma : Array
            Ordering parameters in [-1, 1].

        Returns
        -------
        position : VectorComponents
            Reconstructed position dictionary.

        """
        pos_arr = self.decode(gamma)

        # Unstandardize if parameters are set
        pos_mean = self.pos_mean.get_value()
        pos_std = self.pos_std.get_value()
        if pos_mean is not None and pos_std is not None:
            pos_arr = pos_arr * (pos_std + 1e-8) + pos_mean

        # Convert back to dict (assumes sorted keys)
        keys = [f"d{i}" for i in range(self.n_dims)]
        return {k: pos_arr[..., i] for i, k in enumerate(keys)}

    def predict(
        self, position: VectorComponents, velocity: VectorComponents
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        r"""Predict $\gamma$ and membership probability for phase-space points.

        Parameters
        ----------
        position : VectorComponents
            Position dictionary with 1D array values.
        velocity : VectorComponents
            Velocity dictionary with 1D array values.

        Returns
        -------
        gamma : Array
            Predicted ordering parameters in [-1, 1].
        prob : Array
            Predicted membership probabilities in [0, 1].

        """
        return self.encode(position, velocity)


class AutoencoderResult(TypedDict):
    """Result of autoencoder training and prediction.

    Keys
    ----
    gamma : Array
        Ordering parameters for all tracers.
    membership_prob : Array
        Membership probabilities for all tracers.
    position : VectorComponents
        Original position data (dict with 1D arrays).
    velocity : VectorComponents
        Original velocity data (dict with 1D arrays).
    ordered_indices : tuple[int, ...]
        Indices sorted by gamma value.

    """

    gamma: Array
    membership_prob: Array
    position: dict[str, Array]
    velocity: dict[str, Array]
    ordered_indices: tuple[int, ...]


def _stack_phase_space(
    position: VectorComponents, velocity: VectorComponents
) -> tuple[Array, Array, tuple[str, ...]]:
    """Stack position and velocity dicts into arrays."""
    keys = tuple(sorted(position.keys()))
    pos_arr = jnp.stack([position[k] for k in keys], axis=-1)
    vel_arr = jnp.stack([velocity[k] for k in keys], axis=-1)
    return pos_arr, vel_arr, keys


def _compute_standardization(
    arr: Array,
) -> tuple[Float[Array, " D"], Float[Array, " D"]]:
    """Compute mean and std for standardization."""
    return jnp.mean(arr, axis=0), jnp.std(arr, axis=0)


def _shuffle_and_batch(
    key: PRNGKeyArray,
    phase_space: Array,
    gamma_target: Array,
    prob_target: Array,
    mask: Array,
    batch_size: int,
) -> tuple[Array, Array, Array, Array, int]:
    """Shuffle data and create padded batches for lax.scan.

    Returns arrays of shape (n_batches, batch_size, ...) suitable for lax.scan.
    The last batch is padded with zeros if needed.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for shuffling.
    phase_space, gamma_target, prob_target, mask : Array
        Training data arrays.
    batch_size : int
        Size of each batch.

    Returns
    -------
    batched_phase_space : Array of shape (n_batches, batch_size, D)
    batched_gamma : Array of shape (n_batches, batch_size)
    batched_prob : Array of shape (n_batches, batch_size)
    batched_mask : Array of shape (n_batches, batch_size)
    n_valid : int
        Number of valid (non-padded) samples in total.

    """
    n = phase_space.shape[0]

    # Shuffle indices
    perm = jr.permutation(key, n)
    phase_space = phase_space[perm]
    gamma_target = gamma_target[perm]
    prob_target = prob_target[perm]
    mask = mask[perm]

    # Compute number of batches (pad to make divisible by batch_size)
    n_batches = (n + batch_size - 1) // batch_size
    padded_n = n_batches * batch_size

    # Pad arrays
    pad_n = padded_n - n
    if pad_n > 0:
        phase_space = jnp.concat(
            [phase_space, jnp.zeros((pad_n, phase_space.shape[1]))], axis=0
        )
        gamma_target = jnp.concat([gamma_target, jnp.zeros(pad_n)], axis=0)
        prob_target = jnp.concat([prob_target, jnp.zeros(pad_n)], axis=0)
        mask = jnp.concat([mask, jnp.zeros(pad_n)], axis=0)

    # Reshape to (n_batches, batch_size, ...)
    batched_phase_space = phase_space.reshape(n_batches, batch_size, -1)
    batched_gamma = gamma_target.reshape(n_batches, batch_size)
    batched_prob = prob_target.reshape(n_batches, batch_size)
    batched_mask = mask.reshape(n_batches, batch_size)

    return batched_phase_space, batched_gamma, batched_prob, batched_mask, n


@dataclass
class TrainingConfig:
    r"""Configuration for autoencoder training.

    Parameters
    ----------
    learning_rate : float
        Learning rate for optimizer. Default: 1e-3.
    n_epochs : int
        Number of training epochs. Default: 500.
    batch_size : int
        Batch size for training. Default: 32.
    lambda_momentum : float
        Weight for momentum loss term ($\lambda$ in paper). Default: 100.0.
    n_random_samples : int
        Number of random samples for membership training. Default: 100.
    phase1_epochs : int
        Epochs for phase 1 (interpolation only). Default: 200.
    progress_bar : bool
        Whether to show a progress bar during training. Default: True.

    """

    learning_rate: float = 1e-3
    n_epochs: int = 500
    batch_size: int = 32
    lambda_momentum: float = 100.0
    n_random_samples: int = 100
    phase1_epochs: int = 200
    progress_bar: bool = True


def _assign_gamma_init(
    ordered_indices: tuple[int, ...],
    n_total: int,
    position: Float[Array, "N D"],
) -> Float[Array, " N"]:
    r"""Assign initial $\gamma$ values to ordered tracers based on arc-length.

    Following Equation (A1) from the paper:
    $\gamma$ increases monotonically along the stream from -1 to 1,
    scaled by cumulative arc-length along the ordered path.

    Parameters
    ----------
    ordered_indices : tuple[int, ...]
        Indices from kNN+p in order.
    n_total : int
        Total number of tracers.
    position : Array
        Position array of shape (N, n_dims).

    Returns
    -------
    gamma : Array
        Initial $\gamma$ values (NaN for skipped tracers).

    """
    n_ordered = len(ordered_indices)

    if n_ordered <= 1:
        # Edge case: single point
        gamma = jnp.full(n_total, jnp.nan)
        if n_ordered == 1:
            gamma = gamma.at[ordered_indices[0]].set(0.0)
        return gamma

    # Get ordered positions
    ordered_pos = position[jnp.array(ordered_indices)]

    # Compute arc-length: cumulative distance along the ordered path
    # Distance between consecutive points
    diffs = jnp.diff(ordered_pos, axis=0)
    segment_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=-1))

    # Cumulative arc-length (starting from 0)
    arc_length = jnp.concat([jnp.array([0.0]), jnp.cumsum(segment_lengths)])

    # Scale to [-1, 1]
    total_length = arc_length[-1]
    gamma_values = 2.0 * (arc_length / (total_length + 1e-8)) - 1.0

    # Initialize with NaN for skipped tracers
    gamma = jnp.full(n_total, jnp.nan)

    # Assign γ values to ordered tracers
    for i, idx in enumerate(ordered_indices):
        gamma = gamma.at[idx].set(gamma_values[i])

    return gamma


def _interpolation_loss(
    encoder: InterpolationNetwork,
    phase_space: Float[Array, "N D"],
    gamma_target: Float[Array, " N"],
    prob_target: Float[Array, " N"],
    mask: Float[Array, " N"],
    is_random: Float[Array, " N"] | None = None,
) -> Float[Array, ""]:
    r"""Compute loss for interpolation network training.

    Loss from Equation (A2):
    $L = \sum_i |\gamma_\theta(w_i) - \gamma_{\text{init},i}|^2 + |p_\theta(w_i) - 1|^2
      + \sum_{\text{rand}} |p_\theta(w_{\text{rand}}) - 0|^2$

    IMPORTANT: The probability loss is only applied to:
    1. Ordered tracers (prob_target=1)
    2. Random samples (prob_target=0)

    Skipped tracers (mask=0 but not random) do NOT have probability loss.
    Their probability is learned indirectly through the momentum condition
    and reconstruction loss.

    Parameters
    ----------
    encoder : InterpolationNetwork
        The interpolation network.
    phase_space : Array
        Phase-space coordinates of shape (N, 2*n_dims).
    gamma_target : Array
        Target $\gamma$ values for ordered tracers.
    prob_target : Array
        Target probability (1 for ordered, 0 for random).
    mask : Array
        Mask for valid (ordered) tracers.
    is_random : Array, optional
        Mask indicating which points are random samples (1) vs actual tracers (0).
        If None, all points with mask=0 are assumed to be random samples.

    Returns
    -------
    loss : Array
        Scalar loss value.

    """
    gamma_pred, prob_pred = encoder(phase_space)

    # γ loss only for ordered tracers
    gamma_loss = jnp.where(mask > 0.5, (gamma_pred - gamma_target) ** 2, 0.0)

    # Probability loss for ordered (prob=1) and random (prob=0) only
    # NOT for skipped tracers (mask=0 but not random)
    if is_random is None:
        # Backward compatibility: assume mask=0 means random sample
        prob_loss = (prob_pred - prob_target) ** 2
    else:
        # Only apply prob loss to ordered (mask=1) or random (is_random=1)
        prob_loss_mask = (mask > 0.5) | (is_random > 0.5)
        prob_loss = jnp.where(prob_loss_mask, (prob_pred - prob_target) ** 2, 0.0)

    return jnp.mean(gamma_loss) + jnp.mean(prob_loss)


def _momentum_loss(
    autoencoder: Autoencoder,
    phase_space: Float[Array, "N D"],
    velocity: Float[Array, "N D"],
    pos_std: Float[Array, " D"],
    dgamma: float = 0.01,
) -> Float[Array, ""]:
    r"""Compute momentum loss for joint training.

    The momentum condition from Equation (A3) encourages the tangent
    to the decoded track ($dx/d\gamma$) to align with the velocity direction.
    This is crucial for properly ordering ALL tracers, including skipped ones.

    The loss is computed on ALL tracers (not just ordered ones) so that the
    network learns to predict correct $\gamma$ values for skipped tracers.

    Parameters
    ----------
    autoencoder : Autoencoder
        The autoencoder model.
    phase_space : Array
        Phase-space coordinates (standardized).
    velocity : Array
        Velocity vectors (unstandardized, physical units).
    pos_std : Array
        Position standard deviations for unstandardizing the decoder output.
    dgamma : float
        Step size for numerical derivative.

    Returns
    -------
    loss : Array
        Scalar momentum loss.

    """
    gamma_pred, _ = autoencoder.encoder(phase_space)

    # Compute dx/dγ numerically using the decoder
    pos_at_gamma_plus = autoencoder.decoder(gamma_pred + dgamma)
    pos_at_gamma_minus = autoencoder.decoder(gamma_pred - dgamma)

    # Use central difference for better numerical accuracy
    # The decoder outputs standardized positions, so we need to unstandardize
    # to get directions in physical space
    dx_dgamma_std = (pos_at_gamma_plus - pos_at_gamma_minus) / (2 * dgamma)

    # Unstandardize the direction (multiply by std to get physical units)
    # Note: We only care about direction, but different std values in each
    # dimension change the direction, so we must unstandardize
    pos_std_safe = pos_std + 1e-8
    dx_dgamma = dx_dgamma_std * pos_std_safe

    # Normalize to get unit tangent direction
    dx_norm = jnp.sqrt(jnp.sum(dx_dgamma**2, axis=-1, keepdims=True))
    T_hat = dx_dgamma / (dx_norm + 1e-8)

    # Normalize velocity to get unit velocity direction
    vel_norm = jnp.sqrt(jnp.sum(velocity**2, axis=-1, keepdims=True))
    v_hat = velocity / (vel_norm + 1e-8)

    # Use 1 - |cos(θ)| as loss to encourage alignment (either direction)
    # This is more robust than ||T_hat - v_hat||² which penalizes antiparallel
    cos_sim = jnp.sum(T_hat * v_hat, axis=-1)
    alignment_loss = 1.0 - jnp.abs(cos_sim)

    return jnp.mean(alignment_loss)


def _reconstruction_loss(
    autoencoder: Autoencoder,
    phase_space: Float[Array, "N D"],
    position: Float[Array, "N D"],
    pos_mean: Float[Array, " D"],
    pos_std: Float[Array, " D"],
) -> Float[Array, ""]:
    """Compute reconstruction loss.

    Parameters
    ----------
    autoencoder : Autoencoder
        The autoencoder model.
    phase_space : Array
        Phase-space coordinates.
    position : Array
        Target position coordinates.
    pos_mean, pos_std : Array
        Standardization parameters.

    Returns
    -------
    loss : Array
        Scalar reconstruction loss.

    """
    gamma_pred, _ = autoencoder.encoder(phase_space)
    pos_recon = autoencoder.decoder(gamma_pred)

    # Standardize target for comparison
    pos_std_safe = pos_std + 1e-8
    position_standardized = (position - pos_mean) / pos_std_safe

    return jnp.mean((pos_recon - position_standardized) ** 2)


def train_autoencoder(
    autoencoder: Autoencoder,
    knnp_result: KNNPResult,
    config: TrainingConfig | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Autoencoder, Float[Array, " E"]]:
    """Train the autoencoder on kNN+p results.

    Training follows the two-phase process from Appendix A.2:
    1. Train interpolation network on ordered tracers
    2. Joint training with momentum condition

    Parameters
    ----------
    autoencoder : Autoencoder
        The autoencoder model to train.
    knnp_result : KNNPResult
        Result from nearest_neighbors_with_momentum.
    config : TrainingConfig, optional
        Training configuration. Default: TrainingConfig().
    key : PRNGKeyArray, optional
        Random key for training. Default: jax.random.PRNGKey(0).

    Returns
    -------
    trained_autoencoder : Autoencoder
        The trained autoencoder model.
    losses : Array
        Training losses per epoch.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from knnp import nearest_neighbors_with_momentum
    >>> from knnp.autoencoder import Autoencoder, train_autoencoder

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)
    >>> autoencoder = Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
    >>> trained, losses = train_autoencoder(autoencoder, result)

    """
    if config is None:
        config = TrainingConfig()
    if key is None:
        key = jr.PRNGKey(0)

    position = knnp_result["position"]
    velocity = knnp_result["velocity"]
    ordered_indices = knnp_result["ordered_indices"]

    # Stack data
    pos_arr, vel_arr, keys = _stack_phase_space(position, velocity)
    n_total = pos_arr.shape[0]
    n_dims = pos_arr.shape[1]

    # Compute standardization parameters
    pos_mean, pos_std = _compute_standardization(pos_arr)
    vel_mean, vel_std = _compute_standardization(vel_arr)

    # Store in autoencoder (using set_value for nnx.Variable in Flax 0.12+)
    autoencoder.pos_mean.set_value(pos_mean)
    autoencoder.pos_std.set_value(pos_std)
    autoencoder.vel_mean.set_value(vel_mean)
    autoencoder.vel_std.set_value(vel_std)

    # Standardize data
    pos_std_safe = pos_std + 1e-8
    vel_std_safe = vel_std + 1e-8
    pos_standardized = (pos_arr - pos_mean) / pos_std_safe
    vel_standardized = (vel_arr - vel_mean) / vel_std_safe
    phase_space = jnp.concat([pos_standardized, vel_standardized], axis=-1)

    # Assign initial γ values based on arc-length
    gamma_init = _assign_gamma_init(ordered_indices, n_total, pos_arr)
    mask = ~jnp.isnan(gamma_init)
    gamma_init_safe = jnp.where(mask, gamma_init, 0.0)

    # Probability targets: 1 for ordered, 0 for random samples
    prob_target = mask.astype(jnp.float32)

    # Prepare Phase 1 data: ONLY ordered tracers
    # Per the paper (Appendix A.2), Phase 1 trains only on ordered tracers (p=1)
    # and random samples (p=0). Skipped tracers are NOT included in Phase 1.
    ordered_mask_bool = mask.astype(bool)
    phase1_phase_space = phase_space[ordered_mask_bool]
    phase1_gamma = gamma_init_safe[ordered_mask_bool]
    phase1_prob = prob_target[ordered_mask_bool]
    phase1_mask = mask[ordered_mask_bool].astype(jnp.float32)

    # Phase 2 data: ALL tracers
    phase2_phase_space = phase_space
    phase2_gamma = gamma_init_safe
    phase2_prob = prob_target
    phase2_mask = mask.astype(jnp.float32)

    # Pre-compute data bounds ONCE (not per batch)
    pos_min = jnp.min(pos_arr, axis=0)
    pos_max = jnp.max(pos_arr, axis=0)
    vel_min = jnp.min(vel_arr, axis=0)
    vel_max = jnp.max(vel_arr, axis=0)

    # Create optimizer with wrt=nnx.Param (required in Flax 0.11+)
    optimizer = nnx.Optimizer(
        autoencoder, optax.adam(config.learning_rate), wrt=nnx.Param
    )

    losses = []

    # Single batch step for phase 1 (used with lax.scan)
    def phase1_batch_step(
        model: Autoencoder,
        opt: nnx.Optimizer,
        batch_phase_space: Array,
        batch_gamma_target: Array,
        batch_prob_target: Array,
        batch_mask: Array,
        key: Array,
    ) -> Float[Array, ""]:
        """Single batched training step for phase 1."""
        # Generate random samples for membership training
        rand_key, _ = jr.split(key)
        n_rand = config.n_random_samples

        rand_keys = jr.split(rand_key, 2)
        rand_pos = jr.uniform(
            rand_keys[0], (n_rand, n_dims), minval=pos_min, maxval=pos_max
        )
        rand_vel = jr.uniform(
            rand_keys[1], (n_rand, n_dims), minval=vel_min, maxval=vel_max
        )

        # Standardize random samples
        rand_pos_std = (rand_pos - pos_mean) / pos_std_safe
        rand_vel_std = (rand_vel - vel_mean) / vel_std_safe
        rand_phase_space = jnp.concat([rand_pos_std, rand_vel_std], axis=-1)

        # Combined batch + random data
        all_phase_space = jnp.concat([batch_phase_space, rand_phase_space], axis=0)
        all_gamma_target = jnp.concat([batch_gamma_target, jnp.zeros(n_rand)], axis=0)
        all_prob_target = jnp.concat([batch_prob_target, jnp.zeros(n_rand)], axis=0)
        all_mask = jnp.concat([batch_mask, jnp.zeros(n_rand)], axis=0)

        def loss_fn(m):
            return _interpolation_loss(
                m.encoder,
                all_phase_space,
                all_gamma_target,
                all_prob_target,
                all_mask,
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    # Single batch step for phase 2 (used with lax.scan)
    def phase2_batch_step(
        model: Autoencoder,
        opt: nnx.Optimizer,
        batch_phase_space: Array,
        batch_gamma_target: Array,
        batch_prob_target: Array,
        batch_mask: Array,
        key: Array,
    ) -> Float[Array, ""]:
        r"""Single batched training step for phase 2.

        Phase 2 includes:
        - Interpolation loss (to maintain $\gamma$ predictions)
        - Reconstruction loss (decoder accuracy)
        - Momentum loss (tangent-velocity alignment)
        """
        # Generate random samples
        rand_key, _ = jr.split(key)
        n_rand = config.n_random_samples

        rand_keys = jr.split(rand_key, 2)
        rand_pos = jr.uniform(
            rand_keys[0], (n_rand, n_dims), minval=pos_min, maxval=pos_max
        )
        rand_vel = jr.uniform(
            rand_keys[1], (n_rand, n_dims), minval=vel_min, maxval=vel_max
        )

        # Standardize random samples
        rand_pos_std = (rand_pos - pos_mean) / pos_std_safe
        rand_vel_std = (rand_vel - vel_mean) / vel_std_safe
        rand_phase_space = jnp.concat([rand_pos_std, rand_vel_std], axis=-1)

        # Unstandardize position and velocity for phase 2 losses
        batch_position = batch_phase_space[..., :n_dims] * pos_std_safe + pos_mean
        batch_velocity = batch_phase_space[..., n_dims:] * vel_std_safe + vel_mean

        def loss_fn(m):
            # Interpolation loss - same as phase 1 (includes random samples)
            n_batch = batch_phase_space.shape[0]

            all_phase_space = jnp.concat([batch_phase_space, rand_phase_space], axis=0)
            all_gamma_target = jnp.concat(
                [batch_gamma_target, jnp.zeros(n_rand)], axis=0
            )
            all_prob_target = jnp.concat([batch_prob_target, jnp.zeros(n_rand)], axis=0)
            all_mask = jnp.concat([batch_mask, jnp.zeros(n_rand)], axis=0)
            # Mark which points are random samples (prob loss applies)
            # Batch points are NOT random, even if mask=0 (those are skipped tracers)
            all_is_random = jnp.concat([jnp.zeros(n_batch), jnp.ones(n_rand)], axis=0)
            interp_loss = _interpolation_loss(
                m.encoder,
                all_phase_space,
                all_gamma_target,
                all_prob_target,
                all_mask,
                is_random=all_is_random,
            )

            # Reconstruction loss (on batch data only, not random)
            recon_loss = _reconstruction_loss(
                m, batch_phase_space, batch_position, pos_mean, pos_std
            )

            # Momentum loss on ALL tracers (including skipped) - this is the key
            # to learning correct γ values for skipped tracers
            mom_loss = _momentum_loss(m, batch_phase_space, batch_velocity, pos_std)

            return interp_loss + recon_loss + config.lambda_momentum * mom_loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    # JIT-compile the batch step functions
    phase1_step_jit = nnx.jit(phase1_batch_step)
    phase2_step_jit = nnx.jit(phase2_batch_step)

    # Training loop
    epoch_iter = range(config.n_epochs)
    if config.progress_bar:
        epoch_iter = tqdm(
            epoch_iter,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

    for epoch in epoch_iter:
        key, epoch_key = jr.split(key)

        # Choose appropriate data for current phase
        if epoch < config.phase1_epochs:
            # Phase 1: Only ordered tracers
            (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                _,
            ) = _shuffle_and_batch(
                epoch_key,
                phase1_phase_space,
                phase1_gamma,
                phase1_prob,
                phase1_mask,
                config.batch_size,
            )
            n_batches = batched_phase_space.shape[0]

            # Generate keys for each batch
            key, keys_key = jr.split(key)
            batch_keys = jr.split(keys_key, n_batches)

            # Run batches with JIT-compiled step function
            epoch_losses = []
            for i in range(n_batches):
                loss = phase1_step_jit(
                    autoencoder,
                    optimizer,
                    batched_phase_space[i],
                    batched_gamma[i],
                    batched_prob[i],
                    batched_mask[i],
                    batch_keys[i],
                )
                epoch_losses.append(loss)
            batch_losses = jnp.array(epoch_losses)
        else:
            # Phase 2: All tracers
            (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                _,
            ) = _shuffle_and_batch(
                epoch_key,
                phase2_phase_space,
                phase2_gamma,
                phase2_prob,
                phase2_mask,
                config.batch_size,
            )
            n_batches = batched_phase_space.shape[0]

            # Generate keys for each batch
            key, keys_key = jr.split(key)
            batch_keys = jr.split(keys_key, n_batches)

            # Run batches with JIT-compiled step function
            epoch_losses = []
            for i in range(n_batches):
                loss = phase2_step_jit(
                    autoencoder,
                    optimizer,
                    batched_phase_space[i],
                    batched_gamma[i],
                    batched_prob[i],
                    batched_mask[i],
                    batch_keys[i],
                )
                epoch_losses.append(loss)
            batch_losses = jnp.array(epoch_losses)

        # Average loss for this epoch
        avg_loss = float(jnp.mean(batch_losses))
        losses.append(avg_loss)

        # Update progress bar with loss info
        if config.progress_bar and hasattr(epoch_iter, "set_postfix"):
            phase = "Phase 1" if epoch < config.phase1_epochs else "Phase 2"
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", phase=phase)

    return autoencoder, jnp.array(losses)


def fill_ordering_gaps(
    autoencoder: Autoencoder,
    knnp_result: KNNPResult,
    prob_threshold: float = 0.5,
) -> AutoencoderResult:
    r"""Use trained autoencoder to fill gaps in kNN+p ordering.

    This function predicts $\gamma$ values for all tracers (including those
    skipped by kNN+p) and returns a complete ordering.

    Parameters
    ----------
    autoencoder : Autoencoder
        Trained autoencoder model.
    knnp_result : KNNPResult
        Result from nearest_neighbors_with_momentum.
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
    >>> from knnp import nearest_neighbors_with_momentum
    >>> from knnp.autoencoder import (
    ...     Autoencoder,
    ...     train_autoencoder,
    ...     fill_ordering_gaps,
    ... )

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)
    >>> autoencoder = Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
    >>> trained, _ = train_autoencoder(autoencoder, result)
    >>> full_ordering = fill_ordering_gaps(trained, result)

    """
    position = knnp_result["position"]
    velocity = knnp_result["velocity"]

    # Predict gamma and probability for all tracers
    gamma, prob = autoencoder.predict(position, velocity)

    # Sort by gamma to get ordering
    sorted_indices = jnp.argsort(gamma)

    # Filter by probability threshold
    high_prob_mask = prob[sorted_indices] >= prob_threshold
    filtered_indices = sorted_indices[high_prob_mask]

    return AutoencoderResult(
        gamma=gamma,
        membership_prob=prob,
        position=dict(position),
        velocity=dict(velocity),
        ordered_indices=tuple(int(i) for i in filtered_indices),
    )
