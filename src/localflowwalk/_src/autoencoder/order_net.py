r"""Ordering Network for interpolating skipped tracers."""

__all__: tuple[str, ...] = (
    # Network components
    "OrderingNet",
    # Training functions
    "train_ordering_net",
    "make_step",
    "shuffle_and_batch",
    "OrderingTrainingConfig",
    # Loss functions
    "encoder_loss",
    "compute_loss",
)

import functools as ft
from dataclasses import dataclass
from typing import ClassVar, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_tqdm
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .scanmlp import ScanOverMLP
from .utils import shuffle_and_batch
from localflowwalk._src.custom_types import FSz0, FSzN


class OrderingNet(eqx.Module):
    r"""Interpolation network:$(x, v) \;\mapsto\; (\gamma, p),$.

    This network takes N-D phase-space coordinates and outputs:

    - $\gamma \in [-1, 1]$: The ordering parameter along the stream
    - $p \in [0, 1]$: The membership probability (1 = likely stream member)

    The architecture follows Appendix B.3 of Nibauer et al. (2022).

    Uses scan-over-layers for improved compilation speed. See:
    https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers

    Parameters
    ----------
    in_size : int
        Number of spatial + kinematic dimensions (6 for 3D: x, y, z, vx, vy,
        vz).
    width_size : int
        The size of each hidden layer.
    depth : int, optional
        The number of hidden layers, not include the input layer or output
        heads.  For example, `depth=2` results in an network with layers:

        ``[Linear(in_size, width_size), Linear(width_size, width_size),
        Linear(width_size, out_size), (output_heads)]``

    key : PRNGKeyArray
        JAX random key for initialization.

    """

    mlp: ScanOverMLP
    gamma_head: eqx.nn.Linear
    prob_head: eqx.nn.Linear

    in_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    __citation__: ClassVar[str] = (
        "https://ui.adsabs.harvard.edu/abs/2022ApJ...940...22N/abstract"
    )

    def __init__(
        self,
        in_size: int = 6,
        width_size: int = 100,
        depth: int = 2,
        *,
        key: PRNGKeyArray,
    ) -> None:
        # Store parameters
        self.in_size = in_size
        self.width_size = width_size
        self.depth = depth

        keys = jr.split(key, 3)

        # MLP backbone: input -> hidden layers -> (hidden_size,) The output of
        # this MLP has size hidden_size, which feeds to both output heads
        self.mlp = ScanOverMLP(
            in_size=in_size,
            width_size=width_size,
            out_size=width_size,  # Output fed to the two heads
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=jax.nn.tanh,
            use_bias=True,
            use_final_bias=True,
            key=keys[0],
        )

        # Output heads: hidden_size -> 1 (to be squeezed)
        self.gamma_head = eqx.nn.Linear(width_size, 1, key=keys[1])
        self.prob_head = eqx.nn.Linear(width_size, 1, key=keys[2])

    @ft.partial(eqx.filter_jit)
    def __call__(
        self, ws: Float[Array, " {self.n_dims}"], /, *, key: PRNGKeyArray | None = None
    ) -> tuple[FSz0, FSz0]:
        """Forward pass through the interpolation network.

        Parameters
        ----------
        ws : Array
            Phase-space coordinates of shape (..., 2*n_dims).
            For each point: [x, y, z, vx, vy, vz] (or 2D equivalent).
        key : PRNGKeyArray | None
            Optional key.

        Returns
        -------
        gamma : Array
            Ordering parameter in [-1, 1], shape (...).
        prob : Array
            Membership probability in [0, 1], shape (...).

        """
        # MLP backbone
        x = self.mlp(ws, key=key)

        # Output heads with appropriate activations
        gamma = jax.nn.tanh(self.gamma_head(x)).squeeze(-1)
        prob = jax.nn.sigmoid(self.prob_head(x)).squeeze(-1)

        return gamma, prob


# ===================================================================


def masked_mean(arr: Float[Array, " N"], mask: Bool[Array, " N"]) -> FSz0:
    r"""Compute the mean of an array over only the masked elements.

    Parameters
    ----------
    arr : Array, shape (N,)
        Input array.
    mask : Array, shape (N,)
        Binary mask where True = include in mean, False = exclude.

    Returns
    -------
    mean : Array
        Scalar mean value over masked elements.

    """
    n_real = jnp.sum(mask)
    return jnp.sum(arr * mask) / n_real


@eqx.filter_jit
def encoder_loss(
    gamma_true: FSzN,
    gamma_pred: FSzN,
    prob_pred_ordered: FSzN,
    prob_pred_random: FSzN,
    mask: Bool[Array, " N"],
    *,
    lambda_prob: float = 1.0,
) -> FSz0:
    r"""Compute loss for interpolation network training.

    Parameters
    ----------
    gamma_true : Array, shape (N,)
        True $\gamma$ values for ordered stream tracers.
    gamma_pred : Array, shape (N,)
        Predicted $\gamma$ values from the interpolation network.
    prob_pred_ordered : Array, shape (N,)
        Predicted membership probabilities for ordered stream tracers.
        Should be pushed toward 1.
    prob_pred_random : Array, shape (M,)
        Predicted membership probabilities for random phase-space samples.
        Should be pushed toward 0.
    mask : Array, shape (N,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    lambda_prob : float, optional
        Weight for probability penalties. Default: 1.0.

    Returns
    -------
    loss : Array
        Scalar loss value.

    """
    # Compute per-element losses
    gamma_sq_error = jnp.square(gamma_true - gamma_pred)
    prob_ordered_sq_error = jnp.square(1.0 - prob_pred_ordered)
    prob_random_sq_error = jnp.square(prob_pred_random)

    # Compute masked mean only over real data
    gamma_loss = masked_mean(gamma_sq_error, mask)
    prob_ordered_penalty = masked_mean(prob_ordered_sq_error, mask)
    prob_random_penalty = masked_mean(prob_random_sq_error, mask)

    return gamma_loss + lambda_prob * (prob_ordered_penalty + prob_random_penalty)


@eqx.filter_value_and_grad
def compute_loss(
    model: OrderingNet,
    batch_ws: Float[Array, " B TwoF"],
    batch_gamma: Float[Array, " B"],
    rand_ws: Float[Array, "B TwoF"],
    batch_mask: Bool[Array, " B"],
    lambda_prob: float = 1.0,
) -> FSz0:
    r"""Compute interpolation network loss with gradients.

    This function is decorated with ``@eqx.filter_value_and_grad`` to compute
    both the loss value and gradients with respect to the model parameters in
    a single pass. This is the recommended pattern for low-overhead training
    loops in Equinox.

    Parameters
    ----------
    model : OrderingNet
        The interpolation network being trained.
    batch_ws : Array, shape (B, 2*n_dims)
        Batch of ordered phase-space coordinates from stream tracers.
    batch_gamma : Array, shape (B,)
        Target $\gamma$ values for the ordered stream tracers.
    rand_ws : Array, shape (B, 2*n_dims)
        Batch of random phase-space samples (not on stream).
    batch_mask : Array, shape (B,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    lambda_prob : float, optional
        Weight for probability loss terms. Default: 1.0.

    Returns
    -------
    loss : Array
        Scalar loss value.

    Notes
    -----
    Due to the ``@eqx.filter_value_and_grad`` decorator, calling this function
    returns a tuple ``(loss, grads)`` where ``grads`` contains gradients with
    respect to the trainable parameters of ``model``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from localflowwalk.nn import OrderingNet

    >>> key = jax.random.key(0)
    >>> net = OrderingNet(in_size=4, key=key)
    >>> batch_ws = jax.random.normal(key, (10, 4))
    >>> batch_gamma = jnp.linspace(-0.5, 0.5, 10)
    >>> rand_ws = jax.random.normal(key, (10, 4))
    >>> batch_mask = jnp.ones(10, dtype=bool)

    >>> loss, grads = compute_loss(
    ...     net, batch_ws, batch_gamma, rand_ws, batch_mask, lambda_prob=1.0
    ... )

    """
    # TODO: the models accept (but don't currently use) a key. It would be safer
    # to pass a key.
    # Predictions on ordered stream tracers (vectorized over batch)
    gamma_pred, prob_pred_ordered = jax.vmap(model)(batch_ws)

    # Predictions on random phase-space samples (vectorized over batch)
    _, prob_pred_random = jax.vmap(model)(rand_ws)

    # Compute loss using the loss function
    return encoder_loss(
        gamma_true=batch_gamma,
        gamma_pred=gamma_pred,
        prob_pred_ordered=prob_pred_ordered,
        prob_pred_random=prob_pred_random,
        mask=batch_mask,
        lambda_prob=lambda_prob,
    )


# TODO: https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops
@eqx.filter_jit
def make_step(
    model_dynamic: OrderingNet,
    model_static: OrderingNet,
    /,
    ord_ws: Float[Array, "B 2D"],
    ord_gamma: Float[Array, " B"],
    rand_ws: Float[Array, "B 2D"],
    mask: Bool[Array, " B"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    *,
    lambda_prob: float,
) -> tuple[FSz0, OrderingNet, optax.OptState]:
    r"""Run a single optimization step for the interpolation network.

    Parameters
    ----------
    model_dynamic, model_static : OrderingNet
        The dynamic and static components of the ordering network being trained.
    ord_ws : Array, shape (B, 2*n_dims)
        Batch of ordered phase-space coordinates from stream tracers.
    ord_gamma : Array, shape (B,)
        Target $\gamma$ values for the ordered stream tracers.
    rand_ws : Array, shape (B, 2*n_dims)
        Batch of random phase-space samples (not on stream).
    mask : Array, shape (B,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    opt_state : optax.OptState
        Optimizer state.
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    lambda_prob : float, optional
        Weight for probability loss terms. Default: 1.0.

    Returns
    -------
    loss : Array
        Scalar loss value.
    model : OrderingNet
        Updated model after applying gradients.
    opt_state : optax.OptState
        Updated optimizer state.

    """
    # Reconstruct full model from dynamic and static parts
    model = eqx.combine(model_dynamic, model_static)

    # TODO: the models accept (but don't currently use) a key. It would be safer
    # to pass a key.
    loss, grads = compute_loss(
        model, ord_ws, ord_gamma, rand_ws, mask, lambda_prob=lambda_prob
    )
    # Update the dynamic components of the model
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = cast("OrderingNet", eqx.apply_updates(model_dynamic, updates))
    return loss, model_dynamic, opt_state


default_optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-7)


@dataclass
class OrderingTrainingConfig:
    r"""Configuration for interpolation network training.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optax optimizer for training. Default: AdamW with lr=1e-3, weight_decay=1e-7.
    n_epochs : int
        Number of training epochs. Default: 3000.
    batch_size : int
        Batch size for training. Default: 100.
    lambda_prob : float
        Weight for probability loss terms. Default: 1.0.
    gamma_min : float
        Minimum $\gamma$ value for ordered tracers. Default: -0.75.
    gamma_max : float
        Maximum $\gamma$ value for ordered tracers. Default: 0.75.
    show_pbar : bool
        Whether to show an epoch progress bar via tqdm. Default: True.

    """

    optimizer: optax.GradientTransformation = default_optimizer

    n_epochs: int = 800
    """Number of training epochs."""

    batch_size: int = 100
    """Batch size for training."""

    lambda_prob: float = 1.0
    """Weight for probability loss terms."""

    gamma_range: tuple[float, float] = (-0.75, 0.75)
    r"""Minimum/maximum $\gamma$ value for ordered tracers."""

    show_pbar: bool = True
    """Show an epoch progress bar via tqdm."""


BatchScanCarry: TypeAlias = tuple[eqx.Module, optax.OptState]  # noqa: UP040
BatchScanInputs: TypeAlias = tuple[  # noqa: UP040
    Bool[Array, " B"],  # batch_mask: True where batch has data, False where padded
    Float[Array, "B 2D"],  # batch_ws: ordered phase-space samples
    Float[Array, " B"],  # batch_gamma: target ordering parameter for batch_ws
    Float[Array, "B 2D"],  # batch_rand_ws: random phase-space samples
]


# def linear_map_from_ordering(ordering, vmin, vmax, *, fill=0):
#     ordering = jnp.asarray(ordering)
#     # Mask for valid (ordered) entries
#     mask = ordering >= 0
#     # Maximum ordering index (assumes at least one valid entry)
#     kmax = ordering.max()
#     # Linear map: [0, kmax] -> [vmin, vmax]
#     mapped = vmin + (vmax - vmin) * (ordering / kmax)
#     # Fill unordered entries
#     return jnp.where(mask, mapped, fill)


def train_ordering_net(
    model: OrderingNet,
    all_ws: Float[Array, "N TwoF"],
    ordering_indices: Int[Array, "N"],
    /,
    config: OrderingTrainingConfig,
    *,
    key: PRNGKeyArray,
) -> tuple[OrderingNet, optax.OptState, Float[Array, "E"]]:
    r"""Train the interpolation network on ordered stream tracers.

    This implementation uses lax.scan for efficient batching and supports Orbax
    checkpointing. The training follows the pattern from the original autoencoder
    for maximum performance.

    Parameters
    ----------
    model : OrderingNet
        The interpolation network to train.
    all_ws : Array, shape (N, 2*n_dims)
        Phase-space coordinates from the walk algorithm.
    ordering_indices : Array, shape (N,)
        Indices representing the ordering of tracers.
        Unvisited tracers have indices of -1.
    config : OrderingTrainingConfig
        Training configuration.
    key : PRNGKeyArray
        Random key for shuffling and random sampling.

    Returns
    -------
    trained_net : OrderingNet
        The trained interpolation network.
    opt_state : optax.OptState
        The optimizer state after training.
    losses : Array, shape (n_epochs,)
        Training loss per epoch.

    """
    # TODO: not need to slice. Instead use a mask or something.
    # But changing this goes from ~800 epochs/s to ~30 epochs/s.
    is_ordered = ordering_indices >= 0
    ordered_ws = all_ws[ordering_indices][is_ordered]
    # ordered_ws = all_ws[is_ordered]

    # Shapes
    shape = ordered_ws.shape
    n_total = shape[0]

    # Compute phase-space bounds for random sampling
    ws_min = jnp.min(jnp.where(is_ordered[:, None], all_ws, jnp.inf), axis=0)
    ws_max = jnp.max(jnp.where(is_ordered[:, None], all_ws, -jnp.inf), axis=0)

    # Initialize gamma targets: uniform spacing in [gamma_min, gamma_max]
    gamma_target = jnp.linspace(*config.gamma_range, n_total)
    # # TODO: why is gamma_target.min() != config.gamma_range[0] ?
    # gamma_target = linear_map_from_ordering(
    #     jnp.arange(len(all_ws))[ordering_indices], *config.gamma_range, fill=0
    # )[is_ordered]

    # Model surgery: partition out static components of the model
    filter_spec = eqx.is_array
    model_dynamic, model_static = eqx.partition(model, filter_spec)

    # Optimizer setup
    optimizer = config.optimizer
    opt_state = optimizer.init(model_dynamic)

    # ----------------------------------------
    # Epoch Scan Function (per-epoch scan)

    batch_size = config.batch_size
    ordered_mask = jnp.ones(n_total, dtype=bool)  # True since using ordered_ws

    def epoch_scan_fn(
        carry: BatchScanCarry, epoch_idx: int
    ) -> tuple[BatchScanCarry, FSz0]:
        """Run one scanned epoch (shuffle, batch, and train)."""
        # Unpack the carry
        model_dyn, opt_state = carry

        # Derive epoch-specific key from root key using fold_in
        epoch_key = jr.fold_in(key, epoch_idx)
        keys = jr.split(epoch_key)

        # Make random ws within bounds of ordered_ws
        random_ws = jr.uniform(keys[0], shape=shape, minval=ws_min, maxval=ws_max)

        # Shuffle and batch data
        b_mask, (b_ordered_ws, b_gamma, b_random_ws) = shuffle_and_batch(
            ordered_mask,
            ordered_ws,
            gamma_target,
            random_ws,
            batch_size=batch_size,
            key=keys[1],
        )

        # Scan over batches
        (model_dyn, opt_state), batch_losses = jax.lax.scan(
            cond_batch_scan_fn,
            (model_dyn, opt_state),
            (b_mask, b_ordered_ws, b_gamma, b_random_ws),
        )

        # Use mean loss across all batches for this epoch
        avg_loss = jnp.mean(batch_losses)
        return (model_dyn, opt_state), avg_loss

    # ----------------------------------------
    # Conditionally Run Batch Scan Function

    def cond_batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Run scanned batch step if there's data."""
        mask = inputs[0]
        return jax.lax.cond(
            jnp.any(mask), batch_scan_fn, null_batch_scan_fn, carry, inputs
        )

    def null_batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Don't run scanned batch step."""
        loss = jnp.array(0, dtype=jnp.result_type(*inputs[1:]))
        return carry, loss

    # ----------------------------------------
    # Batch Scan Function (per-batch scan)

    lambda_prob = config.lambda_prob

    def batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Run one scanned batch step.

        Notes
        -----
        Uses a partitioned model to keep the scan carry as arrays-only where
        possible, then re-combines with static structure for each step.

        """
        model_dyn, opt_state = carry
        mask, ord_ws, ord_gamma, rand_ws = inputs

        # Single training step for this batch
        loss, model_dyn, opt_state = make_step(
            model_dyn,
            model_static,
            ord_ws=ord_ws,
            ord_gamma=ord_gamma,
            rand_ws=rand_ws,
            mask=mask,
            opt_state=opt_state,
            optimizer=optimizer,
            lambda_prob=lambda_prob,
        )

        return (model_dyn, opt_state), loss

    # Optionally wrap the epoch scan with a progress bar
    if config.show_pbar:
        epoch_scan_wrapped = jax_tqdm.scan_tqdm(
            config.n_epochs, desc="Training", unit="epoch", dynamic_ncols=True
        )(epoch_scan_fn)
    else:
        epoch_scan_wrapped = epoch_scan_fn

    # Prepare epoch indices and run scan over epochs, which scans over batches
    epoch_indices = jnp.arange(config.n_epochs)
    (model_dynamic, opt_state), epoch_losses = jax.lax.scan(
        epoch_scan_wrapped, (model_dynamic, opt_state), epoch_indices
    )

    # Reconstruct model
    model = cast("OrderingNet", eqx.combine(model_dynamic, model_static))

    return model, opt_state, epoch_losses
