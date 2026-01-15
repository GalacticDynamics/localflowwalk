r"""Interpolation Network for interpolating skipped tracers."""

__all__: tuple[str, ...] = ("TrackNet", "decoder_loss", "TrackTrainingConfig")


import functools as ft
from collections.abc import Mapping
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real

from .order_net import default_optimizer, masked_mean
from .scanmlp import ScanOverMLP
from localflowwalk._src.custom_types import FSz0, RSz0, RSzN


class TrackNet(eqx.Module):
    r"""Param-Net (decoder): maps $\gamma \to$ position (x, y, z).

    This network reconstructs the stream track position from the ordering
    parameter $\gamma$. It serves as the second half of the autoencoder.

    The architecture follows Appendix B.1 of Nibauer et al. (2022).

    Uses scan-over-layers for improved compilation speed. See:
    https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key for initialization.
    out_size : int
        Number of spatial dimensions (2 for 2D, 3 for 3D) for the track
        speed.
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    mlp: ScanOverMLP

    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        out_size: int = 3,
        width_size: int = 100,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        # Store static information
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth

        self.mlp = ScanOverMLP(
            in_size="scalar",
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    @ft.partial(eqx.filter_jit)
    def __call__(
        self, gamma: RSz0, /, *, key: PRNGKeyArray | None = None
    ) -> tuple[Float[Array, " {self.out_size}"], FSz0]:
        """Forward pass through Param-Net.

        Parameters
        ----------
        gamma : RSz0
            Ordering parameter in [-1, 1], shape (...).
        key : PRNGKeyArray | None
            Optional key.

        Returns
        -------
        position : Array
            Reconstructed position of shape (..., out_size).

        """
        return self.mlp(gamma, key=key)


# ===================================================================


@eqx.filter_jit
def decoder_loss(
    qs_meas: Real[Array, "N D"],
    weights: RSzN,
    qs_pred: Real[Array, "N D"],
    t_hat: RSzN,
    p_hat: RSzN,
    mask: Bool[Array, " N"],
    *,
    lambda_q: float = 100,
    lambda_p: float = 100,
) -> RSz0:
    r"""Loss for Param-Net decoder: position + momentum direction matching.

    Computes the reconstruction loss as defined in Appendix A.3 of Nibauer
    et al. (2022). The decoder reconstructs the stream track position and
    momentum direction (tangent vectors) from the ordering parameter $gamma$.

    The loss has two components:

    1. **Position MSE**: Squared error between true and predicted phase-space
       coordinates (position + velocity).
    2. **Momentum direction**: Squared error between true and predicted
       momentum direction (tangent vectors T and T_θ).

    The loss is weighted by element-wise weights to handle variable batch
    sizes with padding (only real data contributes).

    Parameters
    ----------
    qs_meas : Array, shape (N, D)
        True coordinate samples for the batch.
    weights : Array, shape (N,)
        Binary or continuous weights for each sample. Real data receives
        weight 1, padded samples receive weight 0.
    qs_pred : Array, shape (N, D)
        Predicted coordinates from Param-Net decoder.
    t_hat : Array, shape (N,)
        True momentum direction unit vectors (tangent vectors) for each sample.
    p_hat : Array, shape (N,)
        Predicted momentum direction unit vectors from Param-Net decoder.
    mask : Array, shape (N,)
        Mask for which values should be used, and which should be ignored.
    lambda_q : float, optional
        Weight for position reconstruction loss. Default: 1.0.
    lambda_p : float, optional
        Weight for momentum direction loss. Default: 100.0.

    Returns
    -------
    loss : Array, shape ()
        Scalar loss value.

    Notes
    -----
    The mathematical form is:

    $$ \\ell_\\theta(\\theta) = \\sum_{n=1}^N w_n \\left[
        \\lambda_q \\|x_n - x_\\theta(\\gamma_\\theta(x_n, v_n))\\|^2
        + \\lambda_p \\|T_n - T_\\theta(\\gamma_\\theta(x_n, v_n))\\|^2
    \\right] $$

    where:
    - $w_n$ are the sample weights
    - $x_n$ are true coordinates, $x_\\theta$ are predicted
    - $T_n$ are true tangent vectors, $T_\\theta$ are predicted
    - $\\lambda_q$ and $\\lambda_p$ control relative importance

    References
    ----------
    Nibauer et al. (2022), Appendix A.3: Decoder loss formulation for
    reconstructing stream track position and momentum direction.

    """
    # Spatial component
    sq_spatial_dist = jnp.sum(jnp.square(qs_meas - qs_pred), axis=1)
    spatial_l2 = masked_mean(weights * sq_spatial_dist, mask)

    # Velocity Alignment
    # TODO: replace this with the metric information
    sq_tangent_dist = jnp.sum(jnp.square(t_hat - p_hat), axis=1)
    tangent_l2 = masked_mean(weights * sq_tangent_dist, mask)

    # Full loss
    prefactor = 1 / (lambda_q + lambda_p)
    return prefactor * (lambda_q * spatial_l2 + lambda_p * tangent_l2)


# ===================================================================


@dataclass
class TrackTrainingConfig:
    r"""Configuration for Phase 2 (TrackNet/Decoder) training.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optax optimizer for training. Default: AdamW with lr=1e-3,
        weight_decay=1e-7.
    n_epochs : int
        Number of training epochs. Default: 1000.
    batch_size : int
        Batch size for training. Default: 100.
    lambda_q : float
        Weight for spatial reconstruction loss. Default: 1.0.
    lambda_p : tuple[float, float]
        Weight range (min, max) for velocity alignment loss.  Linearly ramped
        from min to max over n_epochs. Default: (1.0, 1.0).
    member_threshold : float
        Membership probability threshold for filtering stream members.  Points
        with probability below this threshold are excluded from loss.
    freeze_encoder : bool
        Whether to freeze the encoder during training. Default: False.
    weight_by_density : bool or Mapping, optional
        If False: use uniform weights (default).  If True: use KDE inverse
        density weighting with default bandwidth.  If Mapping: use KDE with
        custom bandwidth via `**weight_by_density`.
    show_pbar : bool
        Whether to show an epoch progress bar via tqdm. Default: True.

    """

    optimizer: optax.GradientTransformation = default_optimizer
    """Optax optimizer for training."""

    n_epochs: int = 300
    """Number of epochs for training."""

    batch_size: int = 100
    """Batch size for training."""

    lambda_q: float = 1.0
    """Weight for spatial reconstruction loss."""

    lambda_p: tuple[float, float] = (1.0, 1.0)
    """Weight schedule (start, stop) for velocity alignment loss."""

    member_threshold: float = 0.5
    """Membership p > threshold for identifying stream members."""

    freeze_encoder: bool = False
    """Whether to freeze the encoder during phase 2 training."""

    weight_by_density: bool | Mapping[str, object] = False
    """Whether to inverse density weight the samples. USE WITH CARE."""

    show_pbar: bool = True
    """Show an epoch progress bar via `tqdm`."""
