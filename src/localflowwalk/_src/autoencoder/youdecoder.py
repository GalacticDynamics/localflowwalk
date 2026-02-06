"""Simple autoencoder with user-provided decoder function."""

__all__: tuple[str, ...] = (
    "EncoderYouDecoder",
    "RunningMeanDecoder",
    "train_simple_autoencoder",
)

from collections.abc import Callable, Mapping
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import plum
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .normalize import AbstractNormalizer
from .order_net import (
    OrderingNet,
    OrderingTrainingConfig,
    train_ordering_net,
)
from localflowwalk._src.algorithm import LocalFlowWalkResult
from localflowwalk._src.custom_types import FSzN, VectorComponents

Gamma: TypeAlias = FSzN  # noqa: UP040


class RunningMeanDecoder(eqx.Module):
    """Running-mean decoder for non-parametric position reconstruction.

    This decoder stores the training data and reconstructs positions
    from gamma values using a windowed running mean.

    Attributes
    ----------
    gamma_train : Array
        Precomputed gamma values for all training samples.
    positions_train : Array
        Training positions (normalized).
    window_size : float
        Window size in gamma-space.

    """

    window_size: float
    gamma_train: Float[Array, " N"] | None = None
    positions_train: Float[Array, " N D"] | None = None

    def __call__(self, gamma: Float[Array, ""]) -> Float[Array, " D"]:
        """Decode a single gamma value to position using running mean.

        Parameters
        ----------
        gamma : Array, shape ()
            Single gamma value to decode.

        Returns
        -------
        position : Array, shape (D,)
            Reconstructed position.

        """
        gamma_train = eqx.error_if(
            self.gamma_train,
            self.gamma_train is None,
            "Decoder not initialized with training data.",
        )
        positions_train = eqx.error_if(
            self.positions_train,
            self.positions_train is None,
            "Decoder not initialized with training data.",
        )

        # Find samples within the window
        in_window = jnp.abs(gamma_train - gamma) < (self.window_size / 2)

        # Compute weighted mean (uniform weights within window)
        weights = in_window.astype(positions_train.dtype)
        total_weight = jnp.sum(weights) + 1e-10

        # Weighted mean position
        weighted_pos = jnp.sum(positions_train * weights[:, None], axis=0)
        return weighted_pos / total_weight

    @classmethod
    def make(
        cls,
        encoder: OrderingNet,
        normalizer: AbstractNormalizer,
        positions: Mapping[str, Array] | Float[Array, "N D"],
        velocities: Mapping[str, Array] | Float[Array, "N D"],
        /,
        *,
        window_size: float = 0.1,
    ) -> "RunningMeanDecoder":
        r"""Create a running-mean decoder for non-parametric position reconstruction.

        This classmethod constructs a decoder that reconstructs positions from
        $\gamma$ values using a windowed running mean in $\gamma$-space. For each
        query $\gamma$, the decoder:

        1. Predicts $\gamma$ values for all training samples using the encoder
        2. Finds samples whose $\gamma$ is within [$\gamma$ - window_size/2,
           $\gamma$ + window_size/2]
        3. Returns the mean position of those samples

        This provides a simple, non-parametric alternative to training a neural
        network decoder. It works well when the stream is smooth and well-sampled.

        Parameters
        ----------
        encoder : OrderingNet
            Trained encoder network that predicts $\gamma$ from phase-space.
        normalizer : AbstractNormalizer
            Normalizer used to preprocess the phase-space data.
        positions : Mapping[str, Array] or Array, shape (N, D)
            Training positions. If Mapping, will be transformed by normalizer.
            If Array, assumed to be already normalized.
        velocities : Mapping[str, Array] or Array, shape (N, D)
            Training velocities. Same format as positions.
        window_size : float, default=0.1
            Window size in $\gamma$-space for computing running mean.
            Smaller values give more localized (less smooth) reconstruction.

        Returns
        -------
        decoder : RunningMeanDecoder
            Decoder module mapping $\gamma$ to positions.
            The returned decoder is JIT-compatible and vmappable.

        Notes
        -----
        The running mean is computed using a uniform (rectangular) kernel:

        .. math::

            \hat{x}(\gamma) = \frac{\sum_i x_i \cdot \mathbb{1}_{|\gamma_i - \gamma| < w/2}}
                                   {\sum_i \mathbb{1}_{|\gamma_i - \gamma| < w/2}}

        where $w$ is the window size.

        The decoder stores the encoder and training data, so it can be used
        independently after creation.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from localflowwalk._src.autoencoder import (
        ...     OrderingNet,
        ...     StandardScalerNormalizer,
        ...     RunningMeanDecoder,
        ... )
        >>> # Create sample stream data
        >>> key = jr.key(0)
        >>> N = 100
        >>> t = jnp.linspace(0, 2 * jnp.pi, N)
        >>> positions = {"x": jnp.cos(t), "y": jnp.sin(t)}
        >>> velocities = {"x": -jnp.sin(t), "y": jnp.cos(t)}
        >>> # Create and "train" encoder (skipped for brevity)
        >>> normalizer = StandardScalerNormalizer(positions, velocities)
        >>> encoder = OrderingNet(in_size=4, width_size=32, depth=2, key=key)
        >>> # Create running-mean decoder
        >>> decoder = RunningMeanDecoder.make(
        ...     encoder, normalizer, positions, velocities, window_size=0.15
        ... )
        >>> # Use decoder to reconstruct positions for gamma values
        >>> gamma_test = jnp.array([-0.5, 0.0, 0.5])
        >>> reconstructed = jax.vmap(decoder)(gamma_test)
        >>> reconstructed.shape
        (3, 2)

        """
        # Preprocess the data
        if isinstance(positions, Mapping):
            qs_norm, ps_norm = normalizer.transform(positions, velocities)
        else:
            # Assume already normalized array form
            qs_norm = positions
            ps_norm = velocities

        # Concatenate into phase-space vectors
        ws_norm = jnp.concatenate([qs_norm, ps_norm], axis=1)

        # Predict gamma for all training samples (only need gamma, not prob)
        gamma_train, _ = jax.vmap(encoder)(ws_norm)

        # Create the decoder module
        return cls(
            gamma_train=gamma_train,
            positions_train=qs_norm,
            window_size=window_size,
        )


class EncoderYouDecoder(eqx.Module):
    r"""Autoencoder with trained encoder and user-provided decoder function.

    This class provides a flexible alternative to `PathAutoencoder` where the
    decoder is a user-provided function rather than a trained neural network.
    The most common use case is with a running-mean decoder that performs
    non-parametric interpolation in $\gamma$-space.

    The encoder (OrderingNet) is trained to predict $\gamma$ and membership
    probability $p$ from phase-space coordinates. The decoder function maps
    $\gamma$ values back to positions.

    Attributes
    ----------
    encoder : OrderingNet
        Neural network mapping $(x, v) \to (\gamma, p)$.
    decoder : Callable[[Array], Array]
        Function mapping $\gamma \to x$. Should be vmappable and JIT-compatible.
    normalizer : AbstractNormalizer
        Data normalizer for preprocessing phase-space coordinates.

    See Also
    --------
    PathAutoencoder : Full autoencoder with trainable decoder network.
    RunningMeanDecoder.make : Create a running-mean decoder.
    train_simple_autoencoder : Train the encoder and create decoder.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from localflowwalk._src.autoencoder import (
    ...     EncoderYouDecoder,
    ...     RunningMeanDecoder,
    ...     train_simple_autoencoder,
    ...     StandardScalerNormalizer,
    ...     OrderingNet,
    ... )
    >>> # Create sample data
    >>> key = jr.key(0)
    >>> positions = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
    >>> velocities = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
    >>> ordering = jnp.array([0, 1, 2])
    >>> # Create normalizer and encoder
    >>> normalizer = StandardScalerNormalizer(positions, velocities)
    >>> encoder = OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(1))
    >>> # Create decoder function
    >>> decoder = RunningMeanDecoder.make(
    ...     encoder, normalizer, positions, velocities, window_size=0.2
    ... )
    >>> # Create model
    >>> model = EncoderYouDecoder(
    ...     encoder=encoder, decoder=decoder, normalizer=normalizer
    ... )

    """

    encoder: OrderingNet
    decoder: Callable[[Float[Array, ""]], Float[Array, " D"]]
    normalizer: AbstractNormalizer

    def encode(
        self,
        qs: VectorComponents,
        ps: VectorComponents,
        /,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Gamma, FSzN]:
        r"""Encode phase-space coordinates to ($\gamma$, $p$).

        Parameters
        ----------
        qs, ps : VectorComponents
            Spatial / velocity coordinates.
        key : PRNGKeyArray, optional
            JAX random key for stochastic encoding (if applicable).

        Returns
        -------
        gamma : Array, shape (N,)
            Ordering parameter in [-1, 1].
        prob : Array, shape (N,)
            Membership probability in [0, 1].

        """
        qs_norm, ps_norm = self.normalizer.transform(qs, ps)
        ws_norm = jnp.concatenate([qs_norm, ps_norm], axis=1)
        out = jax.vmap(lambda w: self.encoder(w, key=key))(ws_norm)
        return out  # noqa: RET504

    def decode(
        self,
        gamma: Gamma,
        /,
        *,
        key: PRNGKeyArray | None = None,  # noqa: ARG002
    ) -> Float[Array, "N D"]:
        r"""Decode $\gamma$ to reconstructed position.

        Parameters
        ----------
        gamma : Array, shape (N,)
            Ordering parameter in [-1, 1].
        key : PRNGKeyArray, optional
            JAX random key (unused, included for API compatibility).

        Returns
        -------
        position : Array, shape (N, n_dims)
            Reconstructed positions.

        """
        return self.decoder(gamma)


@plum.dispatch
def train_autoencoder(
    model: EncoderYouDecoder,
    all_ws: Float[Array, " N TwoF"],
    ordering_indices: Int[Array, " N"],
    /,
    *,
    config: OrderingTrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[
    EncoderYouDecoder, dict[str, optax.OptState], Float[Array, " {config.n_epochs}"]
]:
    """Train the EncoderYouDecoder encoder and create running-mean decoder.

    This function provides a simplified training workflow:

    1. Train the encoder (OrderingNet) using supervised learning from ordering indices
    2. Create a running-mean decoder using the trained encoder and training data

    Unlike `train_autoencoder` for `PathAutoencoder`, this does not train a decoder
    network. Instead, it uses the provided (or default) decoder function.

    Parameters
    ----------
    model : EncoderYouDecoder
        The autoencoder model to train. Its encoder will be updated.
    all_ws : Array, shape (N, 2*n_dims)
        All phase-space coordinates (positions + velocities) in normalized form.
    ordering_indices : Array, shape (N,)
        Ordering indices from walk algorithm. Valid indices (>= 0) indicate
        ordered tracers; -1 indicates skipped/unordered tracers.
    config : OrderingTrainingConfig, optional
        Training configuration for the encoder. If None, uses default config.
    decoder_kwargs : Mapping, optional
        Keyword arguments passed to decoder function creation. For running-mean
        decoder, can include 'window_size'. If None, uses defaults.
    key : PRNGKeyArray
        Random key for training.

    Returns
    -------
    model : EncoderYouDecoder
        Trained autoencoder with updated encoder and decoder.
    opt_state : optax.OptState
        Final optimizer state from encoder training.
    losses : Array, shape (n_epochs,)
        Training losses from encoder training.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from localflowwalk._src.autoencoder import (
    ...     EncoderYouDecoder,
    ...     OrderingNet,
    ...     StandardScalerNormalizer,
    ...     RunningMeanDecoder,
    ...     train_simple_autoencoder,
    ...     OrderingTrainingConfig,
    ... )
    >>> # Create sample data
    >>> key = jr.key(0)
    >>> N = 50
    >>> positions = {"x": jnp.arange(N, dtype=float), "y": jnp.zeros(N)}
    >>> velocities = {"x": jnp.ones(N), "y": jnp.zeros(N)}
    >>> ordering = jnp.arange(N)
    >>> # Setup model
    >>> normalizer = StandardScalerNormalizer(positions, velocities)
    >>> encoder = OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(1))
    >>> decoder = RunningMeanDecoder.make(encoder, normalizer, positions, velocities)
    >>> model = EncoderYouDecoder(encoder, decoder, normalizer)
    >>> # Prepare normalized data
    >>> qs_norm, ps_norm = normalizer.transform(positions, velocities)
    >>> ws_norm = jnp.concatenate([qs_norm, ps_norm], axis=1)
    >>> # Train (with minimal epochs for demonstration)
    >>> config = OrderingTrainingConfig(n_epochs=10, batch_size=16)
    >>> trained_model, opt_state, losses = train_simple_autoencoder(
    ...     model, ws_norm, ordering, config=config, key=jr.key(2)
    ... )
    >>> losses.shape
    (10,)

    """
    # Use default config if none provided
    if config is None:
        config = OrderingTrainingConfig()

    # Train the encoder
    encoder, opt_state, losses = train_ordering_net(
        model.encoder, all_ws, ordering_indices, config=config, key=key
    )

    # Update encoder in model
    model = eqx.tree_at(lambda m: m.encoder, model, encoder)

    # Recreate decoder with trained encoder
    # Extract positions and velocities from ws
    D = all_ws.shape[1] // 2
    qs_norm = all_ws[:, :D]
    ps_norm = all_ws[:, D:]

    # Update decoder with trained encoder
    decoder = RunningMeanDecoder.make(
        encoder,
        model.normalizer,
        qs_norm,
        ps_norm,
        window_size=model.decoder.window_size,
    )

    # Update decoder in model
    model = eqx.tree_at(lambda m: m.decoder, model, decoder)

    return model, opt_state, losses


@plum.dispatch
def train_autoencoder(  # noqa: ANN202
    model: EncoderYouDecoder,
    walk_results: LocalFlowWalkResult,
    /,
    *,
    config: OrderingTrainingConfig | None = None,
    decoder_kwargs: Mapping[str, object] | None = None,
    key: PRNGKeyArray,
):
    """Train EncoderYouDecoder from LocalFlowWalkResult.

    Convenience overload that extracts phase-space data from walk results.

    Parameters
    ----------
    model : EncoderYouDecoder
        The autoencoder model to train.
    walk_results : LocalFlowWalkResult
        Results from the local flow walk algorithm containing positions,
        velocities, and ordering indices.
    config : OrderingTrainingConfig, optional
        Training configuration. If None, uses default.
    decoder_kwargs : Mapping, optional
        Keyword arguments for decoder creation. If None, uses defaults.
    key : PRNGKeyArray
        Random key for training.

    Returns
    -------
    model : EncoderYouDecoder
        Trained autoencoder.
    opt_state : optax.OptState
        Final optimizer state.
    losses : Array, shape (n_epochs,)
        Training losses.

    """
    # Transform data using normalizer
    qs, ps = model.normalizer.transform(walk_results.positions, walk_results.velocities)
    ws = jnp.concatenate([qs, ps], axis=1)

    return train_simple_autoencoder(
        model,
        ws,
        walk_results.indices,
        config=config,
        decoder_kwargs=decoder_kwargs,
        key=key,
    )
