"""Normalizers for autoencoder data preprocessing."""

__all__: tuple[str, ...] = ("AbstractNormalizer", "StandardScalerNormalizer")

import abc

import jax.numpy as jnp
import plum
from jaxtyping import Array, Float

from localflowwalk._src.custom_types import VectorComponents


class AbstractNormalizer(metaclass=abc.ABCMeta):
    """Abstract base class for normalizers used in autoencoders."""

    @abc.abstractmethod
    def __init__(
        self, qs: VectorComponents, ps: VectorComponents, **kwargs: object
    ) -> None:
        """Initialize the normalizer with data to compute statistics if needed."""

    @abc.abstractmethod
    def transform(
        self, qs: VectorComponents, ps: VectorComponents
    ) -> tuple[Float[Array, "N D"], Float[Array, "N D"]]:
        """Transform the data using the fitted normalizer."""

    @abc.abstractmethod
    def inverse_transform(
        self, qs: Float[Array, "N D"], ps: Float[Array, "N D"]
    ) -> tuple[VectorComponents, VectorComponents]:
        """Inverse transform the data back to original space."""

    @property
    @abc.abstractmethod
    def n_spatial_dims(self) -> int:
        """Number of spatial dimensions the normalizer was fitted on."""


class StandardScalerNormalizer(AbstractNormalizer):
    """A normalizer that standardizes data to zero mean and unit variance."""

    @plum.dispatch
    def __init__(
        self: "StandardScalerNormalizer",
        qs: VectorComponents,
        ps: VectorComponents,
        /,
        **_: object,
    ) -> None:
        self.q_comps = list(qs.keys())
        xs = jnp.stack(list(qs.values()), axis=1)
        self.q_mean = jnp.mean(xs, axis=0)
        self.q_std = jnp.std(xs, axis=0) + 1e-8  # Prevent division by zero

        self.p_comps = list(ps.keys())
        vs = jnp.stack(list(ps.values()), axis=1)
        self.p_mean = jnp.mean(vs, axis=0)
        self.p_std = jnp.std(vs, axis=0) + 1e-8  # Prevent division by zero

    @plum.dispatch
    def transform(
        self, qs: VectorComponents, ps: VectorComponents
    ) -> tuple[Float[Array, "N D"], Float[Array, "N D"]]:
        out_qs = jnp.stack([qs[k] for k in self.q_comps], axis=1)
        out_qs = (out_qs - self.q_mean) / self.q_std

        out_ps = jnp.stack([ps[k] for k in self.p_comps], axis=1)
        out_ps = (out_ps - self.p_mean) / self.p_std

        return out_qs, out_ps

    def inverse_transform(
        self, qs: Float[Array, "N D"], ps: Float[Array, "N D"]
    ) -> tuple[VectorComponents, VectorComponents]:
        qs = (qs * self.q_std) + self.q_mean
        out_qs = {k: qs[:, i] for i, k in enumerate(self.q_comps)}

        ps = (ps * self.p_std) + self.p_mean
        out_ps = {k: ps[:, i] for i, k in enumerate(self.p_comps)}

        return out_qs, out_ps

    @property
    def n_spatial_dims(self) -> int:
        """Number of spatial dimensions the normalizer was fitted on."""
        return len(self.q_comps)
