"""Interop with unxt Quantity for phasespace functions.

This module registers plum dispatch overloads for phasespace functions to work
with unxt Quantity-valued component dictionaries. This enables seamless use of
physical units throughout phase-space calculations.

When unxt is installed, these dispatches automatically handle:
- Distance calculations preserving units
- Direction vectors (unitless by nature)
- Velocity norms with proper unit handling
- Cosine similarity normalized correctly

Examples
--------
>>> import jax.numpy as jnp
>>> import unxt as u
>>> from knnp._src.phasespace import euclidean_distance

With Quantity-valued components:

>>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
>>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
>>> result = euclidean_distance(q_a, q_b)
>>> float(result.to("m").value)
5.0

"""

__all__: tuple[str, ...] = ()

from collections.abc import Mapping
from typing import TypeAlias

import plum
import quax
from jaxtyping import Real

import unxt as u
from unxt import AbstractQuantity as AbcQ

from knnp._src import algorithm, phasespace

ScalarQComponents: TypeAlias = Mapping[str, Real[u.Q, " "]]  # noqa: UP040
VectorQComponents: TypeAlias = Mapping[str, Real[u.Q, " N"]]  # noqa: UP040


@plum.dispatch
def euclidean_distance(q_a: ScalarQComponents, q_b: ScalarQComponents, /) -> AbcQ:
    """Euclidean distance between Quantity-valued component dictionaries.

    Computes the distance between two phase-space positions represented as
    dictionaries with unxt Quantity scalar values.

    Parameters
    ----------
    q_a, q_b : Mapping[str, unxt.AbstractQuantity]
        Position dictionaries with Quantity-valued components. Must have the
        same keys. All values must have compatible length dimensions.

    Returns
    -------
    unxt.Quantity
        The Euclidean distance with the unit of the input components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> result = euclidean_distance(q_a, q_b)
    >>> result

    """
    return quax.quaxify(phasespace.euclidean_distance)(q_a, q_b)


@plum.dispatch
def unit_direction(
    q_a: ScalarQComponents, q_b: ScalarQComponents, /
) -> ScalarQComponents:
    """Compute unit direction vector from q_a to q_b for Quantity-valued components.

    Computes the unit direction vector pointing from position `q_a` to `q_b`,
    where both positions are represented as dictionaries with unxt Quantity
    scalar values.

    Parameters
    ----------
    q_a, q_b : Mapping[str, unxt.AbstractQuantity]
        Position dictionaries with Quantity-valued components. Must have the
        same keys. All values must have compatible length dimensions.

    Returns
    -------
    Mapping[str, unxt.AbstractQuantity]
        A dictionary representing the unit direction vector. The components
        are dimensionless Quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> result = unit_direction(q_a, q_b)
    >>> result["x"].value  # approximately 0.6
    0.6
    >>> result["y"].value  # approximately 0.8
    0.8

    """
    return quax.quaxify(phasespace.unit_direction)(q_a, q_b)


@plum.dispatch
def velocity_norm(velocity: ScalarQComponents, /) -> AbcQ:
    """Compute the norm of a Quantity-valued velocity vector.

    Computes the Euclidean norm of a velocity vector represented as a
    dictionary with unxt Quantity scalar values.

    Parameters
    ----------
    velocity : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).

    Returns
    -------
    unxt.Quantity
        The Euclidean norm of the velocity with appropriate units.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> result = velocity_norm(vel)
    >>> result.to("m/s").value
    5.0

    """
    return quax.quaxify(phasespace.velocity_norm)(velocity)


@plum.dispatch
def unit_velocity(velocity: ScalarQComponents, /) -> ScalarQComponents:
    """Compute unit velocity vector for Quantity-valued components.

    Computes the unit velocity vector from a velocity represented as a
    dictionary with unxt Quantity scalar values.

    Parameters
    ----------
    velocity : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).

    Returns
    -------
    Mapping[str, unxt.AbstractQuantity]
        A dictionary representing the unit velocity vector. The components
        are dimensionless Quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> result = unit_velocity(vel)
    >>> result["x"].value  # approximately 0.6
    0.6
    >>> result["y"].value  # approximately 0.8
    0.8

    """
    return quax.quaxify(phasespace.unit_velocity)(velocity)


@plum.dispatch
def cosine_similarity(vel_a: ScalarQComponents, vel_b: ScalarQComponents, /) -> AbcQ:
    """Compute cosine similarity between Quantity-valued velocity components.

    Computes the cosine similarity between two velocity vectors represented
    as dictionaries with unxt Quantity scalar values.

    Parameters
    ----------
    vel_a, vel_b : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionaries with Quantity-valued components. Must have the
        same keys. All values must have compatible velocity dimensions
        (length/time).

    Returns
    -------
    unxt.Quantity
        The cosine similarity (dimensionless) between the two velocities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> vel_a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
    >>> vel_b = {"x": u.Q(0.0, "m/s"), "y": u.Q(1.0, "m/s")}
    >>> result = cosine_similarity(vel_a, vel_b)
    >>> result.value
    0.0

    """
    return quax.quaxify(phasespace.cosine_similarity)(vel_a, vel_b)


@plum.dispatch
def nearest_neighbors_with_momentum(
    position: VectorQComponents,
    velocity: VectorQComponents,
    /,
    *,
    start_idx: int,
    lam: float,
) -> algorithm.KNNPResult:
    """kNN+p algorithm for Quantity-valued phase-space data.

    Orders phase-space observations represented as dictionaries with unxt
    Quantity scalar values using the kNN+p algorithm.

    Parameters
    ----------
    position : Mapping[str, unxt.AbstractQuantity]
        Position dictionary with Quantity-valued components. All values must
        have compatible length dimensions.
    velocity : Mapping[str, unxt.AbstractQuantity]
        Velocity dictionary with Quantity-valued components. All values must
        have compatible velocity dimensions (length/time).
    start_idx : int
        Index of the starting observation.
    lam : float
        Momentum weighting parameter.

    Returns
    -------
    KNNPResult
        Result container with ordered indices and original data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> pos = {
    ...     "x": u.Q(jnp.array([0.0, 1.0, 2.0]), "m"),
    ...     "y": u.Q(jnp.array([0.0, 0.5, 1.0]), "m"),
    ... }
    >>> vel = {
    ...     "x": u.Q(jnp.array([1.0, 1.0, 1.0]), "m/s"),
    ...     "y": u.Q(jnp.array([0.5, 0.5, 0.5]), "m/s"),
    ... }
    >>> result = nearest_neighbors_with_momentum(
    ...     position=pos, velocity=vel, start_idx=0, lam=1.0
    ... )
    >>> result["ordered_indices"]
    (0, 1, 2)

    """
    return quax.quaxify(algorithm.nearest_neighbors_with_momentum)(
        position, velocity, start_idx=start_idx, lam=lam
    )
