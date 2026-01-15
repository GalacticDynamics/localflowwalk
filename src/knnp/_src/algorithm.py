"""Nearest Neighbors with Momentum algorithm implementation.

This module implements the Nearest Neighbors with Momentum (kNN+p) algorithm
from Nibauer et al. (2022) for ordering phase-space observations in stellar
streams using both spatial proximity and velocity momentum.

The algorithm finds a path through phase-space observations that balances:
1. Spatial proximity (distance between neighboring points)
2. Velocity momentum (alignment of the velocity with the direction to the next point)

This is particularly useful for tracing stellar streams where stars follow
coherent trajectories through phase-space.

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning."

Examples
--------
>>> import jax.numpy as jnp
>>> from knnp import nearest_neighbors_with_momentum

Create some example phase-space data:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0]), "y": jnp.array([0.0, 0.5, 0.8, 1.2])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0]), "y": jnp.array([0.2, 0.2, 0.2, 0.2])}

Run the algorithm:

>>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=0.5)
>>> len(result["ordered_indices"])
4

"""

__all__: tuple[str, ...] = (
    "KNNPResult",
    "nearest_neighbors_with_momentum",
)

from collections.abc import Set
from typing import TypedDict

import jax
import jax.lax
import jax.numpy as jnp
import jax.tree as jtu
import plum
from jaxtyping import Array

from zeroth import zeroth

from .custom_types import VectorComponents
from .phasespace import (
    cosine_similarity,
    euclidean_distance,
    get_w_at,
    unit_direction,
    unit_velocity,
)


class KNNPResult(TypedDict):
    """Result of the Nearest Neighbors with Momentum algorithm.

    Keys
    ----
    ordered_indices : tuple[int, ...]
        The indices of observations in the order found by the algorithm.
        May contain fewer indices than total observations if the algorithm
        terminated early due to max_dist.
    skipped_indices : tuple[int, ...]
        The indices of observations that were skipped (not visited).
        As noted in the paper: "Due to the momentum condition, the algorithm
        inevitably passes over some stream particles without incorporating
        them into the nearest neighbors graph."
    position : dict[str, Array]
        The original position data.
    velocity : dict[str, Array]
        The original velocity data.

    """

    ordered_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]
    position: VectorComponents
    velocity: VectorComponents


@plum.dispatch
def nearest_neighbors_with_momentum(
    position: VectorComponents,
    velocity: VectorComponents,
    /,
    *,
    start_idx: int = 0,
    lam: float = 1.0,
    max_dist: float | None = None,
    terminate_indices: Set[int] | None = None,
    n_max: int | None = None,
) -> KNNPResult:
    r"""Find an ordered path through phase-space using nearest neighbors with momentum.

    This implements Algorithm 1 from Nibauer et al. (2022). The algorithm
    greedily selects the next point in the sequence by minimizing a distance
    metric that combines spatial proximity with velocity alignment.

    Due to the momentum condition, the algorithm may terminate before visiting
    all points. Points that would require "going backwards" (against the
    velocity direction) receive high momentum penalties and may be skipped.
    As stated in the paper: "Due to the momentum condition, the algorithm
    inevitably passes over some stream particles without incorporating them
    into the nearest neighbors graph."

    Parameters
    ----------
    position : Mapping[str, Array]
        Position dictionary with 1D array values of shape (N,).
    velocity : Mapping[str, Array]
        Velocity dictionary with 1D array values of shape (N,).
    start_idx : int, optional
        The index of the starting observation (default: 0).
    lam : float, optional
        The momentum weight ($\lambda$). Higher values favor points whose direction
        from the current point aligns with the current velocity. Default: 1.0.
    max_dist : float or None, optional
        Maximum allowable distance between neighbors. If the minimum distance
        exceeds this value, the algorithm terminates, leaving remaining points
        unvisited. This is key to the algorithm's ability to skip outliers.
        Default: None (no limit).
    terminate_indices : Set[int] or None, optional
        Set of indices at which to terminate the algorithm if reached.
        Default: None.
    n_max : int or None, optional
        Maximum number of iterations. Default: None (process all points).

    Returns
    -------
    KNNPResult
        TypedDict with keys:
        - "ordered_indices": tuple of indices in order (may be fewer than
          total points if algorithm terminated early due to max_dist)
        - "position": original position dict
        - "velocity": original velocity dict

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from knnp import nearest_neighbors_with_momentum

    Create phase-space data for a simple stream:

    >>> pos = {
    ...     "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    ...     "y": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    ... }
    >>> vel = {
    ...     "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    ...     "y": jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    ... }

    Run the algorithm starting from index 0:

    >>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=0.5)
    >>> result["ordered_indices"]
    (0, 1, 2, 3, 4)

    """
    # Get number of observations from first position array
    first_key = zeroth(position)
    n_obs = position[first_key].shape[0]

    if start_idx < 0 or start_idx >= n_obs:
        msg = f"start_idx {start_idx} out of bounds for data with {n_obs} observations."
        raise ValueError(msg)

    if n_max is None:
        n_max = n_obs

    if terminate_indices is None:
        terminate_indices = set()

    # Convert max_dist to a JAX-compatible value
    max_dist_value = jnp.inf if max_dist is None else float(max_dist)

    # Create ordered_indices array (-1 for unused slots)
    ordered_arr = -jnp.ones(n_obs, dtype=jnp.int32)
    ordered_arr = ordered_arr.at[0].set(start_idx)

    # Create visited mask (0.0 for visited, 1.0 for available)
    visited = jnp.ones(n_obs)
    visited = visited.at[start_idx].set(0.0)

    # Convert terminate_indices to array outside the loop
    terminate_arr = (
        jnp.array(list(terminate_indices))
        if terminate_indices
        else jnp.array([], dtype=jnp.int32)
    )

    # State: (ordered_indices, visited_mask, current_idx, step, should_stop)
    state = (ordered_arr, visited, start_idx, 1, False)

    def cond_fn(state: tuple, /) -> bool:
        """Continue looping if there are remaining points and iterations left."""
        ordered_indices, visited_mask, current_idx, step, should_stop = state

        # Check if current index should terminate
        should_not_terminate = jnp.where(
            terminate_arr.size > 0,
            jnp.all(current_idx != terminate_arr),
            jnp.array(True),
        )

        return jnp.logical_and(
            jnp.logical_and(step < n_max, jnp.logical_not(should_stop)),
            should_not_terminate,
        )

    def body_fn(state: tuple, /) -> tuple:
        """Process one iteration using dict-based phase-space operations."""
        ordered_indices, visited_mask, current_idx, step, should_stop = state

        # Get current position and velocity (scalar dicts)
        current_pos, current_vel = get_w_at(position, velocity, current_idx)

        # Compute distances from current point to all points (vmap over array)
        d0 = jax.vmap(lambda q: euclidean_distance(current_pos, q))(position)

        # Compute unit directions from current point to all points (vmap over array)
        unit_dirs = jax.vmap(lambda q: unit_direction(current_pos, q))(position)

        # Compute unit velocity of current point (scalar operation)
        unit_vel = unit_velocity(current_vel)

        # Compute cosine similarity between unit velocity and all unit directions (vmap)
        cos_sim = jax.vmap(lambda ud: cosine_similarity(unit_vel, ud))(unit_dirs)

        # Momentum distance: d = d0 + λ * (1 - cos_sim)
        distances = d0 + lam * (1.0 - cos_sim)

        # Mask visited points (where mask is 0)
        distances_masked = jnp.where(visited_mask > 0.5, distances, jnp.inf)

        # Find nearest neighbor
        min_dist = jnp.min(distances_masked)
        best_idx = jnp.argmin(distances_masked)

        # Check termination BEFORE adding the point
        new_should_stop = min_dist > max_dist_value

        # Conditional update: only add if not terminating
        new_ordered = jnp.where(
            new_should_stop,
            ordered_indices,
            ordered_indices.at[step].set(best_idx),
        )
        new_mask = jnp.where(
            new_should_stop,
            visited_mask,
            visited_mask.at[best_idx].set(0.0),
        )
        new_step = jnp.where(
            new_should_stop,
            step,
            step + 1,
        )

        return (new_ordered, new_mask, best_idx, new_step, new_should_stop)

    # Use jax.lax.while_loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, state)

    # Extract results
    final_ordered, final_visited, _, final_step, _ = final_state
    ordered_tuple = tuple(int(i) for i in final_ordered[:final_step] if i >= 0)
    skipped_tuple = tuple(int(i) for i in range(n_obs) if final_visited[i] > 0.5)

    return KNNPResult(
        ordered_indices=ordered_tuple,
        skipped_indices=skipped_tuple,
        position=dict(position),
        velocity=dict(velocity),
    )


def get_ordered_w(res: KNNPResult, /) -> tuple[dict[str, Array], dict[str, Array]]:
    """Get position and velocity in the ordered sequence from a KNNPResult.

    Parameters
    ----------
    res
        The result from nearest_neighbors_with_momentum.

    Returns
    -------
    position : dict[str, Array]
        Position arrays reordered according to the algorithm's output.
    velocity : dict[str, Array]
        Velocity arrays reordered according to the algorithm's output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from knnp import nearest_neighbors_with_momentum, get_ordered_w
    >>> pos = {"x": jnp.array([3.0, 1.0, 2.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=1, lam=0.0)
    >>> ordered_pos, ordered_vel = get_ordered_w(result)

    """
    indices = jnp.array(res["ordered_indices"])
    f = lambda v: v[indices]  # noqa: E731
    return jtu.map(f, res["position"]), jtu.map(f, res["velocity"])
