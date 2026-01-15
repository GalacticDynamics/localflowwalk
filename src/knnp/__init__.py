"""knnp: Nearest Neighbors with Momentum for phase-space ordering.

This library implements the Nearest Neighbors with Momentum (kNN+p) algorithm
from Nibauer et al. (2022) for ordering phase-space observations in stellar
streams. The algorithm uses both spatial proximity and velocity momentum to
trace coherent structures through phase-space.

Phase-space data is represented as two dictionaries:
- `position`: Maps component names to position arrays (e.g., {"x": array, "y": array})
- `velocity`: Maps component names to velocity arrays (same keys as position)

Main Components
---------------
nearest_neighbors_with_momentum : function
    The main algorithm for ordering phase-space observations.
KNNPResult : TypedDict
    Result container with ordered indices and original data.
get_ordered_w : function
    Extract reordered position and velocity arrays from results.

Submodules
----------
knnp.phasespace : module
    Low-level phase-space operations (distances, directions, similarities).
knnp.autoencoder : module
    Neural network for interpolating skipped tracers.

Examples
--------
>>> import jax.numpy as jnp
>>> from knnp import nearest_neighbors_with_momentum

Create phase-space observations as dictionaries:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

Order the observations:

>>> result = nearest_neighbors_with_momentum(pos, vel, start_idx=0, lam=1.0)
>>> result["ordered_indices"]
(0, 1, 2)

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning."

"""

__all__: tuple[str, ...] = (
    # Version
    "__version__",
    # Type aliases
    "ScalarComponents",
    "VectorComponents",
    # Algorithm
    "nearest_neighbors_with_momentum",
    "KNNPResult",
    # Result accessor
    "get_ordered_w",
)

from ._src.algorithm import (
    KNNPResult,
    get_ordered_w,
    nearest_neighbors_with_momentum,
)
from ._src.custom_types import ScalarComponents, VectorComponents
from ._version import version as __version__

# isort: split
# Optional interop registrations (e.g., unxt)
from . import _interop  # noqa: F401
