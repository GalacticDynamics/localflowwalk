"""Type aliases for knnp.

This module defines common type aliases used throughout the library.
"""

__all__: tuple[str, ...] = (
    "FloatScalar",
    "FloatArray",
    "ScalarComponents",
    "VectorComponents",
)


from typing import TypeAlias

from jaxtyping import ArrayLike, Float

# Scalar float type
FloatScalar: TypeAlias = Float[ArrayLike, " "]  # noqa: UP040

# 1D array of floats
FloatArray: TypeAlias = Float[ArrayLike, " N"]  # noqa: UP040

# Type aliases for component dictionaries
ScalarComponents: TypeAlias = dict[str, FloatScalar]  # noqa: UP040
"""dict of component names to scalar arrays (single phase-space point)."""

VectorComponents: TypeAlias = dict[str, FloatArray]  # noqa: UP040
"""dict of component names to 1D arrays (multiple phase-space points)."""
