"""Tests for unxt interoperability with phasespace functions."""

import quaxed.numpy as jnp
import unxt as u

from knnp.phasespace import (
    cosine_similarity,
    euclidean_distance,
    unit_direction,
    unit_velocity,
    velocity_norm,
)


class TestEuclideanDistanceQuantity:
    """Tests for euclidean_distance with Quantity-valued components."""

    def test_simple_distance(self):
        """Test distance calculation with simple Quantity values."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
        q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
        result = euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(5.0, "m"), atol=u.Q(1e-6, "m"))

    def test_distance_preserves_units(self):
        """Test that distance preserves input units."""
        q_a = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km")}
        q_b = {"x": u.Q(3.0, "km"), "y": u.Q(4.0, "km")}
        result = euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(5.0, "km"), atol=u.Q(1e-6, "km"))

    def test_identical_points(self):
        """Test that distance between identical points is zero."""
        q = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        result = euclidean_distance(q, q)

        assert jnp.allclose(result, u.Q(0.0, "m"), atol=u.Q(1e-10, "m"))

    def test_3d_distance(self):
        """Test distance with 3D coordinates."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        q_b = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(2.0, "m")}
        result = euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(3.0, "m"), atol=u.Q(1e-6, "m"))


class TestUnitDirectionQuantity:
    """Tests for unit_direction with Quantity-valued components."""

    def test_simple_direction(self):
        """Test unit direction calculation."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
        q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
        result = unit_direction(q_a, q_b)

        assert jnp.allclose(result["x"], u.Q(0.6, ""), atol=u.Q(1e-6, ""))
        assert jnp.allclose(result["y"], u.Q(0.8, ""), atol=u.Q(1e-6, ""))

    def test_direction_is_dimensionless(self):
        """Test that unit direction is dimensionless."""
        q_a = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km")}
        q_b = {"x": u.Q(3.0, "km"), "y": u.Q(4.0, "km")}
        result = unit_direction(q_a, q_b)

        # Check dimensionless (unit should be dimensionless)
        assert result["x"].unit == u.unit("")
        assert result["y"].unit == u.unit("")

    def test_direction_has_unit_length(self):
        """Test that direction vector has unit norm."""
        q_a = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        q_b = {"x": u.Q(4.0, "m"), "y": u.Q(6.0, "m")}
        result = unit_direction(q_a, q_b)

        # Compute norm
        norm_sq = result["x"] ** 2 + result["y"] ** 2
        assert jnp.allclose(
            norm_sq,
            u.Q(1.0, ""),
            atol=u.Q(1e-6, ""),
        )


class TestVelocityNormQuantity:
    """Tests for velocity_norm with Quantity-valued components."""

    def test_simple_norm(self):
        """Test velocity norm calculation."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = velocity_norm(vel)

        assert jnp.allclose(result, u.Q(5.0, "m/s"), atol=u.Q(1e-6, "m/s"))

    def test_zero_velocity(self):
        """Test norm of zero velocity."""
        vel = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = velocity_norm(vel)

        assert jnp.allclose(result, u.Q(0.0, "m/s"), atol=u.Q(1e-10, "m/s"))

    def test_3d_norm(self):
        """Test velocity norm with 3D velocity."""
        vel = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(2.0, "m/s")}
        result = velocity_norm(vel)

        assert jnp.allclose(
            result, u.Q(3.0, "m/s"), atol=u.Q(1e-6, "m/s")
        )  # sqrt(1 + 4 + 4) = 3


class TestUnitVelocityQuantity:
    """Tests for unit_velocity with Quantity-valued components."""

    def test_simple_unit_velocity(self):
        """Test unit velocity calculation."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = unit_velocity(vel)

        assert jnp.allclose(result["x"], u.Q(0.6, ""), atol=u.Q(1e-6, ""))
        assert jnp.allclose(result["y"], u.Q(0.8, ""), atol=u.Q(1e-6, ""))

    def test_unit_velocity_is_dimensionless(self):
        """Test that unit velocity is dimensionless."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = unit_velocity(vel)

        assert result["x"].unit == u.unit("")
        assert result["y"].unit == u.unit("")

    def test_unit_velocity_has_unit_length(self):
        """Test that unit velocity vector has unit norm."""
        vel = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(2.0, "m/s")}
        result = unit_velocity(vel)

        norm_sq = result["x"] ** 2 + result["y"] ** 2 + result["z"] ** 2
        assert jnp.allclose(
            norm_sq,
            u.Q(1.0, ""),
            atol=u.Q(1e-6, ""),
        )


class TestCosineSimilarityQuantity:
    """Tests for cosine_similarity with Quantity-valued components."""

    def test_parallel_vectors(self):
        """Test cosine similarity of parallel vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = cosine_similarity(a, b)

        # Result is dot product of vectors, so has unit m²/s²
        assert jnp.allclose(result, u.Q(1.0, "m**2/s**2"), atol=u.Q(1e-6, "m**2/s**2"))

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(0.0, "m/s"), "y": u.Q(1.0, "m/s")}
        result = cosine_similarity(a, b)

        # Orthogonal vectors have zero dot product
        assert jnp.allclose(result, u.Q(0.0, "m**2/s**2"), atol=u.Q(1e-6, "m**2/s**2"))

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(-1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = cosine_similarity(a, b)

        # Opposite vectors have negative dot product
        assert jnp.allclose(result, u.Q(-1.0, "m**2/s**2"), atol=u.Q(1e-6, "m**2/s**2"))

    def test_similarity_is_dimensionless(self):
        """Test that cosine similarity has expected unit from dot product."""
        a = {"x": u.Q(3.0, "km/s"), "y": u.Q(4.0, "km/s")}
        b = {"x": u.Q(3.0, "km/s"), "y": u.Q(4.0, "km/s")}
        result = cosine_similarity(a, b)

        # Dot product of km/s vectors has unit km²/s²
        assert result.unit == u.unit("km**2/s**2")
