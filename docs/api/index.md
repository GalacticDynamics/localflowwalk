# API Reference

Complete API documentation for knnp.

```{eval-rst}
.. currentmodule:: knnp
```

## Main Function

```{eval-rst}
.. autofunction:: nearest_neighbors_with_momentum
   :no-index:
```

## Result Accessor

Helper function to extract ordered data from results.

```{eval-rst}
.. autofunction:: get_ordered_w
   :no-index:
```

## Phase-Space Utilities

Low-level functions for phase-space operations. Available in the `knnp.phasespace` submodule.

```{eval-rst}
.. currentmodule:: knnp.phasespace

.. autofunction:: euclidean_distance
   :no-index:

.. autofunction:: unit_direction
   :no-index:

.. autofunction:: unit_velocity
   :no-index:

.. autofunction:: cosine_similarity
   :no-index:

.. autofunction:: get_w_at
   :no-index:

.. currentmodule:: knnp
```

## Types

```{eval-rst}
.. autoclass:: KNNPResult
   :no-index:
   :members:
   :show-inheritance:

.. data:: ScalarComponents
   :annotation: : TypeAlias = Mapping[str, FloatScalar]

   Type alias for dictionaries mapping component names to scalar JAX arrays.

   Used for single phase-space points. Keys are coordinate/component names
   (e.g., "x", "y", "z"), values are 0-dimensional JAX arrays.

   Example::

       position: ScalarComponents = {
           "x": jnp.array(1.0),
           "y": jnp.array(2.0),
       }

.. data:: VectorComponents
   :annotation: : TypeAlias = Mapping[str, FloatArray]

   Type alias for dictionaries mapping component names to 1D JAX arrays.

   Used for arrays of phase-space points. Keys are coordinate/component names
   (e.g., "x", "y", "z"), values are 1-dimensional JAX arrays of shape (N,).

   Example::

       position: VectorComponents = {
           "x": jnp.array([0.0, 1.0, 2.0]),
           "y": jnp.array([0.0, 1.0, 2.0]),
       }
```

## Autoencoder Module

Neural network for interpolating skipped tracers. See [Autoencoder Guide](../guides/autoencoder.md) for details.

### Classes

```{eval-rst}
.. autoclass:: knnp.autoencoder.Autoencoder
   :no-index:
   :members: encode, decode, decode_position, predict
   :show-inheritance:

.. autoclass:: knnp.autoencoder.InterpolationNetwork
   :no-index:
   :members: __call__
   :show-inheritance:

.. autoclass:: knnp.autoencoder.ParamNet
   :no-index:
   :members: __call__
   :show-inheritance:

.. autoclass:: knnp.autoencoder.PhaseSpaceDataSource
   :no-index:
   :members: __len__, __getitem__
   :show-inheritance:

.. autoclass:: knnp.autoencoder.TrainingConfig
   :no-index:
   :members:

.. autoclass:: knnp.autoencoder.AutoencoderResult
   :no-index:
   :members:
```

### Training Functions

```{eval-rst}
.. autofunction:: knnp.autoencoder.train_autoencoder
   :no-index:

.. autofunction:: knnp.autoencoder.fill_ordering_gaps
   :no-index:
```

## Type Annotations

knnp uses extensive type annotations for clarity and static checking:

- **jaxtyping.Array**: JAX array type
- **ArrayLike**: Values that can be converted to JAX arrays
- **ScalarComponents**: Mapping[str, FloatScalar] - single phase-space point
- **VectorComponents**: Mapping[str, FloatArray] - array of phase-space points
- **TypedDict**: Structured dictionary types (used for KNNPResult)

Type checkers like mypy and pyright can verify knnp code for correctness.

## Examples

See the [Examples Gallery](../guides/examples.md) for comprehensive usage examples.

## Index

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
