---
title:
  "knnp: A High-Performance JAX Implementation of Nearest Neighbors with
  Momentum for Phase-Space Ordering in Stellar Streams"
tags:
  - Python
  - JAX
  - astronomy
  - stellar streams
  - galactic dynamics
  - phase-space analysis
authors:
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    corresponding: true
    affiliation: "1, 2"
  - name: Jacob Nibauer
    orcid: 0000-0002-5408-3992
    affiliation: "3"
  - name: Sirui Wu
    orcid: 0009-0003-4675-3622
    affiliation: "4"
affiliations:
  - name:
      MIT Kavli Institute for Astrophysics and Space Research, Massachusetts
      Institute of Technology, Cambridge, MA 02139, USA
    index: 1
  - name:
      Department of Astronomy, Case Western Reserve University, Cleveland, OH
      44106, USA
    index: 2
  - name: Department of Physics, Princeton University, Princeton, NJ 08544, USA
    index: 3
  - name:
      DARK, Niels Bohr Institute, University of Copenhagen, Jagtvej 155, DK-2200
      Copenhagen, Denmark
    index: 4
date: 14 January 2026
bibliography: paper.bib
---

# Summary

Stellar streams—elongated structures of gravitationally unbound stars stripped
from dwarf galaxies and globular clusters—trace the gravitational potential of
the Milky Way with remarkable fidelity. As sensitive probes of galactic
structure and dark matter distribution, streams have become essential tools in
near-field cosmology [@bonaca2020stellar]. A critical preprocessing step in
stream analysis is ordering the discrete, noisy observations of stream member
stars along the stream's one-dimensional trajectory through six-dimensional
phase-space.

`knnp` is an open-source Python package that implements the Nearest Neighbors
with Momentum (kNN+p) algorithm [@nibauer2022charting] using JAX
[@jax2018github], Google's high-performance numerical computing library. By
exploiting JAX's just-in-time (JIT) compilation, automatic vectorization, and
hardware acceleration, `knnp` delivers high performance while maintaining a
clean, Pythonic API suitable for integration into modern astrophysical analysis
pipelines.

# Statement of Need

The kNN+p algorithm addresses a fundamental challenge in stellar stream
analysis: how to trace coherent structures through six-dimensional phase-space
(position $\mathbf{q} = (x, y, z)$ and velocity $\mathbf{p} = (v_x, v_y, v_z)$)
when observations are discrete, noisy, and potentially contaminated by field
stars. Traditional $k$-nearest-neighbor approaches consider only spatial
proximity, ignoring the crucial velocity information that distinguishes stream
members following coherent orbital trajectories. The kNN+p algorithm
incorporates both spatial proximity and velocity momentum—quantified by the
alignment between a star's velocity vector and the direction to potential
neighbors—yielding substantially improved ordering for stream analysis.

The original implementation accompanying @nibauer2022charting demonstrated the
algorithm's effectiveness but was not optimized for the computational demands of
modern surveys. With Gaia [@gaia2016mission; @gaia2023dr3] providing proper
motions and parallaxes for nearly two billion stars, and upcoming spectroscopic
surveys promising radial velocities for millions more, there is a pressing need
for implementations that scale efficiently to large datasets and can leverage
modern hardware accelerators.

`knnp` addresses this need by providing a JAX-based implementation that is:

1. **Hardware-accelerated**: Transparently executes on CPUs, GPUs, and TPUs
   through XLA compilation
2. **Differentiable**: Supports automatic differentiation, enabling
   gradient-based optimization of the momentum parameter $\lambda$ or
   integration into differentiable simulation frameworks
3. **Composable**: Integrates seamlessly with the JAX ecosystem, including
   Equinox [@kidger2021equinox], Diffrax [@kidger2022diffrax], and other
   scientific libraries
4. **Coordinate-agnostic**: Operates on arbitrary phase-space representations
   through JAX's PyTree abstraction
5. **Rigorously tested**: Comprehensive test suite with property-based testing
   via Hypothesis [@maciver2019hypothesis]

# Algorithm and Implementation

The kNN+p algorithm constructs an ordered path through phase-space observations
by greedily selecting successive points that minimize a cost function combining
spatial distance and velocity misalignment. At each step $i$, the algorithm
selects the next point $j$ by minimizing:

$$
C(i, j) = d(\mathbf{q}_i, \mathbf{q}_j) + \lambda \left(1 - \cos\theta_{ij}\right)
$$

where $d(\mathbf{q}_i, \mathbf{q}_j)$ is the Euclidean distance between
positions, $\theta_{ij}$ is the angle between the velocity vector $\mathbf{p}_i$
and the displacement $\mathbf{q}_j - \mathbf{q}_i$, and $\lambda \geq 0$ is a
tunable parameter controlling the relative importance of momentum. When
$\lambda = 0$, the algorithm reduces to standard nearest-neighbor search; as
$\lambda$ increases, points whose directions align with the current velocity are
increasingly favored.

The momentum penalty $1 - \cos\theta_{ij}$ ranges from 0 (perfect alignment) to
2 (anti-alignment), naturally penalizing points that would require
"backtracking" against the stream's flow direction. This formulation is
particularly effective for streams, where stars follow approximately parallel
orbits and velocities vary smoothly along the structure.

## Implementation Details

`knnp` represents phase-space data as pairs of Python dictionaries mapping
component names to JAX arrays:

```python
import jax.numpy as jnp
from knnp import nearest_neighbors_with_momentum

# Full 6D phase-space
position = {"x": x, "y": y, "z": z}
velocity = {"vx": vx, "vy": vy, "vz": vz}

# Order the observations
result = nearest_neighbors_with_momentum(
    position,
    velocity,
    start_idx=0,  # Index of starting point
    lam=1.0,  # Momentum weight parameter
    max_dist=5.0,  # Optional early termination threshold
)

# Access ordered indices and any skipped points
ordered = result["ordered_indices"]
skipped = result["skipped_indices"]
```

This dictionary-based representation leverages JAX's PyTree abstraction,
enabling the same code to handle arbitrary coordinate systems—Cartesian,
spherical, Galactocentric, or custom projections—without modification. All
geometric operations (distances, angles, normalizations) are implemented as
tree-mapped functions that broadcast appropriately across dictionary structures.

The core algorithm loop uses `jax.lax.while_loop` for efficient iteration with
XLA compilation. Distance and angle computations are fully vectorized, operating
on entire arrays simultaneously rather than through Python loops. The
implementation carefully handles numerical edge cases, including zero-velocity
vectors and coincident points, to ensure stability across realistic datasets.

Key JAX features exploited by `knnp`:

- **JIT compilation via `@jax.jit`**: The inner loop compiles to optimized XLA
  code, eliminating Python overhead
- **Automatic vectorization**: All geometric operations vectorize cleanly over
  observation arrays
- **Hardware portability**: The same code executes on CPU, GPU, or TPU with
  appropriate JAX backend
- **Functional purity**: Stateless implementation enables safe parallelization
  and caching

# Performance Characteristics

The algorithm has $O(N^2)$ worst-case complexity per iteration (computing
distances to all unvisited points), with $N$ iterations for $N$ points. However,
JAX's vectorization and compilation provide substantial constant-factor
improvements:

- Vectorized distance and angle computations avoid Python loop overhead
- JIT compilation eliminates interpreter costs and enables XLA optimization
- GPU execution parallelizes the distance computations across thousands of cores

In practice, we observe 10–100× speedup on CPU compared to NumPy-based
implementations for datasets exceeding 1,000 points, with additional 5–10×
improvement on GPU for datasets exceeding 10,000 points. The implementation also
supports `jax.vmap` for parallel analysis of multiple streams or parameter
sweeps without code modification.

# Research Applications

`knnp` enables several research applications in stellar stream analysis:

- **Stream membership refinement**: Ordering observations helps identify
  outliers and contaminants that disrupt the smooth phase-space trajectory
- **Density and gap detection**: Once ordered, one-dimensional density
  estimation along the stream reveals gaps potentially induced by dark matter
  subhalos [@bonaca2019spur]
- **Acceleration measurement**: The kNN+p ordering is a prerequisite for the
  StreamSculptor framework [@nibauer2022charting], which constrains Galactic
  accelerations from stream morphology
- **Simulation comparison**: Ordered streams facilitate comparison with N-body
  simulations of stream formation and evolution

Beyond stellar streams, the algorithm applies to any trajectory ordering problem
in phase-space, including tidal debris, galactic outflows, and particle tracking
in laboratory physics.

# Acknowledgements

N.S. is supported by the Brinson Prize Fellowship at MIT. This work made use of
the JAX, NumPy, and Astropy software packages. We thank Jacob Nibauer for
developing the original kNN+p algorithm and for helpful discussions during this
implementation.

# References
