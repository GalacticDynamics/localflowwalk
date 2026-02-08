---
title:
  "localflowwalk: A High-Performance JAX Framework for Phase-Space Walks with
  Pluggable Metrics and Strategies"
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
date: 14 January 2026
bibliography: paper.bib
---

# Summary

Filamentary structures are ubiquitous in the physical sciences, ranging from
coherent streams of stars called stellar streams to elongated structures in
turbulent fluids, plasmas, and the interstellar medium. In the context of
stellar streams, a common preprocessing step in stream analysis, be it of
simulations or observations, is ordering the discrete observations of stream
member stars and inferring the mean path of the stream's trajectory through
phase-space.

`localflowwalk` is an open-source Python package for constructing such orderings
and paths by _walking along the local phase-space flow_ using JAX
[@jax2018github]. There are two core components: the first is `walk_local_flow`
which builds an approximate ordering and trajectory but which might miss some of
the data; and the second is a fast-to-train autoencoder that imputes the full
ordering and trajectory.

`walk_local_flow` is very modular and can be customized to use any:

1. **distance metric** that scores candidate next steps in phase space, and
2. **query strategy**, like brute-force or KD-trees, that proposes which
   candidates to consider.

This design makes the Nearest Neighbors with Momentum (NN+p) method from
[@nibauer2022charting] one particular configuration (via
`localflowwalk.metrics.AlignedMomentumDistanceMetric`), while enabling
alternative metrics and search strategies better matched to different data
scenarios and performance constraints.

In addition to the walk itself, `localflowwalk` packages a neural-network
gap-filling component which assigns a continuous ordering parameter to data
skipped during the walk and reconstructs the spatial mean path of the structure.
This encoder develops upon the one in [@nibauer2022charting]: speeding up
different components of training by between 1 and 3 orders of magnitude; adding
an intermediate decoder-only training that quarters the epochs necessary for
training the full autoencoder, halving the overall training time; and adding
stabilization of the loss function across training phases.

# Statement of Need

Ordering stream constituent members is challenging, even in forward-model
simulations where all variables can be controlled. For most models above the
complexity of streaklines -- with zero velocity dispersion at particle release
-- no property intrinsic to the stream may be used to determine the path or
path-order. Therefore, it is necessary to develop algorithms which can infer the
path-ordering and path of the stream. Moreover, it is necessary for these
methods to be performant to support high performance stream simulators, and
auto-differentiable to support inference routines.

The original phase-flow walk implementation accompanying [@nibauer2022charting]
demonstrated the efficacy of the core algorithms, but was not sufficiently
performant nor autodifferentiable, was inflexible regarding the distance metrics
and search strategies, and most importantly was not built as a reusable,
extensible library.

`localflowwalk` addresses these many needs, providing an implementation that
has:

1. **A modular API** with user-selectable metrics and neighbor-query strategies;
2. **Hardware acceleration and composability** through JAX transformations and
   XLA compilation, interoperating with libraries like Equinox
   [@kidger2021equinox];
3. **Differentiability** of metrics and neural components for gradient-based
   tuning and downstream integration;
4. **Coordinate-agnostic inputs** through JAX PyTrees (e.g., `dict`s of
   components);
5. **Rigorous tests** with a property-based test suite via `hypothesis`
   [@maciver2019hypothesis]; and
6. **Interoperability** with `unxt` [@Starkman2025] for unit support in JAX.

# Package Design

With `localflowwalk`, fitting an affine parameter that orders a stream and
inferring the mean path requires 5 lines of code:

1. to construct an initial walk
2. to normalize the data
3. to define the autoencoder
4. to train the autoencoder
5. to get the final result

```python
import jax
import localflowwalk as lfw

walkresult = lfw.walk_local_flow(pos, vel, ...)
normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
ae = lfw.nn.PathAutoencoder.make(normalizer, key=jax.random.key(0))
ae, *_ = lfw.nn.train_autoencoder(ae, walkresult, key=jax.random.key(1))
result = lfw.nn.fill_ordering_gaps(ae, walkresult)
```

# Performance Characteristics

In practice, we observe high performance on datasets with tens of thousands of
points. The implementation also supports `jax.vmap` for parallel analysis of
multiple streams or parameter sweeps without code modification.

- `walk_local_flow` runs in under a second.
- Training the gap-filling path-ordering encoder takes under a second.
- Training the decoder to reconstruct a running mean takes under 4 seconds.
- Training the encoder and full path-inferring decoder takes a little under 16
  seconds.

The slowest step is the path-inferring decoder. Due to `localflowwalk`'s modular
design this step is easily skipped and users can use the gap-filling encoder
with alternate or custom path-inferring functions. An included alternate
function is a path-ordered rolling mean, which consumes a small fraction of a
second after training the encoder.

# Research Applications

`localflowwalk` enables easy data-driven reconstruction of stream paths in
simulations. Beyond stellar streams, the same walk abstraction applies to
ordering problems in other phase-space datasets where a coherent local flow is
expected.

# Acknowledgements

N.S. is supported by the Brinson Prize Fellowship at MIT. J.N. ... This work
made use of the `JAX`, `NumPy`, `Astropy`, and `unxt` [@Starkman2025] software
packages.

# References
