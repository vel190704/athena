# ADR-005: Sparse Masked Indexing over Dense Grids for the Spatial Mathematics Core

## Status
Accepted

## Context
Module 1 (Spatial Mathematics Core) evaluates pitch-control and time-to-intercept
computations over a discretized pitch grid. A dense representation at typical
resolution is a `[100, 68]` grid per batch element, per player (22 players), which
scales as `O(100 × 68 × 22)`. At this density, the production Triton kernel misses the
sub-50ms real-time latency budget: most of the grid is spatially irrelevant to any given
player-ball interaction, since interception and pitch-control dynamics decay sharply
with distance from the ball.

Dynamic Adaptive Mesh Refinement (AMR) was evaluated as an alternative to reduce
wasted compute, but rejected — AMR produces variable-shaped, data-dependent tensors,
which causes Triton memory thrashing (recompilation and irregular memory access
patterns) and defeats the kernel fusion the production path relies on.

## Decision
Before the kernel launches, we extract indices where `distance_to_ball <= 30m` and pass
only these sparse coordinates (empirically ~2,000 of the ~6,800 total grid cells) to the
solver. This reduces compute complexity from `O(100 × 68 × 22)` to `O(2000 × 22)`.

Critically, sparsity is achieved through **masked indexing, not shape reduction**: a
binary spatial mask of shape `[Batch, 100, 68]` is carried alongside the sparse
coordinate list, so the tensor shape presented to Triton remains static regardless of
how many cells are actually within the 30m radius on a given frame. Static shapes avoid
kernel recompilation and preserve the memory-access regularity Triton requires for
consistent latency.

## Consequences
- The 30m cutoff is a physically-motivated but tunable hyperparameter — it must be
  validated to ensure no relevant interception geometry is being discarded (e.g. a very
  fast player converging on a long ball from >30m out). This is a candidate for a
  follow-up ADR once Milestone 2 profiling data exists.
- Masked-out cells still occupy memory in the static-shape tensor; the latency win comes
  from the solver skipping compute on masked cells, not from a smaller tensor footprint.
  This is an intentional trade of memory for latency stability.
- Any downstream consumer of the spatial field (e.g. the GNN edge feature extraction in
  Module 1, via `grid_sample`) must respect the mask and must not implicitly treat
  masked-out cells as zero-valued signal.

## Alternatives Considered
- **Dense `[100, 68]` grid per player**: rejected — fails the sub-50ms latency budget.
- **Dynamic AMR**: rejected — variable tensor shapes cause Triton memory thrashing.
- **Fixed coarser grid resolution**: rejected — reduces spatial fidelity uniformly
  across the pitch rather than concentrating fidelity where it is needed (near the
  ball), and does not address the underlying O(grid × players) scaling problem.
