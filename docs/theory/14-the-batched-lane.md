# Many optimal control paths at once

`spinoct.numeric.BatchImageOCPSolver` solves a batch of independent optimal control problems as one
tensor, on a GPU when there is one. It needs the optional extra, `pip install "spinoct[torch]"`.

## What it is for, and what it is not for

One optimal control path is a small problem: a chain of a few hundred unit vectors, minimized in a
second or two on one core. A GPU does nothing for it, and this lane is not a way to make a single solve
faster.

What the sweeps in front of this package actually do is solve hundreds of such problems that differ
only in their parameters: a cost front per material, a map over damping and hard-axis ratio, a seed
sweep over the coexisting optimal paths of a biaxial system. Those are independent, identically shaped
and embarrassingly parallel. That is the shape this lane is for.

## The same functional, term for term

The functional and its discretization are those of Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B
107, 214448 (2023), https://doi.org/10.1103/PhysRevB.107.214448, exactly as in
`docs/theory/05-numerical-optimal-control-path.md`; this lane changes how it is evaluated, not what is
evaluated. The acceptance gate is the closed-form uniaxial solution of Kwiatkowski, Badarneh, Berkov
and Bessarab, Phys. Rev. Lett. 126, 177206 (2021),
https://doi.org/10.1103/PhysRevLett.126.177206.

The physics is unchanged:

```
s_{p+1/2}     = (s_{p+1} + s_p) / |s_{p+1} + s_p|
s_dot_{p+1/2} = (delta_p / dt) (s_{p+1} - s_p) / |s_{p+1} - s_p|
b             = (alpha / gamma) s_dot + (1 / gamma) [s x s_dot] - b_i_perp(s)
Phi           = dt sum_p |b_{p+1/2}|^2
```

carried over a leading batch axis, so every quantity above has shape `(B, Q + 1, ...)` and the systems
in the batch may differ in every parameter: moment, anisotropy, damping, hard-axis ratio, switching
time.

Two things do differ from the CPU lane, both on purpose.

**The gradient is exact.** The CPU solver differences the cost locally with a step of `1e-7` per
component, which is what makes its iteration `O(Q)` instead of `O(Q^2)`. Automatic differentiation
gives the same derivative without the truncation, and on a batch the backward pass is as parallel as
the forward one. The two gradients are held to agreement in `tests/test_batch_ocp.py`: measured worst
difference `4.6e-06` of the gradient scale, on a biaxial chain of 40 images. That is the floor of a
central difference with a `1e-7` step on a unit vector, not a disagreement about the derivative.

**One tensor means one discretization.** Every problem in a batch is discretized with the same `Q`.
`BatchImageOCPSolver.recommended_images` takes the largest count any problem in the batch asks for on
its own, which over-resolves the easy problems (costing time) rather than under-resolving the hard ones
(costing correctness).

## Convergence is reported per problem

A batch is minimized jointly: the objective handed to L-BFGS is the sum of the per-problem costs, each
divided by its own starting cost. The sum is separable, so its minimizer is each problem's minimizer,
and the normalization is what stops one expensive problem in the batch from setting the step size for
all of them. It is the same rescaling the single-problem solver applies.

But a joint minimization can leave individual rows short, and a single verdict for the whole batch
would hide them. So each problem reports its own residual: the largest tangent-gradient magnitude over
its chain, divided by its own cost, which is dimensionless and therefore the same test at every
switching time.

The threshold, `1e-4`, is measured rather than chosen. On the CPU lane's own paths at `Q = 80`,
uniaxial and biaxial, `T` from 1 to 100 `tau0`, the five solves the reference itself calls converged
sit between `3.7e-06` and `5.5e-05`, and the one that hits its iteration cap sits at `2.2e-03`. The
threshold is above every converged path of the reference and more than an order of magnitude below a
stalled one.

## What it is accepted against

A second implementation of a solved problem is a liability unless it is pinned to the first one. This
one is pinned twice:

- **against the closed form**, on a batch of uniaxial systems, where the exact cost is known: every row
  lands above its exact cost and within 1 per cent at 80 images, the same midpoint-discretization
  signature the CPU lane is held to;
- **against the CPU lane problem by problem**, on biaxial systems where there is no closed form. Over a
  72-problem sweep (12 dampings by 6 hard-axis ratios at `T = 5 tau0`, `Q = 60`), the worst deviation
  from the CPU lane's own answer was `8.5e-07` relative.

`BatchOCPResult` records the device and the engine version that produced the numbers, because "the GPU
said so" is not provenance.

## Measured speedup, and where the GPU actually starts to pay

Two measurements, on a laptop with an RTX 4070 (8 GB) and torch 2.14.

**Batching is worth more than the device.** 72 independent biaxial problems at `Q = 60`, each solved to
its own convergence:

| lane | wall clock | speedup |
|---|---|---|
| CPU, one problem at a time | 43.4 s | 1.0x |
| batch, on the CPU | 11.7 s | 3.7x |

The per-iteration overhead of a Python-level solve is paid once for the whole batch instead of once per
problem, and that alone is most of what there is to win at this size.

**The GPU only pays on large batches.** Forcing every configuration to the same 100 L-BFGS iterations,
so the comparison is throughput and not convergence luck (the iteration count each lane actually took
is beside its time):

| problems | images | CPU float64 | CUDA float64 | CUDA float32 |
|---|---|---|---|---|
| 72 | 60 | **1.6 s** | 3.7 s | 3.1 s |
| 72 | 240 | **1.8 s** | 2.8 s | 4.4 s |
| 288 | 60 | **1.9 s** | 4.3 s | 4.3 s |
| 288 | 240 | **3.9 s** | 4.8 s | 4.1 s |
| 1,152 | 60 | **3.5 s** | 3.7 s | 3.9 s |
| 1,152 | 240 | 10.8 s | 6.9 s | **5.8 s** |
| 4,608 | 60 | 11.3 s | 7.2 s | **6.3 s** |

The crossover is at a few hundred thousand image degrees of freedom, around `1,152 x 240`, and past it
the GPU wins by 1.6x in double precision and 1.9x in single. Below it the CPU is faster, sometimes by
more than a factor of two, because each L-BFGS iteration is a few dozen small kernels and launch
latency dominates a tensor this size. On a consumer card the double-precision rate is a sixty-fourth of
the single-precision rate, which is why float32 leads only where arithmetic, not latency, is the cost.

**The largest configuration did not finish.** `4,608 x 240` in float64 is 27 MB of chain, and L-BFGS
keeps a history of it: fifty vectors is 1.3 GB before the autograd graph. It sat at 100 per cent
utilization and 7.8 of the card's 8 GB for eight minutes without completing a hundred iterations, so it
was stopped. The binding constraint at that size is memory, not arithmetic, and the honest answer for a
batch that large on this card is to split it.

**What to take from this.** Use the batch lane always, and the device only when the batch is large:
`BatchImageOCPSolver(..., device="cpu")` is the right call for a sweep of a few hundred problems, which
is what most sweeps in front of this package are.

## Which lane publishes

The CPU lane. Every number this package's dependants publish is baked through
`ImageOCPSolver`, the reference implementation, and the batch lane is an accelerator for exploration
and sweeps whose agreement with that reference is measured here. CI does not install torch, and a GPU
lane is verified on a machine that has one rather than on every push (ADR-0074, CI/CD budget).
