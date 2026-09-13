# The numerical optimal control path

For any system past the uniaxial macrospin there is no closed form, and the optimal control path must
be found numerically. `spinoct.numeric.ImageOCPSolver` implements the image-based direct
minimization of Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B 107, 214448 (2023),
https://doi.org/10.1103/PhysRevB.107.214448.

## The representation

The trajectory is a chain of `Q + 2` unit vectors (images) on the sphere. The two endpoints are the
initial and final states, held fixed. The `Q` interior images move, and the cost is minimized
directly over them. The discretized cost is the midpoint rule

```
Phi = sum_p |B_{p+1/2}|^2 (t_{p+1} - t_p)
```

where the field at each interval midpoint comes from inverting the equation of motion at the midpoint
position and velocity:

```
s_{p+1/2}     = (s_{p+1} + s_p) / |s_{p+1} + s_p|
s_dot_{p+1/2} = (delta_p / dt) (s_{p+1} - s_p) / |s_{p+1} - s_p|
```

with `delta_p` the angle between neighbouring images. The velocity magnitude is the finite-difference
angular velocity and its direction is the chord, which is orthogonal to the midpoint.

## The minimization

Descent runs on the curved manifold, the product of unit spheres. The gradient is projected onto each
image's tangent plane,

```
grad_perp_p = grad_p - s_p (s_p . grad_p)
```

and each step is taken in the tangent plane and mapped back to the sphere by normalization (the
first-order retraction). A backtracking line search sets the step, so no learning rate is tuned by
hand. Convergence is judged by the **relative** decrease in cost, which is dimensionless; a gradient
magnitude tolerance would be unit-dependent, since the gradient carries units of cost per radian and
the cost scale spans orders of magnitude with the switching time.

The gradient is local. An interior image appears in exactly two midpoint fields, so its gradient
depends only on its two neighbouring intervals. Exploiting that makes each iteration `O(Q)` rather
than the `O(Q^2)` of a naive full-cost finite difference, which is the difference between a solver
that runs in seconds and one that runs in minutes at the image counts the reference uses.

## Symmetry breaking is a correctness requirement, not a tuning knob

The pole-to-pole geodesic lies in a symmetry plane where the cost gradient vanishes to first order. A
solve started exactly on it reports the meridian as converged, even though the true optimal control
path precesses away from it and costs less. `ImageOCPSolver.solve` therefore perturbs the initial
chain by default, and this is not optional: the reference states it explicitly ("add small random
noise to avoid convergence on maxima or saddle points due to possible symmetries").

## Multiple optimal control paths coexist: use `solve_best`

For a biaxial system several optimal control paths coexist, and the asymmetric ones can be the global
optimum while the symmetric one is only a local minimum (the reference finds up to six for one
parameter set). A single seed reports whichever basin it fell into. `ImageOCPSolver.solve_best`
sweeps several seeds and keeps the lowest cost, which is the guard against silently reporting a local
minimum as the answer. This is the numerical embodiment of a lesson learned the hard way across these
projects: a fitted result that agrees with itself can still be wrong, so the optimum is sought from
several independent starts and cross-checked, here against the analytic uniaxial cost.

## What it reproduces

On a uniaxial system the numerical cost converges to the exact closed-form cost from above, with the
gap falling as the image count rises. That agreement is the acceptance gate in
`tests/test_image_ocp.py`. Only after it passes is the solver used on the biaxial and, later, the
lattice problems where the closed form does not exist and the numerical answer is the only one there
is.
