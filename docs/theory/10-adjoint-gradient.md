# The discrete adjoint gradient

The finite-difference gradient the constrained solvers use costs one forward integration per control
parameter. The adjoint method computes the exact gradient with respect to every parameter in a single
backward pass, independent of the number of parameters. It is the classical route to gradient-based
optimal control, and `spinoct.adjoint` implements it in pure numpy.

## The method

The forward integration is a norm-projected Euler step of the Landau-Lifshitz-Gilbert equation with a
piecewise-constant control field. The objective is the switching cost plus a reversal-fidelity penalty.
The adjoint recursion propagates the cost sensitivity backward through the trajectory and reads off the
gradient with respect to each slice. Every Jacobian is analytic, because the LLG right-hand side is a
sum of cross products, which are linear in each argument, so no automatic-differentiation dependency is
needed and the core stays pure numpy. The same reverse pass runs elementwise, so it batches and ports
to a GPU tensor library unchanged.

## Validation

The decisive check is that the hand-derived reverse-mode gradient matches a finite-difference gradient
of the same objective. It does, to within one part in ten to the eighth across damping values
(`tests/test_adjoint.py`).

## The optimizer, and a unit-scale lesson

The exact gradient is fed to L-BFGS-B, which converges onto the analytic optimum: for a uniaxial
macrospin it finds a reversal at a switching cost within a few percent of the closed-form optimal cost,
in a fraction of a second.

Making that work required non-dimensionalizing the problem. The switching cost is of order
$10^{-12}$~T$^2$~s and the gradient of similar magnitude, so a general-purpose optimizer with default
absolute tolerances declares convergence before it takes a single step. Scaling the field by the
anisotropy field and the objective by the free-macrospin cost, so the optimizer works in order-unity
variables, fixes it. This is the same dimensional discipline the whole package enforces: a number
whose scale is not managed will be silently mishandled, here by the optimizer's own stopping rule.

## Sources

The adjoint approach for optimal control of far-from-equilibrium systems, including magnetic spin
lattices, is developed by Engel, Smith and Brenner, Optimal Control of Nonequilibrium Systems through
Automatic Differentiation, Phys. Rev. X 13, 041032 (2023),
https://doi.org/10.1103/PhysRevX.13.041032. The rigorous PDE-level optimal control of the
Landau-Lifshitz-Gilbert equation with a field control is treated by Patnaik and Sakthivel, Optimal
control of the 2D Landau-Lifshitz-Gilbert equation, Mathematical Control and Related Fields 15, 429
(2025), https://doi.org/10.3934/mcrf.2024018.
