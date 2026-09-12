# The optimal control path

This is the object `spinoct` is built around. This page states it precisely, transcribed from the
primary sources, so the code and the physics can be read against each other.

## The problem

A magnetic moment has a direction `s`, a unit vector. It sits in one of two stable states and we
want to flip it to the other in a fixed time `T`, spending as little energy as possible in the
circuit that drives it. The dynamics obey the Landau-Lifshitz-Gilbert (LLG) equation, in the Gilbert
form used throughout this literature (Kwiatkowski, Badarneh, Berkov, Bessarab, Phys. Rev. Lett. 126,
177206 (2021), Eq. 2):

```
(1 + alpha^2) s' = -gamma s x (b_i + b) - alpha gamma s x [s x (b_i + b)]
```

with `alpha` the Gilbert damping, `gamma` the gyromagnetic ratio, `b_i = -(1/mu) dE/ds` the internal
(anisotropy) field, and `b` the applied control field. The cost is the Joule heating of the drive
circuit, which is proportional to

```
Phi = int_0^T |b(t)|^2 dt
```

See [what the cost functional measures](03-what-the-cost-functional-measures.md) for why `Phi` is
not, on its own, an energy.

## The trick that makes it tractable

Minimizing `Phi` over `(b, s)` subject to the LLG equation is a constrained problem. The key move is
to **invert the equation of motion**: solve it for `b` in terms of the trajectory,

```
b(s, s') = (alpha / gamma) s' + (1/gamma) [s x s'] - b_i_perp
```

where `b_i_perp` is the transverse part of the internal field (the longitudinal part does not affect
the dynamics). Substituting this into `Phi` turns it into a functional of the trajectory `s(t)`
alone, with no constraint left except `|s| = 1`. Its minimizer is the **optimal control path**
(OCP). The optimal pulse is recovered by putting the OCP back into the inversion.

In `spinoct` the inversion is `spinoct.dynamics.llg.field_from_trajectory`, and the test
`test_inverting_the_equation_of_motion_recovers_the_optimal_field` confirms that applying it to the
analytic path returns the analytic pulse.

## The OCP is not the minimum energy path

A tempting mistake is to imagine the cheapest reversal follows the lowest ridge over the energy
barrier, the minimum energy path (MEP). It does not. The OCP is a **dynamical** trajectory set by
the switching time and the damping, and it deliberately climbs **higher** than the saddle point when
doing so lets the material's own internal torque assist the reversal (Phys. Rev. Lett. 126, 177206,
Fig. 4). The MEP is a property of the energy surface alone; the OCP is a property of the whole
control problem. Contrasting them is one of the product's planned analyses, using Spirit's GNEB for
the MEP and `spinoct` for the OCP.

## The one case with a closed form

For a purely uniaxial magnet, `E = -K s_z^2`, the Euler-Lagrange equations separate in spherical
coordinates and the OCP is exact. The polar angle solves the sine-Gordon equation and is a Jacobi
amplitude; the pulse is a combination of `dn` and `sn`; and the cost is a combination of complete
elliptic integrals. All of it is in `spinoct.analytic.uniaxial`, and all of it is
[the package's positive controls](04-positive-controls.md). Everything numerical is checked against
this before it is trusted.

## Sources

- G. J. Kwiatkowski, M. H. A. Badarneh, D. V. Berkov, P. F. Bessarab, *Optimal Control of
  Magnetization Reversal in a Monodomain Particle by Means of Applied Magnetic Field*, Phys. Rev.
  Lett. 126, 177206 (2021). https://doi.org/10.1103/PhysRevLett.126.177206
- M. H. A. Badarneh, G. J. Kwiatkowski, P. F. Bessarab, *Reduction of energy cost of magnetization
  switching in a biaxial nanoparticle by use of internal dynamics*, Phys. Rev. B 107, 214448 (2023).
  https://doi.org/10.1103/PhysRevB.107.214448
