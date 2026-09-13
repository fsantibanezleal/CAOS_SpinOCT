# Joint field-plus-current optimal control

The kickoff paper ends by naming a design space and leaving it open: optimal-control field
implementations "could complement or hybridize with current- and light-driven approaches ... where
magnetic switching is tailored by the interplay of fields, currents, and photons" (Badarneh, Cai,
Santos, Adv. Mater. e23059, 2026). The field-only problem and the current-only problem are each solved
in closed form; the joint problem is not. `spinoct.control.hybrid` solves it numerically.

## The two-term cost

The hybrid weighs the Joule heating of each source separately,
```
Phi = C_b integral |b|^2 dt  +  C_j integral |j|^2 dt,
```
with `C_b` and `C_j` the circuit constants of the field and current sources. Their ratio is the design
knob: it sets how expensive a tesla of field is relative to an amp of current, and sweeping it traces
whether a hybrid ever undercuts the better of the two pure protocols. The honest prior is that it might
not, and a null result would be a real finding.

## Dynamics

The spin-orbit torque enters the Landau-Lifshitz-Gilbert equation through a field-like and a
damping-like term driven by an in-plane current (Vlasov et al., Phys. Rev. B 105, 134404, 2022,
https://doi.org/10.1103/PhysRevB.105.134404):
```
s_dot = -gamma s x b_tot + alpha s x s_dot
        + gamma xi_F s x (j x e_z) + gamma xi_D s x [s x (j x e_z)].
```
`spinoct.control.hybrid.integrate_llg_sot` integrates this with a norm-preserving RK4. A useful check
of the SOT sign: a pure damping-like current drives the moment from the pole toward the equator but
cannot complete the reversal on its own, which is exactly why conventional spin-orbit-torque switching
needs a symmetry-breaking field, and part of why the hybrid is interesting.

## The solver and what it finds

`HybridSolver` co-optimizes a band-limited field and a band-limited current at once, with both pulses
expanded in a few Fourier harmonics so they stay realizable. For the balanced case (`C_b = C_j`) it
finds a reversing protocol that spends on both controls, a genuine hybrid rather than a degenerate
field-only or current-only solution. Because the SOT dynamics is integrated in pure Python, the solver
runs at modest scale, so it is a reference implementation for the design-space question rather than a
tool for a large sweep; the tractable pure protocols carry the material-scale results.
