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
`spinoct.control.hybrid.integrate_llg_sot` integrates this with a norm-preserving RK4.

The equation is implicit, and the couplings in it are Gilbert-form couplings. Solving it for a unit
moment gives `(1 + alpha^2) s_dot = tau + alpha s x tau`, and the cross product mixes the two torque
channels, so the coefficients of the explicit equation are, each over `1 + alpha^2`,
```
field-like:    xi_F - alpha xi_D
damping-like:  xi_D + alpha xi_F.
```
The source's sweet spot is the check: at its ideal ratio `xi_D = -alpha xi_F` the explicit damping-like
term vanishes, which is why the source finds the problem "becomes identical to" field-driven switching,
and at its forbidden ratio `xi_F = alpha xi_D` the field-like term vanishes. The integrator applies
this conversion (`spinoct.dynamics.sot_torque.explicit_sot_coefficients`), and a test compares it with
a direct linear solve of the implicit equation to machine precision. It did not always: the first
integrator substituted the couplings directly into the explicit form while citing the implicit one,
which is 2 to 22 per cent off at `alpha = 0.1`, most where the damping-like coupling dominates. The
hybrid solver, its adjoint gradient, and the stochastic current ensemble all share the corrected term.

## The simplified chirped protocol (R04), and a replication that does not reproduce

At the ideal ratio the optimal current rotates at the precession frequency, and its frequency falls
through zero at the barrier crossing. The source replaces it by a rotating current of constant
amplitude whose frequency sweeps linearly from `f_max` to `-f_max` (its Eq. 15),
```
j(t) = j_s (cos Omega(t), sin Omega(t), 0),    Omega(t) = 2 pi f_max (t - t^2 / T),
```
implemented as `spinoct.analytic.sot.ChirpedRotatingCurrent`, with the source's `T = T0` and
`f_max = 1.4 f_r` available through `at_source_settings`. The source reports that at `alpha = 0.1` and a
thermal stability factor of 60 the switching probability is 0.89 at 0.17 `j0`, 0.97 at 0.18 `j0` and
practically one at 0.20 `j0`.

This engine does not reproduce those numbers. Over 1,000 stochastic copies per point
(`spinoct.thermal.sot_switching_success_rate`), at the same settings, it switches 0.9 per cent of
copies at 0.17 `j0`, 4.3 per cent at 0.18, 22 per cent at 0.20, 59 per cent at 0.22 and 90 per cent at
0.25: the same curve, shifted up by a factor of about 1.4 in amplitude. The zero-temperature
threshold is 0.21 `j0`. Everything within reach was checked and ruled out: only the co-rotating sense
switches at all; neither the starting tilt nor its azimuth changes the outcome up to 0.25 rad; the
Gilbert and the explicit coupling conventions agree to 0.01 in probability at the ideal ratio, because
the explicit damping-like term is nearly zero there in both; the threshold is lowest at 1.2 to 1.4
`f_r`, where the source placed `f_max`, so the chirp is tuned; the stochastic 50 per cent point sits on
the deterministic threshold, so the gap is not thermal; running the pulse past `T` changes nothing, as
the source says; and doubling the time unit removes switching altogether rather than lowering the
threshold. The current scale is the source's own, `j0 = K / (mu xi)`.

The gap is therefore either in a detail of the source's simulation that the text does not give (its
supplementary material is not available to us), or in this engine in a way none of the checks above
touches. It is recorded as a non-replication, with the published values pinned beside the engine's
behaviour in `tests/test_sot_chirp.py`, so a change on either side is visible. A useful check
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
