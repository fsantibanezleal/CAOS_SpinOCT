# Thermal dynamics and the cost of reliability

The optimal control paths are computed at zero temperature. Whether a pulse actually switches a bit at
a finite temperature is a separate, statistical question, and it is where a protocol's reliability is
decided. `spinoct.thermal` adds the fluctuating thermal field to the equation of motion and measures
the switching success rate, then quantifies the cost of making a pulse reliable.

## The thermostat

The fluctuation-dissipation theorem fixes the strength of the thermal field: the same Gilbert damping
that dissipates energy injects white noise with covariance

```
<b_th_i(t) b_th_j(t')> = (2 alpha k_B T) / (gamma mu) delta_ij delta(t - t')
```

`spinoct.thermal.stochastic` integrates this with a stochastic Heun scheme, renormalizing each step.
The decisive check is that it reproduces the Boltzmann distribution it must: near a uniaxial minimum
the mean square polar angle is `k_B T / K`, and the thermostat recovers that to ten percent
(`tests/test_thermal.py`). Reference for the thermostat: Evans et al., J. Phys.: Condens. Matter 26,
103202 (2014), https://doi.org/10.1088/0953-8984/26/10/103202.

## The reliability problem, and the cost of solving it

The optimal pulse is always perpendicular to the moment, so it does nothing to damp the perturbations
thermal noise excites. Linearizing about the optimal control path gives a two-component perturbation
governed by the eigenvalues of the energy Hessian shifted by the longitudinal field `B_r`:

```
w1 = B_r + (K / mu) cos(2 theta)
w2 = B_r + (K / mu) cos^2(theta)
```

Perturbations are bounded when `w1 w2 > 0` and divergent (hyperbolic) when `w1 w2 <= 0`. At `B_r = 0`
half the reversal, `pi/4 < theta < 3pi/4`, is hyperbolic, and that hyperbolicity, not the energy
barrier, is the primary cause of the pulse and the moment losing phase lock (Badarneh, Kwiatkowski and
Bessarab, arXiv:2312.11293, 2023). Adding a longitudinal field with `|B_r| > K/mu` removes the
hyperbolic domain and drives the success rate to unity.

But `B_r` is not free. Because the optimal pulse is perpendicular, `B_r` is invisible to the
leading-order dynamics, yet it still costs `integral B_r^2 dt`. That trade between reliability and cost
is the object `spinoct.thermal.stabilize.br_cost_reliability_front` quantifies, and it is not in the
literature. The reproduced physics and the new number:

- the bare path is dynamically unstable over half its length, and stabilizes to zero hyperbolic
  fraction once `B_r >= K/mu`;
- the success rate has a counterintuitive dip near `B_r ~ 0.5 K/mu`, where `w1/w2 = -1` at the barrier
  top, before recovering to unity at larger field, exactly as the reference reports;
- the added cost grows as the square of the field, which is the price of that reliability, computed
  here for the first time.

## The instability-penalized optimal control path

`spinoct.thermal.stabilize.instability_penalty` is the deterministic hyperbolicity integral
`integral max(0, -w1 w2) dt` along a path. It costs nothing extra to evaluate, because `w1` and `w2`
come from the same Hessian the solver already uses. The proposed claim, whose test is the natural next
experiment, is that penalizing it predicts the Monte-Carlo success rate without running an ensemble, so
a cheap deterministic term substitutes for an expensive stochastic sweep.
