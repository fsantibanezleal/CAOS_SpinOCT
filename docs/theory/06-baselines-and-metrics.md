# Baselines and the metric suite

Every optimal control result is only as honest as the baseline it beats. This page describes the
conventional protocols the package implements and the metrics it scores everything on.

## The conventional baselines

- **Static antiparallel field** (`control.ConstantFieldProtocol`). A constant field in the reversing
  hemisphere, above the switching field `K/mu`. The textbook Stoner-Wohlfarth reversal: it works but
  is slow and damping-limited, and it costs far more than an optimal pulse for the same switching
  time.
- **Sun-Wang minimal field** (`control.sun_wang_minimal_field`). The theoretical minimum
  constant-amplitude switching field, half the anisotropy field at the astroid-optimal 45-degree
  direction (Sun and Wang, Phys. Rev. Lett. 97, 077205 (2006),
  https://doi.org/10.1103/PhysRevLett.97.077205). The strongest constant-field baseline, and the fair
  static comparison.
- **Precessional** (`control.PrecessionalProtocol`). A transverse field pulse whose duration is tuned
  so precession carries the moment across the barrier, then removed. Faster than a static reversal but
  duration-sensitive, which is exactly why it is a baseline and not an optimum: sweep the duration and
  some values reverse cleanly while others do not.

The reduction factor a product quotes must be against the strongest of these, never the weakest.
Quoting only against the static field flatters the optimal result.

## The analytic SOT protocol

`analytic.SOTOptimalControl` carries the closed-form results of Vlasov, Kwiatkowski, Lobanov, Uzdin
and Bessarab, Phys. Rev. B 105, 134404 (2022), https://doi.org/10.1103/PhysRevB.105.134404, for
spin-orbit-torque switching driven by an electric current. The field-like and damping-like torques
are balanced by an angle `beta`, and two ratios matter: the ideal ratio `xi_D = -alpha xi_F`
(`beta* = -arctan(alpha)`), for which the current torque points entirely along the switching direction
and the problem collapses onto the field-driven one, and the forbidden ratio `xi_F = alpha xi_D`, for
which the cost diverges and no switching is possible. The average current at the ideal ratio scales
linearly in the damping, which is why optimal SOT can undercut the conventional critical current at
low damping.

## The metrics

`metrics.ProtocolMetrics` scores every protocol on the same quantities, dimensionless where possible
so they compare across materials:

- `cost`, the switching cost in tesla-squared-seconds.
- `cost_over_floor`, relative to the universal long-time floor `4 alpha K / (gamma mu)`.
- `cost_over_free`, relative to the free-macrospin cost. Below one means the material's internal
  dynamics is paying for part of the reversal, which a uniaxial magnet can never achieve and a hard
  axis can.
- `peak_amplitude`, the largest field magnitude, the hardware feasibility constraint.
- `bandwidth_hz`, the 99 percent spectral energy bandwidth, the realizability constraint that the
  constrained solvers exist to respect.
- `switched` and `final_sz`, whether and how completely the reversal completed.
