# What the cost functional measures

This page exists because the most common way to be wrong in this field is to treat the switching
cost as an energy without saying what circuit turns it into one.

## `Phi` is not an energy

The functional is

```
Phi = int_0^T |b(t)|^2 dt
```

Its units are tesla-squared-seconds, not joules. It is proportional to the energy dissipated in the
circuit that generates the field `b`, because that circuit carries a current linearly related to `b`
and dissipates Joule heat as current squared, integrated over time. The proportionality constant has
units of joules per tesla-squared-second and encodes the inductor geometry, the resistance and the
coupling efficiency to the magnetic volume.

In `spinoct` that constant is `spinoct.units.CircuitModel.joules_per_tesla2_second`, and
`cost_to_joules` refuses to run without one. The signature is the guard: you cannot accidentally
convert a cost to joules by multiplying by a bare number, because the function raises `TypeError` if
you try.

## Four different "switching energies"

The literature calls at least four distinct quantities "the switching energy". `spinoct` keeps them
separate and every figure in the companion product names which one it shows.

1. **Source energy** is what `Phi` measures: the energy the drive circuit dissipates.
2. **Magnetic dissipation** is what the magnet itself loses through Gilbert damping, proportional to
   `alpha int |s'|^2 dt`. At zero damping this is exactly zero while `Phi` is not, a point the 2021
   PRL makes explicitly: a lossless magnet still requires a nonzero pulse, and that pulse still costs
   the source energy.
3. **Zeeman work** is the work the field does on the moment, which integrates to the energy
   difference between the endpoints, and is therefore zero for a symmetric reversal.
4. **Cell energy** is the device-level number quoted for STT-MRAM, SOT-MRAM or DRAM, dominated by
   the access transistor and interconnect and largely unrelated to the physics being optimized.

A comparison between a computed `Phi` (times an assumed circuit) and a measured cell energy is a
comparison between two different objects. The honest way to report it is as a band that carries both
the parameter uncertainty and the circuit-model uncertainty, never as a single headline factor.

## The universal floor is linear in the damping

The long-time limit of the minimum cost is

```
Phi_infinity = 4 alpha K / (gamma mu)
```

which is **linear in the Gilbert damping**. In two-dimensional van der Waals magnets the damping is
the least well pinned parameter of the whole problem, uncertain by an order of magnitude between
materials and often unmeasured. So any single-number claim about the reachable energy floor in these
materials is, at bottom, a claim about a damping nobody has measured to better than a factor of ten.
`spinoct` reports the floor and every cost that depends on it as a function of `alpha`, so the
uncertainty is visible rather than hidden in a point estimate.

## The Landauer reference line

`spinoct.units.landauer_limit_j` returns `k_B T ln 2`, which is 2.87 zeptojoules at 300 K. It is a
thermodynamic reference for the cost of **erasing** one bit. Switching a bit is not erasing it, so
the Landauer limit is a reference line to plot, never a target to claim proximity to. The package
provides it so the reference is drawn correctly, and the product's honesty contract forbids the
"closer to the Landauer limit" framing that appeared in press coverage of the source paper.

## Sources

- G. J. Kwiatkowski, M. H. A. Badarneh, D. V. Berkov, P. F. Bessarab, *Optimal Control of
  Magnetization Reversal in a Monodomain Particle by Means of Applied Magnetic Field*, Phys. Rev.
  Lett. 126, 177206 (2021), https://doi.org/10.1103/PhysRevLett.126.177206. The source-energy
  reading of the cost functional, and the observation that a lossless magnet still requires a
  nonzero, nonzero-cost pulse.
- M. H. Badarneh, P. Cai, E. J. G. Santos, *Optimal Control Drives Ultrafast and Energy-Efficient
  Magnetization Switching in Van der Waals Magnets*, Adv. Mater. e23059 (2026),
  https://doi.org/10.1002/adma.202523059. The application to two-dimensional magnets whose energy
  claims motivate the honesty contract on this page.
