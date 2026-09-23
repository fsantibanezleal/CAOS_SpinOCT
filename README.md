# spinoct

Optimal control paths and energy-efficient switching pulses for classical spin dynamics
(Landau-Lifshitz-Gilbert).

`spinoct` computes the control (an applied magnetic field, an electric current, or both) that
drives a magnetic moment from one state to another in a given time for the least dissipated energy.
It is the reusable engine behind [Espira](https://github.com/fsantibanezleal/CAOS_RES_Espira), and
it is deliberately independent of any material database, so it works on any spin Hamiltonian.

## Why this exists

The optimal control of magnetization switching has a small, rigorous literature (Kwiatkowski,
Badarneh, Berkov and Bessarab, Phys. Rev. Lett. 126, 177206 (2021), and the papers that follow it),
but no open implementation. The analytic solutions live only as equations in journal articles, and
the group's own code is not public. `spinoct` is that implementation, with the closed-form results
built in as positive controls that every numerical solver is validated against before it is trusted.

## Install

```bash
pip install spinoct              # core: numpy + scipy
pip install "spinoct[torch]"     # add the batched GPU solvers
```

## The dimensional contract

Read `spinoct.units` first. The same physical quantity is written four different ways across this
literature, and the central quantity of the package, the switching cost `Phi = int |b|^2 dt`, is in
tesla-squared-seconds, not joules. It becomes an energy only through an explicit circuit model. The
package refuses to hide that assumption: `spinoct.units.CircuitModel` is a required, described
object, never a buried constant.

## Status

Pre-1.0, under active development. The analytic uniaxial optimal control path, its closed-form pulse
and cost, and the negative-parameter Jacobi elliptic machinery it needs are complete and validated, as
are the numerical image-based solver and the constrained solvers. GRAPE, CRAB and the field-plus-current
hybrid are driven by the exact adjoint gradient through their linear control bases, and each answer is
required to be a real reversal before its cost is reported; see
`docs/theory/13-constrained-control-and-the-price-of-realizability.md` for what they measure and for the
two defects that shipped before they did. The batched lane (`spinoct[torch]`) solves many
independent optimal control problems as one tensor, on a GPU when one is present, and is accepted
against the CPU lane problem by problem; `docs/theory/14-the-batched-lane.md` has what it is for,
what it is not for, and where the crossover actually is.

## License

MIT. See `LICENSE`.
