# Beyond the macrospin: the spin chain

The macrospin approximation treats a magnetic element as a single rotating moment. The authors of the
optimal-control method state, in print, that this breaks down with size and that the reversal may
instead involve nonuniform rotation, domain-wall nucleation and propagation, or spin waves, and that
"it remains to be seen under what conditions these ... switching mechanisms become optimal in terms of
energy efficiency" (Phys. Rev. B 107, 214448, 2023). That open problem is the reason `spinoct.lattice`
exists.

## The chain

`spinoct.lattice.SpinChain` is a 1D ferromagnetic chain of unit moments with nearest-neighbour exchange
and uniaxial anisotropy,

```
E = -K sum_i s_{i,z}^2  -  J sum_<ij> s_i . s_j
```

The internal field at a site adds an exchange term `(J/mu)(s_{i-1} + s_{i+1})` to the single-site
anisotropy field. In the limit of one site with no exchange the chain IS a macrospin, and the lattice
cost reproduces the analytic macrospin cost exactly, which is the validation gate.

## Two reversal modes

The chain can reverse from all-up to all-down in two qualitatively different ways:

- **Uniform rotation.** Every site follows the same macrospin optimal control path together. Exchange
  is never paid, because neighbours stay parallel throughout, and the cost scales linearly in the
  number of sites.
- **Domain-wall sweep.** A reversed domain nucleates at one end and its wall propagates to the other.
  Neighbours across the wall are not parallel, so exchange is paid, and only a few sites move at once,
  but they must flip fast for the wall to cross the chain in time.

The cost of each is computed the same way as everywhere in the package: build the trajectory, invert
the equation of motion at every site to get the field that site needs (with exchange in the internal
field), and integrate the summed squared field.

## The result: uniform rotation is the field-cost optimum

For the switching-COST metric, the Joule heating of the source, uniform rotation is cheaper than a
domain-wall sweep across every chain length (2 to 128 sites) and exchange strength (J/K from 0.2 to 10)
tested. The reason is physical: the domain wall forces fast local flips and bends the bonds across
itself, while uniform rotation moves every site slowly and never pays exchange. The wall's cost grows
at least as fast as the uniform cost with length, so lengthening the chain does not make the wall the
optimum in this ansatz.

This is worth stating carefully, because it is the opposite of a common intuition. Domain-wall motion
dominates real, thermally-driven magnetic switching, but that is because the wall lowers the energy
BARRIER for nucleation and propagation, a thermal-stability question. It does not lower the field COST,
the Joule heating, which is what optimal control minimizes here. For the cost functional, moving
everything slowly and uniformly is the cheap thing to do.

## What this is and is not

This is a comparison of two specific reversal modes, a uniform rotation and a constant-speed tanh
domain wall, not a free search over all chain trajectories. Its conclusion holds only for that ansatz.

**Superseded by the free search.** The free chain optimal control path
([12](12-free-chain-optimal-control-and-the-barrier-floor.md)) finds that above a crossover length and
at long switching time the optimal reversal IS a domain wall, strictly cheaper than uniform rotation
(at least 16 percent for 16 sites, J/K = 10, alpha = 0.1, T = 150 tau0, verified by grid refinement,
local consistency and forward dynamics). The constant-speed wall of this page is not the optimal wall.

## Sources

- M. H. A. Badarneh, G. J. Kwiatkowski, P. F. Bessarab, Reduction of energy cost of magnetization
  switching in a biaxial nanoparticle by use of internal dynamics, Phys. Rev. B 107, 214448 (2023),
  https://doi.org/10.1103/PhysRevB.107.214448. States the beyond-macrospin problem as future work.
- M. H. A. Badarneh, G. J. Kwiatkowski, P. F. Bessarab, Mechanisms of energy-efficient magnetization
  switching in a bistable nanowire, Nanosystems: Physics, Chemistry, Mathematics 11, 294 (2020). The
  single prior 1D study, which found a standing spin wave above a critical length.
