# Conventions and units

This page fixes the conventions `spinoct` uses internally, because the same physical quantity is
written at least four ways across this literature and a number moved between two of them without
conversion is silently wrong while still producing a plausible plot.

## The dimensional contract

Everything the package passes around carries a declared unit. The table is in `spinoct.units.UNITS`
and is asserted by `self_check()`.

| Symbol | Meaning | Unit |
|---|---|---|
| `mu` | magnetic moment magnitude | J/T (equivalently A m^2) |
| `gamma` | gyromagnetic ratio | rad/(s T) |
| `alpha` | Gilbert damping | dimensionless |
| `K` | anisotropy energy scale | J |
| `xi` | hard-axis to easy-axis ratio | dimensionless |
| `b`, `B` | magnetic field | T |
| `t`, `T` | time, switching time | s |
| `tau0 = mu/(2 gamma K)` | Larmor timescale | s |
| `K/mu` | anisotropy field | T |
| `Phi = int abs(b)^2 dt` | switching cost | T^2 s |

`tau0` is sub-picosecond for the two-dimensional magnets of interest, and `self_check()` asserts a
computed `tau0` falls in `[1e-15, 1e-9]` s. A value outside that window means the inputs were handed
over in the wrong units, which is caught immediately rather than after a whole bake.

## The exchange sign trap

Exchange constants appear in the literature in incompatible conventions. The same coupling is written
as any of

```
H = +sum_<ij> J S_i . S_j        (J < 0 means ferromagnetic; used by Scheie et al. for CrSBr)
H = -sum_<ij> J S_i . S_j        (J > 0 means ferromagnetic)
H = -(1/2) sum_{i!=j} J S_i . S_j    (double counting absorbed into the one half)
H = -sum_{i<j} J e_i . e_j       (unit vectors, S absorbed into J; VAMPIRE's convention)
```

and the spin length matters: for Cr(III) in CrSBr, `S = 3/2`, so a `J` quoted per `S.S` differs from
one quoted per unit-vector by `S^2 = 2.25`. Anisotropy is quoted as meV per atom, meV per formula
unit, joules per cubic metre, or as an effective field in tesla.

The rule in `spinoct` and in the Espira parameter database: every parameter carries its convention,
its spin length, its per-what, its units, its source DOI and its uncertainty, plus a value
canonicalized to one internal form. A unit round-trip gate asserts the conversion. This is the guard
against the failure recorded in the CAOS conventions as constants written for one observable's units
and then applied to another's.

## Constants

`spinoct.units` holds CODATA 2022 recommended values (Bohr magneton, electron gyromagnetic ratio,
reduced Planck constant, elementary charge) and the exact SI definitions (Boltzmann constant). They
are the only place bare physical numbers live, and `scripts/check_unit_constants.py` enforces that no
other module introduces an unreviewed dimensional constant.

## Sources

- A. Scheie et al., *Spin Waves and Magnetic Exchange Hamiltonian in CrSBr*, Adv. Sci. 9, 2202467
  (2022), https://doi.org/10.1002/advs.202202467. Uses the `+sum J S.S` convention with `S = 3/2`;
  the eighth-neighbour exchange set this package's parameter database canonicalizes.
- R. F. L. Evans et al., *Atomistic spin model simulations of magnetic nanomaterials*, J. Phys.:
  Condens. Matter 26, 103202 (2014), https://doi.org/10.1088/0953-8984/26/10/103202. VAMPIRE, whose
  unit-vector exchange convention is one of the four this page reconciles.
