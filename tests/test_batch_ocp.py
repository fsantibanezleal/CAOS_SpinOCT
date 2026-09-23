"""The batched GPU lane, accepted only against the lanes that already have a gate.

A second implementation of a solved problem is a liability unless it is pinned to the first one. These
tests pin it twice: to the closed-form uniaxial cost, which is exact, and to the CPU solver problem by
problem on biaxial systems, where there is no closed form and the CPU lane is the reference every
published number came from.

torch is an optional dependency and CI does not install it, so the whole module skips without it. That
is deliberate: the lane is verified on a machine with a GPU, and its results are never the ones a
release publishes unless they match the CPU lane here.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl
from spinoct.dynamics import MacrospinSystem
from spinoct.numeric import BatchImageOCPSolver, ImageOCPSolver
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

torch = pytest.importorskip("torch", reason="the batch lane needs the optional spinoct[torch] extra")


def make_system(alpha: float, hard_axis_ratio: float = 0.0) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0),
        anisotropy_j=mev_to_joules(0.15),
        alpha=alpha,
        hard_axis_ratio=hard_axis_ratio,
    )


@pytest.fixture(scope="module")
def device() -> str:
    """The lane runs wherever it can; a machine without CUDA still tests the batching itself."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------- the acceptance gates


def test_a_batch_of_uniaxial_systems_reproduces_the_closed_form(device: str) -> None:
    """The keystone gate, batched: every row of one tensor lands on its own exact cost."""
    systems = [make_system(alpha) for alpha in (0.05, 0.1, 0.2, 0.3)]
    times = [s.switching_time_from_tau0(3.0) for s in systems]
    exact = [
        UniaxialOptimalControl.for_switching_time(s, t).cost()
        for s, t in zip(systems, times, strict=True)
    ]

    result = BatchImageOCPSolver(systems, times, n_images=80, device=device).solve(max_iterations=600)

    assert result.all_converged, f"residuals {result.residuals}"
    ratios = result.costs / np.asarray(exact)
    # Above the exact cost and converging to it from above, the same midpoint-discretization signature
    # the CPU lane is held to at the same image count.
    assert np.all(ratios >= 1.0) and np.all(ratios < 1.01), f"ratios {ratios}"


def test_the_batch_agrees_with_the_cpu_lane_problem_by_problem(device: str) -> None:
    """On biaxial systems there is no closed form, so the CPU solver is the reference. The two lanes
    start from the same chain and must land on the same cost; a disagreement here is a finding about
    one of them, not a tolerance to widen."""
    systems = [make_system(0.1, xi) for xi in (1.0, 3.0, 5.0)]
    times = [s.switching_time_from_tau0(5.0) for s in systems]

    batch = BatchImageOCPSolver(systems, times, n_images=60, device=device).solve(
        max_iterations=1500, seed=0, noise=0.05
    )
    for index, (system, time) in enumerate(zip(systems, times, strict=True)):
        single = ImageOCPSolver(system, n_images=60, switching_time=time).solve(
            max_iterations=4000, seed=0, noise=0.05
        )
        assert batch.costs[index] == pytest.approx(single.cost, rel=2e-3), (
            f"xi = {system.hard_axis_ratio}: batch {batch.costs[index]:.6e} against CPU {single.cost:.6e}"
        )


def test_the_autograd_gradient_matches_the_finite_difference_gradient(device: str) -> None:
    """The one place the two lanes are not the same computation. The CPU lane differences the cost
    locally; this one differentiates it. They have to agree, or the batch is minimizing a different
    functional from the one the product publishes."""
    system = make_system(0.1, 3.0)
    switching_time = system.switching_time_from_tau0(4.0)
    cpu = ImageOCPSolver(system, n_images=40, switching_time=switching_time)
    batch = BatchImageOCPSolver([system], switching_time, n_images=40, device=device)

    chain = cpu.solve(max_iterations=200, seed=1, noise=0.1).images
    finite_difference = cpu._tangent_gradient(chain)

    interior = torch.tensor(
        chain[None, 1:-1], dtype=torch.float64, device=device, requires_grad=True
    )
    cost = batch._costs_tensor(batch._chain_from(interior)).sum()
    (exact,) = torch.autograd.grad(cost, interior)
    unit = interior.detach()
    tangent = exact - torch.sum(exact * unit, dim=-1, keepdim=True) * unit
    autograd = tangent[0].cpu().numpy()

    scale = np.max(np.linalg.norm(finite_difference, axis=-1))
    worst = np.max(np.linalg.norm(autograd - finite_difference, axis=-1)) / scale
    # The CPU gradient is a central difference with a 1e-7 step on a unit vector, so its own truncation
    # and rounding floor is around 1e-6 relative; agreement at that level is agreement. Measured here:
    # 4.6e-06 of the gradient scale.
    assert worst < 1e-5, f"the two gradients differ by {worst:.2e} of the gradient scale"


def test_the_batch_costs_agree_with_the_cpu_cost_on_the_same_chain(device: str) -> None:
    """Before any minimization: the same chain must have the same cost in both lanes, or every later
    comparison is between two different functionals that happen to be close."""
    system = make_system(0.15, 2.0)
    switching_time = system.switching_time_from_tau0(7.0)
    cpu = ImageOCPSolver(system, n_images=50, switching_time=switching_time)
    batch = BatchImageOCPSolver([system, system], switching_time, n_images=50, device=device)

    chain = cpu.solve(max_iterations=50, seed=3, noise=0.2).images
    assert batch.cost_of(np.stack([chain, chain]))[0] == pytest.approx(cpu.cost_of(chain), rel=1e-12)


# ---------------------------------------------------------------------------- the reporting


def test_convergence_is_reported_per_problem(device: str) -> None:
    """A batch is minimized jointly, so a single verdict for all of it would hide the rows that did
    not make it. One iteration cannot converge anything, and the result has to say so."""
    systems = [make_system(0.1, xi) for xi in (0.0, 4.0)]
    times = [s.switching_time_from_tau0(20.0) for s in systems]
    stopped = BatchImageOCPSolver(systems, times, n_images=60, device=device).solve(max_iterations=1)
    assert not stopped.all_converged
    assert np.all(stopped.residuals > 1e-4)


def test_the_result_records_where_it_ran(device: str) -> None:
    """The GPU saying so is not provenance: a batched result carries its device and engine."""
    system = make_system(0.1)
    result = BatchImageOCPSolver(
        [system], system.switching_time_from_tau0(3.0), n_images=40, device=device
    ).solve(max_iterations=200)
    assert result.device == device
    assert result.engine.startswith("torch ")
    assert result.seconds > 0.0
    assert result.images.shape == (1, 42, 3)
    assert np.allclose(np.linalg.norm(result.images, axis=-1), 1.0)
    assert np.allclose(result.images[:, 0], [0.0, 0.0, 1.0])
    assert np.allclose(result.images[:, -1], [0.0, 0.0, -1.0])


def test_recommended_images_takes_the_largest_any_problem_asks_for() -> None:
    """One tensor means one discretization, and under-resolving the hard problem is the failure that
    matters. The batch resolution is the maximum, never the mean."""
    systems = [make_system(0.1), make_system(0.1)]
    times = [systems[0].switching_time_from_tau0(2.0), systems[1].switching_time_from_tau0(100.0)]
    singles = [
        ImageOCPSolver.recommended_images(system, time)
        for system, time in zip(systems, times, strict=True)
    ]
    assert BatchImageOCPSolver.recommended_images(systems, times) == max(singles)


def test_seed_sweeps_keep_each_problems_cheapest_path(device: str) -> None:
    """``solve_best`` is where the lane pays: the seeds are more rows of one tensor. Each problem must
    keep its own cheapest path and record which seed found it, not the batch's best seed."""
    systems = [make_system(0.1, xi) for xi in (3.0, 6.0)]
    times = [s.switching_time_from_tau0(6.0) for s in systems]
    solver = BatchImageOCPSolver(systems, times, n_images=60, device=device)
    best = solver.solve_best(n_seeds=3, base_seed=0, noise=0.15, max_iterations=800)

    for seed in range(3):
        single = solver.solve(seed=seed, noise=0.15, max_iterations=800)
        assert np.all(best.costs <= single.costs * (1.0 + 1e-12))
    assert set(np.unique(best.seeds)).issubset({0, 1, 2})


def test_a_batch_rejects_a_mismatched_time_list() -> None:
    systems = [make_system(0.1), make_system(0.2)]
    with pytest.raises(ValueError, match="one entry per system"):
        BatchImageOCPSolver(systems, [1e-9, 2e-9, 3e-9], n_images=10)
