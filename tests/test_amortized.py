"""The amortized learned policy (R15), gated against the analytic optimum.

The pre-declared acceptance criterion: a learned controller that cannot reach the analytic optimum on
the uniaxial case, where the optimum is known in closed form, cannot be trusted on harder cases. The
policy that amortizes onto the pulse's shape parameter passes this gate; the tests enforce it.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.amortized import evaluate_policy, train_amortized_policy
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

MU = bohr_magnetons_to_j_per_t(3.0)
K = mev_to_joules(0.15)
ALPHAS_TRAIN = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
TIMES_TRAIN = np.array([2.0, 4.0, 8.0, 16.0, 32.0])


def trained_policy():
    return train_amortized_policy(MU, K, ALPHAS_TRAIN, TIMES_TRAIN, epochs=8000, seed=0)


def system(alpha: float) -> MacrospinSystem:
    return MacrospinSystem(mu=MU, anisotropy_j=K, alpha=alpha)


@pytest.mark.parametrize("alpha,t_tau0", [(0.07, 6.0), (0.15, 12.0), (0.3, 20.0)])
def test_policy_reverses_the_moment_on_held_out_parameters(alpha: float, t_tau0: float) -> None:
    """The emitted pulse actually switches the moment for parameters not seen in training."""
    policy = trained_policy()
    sys = system(alpha)
    switching_time = sys.switching_time_from_tau0(t_tau0)
    result = evaluate_policy(policy, sys, switching_time)
    assert result.switched
    assert result.final_sz < -0.9


@pytest.mark.parametrize("alpha,t_tau0", [(0.07, 6.0), (0.15, 12.0), (0.1, 10.0)])
def test_policy_reaches_near_optimal_cost(alpha: float, t_tau0: float) -> None:
    """The emitted pulse costs within a modest tolerance of the analytic optimum (the acceptance gate)."""
    policy = trained_policy()
    sys = system(alpha)
    switching_time = sys.switching_time_from_tau0(t_tau0)
    result = evaluate_policy(policy, sys, switching_time)
    assert 0.85 < result.cost_ratio < 1.15


def test_predicted_shape_parameter_is_close_to_the_true_one() -> None:
    policy = trained_policy()
    sys = system(0.12)
    switching_time = sys.switching_time_from_tau0(9.0)
    result = evaluate_policy(policy, sys, switching_time)
    assert result.predicted_p == pytest.approx(result.true_p, rel=0.25)


def test_training_is_reproducible() -> None:
    a = trained_policy()
    b = trained_policy()
    np.testing.assert_array_equal(a.w1, b.w1)
    np.testing.assert_array_equal(a.w2, b.w2)
