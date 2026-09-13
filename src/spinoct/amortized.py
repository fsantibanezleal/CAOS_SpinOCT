"""An amortized learned policy for optimal switching pulses (R15).

Every solver in this package re-optimizes from scratch for each new set of material parameters. An
amortized policy instead learns the map from parameters to the optimal pulse once, and then emits a
near-optimal pulse for any new parameters instantly, with no optimization at inference. This is useful
when a pulse must be chosen online, and it is a clean testbed for a learned controller because the
uniaxial optimum is known in closed form, so the policy's claim is checkable rather than merely
plausible.

The output representation matters, and getting it wrong is instructive. Regressing a learned model onto
the raw pulse amplitude profile fails: switching is a threshold phenomenon, and a profile fit in a
least-squares sense is often a few percent too weak to complete the reversal, so the emitted pulse does
not switch even though it looks close. The fix is to amortize the map onto the pulse's single
physically-meaningful shape parameter, the elliptic parameter ``p`` that the closed-form solution is
built from. Any ``p`` yields a genuine optimal-control pulse that reverses the moment; a slightly wrong
``p`` yields the optimal pulse for a slightly different switching time, which still switches and is only
slightly suboptimal. So the shape-parameter policy is both reliable and near-optimal, while the
profile policy is neither. That contrast is the finding.

The policy is a small multilayer perceptron in numpy, so the package keeps no autodiff dependency. It
maps the two dimensionless parameters that determine the optimal uniaxial pulse, the damping ``alpha``
and the log switching time ``ln(T/tau0)``, to ``ln(p)``, and is trained by regression against the
analytic shape parameter on a grid. It is then validated the only honest way: the emitted pulse is fed
to the equation of motion and its cost and outcome are compared to the analytic optimum on parameter
combinations the policy never saw in training.

The pre-declared acceptance criterion (from the project's research dossier): a learned controller that
cannot reach the analytic optimum on the uniaxial case, where the optimum is known, has no business
being trusted on the harder cases where it is not. This module is that gate, and the shape-parameter
policy passes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic.uniaxial import UniaxialOptimalControl, solve_shape_parameter
from .dynamics.llg import integrate_llg_tabulated, switching_cost
from .dynamics.system import MacrospinSystem

__all__ = ["AmortizedPolicy", "PolicyEvaluation", "evaluate_policy", "train_amortized_policy"]

#: The number of amplitude samples the emitted pulse is reconstructed on for integration.
_N_SAMPLES = 200


def _features(system: MacrospinSystem, switching_time: float) -> np.ndarray:
    """The two dimensionless inputs the policy sees: ``[alpha, ln(T/tau0)]``."""
    return np.array([system.alpha, np.log(switching_time / system.tau0)])


@dataclass
class AmortizedPolicy:
    """A trained policy that emits the optimal-pulse shape parameter for given parameters.

    Attributes:
        w1, b1, w2, b2: the two-layer network weights and biases.
        input_mean, input_std: the input normalization.
        target_mean, target_std: the ``ln(p)`` target normalization.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float

    def _forward(self, features: np.ndarray) -> float:
        x = (features - self.input_mean) / self.input_std
        h = np.tanh(x @ self.w1 + self.b1)
        y = float(h @ self.w2 + self.b2)
        return y * self.target_std + self.target_mean

    def shape_parameter(self, system: MacrospinSystem, switching_time: float) -> float:
        """The predicted shape parameter ``p`` for a system and switching time."""
        return float(np.exp(self._forward(_features(system, switching_time))))

    def pulse(self, system: MacrospinSystem, switching_time: float) -> UniaxialOptimalControl:
        """The optimal-control pulse the policy emits, built from the predicted shape parameter."""
        return UniaxialOptimalControl(
            system=system, switching_time=switching_time, p=self.shape_parameter(system, switching_time)
        )


@dataclass(frozen=True)
class PolicyEvaluation:
    """The result of applying the policy's pulse to the equation of motion.

    Attributes:
        cost: the switching cost of the emitted pulse, T^2 s.
        analytic_cost: the closed-form optimal cost for the same system and switching time, T^2 s.
        cost_ratio: ``cost / analytic_cost``, at least 1 for a valid pulse up to discretization.
        predicted_p: the shape parameter the policy emitted.
        true_p: the exact shape parameter for these parameters.
        final_sz: the final z-component under the emitted pulse.
        switched: whether the emitted pulse reversed the moment.
    """

    cost: float
    analytic_cost: float
    cost_ratio: float
    predicted_p: float
    true_p: float
    final_sz: float
    switched: bool


def evaluate_policy(
    policy: AmortizedPolicy, system: MacrospinSystem, switching_time: float
) -> PolicyEvaluation:
    """Apply the policy's emitted pulse to the equation of motion and score it against the optimum.

    Args:
        policy: the trained policy.
        system: the macrospin (uniaxial).
        switching_time: ``T`` in s.

    Returns:
        The :class:`PolicyEvaluation`.
    """
    emitted = policy.pulse(system, switching_time)
    optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)
    grid = np.linspace(0.0, switching_time, _N_SAMPLES)
    field_table = emitted.field_vector(grid)
    trajectory = integrate_llg_tabulated(np.array([0.0, 0.0, 1.0]), field_table, grid, system)
    final_sz = float(trajectory[-1, 2])
    cost = switching_cost(grid, field_table)
    analytic_cost = optimal.cost()
    return PolicyEvaluation(
        cost=cost,
        analytic_cost=analytic_cost,
        cost_ratio=cost / analytic_cost if analytic_cost > 0 else float("inf"),
        predicted_p=emitted.p,
        true_p=optimal.p,
        final_sz=final_sz,
        switched=final_sz < 0.0,
    )


def train_amortized_policy(
    mu: float,
    anisotropy_j: float,
    alphas: np.ndarray,
    switching_times_tau0: np.ndarray,
    hidden: int = 24,
    epochs: int = 6000,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> AmortizedPolicy:
    """Train the amortized policy by regression onto the analytic shape parameter on a grid.

    Args:
        mu: magnetic moment, J/T (fixed across the training set).
        anisotropy_j: anisotropy energy, J (fixed).
        alphas: the damping values to train on.
        switching_times_tau0: the switching times to train on, in units of tau0.
        hidden: the hidden-layer width.
        epochs: the number of full-batch gradient steps.
        learning_rate: the step size.
        seed: the weight-initialization seed.

    Returns:
        The trained :class:`AmortizedPolicy`.
    """
    features = []
    targets = []
    for alpha in alphas:
        system = MacrospinSystem(mu=mu, anisotropy_j=anisotropy_j, alpha=float(alpha))
        for t_tau0 in switching_times_tau0:
            switching_time = system.switching_time_from_tau0(float(t_tau0))
            features.append(_features(system, switching_time))
            targets.append(np.log(solve_shape_parameter(switching_time, system)))
    features = np.array(features)
    targets = np.array(targets)

    input_mean, input_std = features.mean(0), features.std(0) + 1e-12
    target_mean, target_std = float(targets.mean()), float(targets.std() + 1e-12)
    x = (features - input_mean) / input_std
    y = (targets - target_mean) / target_std

    rng = np.random.default_rng(seed)
    n_in = x.shape[1]
    w1 = rng.normal(scale=1.0 / np.sqrt(n_in), size=(n_in, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(scale=1.0 / np.sqrt(hidden), size=hidden)
    b2 = 0.0

    n = x.shape[0]
    for _ in range(epochs):
        h_pre = x @ w1 + b1
        h = np.tanh(h_pre)
        pred = h @ w2 + b2
        error = pred - y
        grad_w2 = h.T @ error / n
        grad_b2 = float(error.mean())
        grad_h = np.outer(error, w2) * (1.0 - h**2)
        grad_w1 = x.T @ grad_h / n
        grad_b1 = grad_h.mean(0)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    return AmortizedPolicy(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
