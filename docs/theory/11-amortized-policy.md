# The amortized learned policy

Every solver in this package re-optimizes from scratch for each new set of parameters. An amortized
policy instead learns the map from parameters to the optimal pulse once, and then emits a near-optimal
pulse for any new parameters instantly, with no optimization at inference. The uniaxial macrospin is a
clean testbed for it, because the optimum is known in closed form, so the policy's claim is checkable
rather than plausible.

## The output representation is the whole game

Regressing a learned model onto the raw pulse amplitude profile fails, and the failure is instructive.
Switching is a threshold phenomenon: a profile fit in a least-squares sense is often a few percent too
weak to complete the reversal, so the emitted pulse does not switch even though it looks close. We
observed exactly this, including at training points.

The fix is to amortize onto the pulse's single physically-meaningful shape parameter, the elliptic
parameter that the closed-form solution is built from. Any value of it yields a genuine optimal-control
pulse that reverses the moment; a slightly wrong value yields the optimal pulse for a slightly
different switching time, which still switches and is only slightly suboptimal. So the shape-parameter
policy is both reliable and near-optimal, while the profile policy is neither. Choosing the
physically-informed representation is what makes amortization work.

## The policy and its gate

The policy is a small two-layer perceptron in numpy (no autodiff dependency), mapping the damping and
the log switching time to the log shape parameter, trained by regression on a grid. On parameter
combinations it never saw in training it emits pulses that reverse the moment at a cost within about
ten percent of the analytic optimum. That is the pre-declared acceptance gate from the research
dossier: a learned controller that cannot reach the analytic optimum on the uniaxial case, where the
optimum is known, cannot be trusted on the harder cases where it is not. The shape-parameter policy
passes it; the profile policy does not.

## Sources

The amortized-neural-operator framing for optimal control is developed in Self-Supervised Amortized
Neural Operators for Optimal Control, arXiv:2512.24897. Reinforcement-learning controllers for
magnetization switching by spin-orbit torque, which target speed and field-free operation rather than
energy optimality against a known floor, appear in Phys. Rev. B (2026),
https://doi.org/10.1103/7hqy-q23t.
