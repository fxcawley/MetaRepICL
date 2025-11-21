from experiments.probes.state_probes import generate_cg_states, fit_linear_probe
import numpy as np


def test_negative_control_probe_behavior():
	# Use n=8 (state dim 3*8=24) and steps=200 so that N >> D.
	# This ensures random noise cannot perfectly overfit the target.
	states, acts_true, acts_ctrl = generate_cg_states(n=8, p=4, steps=200, seed=321)
	target = []
	for (a, r, p) in states:
		target.append(np.concatenate([a, r, p]))
	target = np.array(target)
	cos_true = fit_linear_probe(acts_true, target)
	cos_ctrl = fit_linear_probe(acts_ctrl, target)
	assert cos_true > 0.9
	assert cos_ctrl < 0.5
