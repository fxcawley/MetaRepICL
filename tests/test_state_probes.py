from experiments.probes.state_probes import generate_cg_states, fit_linear_probe
import numpy as np


def test_negative_control_probe_behavior():
	states, acts_true, acts_ctrl = generate_cg_states(seed=321)
	target = []
	for (a, r, p) in states:
		target.append(np.concatenate([a, r, p]))
	target = np.array(target)
	cos_true = fit_linear_probe(acts_true, target.flatten())
	cos_ctrl = fit_linear_probe(acts_ctrl, target.flatten())
	assert cos_true > 0.9
	assert cos_ctrl < 0.3
