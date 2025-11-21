from experiments.probes.state_probes import generate_cg_dataset, fit_linear_probe
import numpy as np


def test_negative_control_probe_behavior():
	# generate_cg_dataset returns (targets, acts_true, acts_ctrl)
	# num_tasks determines the number of samples.
	# we want enough samples to ensure control probe doesn't accidentally fit.
	# n=8, p=4. State dim is 3*n = 24.
	# num_tasks=20, steps=10 => 200 samples. 200 > 24.
	
	targets, acts_true, acts_ctrl = generate_cg_dataset(num_tasks=20, n=8, p=4, steps=10, seed=321)
	
	cos_true = fit_linear_probe(acts_true, targets)
	cos_ctrl = fit_linear_probe(acts_ctrl, targets)
	
	# True probe should recover state perfectly (since acts_true is linear transform of state)
	assert cos_true > 0.9
	# Control probe (random noise) should fail
	assert cos_ctrl < 0.5
