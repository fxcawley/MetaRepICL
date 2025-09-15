import numpy as np
from experiments.width_rank import run_width_rank


def test_width_rank_error_monotone_on_average():
	res = run_width_rank(ms=[8, 16, 24, 32])
	errs = res["pred_err"]
	assert len(errs) == 4
	# Weak check: last error is <= first error
	assert errs[-1] <= errs[0] + 1e-9
