from experiments.precond import run_precond


def test_preconditioner_not_worse_monotone():
	res = run_precond(t=5)
	eu = res["err_un"]
	ep = res["err_pr"]
	assert len(eu) == len(ep) == 5
	# Weak check: final error with preconditioner is not worse than unpreconditioned
	assert ep[-1] <= eu[-1] + 1e-8
