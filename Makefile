PY=python

.PHONY: baselines width_rank route_a route_a_plot smoke

baselines:
	$(PY) experiments/run_eval.py --target baselines

width_rank:
	$(PY) experiments/run_eval.py --target width_rank --plot

route_a:
	$(PY) experiments/run_eval.py --target route_a

route_a_plot:
	$(PY) experiments/route_a_minimal.py --plot hydra.run.dir=.

smoke:
	$(PY) - << 'PY'
print('Smoke OK')
PY
