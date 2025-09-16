PY=python

.PHONY: baselines width_rank route_a smoke

baselines:
	$(PY) experiments/run_eval.py --target baselines

width_rank:
	$(PY) experiments/run_eval.py --target width_rank --plot

route_a:
	$(PY) experiments/run_eval.py --target route_a

smoke:
	$(PY) - << 'PY'
print('Smoke OK')
PY
