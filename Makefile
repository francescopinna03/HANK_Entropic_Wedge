PY ?= python3

all: results.json results_micro.json results_economics.json figs/F4_risoluzione.pdf results_ge.json results_hank.json

results.json: model.py entropic.py run_all.py
	$(PY) run_all.py

results_micro.json: results.json run_micro.py
	$(PY) run_micro.py

results_ge.json: model.py entropic.py closure.py run_ge.py
	$(PY) run_ge.py

results_hank.json: model.py entropic.py closure.py run_ge.py
	$(PY) run_ge.py --closure hank

results_ge_%.json: model.py entropic.py closure.py run_ge.py
	HANK_CALIBRATION=$* $(PY) run_ge.py

results_hank_%.json: model.py entropic.py closure.py run_ge.py
	HANK_CALIBRATION=$* $(PY) run_ge.py --closure hank

results_channels_hank.json: model.py entropic.py closure.py channels.py run_channels.py
	$(PY) run_channels.py --closure hank

results_channels_hank_expectations.json: model.py entropic.py closure.py channels.py run_channels.py
	$(PY) run_channels.py --closure hank --expectations

results_exact_reopt.json: model.py exact.py channels.py run_exact_reopt.py
	$(PY) run_exact_reopt.py

results_economics.json: model.py economics.py plotstyle.py
	$(PY) economics.py

figs/F4_risoluzione.pdf: results.json results_micro.json figures.py plotstyle.py
	$(PY) figures.py

docs: all
	cd docs && latexmk -pdf -interaction=nonstopmode nota_codice.tex appendice_numerica_toy.tex

clean:
	rm -f *.npy results_hank*.json results_ge_*.json
	rm -rf __pycache__

.PHONY: all docs clean
