PY ?= python3

all: results.json results_micro.json results_economics.json figs/F4_risoluzione.pdf

results.json: model.py entropic.py run_all.py
	$(PY) run_all.py

results_micro.json: results.json run_micro.py
	$(PY) run_micro.py

results_economics.json: model.py economics.py plotstyle.py
	$(PY) economics.py

figs/F4_risoluzione.pdf: results.json results_micro.json figures.py plotstyle.py
	$(PY) figures.py

docs: all
	cd docs && latexmk -pdf -interaction=nonstopmode nota_codice.tex appendice_numerica_toy.tex

clean:
	rm -f *.npy results*.json
	rm -rf __pycache__

.PHONY: all docs clean
