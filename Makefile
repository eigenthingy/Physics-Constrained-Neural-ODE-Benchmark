PY ?= python
SHARDS ?= 8

.PHONY: help install test demo quick grid figures all clean

help:
	@echo "make install   - pip install -e .[dev]"
	@echo "make test      - pytest (includes the conservation-guarantee tests)"
	@echo "make demo      - 60s: train all four models on the pendulum, print the table"
	@echo "make quick     - ~5 min reduced grid, then figures"
	@echo "make grid      - the full study, SHARDS=$(SHARDS) parallel workers"
	@echo "make figures   - rebuild every figure from results/"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

demo:
	$(PY) scripts/demo.py

quick:
	$(PY) scripts/run_experiments.py --quick
	$(PY) scripts/make_figures.py

grid:
	@for i in $$(seq 0 $$(( $(SHARDS) - 1 ))); do \
		OMP_NUM_THREADS=1 PCNN_THREADS=1 $(PY) scripts/run_experiments.py \
			--shard $$i --n-shards $(SHARDS) > results/logs/shard$$i.log 2>&1 & \
	done; wait
	$(PY) scripts/make_figures.py

figures:
	$(PY) scripts/make_figures.py

all: test grid

clean:
	rm -rf results/models results/logs figures/*.png
