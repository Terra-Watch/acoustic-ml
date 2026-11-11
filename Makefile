PY ?= .venv/bin/python

test:
	$(PY) -m pytest -q
check:  # every file compiles, no audio needed
	$(PY) -m compileall -q src scripts tests
