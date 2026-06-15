# Docstring notebooks -> .rst builder
#
# This delegates to doc/build.py, which is the unified entry point for all
# doc builds. It ensures SEABORN_DATA, random seed, matplotlib backend, and
# image params are the same everywhere (local and CI).

PYTHON     ?= python
DOC_ROOT   := $(abspath $(CURDIR)/..)
BUILD_WRAP  = $(DOC_ROOT)/build.py

JOBS       ?=
SPHINXOPTS ?=

WRAP_ARGS :=
ifneq ($(JOBS),)
    WRAP_ARGS += --jobs $(JOBS)
endif
ifneq ($(SPHINXOPTS),)
    WRAP_ARGS += --sphinx-opts "$(SPHINXOPTS)"
endif

rst_files := $(patsubst %.ipynb,$(DOC_ROOT)/docstrings/%.rst,$(wildcard *.ipynb))

.PHONY: docstrings clean

docstrings: $(rst_files)

$(DOC_ROOT)/docstrings/%.rst: %.ipynb
	$(PYTHON) $(BUILD_WRAP) docstrings $(WRAP_ARGS)

clean:
	$(PYTHON) $(BUILD_WRAP) clean
