# Tutorial notebooks -> .rst builder
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

rst_files := $(patsubst %.ipynb,$(DOC_ROOT)/tutorial/%.rst,$(wildcard *.ipynb))

.PHONY: tutorial clean

tutorial: $(rst_files)

$(DOC_ROOT)/tutorial/%.rst: %.ipynb
	$(PYTHON) $(BUILD_WRAP) tutorials $(WRAP_ARGS)

clean:
	$(PYTHON) $(BUILD_WRAP) clean
