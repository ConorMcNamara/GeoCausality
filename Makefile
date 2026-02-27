.DEFAULT_GOAL := help

# ── variables ────────────────────────────────────────────────────────────────
PYTHON      := python3
SRC         := GeoCausality
TESTS       := test
SPHINXBUILD := sphinx-build
BUILDDIR    := _build
SPHINXOPTS  :=
ALLSPHINXOPTS := -d $(BUILDDIR)/doctrees $(SPHINXOPTS) docs

# ── help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Development"
	@echo "  install       Install runtime dependencies"
	@echo "  install-dev   Install runtime + dev dependencies (pytest, ruff, mypy)"
	@echo "  format        Auto-format code with ruff"
	@echo "  lint          Check code style with ruff"
	@echo "  type-check    Run mypy static type checker"
	@echo "  check         Run lint + type-check (mirrors CI)"
	@echo "  test          Run the test suite with pytest + coverage summary"
	@echo "  coverage      Run tests and open an HTML coverage report"
	@echo "  clean         Remove build artefacts and caches"
	@echo ""
	@echo "Documentation"
	@echo "  docs          Build HTML documentation with Sphinx"
	@echo "  docs-clean    Remove the Sphinx build directory"

# ── install ───────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PYTHON) -m pip install -e .

.PHONY: install-dev
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install ruff mypy pydocstringformatter

# ── code quality ─────────────────────────────────────────────────────────────
.PHONY: format
format:
	ruff format $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)

.PHONY: lint
lint:
	ruff check $(SRC) $(TESTS)

.PHONY: type-check
type-check:
	mypy $(SRC) --ignore-missing-imports

.PHONY: check
check: lint type-check

# ── tests ─────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTHON) -m pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing

.PHONY: coverage
coverage:
	$(PYTHON) -m pytest $(TESTS) -v --cov=$(SRC) --cov-report=html --cov-report=term-missing
	@echo "HTML report: htmlcov/index.html"

# ── clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -rf dist build *.egg-info
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -name "*.pyc" -delete

# ── documentation ────────────────────────────────────────────────────────────
.PHONY: docs
docs:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo "HTML pages are in $(BUILDDIR)/html."

.PHONY: docs-clean
docs-clean:
	rm -rf $(BUILDDIR)
