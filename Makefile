# Entry points for the steps in the README's Reproduce section.
.PHONY: test test-local build models test-model bench

# Everything in scripts/test.sh (needs Docker and make test-model).
test:
	scripts/test.sh

# Only what needs no Docker: the pure-Python tests and the Go tests with a local Go toolchain.
test-local:
	python3 -m unittest tests.test_policy tests.test_report tests.test_audit
	go vet ./proxy/... ./loadgen/... ./cmd/...
	go test ./proxy/... ./loadgen/... ./cmd/...

# Static gateway and load-generator binaries in bin/, built in the Go container.
build:
	scripts/build-go.sh

# Pinned model revisions into .cache/huggingface (about 15 GB).
models:
	scripts/fetch-models.sh

# Only the test model's config.json, enough for make test.
test-model:
	scripts/fetch-models.sh --test-only

# matplotlib for bench.report --plots, as used by scripts/run-all.sh.
.venv/bin/python:
	python3 -m venv .venv
	.venv/bin/pip install matplotlib==3.11.2

# Every experiment in the README (about 2.5 h on one RTX 4090; run make models first).
bench: build .venv/bin/python
	scripts/run-all.sh
