#!/usr/bin/env bash
# Build static gateway and load-generator binaries into bin/ using the Go container.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
  -e CGO_ENABLED=0 -e GOFLAGS=-buildvcs=false -e GOCACHE=/tmp/gocache \
  golang:1.26.8-bookworm sh -c 'go build -o bin/fairsched-proxy ./cmd/fairsched-proxy && go build -o bin/fairsched-loadgen ./cmd/fairsched-loadgen'
ls -l bin
