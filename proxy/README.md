# Go gateway

Requires Go 1.26; the module has no third-party dependencies.

```sh
mkdir -p logs
go run -buildvcs=false ./cmd/fairsched-proxy -config proxy/config.example.json
```

The listener defaults to loopback. Replace the example API keys before exposing
the service. `upstream` is a base URL without `/v1`; only POST
`/v1/chat/completions` and `/v1/completions` are forwarded. `/healthz` is a local
liveness endpoint. No client headers are forwarded. Set `upstream_api_key` only
when the engine requires a separate credential.

The JSON config binds API keys to tenant IDs, and tenant IDs to class, TTFT
deadline in milliseconds, `max_inflight`, `max_queued`, and optional baseline
priority. Limits apply per tenant, with a bounded FIFO queue. The deadline starts
at arrival, so time spent uploading, parsing and queueing counts against it.
The HTTP read deadline is cleared after a successful body read within the
deadline. Set deadline to 0 for no TTFT deadline. Set `priority_baseline` to true to inject trusted
priorities (A=-1, B=0, C=1 in the example). Otherwise the gateway strips priority.

Client `fairsched*` keys, `vllm_xargs`, and `priority` fields (any case) are
removed from the top level of the body, where vLLM reads them, and from a
top-level `extra_body` object. Nested content, such as message fields or a
tool's JSON schema with a `priority` property, is forwarded unchanged. The
gateway injects trusted
top-level `vllm_xargs` containing `fairsched_tenant`, `fairsched_class`, `fairsched_rid`, and optional
`fairsched_deadline_ms`. The latter is an **absolute Unix timestamp in milliseconds**.
The gateway returns the generated join ID in `X-Fairsched-Request-ID`.
`X-Fairsched-Client-ID` is an optional, untrusted correlation label (at most 128
printable ASCII bytes). It appears only in proxy logs, never in trusted engine
metadata or routing policy. The load generator sets it to the trace row ID.

The upstream always streams. First content, reasoning, or tool-call output stops
the TTFT timer; a nonempty generated `token_ids` array also counts, even when the
text is empty. Prompt token IDs, role-only and usage-only events do not count. Streaming clients receive
SSE, while non-streaming clients receive aggregated OpenAI JSON after `[DONE]`.
This avoids incorrectly applying TTFT to the entire non-streaming generation.
Every requested choice must have a finish reason before `[DONE]`; premature
markers, EOF, upstream errors, and malformed SSE are errors. A downstream
disconnect, TTFT expiration, or process termination cancels the upstream request.
This proves HTTP cancellation propagation, not the GPU's eventual stop time.

Each request first writes an immediate `request_arrival` JSONL entry containing
the generated RID, optional client ID, trusted tenant, and arrival time. The CLI
writes directly to its log file, so a process kill preserves the arrival even
if no response headers or terminal record were sent. Join unfinished arrivals
to client failures by client ID; do not invent a completion for a killed process.
This survives a process restart, not a power loss.

Each request also produces exactly one `request_terminal` entry when its handler
returns (health checks excluded), with
arrival, upstream start, first output, completion time, reason, HTTP status, and
state (`completed`, `rejected`, `timeout`, `cancelled`, or `error`).
`output_chunks` counts SSE events containing output, **not model tokens**.
After response headers are sent, a stream failure is an SSE `error` event; the
terminal log records its logical failure status. The counter `LogErrorCount`
surfaces write failures for an embedding application's monitoring.
SIGTERM/SIGINT cancels active requests with reason `proxy_shutdown` and waits for
terminal logging. SIGKILL can leave an arrival without a terminal, as expected.
Separate cancellation/trigger/response-body-close events describe local
observations, not remote abort acknowledgements; see [lifecycle-schema.md](lifecycle-schema.md).
The terminal reason `request_context_cancelled` does not uniquely identify a
client disconnect because local socket failures also cancel that context.

`write_timeout_ms` bounds each downstream write and flush (default 30000).
Successful SSE flushes clear that write deadline while awaiting the next model
event; it is separate from TTFT and generation duration. A blocked slow reader
times out, cancels the upstream, and releases its quota slot. The SSE parser
limits the complete event, including multiline data, to 8 MiB.

## Optional predictive admission

The gateway can reject requests that would miss their TTFT deadline, given an
engine-state snapshot that some producer writes to `snapshot_path` (write a
temporary file, then rename it). No producer ships with this repository and
the experiments in the main README run with the path empty, which disables
predictive admission and keeps quotas.

```json
{
  "calibrated": true,
  "step_seq": 12,
  "timestamp_unix_ms": 1791910000000,
  "prefill_tokens_per_second": 5000,
  "step_ms": 12.5,
  "free_kv_tokens": 8192,
  "tenants": {
    "A": {
      "prefill_tokens_ahead": 100,
      "output_tokens_quantile": {"interactive": 128, "analysis": 1024, "batch": 256}
    }
  }
}
```

Uncalibrated, unavailable, invalid, or older-than-one-second snapshots fail open
and increment `SnapshotFailOpenCount`, with the cause recorded in the request
log. Do not set `calibrated=true` until rate, step duration, and prompt estimation
error have been measured for the target environment.

Estimated TTFT is `(prefill_tokens_ahead + estimated_prompt_tokens) / rate + step`.
The local prompt estimate uses three Unicode characters per token plus chat
framing; integer token-ID prompts use their exact length. It is a documented
estimate, not model tokenization. KV feasibility uses estimated prompt length
plus `min(declared_max_tokens, measured_output_length_quantile)`. Missing
`max_tokens` uses the measured quantile. Declared output limits never change
tenant ordering or TTFT estimates. Concurrent arrivals can share a snapshot, so
this is predictive admission, not a hard engine KV reservation.

```sh
GOCACHE=/tmp/fairsched-go-cache go test -race ./...
```

Tests bind localhost ports using `httptest`. They cover metadata stripping,
streaming/non-streaming TTFT, role-only events, HTTP cancellation, bounded queues,
snapshot rejection/fail-open, malformed and interrupted SSE, tool aggregation,
early `[DONE]`, arrival persistence, slow-read timeouts, token IDs, shutdown
attribution, and end-to-end join IDs. No GPU timing claims follow from these tests.
