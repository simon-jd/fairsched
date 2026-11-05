# Open-arrival load generator

```sh
mkdir -p logs
go run ./cmd/fairsched-loadgen -url http://127.0.0.1:8080 \
  -trace trace.jsonl -output logs/client.jsonl \
  -report logs/client-report.json
```

`bench/workloads.py` generates traces in this format; `bench/run.py` writes
one per run to `results/raw/<workload>/<arm>-s<seed>/trace.jsonl`. Each JSONL
row is one arrival relative to a common run start:

```json
{"id":"A-1","at_ms":0,"tenant":"A","api_key":"local-A","path":"/v1/chat/completions","deadline_ms":2000,"body":{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"stream":true}}
{"id":"C-1","at_ms":25,"tenant":"C","api_key":"local-C","cancel_after_output_chunks":2,"timeout_ms":30000,"body":{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Count to one hundred."}],"max_tokens":256}}
```

Rows are sorted by `at_ms`, with stable ties and unique nonempty printable-ASCII
IDs of at most 128 bytes. Each ID is sent as `X-Fairsched-Client-ID` for correlation
with an early proxy arrival record, even if the proxy dies before returning its
RID response header. This ID confers no tenant or policy authority. Requests
start at their scheduled time regardless of earlier requests' completion. There
is no concurrency gate. Dispatch lag is measured, so overloaded generators are
visible in reports. Every arrival yields a terminal result, including rejected,
failed, timed-out, and arrivals not sent because the whole run was cancelled.

`session` (optional, at most 128 printable ASCII bytes) is sent as
`X-Fairsched-Session-ID`, which the gateway's sticky routing hashes to pick a
replica; it confers no tenant authority either.

`stream` defaults to true, and usage streaming is requested. `return_token_ids`
defaults to true using the verified vLLM v0.31.0 request field. Set
`cancel_after_ms` for elapsed-time cancellation or `cancel_after_output_chunks`
for cancellation after a specified number of output-bearing SSE events. Both
can be set; the first trigger wins. `timeout_ms` is a whole-request timeout,
distinct from the proxy's TTFT deadline. `deadline_ms` in the trace is the SLO
used in reports, and does not override trusted tenant policy. Whole-run SIGINT
or SIGTERM cancellation still writes terminal records for every arrival.

`read_delay_ms` delays each response-body read; `read_chunk_bytes` limits bytes
consumed per read (default 4096 when a delay is configured). These inject slow
reading without imposing a client-side arrival gate. `half_close_after_ms`
closes the TCP write half after the request is fully written and the specified
delay elapses. It is supported only for cleartext HTTP/1.1, uses a dedicated
connection, and continues reading the response after FIN. Half-close is recorded
separately from cancellation; it is not assumed to mean the server aborted.

Results contain scheduled/send/first-output/last-output/finish timestamps in
Unix nanoseconds, proxy join ID, terminal state, exact SSE event data, event
receipt timestamps, output-chunk count, and server-reported completion-token
usage when present. Each generated `choices[*].token_ids` entry is recorded with
its choice index and containing chunk's receipt timestamp; prompt IDs are
excluded. All IDs in one chunk share that timestamp. These are client receipt
times, not GPU generation times. `observed_output_tokens` counts IDs actually
seen and may be incomplete when the server does not provide IDs. An SSE chunk
can contain several model tokens. The load generator never calls a chunk a token.
Non-streaming responses retain their
JSON body and completion usage, but cannot expose client-observed TTFT.

The `-report` summary groups by tenant and uses **all arrivals** as success,
timeout, rejection, and SLO-fraction denominators. SLO goodput is successful streaming requests
meeting their declared TTFT target per second. This report omits p99 with
fewer than 100 samples. (The tables in the top-level README come from
`bench/report.py` instead, which reports a nearest-rank p99 for any sample
count.) Mean TPOT is an explicitly labeled per-request estimate from server
usage and first/last output-event times, not token-by-token latency.
Missing usage is counted; it is never silently replaced with chunk counts.

The test suite includes open-loop arrival independence, SSE event parsing,
programmable cancellation reaching the server, whole-request timeouts, all
arrival denominators, EOF/HTTP/SSE errors, and a real localhost proxy-to-engine
integration. Additional regressions cover early `[DONE]`, per-choice finish
reasons, token-ID timestamps, slow reads, and continued reading after TCP
half-close. GPU tokenization, engine abort, and KV reuse are outside what
these client records can show.
