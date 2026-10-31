# Proxy lifecycle events

Each request first writes `request_arrival`, keyed by trusted `rid` and optional
untrusted trace join key `client_id`. A normal handler return writes exactly one
`request_terminal`. Process death may prevent the terminal record; the arrival and
independent load-generator result must remain in the denominator.

The following events describe local proxy observations. They do not acknowledge a
remote network abort, scheduler abort, GPU completion, or KV release:

| Event | Meaning |
| --- | --- |
| `request_context_cancel_observed` | The request context was observed cancelled; `reason` is its first cause. |
| `request_cancel_trigger_observed` | The first explicit local timeout/write-failure trigger was observed. |
| `upstream_body_close_completed` | An acquired upstream response body's `Close()` returned; optional `error` records failure. This also occurs on success. |

All three include `rid`, optional `client_id` and `tenant`, and
`observed_unix_ms`. Millisecond timestamps can tie and do not establish cross-host
ordering. Terminal copies are `context_cancel_observed_unix_ms` plus
`context_cancel_reason`, `cancel_trigger_observed_unix_ms` plus
`cancel_trigger_reason`, and `upstream_body_close_completed_unix_ms`.
Event and terminal copies represent the same observation and must not be counted
twice. Each observation type occurs at most once per request.

Context reason `request_context_cancelled` is deliberately ambiguous: Go's HTTP
server cancels request contexts for peer disconnects and local socket read/write
failures. Proven causes retain `ttft_deadline_exceeded`, `proxy_shutdown`,
`downstream_write_timeout`, or `downstream_write_error`; errors initiated by stream
processing use `upstream_failure`. Parent context deadlines and unrecognized
parent causes retain `request_context_deadline_exceeded` and
`request_context_cancelled_other`. Separate trigger reasons are
`ttft_timer_elapsed`, `ttft_cutoff_observed`,
`request_read_deadline_exceeded`, `downstream_write_timeout`, and
`downstream_write_error`. A local socket timeout can cancel the context before its
explicit trigger is observed, so the generic context cause and specific trigger
may coexist. Body-close records are absent if no upstream response was acquired.
Successful cleanup does not generate cancellation records, and cleanup latency
does not extend the TTFT timer.

Parsed snapshot identifiers are copied before calibration/freshness checks into
`snapshot_step_seq`, `snapshot_timestamp_unix_ms`, `snapshot_policy_version`, and
`snapshot_engine_instance_id`. This identifies even stale or uncalibrated
snapshots. Missing/unparseable snapshots cannot supply these identifiers; an engine
restart may reuse a step number, so a step alone is not a unique snapshot key.
