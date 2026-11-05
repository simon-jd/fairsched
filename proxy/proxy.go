// Package proxy implements the authenticated, bounded fairsched OpenAI-compatible gateway.
package proxy

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

var errDeadline = errors.New("TTFT deadline exceeded")
var errDone = errors.New("SSE done")
var ErrServerShutdown = errors.New("proxy server shutdown")

type ArrivalRecord struct {
	Event         string `json:"event"`
	RID           string `json:"rid"`
	ClientID      string `json:"client_id,omitempty"`
	Tenant        string `json:"tenant,omitempty"`
	Class         string `json:"class,omitempty"`
	ArrivedUnixMS int64  `json:"arrived_unix_ms"`
}

type Terminal struct {
	Event                            string `json:"event"`
	RID                              string `json:"rid"`
	ClientID                         string `json:"client_id,omitempty"`
	Tenant                           string `json:"tenant,omitempty"`
	Class                            string `json:"class,omitempty"`
	ArrivedUnixMS                    int64  `json:"arrived_unix_ms"`
	Upstream                         string `json:"upstream,omitempty"`
	UpstreamStartedUnixMS            int64  `json:"upstream_started_unix_ms,omitempty"`
	FirstTokenUnixMS                 int64  `json:"first_token_unix_ms,omitempty"`
	FinishedUnixMS                   int64  `json:"finished_unix_ms"`
	State                            string `json:"state"`
	Status                           int    `json:"status"`
	Reason                           string `json:"reason,omitempty"`
	OutputChunks                     int    `json:"output_chunks"`
	SnapshotFailOpen                 string `json:"snapshot_fail_open,omitempty"`
	SnapshotStepSeq                  int64  `json:"snapshot_step_seq,omitempty"`
	SnapshotTimestampUnixMS          int64  `json:"snapshot_timestamp_unix_ms,omitempty"`
	SnapshotPolicyVersion            string `json:"snapshot_policy_version,omitempty"`
	SnapshotEngineInstanceID         string `json:"snapshot_engine_instance_id,omitempty"`
	ContextCancelObservedUnixMS      int64  `json:"context_cancel_observed_unix_ms,omitempty"`
	ContextCancelReason              string `json:"context_cancel_reason,omitempty"`
	CancelTriggerObservedUnixMS      int64  `json:"cancel_trigger_observed_unix_ms,omitempty"`
	CancelTriggerReason              string `json:"cancel_trigger_reason,omitempty"`
	UpstreamBodyCloseCompletedUnixMS int64  `json:"upstream_body_close_completed_unix_ms,omitempty"`
	PromptTokensEstimate             int    `json:"prompt_tokens_estimate"`
}
type Handler struct {
	cfg       Config
	client    *http.Client
	gates     map[string]*gate
	log       io.Writer
	logMu     sync.Mutex
	failOpen  atomic.Uint64
	logErrors atomic.Uint64
	active    sync.WaitGroup
	upstreams []string
	rr        map[string]*atomic.Uint64 // per-tenant round-robin position
}

func New(c Config, client *http.Client, log io.Writer) (*Handler, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if client == nil {
		client = &http.Client{}
	}
	copyClient := *client
	copyClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if log == nil {
		log = io.Discard
	}
	h := &Handler{cfg: c, client: &copyClient, log: log, gates: map[string]*gate{},
		upstreams: c.UpstreamURLs(), rr: map[string]*atomic.Uint64{}}
	for t, p := range c.Tenants {
		h.gates[t] = &gate{max: p.MaxInflight, capacity: p.MaxQueued}
		h.rr[t] = &atomic.Uint64{}
	}
	return h, nil
}

// SessionHash is FNV-1a 64 of the session ID; sticky routing sends a session
// to upstream SessionHash(id) mod len(upstreams). bench/workloads.py computes
// the same hash to place sessions.
func SessionHash(session string) uint64 {
	f := fnv.New64a()
	_, _ = f.Write([]byte(session))
	return f.Sum64()
}

// route picks the upstream for one admitted request of a known tenant.
func (h *Handler) route(tenant, session string) string {
	n := uint64(len(h.upstreams))
	if n == 1 {
		return h.upstreams[0]
	}
	if h.cfg.Routing == RoutingSticky && session != "" {
		return h.upstreams[SessionHash(session)%n]
	}
	return h.upstreams[(h.rr[tenant].Add(1)-1)%n]
}
func (h *Handler) SnapshotFailOpenCount() uint64 { return h.failOpen.Load() }
func (h *Handler) LogErrorCount() uint64         { return h.logErrors.Load() }

// Wait is used after the HTTP server stops accepting new requests.
func (h *Handler) Wait() { h.active.Wait() }
func (h *Handler) writeLog(r any) {
	h.logMu.Lock()
	defer h.logMu.Unlock()
	if json.NewEncoder(h.log).Encode(r) != nil {
		h.logErrors.Add(1)
	}
}
func clientID(header string) string {
	if len(header) > 128 {
		return ""
	}
	for _, b := range []byte(header) {
		if b < 32 || b > 126 {
			return ""
		}
	}
	return header
}
func requestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}
func writeError(w http.ResponseWriter, status int, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": reason, "type": "fairsched_error", "code": reason}})
}

// routingField reports whether a client-supplied key could set the tenant, the
// request identity or the priority: any fairsched-prefixed key, vllm_xargs
// (vLLM passes it to the scheduler as SamplingParams.extra_args, where the
// plugin reads the tenant) and priority. Matching ignores case.
func routingField(k string) bool {
	lower := strings.ToLower(k)
	return strings.HasPrefix(lower, "fairsched") || lower == "vllm_xargs" || lower == "priority"
}

// sanitize removes routing fields from the top level of the request body, where
// vLLM reads them, and from a top-level extra_body object, in case a client
// sends the OpenAI SDK's wrapper literally. Nested content such as messages or
// tool JSON schemas is forwarded unchanged, even when it uses these key names.
// The gateway then adds its own trusted top-level vllm_xargs (and priority).
func sanitize(request map[string]any) {
	for k := range request {
		if routingField(k) {
			delete(request, k)
		}
	}
	if extra, ok := request["extra_body"].(map[string]any); ok {
		for k := range extra {
			if routingField(k) {
				delete(extra, k)
			}
		}
	}
}
func estimatePrompt(v map[string]any) int {
	if p, ok := v["prompt"].([]any); ok {
		all := true
		for _, x := range p {
			if _, ok := x.(float64); !ok {
				all = false
			}
		}
		if all {
			return len(p)
		}
	}
	// This is an explicit estimate, not a tokenizer count. Calibrate its error on
	// the target model's trace before enabling deadline admission.
	total := 0
	var visit func(any)
	visit = func(x any) {
		switch x := x.(type) {
		case string:
			total += (utf8.RuneCountInString(x) + 2) / 3
		case []any:
			for _, z := range x {
				visit(z)
			}
		case map[string]any:
			for k, z := range x {
				if k == "content" || k == "text" || k == "prompt" {
					visit(z)
				}
			}
		}
	}
	if p, ok := v["prompt"]; ok {
		visit(p)
	}
	if m, ok := v["messages"].([]any); ok {
		visit(m)
		total += len(m)*8 + 4
	}
	if total < 1 {
		return 1
	}
	return total
}
func (h *Handler) admit(v map[string]any, t string, p TenantPolicy, deadline int64, rec *Terminal) string {
	if h.cfg.SnapshotPath == "" {
		return ""
	}
	fail := func(reason string) string { rec.SnapshotFailOpen = reason; h.failOpen.Add(1); return "" }
	b, err := os.ReadFile(h.cfg.SnapshotPath)
	if err != nil {
		return fail("snapshot_unavailable")
	}
	var s AdmissionSnapshot
	if json.Unmarshal(b, &s) != nil {
		return fail("snapshot_invalid")
	}
	// Preserve the exact parsed snapshot identity even when freshness or calibration
	// makes admission fail open. A restarted engine can reuse step numbers.
	rec.SnapshotStepSeq = s.StepSeq
	rec.SnapshotTimestampUnixMS = s.TimestampUnixMS
	rec.SnapshotPolicyVersion = s.PolicyVersion
	rec.SnapshotEngineInstanceID = s.EngineInstanceID
	now := time.Now().UnixMilli()
	if now-s.TimestampUnixMS > 1000 || s.TimestampUnixMS > now+1000 {
		return fail("snapshot_stale")
	}
	if !s.Calibrated {
		return fail("snapshot_uncalibrated")
	}
	st, ok := s.Tenants[t]
	if !ok || s.PrefillTokensPerSecond <= 0 || s.StepMS < 0 || st.PrefillTokensAhead < 0 || s.FreeKVTokens < 0 {
		return fail("snapshot_invalid")
	}
	q, ok := st.OutputTokensQuantile[p.Class]
	if !ok || q < 0 {
		return fail("snapshot_missing_length_quantile")
	}
	estMS := (st.PrefillTokensAhead+float64(rec.PromptTokensEstimate))/s.PrefillTokensPerSecond*1000 + s.StepMS
	if deadline > 0 && estMS > float64(deadline-now) {
		return "predicted_ttft_exceeds_deadline"
	}
	output := q
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		if n, ok := v[key].(float64); ok && n > 0 {
			output = math.Min(output, n)
		}
	}
	if float64(rec.PromptTokensEstimate)+output > float64(s.FreeKVTokens) {
		return "insufficient_kv_capacity"
	}
	return ""
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.active.Add(1)
	defer h.active.Done()
	if r.URL.Path == "/healthz" {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok\n")
		return
	}
	rec := Terminal{Event: "request_terminal", RID: requestID(), ClientID: clientID(r.Header.Get("X-Fairsched-Client-ID")), ArrivedUnixMS: time.Now().UnixMilli(), State: "error", Status: 500}
	auth := r.Header.Get("Authorization")
	key, ok := strings.CutPrefix(auth, "Bearer ")
	tenant, known := h.cfg.APIKeys[key]
	p := h.cfg.Tenants[tenant]
	if ok && known {
		rec.Tenant = tenant
		rec.Class = p.Class
	}
	h.writeLog(ArrivalRecord{Event: "request_arrival", RID: rec.RID, ClientID: rec.ClientID, Tenant: rec.Tenant, Class: rec.Class, ArrivedUnixMS: rec.ArrivedUnixMS})
	w.Header().Set("X-Fairsched-Request-ID", rec.RID)
	ctx, cancel := context.WithCancelCause(r.Context())
	lifecycle := newLocalLifecycle(h, rec)
	watchDone := make(chan struct{})
	stopWatch := context.AfterFunc(ctx, func() {
		defer close(watchDone)
		lifecycle.observeCancellation(context.Cause(ctx))
	})
	defer func() {
		// Join any callback before copying its fields. Stop the watcher before
		// cleanup cancellation so normal completion never looks like an abort.
		if !stopWatch() {
			<-watchDone
		}
		if ctx.Err() != nil {
			lifecycle.observeCancellation(context.Cause(ctx))
		}
		cancel(nil)
		lifecycle.copyTo(&rec)
		rec.FinishedUnixMS = time.Now().UnixMilli()
		h.writeLog(rec)
	}()
	writeTimeout := time.Duration(h.cfg.WriteTimeoutMS) * time.Millisecond
	if writeTimeout == 0 {
		writeTimeout = 30 * time.Second
	}
	writeDeadline := func() error { return http.NewResponseController(w).SetWriteDeadline(time.Now().Add(writeTimeout)) }
	fail := func(status int, state, reason string) {
		rec.Status = status
		rec.State = state
		rec.Reason = reason
		if writeDeadline() == nil {
			writeError(w, status, reason)
		}
	}
	if r.Method != "POST" || (r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/v1/completions") {
		fail(404, "rejected", "unsupported_endpoint")
		return
	}
	if !ok || !known {
		fail(401, "rejected", "invalid_api_key")
		return
	}
	deadline := int64(0)
	if p.DefaultDeadlineMS > 0 {
		deadline = rec.ArrivedUnixMS + p.DefaultDeadlineMS
		if err := http.NewResponseController(w).SetReadDeadline(time.UnixMilli(deadline)); err != nil {
			fail(500, "error", "request_read_deadline_unavailable")
			return
		}
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 8<<20))
	if err != nil {
		if errors.Is(err, os.ErrDeadlineExceeded) {
			lifecycle.trigger("request_read_deadline_exceeded")
			cancel(errDeadline)
			fail(504, "timeout", "ttft_deadline_exceeded")
		} else if errors.Is(context.Cause(r.Context()), ErrServerShutdown) {
			fail(503, "error", "proxy_shutdown")
		} else if r.Context().Err() != nil {
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "request_context_cancelled"
		} else {
			var limit *http.MaxBytesError
			if errors.As(err, &limit) {
				fail(413, "rejected", "request_body_too_large")
			} else {
				fail(400, "rejected", "request_body_read_error")
			}
		}
		return
	}
	if deadline > 0 {
		if time.Now().UnixMilli() >= deadline {
			lifecycle.trigger("ttft_cutoff_observed")
			cancel(errDeadline)
			fail(504, "timeout", "ttft_deadline_exceeded")
			return
		}
		_ = http.NewResponseController(w).SetReadDeadline(time.Time{})
	}
	var request map[string]any
	if json.Unmarshal(body, &request) != nil || request == nil {
		fail(400, "rejected", "invalid_json")
		return
	}
	stream, _ := request["stream"].(bool)
	sanitize(request)
	rec.PromptTokensEstimate = estimatePrompt(request)
	if reason := h.admit(request, tenant, p, deadline, &rec); reason != "" {
		fail(429, "rejected", reason)
		return
	}
	var deadlineMu sync.Mutex
	first := false
	expired := false
	deadlineFinished := false
	var timer *time.Timer
	stopDeadline := func() {
		deadlineMu.Lock()
		deadlineFinished = true
		if timer != nil {
			timer.Stop()
		}
		deadlineMu.Unlock()
	}
	if deadline > 0 {
		timer = time.AfterFunc(time.Until(time.UnixMilli(deadline)), func() {
			deadlineMu.Lock()
			defer deadlineMu.Unlock()
			if !first && !deadlineFinished {
				expired = true
				lifecycle.trigger("ttft_timer_elapsed")
				cancel(errDeadline)
			}
		})
		defer stopDeadline()
	}
	markFirst := func() bool {
		deadlineMu.Lock()
		defer deadlineMu.Unlock()
		if !first && deadline > 0 && time.Now().UnixMilli() >= deadline {
			expired = true
			lifecycle.trigger("ttft_cutoff_observed")
			cancel(errDeadline)
		}
		if expired {
			return false
		}
		if !first {
			first = true
			rec.FirstTokenUnixMS = time.Now().UnixMilli()
			if timer != nil {
				timer.Stop()
			}
		}
		return true
	}
	contextFailure := func() {
		if errors.Is(context.Cause(ctx), errDeadline) {
			fail(504, "timeout", "ttft_deadline_exceeded")
		} else if errors.Is(context.Cause(ctx), ErrServerShutdown) {
			fail(503, "error", "proxy_shutdown")
		} else if r.Context().Err() != nil {
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "request_context_cancelled"
		} else {
			fail(502, "error", "upstream_error")
		}
	}
	if err := h.gates[tenant].acquire(ctx); err != nil {
		if errors.Is(err, errOverloaded) {
			fail(429, "rejected", "tenant_queue_full")
		} else {
			contextFailure()
		}
		return
	}
	defer h.gates[tenant].release()
	if deadline > 0 && time.Now().UnixMilli() >= deadline {
		lifecycle.trigger("ttft_cutoff_observed")
		cancel(errDeadline)
	}
	if ctx.Err() != nil {
		contextFailure()
		return
	}
	xargs := map[string]any{"fairsched_tenant": tenant, "fairsched_class": p.Class, "fairsched_rid": rec.RID}
	if deadline > 0 {
		xargs["fairsched_deadline_ms"] = deadline
	}
	request["vllm_xargs"] = xargs
	if h.cfg.PriorityBaseline {
		request["priority"] = p.Priority
	}
	request["stream"] = true
	request["stream_options"] = map[string]any{"include_usage": true}
	body, _ = json.Marshal(request)
	rec.Upstream = h.route(tenant, clientID(r.Header.Get("X-Fairsched-Session-ID")))
	up, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(rec.Upstream, "/")+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		fail(500, "error", "invalid_upstream_request")
		return
	}
	up.Header.Set("Content-Type", "application/json")
	up.Header.Set("Accept", "text/event-stream")
	if h.cfg.UpstreamAPIKey != "" {
		up.Header.Set("Authorization", "Bearer "+h.cfg.UpstreamAPIKey)
	}
	if deadline > 0 && time.Now().UnixMilli() >= deadline {
		lifecycle.trigger("ttft_cutoff_observed")
		cancel(errDeadline)
		contextFailure()
		return
	}
	rec.UpstreamStartedUnixMS = time.Now().UnixMilli()
	resp, err := h.client.Do(up)
	if err != nil {
		contextFailure()
		return
	}
	defer func() {
		// The handler has already classified the response. Cleanup latency
		// (including a delayed Close) must not turn completion into a timeout.
		stopDeadline()
		if ctx.Err() != nil {
			lifecycle.observeCancellation(context.Cause(ctx))
		}
		lifecycle.bodyCloseCompleted(resp.Body.Close())
	}()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if ctx.Err() != nil {
			contextFailure()
			return
		}
		rec.Status = resp.StatusCode
		rec.State = "error"
		rec.Reason = "upstream_http_error"
		w.Header().Set("Content-Type", "application/json")
		_ = writeDeadline()
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(b)
		return
	}
	if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		fail(502, "error", "upstream_did_not_stream")
		return
	}
	started := false
	done := false
	var pending []SSEEvent
	pendingSize := 0
	aggregate := newAggregate(strings.Contains(r.URL.Path, "chat"))
	tracker := CompletionTracker{ExpectedChoices: ExpectedChoiceCount(request)}
	var downstreamWriteErr error
	writeEvent := func(e SSEEvent) (writeErr error) {
		defer func() {
			if writeErr != nil {
				downstreamWriteErr = writeErr
				lifecycle.trigger(downstreamTriggerReason(writeErr))
				cancel(downstreamCancelCause(writeErr))
			} else {
				_ = http.NewResponseController(w).SetWriteDeadline(time.Time{})
			}
		}()
		if err := writeDeadline(); err != nil {
			return err
		}
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(200)
			started = true
		}
		if err := WriteSSE(w, e); err != nil {
			return err
		}
		return http.NewResponseController(w).Flush()
	}
	err = ReadSSE(resp.Body, func(e SSEEvent) error {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		if e.Data == "[DONE]" {
			if err := tracker.Complete(); err != nil {
				return err
			}
			done = true
			if stream {
				for _, pre := range pending {
					if err := writeEvent(pre); err != nil {
						return err
					}
				}
				pending = nil
				if err := writeEvent(e); err != nil {
					return err
				}
			}
			return errDone
		}
		if err := tracker.Observe(e.Data); err != nil {
			return err
		}
		if HasToken(e.Data) {
			if !markFirst() {
				return errDeadline
			}
			rec.OutputChunks++
		}
		if !stream {
			return aggregate.add([]byte(e.Data))
		}
		if rec.FirstTokenUnixMS == 0 {
			pendingSize += len(e.Data)
			if pendingSize > 1<<20 {
				return errors.New("upstream pretoken buffer exceeded")
			}
			pending = append(pending, e)
			return nil
		}
		for _, pre := range pending {
			if err := writeEvent(pre); err != nil {
				return err
			}
		}
		pending = nil
		return writeEvent(e)
	})
	if !done || !errors.Is(err, errDone) {
		rec.Status = 502
		rec.State = "error"
		rec.Reason = "upstream_stream_incomplete"
		if errors.Is(context.Cause(ctx), errDeadline) {
			rec.Status = 504
			rec.State = "timeout"
			rec.Reason = "ttft_deadline_exceeded"
		} else if errors.Is(context.Cause(ctx), ErrServerShutdown) {
			rec.Status = 503
			rec.State = "error"
			rec.Reason = "proxy_shutdown"
		} else if downstreamWriteErr != nil {
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "downstream_write_error"
			if errors.Is(downstreamWriteErr, os.ErrDeadlineExceeded) {
				rec.Reason = "downstream_write_timeout"
			}
		} else if r.Context().Err() != nil {
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "request_context_cancelled"
		}
		cancel(errUpstreamFailure)
		if !started && rec.State != "cancelled" {
			_ = writeDeadline()
			writeError(w, rec.Status, rec.Reason)
		} else if started && rec.State != "cancelled" {
			b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": rec.Reason, "type": "fairsched_error"}})
			_ = writeEvent(SSEEvent{Event: "error", Data: string(b)})
		}
		return
	}
	if !stream {
		payload, err := aggregate.result()
		if err != nil {
			fail(502, "error", err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err = writeDeadline(); err != nil {
			lifecycle.trigger(downstreamTriggerReason(err))
			cancel(downstreamCancelCause(err))
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "downstream_write_error"
			return
		}
		if _, err = w.Write(payload); err != nil {
			lifecycle.trigger(downstreamTriggerReason(err))
			cancel(downstreamCancelCause(err))
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = "client_write_error"
			return
		}
		if err = http.NewResponseController(w).Flush(); err != nil {
			lifecycle.trigger(downstreamTriggerReason(err))
			cancel(downstreamCancelCause(err))
			rec.Status = 499
			rec.State = "cancelled"
			rec.Reason = downstreamTriggerReason(err)
			return
		}
		_ = http.NewResponseController(w).SetWriteDeadline(time.Time{})
	}
	rec.Status = 200
	rec.State = "completed"
}
