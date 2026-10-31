package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const outputChunk = `{"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`
const finishChunk = `{"id":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}`

func emit(w http.ResponseWriter, data string) {
	_ = WriteSSE(w, SSEEvent{Data: data})
	w.(http.Flusher).Flush()
}
func config(url string) Config {
	return Config{Upstream: url, APIKeys: map[string]string{"secret": "A"}, Tenants: map[string]TenantPolicy{"A": {Class: "interactive", MaxInflight: 2, MaxQueued: 2}}}
}
func doRequest(t *testing.T, url, body string) *http.Response {
	t.Helper()
	r, err := http.NewRequest("POST", url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set("Authorization", "Bearer secret")
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}
func terminals(t *testing.T, b *bytes.Buffer) []Terminal {
	t.Helper()
	var rows []Terminal
	d := json.NewDecoder(bytes.NewReader(b.Bytes()))
	for {
		var r Terminal
		err := d.Decode(&r)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if r.Event == "request_terminal" {
			rows = append(rows, r)
		}
	}
	return rows
}

func TestMetadataBoundaryAndNonstream(t *testing.T) {
	var got map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		if r.Header.Get("Authorization") != "" || r.Header.Get("X-Fairsched-Tenant") != "" {
			t.Error("client header leaked")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, outputChunk)
		emit(w, finishChunk)
		emit(w, "[DONE]")
	}))
	defer up.Close()
	c := config(up.URL)
	c.PriorityBaseline = true
	p := c.Tenants["A"]
	p.Priority = -1
	p.DefaultDeadlineMS = 1000
	c.Tenants["A"] = p
	var logs bytes.Buffer
	h, err := New(c, nil, &logs)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(h)
	resp := doRequest(t, server.URL, `{"messages":[{"role":"user","content":"hi","FAIRSCHED_tenant":"bad"}],"stream":false,"priority":-99,"fairsched_rid":"evil","vllm_xargs":{"fairsched_tenant":"evil","unknown":123},"extra_body":{"vllm_xargs":{"fairsched_rid":"evil"}}}`)
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	server.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}
	var out map[string]any
	if json.Unmarshal(b, &out) != nil {
		t.Fatal(string(b))
	}
	if out["object"] != "chat.completion" || out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"] != "hello" {
		t.Fatal(out)
	}
	x := got["vllm_xargs"].(map[string]any)
	if x["fairsched_tenant"] != "A" || x["fairsched_class"] != "interactive" || x["fairsched_rid"] == "evil" || len(x) != 4 || got["priority"] != float64(-1) || got["stream"] != true {
		t.Fatal(got)
	}
	if _, ok := got["fairsched_rid"]; ok {
		t.Fatal("untrusted metadata retained")
	}
	if len(got["extra_body"].(map[string]any)) != 0 {
		t.Fatal("nested metadata retained")
	}
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].State != "completed" || rows[0].RID != x["fairsched_rid"] || rows[0].FirstTokenUnixMS == 0 {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, "", true)
}

func TestTTFTOnlyBeforeOutput(t *testing.T) {
	for _, stream := range []bool{true, false} {
		t.Run(map[bool]string{true: "stream", false: "nonstream"}[stream], func(t *testing.T) {
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				emit(w, outputChunk)
				select {
				case <-time.After(160 * time.Millisecond):
				case <-r.Context().Done():
					t.Error("deadline cancelled after first output")
					return
				}
				emit(w, finishChunk)
				emit(w, "[DONE]")
			}))
			defer up.Close()
			c := config(up.URL)
			p := c.Tenants["A"]
			p.DefaultDeadlineMS = 80
			c.Tenants["A"] = p
			var logs bytes.Buffer
			h, _ := New(c, nil, &logs)
			s := httptest.NewServer(h)
			payload := `{"messages":[],"stream":false}`
			if stream {
				payload = `{"messages":[],"stream":true}`
			}
			resp := doRequest(t, s.URL, payload)
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			s.Close()
			if resp.StatusCode != 200 {
				t.Fatalf("%d %s", resp.StatusCode, b)
			}
			rows := terminals(t, &logs)
			if len(rows) != 1 || rows[0].State != "completed" {
				t.Fatal(rows)
			}
			assertLocalLifecycle(t, &logs, "", true)
		})
	}
}

func TestRoleChunkDoesNotStopDeadline(t *testing.T) {
	cancelled := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, `{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`)
		<-r.Context().Done()
		close(cancelled)
	}))
	defer up.Close()
	c := config(up.URL)
	p := c.Tenants["A"]
	p.DefaultDeadlineMS = 40
	c.Tenants["A"] = p
	var logs bytes.Buffer
	h, _ := New(c, nil, &logs)
	s := httptest.NewServer(h)
	resp := doRequest(t, s.URL, `{"messages":[],"stream":true}`)
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("upstream not cancelled")
	}
	s.Close()
	if resp.StatusCode != 504 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].State != "timeout" || rows[0].FirstTokenUnixMS != 0 {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, "ttft_deadline_exceeded", true)
}

func TestDownstreamDisconnectCancelsUpstream(t *testing.T) {
	cancelled := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, outputChunk)
		<-r.Context().Done()
		close(cancelled)
	}))
	defer up.Close()
	var logs bytes.Buffer
	h, _ := New(config(up.URL), nil, &logs)
	s := httptest.NewServer(h)
	resp := doRequest(t, s.URL, `{"messages":[],"stream":true}`)
	one := make([]byte, 1)
	_, _ = resp.Body.Read(one)
	resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("upstream not cancelled")
	}
	s.Close()
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].State != "cancelled" {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, "request_context_cancelled", true)
}

func TestSnapshotAdmissionAndFailOpen(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, outputChunk)
		emit(w, finishChunk)
		emit(w, "[DONE]")
	}))
	defer up.Close()
	for _, tc := range []struct {
		name       string
		age        int64
		calibrated bool
		free       int64
		ahead      float64
		want       int
		reason     string
	}{{"feasible", 0, true, 100, 0, 200, ""}, {"deadline", 0, true, 100, 10000, 429, "predicted_ttft_exceeds_deadline"}, {"kv", 0, true, 1, 0, 429, "insufficient_kv_capacity"}, {"stale", 2000, true, 0, 10000, 200, "snapshot_stale"}, {"uncalibrated", 0, false, 0, 10000, 200, "snapshot_uncalibrated"}} {
		t.Run(tc.name, func(t *testing.T) {
			c := config(up.URL)
			p := c.Tenants["A"]
			p.DefaultDeadlineMS = 1000
			c.Tenants["A"] = p
			c.SnapshotPath = filepath.Join(t.TempDir(), "snapshot.json")
			snap := AdmissionSnapshot{Calibrated: tc.calibrated, StepSeq: 17, PolicyVersion: "v2", EngineInstanceID: "engine-A", TimestampUnixMS: time.Now().UnixMilli() - tc.age, PrefillTokensPerSecond: 1000, StepMS: 1, FreeKVTokens: tc.free, Tenants: map[string]SnapshotTenant{"A": {PrefillTokensAhead: tc.ahead, OutputTokensQuantile: map[string]float64{"interactive": 20}}}}
			b, _ := json.Marshal(snap)
			if err := os.WriteFile(c.SnapshotPath, b, 0600); err != nil {
				t.Fatal(err)
			}
			var logs bytes.Buffer
			h, _ := New(c, nil, &logs)
			s := httptest.NewServer(h)
			resp := doRequest(t, s.URL, `{"messages":[{"content":"hi"}],"max_tokens":8192}`)
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			s.Close()
			if resp.StatusCode != tc.want {
				t.Fatal(resp.StatusCode)
			}
			rows := terminals(t, &logs)
			if len(rows) != 1 {
				t.Fatal(rows)
			}
			if rows[0].SnapshotStepSeq != snap.StepSeq || rows[0].SnapshotTimestampUnixMS != snap.TimestampUnixMS || rows[0].SnapshotPolicyVersion != snap.PolicyVersion || rows[0].SnapshotEngineInstanceID != snap.EngineInstanceID {
				t.Fatal("snapshot identity lost", rows)
			}
			if tc.want == 429 && rows[0].Reason != tc.reason {
				t.Fatal(rows)
			}
			if tc.age > 1000 || !tc.calibrated {
				if h.SnapshotFailOpenCount() != 1 || rows[0].SnapshotFailOpen != tc.reason {
					t.Fatal(rows)
				}
			}
		})
	}
}

func TestGateBoundedQueueAndCancellation(t *testing.T) {
	g := &gate{max: 1, capacity: 1}
	if g.acquire(context.Background()) != nil {
		t.Fatal("first")
	}
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { result <- g.acquire(ctx) }()
	deadline := time.Now().Add(time.Second)
	for {
		g.mu.Lock()
		n := len(g.queue)
		g.mu.Unlock()
		if n == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("waiter not queued")
		}
		time.Sleep(time.Millisecond)
	}
	if err := g.acquire(context.Background()); err != errOverloaded {
		t.Fatal(err)
	}
	cancel()
	if <-result != context.Canceled {
		t.Fatal("expected cancel")
	}
	g.release()
	if g.acquire(context.Background()) != nil {
		t.Fatal("slot leaked")
	}
	g.release()
	// Stress simultaneous release/cancel: no leaked or negative active slots.
	for i := 0; i < 100; i++ {
		_ = g.acquire(context.Background())
		ctx, cancel := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			if g.acquire(ctx) == nil {
				g.release()
			}
		}()
		cancel()
		g.release()
		wg.Wait()
	}
	if g.active != 0 || len(g.queue) != 0 {
		t.Fatal(g)
	}
}

func TestIncompleteStreamIsError(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, outputChunk)
	}))
	defer up.Close()
	var logs bytes.Buffer
	h, _ := New(config(up.URL), nil, &logs)
	s := httptest.NewServer(h)
	resp := doRequest(t, s.URL, `{"stream":true}`)
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	s.Close()
	if !strings.Contains(string(b), "upstream_stream_incomplete") {
		t.Fatal(string(b))
	}
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].State != "error" {
		t.Fatal(rows)
	}
}

func TestSSEParser(t *testing.T) {
	var rows []SSEEvent
	err := ReadSSE(strings.NewReader(": heartbeat\r\nevent: token\r\nid: 4\r\ndata: line1\r\ndata: line2\r\n\r\ndata: dropped"), func(e SSEEvent) error { rows = append(rows, e); return nil })
	if err != nil || len(rows) != 1 || rows[0].Data != "line1\nline2" || rows[0].ID != "4" {
		t.Fatal(rows, err)
	}
	large := strings.Repeat("x", 100000)
	rows = nil
	err = ReadSSE(strings.NewReader("data: "+large+"\n\n"), func(e SSEEvent) error { rows = append(rows, e); return nil })
	if err != nil || len(rows) != 1 || rows[0].Data != large {
		t.Fatal("large SSE event failed")
	}
}

func TestAggregateToolCalls(t *testing.T) {
	a := newAggregate(true)
	for _, data := range []string{`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call1","type":"function","function":{"name":"tool","arguments":"{\"x\":"}}]}}]}`, `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`} {
		if err := a.add([]byte(data)); err != nil {
			t.Fatal(err)
		}
	}
	b, err := a.result()
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	_ = json.Unmarshal(b, &v)
	call := v["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if call["function"].(map[string]any)["arguments"] != `{"x":1}` {
		t.Fatal(string(b))
	}
}

func TestHTTPLimitRejectionAndQueuedDeadline(t *testing.T) {
	entered := make(chan struct{})
	var enterOnce sync.Once
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		enterOnce.Do(func() { close(entered) })
		<-r.Context().Done()
	}))
	defer up.Close()
	c := config(up.URL)
	c.Tenants["A"] = TenantPolicy{Class: "interactive", MaxInflight: 1, MaxQueued: 1, DefaultDeadlineMS: 200}
	var logs bytes.Buffer
	h, _ := New(c, nil, &logs)
	s := httptest.NewServer(h)
	responses := make(chan int, 2)
	request := func() {
		resp := doRequest(t, s.URL, `{"stream":true}`)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		responses <- resp.StatusCode
	}
	go request()
	<-entered
	go request()
	deadline := time.Now().Add(time.Second)
	for {
		h.gates["A"].mu.Lock()
		queued := len(h.gates["A"].queue)
		h.gates["A"].mu.Unlock()
		if queued == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("not queued")
		}
		time.Sleep(time.Millisecond)
	}
	resp := doRequest(t, s.URL, `{"stream":true}`)
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatal(resp.StatusCode)
	}
	for i := 0; i < 2; i++ {
		if status := <-responses; status != 504 {
			t.Fatal(status)
		}
	}
	s.Close()
	rows := terminals(t, &logs)
	if len(rows) != 3 {
		t.Fatal(rows)
	}
	ids := map[string]bool{}
	for _, row := range rows {
		if ids[row.RID] {
			t.Fatal("duplicate terminal")
		}
		ids[row.RID] = true
	}
}

func TestDeclaredOutputSizeCannotChangeTTFTEstimate(t *testing.T) {
	c := config("http://127.0.0.1:1")
	c.SnapshotPath = filepath.Join(t.TempDir(), "snapshot.json")
	snap := AdmissionSnapshot{Calibrated: true, TimestampUnixMS: time.Now().UnixMilli(), PrefillTokensPerSecond: 1000, StepMS: 1, FreeKVTokens: 1000, Tenants: map[string]SnapshotTenant{"A": {PrefillTokensAhead: 1000, OutputTokensQuantile: map[string]float64{"interactive": 64}}}}
	b, _ := json.Marshal(snap)
	_ = os.WriteFile(c.SnapshotPath, b, 0600)
	h, _ := New(c, nil, nil)
	for _, limit := range []float64{1, 32, 8192} {
		r := Terminal{PromptTokensEstimate: 10}
		if reason := h.admit(map[string]any{"max_tokens": limit}, "A", c.Tenants["A"], time.Now().UnixMilli()+100, &r); reason != "predicted_ttft_exceeds_deadline" {
			t.Fatal(limit, reason)
		}
	}
}

func TestTextAggregationPreservesLogprobs(t *testing.T) {
	a := newAggregate(false)
	for _, raw := range []string{`{"choices":[{"index":0,"text":"hel","logprobs":{"tokens":["hel"],"token_logprobs":[-0.1]}}]}`, `{"choices":[{"index":0,"text":"lo","logprobs":{"tokens":["lo"],"token_logprobs":[-0.2]},"finish_reason":"stop"}]}`} {
		if err := a.add([]byte(raw)); err != nil {
			t.Fatal(err)
		}
	}
	b, err := a.result()
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	_ = json.Unmarshal(b, &v)
	c := v["choices"].([]any)[0].(map[string]any)
	if c["text"] != "hello" || len(c["logprobs"].(map[string]any)["tokens"].([]any)) != 2 {
		t.Fatal(string(b))
	}
}
