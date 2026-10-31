package loadgen

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/simon-jd/fairsched/proxy"
)

func output(w http.ResponseWriter, data string) {
	_ = proxy.WriteSSE(w, proxy.SSEEvent{Data: data})
	w.(http.Flusher).Flush()
}

const chunk = `{"choices":[{"index":0,"delta":{"content":"hello"}}]}`

func arrival(id string, at int64) Arrival {
	return Arrival{ID: id, Tenant: "A", AtMS: at, Path: "/v1/chat/completions", Body: json.RawMessage(`{"messages":[],"stream":true}`), DeadlineMS: 500}
}

func TestOpenArrivalDoesNotWaitForCompletion(t *testing.T) {
	var mu sync.Mutex
	var entered []time.Time
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		entered = append(entered, time.Now())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(120 * time.Millisecond)
		output(w, chunk)
		output(w, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		output(w, `{"choices":[],"usage":{"completion_tokens":5}}`)
		output(w, "[DONE]")
	}))
	defer s.Close()
	var logs bytes.Buffer
	rs, err := Run(context.Background(), []Arrival{arrival("a", 0), arrival("b", 10), arrival("c", 20)}, Options{BaseURL: s.URL, Results: &logs})
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	span := entered[2].Sub(entered[0])
	mu.Unlock()
	if span > 80*time.Millisecond {
		t.Fatalf("arrivals coupled to completion: %v", span)
	}
	for _, r := range rs {
		if r.State != "completed" || r.FirstOutputUnixNS == 0 || len(r.Events) != 4 || r.OutputChunks != 1 || r.CompletionTokens == nil || *r.CompletionTokens != 5 {
			t.Fatal(r)
		}
	}
	if bytes.Count(logs.Bytes(), []byte("\n")) != 3 {
		t.Fatal(logs.String())
	}
	report := Summarize(rs)
	if report.Arrivals != 3 || report.Tenants["A"].CompletionTokens != 15 || report.Tenants["A"].TTFT.P99MS != nil || report.Tenants["A"].SLOMet != 3 {
		t.Fatal(report)
	}
}

func TestProgrammableCancelAndTimeoutReachServer(t *testing.T) {
	for _, kind := range []string{"chunks", "timer", "timeout"} {
		t.Run(kind, func(t *testing.T) {
			cancelled := make(chan struct{})
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if kind == "chunks" {
					output(w, chunk)
				} else {
					w.(http.Flusher).Flush()
				}
				<-r.Context().Done()
				close(cancelled)
			}))
			defer s.Close()
			a := arrival(kind, 0)
			want := "cancelled"
			if kind == "chunks" {
				a.CancelAfterOutputChunks = 1
			} else if kind == "timer" {
				ms := int64(20)
				a.CancelAfterMS = &ms
			} else {
				a.TimeoutMS = 20
				want = "timeout"
			}
			rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL})
			if err != nil {
				t.Fatal(err)
			}
			if rs[0].State != want {
				t.Fatal(rs)
			}
			if kind != "timeout" && rs[0].CancelledUnixNS == 0 {
				t.Fatal("missing cancellation timestamp")
			}
			select {
			case <-cancelled:
			case <-time.After(time.Second):
				t.Fatal("server not cancelled")
			}
		})
	}
}

func TestAllArrivalsInDenominator(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, `{"error":"overloaded"}`)
	}))
	defer s.Close()
	rs, err := Run(context.Background(), []Arrival{arrival("reject1", 0), arrival("reject2", 0)}, Options{BaseURL: s.URL})
	if err != nil {
		t.Fatal(err)
	}
	report := Summarize(rs)
	if report.Arrivals != 2 || report.Tenants["A"].Arrivals != 2 || report.Tenants["A"].RejectionRate != 1 || report.Tenants["A"].SuccessRate != 0 {
		t.Fatal(report)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var logs bytes.Buffer
	rs, err = Run(ctx, []Arrival{arrival("a", 0), arrival("b", 2000)}, Options{BaseURL: s.URL, Results: &logs})
	if err != nil || len(rs) != 2 || rs[1].State != "cancelled" || rs[1].SentUnixNS != 0 || bytes.Count(logs.Bytes(), []byte("\n")) != 2 {
		t.Fatal(rs, err)
	}
}

func TestTraceValidationAndOrdering(t *testing.T) {
	valid := `{"id":"b","at_ms":20,"tenant":"A","body":{"prompt":"B"}}
{"id":"a","at_ms":0,"tenant":"A","body":{"prompt":"A"}}`
	rows, err := ReadTrace(strings.NewReader(valid))
	if err != nil || len(rows) != 2 || rows[0].ID != "a" {
		t.Fatal(rows, err)
	}
	for _, bad := range []string{`{"id":"a","at_ms":-1,"body":{}}`, `{"id":"a","body":null}`, `{"id":"a","body":{}}
{"id":"a","body":{}}`, `{"id":"a","body":{},"cancel_after_ms":-2}`} {
		if _, err := ReadTrace(strings.NewReader(bad)); err == nil {
			t.Fatalf("accepted %s", bad)
		}
	}
}

func TestNonstreamDoesNotInventTTFT(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"}}],"usage":{"completion_tokens":2}}`)
	}))
	defer s.Close()
	a := arrival("a", 0)
	a.Body = json.RawMessage(`{"stream":false,"messages":[]}`)
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL})
	if err != nil || rs[0].State != "completed" || rs[0].FirstOutputUnixNS != 0 {
		t.Fatal(rs, err)
	}
	report := Summarize(rs)
	if report.Tenants["A"].TTFT.Samples != 0 || report.Tenants["A"].SLOMet != 0 {
		t.Fatal(report)
	}
}

func TestIncompleteSSEAndHTTPFailures(t *testing.T) {
	for _, kind := range []string{"eof", "malformed", "error", "http"} {
		t.Run(kind, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if kind == "http" {
					w.WriteHeader(500)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				switch kind {
				case "eof":
					output(w, chunk)
				case "malformed":
					output(w, "oops")
				case "error":
					output(w, `{"error":{"message":"engine restart"}}`)
				}
			}))
			defer s.Close()
			rs, err := Run(context.Background(), []Arrival{arrival("a", 0)}, Options{BaseURL: s.URL})
			if err != nil || len(rs) != 1 || rs[0].State != "error" {
				t.Fatal(rs, err)
			}
		})
	}
}

func TestEndToEndProxyLoadgenJoin(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["vllm_xargs"].(map[string]any)["fairsched_tenant"] != "A" {
			t.Error("metadata missing")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		output(w, chunk)
		output(w, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}`)
		output(w, "[DONE]")
	}))
	defer engine.Close()
	var proxyLogs bytes.Buffer
	handler, err := proxy.New(proxy.Config{Upstream: engine.URL, APIKeys: map[string]string{"key": "A"}, Tenants: map[string]proxy.TenantPolicy{"A": {Class: "interactive", MaxInflight: 2, MaxQueued: 2, DefaultDeadlineMS: 500}}}, nil, &proxyLogs)
	if err != nil {
		t.Fatal(err)
	}
	gateway := httptest.NewServer(handler)
	a := arrival("a", 0)
	a.APIKey = "key"
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: gateway.URL})
	gateway.Close()
	if err != nil || rs[0].State != "completed" {
		t.Fatal(rs, err)
	}
	var terminal proxy.Terminal
	lines := bytes.Split(bytes.TrimSpace(proxyLogs.Bytes()), []byte("\n"))
	terminalCount := 0
	for _, line := range lines {
		var record proxy.Terminal
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		if record.Event == "request_terminal" {
			terminal = record
			terminalCount++
		}
	}
	if terminalCount != 1 || terminal.RID != rs[0].RID || terminal.State != "completed" || terminal.ClientID != "a" {
		t.Fatal(terminal, rs)
	}
}
