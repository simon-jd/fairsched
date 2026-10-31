package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDONERequiresEveryChoiceFinished(t *testing.T) {
	for _, prefix := range []string{outputChunk, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{"content":"unfinished"}}]}`} {
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			emit(w, prefix)
			emit(w, "[DONE]")
		}))
		var logs bytes.Buffer
		h, _ := New(config(up.URL), nil, &logs)
		s := httptest.NewServer(h)
		resp := doRequest(t, s.URL, `{"stream":true}`)
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		s.Close()
		up.Close()
		rows := terminals(t, &logs)
		if len(rows) != 1 || rows[0].State != "error" || strings.Contains(string(b), "data: [DONE]") {
			t.Fatal(rows, string(b))
		}
	}
	tracker := CompletionTracker{ExpectedChoices: 2}
	_ = tracker.Observe(finishChunk)
	if tracker.Complete() == nil {
		t.Fatal("missing requested choice accepted")
	}
}

func TestSSEMultilineEventLimit(t *testing.T) {
	line := "data: " + strings.Repeat("x", 1024) + "\n"
	err := ReadSSE(io.MultiReader(strings.NewReader(strings.Repeat(line, MaxSSEEventBytes/len(line)+1)), strings.NewReader("\n")), func(SSEEvent) error { t.Fatal("oversized event dispatched"); return nil })
	if err == nil || !strings.Contains(err.Error(), "maximum size") {
		t.Fatal(err)
	}
}

func TestArrivalIsWrittenBeforeFirstOutput(t *testing.T) {
	entered := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
	}))
	defer up.Close()
	path := filepath.Join(t.TempDir(), "proxy.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h, _ := New(config(up.URL), nil, f)
	s := httptest.NewServer(h)
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", s.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("X-Fairsched-Client-ID", "client-123")
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		resp, _ := http.DefaultClient.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
	}()
	<-entered
	b, _ := os.ReadFile(path)
	var arrival ArrivalRecord
	if json.Unmarshal(bytes.TrimSpace(b), &arrival) != nil || arrival.Event != "request_arrival" || arrival.ClientID != "client-123" || arrival.RID == "" || arrival.Tenant != "A" {
		t.Fatal(string(b))
	}
	cancel()
	<-finished
	s.Close()
	h.Wait()
	b, _ = os.ReadFile(path)
	if bytes.Count(b, []byte(`"event":"request_terminal"`)) != 1 {
		t.Fatal(string(b))
	}
}

func TestSlowReaderWriteDeadlineReleasesSlot(t *testing.T) {
	cancelled := make(chan struct{})
	big := `{"choices":[{"index":0,"delta":{"content":"` + strings.Repeat("x", 256<<10) + `"}}]}`
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 256; i++ {
			if r.Context().Err() != nil {
				break
			}
			if err := WriteSSE(w, SSEEvent{Data: big}); err != nil {
				break
			}
			w.(http.Flusher).Flush()
		}
		<-r.Context().Done()
		close(cancelled)
	}))
	defer up.Close()
	c := config(up.URL)
	c.WriteTimeoutMS = 30
	p := c.Tenants["A"]
	p.MaxInflight = 1
	c.Tenants["A"] = p
	var logs bytes.Buffer
	h, _ := New(c, nil, &logs)
	s := httptest.NewServer(h)
	conn, err := net.Dial("tcp", strings.TrimPrefix(s.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.(*net.TCPConn).SetReadBuffer(1024)
	body := `{"stream":true}`
	_, _ = fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", len(body), body)
	reader := bufio.NewReader(conn)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if line == "\r\n" {
			break
		}
	}
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		conn.Close()
		t.Fatal("slow reader left upstream alive")
	}
	conn.Close()
	s.Close()
	h.Wait()
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].State != "cancelled" || rows[0].Reason != "downstream_write_timeout" {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, rows[0].ContextCancelReason, true)
	if rows[0].ContextCancelReason != "request_context_cancelled" && rows[0].ContextCancelReason != "downstream_write_timeout" {
		t.Fatal(rows)
	}
	if rows[0].CancelTriggerReason != "downstream_write_timeout" {
		t.Fatal(rows)
	}
	if h.gates["A"].active != 0 {
		t.Fatal("slot leaked")
	}
}

func TestTokenIDsStopTTFTAndSurviveAggregation(t *testing.T) {
	if !HasToken(`{"choices":[{"index":0,"delta":{"content":""},"token_ids":[42]}]}`) || HasToken(`{"choices":[{"index":0,"prompt_token_ids":[42]}]}`) {
		t.Fatal("output/prompt ID boundary")
	}
	a := newAggregate(true)
	_ = a.add([]byte(`{"prompt_token_ids":[1],"choices":[{"index":0,"delta":{"content":""},"token_ids":[42]}]}`))
	_ = a.add([]byte(`{"choices":[{"index":0,"delta":{"content":"x"},"token_ids":[43,44],"finish_reason":"stop"}]}`))
	b, err := a.result()
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	_ = json.Unmarshal(b, &v)
	c := v["choices"].([]any)[0].(map[string]any)
	if len(c["token_ids"].([]any)) != 3 || len(v["prompt_token_ids"].([]any)) != 1 {
		t.Fatal(string(b))
	}
}

func TestServerShutdownReason(t *testing.T) {
	entered := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
	}))
	defer up.Close()
	var logs bytes.Buffer
	h, _ := New(config(up.URL), nil, &logs)
	ctx, cancel := context.WithCancelCause(context.Background())
	s := httptest.NewUnstartedServer(h)
	s.Config.BaseContext = func(net.Listener) context.Context { return ctx }
	s.Start()
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		resp := doRequest(t, s.URL, `{"stream":true}`)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	<-entered
	cancel(ErrServerShutdown)
	<-finished
	s.Close()
	h.Wait()
	rows := terminals(t, &logs)
	if len(rows) != 1 || rows[0].Reason != "proxy_shutdown" || rows[0].State != "error" {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, "proxy_shutdown", rows[0].UpstreamBodyCloseCompletedUnixMS != 0)
}

func TestTTFTDeadlineAlsoBoundsRequestBodyRead(t *testing.T) {
	c := config("http://127.0.0.1:1")
	p := c.Tenants["A"]
	p.DefaultDeadlineMS = 30
	c.Tenants["A"] = p
	var logs bytes.Buffer
	h, _ := New(c, nil, &logs)
	s := httptest.NewServer(h)
	conn, err := net.Dial("tcp", strings.TrimPrefix(s.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(time.Second))
	_, _ = io.WriteString(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 1000\r\n\r\n{")
	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		conn.Close()
		s.Close()
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	conn.Close()
	s.Close()
	h.Wait()
	rows := terminals(t, &logs)
	if resp.StatusCode != 504 || len(rows) != 1 || rows[0].State != "timeout" || rows[0].UpstreamStartedUnixMS != 0 {
		t.Fatal(resp.StatusCode, rows)
	}
	assertLocalLifecycle(t, &logs, rows[0].ContextCancelReason, false)
	if rows[0].ContextCancelReason != "request_context_cancelled" && rows[0].ContextCancelReason != "ttft_deadline_exceeded" {
		t.Fatal(rows)
	}
	if rows[0].CancelTriggerReason != "request_read_deadline_exceeded" {
		t.Fatal(rows)
	}
}

type deadlineRecorder struct{ *httptest.ResponseRecorder }

func (deadlineRecorder) SetReadDeadline(time.Time) error  { return nil }
func (deadlineRecorder) SetWriteDeadline(time.Time) error { return nil }

type lateBody struct{ done bool }

func (b *lateBody) Read(p []byte) (int, error) {
	if b.done {
		return 0, io.EOF
	}
	b.done = true
	time.Sleep(30 * time.Millisecond)
	return copy(p, `{"stream":true}`), nil
}
func (*lateBody) Close() error { return nil }

type rejectingTransport struct{ called bool }

func (t *rejectingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.called = true
	return nil, fmt.Errorf("request should not be sent")
}

func TestBodyCompletingAfterDeadlineDoesNotReachUpstream(t *testing.T) {
	c := config("http://127.0.0.1:1")
	p := c.Tenants["A"]
	p.DefaultDeadlineMS = 10
	c.Tenants["A"] = p
	transport := &rejectingTransport{}
	var logs bytes.Buffer
	h, _ := New(c, &http.Client{Transport: transport}, &logs)
	req := httptest.NewRequest("POST", "http://proxy/v1/chat/completions", nil)
	req.Body = &lateBody{}
	req.Header.Set("Authorization", "Bearer secret")
	w := deadlineRecorder{httptest.NewRecorder()}
	h.ServeHTTP(w, req)
	rows := terminals(t, &logs)
	if w.Code != 504 || transport.called || len(rows) != 1 || rows[0].State != "timeout" || rows[0].UpstreamStartedUnixMS != 0 {
		t.Fatal(w.Code, transport.called, rows)
	}
}
