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
	"testing"
	"time"
)

func assertLocalLifecycle(t *testing.T, logs *bytes.Buffer, reason string, closed bool) {
	t.Helper()
	rows := terminals(t, logs)
	if len(rows) != 1 {
		t.Fatalf("terminal count: %v", rows)
	}
	rec := rows[0]
	var cancellations, closes, triggers int
	dec := json.NewDecoder(bytes.NewReader(logs.Bytes()))
	for {
		var row LocalLifecycleRecord
		if err := dec.Decode(&row); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if row.Event != "request_context_cancel_observed" && row.Event != "upstream_body_close_completed" && row.Event != "request_cancel_trigger_observed" {
			continue
		}
		if row.RID != rec.RID || row.ObservedUnixMS < rec.ArrivedUnixMS || row.ObservedUnixMS > rec.FinishedUnixMS {
			t.Fatal(row, rec)
		}
		switch row.Event {
		case "request_context_cancel_observed":
			cancellations++
			if row.Reason != reason || row.ObservedUnixMS != rec.ContextCancelObservedUnixMS {
				t.Fatal(row, rec)
			}
		case "upstream_body_close_completed":
			closes++
			if row.ObservedUnixMS != rec.UpstreamBodyCloseCompletedUnixMS {
				t.Fatal(row, rec)
			}
		case "request_cancel_trigger_observed":
			triggers++
			if row.ObservedUnixMS != rec.CancelTriggerObservedUnixMS || row.Reason != rec.CancelTriggerReason {
				t.Fatal(row, rec)
			}
		}
	}
	if reason == "" {
		if cancellations != 0 || rec.ContextCancelReason != "" || rec.ContextCancelObservedUnixMS != 0 || triggers != 0 || rec.CancelTriggerReason != "" {
			t.Fatal("normal completion mislabeled as cancellation", rec)
		}
	} else if cancellations != 1 || rec.ContextCancelReason != reason || rec.ContextCancelObservedUnixMS == 0 {
		t.Fatal("missing cancellation observation", rec)
	}
	if rec.CancelTriggerObservedUnixMS != 0 && triggers != 1 {
		t.Fatal("missing trigger observation", rec)
	}
	if closed {
		if closes != 1 || rec.UpstreamBodyCloseCompletedUnixMS == 0 {
			t.Fatal("missing body close observation", rec)
		}
	} else if closes != 0 || rec.UpstreamBodyCloseCompletedUnixMS != 0 {
		t.Fatal("body close fabricated before upstream response", rec)
	}
}

type heldCloseBody struct {
	io.Reader
	closeEntered chan struct{}
	releaseClose chan struct{}
}

func (b *heldCloseBody) Close() error { close(b.closeEntered); <-b.releaseClose; return nil }

func TestCompletedEmptyOutputDoesNotAcquireTimeoutDuringBodyCleanup(t *testing.T) {
	closeEntered, releaseClose := make(chan struct{}), make(chan struct{})
	transport := transportFunc(func(r *http.Request) (*http.Response, error) {
		body := &heldCloseBody{strings.NewReader("data: " + finishChunk + "\n\ndata: [DONE]\n\n"), closeEntered, releaseClose}
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: body}, nil
	})
	c := config("http://unused.invalid")
	p := c.Tenants["A"]
	p.DefaultDeadlineMS = 25
	c.Tenants["A"] = p
	var logs bytes.Buffer
	h, _ := New(c, &http.Client{Transport: transport}, &logs)
	req := httptest.NewRequest("POST", "http://proxy/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("Authorization", "Bearer secret")
	done := make(chan struct{})
	go func() { defer close(done); h.ServeHTTP(deadlineRecorder{httptest.NewRecorder()}, req) }()
	select {
	case <-closeEntered:
	case <-time.After(time.Second):
		t.Fatal("close not reached")
	}
	time.Sleep(50 * time.Millisecond)
	close(releaseClose)
	<-done
	if rows := terminals(t, &logs); rows[0].State != "completed" {
		t.Fatal(rows)
	}
	assertLocalLifecycle(t, &logs, "", true)
}

type cancelReadBody struct {
	ctx          context.Context
	closeEntered chan struct{}
	releaseClose chan struct{}
}

func (b *cancelReadBody) Read([]byte) (int, error) { <-b.ctx.Done(); return 0, b.ctx.Err() }
func (b *cancelReadBody) Close() error             { close(b.closeEntered); <-b.releaseClose; return nil }

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCancellationEventSurvivesBlockedBodyCloseBeforeTerminal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "proxy.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	closeEntered, releaseClose := make(chan struct{}), make(chan struct{})
	transport := transportFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: &cancelReadBody{r.Context(), closeEntered, releaseClose}}, nil
	})
	c := config("http://unused.invalid")
	p := c.Tenants["A"]
	p.DefaultDeadlineMS = 20
	c.Tenants["A"] = p
	h, _ := New(c, &http.Client{Transport: transport}, f)
	req := httptest.NewRequest("POST", "http://proxy/v1/chat/completions", bytes.NewBufferString(`{"stream":true}`))
	req.Header.Set("Authorization", "Bearer secret")
	done := make(chan struct{})
	go func() { defer close(done); h.ServeHTTP(deadlineRecorder{httptest.NewRecorder()}, req) }()
	select {
	case <-closeEntered:
	case <-time.After(time.Second):
		t.Fatal("close not reached")
	}
	b, _ := os.ReadFile(path)
	if bytes.Count(b, []byte(`"event":"request_context_cancel_observed"`)) != 1 || bytes.Contains(b, []byte(`"event":"upstream_body_close_completed"`)) || bytes.Contains(b, []byte(`"event":"request_terminal"`)) {
		close(releaseClose)
		<-done
		t.Fatal(string(b))
	}
	close(releaseClose)
	<-done
	b, _ = os.ReadFile(path)
	assertLocalLifecycle(t, bytes.NewBuffer(b), "ttft_deadline_exceeded", true)
}
