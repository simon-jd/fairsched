package loadgen

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDONEWithoutFinishIsError(t *testing.T) {
	for _, data := range []string{chunk, `{"choices":[{"index":0,"finish_reason":"stop"},{"index":1,"delta":{"content":"unfinished"}}]}`} {
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			output(w, data)
			output(w, "[DONE]")
		}))
		rs, err := Run(context.Background(), []Arrival{arrival("a", 0)}, Options{BaseURL: s.URL})
		s.Close()
		if err != nil || rs[0].State != "error" {
			t.Fatal(rs, err)
		}
	}
}

func TestTokenIDsHaveChunkReceiptTimestamps(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["return_token_ids"] != true || r.Header.Get("X-Fairsched-Client-ID") != "ids" {
			t.Error("missing requested IDs/correlation")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		output(w, `{"prompt_token_ids":[1,2],"choices":[{"index":0,"delta":{"content":""},"token_ids":[41,42]}]}`)
		output(w, `{"choices":[{"index":0,"delta":{"content":"x"},"token_ids":[43],"finish_reason":"stop"}],"usage":{"completion_tokens":3}}`)
		output(w, "[DONE]")
	}))
	defer s.Close()
	rs, err := Run(context.Background(), []Arrival{arrival("ids", 0)}, Options{BaseURL: s.URL})
	if err != nil || rs[0].State != "completed" || rs[0].ObservedOutputTokens != 3 || rs[0].OutputChunks != 2 || len(rs[0].Events[0].Tokens) != 2 {
		t.Fatal(rs, err)
	}
	for _, event := range rs[0].Events {
		for _, token := range event.Tokens {
			if token.AtUnixNS != event.AtUnixNS || token.ChoiceIndex != 0 || token.ID < 41 {
				t.Fatal(token, event)
			}
		}
	}
	if rs[0].FirstOutputUnixNS != rs[0].Events[0].AtUnixNS {
		t.Fatal("empty text token did not start TTFT")
	}
}

func TestHalfCloseKeepsReadingResponse(t *testing.T) {
	halfClosed := make(chan struct{})
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		conn, rw, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		_ = conn.SetDeadline(time.Now().Add(2 * time.Second))
		_, err = io.ReadAll(rw)
		if err != nil {
			t.Error(err)
			return
		}
		close(halfClosed)
		body := "data: " + chunk + "\n\ndata: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
		_, _ = fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", len(body), body)
	}))
	defer s.Close()
	a := arrival("half", 0)
	delay := int64(10)
	a.HalfCloseAfterMS = &delay
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL})
	if err != nil || rs[0].State != "completed" || rs[0].HalfClosedUnixNS == 0 || rs[0].HalfCloseError != "" || rs[0].CancelledUnixNS != 0 {
		t.Fatal(rs, err)
	}
	select {
	case <-halfClosed:
	default:
		t.Fatal("server did not observe FIN")
	}
}

func TestSlowReadInjectionAndCancellation(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		output(w, chunk)
		output(w, `{"choices":[{"index":0,"finish_reason":"stop"}]}`)
		output(w, "[DONE]")
	}))
	defer s.Close()
	a := arrival("slow", 0)
	a.ReadDelayMS = 2
	a.ReadChunkBytes = 16
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL})
	if err != nil || rs[0].State != "completed" || rs[0].FinishedUnixNS-rs[0].SentUnixNS < int64(10*time.Millisecond) {
		t.Fatal(rs, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := slowReader{ctx: ctx, reader: strings.NewReader("x"), delay: time.Second, chunk: 1}
	before := time.Now()
	if _, err := r.Read(make([]byte, 1)); err != context.Canceled || time.Since(before) > 100*time.Millisecond {
		t.Fatal("slow reader ignored cancellation", err)
	}
}
