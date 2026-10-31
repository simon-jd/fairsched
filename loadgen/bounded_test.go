package loadgen

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

type errorWriter struct{}

func (errorWriter) Write(p []byte) (int, error) { return 0, errors.New("write failed") }
func TestBoundedRecordIOAndByteFailure(t *testing.T) {
	r := Result{ID: "r", State: "completed", Events: []Event{{Data: "[DONE]"}}}
	if err := writeRecords(errorWriter{}, r, 128); err == nil {
		t.Fatal("lost writer error")
	}
	if err := writeRecords(&bytes.Buffer{}, r, 2049); err == nil {
		t.Fatal("unbounded output accepted")
	}
	var b bytes.Buffer
	if err := writeRecords(&b, r, 128); err != nil {
		t.Fatal(err)
	}
	if bytes.Count(b.Bytes(), []byte("\n")) != 2 {
		t.Fatal("missing event or terminal")
	}
}
func TestBoundedRecordsActualRun(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		output(w, `{"choices":[{"index":0,"token_ids":[7],"finish_reason":null}]}`)
		output(w, `{"choices":[{"index":0,"token_ids":[],"finish_reason":"length"}],"usage":{"completion_tokens":1}}`)
		output(w, "[DONE]")
	}))
	defer s.Close()
	a := arrival("r", 0)
	a.Body = json.RawMessage(`{"stream":true,"max_tokens":1}`)
	var b bytes.Buffer
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL, Results: &b, BoundedRecords: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 || rs[0].State != "completed" || rs[0].ObservedOutputTokens != 1 || len(rs[0].Events) != 0 {
		t.Fatal(rs)
	}
	if bytes.Count(b.Bytes(), []byte("\n")) != 4 {
		t.Fatal("incomplete rows")
	}
	for _, line := range bytes.Split(bytes.TrimSpace(b.Bytes()), []byte("\n")) {
		if len(line) > 65536 || !json.Valid(line) {
			t.Fatal("bad bounded JSON")
		}
	}
}
func TestBoundedRecordsRejectAStreamOverItsEventBudget(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 5; i++ { // max_tokens 1 allows at most 3 events
			output(w, `{"choices":[{"index":0,"token_ids":[7],"finish_reason":null}]}`)
		}
		output(w, "[DONE]")
	}))
	defer s.Close()
	a := arrival("r", 0)
	a.Body = json.RawMessage(`{"stream":true,"max_tokens":1}`)
	rs, err := Run(context.Background(), []Arrival{a}, Options{BaseURL: s.URL, Results: &bytes.Buffer{}, BoundedRecords: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 || rs[0].State != "error" || rs[0].Reason != "bounded_records_event_budget_exceeded" {
		t.Fatal(rs)
	}
}
