package loadgen

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSessionIsSentAsHeader(t *testing.T) {
	got := make(chan string, 2)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.Header.Get("X-Fairsched-Session-ID")
		w.Header().Set("Content-Type", "text/event-stream")
		output(w, chunk)
		output(w, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		output(w, "[DONE]")
	}))
	defer s.Close()
	with := arrival("with", 0)
	with.Session = "B-s3"
	rs, err := Run(context.Background(), []Arrival{with, arrival("without", 0)}, Options{BaseURL: s.URL})
	if err != nil || rs[0].State != "completed" || rs[1].State != "completed" {
		t.Fatal(rs, err)
	}
	seen := map[string]bool{<-got: true, <-got: true}
	if !seen["B-s3"] || !seen[""] {
		t.Fatalf("session headers %v", seen)
	}
}

func TestSessionMustBePrintableASCII(t *testing.T) {
	row := func(session string) string {
		return `{"id":"a","at_ms":0,"tenant":"A","api_key":"k","path":"/v1/completions","body":{},"session":` + session + "}\n"
	}
	if rows, err := ReadTrace(strings.NewReader(row(`"B-s1"`))); err != nil || rows[0].Session != "B-s1" {
		t.Fatal(rows, err)
	}
	for _, session := range []string{`"a\u0007"`, `"` + strings.Repeat("s", 129) + `"`} {
		if _, err := ReadTrace(strings.NewReader(row(session))); err == nil {
			t.Fatalf("session %s accepted", session)
		}
	}
}
