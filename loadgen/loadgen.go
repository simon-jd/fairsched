// Package loadgen replays open-loop arrivals. Every trace row yields exactly one
// result, including arrivals rejected, cancelled, or never sent after run cancellation.
package loadgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/simon-jd/fairsched/proxy"
)

type Arrival struct {
	ID                      string          `json:"id"`
	AtMS                    int64           `json:"at_ms"`
	Tenant                  string          `json:"tenant"`
	APIKey                  string          `json:"api_key"`
	Path                    string          `json:"path"`
	Body                    json.RawMessage `json:"body"`
	CancelAfterMS           *int64          `json:"cancel_after_ms,omitempty"`
	CancelAfterOutputChunks int             `json:"cancel_after_output_chunks,omitempty"`
	TimeoutMS               int64           `json:"timeout_ms,omitempty"`
	DeadlineMS              int64           `json:"deadline_ms,omitempty"`
	ReadDelayMS             int64           `json:"read_delay_ms,omitempty"`
	ReadChunkBytes          int             `json:"read_chunk_bytes,omitempty"`
	HalfCloseAfterMS        *int64          `json:"half_close_after_ms,omitempty"`
}
type Token struct {
	ChoiceIndex int   `json:"choice_index"`
	ID          int   `json:"token_id"`
	AtUnixNS    int64 `json:"at_unix_ns"`
}
type Event struct {
	AtUnixNS  int64   `json:"at_unix_ns"`
	Event     string  `json:"event,omitempty"`
	Data      string  `json:"data"`
	HasOutput bool    `json:"has_output"`
	Tokens    []Token `json:"tokens,omitempty"`
}
type Result struct {
	ID                   string          `json:"id"`
	RID                  string          `json:"rid,omitempty"`
	Tenant               string          `json:"tenant"`
	ScheduledUnixNS      int64           `json:"scheduled_unix_ns"`
	SentUnixNS           int64           `json:"sent_unix_ns,omitempty"`
	FirstOutputUnixNS    int64           `json:"first_output_unix_ns,omitempty"`
	LastOutputUnixNS     int64           `json:"last_output_unix_ns,omitempty"`
	FinishedUnixNS       int64           `json:"finished_unix_ns"`
	CancelledUnixNS      int64           `json:"cancelled_unix_ns,omitempty"`
	HalfClosedUnixNS     int64           `json:"half_closed_unix_ns,omitempty"`
	HalfCloseError       string          `json:"half_close_error,omitempty"`
	State                string          `json:"state"`
	Status               int             `json:"status,omitempty"`
	Reason               string          `json:"reason,omitempty"`
	DeadlineMS           int64           `json:"deadline_ms,omitempty"`
	Stream               bool            `json:"stream"`
	OutputChunks         int             `json:"output_chunks"`
	ObservedOutputTokens int             `json:"observed_output_tokens"`
	CompletionTokens     *int            `json:"completion_tokens,omitempty"`
	Events               []Event         `json:"events,omitempty"`
	Response             json.RawMessage `json:"response,omitempty"`
}

func ReadTrace(r io.Reader) ([]Arrival, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 16<<20)
	var rows []Arrival
	seen := map[string]bool{}
	line := 0
	for s.Scan() {
		line++
		if len(bytes.TrimSpace(s.Bytes())) == 0 {
			continue
		}
		var a Arrival
		if err := json.Unmarshal(s.Bytes(), &a); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if a.ID == "" || seen[a.ID] {
			return nil, fmt.Errorf("line %d: empty or duplicate id", line)
		}
		if len(a.ID) > 128 {
			return nil, fmt.Errorf("line %d: id must be at most 128 printable ASCII bytes", line)
		}
		for _, b := range []byte(a.ID) {
			if b < 32 || b > 126 {
				return nil, fmt.Errorf("line %d: id must be printable ASCII", line)
			}
		}
		seen[a.ID] = true
		if a.AtMS < 0 || a.AtMS > int64((7*24*time.Hour)/time.Millisecond) || a.TimeoutMS < 0 || a.TimeoutMS > int64((24*time.Hour)/time.Millisecond) || a.DeadlineMS < 0 || a.CancelAfterOutputChunks < 0 || (a.CancelAfterMS != nil && (*a.CancelAfterMS < 0 || *a.CancelAfterMS > int64((24*time.Hour)/time.Millisecond))) {
			return nil, fmt.Errorf("line %d: invalid timing", line)
		}
		if a.ReadDelayMS < 0 || a.ReadDelayMS > int64((24*time.Hour)/time.Millisecond) || a.ReadChunkBytes < 0 || a.ReadChunkBytes > 1<<20 || (a.HalfCloseAfterMS != nil && (*a.HalfCloseAfterMS < 0 || *a.HalfCloseAfterMS > int64((24*time.Hour)/time.Millisecond))) {
			return nil, fmt.Errorf("line %d: invalid fault injection", line)
		}
		if a.Path == "" {
			a.Path = "/v1/chat/completions"
		}
		if a.Path != "/v1/chat/completions" && a.Path != "/v1/completions" {
			return nil, fmt.Errorf("line %d: unsupported path", line)
		}
		var body map[string]any
		if json.Unmarshal(a.Body, &body) != nil || body == nil {
			return nil, fmt.Errorf("line %d: body must be JSON object", line)
		}
		rows = append(rows, a)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].AtMS < rows[j].AtMS })
	return rows, nil
}

type Options struct {
	BaseURL string
	Client  *http.Client
	Timeout time.Duration
	Results io.Writer
	// Optional bounded record mode. Ordinary JSONL output and file permissions stay
	// unchanged. Events are separate <=64KiB records, never a giant result row.
	BoundedRecords bool
}

var errProgrammed = errors.New("programmed_cancel")
var errComplete = errors.New("complete")

// Run schedules against one fixed origin. A slow request never delays another
// arrival. There is deliberately no client-side concurrency gate.
func Run(ctx context.Context, arrivals []Arrival, opt Options) ([]Result, error) {
	if opt.Client == nil {
		opt.Client = &http.Client{}
	}
	if opt.Timeout <= 0 {
		opt.Timeout = 2 * time.Minute
	}
	start := time.Now()
	results := make([]Result, len(arrivals))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var writeErr error
	store := func(i int, r Result) {
		mu.Lock()
		defer mu.Unlock()
		results[i] = r
		if opt.Results != nil {
			var err error
			if opt.BoundedRecords {
				var b struct {
					MaxTokens int `json:"max_tokens"`
				}
				_ = json.Unmarshal(arrivals[i].Body, &b)
				err = writeRecords(opt.Results, r, b.MaxTokens)
			} else {
				err = json.NewEncoder(opt.Results).Encode(r)
			}
			if err != nil && writeErr == nil {
				writeErr = err
			}
		}
		if opt.BoundedRecords {
			results[i].Events = nil
		}
	}
	for i, a := range arrivals {
		due := start.Add(time.Duration(a.AtMS) * time.Millisecond)
		timer := time.NewTimer(time.Until(due))
		select {
		case <-timer.C:
		case <-ctx.Done():
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		}
		if ctx.Err() != nil {
			store(i, Result{ID: a.ID, Tenant: a.Tenant, ScheduledUnixNS: due.UnixNano(), FinishedUnixNS: time.Now().UnixNano(), State: "cancelled", Reason: "run_cancelled_before_send", DeadlineMS: a.DeadlineMS})
			continue
		}
		wg.Add(1)
		go func(i int, a Arrival, due time.Time) { defer wg.Done(); store(i, runOne(ctx, a, due, opt)) }(i, a, due)
	}
	wg.Wait()
	return results, writeErr
}
func runOne(parent context.Context, a Arrival, due time.Time, opt Options) Result {
	r := Result{ID: a.ID, Tenant: a.Tenant, ScheduledUnixNS: due.UnixNano(), State: "error", DeadlineMS: a.DeadlineMS}
	var finishHalfClose func() halfCloseResult
	finish := func() Result {
		if finishHalfClose != nil {
			result := finishHalfClose()
			r.HalfClosedUnixNS = result.at
			r.HalfCloseError = result.err
		}
		r.FinishedUnixNS = time.Now().UnixNano()
		return r
	}
	timeout := opt.Timeout
	if a.TimeoutMS > 0 {
		timeout = time.Duration(a.TimeoutMS) * time.Millisecond
	}
	timedCtx, stop := context.WithTimeout(parent, timeout)
	defer stop()
	ctx, cancel := context.WithCancelCause(timedCtx)
	defer cancel(nil)
	var cancelMu sync.Mutex
	var cancelledAt int64
	programCancel := func() {
		cancelMu.Lock()
		if cancelledAt == 0 {
			cancelledAt = time.Now().UnixNano()
		}
		cancelMu.Unlock()
		cancel(errProgrammed)
	}
	classify := func(reason string) {
		switch {
		case errors.Is(context.Cause(ctx), errProgrammed):
			r.State = "cancelled"
			r.Reason = "programmed_cancel"
		case errors.Is(ctx.Err(), context.DeadlineExceeded):
			r.State = "timeout"
			r.Reason = "request_timeout"
		case ctx.Err() != nil:
			r.State = "cancelled"
			r.Reason = "run_cancelled"
		default:
			r.State = "error"
			r.Reason = reason
		}
		cancelMu.Lock()
		r.CancelledUnixNS = cancelledAt
		cancelMu.Unlock()
	}
	var body map[string]any
	_ = json.Unmarshal(a.Body, &body)
	if opt.BoundedRecords {
		limit, ok := body["max_tokens"].(float64)
		if !ok || limit < 1 || limit > 2048 || limit != float64(int(limit)) || body["stream"] != true {
			r.Reason = "bounded_records_requires_stream_max_tokens_1_to_2048"
			return finish()
		}
	}
	if _, exists := body["return_token_ids"]; !exists {
		body["return_token_ids"] = true
	}
	stream, exists := body["stream"].(bool)
	if !exists {
		stream = true
		body["stream"] = true
	}
	r.Stream = stream
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	payload, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(opt.BaseURL, "/")+a.Path, bytes.NewReader(payload))
	if err != nil {
		r.Reason = err.Error()
		return finish()
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.APIKey)
	req.Header.Set("X-Fairsched-Client-ID", a.ID)
	client := opt.Client
	if a.HalfCloseAfterMS != nil {
		client, finishHalfClose, err = halfCloseClient(req, client, time.Duration(*a.HalfCloseAfterMS)*time.Millisecond)
		if err != nil {
			r.Reason = err.Error()
			return finish()
		}
	}
	r.SentUnixNS = time.Now().UnixNano()
	if a.CancelAfterMS != nil {
		timer := time.AfterFunc(time.Duration(*a.CancelAfterMS)*time.Millisecond, programCancel)
		defer timer.Stop()
	}
	resp, err := client.Do(req)
	if err != nil {
		classify("http_request_failed: " + err.Error())
		return finish()
	}
	defer resp.Body.Close()
	r.Status = resp.StatusCode
	r.RID = resp.Header.Get("X-Fairsched-Request-ID")
	var responseReader io.Reader = resp.Body
	if a.ReadDelayMS > 0 || a.ReadChunkBytes > 0 {
		chunk := a.ReadChunkBytes
		if chunk == 0 {
			chunk = 4096
		}
		responseReader = slowReader{ctx: ctx, reader: resp.Body, delay: time.Duration(a.ReadDelayMS) * time.Millisecond, chunk: chunk}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, readErr := io.ReadAll(io.LimitReader(responseReader, 1<<20))
		if readErr != nil {
			classify("error_body_read_failed")
			return finish()
		}
		if json.Valid(b) {
			r.Response = b
		}
		r.State = "error"
		r.Reason = "upstream_http_error"
		if resp.StatusCode == 429 || resp.StatusCode == 401 || resp.StatusCode == 413 || resp.StatusCode == 400 {
			r.State = "rejected"
		}
		if resp.StatusCode == 504 {
			r.State = "timeout"
		}
		return finish()
	}
	usage := func(data []byte) {
		var v struct {
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal(data, &v) == nil && v.Usage != nil {
			n := v.Usage.CompletionTokens
			r.CompletionTokens = &n
		}
	}
	if !stream {
		b, err := io.ReadAll(io.LimitReader(responseReader, 16<<20))
		if err != nil {
			classify("response_read_failed")
			return finish()
		}
		if !json.Valid(b) {
			r.Reason = "invalid_json_response"
			return finish()
		}
		r.Response = b
		usage(b)
		r.State = "completed"
		return finish()
	}
	if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		r.Reason = "expected_event_stream"
		return finish()
	}
	done := false
	recordedBytes := 0
	tracker := proxy.CompletionTracker{ExpectedChoices: proxy.ExpectedChoiceCount(body)}
	err = proxy.ReadSSE(responseReader, func(e proxy.SSEEvent) error {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		now := time.Now().UnixNano()
		out := proxy.HasToken(e.Data)
		event := Event{AtUnixNS: now, Event: e.Event, Data: e.Data, HasOutput: out}
		for _, batch := range proxy.OutputTokenIDs(e.Data) {
			for _, id := range batch.IDs {
				event.Tokens = append(event.Tokens, Token{ChoiceIndex: batch.ChoiceIndex, ID: id, AtUnixNS: now})
				r.ObservedOutputTokens++
			}
		}
		if opt.BoundedRecords && (len(r.Events) >= int(body["max_tokens"].(float64))+2 || len(event.Tokens) > 64) {
			return errors.New("bounded_records_event_budget_exceeded")
		}
		if opt.BoundedRecords {
			b, e := json.Marshal(event)
			if e != nil {
				return e
			}
			recordedBytes += len(b) + 256
			if len(b) > 65536-256 || recordedBytes > int(body["max_tokens"].(float64))*1024+2048 {
				return errors.New("bounded_records_byte_budget_exceeded")
			}
		}
		r.Events = append(r.Events, event)
		if e.Data == "[DONE]" {
			if err := tracker.Complete(); err != nil {
				return err
			}
			done = true
			return errComplete
		}
		if err := tracker.Observe(e.Data); err != nil {
			return err
		}
		usage([]byte(e.Data))
		if out {
			r.OutputChunks++
			if r.FirstOutputUnixNS == 0 {
				r.FirstOutputUnixNS = now
			}
			r.LastOutputUnixNS = now
			if a.CancelAfterOutputChunks > 0 && r.OutputChunks >= a.CancelAfterOutputChunks {
				programCancel()
				return errProgrammed
			}
		}
		return nil
	})
	if done && errors.Is(err, errComplete) {
		r.State = "completed"
	} else {
		reason := "incomplete_sse_stream"
		if err != nil {
			reason = err.Error()
		}
		classify(reason)
	}
	return finish()
}

func writeRecords(w io.Writer, r Result, maxTokens int) error {
	// Bounded records allow at most 2048 tokens per request. Bound accepted bytes
	// independently of event count, including malformed upstream data.
	if maxTokens < 1 || maxTokens > 2048 {
		return errors.New("bounded_records_max_tokens_out_of_range")
	}
	budget := maxTokens*1024 + 4096
	for i, e := range r.Events {
		b, err := json.Marshal(map[string]any{"schema": 1, "kind": "client_event", "id": r.ID, "seq": i, "event": e})
		if err != nil {
			return err
		}
		if len(b) > 65536 || len(b)+1 > budget {
			return errors.New("bounded_records_event_row_budget_exceeded")
		}
		budget -= len(b) + 1
		if _, err = w.Write(append(b, '\n')); err != nil {
			return err
		}
	}
	n := len(r.Events)
	r.Events = nil
	b, err := json.Marshal(map[string]any{"schema": 1, "kind": "client_terminal", "id": r.ID, "event_rows": n, "result": r})
	if err != nil {
		return err
	}
	if len(b) > 65536 || len(b)+1 > budget {
		return errors.New("bounded_records_terminal_row_budget_exceeded")
	}
	_, err = w.Write(append(b, '\n'))
	return err
}
