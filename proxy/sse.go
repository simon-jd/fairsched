package proxy

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// SSEEvent is one complete event, not a network read or a model token.
type SSEEvent struct{ Event, ID, Data string }

const MaxSSEEventBytes = 8 << 20

// ReadSSE supports CRLF, comments, multiline data, and events larger than 64 KiB.
// Events without a data field are ignored as required by the event-stream format.
func ReadSSE(r io.Reader, visit func(SSEEvent) error) error {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), MaxSSEEventBytes)
	var e SSEEvent
	var data []string
	eventBytes := 0
	dispatch := func() error {
		eventBytes = 0
		if len(data) == 0 {
			e = SSEEvent{}
			return nil
		}
		e.Data = strings.Join(data, "\n")
		err := visit(e)
		e = SSEEvent{}
		data = nil
		return err
	}
	for s.Scan() {
		line := strings.TrimSuffix(s.Text(), "\r")
		eventBytes += len(line) + 1
		if eventBytes > MaxSSEEventBytes {
			return errors.New("SSE event exceeds maximum size")
		}
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			e.Event = value
		case "id":
			e.ID = value
		case "data":
			data = append(data, value)
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	// An unterminated last event is not a complete SSE event.
	return nil
}

func WriteSSE(w io.Writer, e SSEEvent) error {
	if e.Event != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", e.Event); err != nil {
			return err
		}
	}
	if e.ID != "" {
		if _, err := fmt.Fprintf(w, "id: %s\n", e.ID); err != nil {
			return err
		}
	}
	for _, line := range strings.Split(e.Data, "\n") {
		if _, err := fmt.Fprintf(w, "data: %s\n", line); err != nil {
			return err
		}
	}
	_, err := io.WriteString(w, "\n")
	return err
}

// HasToken excludes role-only and usage chunks. A content or tool-call delta is
// evidence of first generated output; SSE chunks may contain multiple tokens.
func HasToken(data string) bool {
	var v struct {
		Choices []struct {
			Text     string                     `json:"text"`
			TokenIDs []int                      `json:"token_ids"`
			Delta    map[string]json.RawMessage `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(data), &v) != nil {
		return false
	}
	for _, c := range v.Choices {
		if c.Text != "" || len(c.TokenIDs) > 0 {
			return true
		}
		for _, key := range []string{"content", "reasoning_content", "reasoning", "tool_calls", "function_call"} {
			if b := c.Delta[key]; len(b) > 0 && string(b) != "null" && string(b) != "\"\"" && string(b) != "[]" && string(b) != "{}" {
				return true
			}
		}
	}
	return false
}

// CompletionTracker validates every observed choice before accepting [DONE].
// The marker alone cannot distinguish a complete response from truncation.
type CompletionTracker struct {
	choices         map[int]bool
	ExpectedChoices int
}

func (t *CompletionTracker) Observe(data string) error {
	var v struct {
		Error   json.RawMessage `json:"error"`
		Choices []struct {
			Index        *int    `json:"index"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(data), &v); err != nil {
		return errors.New("invalid upstream SSE JSON")
	}
	if len(v.Error) > 0 && string(v.Error) != "null" {
		return errors.New("upstream SSE error")
	}
	if t.choices == nil {
		t.choices = make(map[int]bool)
	}
	for _, choice := range v.Choices {
		if choice.Index == nil || *choice.Index < 0 {
			return errors.New("invalid or missing choice index")
		}
		finished := choice.FinishReason != nil && *choice.FinishReason != ""
		t.choices[*choice.Index] = t.choices[*choice.Index] || finished
	}
	return nil
}
func (t *CompletionTracker) Complete() error {
	if len(t.choices) == 0 {
		return errors.New("upstream_empty_choices")
	}
	if t.ExpectedChoices > 0 && len(t.choices) != t.ExpectedChoices {
		return errors.New("upstream_missing_choices")
	}
	for index := 0; index < t.ExpectedChoices; index++ {
		if _, exists := t.choices[index]; !exists {
			return errors.New("upstream_missing_choices")
		}
	}
	for _, finished := range t.choices {
		if !finished {
			return errors.New("upstream_missing_finish_reason")
		}
	}
	return nil
}

func ExpectedChoiceCount(body map[string]any) int {
	n := 1
	if count, ok := body["n"].(float64); ok && count > 0 && count <= 100000 && count == float64(int(count)) {
		n = int(count)
	}
	if prompts, ok := body["prompt"].([]any); ok && len(prompts) > 0 {
		if _, tokenIDs := prompts[0].(float64); !tokenIDs {
			n *= len(prompts)
		}
	}
	return n
}

type TokenIDs struct {
	ChoiceIndex int   `json:"choice_index"`
	IDs         []int `json:"token_ids"`
}

// OutputTokenIDs excludes prompt_token_ids, which are input rather than output.
func OutputTokenIDs(data string) []TokenIDs {
	var v struct {
		Choices []struct {
			Index int   `json:"index"`
			IDs   []int `json:"token_ids"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(data), &v) != nil {
		return nil
	}
	var result []TokenIDs
	for _, c := range v.Choices {
		if len(c.IDs) > 0 {
			result = append(result, TokenIDs{ChoiceIndex: c.Index, IDs: c.IDs})
		}
	}
	return result
}
