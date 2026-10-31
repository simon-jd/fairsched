package proxy

import (
	"encoding/json"
	"errors"
	"sort"
)

type aggregate struct {
	chat    bool
	root    map[string]any
	choices map[int]map[string]any
}

func newAggregate(chat bool) *aggregate {
	return &aggregate{chat: chat, root: map[string]any{}, choices: map[int]map[string]any{}}
}
func appendFields(dst, src map[string]any) {
	for k, v := range src {
		if str, ok := v.(string); ok {
			old, _ := dst[k].(string)
			dst[k] = old + str
		} else if v != nil {
			dst[k] = v
		}
	}
}
func (a *aggregate) add(b []byte) error {
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	for _, k := range []string{"id", "created", "model", "system_fingerprint", "service_tier", "usage", "prompt_logprobs", "prompt_token_ids"} {
		if x, ok := v[k]; ok && x != nil {
			a.root[k] = x
		}
	}
	cs, _ := v["choices"].([]any)
	for _, raw := range cs {
		c, ok := raw.(map[string]any)
		if !ok {
			return errors.New("invalid upstream choice")
		}
		idxF, ok := c["index"].(float64)
		if !ok {
			return errors.New("missing choice index")
		}
		idx := int(idxF)
		out := a.choices[idx]
		if out == nil {
			out = map[string]any{"index": idx, "finish_reason": nil}
			if a.chat {
				out["message"] = map[string]any{"role": "assistant", "content": ""}
			} else {
				out["text"] = ""
			}
			a.choices[idx] = out
		}
		if f := c["finish_reason"]; f != nil {
			out["finish_reason"] = f
		}
		if ids, ok := c["token_ids"].([]any); ok {
			previous, _ := out["token_ids"].([]any)
			out["token_ids"] = append(previous, ids...)
		}
		for _, key := range []string{"prompt_token_ids", "prompt_logprobs", "stop_reason"} {
			if value, exists := c[key]; exists && value != nil {
				out[key] = value
			}
		}
		if logs, ok := c["logprobs"].(map[string]any); ok {
			combined, _ := out["logprobs"].(map[string]any)
			if combined == nil {
				combined = map[string]any{}
				out["logprobs"] = combined
			}
			for key, value := range logs {
				if values, ok := value.([]any); ok {
					previous, _ := combined[key].([]any)
					combined[key] = append(previous, values...)
				} else if value != nil {
					combined[key] = value
				}
			}
		}
		if !a.chat {
			if s, ok := c["text"].(string); ok {
				out["text"] = out["text"].(string) + s
			}
			continue
		}
		delta, _ := c["delta"].(map[string]any)
		msg := out["message"].(map[string]any)
		for k, val := range delta {
			switch k {
			case "role":
				msg[k] = val
			case "tool_calls":
				calls, _ := val.([]any)
				stored, _ := msg[k].(map[int]map[string]any)
				if stored == nil {
					stored = map[int]map[string]any{}
				}
				for _, raw := range calls {
					tc, ok := raw.(map[string]any)
					if !ok {
						return errors.New("invalid tool delta")
					}
					idxN, ok := tc["index"].(float64)
					if !ok {
						return errors.New("missing tool index")
					}
					i := int(idxN)
					t := stored[i]
					if t == nil {
						t = map[string]any{}
						stored[i] = t
					}
					for key, z := range tc {
						if key == "index" {
							continue
						}
						if key == "function" {
							fn, _ := t[key].(map[string]any)
							if fn == nil {
								fn = map[string]any{}
								t[key] = fn
							}
							src, _ := z.(map[string]any)
							appendFields(fn, src)
						} else {
							t[key] = z
						}
					}
				}
				msg[k] = stored
			case "function_call":
				fn, _ := msg[k].(map[string]any)
				if fn == nil {
					fn = map[string]any{}
					msg[k] = fn
				}
				src, _ := val.(map[string]any)
				appendFields(fn, src)
			default:
				if s, ok := val.(string); ok {
					prev, _ := msg[k].(string)
					msg[k] = prev + s
				} else if val != nil {
					msg[k] = val
				}
			}
		}
	}
	return nil
}
func (a *aggregate) result() ([]byte, error) {
	if len(a.choices) == 0 {
		return nil, errors.New("upstream_empty_choices")
	}
	indices := make([]int, 0, len(a.choices))
	for i := range a.choices {
		indices = append(indices, i)
	}
	sort.Ints(indices)
	cs := make([]any, 0, len(indices))
	for _, i := range indices {
		c := a.choices[i]
		if c["finish_reason"] == nil {
			return nil, errors.New("upstream_missing_finish_reason")
		}
		if a.chat {
			msg := c["message"].(map[string]any)
			if tools, ok := msg["tool_calls"].(map[int]map[string]any); ok {
				ids := make([]int, 0, len(tools))
				for id := range tools {
					ids = append(ids, id)
				}
				sort.Ints(ids)
				list := make([]any, 0, len(ids))
				for _, id := range ids {
					list = append(list, tools[id])
				}
				msg["tool_calls"] = list
			}
		}
		cs = append(cs, c)
	}
	a.root["choices"] = cs
	a.root["object"] = "text_completion"
	if a.chat {
		a.root["object"] = "chat.completion"
	}
	return json.Marshal(a.root)
}
