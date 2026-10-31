package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"time"
)

type TenantPolicy struct {
	Class             string `json:"class"`
	DefaultDeadlineMS int64  `json:"default_deadline_ms"`
	MaxInflight       int    `json:"max_inflight"`
	MaxQueued         int    `json:"max_queued"`
	Priority          int    `json:"priority"`
}
type Config struct {
	WriteTimeoutMS   int64                   `json:"write_timeout_ms"`
	Upstream         string                  `json:"upstream"`
	Listen           string                  `json:"listen"`
	APIKeys          map[string]string       `json:"api_keys"`
	Tenants          map[string]TenantPolicy `json:"tenants"`
	SnapshotPath     string                  `json:"snapshot_path"`
	LogPath          string                  `json:"log_path"`
	PriorityBaseline bool                    `json:"priority_baseline"`
	UpstreamAPIKey   string                  `json:"upstream_api_key"`
}

func (c Config) Validate() error {
	if c.WriteTimeoutMS < 0 || c.WriteTimeoutMS > int64((24*time.Hour)/time.Millisecond) {
		return errors.New("invalid write_timeout_ms")
	}
	u, err := url.Parse(c.Upstream)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("upstream must be an http(s) base URL without query/fragment")
	}
	if len(c.APIKeys) == 0 {
		return errors.New("api_keys is empty")
	}
	for key, tenant := range c.APIKeys {
		if key == "" || tenant == "" {
			return errors.New("empty API key or tenant")
		}
		if _, ok := c.Tenants[tenant]; !ok {
			return fmt.Errorf("unknown tenant %q", tenant)
		}
	}
	for name, p := range c.Tenants {
		if p.MaxInflight < 1 || p.MaxQueued < 0 || p.DefaultDeadlineMS < 0 || p.DefaultDeadlineMS > int64((24*time.Hour)/time.Millisecond) {
			return fmt.Errorf("invalid limits for tenant %q", name)
		}
		if p.Class != "interactive" && p.Class != "analysis" && p.Class != "batch" {
			return fmt.Errorf("invalid class for tenant %q", name)
		}
	}
	return nil
}
func LoadConfig(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(b, &c); err != nil {
		return c, err
	}
	return c, c.Validate()
}

type SnapshotTenant struct {
	PrefillTokensAhead   float64            `json:"prefill_tokens_ahead"`
	OutputTokensQuantile map[string]float64 `json:"output_tokens_quantile"`
}

// Producers replace the entire JSON file using rename, never truncate in place.
type AdmissionSnapshot struct {
	Calibrated             bool                      `json:"calibrated"`
	PolicyVersion          string                    `json:"policy_version"`
	EngineInstanceID       string                    `json:"engine_instance_id"`
	StepSeq                int64                     `json:"step_seq"`
	TimestampUnixMS        int64                     `json:"timestamp_unix_ms"`
	PrefillTokensPerSecond float64                   `json:"prefill_tokens_per_second"`
	StepMS                 float64                   `json:"step_ms"`
	FreeKVTokens           int64                     `json:"free_kv_tokens"`
	Tenants                map[string]SnapshotTenant `json:"tenants"`
}
