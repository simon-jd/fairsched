package loadgen

import (
	"math"
	"sort"
)

type Distribution struct {
	Samples int      `json:"samples"`
	P50MS   *float64 `json:"p50_ms,omitempty"`
	P99MS   *float64 `json:"p99_ms,omitempty"`
}
type TenantReport struct {
	Arrivals                      int            `json:"arrivals"`
	States                        map[string]int `json:"states"`
	Completed                     int            `json:"completed"`
	SuccessRate                   float64        `json:"success_rate"`
	RejectionRate                 float64        `json:"rejection_rate"`
	TimeoutRate                   float64        `json:"timeout_rate"`
	SLOMet                        int            `json:"slo_met"`
	SLOMeasured                   int            `json:"slo_measured"`
	SLOGoodputPerSecond           float64        `json:"slo_goodput_per_second"`
	SLOSuccessFractionAllArrivals float64        `json:"slo_success_fraction_all_arrivals"`
	TTFT                          Distribution   `json:"ttft"`
	MeanTPOT                      Distribution   `json:"mean_tpot"`
	DispatchLag                   Distribution   `json:"dispatch_lag"`
	CompletionTokens              int            `json:"completion_tokens"`
	MissingTokenUsage             int            `json:"missing_token_usage"`
	OutputChunks                  int            `json:"output_chunks"`
	ObservedOutputTokens          int            `json:"observed_output_tokens"`
}
type Report struct {
	Arrivals        int                     `json:"arrivals"`
	DurationSeconds float64                 `json:"duration_seconds"`
	Tenants         map[string]TenantReport `json:"tenants"`
	Notes           []string                `json:"notes"`
}

func distribution(samples []float64) Distribution {
	d := Distribution{Samples: len(samples)}
	if len(samples) == 0 {
		return d
	}
	sort.Float64s(samples)
	p50 := samples[int(math.Ceil(float64(len(samples))*.5))-1]
	d.P50MS = &p50
	if len(samples) >= 100 {
		p99 := samples[int(math.Ceil(float64(len(samples))*.99))-1]
		d.P99MS = &p99
	}
	return d
}
func Summarize(results []Result) Report {
	r := Report{Arrivals: len(results), Tenants: map[string]TenantReport{}, Notes: []string{"SSE output chunks are not model tokens; completion_tokens uses server usage only.", "p99 is omitted for fewer than 100 samples. TTFT is measured only from streaming content/tool output.", "mean_tpot uses (last output chunk time - first output chunk time)/(completion_tokens - 1); it is a per-request estimate, not individual token latency.", "Every arrival is included in success/rejection/timeout/SLO denominators. TTFT samples include requests that later fail.", "Non-streaming requests cannot expose client-observed TTFT and are excluded from SLO successes."}}
	if len(results) == 0 {
		return r
	}
	start := results[0].ScheduledUnixNS
	end := start
	by := map[string][]Result{}
	for _, x := range results {
		if x.ScheduledUnixNS < start {
			start = x.ScheduledUnixNS
		}
		if x.FinishedUnixNS > end {
			end = x.FinishedUnixNS
		}
		by[x.Tenant] = append(by[x.Tenant], x)
	}
	r.DurationSeconds = float64(end-start) / 1e9
	for tenant, rows := range by {
		t := TenantReport{Arrivals: len(rows), States: map[string]int{}}
		var ttft, tpot, lag []float64
		for _, x := range rows {
			t.States[x.State]++
			t.OutputChunks += x.OutputChunks
			t.ObservedOutputTokens += x.ObservedOutputTokens
			if x.State == "completed" {
				t.Completed++
			}
			if x.CompletionTokens == nil {
				t.MissingTokenUsage++
			} else {
				t.CompletionTokens += *x.CompletionTokens
			}
			if x.SentUnixNS > 0 {
				lag = append(lag, float64(x.SentUnixNS-x.ScheduledUnixNS)/1e6)
			}
			if x.FirstOutputUnixNS > 0 {
				v := float64(x.FirstOutputUnixNS-x.SentUnixNS) / 1e6
				ttft = append(ttft, v)
				if x.DeadlineMS > 0 {
					t.SLOMeasured++
					if x.State == "completed" && v <= float64(x.DeadlineMS) {
						t.SLOMet++
					}
				}
				if x.State == "completed" && x.CompletionTokens != nil && *x.CompletionTokens > 1 {
					tpot = append(tpot, float64(x.LastOutputUnixNS-x.FirstOutputUnixNS)/1e6/float64(*x.CompletionTokens-1))
				}
			}
		}
		den := float64(t.Arrivals)
		t.SuccessRate = float64(t.Completed) / den
		t.RejectionRate = float64(t.States["rejected"]) / den
		t.TimeoutRate = float64(t.States["timeout"]) / den
		t.SLOSuccessFractionAllArrivals = float64(t.SLOMet) / den
		if r.DurationSeconds > 0 {
			t.SLOGoodputPerSecond = float64(t.SLOMet) / r.DurationSeconds
		}
		t.TTFT = distribution(ttft)
		t.MeanTPOT = distribution(tpot)
		t.DispatchLag = distribution(lag)
		r.Tenants[tenant] = t
	}
	return r
}
