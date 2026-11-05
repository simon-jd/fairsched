package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// replicas starts n fake engines that count the requests they receive.
func replicas(t *testing.T, n int) ([]*httptest.Server, []string, func() []int) {
	t.Helper()
	var mu sync.Mutex
	counts := make([]int, n)
	servers := make([]*httptest.Server, n)
	urls := make([]string, n)
	for i := range servers {
		i := i
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("X-Fairsched-Session-ID") != "" {
				t.Error("session header forwarded to the engine")
			}
			mu.Lock()
			counts[i]++
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			emit(w, outputChunk)
			emit(w, finishChunk)
			emit(w, "[DONE]")
		}))
		t.Cleanup(servers[i].Close)
		urls[i] = servers[i].URL
	}
	return servers, urls, func() []int {
		mu.Lock()
		defer mu.Unlock()
		return append([]int(nil), counts...)
	}
}

func twoTenantConfig(urls []string, routing string) Config {
	policy := TenantPolicy{Class: "batch", MaxInflight: 4, MaxQueued: 4}
	return Config{Upstreams: urls, Routing: routing,
		APIKeys: map[string]string{"key-A": "A", "key-B": "B"},
		Tenants: map[string]TenantPolicy{"A": policy, "B": policy}}
}

func send(t *testing.T, url, key, session string) {
	t.Helper()
	r, _ := http.NewRequest("POST", url+"/v1/completions", strings.NewReader(`{"prompt":"x","max_tokens":1}`))
	r.Header.Set("Authorization", "Bearer "+key)
	if session != "" {
		r.Header.Set("X-Fairsched-Session-ID", session)
		r.Header.Set("X-Fairsched-Client-ID", session)
	}
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestRoundRobinAlternatesEachTenant(t *testing.T) {
	_, urls, counts := replicas(t, 2)
	var logs bytes.Buffer
	h, err := New(twoTenantConfig(urls, RoutingRoundRobin), nil, &logs)
	if err != nil {
		t.Fatal(err)
	}
	gw := httptest.NewServer(h)
	defer gw.Close()
	// A, B, B, B repeated: global round-robin would send every A to one replica.
	for i := 0; i < 4; i++ {
		send(t, gw.URL, "key-A", "")
		send(t, gw.URL, "key-B", "")
		send(t, gw.URL, "key-B", "")
		send(t, gw.URL, "key-B", "")
	}
	h.Wait()
	perTenant := map[string]map[string]int{}
	for _, rec := range terminals(t, &logs) {
		if perTenant[rec.Tenant] == nil {
			perTenant[rec.Tenant] = map[string]int{}
		}
		perTenant[rec.Tenant][rec.Upstream]++
	}
	if perTenant["A"][urls[0]] != 2 || perTenant["A"][urls[1]] != 2 || perTenant["B"][urls[0]] != 6 || perTenant["B"][urls[1]] != 6 {
		t.Fatalf("uneven split: %v", perTenant)
	}
	if c := counts(); c[0] != 8 || c[1] != 8 {
		t.Fatalf("engine counts %v", c)
	}
}

func TestStickyRoutingFollowsTheSessionHash(t *testing.T) {
	_, urls, counts := replicas(t, 2)
	var logs bytes.Buffer
	h, err := New(twoTenantConfig(urls, RoutingSticky), nil, &logs)
	if err != nil {
		t.Fatal(err)
	}
	gw := httptest.NewServer(h)
	defer gw.Close()
	sessions := []string{"A-s0", "A-s1", "B-s0", "B-s1", "B-s2"}
	for round := 0; round < 3; round++ {
		for _, s := range sessions {
			send(t, gw.URL, "key-"+s[:1], s)
		}
	}
	send(t, gw.URL, "key-A", "") // no session: round-robin
	h.Wait()
	rows := terminals(t, &logs)
	if len(rows) != 16 {
		t.Fatalf("%d terminal records", len(rows))
	}
	for _, rec := range rows {
		if rec.ClientID == "" {
			if rec.Upstream == "" {
				t.Fatal("terminal record without upstream")
			}
			continue
		}
		if want := urls[SessionHash(rec.ClientID)%2]; rec.Upstream != want {
			t.Fatalf("session %s went to %s, want %s", rec.ClientID, rec.Upstream, want)
		}
	}
	if c := counts(); c[0]+c[1] != 16 {
		t.Fatalf("engine counts %v", c)
	}
}

func TestSessionHashIsFNV1a64(t *testing.T) {
	// Reference values of FNV-1a 64; bench/workloads.py must agree (tests/test_replicas.py).
	if SessionHash("") != 0xcbf29ce484222325 || SessionHash("a") != 0xaf63dc4c8601ec8c {
		t.Fatalf("hash %x %x", SessionHash(""), SessionHash("a"))
	}
}

func TestUpstreamConfigValidation(t *testing.T) {
	base := twoTenantConfig([]string{"http://127.0.0.1:1", "http://127.0.0.1:2"}, RoutingSticky)
	if err := base.Validate(); err != nil {
		t.Fatal(err)
	}
	both := base
	both.Upstream = "http://127.0.0.1:3"
	if both.Validate() == nil {
		t.Fatal("upstream and upstreams together accepted")
	}
	none := base
	none.Upstreams = nil
	if none.Validate() == nil {
		t.Fatal("no upstream accepted")
	}
	bad := base
	bad.Upstreams = []string{"http://127.0.0.1:1", "ftp://x"}
	if bad.Validate() == nil {
		t.Fatal("bad upstream URL accepted")
	}
	routing := base
	routing.Routing = "random"
	if routing.Validate() == nil {
		t.Fatal("unknown routing accepted")
	}
	if got := config("http://127.0.0.1:9").UpstreamURLs(); len(got) != 1 || got[0] != "http://127.0.0.1:9" {
		t.Fatalf("single upstream %v", got)
	}
}
