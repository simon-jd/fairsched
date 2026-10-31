package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/simon-jd/fairsched/loadgen"
)

func main() {
	trace := flag.String("trace", "trace.jsonl", "JSONL arrival trace")
	url := flag.String("url", "http://127.0.0.1:8080", "proxy base URL")
	output := flag.String("output", "logs/client.jsonl", "results JSONL")
	report := flag.String("report", "logs/client-report.json", "summary JSON")
	timeout := flag.Duration("timeout", 2*time.Minute, "whole-request timeout")
	bounded := flag.Bool("bounded-records", false, "cap each event/terminal row at 64 KiB")
	flag.Parse()
	f, err := os.Open(*trace)
	if err != nil {
		log.Fatal(err)
	}
	arrivals, err := loadgen.ReadTrace(f)
	_ = f.Close()
	if err != nil {
		log.Fatal(err)
	}
	out, err := os.OpenFile(*output, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	results, err := loadgen.Run(ctx, arrivals, loadgen.Options{BaseURL: *url, Timeout: *timeout, Results: out, BoundedRecords: *bounded})
	if err != nil {
		log.Fatal(err)
	}
	b, err := json.MarshalIndent(loadgen.Summarize(results), "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err = os.WriteFile(*report, append(b, '\n'), 0600); err != nil {
		log.Fatal(err)
	}
	log.Printf("recorded %d arrivals in %s", len(results), *output)
}
