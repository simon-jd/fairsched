package main

import (
	"context"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/simon-jd/fairsched/proxy"
)

func main() {
	path := flag.String("config", "proxy/config.example.json", "JSON configuration")
	flag.Parse()
	cfg, err := proxy.LoadConfig(*path)
	if err != nil {
		log.Fatal(err)
	}
	if cfg.Listen == "" {
		cfg.Listen = "127.0.0.1:8080"
	}
	var output io.Writer = os.Stdout
	if cfg.LogPath != "" {
		f, err := os.OpenFile(cfg.LogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		output = f
	}
	h, err := proxy.New(cfg, nil, output)
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{Addr: cfg.Listen, Handler: h, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, stopServer := context.WithCancelCause(context.Background())
	defer stopServer(nil)
	server.BaseContext = func(_ net.Listener) context.Context { return ctx }
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		<-signalCtx.Done()
		stopServer(proxy.ErrServerShutdown)
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if server.Shutdown(shutdown) != nil {
			_ = server.Close()
		}
	}()
	log.Printf("fairsched proxy listening on %s", cfg.Listen)
	if err = server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
	<-shutdownDone
	h.Wait()
}
