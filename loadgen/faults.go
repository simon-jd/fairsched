package loadgen

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"time"
)

type slowReader struct {
	ctx    context.Context
	reader io.Reader
	delay  time.Duration
	chunk  int
}

func (r slowReader) Read(p []byte) (int, error) {
	timer := time.NewTimer(r.delay)
	defer timer.Stop()
	select {
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	case <-timer.C:
	}
	if r.chunk > 0 && len(p) > r.chunk {
		p = p[:r.chunk]
	}
	return r.reader.Read(p)
}

type halfCloseResult struct {
	at  int64
	err string
}

// Half-close is an HTTP/1.1, cleartext TCP fault. It closes only the write half
// after the complete request has been sent, and keeps reading the response.
func halfCloseClient(req *http.Request, base *http.Client, delay time.Duration) (*http.Client, func() halfCloseResult, error) {
	if req.URL.Scheme != "http" {
		return nil, nil, errors.New("half_close_requires_cleartext_http1")
	}
	var transport *http.Transport
	if base.Transport == nil {
		transport = http.DefaultTransport.(*http.Transport).Clone()
	} else if t, ok := base.Transport.(*http.Transport); ok {
		transport = t.Clone()
	} else {
		return nil, nil, errors.New("half_close_requires_http_transport")
	}
	transport.DisableKeepAlives = true
	transport.ForceAttemptHTTP2 = false
	transport.Proxy = nil
	ctx, cancel := context.WithCancel(req.Context())
	connection := make(chan net.Conn, 1)
	written := make(chan struct{}, 1)
	dial := transport.DialContext
	if dial == nil {
		dial = (&net.Dialer{}).DialContext
	}
	transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := dial(ctx, network, addr)
		if err == nil {
			select {
			case connection <- conn:
			default:
			}
		}
		return conn, err
	}
	trace := &httptrace.ClientTrace{WroteRequest: func(info httptrace.WroteRequestInfo) {
		if info.Err == nil {
			select {
			case written <- struct{}{}:
			default:
			}
		}
	}}
	*req = *req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	client := *base
	client.Transport = transport
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	done := make(chan struct{})
	var result halfCloseResult
	go func() {
		defer close(done)
		select {
		case <-ctx.Done():
			return
		case <-written:
		}
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		var conn net.Conn
		select {
		case <-ctx.Done():
			return
		case conn = <-connection:
		}
		writer, ok := conn.(interface{ CloseWrite() error })
		if !ok {
			result.err = "connection_does_not_support_half_close"
			return
		}
		if err := writer.CloseWrite(); err != nil {
			result.err = err.Error()
			return
		}
		result.at = time.Now().UnixNano()
	}()
	finish := func() halfCloseResult { cancel(); <-done; transport.CloseIdleConnections(); return result }
	return &client, finish, nil
}
