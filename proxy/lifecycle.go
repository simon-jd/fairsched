package proxy

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"
)

var errDownstreamWrite = errors.New("downstream write failed")
var errDownstreamWriteTimeout = errors.New("downstream write timeout")
var errUpstreamFailure = errors.New("upstream stream processing failed")

func downstreamCancelCause(err error) error {
	if errors.Is(err, os.ErrDeadlineExceeded) {
		return errDownstreamWriteTimeout
	}
	return errDownstreamWrite
}

func downstreamTriggerReason(err error) string {
	if errors.Is(err, os.ErrDeadlineExceeded) {
		return "downstream_write_timeout"
	}
	return "downstream_write_error"
}

// LocalLifecycleRecord records observations made by this proxy process. In
// particular, Body.Close returning is not a remote abort or GPU acknowledgement.
type LocalLifecycleRecord struct {
	Event          string `json:"event"`
	RID            string `json:"rid"`
	ClientID       string `json:"client_id,omitempty"`
	Tenant         string `json:"tenant,omitempty"`
	ObservedUnixMS int64  `json:"observed_unix_ms"`
	Reason         string `json:"reason,omitempty"`
	Error          string `json:"error,omitempty"`
}

type localLifecycle struct {
	mu            sync.Mutex
	h             *Handler
	base          LocalLifecycleRecord
	cancelAt      int64
	cancelReason  string
	closeAt       int64
	triggerAt     int64
	triggerReason string
}

func newLocalLifecycle(h *Handler, rec Terminal) *localLifecycle {
	return &localLifecycle{h: h, base: LocalLifecycleRecord{RID: rec.RID, ClientID: rec.ClientID, Tenant: rec.Tenant}}
}

func (l *localLifecycle) observeCancellation(cause error) {
	if cause == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cancelAt != 0 {
		return
	}
	l.cancelAt = time.Now().UnixMilli()
	switch {
	case errors.Is(cause, errDeadline):
		l.cancelReason = "ttft_deadline_exceeded"
	case errors.Is(cause, ErrServerShutdown):
		l.cancelReason = "proxy_shutdown"
	case errors.Is(cause, errDownstreamWriteTimeout):
		l.cancelReason = "downstream_write_timeout"
	case errors.Is(cause, errDownstreamWrite):
		l.cancelReason = "downstream_write_error"
	case errors.Is(cause, context.Canceled):
		// net/http also cancels this context after local socket read/write
		// failures. The context alone does not prove a peer disconnect.
		l.cancelReason = "request_context_cancelled"
	case errors.Is(cause, context.DeadlineExceeded):
		l.cancelReason = "request_context_deadline_exceeded"
	case errors.Is(cause, errUpstreamFailure):
		l.cancelReason = "upstream_failure"
	default:
		l.cancelReason = "request_context_cancelled_other"
	}
	e := l.base
	e.Event, e.ObservedUnixMS, e.Reason = "request_context_cancel_observed", l.cancelAt, l.cancelReason
	l.h.writeLog(e)
}

func (l *localLifecycle) trigger(reason string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.triggerAt != 0 {
		return
	}
	l.triggerAt, l.triggerReason = time.Now().UnixMilli(), reason
	e := l.base
	e.Event, e.ObservedUnixMS, e.Reason = "request_cancel_trigger_observed", l.triggerAt, reason
	l.h.writeLog(e)
}

func (l *localLifecycle) bodyCloseCompleted(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closeAt = time.Now().UnixMilli()
	e := l.base
	e.Event, e.ObservedUnixMS = "upstream_body_close_completed", l.closeAt
	if err != nil {
		e.Error = err.Error()
	}
	l.h.writeLog(e)
}

func (l *localLifecycle) copyTo(rec *Terminal) {
	l.mu.Lock()
	defer l.mu.Unlock()
	rec.ContextCancelObservedUnixMS = l.cancelAt
	rec.ContextCancelReason = l.cancelReason
	rec.UpstreamBodyCloseCompletedUnixMS = l.closeAt
	rec.CancelTriggerObservedUnixMS = l.triggerAt
	rec.CancelTriggerReason = l.triggerReason
}
