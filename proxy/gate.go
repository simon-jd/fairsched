package proxy

import (
	"context"
	"errors"
	"sync"
)

var errOverloaded = errors.New("tenant queue full")

type waiter struct {
	ready    chan struct{}
	admitted bool
}
type gate struct {
	mu                    sync.Mutex
	active, max, capacity int
	queue                 []*waiter
}

func (g *gate) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	g.mu.Lock()
	if g.active < g.max {
		g.active++
		g.mu.Unlock()
		return nil
	}
	if len(g.queue) >= g.capacity {
		g.mu.Unlock()
		return errOverloaded
	}
	w := &waiter{ready: make(chan struct{})}
	g.queue = append(g.queue, w)
	g.mu.Unlock()
	select {
	case <-w.ready:
		return nil
	case <-ctx.Done():
		g.mu.Lock()
		if w.admitted {
			g.releaseLocked()
		} else {
			for i, v := range g.queue {
				if v == w {
					g.queue = append(g.queue[:i], g.queue[i+1:]...)
					break
				}
			}
		}
		g.mu.Unlock()
		return ctx.Err()
	}
}
func (g *gate) releaseLocked() {
	if len(g.queue) > 0 {
		w := g.queue[0]
		g.queue = g.queue[1:]
		w.admitted = true
		close(w.ready)
	} else {
		g.active--
	}
}
func (g *gate) release() { g.mu.Lock(); defer g.mu.Unlock(); g.releaseLocked() }
