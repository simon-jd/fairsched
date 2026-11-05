"""Counters shared across engine replicas: a small HTTP service and the engine-side client.

Off by default. With ``FAIRSCHED_COUNTER_URL`` set, each engine's scheduler
starts a :class:`CounterSync` thread that every ``FAIRSCHED_SYNC_MS``
milliseconds (default 100) posts the engine's cumulative weighted service per
tenant and reads back the sum of the other replicas' totals. The scheduler
adds those totals to its own counters for ordering, lift and victim choice.

Replicas publish cumulative totals, not increments, so a repeated or lost post
changes nothing and a restarted service is whole again after one round. Lifts
stay local: only billed service is published.

If the service cannot be reached, the engine keeps scheduling on its local
counters, with the other replicas' totals held at their last known values, and
logs the change; it keeps retrying and logs when the service answers again.

    python -m fairsched.counters --listen 127.0.0.1:8090
"""
from __future__ import annotations

import argparse
from collections import deque
from http.client import HTTPConnection, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import socket
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

PATH = "/v1/counters"
MAX_BODY = 64 * 1024
MAX_NAME = 128
MAX_TENANTS = 1024
MAX_REPLICAS = 256


def _name(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_NAME and value.isprintable()


def parse_publish(body: bytes) -> tuple[str, dict[str, float]]:
    """Validate one post: ``{"replica": name, "served": {tenant: finite nonnegative number}}``."""
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("body must be an object")
    replica, served = data.get("replica"), data.get("served")
    if not _name(replica):
        raise ValueError("replica must be a printable string of 1-128 characters")
    if not isinstance(served, dict) or len(served) > MAX_TENANTS:
        raise ValueError(f"served must be an object of at most {MAX_TENANTS} tenants")
    out = {}
    for tenant, value in served.items():
        if not _name(tenant) or isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError("served maps tenant names to finite nonnegative numbers")
        out[tenant] = float(value)
    return replica, out


class CounterStore:
    """Latest cumulative totals per replica; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.replicas: dict[str, dict[str, float]] = {}
        self.posts = 0

    def publish(self, replica: str, served: dict[str, float]) -> dict[str, Any]:
        with self._lock:
            if replica not in self.replicas and len(self.replicas) >= MAX_REPLICAS:
                raise ValueError("too many replicas")
            self.replicas[replica] = served
            self.posts += 1
            remote: dict[str, float] = {}
            for name, totals in self.replicas.items():
                if name != replica:
                    for tenant, value in totals.items():
                        remote[tenant] = remote.get(tenant, 0.0) + value
            return {"remote": remote, "replicas": len(self.replicas)}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"replicas": {k: dict(v) for k, v in self.replicas.items()}, "posts": self.posts}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive, so a sync is one round trip on an open socket
    store: CounterStore
    open_sockets: set

    def setup(self) -> None:
        super().setup()
        self.open_sockets.add(self.connection)

    def finish(self) -> None:
        self.open_sockets.discard(self.connection)
        super().finish()

    def _reply(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._reply(200, {"ok": True})
        elif self.path == PATH:
            self._reply(200, self.store.snapshot())
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != PATH:
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            self._reply(413 if length > MAX_BODY else 411, {"error": "body must have a length of at most 64 KiB"})
            return
        try:
            replica, served = parse_publish(self.rfile.read(length))
            reply = self.store.publish(replica, served)
        except ValueError as e:      # json.JSONDecodeError is a ValueError
            self._reply(400, {"error": str(e)[:200]})
            return
        self._reply(200, reply)

    def log_message(self, *args: Any) -> None:
        pass


class CounterService:
    """The service on a background thread (tests) or in the foreground (``main``)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.store = CounterStore()
        self.open_sockets: set = set()
        handler = type("Handler", (_Handler,), {"store": self.store, "open_sockets": self.open_sockets})
        self.server = ThreadingHTTPServer((host, port), handler)
        self.server.daemon_threads = True
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "CounterService":
        self.thread = threading.Thread(target=self.server.serve_forever, name="fairsched-counters", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and drop open keep-alive connections, as a killed process would."""
        self.server.shutdown()
        self.server.server_close()
        for sock in list(self.open_sockets):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class CounterSync:
    """Engine side: a thread that publishes this engine's totals and fetches the others'.

    The scheduler thread calls :meth:`exchange` once per step with a copy of
    its served totals and gets back new remote totals, or None when nothing
    new has arrived; it never waits on the network. :meth:`events` returns the
    reachability changes since the last call, for the step log.
    """

    def __init__(self, url: str, replica: str, *, period_s: float = 0.1, timeout_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname or parts.path not in ("", "/"):
            raise ValueError("counter URL must be http://host:port")
        if not _name(replica):
            raise ValueError("replica must be a printable string of 1-128 characters")
        if not (isinstance(period_s, (int, float)) and math.isfinite(period_s) and period_s > 0):
            raise ValueError("sync period must be positive")
        self.host, self.port = parts.hostname, parts.port or 80
        self.replica, self.period_s = replica, float(period_s)
        # A sync must not outlive its period by much, or a hung service delays
        # the fallback; bounded below so a fast period still gets a fair chance.
        self.timeout_s = timeout_s if timeout_s is not None else min(1.0, max(0.05, 2 * self.period_s))
        self.clock = clock
        self._served: dict[str, float] = {}
        self._latest: tuple[int, dict[str, float]] | None = None
        self._applied = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=1024)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: HTTPConnection | None = None
        self.reachable: bool | None = None       # None until the first attempt
        self.ok = self.failed = 0
        self.rtt_us: deque[float] = deque(maxlen=4096)   # recent round trips, for the step log

    # Scheduler thread -------------------------------------------------------

    def exchange(self, served: dict[str, float]) -> dict[str, float] | None:
        self._served = served                    # one reference assignment; the thread only reads it
        latest = self._latest
        if latest is None or latest[0] == self._applied:
            return None
        self._applied = latest[0]
        return latest[1]

    def events(self) -> list[dict[str, Any]]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    # Sync thread ------------------------------------------------------------

    def start(self) -> "CounterSync":
        self._thread = threading.Thread(target=self._run, name="fairsched-counter-sync", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._close()

    def _run(self) -> None:
        next_at = self.clock()
        while not self._stop.is_set():
            self.sync_once()
            next_at += self.period_s
            delay = next_at - self.clock()
            if delay < 0:                        # fell behind (slow or failing service): no catch-up burst
                next_at, delay = self.clock(), 0.0
            self._stop.wait(delay)

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _post(self, body: str) -> dict[str, float]:
        if self._conn is None:
            self._conn = HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        self._conn.request("POST", PATH, body=body, headers={"Content-Type": "application/json"})
        response = self._conn.getresponse()
        payload = response.read()
        if response.status != 200:
            raise OSError(f"HTTP {response.status}")
        remote = json.loads(payload)["remote"]
        if not isinstance(remote, dict):
            raise ValueError("remote must be an object")
        return {str(t): float(v) for t, v in remote.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0}

    def sync_once(self) -> bool:
        """One post; True when the service answered with remote totals."""
        body = json.dumps({"replica": self.replica, "served": self._served}, separators=(",", ":"))
        started = time.perf_counter()
        try:
            reused = self._conn is not None
            try:
                remote = self._post(body)
            except (ConnectionError, RemoteDisconnected):
                if not reused:
                    raise
                # The service closed an idle keep-alive socket (it restarted, say): retry once on a new one.
                self._close()
                remote = self._post(body)
        except Exception as e:                   # noqa: BLE001 - any failure means local counters
            self._close()
            self.failed += 1
            if self.reachable is not False:
                self._events.append({"type": "counter_sync", "state": "unreachable", "t": time.time(),
                                     "error": repr(e)[:200]})
            self.reachable = False
            return False
        self.rtt_us.append((time.perf_counter() - started) * 1e6)
        self.ok += 1
        if self.reachable is False:
            self._events.append({"type": "counter_sync", "state": "restored", "t": time.time()})
        self.reachable = True
        seq = (self._latest[0] if self._latest else 0) + 1
        self._latest = (seq, remote)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen", default="127.0.0.1:8090", help="host:port")
    args = parser.parse_args()
    host, _, port = args.listen.rpartition(":")
    service = CounterService(host or "127.0.0.1", int(port))
    print(f"fairsched counters listening on {service.url}", flush=True)
    try:
        service.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server.server_close()


if __name__ == "__main__":
    main()
