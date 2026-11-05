"""Seeded open-arrival traces for the Go load generator.

Prompts are random token-ID lists sent to ``/v1/completions``, so prompt length
is exact, no tokenizer is involved, and requests never share a cached prefix.
``ignore_eos`` makes every request generate exactly ``max_tokens`` tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    weights: dict[str, float]
    duration_s: float


WORKLOADS = {
    "smoke": Workload(
        "smoke", "10 s sanity check: 20 short requests per tenant.", {"A": 1.0, "B": 1.0}, 10.0),
    "noisy_neighbor": Workload(
        "noisy_neighbor",
        "A: interactive, Poisson 2 req/s, 256-token prompts, 128 output tokens, for 120 s. "
        "B: at t=10 s dumps 300 requests with 2048-token prompts and 256 output tokens; "
        "the client cancels 20% of B's requests after 2 output chunks.",
        {"A": 1.0, "B": 1.0}, 120.0),
    "aggressive": Workload(
        "aggressive",
        "Both tenants backlogged from t=0 with 1024-token prompts and 256 output tokens. "
        "A sends 100 requests, B sends 300 (3x the demand), equal weights.",
        {"A": 1.0, "B": 1.0}, 0.0),
    "weighted": Workload(
        "weighted",
        "Both tenants backlogged from t=0 with 200 requests each, 1024-token prompts and "
        "256 output tokens. A has weight 2, B weight 1.",
        {"A": 2.0, "B": 1.0}, 0.0),
    "mixed_shapes": Workload(
        "mixed_shapes",
        "Both tenants backlogged from t=0 with different shapes and equal weights: A sends 200 "
        "requests with 2048-token prompts and 32 output tokens (prefill-heavy), B 200 requests "
        "with 128-token prompts and 512 output tokens (decode-heavy).",
        {"A": 1.0, "B": 1.0}, 0.0),
    "long_prompt": Workload(
        "long_prompt",
        "B: 6 requests with 6144-token prompts and 128 output tokens at t=0 (the shape that "
        "livelocked the old quantum preemption). A: Poisson 1 req/s short requests for 60 s. "
        "Run with a 520-block (8,320-token) KV cache, about the size of that run, so running requests contend.",
        {"A": 1.0, "B": 1.0}, 60.0),
}


def _row(rng: random.Random, model: str, rid: str, at_s: float, tenant: str, prompt: int, output: int,
         **extra) -> dict:
    body = {"model": model, "prompt": [rng.randrange(1000, 100_000) for _ in range(prompt)],
            "max_tokens": output, "ignore_eos": True, "temperature": 0.0, "stream": True}
    return {"id": rid, "at_ms": int(at_s * 1000), "tenant": tenant, "api_key": f"key-{tenant}",
            "path": "/v1/completions", "timeout_ms": 900_000, "body": body, **extra}


def _poisson(rng: random.Random, rate: float, start: float, end: float):
    t = start
    while True:
        t += rng.expovariate(rate)
        if t >= end:
            return
        yield t


def generate(name: str, seed: int, model: str = MODEL) -> list[dict]:
    """The trace's rows; ``model`` is the served model name in each body."""
    rng = random.Random(seed)
    rows: list[dict] = []
    if name == "smoke":
        for i in range(20):
            rows.append(_row(rng, model, f"A-{i}", i * 0.5, "A", 128, 32))
            rows.append(_row(rng, model, f"B-{i}", i * 0.05, "B", 512, 64, **(
                {"cancel_after_output_chunks": 2} if i % 5 == 0 else {})))
    elif name == "noisy_neighbor":
        for i, t in enumerate(_poisson(rng, 2.0, 0.0, 120.0)):
            rows.append(_row(rng, model, f"A-{i}", t, "A", 256, 128))
        for i in range(300):
            extra = {"cancel_after_output_chunks": 2} if rng.random() < 0.2 else {}
            rows.append(_row(rng, model, f"B-{i}", 10.0 + i * 0.001, "B", 2048, 256, **extra))
    elif name == "aggressive":
        for i in range(100):
            rows.append(_row(rng, model, f"A-{i}", i * 0.003, "A", 1024, 256))
        for i in range(300):
            rows.append(_row(rng, model, f"B-{i}", i * 0.001, "B", 1024, 256))
    elif name == "weighted":
        for i in range(200):
            rows.append(_row(rng, model, f"A-{i}", i * 0.002, "A", 1024, 256))
            rows.append(_row(rng, model, f"B-{i}", i * 0.002 + 0.001, "B", 1024, 256))
    elif name == "mixed_shapes":
        for i in range(200):
            rows.append(_row(rng, model, f"A-{i}", i * 0.002, "A", 2048, 32))
            rows.append(_row(rng, model, f"B-{i}", i * 0.002 + 0.001, "B", 128, 512))
    elif name == "long_prompt":
        for i in range(6):
            rows.append(_row(rng, model, f"B-{i}", i * 0.01, "B", 6144, 128))
        for i, t in enumerate(_poisson(rng, 1.0, 0.5, 60.0)):
            rows.append(_row(rng, model, f"A-{i}", t, "A", 128, 64))
    else:
        raise KeyError(name)
    rows.sort(key=lambda r: (r["at_ms"], r["id"]))
    return rows


def session_hash(session: str) -> int:
    """FNV-1a 64, the gateway's sticky-routing hash (proxy.SessionHash)."""
    h = 0xcbf29ce484222325
    for byte in session.encode():
        h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def session_names(tenant: str, count: int, replicas: int, pinned: int | None = None) -> list[str]:
    """``count`` session IDs for a tenant that sticky routing sends all to
    replica ``pinned``, or, with ``pinned`` None, evenly over the replicas."""
    names: list[str] = []
    per = [0] * replicas
    k = 0
    while len(names) < count:
        name = f"{tenant}-s{k}"
        k += 1
        replica = session_hash(name) % replicas
        if (replica == pinned) if pinned is not None else per[replica] < -(-count // replicas):
            names.append(name)
            per[replica] += 1
    return names


def assign_sessions(rows: list[dict], replicas: int, per_tenant: int = 8,
                    pinned: dict[str, int] | None = None) -> list[dict]:
    """Give each row a session; a tenant's i-th request uses its (i mod per_tenant)-th session.

    Tenants named in ``pinned`` get sessions that all hash to that replica
    (the case where one tenant's traffic has affinity to one replica); the
    others' sessions are spread evenly. Only sticky routing reads sessions.
    """
    pinned = pinned or {}
    names: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    for row in rows:
        tenant = row["tenant"]
        if tenant not in names:
            names[tenant] = session_names(tenant, per_tenant, replicas, pinned.get(tenant))
        i = seen.get(tenant, 0)
        seen[tenant] = i + 1
        row["session"] = names[tenant][i % per_tenant]
    return rows


def write(name: str, seed: int, path: Path, model: str = MODEL, replicas: int = 1,
          pinned: dict[str, int] | None = None) -> Path:
    """Write the trace; with ``replicas`` > 1 each row also gets a session (``assign_sessions``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(name, seed, model)
    if replicas > 1:
        assign_sessions(rows, replicas, pinned=pinned)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path
