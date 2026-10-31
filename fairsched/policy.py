"""Pure-Python VTC policy: per-tenant virtual token counters and cost accounts.

Counters decide scheduling order (Virtual Token Counter, Sheng et al., OSDI 2024).
Accounts record who paid for executed tokens. The two diverge on purpose. When a
cancelled request's in-flight step is reclassified as wasted, cost moves between
accounts but the counter that already shaped earlier decisions is not rewritten.
When KV pressure forces a recompute, the counter is not charged by default, so
the victim tenant is not penalised twice for a preemption it did not choose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

SYSTEM = "system"
DEFAULT = "default"


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) \
            or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class TenantPolicy:
    weight: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.weight, "weight")


@dataclass(frozen=True)
class PolicyConfig:
    """Tenant weights and the VTC cost model.

    ``w_p``/``w_d`` weight prefill and decode tokens (the VTC paper uses 1 and 2).
    ``delta`` widens the "equally under-served" window when picking a tenant.
    ``charge_recompute`` and ``charge_wasted`` name who pays for tokens recomputed
    after preemption and for tokens computed for an already-cancelled request.
    ``preempt_order`` decides which running request vLLM preempts under KV
    pressure (see ``PolicyState.order_running``): ``"counter"`` (default) the
    newest request of the tenant with the highest counter, ``"kv_share"`` the
    newest request of the highest-counter tenant among those holding more than
    their weighted share of the running requests' KV.
    """
    tenants: dict[str, TenantPolicy] = field(default_factory=lambda: {"default": TenantPolicy()})
    w_p: float = 1.0
    w_d: float = 2.0
    delta: float = 0.0
    charge_recompute: str = SYSTEM
    charge_wasted: str = "tenant"
    preempt_order: str = "counter"

    def __post_init__(self) -> None:
        if not self.tenants or any(not isinstance(k, str) or not k or k == SYSTEM for k in self.tenants):
            raise ValueError("tenants must be nonempty; 'system' is reserved")
        if any(not isinstance(v, TenantPolicy) for v in self.tenants.values()):
            raise ValueError("tenant values must be TenantPolicy instances")
        _positive(self.w_p, "w_p")
        _positive(self.w_d, "w_d")
        _positive(self.delta, "delta", zero=True)
        for name in ("charge_recompute", "charge_wasted"):
            if getattr(self, name) not in {"tenant", SYSTEM}:
                raise ValueError(f"{name} must be 'tenant' or 'system'")
        if self.preempt_order not in {"counter", "kv_share"}:
            raise ValueError("preempt_order must be 'counter' or 'kv_share'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyConfig":
        data = dict(value)
        data["tenants"] = {name: TenantPolicy(**settings)
                           for name, settings in data.get("tenants", {"default": {}}).items()}
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("policy root must be an object")
        return cls.from_dict(data)


def request_metadata(request: Any, config: PolicyConfig) -> dict[str, Any]:
    """Tenant identity comes only from trusted ``vllm_xargs`` set by the gateway.

    A request without a tenant, or naming one the policy does not list, is
    served as the ``default`` tenant (weight 1 unless the policy lists it), and
    an unlisted name is kept as ``claimed_tenant``. Raising instead would fail
    the engine's ``add_request``. Mapping every unlisted name to one tenant
    means a client that reaches the engine without the gateway cannot gain
    share by inventing tenant names.
    """
    extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
    claimed = extra.get("fairsched_tenant")
    tenant = claimed if isinstance(claimed, str) and claimed in config.tenants else DEFAULT
    request_id = str(request.request_id)
    meta = {"request_id": request_id, "rid": str(extra.get("fairsched_rid", request_id)), "tenant": tenant,
            "arrival_time": float(getattr(request, "arrival_time", 0.0))}
    if claimed is not None and claimed != tenant:
        meta["claimed_tenant"] = str(claimed)
    return meta


class PolicyState:
    """Counters, accounts and the per-step charge ledger for one engine."""

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()
        self.counters = {tenant: 0.0 for tenant in self.config.tenants}
        self.accounts = {**self.counters, SYSTEM: 0.0}
        self.wasted = {tenant: 0.0 for tenant in self.config.tenants}   # cumulative cost
        self.requests: dict[str, dict[str, Any]] = {}
        # Requests per tenant in the waiting queue. VTC's lift rule is defined on
        # the queue, not on unfinished work: a tenant with one long request
        # still decoding is not "backlogged" and must not keep a stale counter.
        self.waiting = {tenant: 0 for tenant in self.config.tenants}
        # Prefill cost of requests admitted earlier in the same step, not yet charged.
        self.reservations: dict[str, float] = {}
        # step -> charge records, kept until the async output for that step lands.
        self.ledger: dict[int, list[dict[str, Any]]] = {}
        self.step_seq = 0
        # The paper's l: counter of the last tenant whose queue emptied, the
        # lift target when nobody is waiting.
        self.last_left = 0.0
        # Requests that named a tenant the policy does not list (served as DEFAULT).
        self.unknown_requests = 0

    def weight(self, tenant: str) -> float:
        policy = self.config.tenants.get(tenant)
        return policy.weight if policy is not None else TenantPolicy().weight

    def _add_tenant(self, tenant: str) -> None:
        """Start tracking DEFAULT when the policy does not list it but a request needs it."""
        self.counters[tenant] = 0.0
        self.accounts[tenant] = 0.0
        self.wasted[tenant] = 0.0
        self.waiting[tenant] = 0

    @property
    def effective_counters(self) -> dict[str, float]:
        result = self.counters.copy()
        for request_id, reservation in self.reservations.items():
            result[self.requests[request_id]["tenant"]] += reservation
        return result

    def register(self, request: Any) -> dict[str, Any]:
        """Record a new request; apply the VTC lift if its tenant has nothing waiting.

        Called when the request arrives, before it is enqueued. A tenant that
        joins the queue starts no lower than the least-served waiting tenant
        (or ``last_left`` when the queue is empty), so time spent idle or with
        only running work cannot be banked as credit.
        """
        request_id = str(request.request_id)
        meta = self.requests.get(request_id)
        if meta is not None:
            return meta
        meta = request_metadata(request, self.config)
        tenant = meta["tenant"]
        if tenant not in self.counters:
            self._add_tenant(tenant)
        if "claimed_tenant" in meta:
            self.unknown_requests += 1
        if not self.waiting[tenant]:
            queued = [self.counters[t] for t, n in self.waiting.items() if n]
            self.counters[tenant] = max(self.counters[tenant], min(queued) if queued else self.last_left)
        self.requests[request_id] = meta
        return meta

    def enqueued(self, tenant: str) -> None:
        self.waiting[tenant] += 1

    def dequeued(self, tenant: str) -> None:
        self.waiting[tenant] -= 1
        if self.waiting[tenant] == 0:
            # Includes the reserved prompt of a request just admitted, matching
            # VTC charging a request's input tokens when it is admitted.
            self.last_left = self.effective_counters[tenant]

    def finish(self, request_id: str) -> None:
        self.reservations.pop(str(request_id), None)

    def forget(self, request_id: str) -> None:
        """Drop metadata once the request is finished and no ledger entry needs it."""
        self.requests.pop(str(request_id), None)

    def begin_step(self, step_seq: int) -> None:
        self.step_seq = step_seq
        self.reservations.clear()
        self.ledger.setdefault(step_seq, [])

    def settle_step(self) -> None:
        self.reservations.clear()

    def release_step(self, step_seq: int) -> None:
        self.ledger.pop(step_seq, None)

    def reserve(self, request: Any) -> None:
        """Project an admitted request's prefill so the next pop sees its cost."""
        meta = self.register(request)
        remaining = max(0, int(request.num_prompt_tokens) - int(request.num_computed_tokens))
        self.reservations[meta["request_id"]] = remaining * self.config.w_p / self.weight(meta["tenant"])

    def rollback(self, request_id: str) -> None:
        self.reservations.pop(str(request_id), None)

    def charge(self, request_id: str, prefill: int, decode: int, *,
               recompute_prefill: int = 0, recompute_decode: int = 0) -> dict[str, Any]:
        """Bill one step's scheduled tokens for one request.

        ``prefill``/``decode`` split the scheduled positions at the prompt
        boundary. ``recompute_*`` count the subset that re-executes positions
        this request already computed before a preemption. Recomputed output
        positions run as part of a prefill chunk, so they are priced at ``w_p``.
        """
        # Called for every scheduled request on every step, so it stays lean.
        if not (0 <= recompute_prefill <= prefill and 0 <= recompute_decode <= decode):
            raise ValueError("token counts must be nonnegative and recompute cannot exceed scheduled tokens")
        tenant, c = self.requests[request_id]["tenant"], self.config
        cost = (prefill + recompute_decode) * c.w_p + (decode - recompute_decode) * c.w_d
        system = (recompute_prefill + recompute_decode) * c.w_p if c.charge_recompute == SYSTEM else 0.0
        billed = cost - system
        self.accounts[tenant] += billed
        if system:
            self.accounts[SYSTEM] += system
        delta = billed / self.weight(tenant)
        self.counters[tenant] += delta
        if self.reservations:
            self.reservations.pop(request_id, None)
        record = {"request_id": request_id, "tenant": tenant, "step_seq": self.step_seq,
                  "prefill": prefill, "decode": decode, "recompute": recompute_prefill + recompute_decode,
                  "cost": cost, "accounts": {tenant: billed, SYSTEM: system}, "counter_delta": delta,
                  "wasted": False}
        self.ledger.setdefault(self.step_seq, []).append(record)
        return record

    def mark_wasted(self, step_seq: int, request_id: str) -> float:
        """Reclassify a step's work for a request that finished before its output landed.

        Returns the wasted cost. Idempotent: a record is reclassified at most once.
        """
        wasted = 0.0
        for record in self.ledger.get(step_seq, []):
            if record["request_id"] != str(request_id) or record["wasted"]:
                continue
            record["wasted"] = True
            wasted += record["cost"]
            tenant = record["tenant"]
            self.wasted[tenant] += record["cost"]
            if self.config.charge_wasted == SYSTEM:
                moved = record["accounts"][tenant]
                self.accounts[tenant] -= moved
                self.accounts[SYSTEM] += moved
                record["accounts"] = {tenant: 0.0, SYSTEM: record["cost"]}
        return wasted

    def _key(self, request: Any, counters: dict[str, float]) -> tuple[float, float, str]:
        meta = self.requests.get(str(request.request_id)) or self.register(request)
        return counters[meta["tenant"]], meta["arrival_time"], meta["rid"]

    def select(self, heads: Iterable[Any], *, counters: dict[str, float] | None = None) -> Any | None:
        """Pick the head of the least-served tenant (ties: earliest arrival)."""
        heads = list(heads)
        if not heads:
            return None
        counters = self.effective_counters if counters is None else counters
        keys = {id(r): self._key(r, counters) for r in heads}
        lowest = min(k[0] for k in keys.values())
        window = [r for r in heads if keys[id(r)][0] <= lowest + self.config.delta]
        return min(window, key=lambda r: keys[id(r)][1:])

    def order_running(self, requests: Iterable[Any]) -> list[Any]:
        """Order running requests least-served tenant first.

        vLLM spends the step's token budget in ``running`` order and, under KV
        pressure, preempts ``running[-1]``. Sorting ascending therefore gives
        under-served tenants the budget first and makes the newest request of
        the most over-served tenant the preemption victim, without the policy
        ever preempting anything itself.
        """
        counters = self.effective_counters
        ordered = sorted(requests, key=lambda r: self._key(r, counters))
        if self.config.preempt_order == "counter":
            return ordered
        # kv_share: tenants holding more than their weighted share of the
        # running requests' KV go to the back, keeping counter order inside
        # each group. The victim is then the newest request of the
        # highest-counter tenant among them, never a tenant within its share
        # while such a tenant is running. A light tenant just lifted to the top
        # counter is therefore not preempted for a heavy tenant's KV growth.
        held: dict[str, int] = {}
        for r in ordered:
            tenant = self.requests[str(r.request_id)]["tenant"]
            held[tenant] = held.get(tenant, 0) + int(getattr(r, "num_computed_tokens", 0))
        total, weights = sum(held.values()), sum(self.weight(t) for t in held)
        over = {t for t, n in held.items() if total and n > total * self.weight(t) / weights}
        return sorted(ordered, key=lambda r: self.requests[str(r.request_id)]["tenant"] in over)
