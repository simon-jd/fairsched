"""Per-tenant FIFO queues behind vLLM 0.31.0's ``RequestQueue`` interface.

vLLM admits waiting requests by peeking and popping this queue's head. Here the
head is the oldest request of the least-served tenant, so admission follows VTC
while each tenant's own requests stay first-come-first-served.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Iterator

from .policy import PolicyState


class TenantRequestQueue:
    """Implements the ten ``RequestQueue`` operations the native scheduler calls."""

    def __init__(self, state: PolicyState):
        self.state = state
        self._queues: dict[str, deque[Any]] = {}

    def _tenant(self, request: Any) -> str:
        return self.state.register(request)["tenant"]

    def _heads(self) -> list[Any]:
        return [queue[0] for queue in self._queues.values() if queue]

    def add_request(self, request: Any) -> None:
        tenant = self._tenant(request)      # registers (and lifts) before counting it waiting
        self._queues.setdefault(tenant, deque()).append(request)
        self.state.enqueued(tenant)

    def peek_request(self) -> Any:
        selected = self.state.select(self._heads())
        if selected is None:
            raise IndexError("peek from an empty queue")
        return selected

    def pop_request(self) -> Any:
        selected = self.peek_request()
        tenant = self._tenant(selected)
        self._queues[tenant].popleft()
        # Later pops in this step must see the cost this admission will add,
        # and so must the departure counter if this empties the tenant's queue.
        self.state.reserve(selected)
        self.state.dequeued(tenant)
        return selected

    def prepend_request(self, request: Any) -> None:
        self.state.rollback(request.request_id)
        tenant = self._tenant(request)
        self._queues.setdefault(tenant, deque()).appendleft(request)
        self.state.enqueued(tenant)

    def prepend_requests(self, requests: Iterable[Any]) -> None:
        # Matches FCFSRequestQueue.extendleft: the last input ends up first.
        for request in list(requests):
            self.prepend_request(request)

    def remove_request(self, request: Any) -> None:
        tenant = self._tenant(request)
        self._queues[tenant].remove(request)
        self.state.dequeued(tenant)
        self.state.rollback(request.request_id)

    def remove_requests(self, requests: Iterable[Any]) -> None:
        ids = {str(request.request_id) for request in requests}
        for tenant, queue in self._queues.items():
            kept = deque(r for r in queue if str(r.request_id) not in ids)
            for _ in range(len(queue) - len(kept)):
                self.state.dequeued(tenant)
            self._queues[tenant] = kept
        for request_id in ids:
            self.state.rollback(request_id)

    def __bool__(self) -> bool:
        return any(self._queues.values())

    def __len__(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def __iter__(self) -> Iterator[Any]:
        """Yield in projected admission order, without mutating any state."""
        queues = {tenant: deque(queue) for tenant, queue in self._queues.items()}
        projected = self.state.effective_counters
        while any(queues.values()):
            selected = self.state.select([q[0] for q in queues.values() if q], counters=projected)
            tenant = self._tenant(selected)
            queues[tenant].popleft()
            remaining = max(0, int(selected.num_prompt_tokens) - int(selected.num_computed_tokens))
            projected[tenant] += remaining * self.state.config.w_p / self.state.weight(tenant)
            yield selected

    def counts(self) -> dict[str, int]:
        return {tenant: len(queue) for tenant, queue in self._queues.items()}
