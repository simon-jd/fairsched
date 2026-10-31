"""Multi-tenant VTC scheduling for vLLM V1.

The policy and queue import without vLLM. The scheduler classes need vLLM 0.31.0
and are loaded lazily, so ``--scheduler-cls fairsched.TenantAsyncScheduler`` works.
"""
from .policy import PolicyConfig, PolicyState, TenantPolicy
from .queue import TenantRequestQueue

__all__ = ["PolicyConfig", "PolicyState", "TenantPolicy", "TenantRequestQueue",
           "TenantAsyncScheduler", "TenantSyncScheduler"]


def __getattr__(name):
    if name in {"TenantAsyncScheduler", "TenantSyncScheduler"}:
        from . import scheduler
        return getattr(scheduler, name)
    raise AttributeError(name)
