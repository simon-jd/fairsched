"""Figures for the README (needs matplotlib)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.report import load_client, load_steps, service_timeline  # noqa: E402

ARM_ORDER = ["native", "fcfs", "fcfs-cap", "priority", "priority-b", "vtc", "vtc-kv", "vtc-fit"]
ARM_LABEL = {"native": "vLLM FCFS (no plugin)", "fcfs": "vLLM FCFS", "fcfs-cap": "vLLM FCFS, gateway cap 16/tenant",
             "priority": "vLLM priority (A high)", "priority-b": "vLLM priority (B high)", "vtc": "VTC (this project)",
             "vtc-kv": "VTC, kv_share victim order", "vtc-fit": "VTC, fitted decode weight"}
TENANT_COLOR = {"A": "#2f6fb0", "B": "#d1762b"}


def _first_seed(runs: list[Path]) -> dict[str, Path]:
    chosen: dict[str, Path] = {}
    for run in sorted(runs):
        meta = json.loads((run / "meta.json").read_text())
        chosen.setdefault(meta["arm"], run)
    return {arm: chosen[arm] for arm in ARM_ORDER if arm in chosen}


def plot_ttft_cdf(runs: dict[str, Path], tenant: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for arm, run in runs.items():
        rows = [r for r in load_client(run / "client.jsonl") if r["tenant"] == tenant and r["state"] == "completed"]
        ttft = sorted((r["first_output_unix_ns"] - r["scheduled_unix_ns"]) / 1e9 for r in rows)
        if ttft:
            ax.step(ttft, [(i + 1) / len(ttft) for i in range(len(ttft))], where="post", label=ARM_LABEL[arm])
    ax.set_xscale("log")
    ax.set_xlabel(f"tenant {tenant} time to first token (s, log scale)")
    ax.set_ylabel("fraction of requests")
    ax.grid(alpha=0.3)
    # Below the axes, so the legend never covers a curve.
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_service(runs: dict[str, Path], path: Path, bin_s: float = 2.0) -> None:
    arms = [a for a in runs if a != "native"]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.2), sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        run = runs[arm]
        since = json.loads((run / "meta.json").read_text())["trace_start_unix"]
        timeline = service_timeline(load_steps(run), since, bin_s)
        for tenant in ("A", "B"):
            series = timeline.get(tenant, [])
            ax.plot([i * bin_s for i in range(len(series))], series, color=TENANT_COLOR[tenant], label=f"tenant {tenant}")
        ax.set_title(ARM_LABEL[arm], fontsize=9)
        ax.set_xlabel("time since start (s)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("billed service (weighted tokens/s)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_workload(workload: str, runs: list[Path], out: Path) -> None:
    chosen = _first_seed(runs)
    if workload in {"noisy_neighbor", "long_prompt", "smoke"}:
        plot_ttft_cdf(chosen, "A", out / "ttft_A.png")
    if any(arm != "native" for arm in chosen):
        plot_service(chosen, out / "service.png")
