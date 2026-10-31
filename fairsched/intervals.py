"""Sorted, disjoint half-open position intervals: which KV positions a request has executed."""
from __future__ import annotations


def overlap(intervals: list[tuple[int, int]], start: int, end: int) -> int:
    """Positions in [start, end) already covered by ``intervals``."""
    if not intervals or start >= intervals[-1][1]:
        return 0                       # the usual case: new positions past everything executed
    return sum(max(0, min(b, end) - max(a, start)) for a, b in intervals)


def merge(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    if intervals and intervals[-1][0] <= start <= intervals[-1][1]:
        # Common case (prefill chunks, decode): the new range starts inside or
        # at the end of the last interval, so extending it in place keeps the
        # list sorted and disjoint.
        if end > intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        return intervals
    merged: list[tuple[int, int]] = []
    for a, b in sorted([*intervals, (start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged
