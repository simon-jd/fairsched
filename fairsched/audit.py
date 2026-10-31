"""Per-step scheduler log: records are serialised in schedule(); file writes happen on a writer thread."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
from typing import Any


class StepLog:
    """Bounded queue plus one writer thread, so ``schedule()`` never blocks on disk.

    ``schedule()`` pays only for building and serialising the record (it is not
    included in the logged ``sched_ns``).

    The writer flushes whenever it catches up with the queue, so the file is
    complete up to the last flushed line even if the engine process is killed
    before ``close()`` runs (``docker stop`` does that to EngineCore). If the
    writer falls behind by ``capacity`` records, new records are counted as
    dropped instead of stalling the engine. The first record queued after a
    drop carries the running total as ``dropped``, so the count reaches the
    file while the engine runs, not only in the final ``end`` record.
    """

    def __init__(self, path: str | Path, *, capacity: int = 65536):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dropped = 0
        self._reported = 0      # the dropped total last written into a queued record
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=capacity)
        # "x": never overwrite an earlier run's log.
        self._file = open(self.path, "x", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="fairsched-steplog", daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        dropped = self.dropped
        if dropped != self._reported:
            record = {**record, "dropped": dropped}
        try:
            self._queue.put_nowait(json.dumps(record, separators=(",", ":")))
        except queue.Full:
            self.dropped += 1
        else:
            self._reported = dropped

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(json.dumps({"type": "end", "dropped": self.dropped}))
            self._queue.put(None)
            self._thread.join(timeout=10)

    def _run(self) -> None:
        with self._file as output:
            while (line := self._queue.get()) is not None:
                output.write(line + "\n")
                if self._queue.empty():
                    output.flush()
