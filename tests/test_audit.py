"""Step log behaviour when its writer falls behind (pure Python)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

from fairsched.audit import StepLog


def _wait(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


class StepLogTest(unittest.TestCase):
    def test_drops_are_reported_while_running(self):
        directory = Path(tempfile.mkdtemp(prefix="fairsched-audit-"))
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "steps.jsonl"
        log = StepLog(path, capacity=1)
        gate = threading.Event()
        write = log._file.write
        log._file.write = lambda line: (gate.wait(5), write(line))[1]   # stall the writer thread

        log.write({"n": 0})
        _wait(log._queue.empty)          # the writer holds record 0 and is stalled
        log.write({"n": 1})              # fills the queue
        log.write({"n": 2})              # dropped
        log.write({"n": 3})              # dropped
        self.assertEqual(log.dropped, 2)
        gate.set()
        _wait(log._queue.empty)
        log.write({"n": 4})
        _wait(log._queue.empty)
        log.write({"n": 5})
        log.close()

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r.get("n") for r in rows], [0, 1, 4, 5, None])
        self.assertNotIn("dropped", rows[1])
        self.assertEqual(rows[2]["dropped"], 2)       # first record after the drops
        self.assertNotIn("dropped", rows[3])          # reported once, not repeated
        self.assertEqual(rows[4], {"type": "end", "dropped": 2})


if __name__ == "__main__":
    unittest.main()
