"""CPU worker that holds ``execute_model`` on request, for the PR #55700 probe.

Loaded inside the vLLM server through ``--worker-cls hold_worker.HoldingCPUWorker``;
PR code is not modified. A hold is requested by creating
``$PR55700_CONTROL_DIR/hold_rank<rank>``. The worker consumes that file on its
next ``execute_model`` call, writes ``held_rank<rank>``, and waits until
``release`` exists. The wait uses ``time.sleep`` so the GIL is released and the
watchdog thread in the same process can run; a hold that kept the GIL would
leave no witness and the case would be ``unscored``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from vllm.v1.worker.cpu_worker import CPUWorker

_CONTROL = Path(os.environ["PR55700_CONTROL_DIR"])
_POLL_S = 0.2


class HoldingCPUWorker(CPUWorker):
    def execute_model(self, *args, **kwargs):
        trigger = _CONTROL / f"hold_rank{self.rank}"
        if trigger.exists():
            trigger.unlink()
            (_CONTROL / f"held_rank{self.rank}").write_text(f"{time.time()}\n")
            release = _CONTROL / "release"
            while not release.exists():
                time.sleep(_POLL_S)
            (_CONTROL / f"released_rank{self.rank}").write_text(f"{time.time()}\n")
        return super().execute_model(*args, **kwargs)
