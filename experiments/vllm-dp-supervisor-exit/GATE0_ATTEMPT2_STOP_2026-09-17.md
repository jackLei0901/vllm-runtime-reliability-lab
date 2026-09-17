# Gate 0 attempt 2 stop — 2026-09-17

Attempt 2 was stopped before mechanism observation. Both full-package subject
processes were externally killed with return code `-9` after roughly 20
seconds, before emitting their closed-shape records. The 90-second runner bound
did not fire. The container has a 2-GiB memory limit, and complete vLLM plus
PyTorch import was the only work preceding termination.

This result is not evidence for or against the supervisor hypothesis. The
scored revision instead loads the exact current-main `dp_supervisor.py` source
with minimal stubs for vLLM imports outside the exercised lifecycle path. It
continues to use real asyncio, subprocess, uvloop, aiohttp, FastAPI, uvicorn,
and psutil packages. The limitation is explicit: Gate 0 is a thin lifecycle
test, not an end-to-end serving run.
