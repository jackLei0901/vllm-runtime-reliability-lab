# Gate 0 superseded two-cell scored record

Date: 2026-09-17

The first scored Gate 0 revision covered only intentional shutdown and an
abnormal child exit during startup. Review then identified two missing serving
state cases and that the control called `_handle_signal()` directly instead of
delivering a process signal.

Revision 4 added post-ready child exit and post-ready probe failure, changed
the control to deliver `SIGTERM` with `os.kill()`, and recorded the new
source-derived predictions before executing those cells. The earlier summaries
are superseded but retained by content hash:

| Subject | Superseded summary SHA-256 |
| --- | --- |
| vLLM main `0bfc7a15d095fe83ecc82b50561a93c177fece2d` | `3942161d9066f80025c7da22b22adf2866aced868c19766ccc85baf9d696f71e` |
| #54963 `61950f9589938ffc73e4f4eaf4181c39f66afbaf` | `479ec17e70accc972c8771390f3a012bb95d54ede6e4aa4197e7d9a7934c5a35` |

Those two-cell summaries are historical audit references only. The Gate 0
verdict is based on the later four-cell summaries and their verifier.
