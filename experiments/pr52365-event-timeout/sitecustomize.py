"""Private timing witness for PR #52365; loaded only by the experiment server.

This does not replace the wait algorithm. It wraps the function before V1/V2
import their binding and writes one minimal record after each call. Missing
records are an apparatus failure, never evidence that a wait did not occur.
"""

import builtins
import json
import os
import sys
import time

_TRACE = os.environ.get("LLR_WAIT_TRACE_PATH")
_TARGET = "vllm.v1.worker.gpu.event_utils"

if _TRACE:
    _original_import = builtins.__import__

    def _emit(record):
        try:
            data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
            fd = os.open(_TRACE, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError:
            # Preserve the unchanged vLLM behavior; missing evidence is unscored.
            pass

    def _import_with_witness(name, globals=None, locals=None, fromlist=(), level=0):
        result = _original_import(name, globals, locals, fromlist, level)
        module = sys.modules.get(_TARGET)
        if module is not None and not getattr(module, "_llr_wait_wrapped", False):
            original = getattr(module, "wait_for_gpu_event", None)
            if original is not None:

                def measured_wait(event, operation):
                    started = time.monotonic()
                    status = "returned"
                    caller = sys._getframe(1).f_globals.get("__name__")
                    source = {
                        "vllm.v1.worker.gpu_model_runner": "v1",
                        "vllm.v1.worker.gpu.async_utils": "v2",
                    }.get(caller, "other")
                    try:
                        return original(event, operation)
                    except TimeoutError:
                        status = "timeout"
                        raise
                    except BaseException:
                        status = "error"
                        raise
                    finally:
                        _emit(
                            {
                                "kind": "wait",
                                "seconds": round(time.monotonic() - started, 6),
                                "status": status,
                                "source": source,
                            }
                        )

                module.wait_for_gpu_event = measured_wait
                module._llr_wait_wrapped = True
                _emit({"kind": "installed"})
                builtins.__import__ = _original_import
        return result

    builtins.__import__ = _import_with_witness
