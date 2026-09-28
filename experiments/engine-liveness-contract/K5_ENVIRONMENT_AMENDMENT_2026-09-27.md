# K5 environment amendment after an unscored startup attempt

The frozen K5 TP=1 attempt ended `unscored_server_not_ready` before the
baseline request or arm file. The private server log identifies
`FileNotFoundError: ninja` while FlashInfer was compiling its sampling module.
The executable already exists in the pinned vLLM environment's `bin`
directory; the invoking shell did not include that directory in `PATH`.
No worker hold was injected and no C3 observation was scored.

This amendment permits a separate, explicitly labelled environment-corrected
round in the same booking. Its only change is to prepend the existing vLLM
environment's `bin` directory to `PATH` before invoking the unchanged K5
runner. The model, vLLM build, plugin, source digests, 30-second RPC timeout,
45-second hold, health windows, and scoring code remain unchanged. The
original unscored attempt is retained and is not pooled with the corrected
round. Use fresh output directories and ports, run TP=1 then TP=2 once each,
and stop on another pre-arm failure rather than tuning further.

The correction is needed for runtime compilation, not to alter the target
engine's liveness behavior. It is still a protocol deviation, not part of the
original preregistered run. Publish the original attempt and corrected results
separately, including both private-log digests.
