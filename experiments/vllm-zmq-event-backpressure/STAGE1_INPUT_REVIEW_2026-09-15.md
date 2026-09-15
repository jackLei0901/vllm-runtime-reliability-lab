# Stage 1 exact input review

Status: **review only; not frozen and not executed**

The proposed private inputs are:

- `.stage1-private/server-command.json`
- `.stage1-private/request.json`

They are ignored by Git because the public evidence contract retains only their
SHA-256 values, not the request text or machine-specific model path.

## Launch choices

- Model: the existing local `Qwen3-4B-Instruct-2507` bf16 checkpoint. No model
  download is required.
- One GPU, one sequence, V1 API server.
- Eager execution to avoid compilation as a timing variable.
- Fixed seed 0, maximum model length 512 and GPU-memory utilization 0.70.
- Prefix caching explicitly enabled.
- Block size explicitly fixed at 16.
- Async scheduling explicitly enabled.
- ZMQ KV events explicitly enabled with queue size 1 and a loopback-only
  `tcp://127.0.0.1:5557` endpoint.

The normal checkpoint is preferred over the available FP8 checkpoint so the
experiment does not add quantization kernels to a queue-liveness question.

## Request choices

- Completion API with deterministic temperature 0.
- `ignore_eos=true` and `max_tokens=64`, ensuring the requested output exceeds
  the two-block minimum.
- A fixed 32-word synthetic prompt. Runtime usage must still report at least 16
  prompt tokens; the English word count is not treated as tokenizer evidence.
- Streaming usage explicitly enabled.

The campaign rejects the cell as `trigger_not_reached` unless runtime usage
reports at least 16 prompt tokens, 32 completion tokens and at least one
accepted KV-event batch.

The EngineCore hook records the SHA-256 of the `kv_events.py` that it imported.
The campaign requires that value to equal the selected Git tree's blob, so the
separate preflight process cannot conceal EngineCore-side package shadowing.

## Binary-build identity

The selected source trees contain Python code but still need compiled vLLM
extensions for model serving. The existing local vLLM 0.20.1 wheel is not an
acceptable shortcut: a read-only comparison found 1,111 tracked Python files
whose bytes differ from the pinned baseline tree. Import success would not
establish source/build compatibility.

The exact baseline wheel was downloaded and installed into two separate arm
environments with pinned PyTorch 2.13. The wheel, driver, dependency manifests
and all 19 installed wheel-binary hashes match across arms; both Git trees
remain clean. See
`STAGE1_BUILD_IDENTITY_RESULT_2026-09-15.md` for the result and the disclosed
disk-saving environment construction. Formal execution remains on hold until
that construction is reviewed and the campaign is frozen.
