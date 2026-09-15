# Stage 1a CPU preflight result

Date: 2026-09-15

Verdict: **PASS on both source trees after the EngineCore-side import-hash
contract was added**

This was an integration preflight, not a scored Stage 1 inference result. The
machine had one RTX 4090 visible, but vLLM was forced to the CPU platform and no
model was loaded. GPU memory usage was 0 MiB after the run.

## Source and runtime

- Base tree: `b7061e73a6ed4773e16bd2ae3acf47aebfd1342d`
- Same-base #53883 tree: `46bc6e191b14ce12a04827454b4588ea5d3a435f`
- Python: 3.12.3
- PyTorch: `2.13.0+cu130`, the same pinned runtime as formal Stage 1
- CUDA reported by PyTorch: 13.0
- vLLM versions: `0.1.dev1+g9935dfceb` on base and
  `0.1.dev2+g6bf585185` on fix
- Yama `ptrace_scope`: 1
- Observer authorization: `pr_set_ptracer_observer`
- Both imported `kv_events.py` files matched their respective Git blobs.
- The EngineCore-side hook independently hashed its imported `kv_events.py`;
  both hashes matched the corresponding source-tree blobs.
- Both runs used identical hashes for the plugin, preflight, campaign helpers
  and scoring contract.
- The preflight loads no model and maps no source-tree `.so`; both ready records
  therefore contain an empty mapped-binary set. Formal Stage 1 requires that
  set to be non-empty and match the build identity.

The preflight was repeated after the Stage 1 campaign file changed. Only the
second pair is retained; this keeps the recorded dependency hashes equal to the
current reviewed files.

## Observations

| Observation | Base | #53883 |
| --- | ---: | ---: |
| Accepted before release | 1 | 1 |
| Dropped before release | 0 | 1 |
| Accepted after release | 2 | 1 |
| Dropped after release | 0 | 1 |
| Stack attempts | 1 | 0 (not required) |
| Stack chain matched | yes | not sampled |
| Subject return code | 0 | 0 |
| Process group gone | yes | yes |
| PID/start-time identity gone | yes | yes |

The base stack matched on the first attempt after the pre-registered one-second
settling interval. It contained the disclosed lab frame between production
frames:

```text
threading.Condition.wait
queue.Queue.put
dfx_stage1_backpressure.observed_put
ZmqEventPublisher.publish
```

This closes the Stage 1a authorization question: calling `PR_SET_PTRACER` from
the publisher thread authorizes the external observer process-wide on this
Yama=1 kernel. Formal Stage 1 adopts one post-stall sample. If that sample is
unavailable or does not match, the result remains `evidence_unavailable`; the
CPU `inproc://` preflight does not claim a sampling-success guarantee under GPU
serving load.

## Integrity

- `stage1a-base.json` SHA-256:
  `b435b955c3e7506f201a46ed5de9664666999dd6bcd8b6fe7ea366423aaa57a4`
- `stage1a-fix.json` SHA-256:
  `e2f57a1fbab066ad386069b3dd149df355b94ab6a3aa3445214ebc25996df49c`
- Independent verifier result:
  `PASS: Stage 1a real plugin/publisher preflight verified on both trees`

Raw subject logs and the raw py-spy dump remain private. The public summaries
retain only closed fields and the raw stack hash.
