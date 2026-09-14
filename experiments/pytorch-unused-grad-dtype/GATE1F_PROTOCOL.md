# Gate 1f: rank-1 shutdown-stage diagnostic

Status: **pre-execution**

Gate 1e established the frozen mechanism and termination outcomes in three of
three affected trials, but failed the strict capture gate because only rank 0
produced a Flight Recorder dump. Gate 1f does not reinterpret that failure and
does not rerun the campaign matrix. It performs one diagnostic affected-arm
trial to test the leading source-derived explanation for the missing rank-1
dump.

## Declared diagnostic change

Set `TORCH_CPP_LOG_LEVEL=INFO`. Reduce only the following fixed PyTorch
`ProcessGroupNCCL` messages to per-rank boolean flags; retain no raw log text:

1. `Starting to destroy process group, flushing operations.`
2. `Operations flushed, joining watchdog thread.`
3. `Watchdog joined, destroying NCCL communicators.`
4. `Destroy complete.`
5. `Observed flight recorder dump signal from another rank via TCPStore.`
6. `Broadcasting signal exception_dump to other ranks via TCPStore.`
7. `Failed to broadcast signal exception_dump through TCPStore.`
8. `Flight Recorder trace successfully dumped.`

The scanner accepts a flag only when the message is directly preceded by
PyTorch's `[PG ID ... Rank N]` prefix. An allow-listed message without an
unambiguous rank produces a closed `library_log_scan_error` code. The campaign
still writes the mechanism, stack, Flight Recorder and lifecycle summary, then
returns non-zero; the verifier rejects any non-null scanner error. The preflight also records
`torch.cuda.nccl.version()` and requires the PyTorch 2.13.0+cu130 pinned NCCL
version, 2.29.7.

## Frozen prediction

For rank 0:

| message flag | expected |
| --- | --- |
| dump signal broadcast succeeded | true |
| dump signal broadcast failed | false |
| dump succeeded | true |
| all four shutdown stages | false |
| dump signal observed from another rank | false |

For rank 1:

| message flag | expected |
| --- | --- |
| shutdown started | true |
| operations flushed | true |
| watchdog joined; destroying communicators | true |
| destroy complete | false |
| dump signal observed | false |
| dump signal broadcast succeeded/failed | false/false |
| dump succeeded | false |

The leading explanation is therefore: rank 1 passes `finalize()`, stops its
heartbeat monitor, and then remains inside communicator destruction. The last
sentence is an inference if the flags match; Gate 1f does not instrument NCCL
internals and cannot directly prove the exact blocking call.

The prediction is pinned to PyTorch v2.13.0 source: process-group shutdown
finalizes communicators, stops the watchdog and heartbeat monitor, and only
then destroys communicators
([ProcessGroupNCCL.cpp:1546-1592](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1546-L1592)).
The heartbeat monitor observes the shared-store dump signal
([lines 1870-1903](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1870-L1903));
the requesting rank broadcasts that signal through the store
([lines 2709-2711](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L2709-L2711)).

The rank-0 prediction proves that the cross-rank store request was attempted
successfully and gives `dump_success` a positive scanner control. Its four
negative shutdown-stage flags also fail if rank attribution is swapped.

The Gate 1e mechanism, stack, termination and lifecycle observations must still
match. The expected Flight Recorder boundary is one decodable rank-0 dump and a
missing rank-1 dump. This remains an incomplete dump set, never evidence that
rank 1 did not participate in the collective.

## Decision rule

- matching all frozen rank-0 and rank-1 flags confirms that the dump request was
  sent and supports the shutdown-stage observation on NCCL 2.29.7;
- any flag mismatch is retained and reported as a failed prediction;
- any scanner attribution error is retained as a bounded code and fails closed;
- a mechanism, provenance, lifecycle or privacy mismatch invalidates the run;
- one diagnostic trial is sufficient because Gate 1e already repeated the
  missing-rank pattern three times;
- no result from Gate 1f converts Gate 1e's strict capture result into PASS.

## Commands after review

```bash
python experiments/pytorch-unused-grad-dtype/verify_gate1f_freeze.py
python experiments/pytorch-unused-grad-dtype/gate1f_campaign.py \
  --output results/pytorch-unused-grad-dtype-gate1f-YYYYMMDD
python experiments/pytorch-unused-grad-dtype/verify_gate1f.py \
  results/pytorch-unused-grad-dtype-gate1f-YYYYMMDD
```
