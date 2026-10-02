# SM90 block-FP8 performance experiment: insufficient evidence

Status: frozen first attempt closed; no rerun. The Chinese version is authoritative.

## Freeze and execution

- Public freeze: `c8d767ad98ddda4a899cf46c6a17ad7256a7e282`.
- Manifest SHA-256: `88f98c9c53028132f84372538b8ed90cd9df46d6e3881e9a141fe3942c425c25`.
- Both arms were built and executed on one H800 PCIe, with Torch `2.13.0+cu130` and CUDA toolkit 13.0.
- Approximately 12.4 minutes elapsed from the recorded session start to timing completion, below the user-authorized 90-minute cap. Cumulative work against the 18-hour budget was unverified; the user explicitly authorized this session. Earlier attempts and their budget history remain recorded.
- Both builds succeeded. The candidate predicate, cases and scoring rules were not changed during execution. The public freeze was checked online once; subsequent steps used its offline receipt.

## Results

| Evidence | Result | Meaning |
| --- | --- | --- |
| E1 correctness | 176/176 passed | Correctness passed for covered cases, not all possible inputs |
| E2 actual dispatch | witnessed | The two arms actually execute different kernels at M=64 |
| E3 timing | insufficient_evidence | The completeness gate failed; proposal value is undecided |
| E4 serving support | unwitnessed | Concurrency 1 and 16 passed; 63 and 64 failed the frozen first-ten-graph-step check |

All 80 A-A calibration records and 80 base/variant records were collected, but six calibration cells were unscored, as were their six paired records. The other pairs had descriptive classifications of 22 positive, seven negative and 45 below_positive_gate. These counts cannot bypass completeness or support a selectively reported speedup or rejection.

The six unscored calibration cells all used `(N=24576, K=4096)`, at M=24, 40, 48, 56, 63 and 64. They retained respectively five, five, four, four, three and three valid blocks, below the required six. The reference SM clock was 1755 MHz; some timed blocks exceeded the frozen 10% clock-deviation bound. No blocks were replaced, no extra data were collected and no threshold was relaxed. This does not establish interference by another tenant or a particular hardware fault.

E4 was exported and reviewed using CPU work after E3. The expected kernel-class check failed at concurrency 63 and 64, as detailed below. Exact executed M remains unmeasured; the concurrency-63 padding inference is not confirmed. E4 supplies no quantitative performance or end-to-end benefit evidence.

## Complete descriptive cell table

Descriptive only: the decision stopped at completeness. No frozen outcomes are rescored. P = positive; N = negative; B = below_positive_gate (improvement not demonstrated); U = unscored. `*` marks a changed M; other rows are controls. The four columns cover all 80 paired cells, including the six unscored cells.

| M | (6144,4096) | (4096,4096) | (24576,4096) | (4096,12288) |
| --- | --- | --- | --- | --- |
| 1 | B | B | B | B |
| 2 | B | B | B | B |
| 3 | B | B | B | B |
| 4* | P | P | P | P |
| 5 | B | B | B | B |
| 7 | B | B | B | B |
| 8* | P | P | P | P |
| 15 | B | B | B | B |
| 16* | P | P | P | P |
| 17 | B | B | B | B |
| 24* | P | P | U | P |
| 31 | B | B | B | B |
| 32* | P | P | N | P |
| 33 | B | B | B | B |
| 40* | N | P | U | P |
| 47 | B | B | B | B |
| 48* | N | P | U | P |
| 56* | B | N | U | N |
| 63 | B | B | U | B |
| 64* | B | N | U | N |

### Shape counts by M

These are observed classifications, not acceptance scores. The candidate's seven negatives are all changed cells. Among 43 scored unchanged controls, none exceeds its bound in either direction; the remaining control is unscored. Thus the observed controls do not add a second control-out-of-bound failure, but complete control validation is unavailable.

| M | P | N | B | U |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 4 | 0 |
| 2 | 0 | 0 | 4 | 0 |
| 3 | 0 | 0 | 4 | 0 |
| 4* | 4 | 0 | 0 | 0 |
| 5 | 0 | 0 | 4 | 0 |
| 7 | 0 | 0 | 4 | 0 |
| 8* | 4 | 0 | 0 | 0 |
| 15 | 0 | 0 | 4 | 0 |
| 16* | 4 | 0 | 0 | 0 |
| 17 | 0 | 0 | 4 | 0 |
| 24* | 3 | 0 | 0 | 1 |
| 31 | 0 | 0 | 4 | 0 |
| 32* | 3 | 1 | 0 | 0 |
| 33 | 0 | 0 | 4 | 0 |
| 40* | 2 | 1 | 0 | 1 |
| 47 | 0 | 0 | 4 | 0 |
| 48* | 2 | 1 | 0 | 1 |
| 56* | 0 | 2 | 1 | 1 |
| 63 | 0 | 0 | 3 | 1 |
| 64* | 0 | 2 | 1 | 1 |

### Clock and E4 detail

Failed timing blocks / collected blocks, using the frozen clock test:

| (N,K) | AA | base/variant |
| --- | --- | --- |
| (6144,4096) | 0/140 | 0/140 |
| (4096,4096) | 0/140 | 0/140 |
| (24576,4096) | 19/140 | 9/140 |
| (4096,12288) | 0/140 | 0/140 |

All 28 rejected blocks contain clocks **below** 90% of the 1755 MHz reference; none contains a clock above 110%. The lowest recorded timed-block clock is 1485 MHz, **84.62%** of reference (15.38% below). The six unscored AA cells account for 18 rejected AA blocks; another calibrated AA cell lost one block but retained the required six. A software power-cap flag appears in these samples and is permitted by the frozen mask; it is the clock deviation that invalidates them. Sustained large GEMMs on a power-limited PCIe card are a plausible explanation; power/thermal causality is not established. No inference about tenant interference or hardware defects follows.

An earlier user-provided GPU inventory in this conversation identified H800 PCIe too. The retained summary of the earlier incomplete witness session does not independently establish its form factor; no SXM/PCIe or cross-card performance comparison is claimed.

For each E4 level, the fixed first ten selected graph steps all have 144 target GEMMs, present and unique graph-node IDs, and no ambiguous/failed-launch errors. At concurrency 1 and 16, the expected kernel class passes 10/10. At concurrency 63 and 64, **10/10 steps fail only the expected Cooperative-class test**: each contains 144 Pingpong kernels and zero Cooperative kernels. Their decode labels record one token in all ten steps. At concurrency 16 the labels record five tokens in two steps and 16 in eight; at concurrency 1 all ten record one. A client-concurrency window therefore does not establish that its first ten selected steps have the same number of active decode tokens. Exact kernel M remains unmeasured. These observations describe the frozen checker failure; they do not select later steps or repair E4 after seeing the data.

Measurement lesson: the 10% clock rule was not piloted on this card class. Future, separately scoped timing should lock clocks where permitted or characterize clock-check pass rates on the largest shape before freezing the gate. This is not a proposal to rerun this closed experiment.

### Handover accounting

The user explicitly handed the instance to the INT8 thread at **2026-10-01 21:55:47 America/Los_Angeles** (2026-10-02 04:55:47 UTC, the logged handover-message time). Subsequent GPU minutes belong outside this experiment. INT8 on H800 is a separately authorized operator probe, not a continuation of the archived H20-only investigation; its decision and time cap must be owned by that thread.

## Preservation and conclusion

- Private archive SHA-256: `a698c3679af28bdae10fad3192e6be229f583111de85b6f375e9d1aa6560a966`. It was downloaded and its local digest matches the remote digest. It contains builds, binaries, receipts, raw timing, logs and traces; real paths, PIDs and GPU UUIDs remain private.
- Both downloaded binary digests were independently checked against their build receipts.
- The controller closed as `closed_first_attempt`, with `timing failed; no retry`. The scientific outcome is insufficient evidence, not `no_gap` or a validated speedup.
- The log contains `libgomp: Invalid value for environment variable OMP_NUM_THREADS`; it remains an unattributed warning, without an in-session change or rerun.
- This frozen experiment ends without a third attempt. Evidence does not justify a performance PR or a claim of end-to-end gain in default serving.
- Lab measurement processes have finished; at sealing, GPU memory usage was 0 MiB and the GPU was idle. Under the user's later authorization, the instance remains running for INT8 validation rather than being shut down.
- The frozen packet again reports `MANIFEST_MATCHES`; the protocol files remain unchanged from the freeze.
