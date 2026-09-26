# vLLM TP PyNccl: RAS / Inspector A-B capability result — 2026-09-26

Status: **bounded producer comparison; no Lab C++ probe admitted**. This
follow-up extends the [first RAS capability result](VLLM_TP_RAS_CAPABILITY_RESULT_2026-09-26.md).
It is not a vLLM serving hang, a transport hang, or a performance benchmark.
Each cell below ran once. The only route exercised was direct
`PyNcclCommunicator.all_reduce`, not automatic `CudaCommunicator` dispatch.

## Question and pre-registered distinction

What can existing producers report *while completion is delayed*? The eager
A/B pair tests NCCL issued-call counts. The graph A/B pair tests exported
records, but is **not** an admissible NCCL-layer discrimination test: its
decisive difference is whether the host called `cudaGraphLaunch`.

- **A — peer host call/replay delayed:** rank 1 delays its host call/replay by
  three seconds; rank 0 has issued and awaits the peer.
- **B — both host calls/replays issued:** rank 1 has queued a bounded
  `torch.cuda._sleep(5_000_000_000)` before its collective on the same CUDA
  stream. A CPU barrier confirms both host calls returned. A CUDA event
  confirms the graph is still pending before and after the diagnostic query.
  This is a delayed device dependency, not a transport fault.

The zero-cycle B control completes normally. All arms verify the collective's
output after completion. Graph healthy controls change the input before each
replay and check the corresponding output, rather than trusting a value left
from capture.

## Environment and identity

- Two RTX 4090 GPUs; Linux; vLLM source
  `c8602c79062440074a018c1d5f875a5571eb6881`; torch
  `2.13.0+cu130`; installed NCCL `2.29.7`.
- NCCL Inspector source: NVIDIA tag [`v2.29.7-1`](https://github.com/NVIDIA/nccl/tree/v2.29.7-1/plugins/profiler/inspector),
  commit `b91894bd5b190c874d98a017f93f5daa515b65d0`. Its standalone
  `make -j2` build succeeded. Re-read directly from the retained binary on
  the same host, its SHA-256 is
  `03091146feb5e551e454f65674ea43a5e8cb1aeab42e719dbfb36c541c99895f`.
  The earlier 65-character terminal transcription was invalid and is
  superseded by this 64-character re-read.
- Inspector was configured with `NCCL_PROFILER_PLUGIN` pointing to that
  library, `NCCL_INSPECTOR_ENABLE=1`, a 500-microsecond dump interval,
  `NCCL_INSPECTOR_DUMP_VERBOSE=1`, and a run-private dump directory. Both rank
  files contained parseable Inspector JSON records. The eager healthy cell
  is a positive plugin-loading and record-production control.
- `NCCL_RAS_ENABLE=1` exposed localhost port 28028. RAS and Inspector
  outputs were retained only in private mode-0700 directories; individual
  files and the complete archive had mode 0600. The archive contains 48 raw
  files and has SHA-256
  `881744d83b757c3785a721a2e9231b6a5ad4c299e1b88d0c229ebfb89cfca71a`.
  Raw hostnames, IP addresses, PIDs, communicator IDs, and addresses are not
  published here. The archive is **not** a public replay bundle.

The original seven-cell RAS run recorded these script digests: baseline
`e5496b708c64c1d3554377c1e76d40be8e3d9854614150944bf9c88bbd8751d8`,
peer hold `a70e6d75537014a760e074ac4df6c1f2d809eadbc47eda674d4baadc3ce365ae`,
eager B `914f90285a4af66455867701a03b0a8ad5aa672dc77e309862519805374516a4`,
graph B `1d66f11e6b30387f6d4c519e13361c5044baff72b18254364466d0e4c6143b5e`.
The eager B script (`914f…`) is retained and committed at `1bd7134`.
The original baseline, peer-hold and graph B versions were overwritten in
place and are **not retained as files in this checkout**. A read-only search
of the original host found only their current versions, and the private
evidence archive contains no `.py` files. Those three digests identify the
execution transcript, not independently reviewable source blobs; this
provenance gap cannot be closed from the currently retained artifacts.

The later *in-window A/B* Inspector snapshots imported the current
[`baseline`](../../experiments/vllm-tp-dfx/ras_graph_baseline.py)
`a4122908aecafa27b10c729680906b4e49ae50fc726602b5392da15cd488f516`,
[`A runner`](../../experiments/vllm-tp-dfx/ras_peer_hold.py)
`18fc05beac6f0a6509030189bcf93e4e607a3f4c37fcb53adc506419d3d66ba8`,
and [`graph B runner`](../../experiments/vllm-tp-dfx/ras_graph_all_issued_delay.py)
`3fa7a28fa0095355c20ff377014d7a2714924469ba41962d86b683c2841aa77b`.
The four current runners are committed at Lab `1bd7134`; that commit pins
the **later** script state only and does not recover the earlier overwritten
versions.
The healthy Inspector baseline runs happened **before** the baseline was
changed to `a412…`; they used the unretained `e549…` version.
The only added instrumentation reads complete Inspector JSON lines and
publishes per-rank `AllReduce` record counts; it does not change the fault
or the collective.

## Observations and run provenance

All rows returned success and passed their output assertions. Counts are
per-rank. `before / during / after` refer to bounded observation points in
the runners, not to a continuous time-series. Inspector record counts are
for complete JSON lines already flushed to disk at the query instant.

| Cell and run | RAS `AllReduce` counts | Inspector records | Independent completion check |
| --- | --- | --- | --- |
| Healthy eager, Inspector baseline run (unretained `e549…`) | `2/2 → 22/22` | 20 / 20 after | 20 changing-input outputs correct |
| Healthy graph, Inspector baseline run (unretained `e549…`) | `1/1 → 1/1` | 20 / 20 after | 20 changing-input outputs correct |
| Eager A, original RAS run (`a70e…`) | `2/2 → 3/2 → 3/3` | Not sampled | Released output correct |
| Eager B, original zero-cycle RAS run (`914f…`) | `2/2 → 3/3 → 3/3` | Not sampled | Output correct; no pending event |
| Eager B, original delayed RAS run (`914f…`) | `2/2 → 3/3 → 3/3` | Not sampled | Sleep event pending across RAS query; ~1831 ms to completion |
| Graph A, later in-window Inspector run (`18fc…`) | `2/2 → 2/2 → 2/2` | `1/1 → 1/1 → 2/2` | Released output correct |
| Graph B, original zero-cycle RAS run (`1d66…`) | `2/2 → 2/2 → 2/2` | Not sampled | Output correct; no pending event |
| Graph B, later in-window Inspector run (`3fa7…`) | `2/2 → 2/2 → 2/2` | `1/1 → 1/1 → 2/2` | Both host replays issued; event pending before and after RAS query; ~1831 ms to completion |

The healthy eager and graph RAS observations were also made in the original
seven-cell campaign; the table deliberately uses their **later Inspector
baseline runs** with the same unretained `e549…` script, where RAS and
Inspector were observed together. The committed `a412…` file cannot be used
to inspect those healthy runs. Earlier
standalone Inspector A/B runs did not sample Inspector during the hold and
are not combined with the later in-window rows. The graph A/B RAS baseline
starts at `2/2` because those runners perform an eager warmup *and* graph
capture before the first snapshot; the healthy graph baseline starts at
`1/1` after capture without that eager warmup.

In eager mode, RAS separated A's issued-count asymmetry from B's equal
issued counts during the hold. The B event was still pending when queried,
so an equal issued count is **not** evidence of completed work.

In graph mode, the two mechanisms produced the same RAS counts throughout.
Inspector did record graph collectives: its healthy graph files each held 20
records after 20 verified replays. But in *both* A and B, during the delayed
window its already-flushed records remained `1/1`; the additional record
appeared only after completion. Thus, for these two tested in-window
observations, neither RAS count nor Inspector's completed JSON records
separated A from B. The healthy run establishes post-completion replay
visibility; it does not establish in-flight attribution.

That is specifically an **Inspector export policy**, not absence of NCCL
profiler callbacks. At the pinned tag, Inspector subscribes to
[`ncclProfileColl | ncclProfileKernelCh`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L117),
records [collective](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L203-L232)
and [kernel-channel](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L259-L293)
start events, then sets [`commInfo->dump`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L420)
only after every channel completes;
the [JSON writer](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector.cc#L548)
emits a record only when that flag is set. Verbose mode adds fields to a
completed record, not an in-flight export. Whether `KernelCh` start callbacks
actually fire during this specific graph B delay remains **unmeasured**.

The graph A/B comparison has another, independent limit. In graph A, rank 1
has not called `graph.replay()`. In graph B, it has called replay, but the
same-stream `_sleep` precedes its NCCL kernel. At the observed instant,
neither arm establishes a started rank-1 NCCL kernel. The known distinction
is **host graph-launch state**, which a NCCL-level producer need not observe.
Equal NCCL output here is therefore not a reason to invent a NCCL probe.

## Decision and limits

1. **No probe admission yet.** Eager RAS distinguishes the admitted
   issued-call A/B pair, while graph RAS counts freeze and Inspector exports
   completed records only. The graph A/B pair cannot score NCCL-level
   attribution because its distinguishing fact is host replay issuance.
   `nsys` was unavailable on this host; no CUDA runtime trace was run.
2. **No vLLM-serving or general-NCCL claim.** This is direct PyNccl on NCCL
   2.29.7 and two consumer GPUs. The source suggests vLLM V1 may capture
   decode collectives, but we did not bind an actual serving collective to
   this path and graph segment. Nor does a bounded `_sleep` model a real
   transport failure.
3. **Next gate, split by layer:** for host issue, check whether a CUDA
   runtime/CUPTI trace of `cudaGraphLaunch` or an existing vLLM step marker
   answers the question. For NCCL in-flight state, check whether the already
   documented v5 profiler interface delivers `KernelCh` start callbacks in
   the delayed window, before completion; a lab-local Inspector debug build
   or minimal callback logger is an *interface capability check*, not probe
   admission. Only after testing the right producer against a separable
   pair should we reconsider the minimal C++ probe gate in
   [`LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](../LOW_LEVEL_CAPABILITY_REQUIREMENTS.md).
   Repeating issue counts or adding generic hardware sampling would not
   answer this A/B question.

This is a capability finding and a reviewable negative result, not a claim
that vLLM has a newly discovered hang bug. No upstream issue or PR is opened
from this campaign.
