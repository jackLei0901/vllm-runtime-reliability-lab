# Block 5 native-evidence capability run

Status: Stage A Linux CPU smoke and Stage B v2 scored RTX 4090 validation are
complete. See
[`../../results/native-stack-pair-stage-b-v2-20260922/README.md`](../../results/native-stack-pair-stage-b-v2-20260922/README.md).

This experiment answers one bounded question: can real `py-spy` and PyStack
captures of a deterministic held target satisfy the same attribution rule? It
does not extend the v0.2 bundle or verifier.

## Why A/B/A2

Attach producers run sequentially. The sequence is therefore fixed:

```text
py-spy A -> PyStack B -> py-spy A2
```

A and A2 must later satisfy the same rule predicates and emit the same
`blocked_in`. Otherwise the result is `target_not_stable` and B is not judged.
Frame-sequence equality is deliberately not an acceptance condition.
When no rule has been admitted, three `unmatched` results are `not_scorable`,
not `interchangeable`. An unavailable or failed capture is also
`not_scorable`.

## Environment

- Linux target and observer in the same PID namespace;
- operator-supplied PID and `/proc/<pid>/stat` start ticks;
- `py-spy` and `pystack` on `PATH`;
- ptrace authorization for both tools;
- PyStack native dependencies available (`libdw` and `libelf`);
- a target held at the pre-registered healthy or fault marker for all three
  captures.

PyStack is Linux-only. A container normally needs `CAP_SYS_PTRACE` and a
compatible seccomp policy. Yama scope is recorded separately by the campaign;
it is not used as a preflight permission verdict because scoped target
authorization may still permit attach.

## Capture

First read and retain the target identity:

```bash
pid=<operator-supplied-pid>
start_ticks=$(python -c 'import pathlib,sys; p=pathlib.Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(p[p.rfind(")")+2:].split()[19])' "$pid")

python experiments/native-evidence-capability/run_stack_pair.py \
  --pid "$pid" \
  --start-ticks "$start_ticks" \
  --role engine_core \
  --vllm-version 0.1.dev1+g7e100f011 \
  --pytorch-version 2.13.0+cu130 \
  --pytorch-backend not-applicable \
  --nccl-version not-applicable \
  --topology single-process \
  --output results/native-stack-pair-YYYYMMDD
```

Raw output is written only below `private/`. `capture-record.json` contains
producer provenance, monotonic bounds, typed outcomes, subject binding, and raw
content digests. It intentionally ends with:

```json
{
  "pairing_status": "normalization_pending",
  "claim": null
}
```

The capture command cannot claim interchangeability. That claim requires a
reviewed normalizer, explicit target-runtime constraints, an admitted rule, and
the A/B/A2 comparison in `dfxlab.native_evidence`.

Stage B v2 supplies that reviewed path in `score_stage_b_pair.py` and
`stage_b_queue_wait_rule.json`. The scorer verifies raw digests privately,
normalizes only the admitted predicates, and publishes no raw frame text.

An execution that times out, exceeds its output budget, or otherwise fails may
retain a digest of bounded partial output. That digest identifies only the
retained bytes; it does not change the typed outcome to `produced` and partial
output is not eligible for attribution.

The rule `platform` field describes the observer/producer platform. In this
runbook the observer and target share a PID namespace and host, so it is also
the target platform; that equivalence must not be assumed for a future
cross-host collector.

## First case

Use the retained #53859 Stage 1 base/pause target first. Its known fault window
already has a negative control and a pre-registered queue-wait chain. Do not
create a new fault campaign. The initial run should answer only whether mixed
native/Python frames and GIL state improve the existing `queue_wait`
attribution.

The #196968 teardown target follows after the first pipeline is reviewed. It
may justify one lab-local lifecycle probe only if PyStack, Flight Recorder, and
NCCL RAS cannot expose the required responder/shutdown transition. An
upstream-facing probe remains gated on an explicit #197232 maintainer outcome.
