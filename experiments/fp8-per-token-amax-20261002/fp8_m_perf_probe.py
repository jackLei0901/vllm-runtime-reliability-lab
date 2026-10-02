#!/usr/bin/env python3
"""Operator timing for per-token QuantFP8.forward_native: max vs amax (arm M).

Decision fed: may the #59564 comment claim a per-token speedup, or only the
codegen difference? Not model TPOT evidence. Method follows #25094/#59564:
compile once with the largest batch marked dynamic, no input clone, CUDA-graph
timing. Batch 1 is a separate static context. CUDA forward_cuda is timed in
every process as a within-process reference. No installed vLLM file is modified.
"""

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap

SYMBOLIC_TOKENS = (8, 32, 128, 512, 2048, 8192)
ARMS = ("max", "amax", "max", "amax", "max")
OLD = "x_max, _ = x.abs().max(dim=-1)"
NEW = "x_max = x.abs().amax(dim=-1)"


def worker(args):
    import hashlib
    import torch
    import vllm
    from vllm.benchmarks.lib.utils import default_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.triton_utils import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 9):
        raise RuntimeError("Use an FP8-capable SM89+ device; prefer Hopper")
    torch.manual_seed(42)
    torch._dynamo.config.automatic_dynamic_shapes = False
    source = textwrap.dedent(inspect.getsource(QuantFP8.forward_native))
    if source.count(OLD) != 1:
        raise RuntimeError("Source changed: audit the per-token reduction first")
    original = source
    if args.arm == "amax":
        source = source.replace(OLD, NEW)
    namespace = {}
    exec(compile(source, "<local-M-perf>", "exec"),
         QuantFP8.forward_native.__globals__, namespace)
    native = namespace["forward_native"]

    def median_us(fn):
        samples = [1000 * triton.testing.do_bench_cudagraph(fn)
                   for _ in range(args.samples)]
        return statistics.median(samples), samples

    def digest(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu()
                              .numpy().tobytes()).hexdigest()

    @default_vllm_config()
    def run():
        quant = QuantFP8(False, GroupShape.PER_TOKEN)
        compiled = torch.compile(lambda x: native(quant, x), fullgraph=True)
        symbolic = args.context == "symbolic"
        tokens_list = SYMBOLIC_TOKENS if symbolic else (1,)
        prime = torch.randn(max(tokens_list), args.hidden, device="cuda",
                            dtype=torch.bfloat16)
        if symbolic:
            torch._dynamo.mark_dynamic(prime, 0)
        compiled(prime)
        torch.cuda.synchronize()
        torch._dynamo.config.error_on_recompile = True

        rows = []
        for tokens in tokens_list:
            x = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
            output, scale = compiled(x)
            compiled_us, compiled_samples = median_us(lambda: compiled(x))
            cuda_us, cuda_samples = median_us(lambda: quant.forward_cuda(x))
            rows.append({"tokens": tokens,
                         "compiled_us": compiled_us,
                         "compiled_samples": compiled_samples,
                         "cuda_us": cuda_us, "cuda_samples": cuda_samples,
                         "output_hash": digest(output),
                         "scale_hash": digest(scale)})
        result = {"arm": args.arm, "context": args.context, "hidden": args.hidden,
                  "gpu": torch.cuda.get_device_name(), "capability": capability,
                  "torch": torch.__version__, "vllm": vllm.__version__,
                  "source_hash": hashlib.sha256(original.encode()).hexdigest(),
                  "rows": rows}
        Path(args.result).write_text(json.dumps(result, indent=2))
    run()


def classify(speedup, drift, threshold):
    if speedup >= threshold and speedup >= 3 * drift:
        return "GAIN"
    if speedup <= -threshold and -speedup >= 3 * drift:
        return "LOSS"
    return "FLAT"


def summarize(results, threshold, expected_hidden_sizes=None):
    table, invalid = [], []
    hidden_sizes = (expected_hidden_sizes if expected_hidden_sizes is not None
                    else sorted({r["hidden"] for r in results}))
    if not results or not hidden_sizes:
        invalid.append("No results")
    metadata = ("gpu", "capability", "torch", "vllm", "source_hash")
    for key in metadata:
        if any(key not in r for r in results):
            invalid.append(f"Missing provenance: {key}")
        elif len({json.dumps(r[key], sort_keys=True) for r in results}) > 1:
            invalid.append(f"Provenance differs: {key}")
    keys = [(hidden, context) for hidden in hidden_sizes
            for context in ("symbolic", "batch1")]
    if any((r["hidden"], r["context"]) not in keys for r in results):
        invalid.append("Unexpected hidden/context")
    for hidden, context in keys:
        runs = [r for r in results if r["hidden"] == hidden and r["context"] == context]
        max_runs = [r for r in runs if r["arm"] == "max"]
        amax_runs = [r for r in runs if r["arm"] == "amax"]
        if [r["arm"] for r in runs] != list(ARMS):
            invalid.append(f"{hidden}/{context}: expected A/B/A/B/A")
            continue
        tokens = list(SYMBOLIC_TOKENS if context == "symbolic" else (1,))
        if any([row.get("tokens") for row in r["rows"]] != tokens for r in runs):
            invalid.append(f"{hidden}/{context}: token rows missing or misaligned")
            continue
        for i, row in enumerate(amax_runs[0]["rows"]):
            observed = [r["rows"][i] for r in runs]
            if any(not isinstance(rr.get(field), str) or not rr[field]
                   for rr in observed for field in ("output_hash", "scale_hash")):
                invalid.append(f"{hidden}/{context}/{row['tokens']}: missing output hashes")
                continue
            valid_timing = all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for rr in observed for field in ("compiled_us", "cuda_us")
                for v in [rr.get(field)]
            )
            valid_samples = all(
                isinstance(rr.get(field), list) and len(rr[field]) >= 3
                and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                        for v in rr[field])
                for rr in observed for field in ("compiled_samples", "cuda_samples")
            )
            if not valid_timing or not valid_samples:
                invalid.append(f"{hidden}/{context}/{row['tokens']}: invalid timing")
                continue
            if any(not math.isclose(rr[field], statistics.median(rr[samples]),
                                    rel_tol=1e-9, abs_tol=1e-12)
                   for rr in observed for field, samples in
                   (("compiled_us", "compiled_samples"), ("cuda_us", "cuda_samples"))):
                invalid.append(f"{hidden}/{context}/{row['tokens']}: median mismatch")
                continue
            hashes = {(r["rows"][i]["output_hash"], r["rows"][i]["scale_hash"])
                      for r in runs}
            if len(hashes) != 1:
                invalid.append(f"{hidden}/{context}/{row['tokens']}: outputs differ")
            baseline = [r["rows"][i]["compiled_us"] for r in max_runs]
            candidate = [r["rows"][i]["compiled_us"] for r in amax_runs]
            max_us, amax_us = statistics.median(baseline), statistics.median(candidate)
            drift = (max(baseline) - min(baseline)) / max_us
            candidate_spread = (max(candidate) - min(candidate)) / amax_us
            def relative_mad(rr, field):
                samples = rr[field]
                median = statistics.median(samples)
                return statistics.median(abs(v - median) for v in samples) / median
            sample_noise = max(relative_mad(rr, "compiled_samples") for rr in observed)
            noise = max(drift, candidate_spread, sample_noise)
            reductions = [1 - b / a for a in baseline for b in candidate]
            reduction = 1 - amax_us / max_us
            cuda_values = [rr["cuda_us"] for rr in observed]
            cuda_us = statistics.median(cuda_values)
            cuda_spread = (max(cuda_values) - min(cuda_values)) / cuda_us
            cuda_noise = max(relative_mad(rr, "cuda_samples") for rr in observed)
            if cuda_spread > max(threshold, 3 * cuda_noise):
                verdict = "RECHECK"
            elif all(classify(v, noise, threshold) == "GAIN" for v in reductions):
                verdict = "GAIN"
            elif all(classify(v, noise, threshold) == "LOSS" for v in reductions):
                verdict = "LOSS"
            else:
                verdict = "FLAT"
            table.append({"hidden": hidden, "context": context,
                          "tokens": row["tokens"],
                          "max_us": round(max_us, 3),
                          "amax_us": round(amax_us, 3),
                          "max_run_us": baseline, "amax_run_us": candidate,
                          "cuda_us": round(cuda_us, 3),
                          "cuda_run_us": cuda_values,
                          "cuda_reference_spread": round(cuda_spread, 4),
                          "latency_reduction": round(reduction, 4),
                          "min_pairwise_reduction": round(min(reductions), 4),
                          "max_pairwise_reduction": round(max(reductions), 4),
                          "max_drift": round(drift, 4),
                          "amax_compile_spread": round(candidate_spread, 4),
                          "sample_relative_mad": round(sample_noise, 4),
                          "verdict": verdict})
    verdicts = {row["verdict"] for row in table}
    if invalid:
        overall = "INVALID"
    elif "RECHECK" in verdicts:
        overall = "RECHECK"
    elif len(verdicts) == 1:
        overall = next(iter(verdicts))
    else:
        overall = "MIXED"
    groups = {}
    for name, tokens in (("batch1", (1,)), ("low_batch", (8, 32, 128)),
                         ("large_batch", (512, 2048, 8192))):
        group_rows = [row for row in table if row["tokens"] in tokens]
        counts = {v: sum(row["verdict"] == v for row in group_rows)
                  for v in sorted({row["verdict"] for row in group_rows})}
        groups[name] = {"verdict_counts": counts, "rows": len(group_rows)}
    return {"verdict": overall, "threshold": threshold, "invalid": invalid,
            "table": table, "batch_groups": groups,
            "decision_policy": "Inspect each shape; overall verdict is not PR authorization",
            "drift_policy": "CUDA reference drift is a confound flag, not proof of its cause",
            "scope": "operator timing on this GPU/build; not model TPOT evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fp8-m-perf")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[4096, 7168, 14336])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("max", "amax"))
    parser.add_argument("--context", choices=("symbolic", "batch1"))
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.samples < 3 or not 0 < args.threshold < 1:
        parser.error("Require samples >= 3 and 0 < threshold < 1")
    sizes = [args.hidden] if args.worker else args.hidden_sizes
    if any(size is None or size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        parser.error("Hidden sizes must be positive and unique")
    if args.worker:
        worker(args)
        return
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for hidden in args.hidden_sizes:
        for context in ("symbolic", "batch1"):
            for position, arm in enumerate(ARMS):
                label = f"h{hidden}-{context}-{position}-{arm}"
                result = out / f"{label}.json"
                log = out / f"{label}.log"
                env = dict(os.environ, TORCH_LOGS="recompiles",
                           TORCHINDUCTOR_CACHE_DIR=str(out / f"cache-{label}"))
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--arm", arm, "--context", context,
                           "--hidden", str(hidden), "--samples", str(args.samples),
                           "--result", str(result)]
                with log.open("w") as stream:
                    process = subprocess.run(command, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"{label} failed; inspect {log}")
                results.append(json.loads(result.read_text()))
    summary = summarize(results, args.threshold, args.hidden_sizes)
    summary["results"] = results
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY_JSON " + json.dumps({k: summary[k] for k in
                                        ("verdict", "threshold", "invalid", "batch_groups", "table")}))
    if summary["verdict"] == "INVALID":
        sys.exit(1)


if __name__ == "__main__":
    main()
