#!/usr/bin/env python3
"""Local M probe: actual QuantFP8 native method, not a model benchmark.

Run on a supported Linux CUDA/vLLM environment. Each arm has a fresh process
and compile cache. No installed vLLM file is modified. Generated code is in
the per-arm logs; absence of max_with_index is not a Hopper exclusion proof.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap


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
    old = "x_max, _ = x.abs().max(dim=-1)"
    if source.count(old) != 1:
        raise RuntimeError("Source changed: audit the per-token reduction first")
    original = source
    if args.arm == "amax":
        source = source.replace(old, "x_max = x.abs().amax(dim=-1)")
    namespace = {}
    exec(compile(source, "<local-M-probe>", "exec"),
         QuantFP8.forward_native.__globals__, namespace)
    native = namespace["forward_native"]

    @default_vllm_config()
    def run():
        quant = QuantFP8(False, GroupShape.PER_TOKEN)

        def fn(x):
            if args.case == "norm":
                # Synthetic native RMSNorm composition, not vLLM IR matching.
                xf = x.float()
                x = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True)
                                     + 1e-6)).to(x.dtype)
            return native(quant, x)

        compiled = torch.compile(fn, fullgraph=True)
        batches = (8, 32, 128) if args.shape_mode == "symbolic" else (1,)
        prime_batch = 128 if args.shape_mode == "symbolic" else 1
        prime = torch.randn(prime_batch, args.hidden, device="cuda", dtype=torch.bfloat16)
        if args.shape_mode == "symbolic":
            torch._dynamo.mark_dynamic(prime, 0)
        compiled(prime)
        torch.cuda.synchronize()
        # Batch=1 is an independent specialization, never pooled into this
        # symbolic graph. Any unexpected recompilation within either mode fails.
        torch._dynamo.config.error_on_recompile = True
        rows = []
        for batch in batches:
            x = torch.randn(batch, args.hidden, device="cuda", dtype=torch.bfloat16)
            output, scale = compiled(x)
            ref_output, ref_scale = fn(x)
            # Fusion may legally differ from eager rounding. Record this without
            # hiding generated code; cross-arm exactness is checked separately.
            eager_exact = True
            eager_error = None
            try:
                torch.testing.assert_close(output.float(), ref_output.float(), rtol=0, atol=0)
                torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
            except AssertionError as error:
                eager_exact = False
                eager_error = str(error)
            torch.cuda.synchronize()
            samples = [1000 * triton.testing.do_bench_cudagraph(lambda: compiled(x))
                       for _ in range(3)]
            def digest(t):
                return hashlib.sha256(t.contiguous().view(torch.uint8).cpu()
                                      .numpy().tobytes()).hexdigest()
            rows.append({"batch": batch, "us_samples": samples,
                         "output_hash": digest(output), "scale_hash": digest(scale),
                         "compiled_eager_exact": eager_exact,
                         "compiled_eager_difference": eager_error})
        result = {"arm": args.arm, "case": args.case, "shape_mode": args.shape_mode,
                  "hidden": args.hidden,
                  "gpu": torch.cuda.get_device_name(), "capability": capability,
                  "torch": torch.__version__, "vllm": vllm.__version__,
                  "source_hash": hashlib.sha256(original.encode()).hexdigest(),
                  "rows": rows}
        Path(args.result).write_text(json.dumps(result, indent=2))
    run()


def summarize(results):
    """Keep each codegen context separate; timings never determine a verdict."""
    contexts = []
    for case in ("quant", "norm"):
        for mode in ("symbolic", "batch1"):
            group = [r for r in results if r["case"] == case
                     and r["shape_mode"] == mode]
            equal = bool(group) and all(
                len({(r["rows"][i]["output_hash"], r["rows"][i]["scale_hash"])
                     for r in group}) == 1
                for i in range(len(group[0]["rows"]))
            )
            max_runs = [r for r in group if r["arm"] == "max"]
            amax_runs = [r for r in group if r["arm"] == "amax"]
            if len(max_runs) != 2 or len(amax_runs) != 1 or not equal:
                verdict = "INCONSISTENT"
            elif all(r["max_with_index_count"] > 0 for r in max_runs) and all(
                r["max_with_index_count"] == 0 for r in amax_runs
            ):
                verdict = "CONFIRMED"
            elif all(r["max_with_index_count"] == 0 for r in group):
                verdict = "ABSENT"
            else:
                verdict = "INCONSISTENT"
            contexts.append({"case": case, "shape_mode": mode,
                             "cross_arm_bitwise_equal": equal, "verdict": verdict})
    verdicts = {c["verdict"] for c in contexts}
    if len(verdicts) == 1:
        verdict = next(iter(verdicts))
    elif "INCONSISTENT" in verdicts:
        verdict = "INCONSISTENT"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "contexts": contexts, "results": results,
            "scope": "codegen probe for this GPU/build only; not model TPOT evidence",
            "absence_policy": "ABSENT on SM89 does not exclude a Hopper-specific result",
            "timing_policy": "operator timing is diagnostic only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fp8-m-probe")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("max", "amax"))
    parser.add_argument("--case", choices=("quant", "norm"))
    parser.add_argument("--shape-mode", choices=("symbolic", "batch1"))
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    # A/B/A in four independent contexts prevents pooled codegen conclusions.
    for case in ("quant", "norm"):
        for mode in ("symbolic", "batch1"):
            for position, arm in enumerate(("max", "amax", "max")):
                label = f"{case}-{mode}-{position}-{arm}"
                result = out / f"{label}.json"
                log = out / f"{label}.log"
                env = dict(os.environ, TORCH_LOGS="output_code,recompiles",
                           TORCHINDUCTOR_CACHE_DIR=str(out / f"cache-{label}"))
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--arm", arm, "--case", case, "--shape-mode", mode,
                           "--hidden", str(args.hidden), "--result", str(result)]
                with log.open("w") as stream:
                    process = subprocess.run(command, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"{label} failed; inspect {log}")
                data = json.loads(result.read_text())
                # Counts are log call occurrences, not dynamic instruction counts.
                data["max_with_index_count"] = len(
                    re.findall(r"\bmax_with_index\s*\(", log.read_text())
                )
                results.append(data)
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY_JSON " + json.dumps(summary))
    if any(not c["cross_arm_bitwise_equal"] for c in summary["contexts"]):
        raise RuntimeError("Cross-arm correctness failed")


if __name__ == "__main__":
    main()
