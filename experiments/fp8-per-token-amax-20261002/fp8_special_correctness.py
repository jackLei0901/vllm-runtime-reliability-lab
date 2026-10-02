"""Local correctness check of patched native FP8 against the original reduction."""

import inspect
import json
import textwrap

import torch
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def check_pair(lhs, rhs):
    out_a, scale_a = lhs
    out_b, scale_b = rhs
    assert torch.equal(
        out_a.contiguous().view(torch.uint8), out_b.contiguous().view(torch.uint8)
    )
    torch.testing.assert_close(scale_a, scale_b, rtol=0, atol=0, equal_nan=True)


@torch.inference_mode()
@default_vllm_config()
def main():
    source = textwrap.dedent(inspect.getsource(QuantFP8.forward_native))
    new = "x_max = x.abs().amax(dim=-1)"
    old = "x_max, _ = x.abs().max(dim=-1)"
    assert source.count(new) == 1 and old not in source
    scope = {}
    exec(  # noqa: S102 - recreate the inspected method with only the reduction reverted
        compile(source.replace(new, old), "<original-native>", "exec"),
        QuantFP8.forward_native.__globals__,
        scope,
    )
    original = scope["forward_native"]
    quant = QuantFP8(False, GroupShape.PER_TOKEN)
    rows = []
    torch.manual_seed(42)
    for dtype in (torch.bfloat16, torch.float32):
        for bounded in (False, True):
            torch._dynamo.reset()
            ub = torch.tensor(2.0, device="cuda") if bounded else None

            def baseline(x, ub=ub):
                return original(quant, x, scale_ub=ub)

            def candidate(x, ub=ub):
                return quant.forward_native(x, scale_ub=ub)

            compiled_old = torch.compile(baseline, fullgraph=True)
            compiled_new = torch.compile(candidate, fullgraph=True)
            x = torch.randn(7, 1025, device="cuda", dtype=dtype)
            x[0] = 0
            x[0, 0] = -0.0
            x[1] = 1
            x[1, :2] = torch.tensor([4.0, -4.0], device="cuda", dtype=dtype)
            check_pair(baseline(x), candidate(x))
            check_pair(compiled_old(x), compiled_new(x))
            for value in (float("nan"), float("inf"), -float("inf")):
                special = x.clone()
                special[2, 0] = value
                check_pair(baseline(special), candidate(special))
                check_pair(compiled_old(special), compiled_new(special))
            rows.append(
                {
                    "dtype": str(dtype),
                    "bounded": bounded,
                    "finite_eager_and_compiled_exact": True,
                    "special_eager_and_compiled_behavior_preserved": True,
                }
            )
    print("SUMMARY_JSON " + json.dumps({"rows": rows, "status": "PASS"}))


if __name__ == "__main__":
    main()
