"""CPU-only measurement definitions; not a CUDA timing harness or a freeze."""

from __future__ import annotations

import math
from statistics import median

MODEL = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"
SHAPES = ((6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288))
M_VALUES = (1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 56, 63, 64)
CHANGED_M = (4, 8, 16, 24, 32, 40, 48, 56, 64)
CORRECTNESS_M = tuple(sorted(set(CHANGED_M) | {3, 5, 15, 17, 31, 33, 63}))
GRAPH_M = (4, 64)
FP16_SHAPE = (4096, 4096)
WARMUP = 5
REPLAYS = 9
BLOCKS = 7
REQUIRED_BLOCKS = 6
NOISE_MULTIPLIER = 3.0
RELATIVE_EFFECT = 0.03
ABSOLUTE_EFFECT_US = 0.5
CLOCK_DEVIATION = 0.10
GRAPH_CYCLES = 8


def positive(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("timings must be nonempty, finite, positive microseconds")
    return values


def rotation_plan(l2_bytes: int, set_bytes: int) -> tuple[int, int]:
    if l2_bytes <= 0 or set_bytes <= 0:
        raise ValueError("cache and set sizes must be positive")
    sets = max(3, (2 * l2_bytes + set_bytes - 1) // set_bytes + 1)
    # Whole cycles ensure that graph boundaries do not privilege weight sets.
    return sets, GRAPH_CYCLES * sets


def pair_order(block: int, left: str, right: str) -> tuple[str, ...]:
    if block < 0 or left == right:
        raise ValueError("block index and distinct position labels required")
    return (left, right, right, left) if block % 2 == 0 else (right, left, left, right)


def paired_difference(left, right) -> float:
    """Median of two n-replay positions per side; caller preserves raw samples."""
    left, right = positive(left), positive(right)
    if len(left) != 2 * REPLAYS or len(right) != 2 * REPLAYS:
        raise ValueError("each side needs two fixed n-replay positions")
    return median(left) - median(right)


def decision_bound(reference_us: float, aa_differences) -> float:
    positive([reference_us])
    differences = list(aa_differences)
    if len(differences) < REQUIRED_BLOCKS or any(
        not math.isfinite(x) for x in differences
    ):
        raise ValueError("at least six valid A-A block differences required")
    center = median(differences)
    floor = max(ABSOLUTE_EFFECT_US, RELATIVE_EFFECT * reference_us)
    if abs(center) > floor:
        raise ValueError("A-A position bias exceeds the minimum effect")
    spread = 1.4826 * median(abs(x - center) for x in differences)
    return max(NOISE_MULTIPLIER * spread, floor)


def exceeds_bound(differences, bound: float) -> bool:
    """Exactly seven planned blocks; rejected blocks remain None, never replaced."""
    positive([bound])
    differences = list(differences)
    if len(differences) != BLOCKS:
        raise ValueError("exactly seven planned blocks required")
    if any(x is not None and not math.isfinite(x) for x in differences):
        raise ValueError("non-finite result is not a valid block")
    valid = [x for x in differences if x is not None]
    return (
        len(valid) >= REQUIRED_BLOCKS
        and sum(x > bound for x in valid) >= REQUIRED_BLOCKS
        and median(valid) > bound
    )


def model_shapes(config: dict) -> tuple[tuple[int, int], ...]:
    expected = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("model dimensions differ from the reviewed configuration")
    quant = config.get("quantization_config", {})
    if (
        quant.get("weight_block_size") != [128, 128]
        or quant.get("quant_method") != "fp8"
        or quant.get("fmt") != "e4m3"
        or quant.get("activation_scheme") != "dynamic"
    ):
        raise ValueError("model is not the reviewed dynamic block-FP8 format")
    hidden, intermediate = expected["hidden_size"], expected["intermediate_size"]
    q, kv, dim = (
        expected["num_attention_heads"],
        expected["num_key_value_heads"],
        expected["head_dim"],
    )
    return (
        (dim * (q + 2 * kv), hidden),
        (hidden, dim * q),
        (2 * intermediate, hidden),
        (hidden, intermediate),
    )
