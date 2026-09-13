"""Prepare the three frozen four-GPU source variants from one pinned source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARER_PATH = (
    ROOT / "experiments" / "organic-hang" / "fetch_and_prepare_reproducer.py"
)

MATRIX = {
    "pp2-dp2-divergent": {
        "sha256": "430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1",
        "disable_random_output": False,
        "dp_only": False,
        "expected": "mixed_gradient_dtype_assertion",
    },
    "pp1-dp4-divergent": {
        "sha256": "5ced00d84f64cabdbb9503baf74de05ec294a592609a55b6066d506da72386eb",
        "disable_random_output": False,
        "dp_only": True,
        "expected": "completed",
    },
    "pp2-dp2-uniform": {
        "sha256": "25e84dae1e3514ada1b0047e855fb8c4ef4b08f4664b03f7f9e42d4cb95e5b1b",
        "disable_random_output": True,
        "dp_only": False,
        "expected": "completed",
    },
}


def load_preparer():
    spec = importlib.util.spec_from_file_location("organic_preparer", PREPARER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load organic reproducer preparer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_variants(source: bytes, output_dir: Path) -> dict[str, str]:
    module = load_preparer()
    if hashlib.sha256(source).hexdigest() != module.SOURCE_SHA256:
        raise ValueError("pinned source SHA-256 mismatch")
    base = module.enable_upstream_fix(module.prepare_source(source.decode("utf-8")))
    output_dir.mkdir(parents=True, exist_ok=False)
    observed: dict[str, str] = {}
    for case, contract in MATRIX.items():
        prepared = base
        if contract["disable_random_output"]:
            prepared = module.disable_random_output(prepared)
        if contract["dp_only"]:
            prepared = module.use_dp_only_topology(prepared)
        compile(prepared, str(output_dir / f"{case}.py"), "exec")
        encoded = prepared.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != contract["sha256"]:
            raise ValueError(f"prepared hash mismatch for {case}: {digest}")
        (output_dir / f"{case}.py").write_bytes(encoded)
        observed[case] = digest
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_preparer()
    source = module.read_source(args.source)
    for case, digest in prepare_variants(source, args.output).items():
        print(f"{case}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
