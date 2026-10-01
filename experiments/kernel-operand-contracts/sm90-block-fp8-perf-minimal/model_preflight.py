"""Hash a previously downloaded pinned model on CPU; never downloads implicitly."""

import argparse
import json
from pathlib import Path

import protocol
import runtime


def inspect_model(directory):
    directory = Path(directory).resolve()
    # HF snapshot directories bind the requested revision. A copied arbitrary
    # folder needs its own verified download record; this tool rejects it.
    if directory.name != protocol.MODEL_REVISION:
        raise ValueError("use the pinned Hugging Face snapshot directory")
    config = directory / "config.json"
    if protocol.model_shapes(json.loads(config.read_text())) != protocol.SHAPES:
        raise ValueError("unexpected model shapes")
    index = directory / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values())
    if not names or any(
        Path(n).name != n or not n.endswith(".safetensors") for n in names
    ):
        raise ValueError("unsafe or empty weight index")
    files = {n: runtime.sha(directory / n) for n in sorted(names)}
    files[config.name] = runtime.sha(config)
    files[index.name] = runtime.sha(index)
    return {
        "model": protocol.MODEL,
        "revision": protocol.MODEL_REVISION,
        "files_sha256": files,
        "weights_bytes": sum((directory / n).stat().st_size for n in names),
        "shapes": protocol.SHAPES,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "packet": runtime.packet_hashes()}
    try:
        result.update(inspect_model(args.model_dir))
        result["status"] = "verified_model_files"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "model.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "verified_model_files":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
