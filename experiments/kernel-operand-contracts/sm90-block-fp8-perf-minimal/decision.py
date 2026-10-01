"""Precommitted engineering decision; no GPU imports or equivalence claim."""

import math

import protocol


def cell_key(row):
    return (
        tuple(row["shape"]),
        row["left_m"],
        row["right_m"],
        row["left_cache"],
        row["right_cache"],
    )


def timing_outcome(records):
    result = {"status": "insufficient_evidence", "improved_shape_counts": {}}
    expected = {
        (shape, m, m, "rotating", "rotating")
        for shape in protocol.SHAPES
        for m in protocol.M_VALUES
    }
    groups = {"AA": {}, "variant": {}}
    try:
        for row in records:
            group = groups[row["kind"]]
            key = cell_key(row)
            if key not in expected or key in group:
                raise ValueError("unexpected or duplicate cell")
            group[key] = row
        if any(set(group) != expected for group in groups.values()):
            raise ValueError("missing cell")
        if any(row.get("status") != "calibrated" for row in groups["AA"].values()):
            raise ValueError("unscored calibration")
        pairs = list(groups["variant"].values())
        for row in pairs:
            if row.get("status") not in ("positive", "negative", "below_positive_gate"):
                raise ValueError("unscored pair")
            diff, bound = row["median_difference_us"], row["bound_us"]
            if not math.isfinite(diff) or not math.isfinite(bound) or bound <= 0:
                raise ValueError("invalid bound/difference")
        controls = [r for r in pairs if r["left_m"] not in protocol.CHANGED_M]
        if any(abs(r["median_difference_us"]) > r["bound_us"] for r in controls):
            raise ValueError("unchanged-path control outside bound")
        changed = [r for r in pairs if r["left_m"] in protocol.CHANGED_M]
        counts = {
            m: sum(r["status"] == "positive" for r in changed if r["left_m"] == m)
            for m in protocol.CHANGED_M
        }
        result["improved_shape_counts"] = counts
        if any(r["status"] == "negative" for r in changed):
            return {
                **result,
                "status": "not_worth_proposing",
                "reason": "changed-cell regression",
            }
        improved = sum(count >= 3 for count in counts.values())
        return {
            **result,
            "status": "worth_proposing" if improved >= 7 else "not_worth_proposing",
            "improved_m_count": improved,
            "reason": "engineering acceptance rule; not equivalence",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {**result, "reason": str(exc)}


def correctness_outcome(rows):
    from correctness import cases

    expected = {
        (arm, tuple(shape), m, dtype, mode)
        for arm in ("base", "variant")
        for shape, m, dtype, mode in cases()
    }
    indexed = {}
    try:
        for row in rows:
            shape, m, dtype, mode = row["case"]
            key = row["arm"], tuple(shape), m, dtype, mode
            if key not in expected or key in indexed:
                return "unscored"
            indexed[key] = row["status"]
        if set(indexed) != expected or any(
            v not in ("passed", "failed") for v in indexed.values()
        ):
            return "unscored"
        if all(v == "passed" for v in indexed.values()):
            return "passed"
        if all(v == "passed" for k, v in indexed.items() if k[0] == "base"):
            return "not_worth_proposing"
        return "unscored"
    except (KeyError, TypeError, ValueError):
        return "unscored"
