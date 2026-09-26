"""CPU-only, private rank/PID binding experiment; not a v0.2 verdict.

The caller must independently retain the build, launch configuration, log
digest and verified fresh-log boundary. A prefix alone is never liveness.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

PREFIX = re.compile(r"^\((EngineCore(?:_DP[0-9]+)?) pid=([0-9]+)\) ")
NAME = re.compile(r"EngineCore(?:_DP[0-9]+)?\Z")


@dataclass(frozen=True)
class Process:
    ppid: int
    start_ticks: int
    pid_namespace: str


@dataclass(frozen=True)
class TreeSnapshot:
    boot_id: str
    observer_pid_namespace: str
    root_pid: int
    root_start_ticks: int
    processes: Mapping[int, Process]


@dataclass(frozen=True)
class Binding:
    state: str  # bound | binding_unavailable
    reason: str | None
    pid: int | None  # private, never put in a public bundle
    start_ticks: int | None


def _identity_error(pid: int, first: TreeSnapshot, last: TreeSnapshot) -> str | None:
    a = first.processes.get(pid)
    b = last.processes.get(pid)
    if a is None or b is None:
        return "identity_unstable"
    if (
        not a.pid_namespace
        or a.pid_namespace != first.observer_pid_namespace
        or b.pid_namespace != last.observer_pid_namespace
    ):
        return "namespace_mismatch"
    if (
        a.start_ticks != b.start_ticks
        or a.start_ticks < first.root_start_ticks
        or a.ppid != first.root_pid
        or b.ppid != last.root_pid
    ):
        return "identity_unstable"
    return None


def evaluate_bindings(
    log_text: str | None,
    expected_names: set[str],
    first: TreeSnapshot,
    last: TreeSnapshot,
    *,
    log_custody_verified: bool,
    launcher: str = "multiprocessing",
) -> dict[str, Binding]:
    """Join repeated prefixes to stable live descendants of one launch root.

    Both snapshots must bound the same evaluation window. A restarted child
    needs a new window; this function never splices identities across one.
    """
    if not expected_names or any(NAME.fullmatch(n) is None for n in expected_names):
        raise ValueError("expected_names must be a nonempty exact EngineCore set")

    def unavailable(reason: str) -> dict[str, Binding]:
        return {
            name: Binding("binding_unavailable", reason, None, None)
            for name in sorted(expected_names)
        }

    if launcher != "multiprocessing":
        return unavailable("unsupported_launcher")
    if log_text is None or not log_custody_verified:
        return unavailable("log_producer_unavailable")
    if (
        not first.boot_id
        or first.boot_id != last.boot_id
        or not first.observer_pid_namespace
        or first.observer_pid_namespace != last.observer_pid_namespace
        or first.root_pid != last.root_pid
        or first.root_start_ticks != last.root_start_ticks
        or first.processes.get(first.root_pid) is None
        or last.processes.get(last.root_pid) is None
        or first.processes[first.root_pid].start_ticks != first.root_start_ticks
        or last.processes[last.root_pid].start_ticks != last.root_start_ticks
    ):
        return unavailable("launch_root_unstable")
    if (
        first.processes[first.root_pid].pid_namespace != first.observer_pid_namespace
        or last.processes[last.root_pid].pid_namespace != last.observer_pid_namespace
    ):
        return unavailable("namespace_mismatch")

    witnesses = Counter(
        (m[1], int(m[2]))
        # _add_prefix starts a new record on LF only, not on CR or Unicode separators.
        for line in log_text.split("\n")
        if (m := PREFIX.match(line)) is not None
    )
    errors = {pair: _identity_error(pair[1], first, last) for pair in witnesses}
    # One well-formed contradictory witness is enough to weaken a binding;
    # two consistent witnesses are still required to make a positive claim.
    live = {pair for pair, error in errors.items() if error is None}
    if any(name not in expected_names for name, _ in live):
        return unavailable("unexpected_live_engine")

    names_by_pid: dict[int, set[str]] = defaultdict(set)
    pids_by_name: dict[str, set[int]] = defaultdict(set)
    for name, pid in live:
        names_by_pid[pid].add(name)
        pids_by_name[name].add(pid)

    result: dict[str, Binding] = {}
    for name in sorted(expected_names):
        pids = pids_by_name[name]
        if len(pids) > 1:
            result[name] = Binding("binding_unavailable", "ambiguous_name", None, None)
        elif not pids:
            name_errors = [error for (n, _), error in errors.items() if n == name]
            reason = (
                "namespace_mismatch"
                if "namespace_mismatch" in name_errors
                else "identity_unstable"
                if name_errors
                else "witness_missing"
            )
            result[name] = Binding("binding_unavailable", reason, None, None)
        else:
            pid = next(iter(pids))
            if len(names_by_pid[pid]) > 1:
                result[name] = Binding(
                    "binding_unavailable", "ambiguous_pid", None, None
                )
            elif witnesses[(name, pid)] < 2:
                result[name] = Binding(
                    "binding_unavailable", "witness_missing", None, None
                )
            else:
                result[name] = Binding(
                    "bound", None, pid, last.processes[pid].start_ticks
                )
    return result
