"""Synthetic CPU contract vectors for the G0 secondary identity witness."""

from __future__ import annotations

import unittest

from binding_evidence import Process, TreeSnapshot, evaluate_bindings


def proc(ppid: int, start_ticks: int, namespace: str = "pid:[fixture]") -> Process:
    return Process(ppid, start_ticks, namespace)


def snap(
    children: dict[int, Process] | None = None,
    *,
    boot: str = "boot-a",
    root_ticks: int = 100,
    observer_ns: str = "pid:[fixture]",
) -> TreeSnapshot:
    return TreeSnapshot(
        boot,
        observer_ns,
        10,
        root_ticks,
        {10: proc(1, root_ticks), **(children or {20: proc(10, 101)})},
    )


def lines(name: str = "EngineCore", pid: int = 20) -> str:
    return f"({name} pid={pid}) ready\n({name} pid={pid}) step\n"


class BindingEvidenceTests(unittest.TestCase):
    def check(
        self,
        log: str | None = None,
        names: set[str] | None = None,
        first: TreeSnapshot | None = None,
        last: TreeSnapshot | None = None,
        **kwargs,
    ):
        return evaluate_bindings(
            lines() if log is None else log,
            {"EngineCore"} if names is None else names,
            snap() if first is None else first,
            snap() if last is None else last,
            log_custody_verified=kwargs.pop("custody", True),
            launcher=kwargs.pop("launcher", "multiprocessing"),
        )

    def test_g0_repeated_exact_prefix_binds(self):
        self.assertEqual(self.check()["EngineCore"].state, "bound")

    def test_one_line_does_not_bind(self):
        self.assertEqual(
            self.check(log="(EngineCore pid=20) ready\n")["EngineCore"].reason,
            "witness_missing",
        )

    def test_spliced_and_nonanchored_prefixes_do_not_bind(self):
        bad = (
            "log (EngineCore pid=20) ready\n"
            "(EngineCore pid=20(EngineCore_DP1 pid=21) x\n"
        )
        self.assertEqual(self.check(log=bad)["EngineCore"].state, "binding_unavailable")

    def test_worker_and_api_server_are_not_engine_witnesses(self):
        bad = lines("Worker_TP0") + lines("APIServer_DP0")
        self.assertEqual(self.check(log=bad)["EngineCore"].reason, "witness_missing")

    def test_two_dp_engines_bind_separately(self):
        children = {20: proc(10, 101), 21: proc(10, 102)}
        log = lines("EngineCore_DP0", 20) + lines("EngineCore_DP1", 21)
        result = self.check(
            log, {"EngineCore_DP0", "EngineCore_DP1"}, snap(children), snap(children)
        )
        self.assertEqual(
            {name: b.pid for name, b in result.items()},
            {"EngineCore_DP0": 20, "EngineCore_DP1": 21},
        )

    def test_missing_rank_does_not_erase_other_rank(self):
        result = self.check(
            lines("EngineCore_DP0"), {"EngineCore_DP0", "EngineCore_DP1"}
        )
        self.assertEqual(result["EngineCore_DP0"].state, "bound")
        self.assertEqual(result["EngineCore_DP1"].state, "binding_unavailable")

    def test_same_live_pid_two_names_is_ambiguous(self):
        log = lines("EngineCore_DP0") + lines("EngineCore_DP1")
        result = self.check(log, {"EngineCore_DP0", "EngineCore_DP1"})
        self.assertEqual({b.reason for b in result.values()}, {"ambiguous_pid"})

    def test_one_contradictory_name_blocks_two_positive_witnesses(self):
        log = lines("EngineCore_DP0") + "(EngineCore_DP1 pid=20) conflicting\n"
        result = self.check(log, {"EngineCore_DP0", "EngineCore_DP1"})
        self.assertEqual({b.reason for b in result.values()}, {"ambiguous_pid"})

    def test_same_name_two_live_pids_is_ambiguous(self):
        children = {20: proc(10, 101), 21: proc(10, 102)}
        log = lines("EngineCore", 20) + lines("EngineCore", 21)
        result = self.check(log, first=snap(children), last=snap(children))
        self.assertEqual(result["EngineCore"].reason, "ambiguous_name")

    def test_dead_old_pid_and_stable_new_pid_uses_new_identity(self):
        children = {21: proc(10, 102)}
        log = lines("EngineCore", 20) + lines("EngineCore", 21)
        result = self.check(log, first=snap(children), last=snap(children))
        self.assertEqual(result["EngineCore"].pid, 21)

    def test_restart_inside_window_does_not_splice(self):
        log = lines("EngineCore", 20) + lines("EngineCore", 21)
        result = self.check(
            log, first=snap({20: proc(10, 101)}), last=snap({21: proc(10, 102)})
        )
        self.assertEqual(result["EngineCore"].reason, "identity_unstable")

    def test_pid_reuse_cannot_bind(self):
        result = self.check(last=snap({20: proc(10, 200)}))
        self.assertEqual(result["EngineCore"].reason, "identity_unstable")

    def test_process_must_be_descendant_at_both_ends(self):
        result = self.check(last=snap({20: proc(99, 101)}))
        self.assertEqual(result["EngineCore"].reason, "identity_unstable")

    def test_grandchild_tp_worker_is_not_engine_core(self):
        tree = snap({20: proc(10, 101), 30: proc(20, 102)})
        result = self.check(lines("EngineCore", 30), first=tree, last=tree)
        self.assertEqual(result["EngineCore"].reason, "identity_unstable")

    def test_carriage_return_does_not_create_a_log_record(self):
        tree = snap({20: proc(10, 101), 30: proc(10, 102)})
        fake = "(APIServer pid=10) prompt=x\r(EngineCore pid=30) y\n" * 2
        result = self.check(fake, first=tree, last=tree)
        self.assertEqual(result["EngineCore"].reason, "witness_missing")

    def test_prompt_cr_plus_grandchild_cannot_bind(self):
        tree = snap({20: proc(10, 101), 30: proc(20, 102)})
        fake = "(APIServer pid=10) prompt=x\r(EngineCore pid=30) y\n" * 2
        result = self.check(fake, first=tree, last=tree)
        self.assertEqual(result["EngineCore"].state, "binding_unavailable")

    def test_pid_namespace_mismatch_is_unavailable(self):
        tree = snap({20: proc(10, 101, "pid:[container]")})
        result = self.check(first=tree, last=tree)
        self.assertEqual(result["EngineCore"].reason, "namespace_mismatch")

    def test_boot_or_root_identity_change_blocks_all(self):
        self.assertEqual(
            self.check(last=snap(boot="boot-b"))["EngineCore"].reason,
            "launch_root_unstable",
        )
        self.assertEqual(
            self.check(last=snap(root_ticks=200))["EngineCore"].reason,
            "launch_root_unstable",
        )

    def test_log_without_custody_or_missing_is_unavailable(self):
        self.assertEqual(
            self.check(custody=False)["EngineCore"].reason, "log_producer_unavailable"
        )
        result = evaluate_bindings(
            None, {"EngineCore"}, snap(), snap(), log_custody_verified=True
        )
        self.assertEqual(result["EngineCore"].reason, "log_producer_unavailable")

    def test_ray_has_no_prefix_binding(self):
        self.assertEqual(
            self.check(launcher="ray")["EngineCore"].reason, "unsupported_launcher"
        )

    def test_unexpected_live_engine_fails_cardinality(self):
        children = {20: proc(10, 101), 21: proc(10, 102)}
        log = lines() + lines("EngineCore_DP1", 21)
        result = self.check(log, first=snap(children), last=snap(children))
        self.assertEqual(result["EngineCore"].reason, "unexpected_live_engine")


if __name__ == "__main__":
    unittest.main()
