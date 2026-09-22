from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dfxlab.native_producers import capture_native_stack


class _CompletedProcess:
    def __init__(self, command, *, stdout, payload: bytes, return_code: int = 0, **_):
        self.command = command
        self.stdout = stdout
        self.stdout.write(payload)
        self.stdout.flush()
        self.return_code = return_code

    def wait(self, timeout):
        self.timeout = timeout
        return self.return_code

    def kill(self):
        raise AssertionError("successful capture must not be killed")


class _TimedOutProcess(_CompletedProcess):
    def wait(self, timeout=None):
        if getattr(self, "killed", False):
            return -9
        raise subprocess.TimeoutExpired(self.command, timeout)

    def kill(self):
        self.killed = True


class NativeProducerTest(unittest.TestCase):
    def test_non_linux_is_preflight_unsupported(self) -> None:
        with (
            patch("dfxlab.native_producers.linux_start_ticks", return_value="10"),
            patch("dfxlab.native_producers.platform.system", return_value="Windows"),
        ):
            result = capture_native_stack(
                implementation="pystack",
                pid=123,
                expected_start_ticks="10",
                private_output=Path("unused"),
                declared_role="engine_core",
            )
        self.assertEqual("preflight", result["outcome"]["attempt_stage"])
        self.assertEqual("unsupported", result["outcome"]["outcome_code"])
        self.assertIsNone(result["outcome"]["raw_output_sha256"])

    def test_missing_binary_is_preflight_not_permission_denied(self) -> None:
        with (
            patch("dfxlab.native_producers.linux_start_ticks", return_value="10"),
            patch("dfxlab.native_producers.platform.system", return_value="Linux"),
            patch("dfxlab.native_producers.shutil.which", return_value=None),
        ):
            result = capture_native_stack(
                implementation="pystack",
                pid=123,
                expected_start_ticks="10",
                private_output=Path("unused"),
                declared_role="engine_core",
            )
        self.assertEqual("preflight", result["outcome"]["attempt_stage"])
        self.assertEqual("binary_missing", result["outcome"]["outcome_code"])

    def test_live_identity_is_required_before_any_preflight(self) -> None:
        with (
            patch(
                "dfxlab.native_producers.linux_start_ticks",
                return_value="different",
            ),
            patch("dfxlab.native_producers.platform.system", return_value="Linux"),
        ):
            with self.assertRaisesRegex(ValueError, "identity is not live"):
                capture_native_stack(
                    implementation="pystack",
                    pid=123,
                    expected_start_ticks="10",
                    private_output=Path("unused"),
                    declared_role="engine_core",
                )

    def test_pystack_execution_is_fixed_bounded_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "pystack"
            binary.write_bytes(b"binary")
            output = root / "private" / "capture.txt"
            created = []

            def popen(command, **kwargs):
                process = _CompletedProcess(
                    command, payload=b"private native stack", **kwargs
                )
                created.append(process)
                return process

            with (
                patch(
                    "dfxlab.native_producers.linux_start_ticks",
                    side_effect=["10", "10"],
                ),
                patch("dfxlab.native_producers.platform.system", return_value="Linux"),
                patch("dfxlab.native_producers.shutil.which", return_value=str(binary)),
                patch("dfxlab.native_producers._version", return_value="1.7.1"),
                patch("dfxlab.native_producers.subprocess.Popen", side_effect=popen),
                patch(
                    "dfxlab.native_producers.time.monotonic_ns",
                    side_effect=[100, 150],
                ),
            ):
                result = capture_native_stack(
                    implementation="pystack",
                    pid=123,
                    expected_start_ticks="10",
                    private_output=output,
                    declared_role="engine_core",
                    timeout=2,
                    max_output_bytes=4096,
                )
        self.assertEqual(
            [str(binary), "remote", "--no-color", "--native-all", "123"],
            created[0].command,
        )
        self.assertEqual("execution", result["outcome"]["attempt_stage"])
        self.assertEqual("produced", result["outcome"]["outcome_code"])
        self.assertEqual(
            hashlib.sha256(b"private native stack").hexdigest(),
            result["outcome"]["raw_output_sha256"],
        )
        self.assertNotIn(str(output), str(result))

    def test_permission_is_classified_only_after_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "py-spy"
            binary.write_bytes(b"binary")
            output = root / "private" / "capture.txt"

            def popen(command, **kwargs):
                return _CompletedProcess(
                    command,
                    payload=b"Error: Permission denied while attaching",
                    return_code=1,
                    **kwargs,
                )

            with (
                patch(
                    "dfxlab.native_producers.linux_start_ticks",
                    side_effect=["10", "10"],
                ),
                patch("dfxlab.native_producers.platform.system", return_value="Linux"),
                patch("dfxlab.native_producers.shutil.which", return_value=str(binary)),
                patch("dfxlab.native_producers._version", return_value="0.4.1"),
                patch("dfxlab.native_producers.subprocess.Popen", side_effect=popen),
                patch(
                    "dfxlab.native_producers.time.monotonic_ns",
                    side_effect=[100, 150],
                ),
            ):
                result = capture_native_stack(
                    implementation="py-spy",
                    pid=123,
                    expected_start_ticks="10",
                    private_output=output,
                    declared_role="engine_core",
                )
        self.assertEqual("execution", result["outcome"]["attempt_stage"])
        self.assertEqual("permission_denied", result["outcome"]["outcome_code"])

    def test_timeout_is_an_execution_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "pystack"
            binary.write_bytes(b"binary")
            output = root / "private" / "capture.txt"

            def popen(command, **kwargs):
                return _TimedOutProcess(command, payload=b"partial", **kwargs)

            with (
                patch(
                    "dfxlab.native_producers.linux_start_ticks",
                    side_effect=["10", "10"],
                ),
                patch("dfxlab.native_producers.platform.system", return_value="Linux"),
                patch("dfxlab.native_producers.shutil.which", return_value=str(binary)),
                patch("dfxlab.native_producers._version", return_value="1.7.1"),
                patch("dfxlab.native_producers.subprocess.Popen", side_effect=popen),
                patch(
                    "dfxlab.native_producers.time.monotonic_ns",
                    side_effect=[100, 150],
                ),
            ):
                result = capture_native_stack(
                    implementation="pystack",
                    pid=123,
                    expected_start_ticks="10",
                    private_output=output,
                    declared_role="engine_core",
                )
        self.assertEqual("execution", result["outcome"]["attempt_stage"])
        self.assertEqual("timeout", result["outcome"]["outcome_code"])
        self.assertIsNotNone(result["outcome"]["raw_output_sha256"])


if __name__ == "__main__":
    unittest.main()
