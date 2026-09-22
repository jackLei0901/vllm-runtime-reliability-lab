import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "native-evidence-capability"
    / "stage_b_pair_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("stage_b_pair_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class StageBPairAdapterTest(unittest.TestCase):
    def test_accepts_only_the_frozen_campaign_shape(self) -> None:
        self.assertEqual(adapter.parse_campaign_invocation(["dump", "--pid", "17"]), 17)
        for value in (
            [],
            ["dump"],
            ["dump", "-p", "17"],
            ["dump", "--pid", "0"],
            ["dump", "--pid", "not-a-pid"],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.parse_campaign_invocation(value)

    def test_environment_shape_is_closed_and_required(self) -> None:
        values = {name: name.lower() for name in adapter.REQUIRED_ENV}
        self.assertEqual(adapter.required_environment(values), values)
        values.pop("DFX_STAGE_B_OUTPUT")
        with self.assertRaisesRegex(ValueError, "DFX_STAGE_B_OUTPUT"):
            adapter.required_environment(values)

    @mock.patch.object(adapter, "linux_start_ticks", return_value=None)
    def test_dead_target_fails_before_output_creation(self, _identity: mock.Mock) -> None:
        values = {name: name.lower() for name in adapter.REQUIRED_ENV}
        with mock.patch.dict(os.environ, values, clear=True):
            self.assertEqual(adapter.main(["dump", "--pid", "17"]), 1)


if __name__ == "__main__":
    unittest.main()
