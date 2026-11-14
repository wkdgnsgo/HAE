"""Run real multi-process distributed checks on CPU without external data."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from hae import distributed


ROOT = Path(__file__).resolve().parents[1]


class DistributedTests(unittest.TestCase):
    def test_multi_rank_rejects_explicit_cuda_device_before_initialization(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "1", "LOCAL_RANK": "1"}), \
                mock.patch("torch.cuda.set_device") as set_device, \
                mock.patch("torch.distributed.init_process_group") as init_process_group:
            for device in ("cuda:0", "cuda:1"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "LOCAL_RANK"):
                    distributed.initialize(device)
            set_device.assert_not_called()
            init_process_group.assert_not_called()

    def test_two_rank_accumulation_matches_global_batch(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        with tempfile.TemporaryDirectory() as output:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc-per-node=2", "--local-addr=127.0.0.1",
                       str(ROOT / "tests/smoke_distributed.py"), "--output-dir", output]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                self.fail("Two-process distributed smoke timed out.\n" + stdout + "\n" + stderr)
            self.assertEqual(process.returncode, 0, stdout + "\n" + stderr)
            report = json.loads((Path(output) / "result.json").read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(report["rank_weights_identical"])
            self.assertTrue(report["broadcast_parameters_and_buffers"])
            self.assertTrue(report["rank_rng_roundtrip"])
            self.assertEqual(report["world_size"], 2)
            for rank in report["records"]:
                self.assertEqual(rank["progress"]["generator_updates"], 3)
                self.assertEqual(rank["progress"]["samples_seen"], 24)
                self.assertLess(rank["max_error_vs_global_batch"], 2e-7)


if __name__ == "__main__":
    unittest.main()
