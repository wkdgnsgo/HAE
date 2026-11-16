"""W&B bookkeeping tests: all SDK calls use a fake, with no online access."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hae.checkpoints import capture_rng_state, seed_all
from hae.tracking import WandbTracker


def touch_rng():
    random.random()
    np.random.random()
    torch.rand(1)


class FakeRun:
    def __init__(self, sdk, kwargs):
        self.sdk = sdk
        self.id = kwargs['id']
        self.project = kwargs['project']
        self.entity = kwargs['entity'] or 'resolved-account'
        self.group = kwargs['group']
        self.metrics = []
        self.logs = []
        self.finishes = []

    def define_metric(self, name, **kwargs):
        touch_rng()
        self.metrics.append((name, kwargs))

    def log(self, data, **kwargs):
        touch_rng()
        if self.sdk.fail == 'log':
            raise OSError('simulated transport failure')
        self.logs.append((data, kwargs))

    def finish(self, **kwargs):
        touch_rng()
        self.finishes.append(kwargs)
        if self.sdk.fail == 'finish':
            raise OSError('simulated finish failure')


class FakeSDK:
    Settings = SimpleNamespace

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.runs = []

    def init(self, **kwargs):
        touch_rng()
        self.calls.append(kwargs)
        if self.fail == 'init':
            raise OSError('simulated init failure')
        run = FakeRun(self, kwargs)
        self.runs.append(run)
        return run

    def Image(self, data, **kwargs):
        touch_rng()
        return SimpleNamespace(data=data, **kwargs)


def config(mode='online'):
    return {'stage': 2, 'logging': {'wandb': {
        'enabled': True, 'mode': mode, 'project': 'test-project', 'entity': None,
        'group': 'stage1', 'name': None, 'image_every': 50, 'num_images': 10,
    }}}


def identity(run_id='saved123'):
    return {'format_version': 1, 'id': run_id, 'project': 'original-project',
            'entity': 'original-account', 'group': 'original-group',
            'mode': 'online', 'last_logged_update': 10}


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sdk = FakeSDK()
        self.importer = patch('hae.tracking.import_module', return_value=self.sdk)
        self.mock_import = self.importer.start()
        self.credentials = patch('hae.tracking._has_online_credentials', return_value=True)
        self.credentials.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.importer.stop)
        self.addCleanup(self.credentials.stop)

    def assert_rng_equal(self, before):
        after = capture_rng_state()
        self.assertEqual(before['python'], after['python'])
        self.assertEqual(before['numpy'], after['numpy'])
        self.assertTrue(torch.equal(before['torch_cpu'], after['torch_cpu']))
        if before['torch_cuda'] is None:
            self.assertIsNone(after['torch_cuda'])
        else:
            for name in ('device_count', 'device_index'):
                self.assertEqual(before['torch_cuda'][name], after['torch_cuda'][name])
            self.assertTrue(torch.equal(before['torch_cuda']['state'], after['torch_cuda']['state']))

    def test_disabled_missing_config_and_nonzero_rank_never_import_sdk(self):
        disabled = config()
        disabled['logging']['wandb']['enabled'] = False
        for number, (cfg, rank) in enumerate(((disabled, 0), ({'stage': 1}, 0),
                                             (config(), 1), (config('disabled'), 0))):
            directory = self.root / str(number)
            tracker = WandbTracker(cfg, directory, rank=rank)
            self.assertFalse(tracker.enabled)
            self.assertIsNone(tracker.state_dict())
            tracker.log({'stage_update': 1})
            tracker.finish()
            self.assertFalse(directory.exists())
        self.mock_import.assert_not_called()

    def test_update_axis_is_optimizer_step_not_microbatch_count(self):
        tracker = WandbTracker(config(), self.root)
        record = {'stage_update': 11, 'generator_updates': 11, 'discriminator_updates': 11,
                  'microbatches': 88, 'samples_seen': 11264, 'loss_g': 0.2, 'lr_g': 5e-5}
        tracker.log(record)
        data, kwargs = self.sdk.runs[0].logs[0]
        self.assertEqual(kwargs, {'step': 11, 'commit': True})
        self.assertEqual(data['train/update'], 11)
        self.assertEqual(data['train/microbatches'], 88)
        self.assertEqual(data['train/loss_g'], 0.2)
        self.assertEqual(len(self.sdk.runs[0].logs), 1)
        self.assertIn(('train/*', {'step_metric': 'train/update'}), self.sdk.runs[0].metrics)
        self.assertEqual(tracker.image_every, 50)
        self.assertEqual(tracker.num_images, 10)
        self.assertEqual(json.loads((self.root / 'wandb_run.json').read_text()), tracker.state_dict())
        self.assertEqual(tracker.state_dict()['last_logged_update'], 11)
        tracker.finish()
        self.assertEqual(json.loads((self.root / 'wandb_status.json').read_text())['status'], 'finished')

    def test_checkpoint_identity_precedes_local_file_and_current_config(self):
        (self.root / 'wandb_run.json').write_text(json.dumps(identity('local123')))
        tracker = WandbTracker(config(), self.root, resume=True, checkpoint_state=identity('check123'))
        call = self.sdk.calls[0]
        self.assertEqual(call['id'], 'check123')
        self.assertEqual(call['project'], 'original-project')
        self.assertEqual(call['entity'], 'original-account')
        self.assertEqual(call['group'], 'original-group')
        self.assertEqual(call['resume'], 'must')
        self.assertEqual(tracker.state_dict()['last_logged_update'], 10)

    def test_resume_local_identity_fallback_and_legacy_checkpoint_new_run(self):
        (self.root / 'wandb_run.json').write_text(json.dumps(identity('local123')))
        tracker = WandbTracker(config(), self.root, resume=True)
        self.assertEqual(tracker.state_dict()['id'], 'local123')
        self.assertEqual(self.sdk.calls[-1]['resume'], 'must')
        fresh_dir = self.root / 'legacy'
        fresh = WandbTracker(config(), fresh_dir, resume=True)
        self.assertNotEqual(fresh.state_dict()['id'], 'local123')
        self.assertEqual(self.sdk.calls[-1]['resume'], 'never')
        self.assertTrue(json.loads((fresh_dir / 'wandb_status.json').read_text())['resumed_without_identity'])

    def test_new_phase_ignores_checkpoint_local_and_environment_run_ids(self):
        (self.root / 'wandb_run.json').write_text(json.dumps(identity('local123')))
        with patch.dict('os.environ', {'WANDB_RUN_ID': 'env123', 'WANDB_RESUME': 'must'}):
            first = WandbTracker(config(), self.root, checkpoint_state=identity('check123'))
            second = WandbTracker(config(), self.root / 'stage3')
        first_id, second_id = first.state_dict()['id'], second.state_dict()['id']
        self.assertNotIn(first_id, ('local123', 'check123', 'env123'))
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(self.sdk.calls[0]['resume'], 'never')
        self.assertEqual(self.sdk.calls[0]['reinit'], 'create_new')

    def test_offline_resume_keeps_identity_without_claiming_sdk_resume(self):
        tracker = WandbTracker(config('offline'), self.root, resume=True, checkpoint_state=identity())
        self.assertEqual(tracker.state_dict()['id'], 'saved123')
        self.assertEqual(self.sdk.calls[-1]['mode'], 'offline')
        self.assertEqual(self.sdk.calls[-1]['resume'], 'never')
        status = json.loads((self.root / 'wandb_status.json').read_text())
        self.assertTrue(status['offline_segment'])

    def test_resume_skips_already_logged_updates_without_disabling(self):
        tracker = WandbTracker(config(), self.root, resume=True, checkpoint_state=identity())
        tracker.log({'stage_update': 9})
        tracker.log({'stage_update': 10})
        self.assertTrue(tracker.enabled)
        self.assertEqual(self.sdk.runs[0].logs, [])
        tracker.log({'stage_update': 11})
        self.assertEqual(len(self.sdk.runs[0].logs), 1)
        self.assertEqual(self.sdk.runs[0].logs[0][1]['step'], 11)

    def test_ema_grid_metrics_and_caption_share_one_log_event(self):
        tracker = WandbTracker(config(), self.root)
        grid = np.zeros((32, 64, 3), dtype=np.uint8)
        tracker.log({'stage_update': 50, 'microbatches': 400}, preview={
            'images': grid, 'psnr_db': 26.7, 'image_count': 2,
            'paths': ['a/image.jpg', 'b/image.jpg'], 'weights': 'ema'})
        data, kwargs = self.sdk.runs[0].logs[0]
        self.assertIs(data['Reconstructions_EMA'].data, grid)
        self.assertIn('a/image.jpg', data['Reconstructions_EMA'].caption)
        self.assertEqual(data['preview/psnr_db'], 26.7)
        self.assertEqual(data['preview/count'], 2)
        self.assertEqual(kwargs['step'], 50)
        self.assertEqual(len(self.sdk.runs[0].logs), 1)

    def test_sdk_init_log_image_and_finish_preserve_training_rng(self):
        seed_all(441)
        before = capture_rng_state()
        tracker = WandbTracker(config(), self.root)
        self.assert_rng_equal(before)
        tracker.log({'stage_update': 1}, preview={'images': np.zeros((4, 4, 3), dtype=np.uint8),
                    'psnr_db': 25, 'image_count': 1, 'paths': ['sample.png'], 'weights': 'ema'})
        self.assert_rng_equal(before)
        tracker.finish()
        self.assert_rng_equal(before)

    def test_init_failure_writes_status_and_does_not_raise_or_change_rng(self):
        self.sdk.fail = 'init'
        before = capture_rng_state()
        with self.assertWarnsRegex(RuntimeWarning, 'training continues'):
            tracker = WandbTracker(config(), self.root)
        self.assertFalse(tracker.enabled)
        self.assert_rng_equal(before)
        self.assertEqual(json.loads((self.root / 'wandb_status.json').read_text())['operation'], 'init')
        tracker.log({'stage_update': 1})
        tracker.finish()

    def test_log_failure_disables_and_keeps_identity_for_resume(self):
        tracker = WandbTracker(config(), self.root)
        self.sdk.fail = 'log'
        before = capture_rng_state()
        with self.assertWarnsRegex(RuntimeWarning, 'tracking disabled'):
            tracker.log({'stage_update': 1})
        self.assertFalse(tracker.enabled)
        self.assertIsNotNone(tracker.state_dict())
        self.assert_rng_equal(before)
        self.assertEqual(self.sdk.runs[0].finishes, [{'exit_code': 1}])
        tracker.log({'stage_update': 2})
        self.assertEqual(self.sdk.runs[0].logs, [])

    def test_finish_failure_is_nonfatal_and_preserves_rng(self):
        tracker = WandbTracker(config(), self.root)
        self.sdk.fail = 'finish'
        before = capture_rng_state()
        with self.assertWarnsRegex(RuntimeWarning, 'tracking disabled'):
            tracker.finish()
        self.assertFalse(tracker.enabled)
        self.assert_rng_equal(before)
        self.assertEqual(json.loads((self.root / 'wandb_status.json').read_text())['operation'], 'finish')

    def test_missing_credentials_fail_without_sdk_import_or_prompt(self):
        with patch('hae.tracking._has_online_credentials', return_value=False):
            with self.assertWarnsRegex(RuntimeWarning, 'No W&B credentials'):
                tracker = WandbTracker(config(), self.root)
        self.assertFalse(tracker.enabled)
        self.mock_import.assert_not_called()

    def test_identity_snapshot_is_json_safe_and_independent(self):
        tracker = WandbTracker(config(), self.root)
        snapshot = tracker.state_dict()
        json.dumps(snapshot, allow_nan=False)
        snapshot['id'] = 'modified'
        self.assertNotEqual(tracker.state_dict()['id'], 'modified')
        self.assertLessEqual(self.sdk.calls[0]['settings'].init_timeout, 30)
        self.assertFalse(self.sdk.calls[0]['save_code'])
        self.assertTrue(self.sdk.calls[0]['settings'].disable_git)


if __name__ == '__main__':
    unittest.main()
