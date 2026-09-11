"""Keep the formal entry on G1 without starting GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'main.py'
spec = importlib.util.spec_from_file_location('ours21_formal_entry', PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class FormalEntryTests(unittest.TestCase):
    def test_reject_other_methods_and_empty_jobs(self):
        for jobs in ([], [{}], [dict(kind='candidate', group='G2', state_mode='cnn_G2')],
                     [dict(kind='candidate', group='G1', state_mode='sea7')]):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                entry.validate_jobs(jobs)

    def test_train_routes_only_g1(self):
        with patch.object(entry.sys, 'argv', ['main.py', 'train']), \
                patch.object(entry.subprocess, 'run') as run:
            entry.main()
        self.assertEqual(run.call_args.args[0],
            [entry.sys.executable, str(entry.TRAIN / 'pipeline.py'), 'train_group', '--index', '0'])
        self.assertTrue(run.call_args.kwargs['check'])

    def test_generate_preserves_frozen_jobs_and_rejects_mixed_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'jobs.json'
            jobs = [dict(kind='candidate', group='G1', state_mode='cnn_G1')]
            path.write_text(json.dumps(jobs))
            with patch.object(entry.sys, 'argv', ['main.py', 'generate', '--jobs', str(path)]), \
                    patch.object(entry.subprocess, 'run') as run:
                entry.main()
                self.assertEqual(run.call_args.args[0], [entry.sys.executable,
                    str(entry.INFERENCE / 'generate_worker.py'), '--jobs', str(path.resolve())])
                self.assertEqual(json.loads(path.read_text()), jobs)
                run.reset_mock()
                path.write_text(json.dumps(jobs + [dict(kind='candidate', group='G4', state_mode='cnn_G4')]))
                with self.assertRaises(ValueError):
                    entry.main()
                run.assert_not_called()
