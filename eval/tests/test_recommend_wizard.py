"""No Firebase or model calls: test command construction and execution gates."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('recommend_wizard', ROOT / 'wizard.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class RecommendationWizardTests(unittest.TestCase):
    def command(self, **changes):
        args = dict(user_id='recipient', group_id='', count=1, model='model',
                    agent='album-first@v1', base_url='http://localhost:8092')
        args.update(changes)
        return w.recommendation_command(**args)

    def test_default_preview_and_auto_group(self):
        command = self.command()
        self.assertIn('--dry-run', command)
        self.assertEqual(command[command.index('--group-id') + 1], '')
        self.assertNotIn('--no-spotify', command)

    def test_live_and_options(self):
        command = self.command(preview=False, enrich=True, count=3, group_id='shared')
        self.assertNotIn('--dry-run', command)
        self.assertIn('--enrich', command)
        self.assertEqual(command[command.index('--count') + 1], '3')
        with self.assertRaises(ValueError):
            self.command(preview=False, no_spotify=True)

    def test_invalid_inputs(self):
        for args in [dict(user_id=''), dict(user_id='--help'), dict(count=0)]:
            with self.assertRaises(ValueError):
                self.command(**args)

    def test_argv_preserves_literal_input(self):
        command = self.command(user_id='name;echo hello', model='a:latest')
        self.assertIn('name;echo hello', command)
        self.assertIn('a:latest', command)

    def test_print_only_and_decline_never_run(self):
        for argv in [['wizard', '--dry-run'], ['wizard']]:
            with patch.object(sys, 'argv', argv), patch.object(w.ui, 'choose', return_value=['recommend']), \
                 patch.object(w, 'recommendation_plan', return_value=self.command(preview=False)), \
                 patch.object(w.ui, 'yes', return_value=False), patch.object(w.subprocess, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 0)
                run.assert_not_called()

    def test_plan_defaults_and_auto_override(self):
        with patch.dict(w.os.environ, {'GROUP_ID': 'configured'}), \
             patch.object(w.ui, 'choose', side_effect=[['preview'], ['model'], ['album-first@v1']]), \
             patch.object(w.ui, 'ask', side_effect=['recipient', 'auto', 'http://localhost:8092', '']), \
             patch.object(w.ui, 'positive', return_value=1), patch.object(w.ui, 'discover_models', return_value=[]), \
             patch.object(w.ui, 'yes', side_effect=[False, True]), contextlib.redirect_stdout(io.StringIO()):
            command = w.recommendation_plan()
            self.assertIn('--dry-run', command)
            self.assertEqual(command[command.index('--group-id') + 1], '')


if __name__ == '__main__':
    unittest.main()
