"""Offline launcher contracts: cardinality, CLI compatibility and safe execution."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wizard as w


class WizardTests(unittest.TestCase):
    def build(self, action, **overrides):
        args = dict(profiles=['jazzy_hiphop'], models=['model'], agents=['album-first@v1'], base_url='http://localhost:8092')
        args.update(overrides)
        return w.build_commands(action, **args)

    def test_cardinality(self):
        for action, overrides in [('label', dict(models=['a', 'b'])), ('fast', dict(profiles=['a', 'b'])),
                                  ('duel', {}), ('agents', {}), ('models', dict(models=['a', 'b']))]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.build(action, **overrides)

    def test_profile_encodings(self):
        with self.assertRaises(ValueError):
            self.build('eval', profiles=['a', 'b'])
        command = self.build('eval', profiles=[], all_profiles=True)[0]
        self.assertNotIn('--profile', command)
        self.assertEqual(len(self.build('eval', profiles=[], all_profiles=True)), 1)
        command = self.build('score', profiles=['a', 'b'])[0]
        self.assertEqual(command.count('--profile'), 2)
        command = self.build('agents', profiles=['a', 'b'], agents=['album-first@v1', 'artist-then-album@v1'])[0]
        self.assertEqual(command[command.index('--profiles') + 1], 'a,b')

    def test_contestants(self):
        command = self.build('models', models=['a', 'b'], agents=['album-first@v1', 'artist-then-album@v1'])[0]
        self.assertEqual(command[-1], 'a:album-first@v1,b:artist-then-album@v1')
        with self.assertRaises(ValueError):
            self.build('models', models=['a:latest', 'b'], agents=['album-first@v1'] * 2)

    def test_shell_characters_are_one_argument(self):
        name = 'model; echo surprise'
        command = self.build('eval', models=[name])[0]
        self.assertEqual(command[command.index('--models') + 1], name)

    def test_checkbox_validation_and_toggle(self):
        with patch('builtins.input', side_effect=['0', 'none', '', '1,2', '']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.choose('Test', [('a', 'A'), ('b', 'B')], minimum=2), ['a', 'b'])

    def test_default_single_choice(self):
        with patch('builtins.input', side_effect=['2', '']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.choose('Test', [('a', 'A'), ('b', 'B')], ['a'], maximum=1), ['b'])

    def test_discovery_backends(self):
        for backend, payload, endpoint in [('ollama', '{"models":[{"name":"a:latest"}]}', '/api/tags'),
                                            ('llamacpp', '{"data":[{"id":"a:latest"}]}', '/v1/models')]:
            with patch.object(w, 'urlopen', return_value=io.StringIO(payload)) as fetch:
                self.assertEqual(w.discover_models('http://localhost:8092/', backend), ['a:latest'])
                self.assertEqual(fetch.call_args.args[0], 'http://localhost:8092' + endpoint)

    def test_every_generated_flag_is_supported_by_runner(self):
        # Check advertised runner options without generation or writes.
        # site.py has no parser: invoking it always writes the website.
        for action, _ in w.ACTIONS:
            if action == "site":
                self.assertEqual(self.build(action), [[sys.executable, str(w.ROOT / "eval/site.py")]])
                continue
            overrides = {}
            if action in {'duel', 'models'}:
                overrides['models'] = ['a', 'b']
            if action in {'agents', 'models'}:
                overrides['agents'] = ['album-first@v1', 'artist-then-album@v1']
            for command in self.build(action, **overrides):
                result = subprocess.run([*command, '--help'], capture_output=True, text=True, cwd=w.ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)
                for arg in command[2:]:
                    if arg.startswith("--"):
                        self.assertIn(arg, result.stdout)

    def test_dry_run_and_decline_never_execute(self):
        for argv in [['wizard', '--dry-run'], ['wizard']]:
            with patch.object(sys, 'argv', argv), patch.object(w, 'plan', return_value=[['fake']]), \
                 patch.object(w, 'yes', return_value=False), patch.object(w.subprocess, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 0)
                run.assert_not_called()

    def test_failures_stop_queue(self):
        with patch.object(sys, 'argv', ['wizard']), patch.object(w, 'plan', return_value=[['one'], ['two']]), \
             patch.object(w, 'yes', return_value=True), \
             patch.object(w.subprocess, 'run', return_value=subprocess.CompletedProcess(['one'], 7)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.main(), 7)
            run.assert_called_once_with(['one'], cwd=w.ROOT)


if __name__ == '__main__':
    unittest.main()
