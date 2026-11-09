"""Offline failure/retry/resume regressions; never touch real judging data."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duels as d


class RecoveryTests(unittest.TestCase):
    def test_spotify_failures_are_recoverable_before_generation(self):
        for error in [d.requests.ConnectionError('DNS failed'), d.requests.Timeout(),
                      d.ev.spotify.SpotifyConfigError('401')]:
            with patch.object(d.ev.spotify, 'get_token', side_effect=error), \
                 patch.object(d.t, '_generate') as generate, patch.object(d.he, 'append_duel') as append:
                with self.assertRaisesRegex(d.MatchupError, 'Spotify authentication failed'):
                    d.run_matchup({'name': 'p'}, [], 'http://unused', 0.7, True)
                generate.assert_not_called()
                append.assert_not_called()

    def run_case(self, answers, outcomes):
        args = SimpleNamespace(models='m', profiles='a,b', agent_a='album-first@v1',
                               agent_b='artist-then-album@v1', enrich=False, session='recovery',
                               dry_run=False, unload_wait=0, base_url='http://unused',
                               temperature=0.7, no_spotify=False, grade_winner=False, enricher=None)
        conts = [d.Contestant('m', 'm', 'album-first@v1')]
        matchups = [({'name': name}, conts) for name in ['a', 'b']]
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'SESSIONS_DIR', Path(tmp)), \
             patch.object(d, '_big_model_guard', return_value=True), patch.object(d, '_pause', return_value='next'), \
             patch.object(d, '_ask_line', side_effect=answers), \
             patch.object(d, 'run_matchup', side_effect=outcomes) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            d.run_session('agents', matchups, args)
            calls = [call.args[0]['name'] for call in run.call_args_list]
            state = d.load_state('recovery')
            # Resume only the unfinished matchups, without touching saved ones.
            with patch.object(d, 'run_matchup', return_value='resumed') as resumed:
                d.run_session('agents', matchups, args)
            return calls, state, [call.args[0]['name'] for call in resumed.call_args_list]

    def test_final_duel_retries_in_place(self):
        calls, state, resumed = self.run_case(['r'], ['ok', d.MatchupError('offline'), 'ok'])
        self.assertEqual(calls, ['a', 'b', 'b'])
        self.assertEqual([m['status'] for m in state['matchups']], ['done', 'done'])
        self.assertEqual(resumed, [])

    def test_quit_and_eof_preserve_completed_progress(self):
        for answer in ['q', None]:
            calls, state, resumed = self.run_case([answer], ['ok', d.MatchupError('offline')])
            self.assertEqual(len(state['matchups']), 1)
            self.assertEqual(resumed, ['b'])

    def test_invalid_answer_does_not_advance(self):
        calls, state, resumed = self.run_case(['oops', 'retry'], [d.MatchupError('offline'), 'ok', 'ok'])
        self.assertEqual(calls, ['a', 'a', 'b'])
        self.assertEqual(resumed, [])

    def test_skip_is_explicit_and_persisted(self):
        calls, state, resumed = self.run_case(['skip'], ['ok', d.MatchupError('offline')])
        self.assertEqual(state['matchups'][-1]['status'], 'skipped')
        self.assertEqual(resumed, [])


if __name__ == '__main__':
    unittest.main()
