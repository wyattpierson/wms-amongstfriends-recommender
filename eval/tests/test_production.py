"""Production contract tests: real pipeline and payloads; fake network boundaries."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import recommend
from afrec import firebase, pipeline
from afrec.llm import BaseLLM, GenerationMeta


class LocalModel(BaseLLM):
    name = 'test-local'

    def generate(self, prompt, **kwargs):
        return json.dumps([
            {'artist': 'New Artist', 'album': 'New Album', 'reason': 'Warm drums', 'confidence': .9},
            {'artist': 'Another Artist', 'album': 'Another Album', 'reason': 'Soulful', 'confidence': .8},
        ]), GenerationMeta(name=self.name)


def document(name, members):
    doc = Mock(id=name, exists=True)
    doc.to_dict.return_value = {'members': members}
    return doc


class ProductionTests(unittest.TestCase):
    def run_command(self, flags=(), *, verifier=None, response=None, group_error=None):
        raw = [{'catalog': {'type': 'album', 'metadata': {
            'artistName': f'Old Artist {i}', 'albumName': f'Old Album {i}'}},
            'rating': 5, 'text': 'Warm grooves'} for i in range(3)]
        def remote(name, payload, token):
            if name == 'get_user_reviews':
                return {'success': True, 'reviews': raw}
            return response if response is not None else {'success': True, 'recommendationId': 'rec-1'}
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(firebase, 'initialize'))
            stack.enter_context(patch.object(firebase, 'get_firebase_id_token', return_value='bot-token'))
            stack.enter_context(patch.object(firebase, 'resolve_group', return_value='shared-group',
                                            side_effect=group_error))
            call = stack.enter_context(patch.object(firebase, 'call_function', side_effect=remote))
            stack.enter_context(patch.object(recommend, 'make_llm', return_value=LocalModel()))
            stack.enter_context(patch.object(recommend.spotify, 'get_token', return_value='spotify-token'))
            stack.enter_context(patch.object(recommend.spotify, 'verifier_for', return_value=verifier or (
                lambda a, b: (True, b, 'https://open.spotify.com/album/test'))))
            recommend.main(['recipient', '--agent', 'album-first@v1', *flags])
            return call.call_args_list

    def test_fetch_generate_verify_and_send_one_as_bot(self):
        calls = self.run_command()
        self.assertEqual(calls[0].args, ('get_user_reviews', {'userId': 'recipient'}, 'bot-token'))
        self.assertEqual(len(calls), 2)
        name, payload, token = calls[1].args
        self.assertEqual(name, 'submit_recommendation')
        self.assertEqual(payload, {'fromUserId': firebase.BOT_USER_ID, 'toUserId': 'recipient',
            'groupId': 'shared-group', 'message': 'New Artist — New Album\n\nWarm drums',
            'link': 'https://open.spotify.com/album/test'})
        self.assertEqual(token, 'bot-token')

    def test_dry_run_never_submits(self):
        self.assertEqual(len(self.run_command(['--dry-run'])), 1)

    def test_dry_run_can_preview_without_shared_group(self):
        calls = self.run_command(['--dry-run'], group_error=firebase.FirebaseError('No shared group'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0], 'get_user_reviews')

    def test_live_run_stops_before_generation_without_shared_group(self):
        with patch.object(LocalModel, 'generate') as generate:
            with self.assertRaisesRegex(firebase.FirebaseError, 'No shared group'):
                self.run_command(group_error=firebase.FirebaseError('No shared group'))
            generate.assert_not_called()

    def test_count(self):
        self.assertEqual(len(self.run_command(['--count', '2'])), 3)

    def test_delivery_chooses_highest_confidence_unreviewed_album(self):
        outcome = pipeline.RunOutcome(ok=True, recs=[
            pipeline.Rec('Low Artist', 'Low Album', .6, verified=True),
            pipeline.Rec('Old Artist 0', 'Alternate title', .99, verified=True,
                         canonical_album='Old Album 0'),
            pipeline.Rec('Best Artist', 'Best Album', .95, verified=True),
        ])
        with patch.object(recommend.agents, 'resolve_agent') as resolve:
            resolve.return_value.run.return_value = outcome
            calls = self.run_command()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].args[1]['message'], 'Best Artist — Best Album')

    def test_only_reviewed_candidates_never_submit(self):
        outcome = pipeline.RunOutcome(ok=True, recs=[
            pipeline.Rec('OLD ARTIST 0', 'Old Album 0!', .99, verified=True),
        ])
        with patch.object(recommend.agents, 'resolve_agent') as resolve:
            resolve.return_value.run.return_value = outcome
            with self.assertRaisesRegex(firebase.FirebaseError, 'No unreviewed'):
                self.run_command()

    def test_selection_normalizes_names_and_skips_invalid_confidence(self):
        recs = [pipeline.Rec('Beyoncé', 'Déjà Vu!', .99),
                pipeline.Rec('Other', 'Invalid', float('nan')),
                pipeline.Rec('Other', 'Too high', 99),
                pipeline.Rec('New', 'Album', .8),
                pipeline.Rec('NEW', 'Album!', .9)]
        selected = recommend.select_delivery(
            recs, [{'artist': 'Beyonce', 'album': 'Deja Vu'}], 2)
        self.assertEqual(selected, [recs[-1]])

    def test_reviews_follow_all_pages(self):
        with patch.object(firebase, 'call_function', side_effect=[
            {'reviews': [{'reviewId': 'first'}], 'nextCursor': 'page-2'},
            {'reviews': [{'reviewId': 'older'}], 'nextCursor': None},
        ]) as call:
            self.assertEqual(firebase.get_user_reviews('recipient', 'token'),
                             [{'reviewId': 'first'}, {'reviewId': 'older'}])
        self.assertEqual(call.call_args_list[1].args,
                         ('get_user_reviews', {'userId': 'recipient', 'cursor': 'page-2'}, 'token'))

    def test_repeated_review_cursor_fails_closed(self):
        with patch.object(firebase, 'call_function', return_value={
            'reviews': [], 'nextCursor': 'same',
        }):
            with self.assertRaisesRegex(firebase.FirebaseError, 'pagination cursor'):
                firebase.get_user_reviews('recipient', 'token')

    def test_verification_outage_never_submits(self):
        def unavailable(*args):
            raise RuntimeError('catalog unavailable')
        with self.assertRaisesRegex(firebase.FirebaseError, 'Verification failed'):
            self.run_command(verifier=unavailable)

    def test_false_submission_is_failure(self):
        with self.assertRaisesRegex(firebase.FirebaseError, 'Only 0/1'):
            self.run_command(response={'success': False})

    def test_failed_generation_exits_nonzero(self):
        with patch.object(pipeline, '_run_stage', return_value=pipeline.StageResult('albums_first', False)):
            with self.assertRaises(SystemExit) as raised:
                self.run_command()
            self.assertEqual(raised.exception.code, 1)

    def test_group_preflight(self):
        with patch.object(firebase, 'initialize'), patch.object(firebase.firestore, 'client') as client:
            collection = client.return_value.collection.return_value
            query = collection.where.return_value
            query.get.return_value = [document('shared', [firebase.BOT_USER_ID, 'recipient'])]
            self.assertEqual(firebase.resolve_group('recipient'), 'shared')
            query.get.return_value.append(document('second', [firebase.BOT_USER_ID, 'recipient']))
            with self.assertRaisesRegex(firebase.FirebaseError, 'Multiple shared'):
                firebase.resolve_group('recipient')
            collection.document.return_value.get.return_value = document('wrong', ['recipient'])
            with self.assertRaisesRegex(firebase.FirebaseError, 'No matching group'):
                firebase.resolve_group('recipient', 'wrong')

    def test_callable_protocol(self):
        with patch.object(firebase.requests, 'post') as post:
            post.return_value.ok = True
            for envelope in ('result', 'data'):
                post.return_value.json.return_value = {envelope: {'success': True}}
                self.assertEqual(firebase.call_function('test', {}, 'token'), {'success': True})
            post.return_value.json.return_value = {'error': {'message': 'denied'}}
            with self.assertRaises(firebase.FirebaseError):
                firebase.call_function('test', {}, 'token')


if __name__ == '__main__':
    unittest.main()
