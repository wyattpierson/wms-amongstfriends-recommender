"""Regressions for inconsistent Last.fm tag response shapes; no network."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afrec import lastfm


class LastfmTests(unittest.TestCase):
    def test_empty_or_malformed_tags(self):
        for node in (None, '', 'missing', {}, {'tag': ''}, {'tag': ['bad', None]}):
            with self.subTest(node=node):
                self.assertEqual(lastfm._tags(node, limit=3), [])

    def test_single_tag_and_list(self):
        tag = {'name': 'soul', 'count': '10'}
        self.assertEqual(lastfm._tags({'tag': tag}, limit=3), [('soul', 10)])
        self.assertEqual(lastfm._tags({'tag': [tag, {'name': 'bad', 'count': 'oops'}]}, limit=3),
                         [('soul', 10)])

    def test_album_facts_survive_empty_tags(self):
        with patch.object(lastfm, '_call', return_value={'album': {
            'name': 'Album', 'artist': 'Artist', 'releaseDate': '2001-01-01',
            'tags': '', 'toptags': ''}}):
            album = lastfm.album_info('Artist', 'Album')
        self.assertEqual(album['year'], 2001)
        self.assertEqual(album['tags'], [])


if __name__ == '__main__':
    unittest.main()
