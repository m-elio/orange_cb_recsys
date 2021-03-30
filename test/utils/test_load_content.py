import os
from unittest import TestCase
from orange_cb_recsys.utils.load_content import load_content_instance, remove_not_existent_items
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
contents_path = os.path.join(THIS_DIR, "../../contents/movielens_test1591885241.5520566")


class Test(TestCase):
    def test_load_content_instance(self):
        try:
            content = load_content_instance(contents_path, "tt0112281")
            self.assertEqual(content.content_id, "tt0112281")
            content = load_content_instance("aaa", "1")
            self.assertEqual(content, None)

        except FileNotFoundError:
            self.fail("File not found!")

    def test_remove_not_existent_items(self):
        ratings = pd.DataFrame({'to_id': ['tt0112281', 'aaaa']})
        valid_ratings = remove_not_existent_items(ratings, contents_path)
        self.assertIn('tt0112281', valid_ratings['to_id'].values)
        self.assertNotIn('aaaa', valid_ratings['to_id'].values)
