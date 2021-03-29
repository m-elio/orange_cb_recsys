from unittest import TestCase
import os
import pandas as pd

from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVectorRecommender

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(THIS_DIR, "../../../contents/movies_multiple_repr")

ratings = pd.DataFrame.from_records([
    ("A000", "tt0112281", 0.99, "54654675"),
    ("A000", "tt0112453", 0, "54654675"),
    ("A000", "tt0112641", 0.44, "54654675"),
    ("A000", "tt0112760", -0.68, "54654675"),
    ("A000", "tt0112896", -0.32, "54654675"),
    ("A000", "tt0113041", 0.1, "54654675"),
    ("A000", "tt0113101", -0.87, "54654675")
], columns=["from_id", "to_id", "score", "timestamp"])


class TestItemFieldsRankingAlgorithm(TestCase):

    def test_preprocessing(self):
        # test for not existent item in candidate list
        alg = CentroidVectorRecommender({'Plot': '0'})
        self.assertEqual(len(alg.predict(ratings, 2, path, ['tt0114576', 'test_not_existent'])), 1)

    def test_exceptions(self):

        # test for not valid threshold value (range is [-1, 1])
        alg = CentroidVectorRecommender({'Plot': '0'}, threshold=2)
        self.assertEqual(len(alg.predict(ratings, 2, path)), 0)

        # test for candidate item id list with non existent items only
        alg.threshold = 0
        self.assertEqual(len(alg.predict(ratings, 2, path, ['test_not_existent'])), 0)

        # test for item_fields with a not existent field in the keys
        alg.item_fields = {'test_not_existent': '0'}
        self.assertEqual(len(alg.predict(ratings, 2, path)), 0)

        # test for item_fields with a not existent representation for a field
        alg.item_fields = {'Plot': 'test_not_existent'}
        self.assertEqual(len(alg.predict(ratings, 2, path)), 0)

        # test for item_fields with a not existent representation in the list of representations of a field
        alg.item_fields = {'Plot': ['0', 'test_not_existent']}
        self.assertEqual(len(alg.predict(ratings, 2, path)), 0)

        # test for item_fields with an empty list of representations for a field
        # also tests if the algorithm proceeds normally with an empty candidate list
        alg.item_fields = {'Plot': []}
        self.assertEqual(len(alg.predict(ratings, 2, path, [])), 0)
