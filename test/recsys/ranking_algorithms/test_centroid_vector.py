import os
from unittest import TestCase

from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVectorRecommender

import pandas as pd

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


class TestCentroidVector(TestCase):

    def test_predict(self):
        alg = CentroidVectorRecommender({'Plot': ['0']}, threshold=0)
        result = alg.predict(ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['1']}, threshold=0)
        result = alg.predict(ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['0', '1'], 'Genre': ['0', '1'], 'Director': ['1']}, threshold=0)
        result = alg.predict(ratings=ratings, recs_number=1, items_directory=path,
                             candidate_item_id_list=['tt0114319'])
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['0', '1'], 'Genre': ['0', '1'], 'Director': ['1']}, threshold=0)
        result = alg.predict(ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

    def test_exceptions(self):
        # test for empty ratings
        empty_ratings = pd.DataFrame.from_records([], columns=["from_id", "to_id", "score", "timestamp"])
        t_centroid = CentroidVectorRecommender({'Plot': '1'})
        result = t_centroid.predict(empty_ratings, 2, path)
        self.assertEqual(len(result), 0)

        # test for negative ratings
        negative_ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -0.99, "54654675"),
            ("A000", "tt0112453", -0.22, "54654675"),
            ("A000", "tt0112641", -0.44, "54654675")
        ], columns=["from_id", "to_id", "score", "timestamp"])
        t_centroid = CentroidVectorRecommender({'Plot': '1'}, threshold=0)
        result = t_centroid.predict(negative_ratings, 2, path)
        self.assertEqual(len(result), 0)

        # test for embedding technique with granularity word
        path_word_gran = os.path.join(THIS_DIR, "../../../contents/movies_plot_1_word_granularity")
        t_centroid = CentroidVectorRecommender({'Plot': '1'})
        result = t_centroid.predict(ratings, 2, path_word_gran)
        self.assertEqual(len(result), 0)
