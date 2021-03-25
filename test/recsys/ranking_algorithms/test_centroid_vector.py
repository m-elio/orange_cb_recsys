import lzma
import os
import pickle
from unittest import TestCase

from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVectorRecommender

import pandas as pd


class TestCentroidVector(TestCase):
    def test_predict(self):
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 0.99, "54654675"),
            ("A000", "tt0112453", 0, "54654675"),
            ("A000", "tt0112641", 0.44, "54654675"),
            ("A000", "tt0112760", -0.68, "54654675"),
            ("A000", "tt0112896", -0.32, "54654675"),
            ("A000", "tt0113041", 0.1, "54654675"),
            ("A000", "tt0113101", -0.87, "54654675")
        ], columns=["from_id", "to_id", "score", "timestamp"])

        path = "../../../contents/movies_multiple_repr"
        items = []
        try:
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))
        except FileNotFoundError:
            path = "contents/movies_multiple_repr"
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))

        alg = CentroidVectorRecommender({'Plot': ['0']}, threshold=0)
        result = alg.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['1']}, threshold=0)
        result = alg.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['0', '1'], 'Genre': ['0', '1'], 'Director': ['1']}, threshold=0)
        result = alg.predict('A000', ratings=ratings, recs_number=1, items_directory=path,
                             candidate_item_id_list=['tt0114319'])
        self.assertGreater(result.rating[0], 0)

        alg = CentroidVectorRecommender({'Plot': ['0', '1'], 'Genre': ['0', '1'], 'Director': ['1']}, threshold=0)
        result = alg.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        self.assertGreater(result.rating[0], 0)

    def test_exceptions(self):
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 0.99, "54654675"),
            ("A000", "tt0112453", 0, "54654675"),
            ("A000", "tt0112641", 0.44, "54654675"),
            ("A000", "tt0112760", -0.68, "54654675"),
            ("A000", "tt0112896", -0.32, "54654675"),
            ("A000", "tt0113041", 0.1, "54654675"),
            ("A000", "tt0113101", -0.87, "54654675")
        ], columns=["from_id", "to_id", "score", "timestamp"])

        path = "../../../contents/movies_multiple_repr"
        items = []
        try:
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))
        except FileNotFoundError:
            path = "contents/movies_multiple_repr"
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))

        # Test for not existing field name
        t_centroid = CentroidVectorRecommender({'TestExceptionFieldName': '1'})
        result = t_centroid.predict('A000', ratings, 2, path, [])
        self.assertEqual(len(result), 0)

        # Test for not existing representation value
        t_centroid = CentroidVectorRecommender({'Plot': ['TestExceptionRepresentationName']})
        result = t_centroid.predict('A000', ratings, 2, path)
        self.assertEqual(len(result), 0)

        # Test for empty list representation
        t_centroid = CentroidVectorRecommender({'Plot': []})
        result = t_centroid.predict('A000', ratings, 2, path)
        self.assertEqual(len(result), 0)

        # Test for empty ratings
        empty_ratings = pd.DataFrame.from_records([], columns=["from_id", "to_id", "score", "timestamp"])
        t_centroid = CentroidVectorRecommender({'Plot': '1'})
        result = t_centroid.predict('A000', empty_ratings, 2, path)
        self.assertEqual(len(result), 0)

        # Test for negative ratings
        negative_ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -0.99, "54654675"),
            ("A000", "tt0112453", -0.22, "54654675"),
            ("A000", "tt0112641", -0.44, "54654675")
        ], columns=["from_id", "to_id", "score", "timestamp"])
        t_centroid = CentroidVectorRecommender({'Plot': '1'}, threshold=0)
        result = t_centroid.predict('A000', negative_ratings, 2, path)
        self.assertEqual(len(result), 0)

        # Test for embedding technique with granularity word
        path = "../../../contents/movies_plot_1_word_granularity"
        t_centroid = CentroidVectorRecommender({'Plot': '1'})
        result = t_centroid.predict('A000', ratings, 2, path)
        self.assertEqual(len(result), 0)


