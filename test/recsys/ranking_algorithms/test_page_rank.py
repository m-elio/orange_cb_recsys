import pandas as pd
from unittest import TestCase
import os

from orange_cb_recsys.recsys import NXPageRank
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.utils.feature_selection import FSPageRank

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
contents_path = os.path.join(THIS_DIR, '../../../contents')
ratings_filename = os.path.join(contents_path, 'exo_prop/new_ratings_small.csv')
movies_dir = os.path.join(contents_path, 'exo_prop/movielens_exo_1612956350.7812138/')
user_dir = os.path.join(contents_path, 'exo_prop/user_exo_1612956381.4652517/')

ratings = pd.DataFrame.from_records([
    ("1", "tt0112281", 0.8),
    ("1", "tt0112302", 0.7),
    ("2", "tt0112281", -0.4),
    ("2", "tt0112346", 1.0),
    ("2", "tt0112453", 0.4),
    ("3", "tt0112453", 0.1),
    ("4", "tt0112346", -0.3),
    ("4", "tt0112453", 0.7),
    ("5", "tt0114388", 0.2),
    ("5", "tt0114885", 0.1),
    ("5", "tt0113189", -0.3),
    ("5", "tt0112281", 0.6),
    ("6", "tt0113277", 0.9),
    ("6", "tt0114319", 0.3),
    ("6", "tt0114709", 0.5),
    ("7", "tt0114709", -0.2),
    ("7", "tt0113228", 0.4),
    ("8", "tt0113228", -1.0)
], columns=["from_id", "to_id", "score"])

graph: NXFullGraph = NXFullGraph(ratings,
                                 user_contents_dir=user_dir,
                                 item_contents_dir=movies_dir,
                                 item_exo_representation="0",
                                 user_exo_representation=None,
                                 item_exo_properties=None,
                                 user_exo_properties=['1', '4']
                                 )


class TestNXPageRank(TestCase):
    def test_predict(self):
        alg = NXPageRank(graph)

        user_ratings = ratings[ratings['from_id'] == '2']

        # test for number of recommendations <= 0
        rank = alg.predict(user_ratings, -10)
        self.assertEqual(len(rank), 0)

        # test for standard prediction considering ratings from a user (PageRank with priors)
        rank = alg.predict(user_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with empty dataframe (standard PageRank)
        empty_ratings = pd.DataFrame()
        rank = alg.predict(empty_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with a candidate_item_id_list
        rank = alg.predict(user_ratings, 3, candidate_item_id_list=['tt0113277', 'tt0114709'])
        self.assertEqual(len(rank), 2)

        # test for prediction with feature selection algorithms both for items and users and empty ratings
        alg = NXPageRank(graph=graph, item_feature_selection_algorithm=FSPageRank(1),
                         user_feature_selection_algorithm=FSPageRank(2))
        rank = alg.predict(empty_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with feature selection algorithms both for items and users and user ratings
        alg = NXPageRank(graph=graph, item_feature_selection_algorithm=FSPageRank(1),
                         user_feature_selection_algorithm=FSPageRank(2))
        rank = alg.predict(user_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with feature selection algorithm for items only
        alg = NXPageRank(graph, item_feature_selection_algorithm=FSPageRank(1))
        rank = alg.predict(user_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with feature selection algorithm for users only
        alg = NXPageRank(graph, user_feature_selection_algorithm=FSPageRank(1))
        rank = alg.predict(user_ratings, 3)
        self.assertEqual(len(rank), 3)

        # test for prediction with feature selection algorithm for items and users with k set to 0
        alg = NXPageRank(graph, item_feature_selection_algorithm=FSPageRank(0),
                         user_feature_selection_algorithm=FSPageRank(0))
        rank = alg.predict(user_ratings, 3)
        self.assertEqual(len(rank), 3)


class PageRankAlg(TestCase):
    def test_clean_rank(self):
        user_ratings = ratings[ratings['from_id'] == '1']

        rank = {"1": 0.5, "tt0112281": 0.5, "2": 0.5, "tt0113497": 0.5, "tt0112302": 0.5}
        alg = NXPageRank(graph=graph)

        # doesn't remove any node
        alg.remove_items_in_profile = False
        alg.remove_properties = False
        alg.remove_user_nodes = False
        result = alg.clean_rank(rank, graph, user_ratings, "1")
        self.assertGreaterEqual(len(result.keys()), 0)

        # removes user and property nodes and item nodes already in the user profile
        alg.remove_items_in_profile = True
        alg.remove_properties = True
        alg.remove_user_nodes = True
        result = alg.clean_rank(rank, graph, user_ratings, "1")
        expected = {"tt0113497": 0.5}
        self.assertEqual(expected, result)

    def test_extract_profile(self):
        alg = NXPageRank(graph=graph)
        user_ratings = pd.DataFrame().from_records([
            ("3", "tt0112453", 0.1)], columns=["from_id", "to_id", "score"])
        result = alg.extract_profile("3", graph, user_ratings)

        expected = {"tt0112453": 0.55}

        self.assertEqual(expected, result)
