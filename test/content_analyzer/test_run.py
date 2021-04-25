import os
from unittest import TestCase

from orange_cb_recsys.__main__ import script_run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

item_config_dict = {
    "module": "content_analyzer",
    "content_type": "ITEM",
    "output_directory": "movielens_test",
    "source": {"class": "json", "file_path": "../../datasets/movies_info_reduced.json"},
    "id_field_name_list": "imdbID",
    "field_config_dict": {"Plot": {"pipelines_list": [{"class": "sk_learn_tf-idf"}, {"class": "whoosh_tf-idf"}]}}
}

user_config_dict = {
    "module": "content_analyzer",
    "content_type": "user",
    "output_directory": "user_test",
    "source": {"class": "json", "file_path": "../../datasets/users_info.json"},
    "id_field_name_list": "user_id",
    "field_config_dict": {"name": {}}
}

rating_config_dict = {
    "module": "ratings",
    "source": {"class": "csv", "file_path": "../../datasets/examples/new_ratings.csv"},
    "from_field_name": "user_id",
    "to_field_name": "item_id",
    "timestamp_field_name": "timestamp",
    "output_directory": "../../datasets/test_ratings_file",
    "rating_configs": {"field_name": "points", "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}}
}

recsys_config_dict = {
    "module": "recsys",
    "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
    "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
    "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv",
    "ranking_algorithm": {"class": "classifier", "item_fields": {"Plot": ["0"]}, "classifier": {"class": "knn"}},
    "rankings": {"user_id": "10", "recs_number": 2}
}

eval_config_dict = {
    "module": "eval",
    "eval_type": "ranking_alg_eval_model",
    "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
    "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
    "ranking_algorithm": {"class": "Centroid_Vector", "item_fields": {"Plot": ["0"]}},
    "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv",
    "partitioning": {"class": "k_fold", "n_splits": 4},
    "metric_list": {"class": "fnmeasure", "n": 2}
}

config_list = [item_config_dict, user_config_dict, rating_config_dict, recsys_config_dict, eval_config_dict]


class TestRun(TestCase):

    def test_run(self):
        self.assertEqual(len(script_run(config_list)), 3)

    def test_exceptions(self):
        # test for list not containing dictionaries only
        test_config_list_dict = [set(), dict()]
        with self.assertRaises(ValueError):
            script_run(test_config_list_dict)

        # test for dictionary in the list with no "module" parameter
        test_config_list_dict = {"parameter": "test"}
        with self.assertRaises(KeyError):
            script_run(test_config_list_dict)

        # test for dictionary in the list with "module" parameter but not valid value
        test_config_list_dict = [{"module": "test"}]
        with self.assertRaises(ValueError):
            script_run(test_config_list_dict)

        # test for not valid parameter name in dictionary representing object
        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1618867719.6795104",
                     "items_directory": "../../contents/examples/ex_1/movies_1618909007.3204026",
                     "ranking_algorithm": {"class": "centroid_vector", "test": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv",
                     "rankings": {"user_id": "10", "recs_number": 2}}
        with self.assertRaises(TypeError):
            script_run(test_dict)

        # test for not existing config_line in ratings
        test_dict = {"module": "ratings",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in content_analyzer
        test_dict = {"module": "content_analyzer",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in eval model
        test_dict = {"module": "eval",
                     "eval_type": "ranking_alg_eval_model",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in recsys
        test_dict = {"module": "recsys",
                     "test": "test",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test eval model with no defined eval type
        test_dict = {"module": "eval",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_1618757236.csv"}
        with self.assertRaises(KeyError):
            script_run(test_dict)
