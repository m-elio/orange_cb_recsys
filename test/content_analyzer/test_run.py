import os
from unittest import TestCase

from orange_cb_recsys.__main__ import script_run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(THIS_DIR, "../../datasets/movies_info_reduced.json")
user_info = os.path.join(THIS_DIR, "../../datasets/users_info.json")
new_ratings = os.path.join(THIS_DIR, "../../datasets/examples/new_ratings.csv")
test_ratings_file = os.path.join(THIS_DIR, "../../datasets/test_ratings_file")
users_example_1 = os.path.join(THIS_DIR, "../../contents/examples/ex_1/users_1600355755.1935306")
items_example_1 = os.path.join(THIS_DIR, "../../contents/examples/ex_1/movies_1600355972.49884")
ratings_example = os.path.join(THIS_DIR, "../../datasets/test_ratings/test_ratings_1618757236.csv")


class TestRun(TestCase):

    def setUp(self) -> None:

        item_config_dict = {
            "module": "content_analyzer",
            "content_type": "ITEM",
            "output_directory": "movielens_test",
            "source": {"class": "json", "file_path": movies_info_reduced},
            "id_field_name_list": "imdbID",
            "field_config_dict": {"Plot": {"pipelines_list": [{"class": "sk_learn_tf-idf"}, {"class": "whoosh_tf-idf"}]}}
        }

        user_config_dict = {
            "module": "content_analyzer",
            "content_type": "user",
            "output_directory": "user_test",
            "source": {"class": "json", "file_path": user_info},
            "id_field_name_list": "user_id",
            "field_config_dict": {"name": {}}
        }

        rating_config_dict = {
            "module": "ratings",
            "source": {"class": "csv", "file_path": new_ratings},
            "from_field_name": "user_id",
            "to_field_name": "item_id",
            "timestamp_field_name": "timestamp",
            "output_directory": test_ratings_file,
            "rating_configs": {"field_name": "points", "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}}
        }

        recsys_config_dict = {
            "module": "recsys",
            "users_directory": users_example_1,
            "items_directory": items_example_1,
            "rating_frame": ratings_example,
            "ranking_algorithm": {"class": "classifier", "item_fields": {"Plot": ["0"]}, "classifier": {"class": "knn"}},
            "rankings": {"user_id": "10", "recs_number": 2}
        }

        eval_config_dict = {
            "module": "eval",
            "eval_type": "ranking_alg_eval_model",
            "users_directory": users_example_1,
            "items_directory": items_example_1,
            "ranking_algorithm": {"class": "Centroid_Vector", "item_fields": {"Plot": ["0"]}},
            "rating_frame": ratings_example,
            "partitioning": {"class": "k_fold", "n_splits": 4},
            "metric_list": {"class": "fnmeasure", "n": 2}
        }

        self.config_list = [item_config_dict, user_config_dict, rating_config_dict, recsys_config_dict, eval_config_dict]

    def test_run(self):
        self.assertEqual(len(script_run(self.config_list)), 3)

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
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {"class": "centroid_vector", "test": {"Plot": ["0"]}},
                     "rating_frame": ratings_example,
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
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": ratings_example,
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in recsys
        test_dict = {"module": "recsys",
                     "test": "test",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": ratings_example}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test eval model with no defined eval type
        test_dict = {"module": "eval",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": ratings_example}
        with self.assertRaises(KeyError):
            script_run(test_dict)
