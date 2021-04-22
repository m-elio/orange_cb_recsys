import os
from unittest import TestCase

from orange_cb_recsys.__main__ import script_run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

item_config_dict = {
    "module": "content_analyzer",
    "content_type": "ITEM",
    "output_directory": "movielens_test",
    "raw_source_path": "../../datasets/movies_info_reduced.json",
    "source_type": "json",
    "id_field_name": "imdbID",
    "get_lod_properties": {
        "class": "dbpedia_mapping",
        "mode": 'only_retrieved_evaluated',
        "entity_type": 'Film',
        "lang": 'EN',
        "label_field": 'Title'
    },
    "fields": {"field_name": "Plot",
               "pipeline_list": [{"class": "sk_learn_tf-idf"},
                                 {"class": "whoosh_tf-idf"}]}
}

user_config_dict = {
    "module": "content_analyzer",
    "content_type": "user",
    "output_directory": "user_test",
    "raw_source_path": "../../datasets/users_info.json",
    "source_type": "json",
    "id_field_name": "user_id",
    "fields": {"field_name": "name"}
}

embedding_learner_dict = {
    "module": "embedding_learner",
    "embedding_class": "word2vec",
    "source_type": "json",
    "raw_source_path": "../../datasets/movies_info_reduced.json",
    "preprocessor": {"class": "nltk"},
    "fields": "Plot",
    "additional_parameters": {"alpha": 0.020}
}

rating_config_dict = {
    "module": "rating",
    "source_type": "csv",
    "from_field_name": "user_id",
    "to_field_name": "item_id",
    "timestamp_field_name": "timestamp",
    "raw_source_path": "../../datasets/examples/new_ratings.csv",
    "output_directory": "../../datasets/test_ratings_file",
    "fields": {"field_name": "points", "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}}
}

recsys_config_dict = {
    "module": "recsys",
    "users_directory": "user_test",
    "items_directory": "movielens_test",
    "ranking_algorithm": {"class": "classifier", "item_fields": {"Plot": ["0"]}, "classifier": {"class": "knn"}},
    "rating_frame": "../../datasets/test_ratings_file",
    "rankings": [{"user_id": "10", "recs_number": 2},
                 {"user_id": "10", "recs_number": 4}]
}

eval_config_dict = {
    "module": "eval_model",
    "content_type": "ranking_alg_eval_model",
    "users_directory": "user_test",
    "items_directory": "movielens_test",
    "ranking_algorithm": {"class": "Centroid_Vector", "item_fields": {"Plot": ["0"]}},
    "rating_frame": "../../datasets/test_ratings_file",
    "partitioning": {"class": "k_fold", "n_splits": 4},
    "metric_list": {"class": "fnmeasure", "n": 2}
}

config_list = [item_config_dict, user_config_dict, embedding_learner_dict, rating_config_dict, recsys_config_dict,
               eval_config_dict]


class TestRun(TestCase):

    def test_run(self):
        # self.skipTest("test in the submodules.")
        global config_list
        try:
            # to do: improve test
            script_run(config_list)

        except FileNotFoundError:
            self.skipTest("LOCAL MACHINE")

    def test_content_config_exceptions(self):
        test_dict = {"module": "content_analyzer"}
        with self.assertRaises(KeyError):
            script_run(test_dict)

        # test for not valid source type
        test_dict = {
            "module": "content_analyzer",
            "content_type": "item",
            "output_directory": "items_test",
            "raw_source_path": "../../datasets/movies_info_reduced.json",
            "source_type": "test",
            "id_field_name": "imdbID"
        }
        with self.assertRaises(KeyError):
            script_run(test_dict)

    def test_embedding_learner_exceptions(self):
        test_dict = {"module": "embedding_learner"}
        with self.assertRaises(KeyError):
            script_run(test_dict)

    def test_rating_config_run_exceptions(self):
        test_dict = {"module": "rating"}
        with self.assertRaises(KeyError):
            script_run(test_dict)

        # test for not valid source type
        test_dict = {
            "module": "rating",
            "output_directory": "test_ratings",
            "raw_source_path": "../../datasets/examples/new_ratings.csv",
            "source_type": "csc",
            "from_field_name": "user_id",
            "to_field_name": "item_id",
            "timestamp_field_name": "timestamp"
        }
        with self.assertRaises(KeyError):
            script_run(test_dict)

    def test_eval_config_exceptions(self):
        test_dict = {"module": "eval_model"}
        with self.assertRaises(KeyError):
            script_run(test_dict)

        # test dictionary with not valid eval model class
        test_dict = {"module": "eval_model",
                     "partitioning": "k_fold",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": "0"}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings",
                     "content_type": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

    def test_run_exceptions(self):

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

    def test_recsys_config_exceptions(self):
        test_dict = {"module": "recsys"}
        with self.assertRaises(KeyError):
            script_run(test_dict)

        # test for not valid users and items directory paths
        test_dict = {"module": "recsys", "users_directory": "not_existing_path", "items_directory": "not_existing_path"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for rating frame path
        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": "0"}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_file"}
        self.assertIsNone(script_run(test_dict))

        # test for rating frame object
        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "centroid_vector", "item_fields": {"Plot": "0"}},
                     "rating_frame": {
                         "output_directory": "test_ratings",
                         "raw_source_path": "../../datasets/examples/new_ratings.csv",
                         "source_type": "csv",
                         "from_field_name": "user_id",
                         "to_field_name": "item_id",
                         "timestamp_field_name": "timestamp",
                         "fields": {
                             "field_name": "points",
                             "processor": {"class": "number_normalizer", "min_": 1, "max_": 5}
                         }
                     }}
        self.assertIsNone(script_run(test_dict))

    def test_object_extractor_exceptions(self):
        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"test": "Centroid_Vector", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_file",
                     "rankings": {"user_id": "10", "recs_number": 2}}

        with self.assertRaises(KeyError):
            script_run(test_dict)

        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "test", "item_fields": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_file",
                     "rankings": {"user_id": "10", "recs_number": 2}}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        test_dict = {"module": "recsys",
                     "users_directory": "../../contents/examples/ex_1/users_1600355755.1935306",
                     "items_directory": "../../contents/examples/ex_1/movies_1600355972.49884",
                     "ranking_algorithm": {"class": "Centroid_Vector", "test": {"Plot": ["0"]}},
                     "rating_frame": "../../datasets/test_ratings/test_ratings_file",
                     "rankings": {"user_id": "10", "recs_number": 2}}
        with self.assertRaises(TypeError):
            script_run(test_dict)

        test_dict = {
            "module": "content_analyzer",
            "content_type": "ITEM",
            "output_directory": "movielens_test",
            "raw_source_path": "../../datasets/movies_info_reduced.json",
            "source_type": "json",
            "id_field_name": "imdbID",
            "fields": {"field_name": "Plot",
                       "pipeline_list": [{"class": "sk_learn_tf-idf",
                                          "preprocessor_list": [{"class": "nltk", "test": "test"}]},
                                         {"class": "whoosh_tf-idf"}]}
        }
        with self.assertRaises(TypeError):
            script_run(test_dict)
