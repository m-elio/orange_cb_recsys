import os
from unittest import TestCase

from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import WhooshTfIdf, SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestWhooshTfIdf(TestCase):
    def test_produce_content(self):
        try:
            technique = WhooshTfIdf([NLTK()])
            technique.dataset_refactor(JSONFile(file_path), ["imdbID"], "Plot")
            features_bag_test = technique.produce_content("tt0113497")
            features = features_bag_test.value

            self.assertEqual(features['years'], 0.6989700043360189)
        except AttributeError:
            self.fail("Couldn't load feature bag!")


class TestSkLearnTfIDF(TestCase):
    def test_produce_content(self):
        technique = SkLearnTfIdf([NLTK()])
        technique.dataset_refactor(JSONFile(file_path), ["imdbID"], "Plot")
        features_bag_test = technique.produce_content("tt0113497")
        features = features_bag_test.value

        self.assertLess(features['the'], 0.15)
