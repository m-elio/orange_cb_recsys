import os
from unittest import TestCase

import gensim

from orange_cb_recsys.content_analyzer.embedding_learner.random_indexing import GensimRandomIndexing
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestRandomIndexing(TestCase):
    def test_fit(self):
        file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")
        random_indexing = GensimRandomIndexing(JSONFile(file_path), NLTK(), ['Genre', 'Plot'])
        random_indexing.fit()
        self.assertIsInstance(random_indexing.model, gensim.models.rpmodel.RpModel)
