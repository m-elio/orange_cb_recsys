import os
from unittest import TestCase
import gensim

from orange_cb_recsys.content_analyzer.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestGensimLatentSemanticAnalysis(TestCase):
    def test_fit(self):
        file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")
        preprocessor = NLTK(stopwords_removal=True)
        fields = ["Plot"]
        src = JSONFile(file_path)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
        learner.fit()
        self.assertIsInstance(learner.model, gensim.models.lsimodel.LsiModel)
