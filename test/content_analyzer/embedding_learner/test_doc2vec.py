import os
from unittest import TestCase

import gensim

from orange_cb_recsys.content_analyzer.embedding_learner.doc2vec import GensimDoc2Vec
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestGensimDoc2Vec(TestCase):
    def test_fit(self):
        path = os.path.join(THIS_DIR, "../../../datasets/d2v_test_data.json")
        doc2vec = GensimDoc2Vec(source=JSONFile(file_path=path), preprocessor=NLTK(), field_list=["doc_field"])
        doc2vec.fit()
        self.assertIsInstance(doc2vec.model, gensim.models.doc2vec.Doc2Vec)
