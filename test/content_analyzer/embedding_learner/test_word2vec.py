import os
from unittest import TestCase

import gensim

from orange_cb_recsys.content_analyzer.embedding_learner.word2vec import GensimWord2Vec
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestGensimWord2Vec(TestCase):
    def test_fit(self):
        file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")
        field_list = ['Title', 'Year', 'Genre']
        word2vec = GensimWord2Vec(source=JSONFile(file_path),
                                  preprocessor=NLTK(),
                                  field_list=field_list)
        word2vec.fit()
        self.assertIsInstance(word2vec.model, gensim.models.word2vec.Word2Vec)

    def test_save(self):
        preprocessor = NLTK(stopwords_removal=True)
        fields = ["Plot"]
        file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")
        src = JSONFile(file_path)
        learner = GensimWord2Vec(src, preprocessor, fields)
        learner.fit()
        learner.save()
        self.assertIsInstance(learner.model, gensim.models.word2vec.Word2Vec)
        """
        path = os.path.join(THIS_DIR, "*.model")
        x = sorted(glob.glob(path))[-1]
        dynamic_path = pl.Path(x)
        self.assertEqual((str(dynamic_path), dynamic_path.is_file()), (str(dynamic_path), True))
        """