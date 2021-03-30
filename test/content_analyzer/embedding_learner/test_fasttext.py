import os
from unittest import TestCase

import gensim

from orange_cb_recsys.content_analyzer.embedding_learner.fasttext import GensimFastText
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestGensimFastText(TestCase):
    def test_fit(self):
        field_list = ['Title', 'Year', 'Genre']
        file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")
        fast_text = GensimFastText(source=JSONFile(file_path),
                                   preprocessor=NLTK(),
                                   field_list=field_list)
        fast_text.fit()
        self.assertIsInstance(fast_text.model, gensim.models.fasttext.FastText)


