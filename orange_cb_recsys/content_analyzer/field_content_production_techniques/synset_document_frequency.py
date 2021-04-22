import nltk

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques import SingleContentTechnique
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from typing import List, Union
from pywsd import disambiguate
from collections import Counter


class SynsetDocumentFrequency(SingleContentTechnique):
    """
    Pywsd word sense disambiguation
    """
    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)

    def produce_content(self, field_data) -> FeaturesBagField:
        """
        Produces a bag of features whose key is a wordnet synset
        and whose value is the frequency of the synset in the
        field data text
        """

        field_data = check_not_tokenized(self.preprocess_data(field_data))

        synsets = disambiguate(field_data)
        synsets = [synset for word, synset in synsets if synset is not None]

        return FeaturesBagField(Counter(synsets))

    def __str__(self):
        return "SynsetDocumentFrequency"

    def __repr__(self):
        return "SynsetDocumentFrequency Preprocessor List: " + str(self.preprocessor_list)
