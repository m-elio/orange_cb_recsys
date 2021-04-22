from babelpy.babelfy import BabelfyClient

from orange_cb_recsys.content_analyzer.content_representation.\
    content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SingleContentTechnique, FieldContentProductionTechnique
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from typing import List, Union
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


class BabelPyEntityLinking(SingleContentTechnique):
    """
    Interface for the Babelpy library that wraps some feature of Babelfy entity Linking.

    Args:
        api_key: string obtained by registering to
            babelfy website, with None babelpy key only few
            queries can be executed
    """

    def __init__(self, api_key: str = None,
                 preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)
        self.__api_key = api_key
        self.__babel_client = None

    @FieldContentProductionTechnique.lang.setter
    def lang(self, lang: str):
        FieldContentProductionTechnique.lang.fset(self, lang)
        params = dict()
        params['lang'] = self.lang
        self.__babel_client = BabelfyClient(self.__api_key, params)

    def __str__(self):
        return "BabelPyEntityLinking"

    def __repr__(self):
        return "BabelPyEntityLinking Preprocessor List: " + str(self.preprocessor_list)

    def produce_content(self, field_data) -> FeaturesBagField:
        """
        Produces the field content for this representation,
        bag of features whose keys is babel net synset id and
        values are global score of the sysnset

        Args:
            field_data: Text that will be linked to BabelNet

        Returns:
            feature_bag (FeaturesBagField)
        """
        field_data = check_not_tokenized(self.preprocess_data(field_data))

        self.__babel_client.babelfy(field_data)
        feature_bag = dict()
        try:
            if self.__babel_client.entities is not None:
                try:
                    for entity in self.__babel_client.entities:
                        feature_bag[entity['babelSynsetID']] = entity['globalScore']
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return FeaturesBagField(feature_bag)
