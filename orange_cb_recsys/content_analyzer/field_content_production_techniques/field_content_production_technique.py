from abc import ABC, abstractmethod
from typing import List, Union

import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from orange_cb_recsys.content_analyzer.content_representation.content_field import FieldRepresentation, EmbeddingField,\
    StringField
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized, check_not_tokenized


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that generalizes the techniques to use for producing the semantic description
    of a content's field's representation
    """
    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        if preprocessor_list is None:
            preprocessor_list = []
        self.__preprocessor_list = preprocessor_list

        if not isinstance(self.__preprocessor_list, list):
            self.__preprocessor_list = [self.__preprocessor_list]

        self.__lang = "EN"

    @property
    def lang(self):
        return self.__lang

    @property
    def preprocessor_list(self):
        return self.__preprocessor_list

    @lang.setter
    def lang(self, lang: str):
        self.__lang = lang
        for preprocessor in self.__preprocessor_list:
            preprocessor.lang = self.__lang

    @preprocessor_list.setter
    def preprocessor_list(self, preprocessor_list):
        self.__preprocessor_list = preprocessor_list

    def preprocess_data(self, field_data):
        processed_field_data = field_data
        for preprocessor in self.__preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        return processed_field_data

    @abstractmethod
    def produce_content(self, field_data):
        raise NotImplementedError


class SearchIndexing(FieldContentProductionTechnique):
    """
    Technique used to saved the processed data into a Search Index that can be used for other operations (an example
    would be submitting a query to said index)
    """

    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)
        self.__index = None             # index to write on
        self.__field_name = None        # filed name to write on
        self.__pipeline_id = None       # pipeline id for passing field_name + pipeline_id in the index

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, index: IndexInterface):
        self.__index = index

    @property
    def field_name(self):
        return self.__field_name

    @field_name.setter
    def field_name(self, field_name):
        self.__field_name = field_name

    @property
    def pipeline_id(self):
        return self.__pipeline_id

    @pipeline_id.setter
    def pipeline_id(self, pipeline_id):
        self.__pipeline_id = pipeline_id

    def produce_content(self, field_data):
        """
        Save field data as a document field using the given indexer,
        the resulting can be used for an index query recommender

        Args:
            field_data: Data that will be stored in the index

        """
        field_data = check_not_tokenized(self.preprocess_data(field_data))
        self.__index.new_field(self.__field_name + self.__pipeline_id, field_data)

    def __str__(self):
        return "Indexing for search-engine recommender"

    def __repr__(self):
        return "Indexing for search-engine recommender Preprocessor list:" + str(self.preprocessor_list)


class CollectionBasedTechnique(FieldContentProductionTechnique):
    """
    This class generalizes the techniques that work on the entire content collection,
    such as the tf-idf technique
    """

    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)

    @abstractmethod
    def produce_content(self, content_id: str) -> FieldRepresentation:
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: List[str], field_name: str):
        """
        This method restructures the raw data in a way functional to the final representation.
        This is done only for those field representations that require this phase to be done

        Args:
            information_source (RawInformationSource):
            id_field_names: fields where to find data that compound content's id
            field_name
        """
        raise NotImplementedError

    @abstractmethod
    def delete_refactored(self):
        raise NotImplementedError


class SingleContentTechnique(FieldContentProductionTechnique):

    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)

    @abstractmethod
    def produce_content(self, field_data) -> FieldRepresentation:
        """
        Given data of certain field it returns a complex representation's instance of the field.

        Args:
            field_data: input for the complex representation production

        Returns:
            FieldRepresentation: an instance of FieldRepresentation,
                 the particular type of representation depends from the technique
        """
        raise NotImplementedError


class GenericTechnique(FieldContentProductionTechnique):
    """
    Simple technique which stores the data without applying any complex operation. If a preprocessor list is
    defined it will process the data before returning it in a String Field
    """

    def __init__(self, preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)

    def produce_content(self, field_data) -> StringField:
        return StringField(self.preprocess_data(field_data))

    def __str__(self):
        return "Basic technique for text"

    def __repr__(self):
        return "Basic technique for text Preprocessor list:" + str(self.preprocessor_list)


class CombiningTechnique(ABC):
    """
    Class that generalizes the modality in which loaded embeddings will be
    combined to produce a semantic representation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def combine(self, embedding_matrix: np.ndarray):
        """
        Combine, in a way specified in the implementations,
        the row of the input matrix

        Args:
            embedding_matrix: matrix whose rows will be combined

        Returns:

        """
        raise NotImplementedError


class EmbeddingSource(ABC):
    """
    General class whose purpose is to store the loaded pre-trained embeddings model and
    extract specified words from it

    Args:
        self.__model: embeddings model loaded from source
    """

    def __init__(self):
        self.__model = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model
        the vectors of the words contained in text

        Args:
            text (list<str>): contains words of which vectors will be extracted

        Returns:
            embedding_matrix (np.ndarray): numpy vector, where
                each row is a term vector. Assuming text is a list of N words,
                embedding_matrix will be N-dimensional.
        """
        text = check_tokenized(text)
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        for i, word in enumerate(text):
            word = word.lower()
            try:
                embedding_matrix[i, :] = self.__model[word]
            except KeyError:
                embedding_matrix[i, :] = np.zeros(self.get_vector_size())

        return embedding_matrix

    def get_vector_size(self) -> int:
        return self.__model.vector_size

    def __str__(self):
        return "EmbeddingSource"

    def __repr__(self):
        return "EmbeddingSource " + str(self.__model)


class EmbeddingTechnique(SingleContentTechnique):
    """
    Class that can be used to combine different embeddings coming from various sources
    in order to produce the semantic description.

    Args:
        combining_technique (CombiningTechnique): The technique that will be used
        for combining the embeddings.
        embedding_source (EmbeddingSource):
        Source where the embeddings vectors for the words in field_data are stored.
        granularity (Granularity): It can assume three values, depending on whether
        the framework user wants to combine relatively to words, phrases or documents.
    """

    def __init__(self, combining_technique: CombiningTechnique, embedding_source: EmbeddingSource, granularity: str,
                 preprocessor_list: Union[InformationProcessor, List[InformationProcessor]] = None):
        super().__init__(preprocessor_list)
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source

        self.__granularity: str = granularity.lower()

    def produce_content(self, field_data) -> EmbeddingField:
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.

        Args:
            field_data: The terms whose embeddings will be combined.

        Returns:
            EmbeddingField
        """
        preprocessed_field_data = self.preprocess_data(field_data)

        if self.__granularity == "word":
            doc_matrix = self.__embedding_source.load(preprocessed_field_data)
            return EmbeddingField(doc_matrix)
        if self.__granularity == "sentence":
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = sent_tokenize(preprocessed_field_data)
            for i, sentence in enumerate(sentences):
                sentences[i] = sentence[:len(sentence) - 1]

            sentences_embeddings = \
                np.ndarray(shape=(len(sentences), self.__embedding_source.get_vector_size()))
            for i, sentence in enumerate(sentences):
                sentence_matrix = self.__embedding_source.load(sentence)
                sentences_embeddings[i, :] = self.__combining_technique.combine(sentence_matrix)

            return EmbeddingField(sentences_embeddings)
        if self.__granularity == "doc":
            doc_matrix = self.__embedding_source.load(preprocessed_field_data)
            return EmbeddingField(self.__combining_technique.combine(doc_matrix))
        else:
            raise ValueError("Must specify a valid embedding technique granularity")

    def __str__(self):
        return "EmbeddingTechnique"

    def __repr__(self):
        return "EmbeddingTechnique " \
               + str(self.__combining_technique) + " " + \
               str(self.__embedding_source) + " " \
               + str(self.__granularity) + "Preprocessor List: " + str(self.preprocessor_list)
