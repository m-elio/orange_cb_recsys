import lzma
import os
from abc import ABC, abstractmethod
from typing import Dict
import pickle
import re

from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField
from orange_cb_recsys.utils.const import logger


class ExogenousPropertiesRepresentation(ABC):
    """
    Output of LodPropertiesRetrieval, different representations
    exist according to different techniques

    Args:
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError


class PropertiesDict(ExogenousPropertiesRepresentation):
    """
    Couples <property name, property value>
    retrieved by DBPediaMappingTechnique

    Args:
        features: properties in the specified format
    """

    def __init__(self, features: Dict[str, str] = None):
        super().__init__()
        if features is None:
            features = {}

        self.__features: Dict[str, str] = features

    @property
    def value(self):
        """
        Returns: features dictionary
        """
        return self.__features


class Content:
    """
    Class that represents a content. A content can be an item or a user.
    A content is identified by a string id and is composed by different fields

    Args:
        content_id (str): identifier
        field_dict (dict[str, ContentField]): dictionary
            containing the fields instances for the content,
            and their name as dictionary key
        exogenous_rep_dict (Dict <str, ExogenousProperties>):
            different representations of content obtained
            using ExogenousPropertiesRetrieval, the dictionary key is
            the representation name
    """
    def __init__(self, content_id: str,
                 field_dict: Dict[str, ContentField] = None,
                 exogenous_rep_dict: Dict[str, ExogenousPropertiesRepresentation] = None):
        if field_dict is None:
            field_dict = {}       # list o dict
        if exogenous_rep_dict is None:
            exogenous_rep_dict = {}

        self.__content_id: str = content_id
        self.__index_document_id: int = None
        self.__field_dict: Dict[str, ContentField] = field_dict
        self.__exogenous_rep_dict: Dict[str, ExogenousPropertiesRepresentation] = exogenous_rep_dict

    @property
    def content_id(self):
        return self.__content_id

    @property
    def field_dict(self):
        return self.__field_dict

    @property
    def exogenous_rep_dict(self):
        return self.__exogenous_rep_dict

    @property
    def index_document_id(self) -> int:
        return self.__index_document_id

    @index_document_id.setter
    def index_document_id(self, index_document_id: int):
        self.__index_document_id = index_document_id

    def get_field(self, field_name: str):
        return self.__field_dict[field_name]

    def append_exogenous_rep(self, name: str, exogenous_properties: ExogenousPropertiesRepresentation):
        self.__exogenous_rep_dict[name] = exogenous_properties

    def get_exogenous_rep(self, name):
        return self.__exogenous_rep_dict[name]

    def append(self, field_name: str, field: ContentField):
        self.__field_dict[field_name] = field

    def remove(self, field_name: str):
        """
        Remove the field named field_name from the field dictionary

        Args:
            field_name (str): the name of the field to remove
        """
        self.__field_dict.pop(field_name)

    def serialize(self, output_directory: str):
        """
        Serialize a content instance using lzma compression algorithm,
        so the file extension is .xz

        Args:
            output_directory (str): Name of the directory in which serialize
        """
        logger.info("Serializing content %s in %s", self.__content_id, output_directory)

        file_name = re.sub(r'[^\w\s]', '', self.__content_id)
        path = os.path.join(output_directory, file_name + '.xz')
        with lzma.open(path, 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        content_string = "Content: %s" % self.__content_id
        field_string = ''
        for field, rep in self.__field_dict.items():
            field_string += "\nField: %s %s" % (field, rep)

        return "%s \n\n %s \n##############################" % (content_string, field_string)

    def __eq__(self, other):
        return self.__content_id == other.__content_id and self.__field_dict == other.__field_dict
