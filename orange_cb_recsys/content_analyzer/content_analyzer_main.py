import json
from typing import Dict
import time
import os
import numpy as np

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField, StringField, \
    FeaturesBagField, EmbeddingField
from orange_cb_recsys.content_analyzer.field_content_production_techniques. \
    field_content_production_technique import CollectionBasedTechnique, SingleContentTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.memory_interfaces import SearchIndex
from orange_cb_recsys.utils.const import home_path, DEVELOPING, logger
from orange_cb_recsys.utils.id_merger import id_merger


class ContentAnalyzer:
    """
    Class to whom the control of the content analysis phase is delegated

    Args:
        config (ContentAnalyzerConfig):
            configuration for processing the item fields. This parameter provides the possibility
            of customizing the way in which the input data is processed.
    """

    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def __preprocessing_operations(self, output_directory: str):
        """
        The Collection Based and the Search Index techniques both need a preliminary phase in order to be able to
        function (in particular collection based techniques need to refactor the dataset in order to have
        a modified representation of the field they will work on, search indexing techniques need to set up
        the same external index so that they operate on the same one)

        This preparatory operations are done here by cycling over each field that has to be considered and their
        respective pipelines (which are Field Content Production Techniques)

        Args:
            output_directory (str): name of the directory where the processed items will be stored, it is used so that,
                if the search index is need, it will be saved in that directory

        Returns:
            None if the index wasn't created during this process, the index itself otherwise
        """
        index = None
        index_need = False
        for field_name in self.__config.get_field_name_list():
            for i, pipeline in enumerate(self.__config.get_pipeline_list(field_name)):

                if isinstance(pipeline, CollectionBasedTechnique):
                    logger.info("Creating collection for technique: %s on field %s, "
                                "representation: %s", pipeline, field_name, str(i))
                    pipeline.dataset_refactor(self.__config.source, self.__config.id_field_name_list, field_name)

                if isinstance(pipeline, SearchIndexing):

                    if not index_need:
                        index_need = True
                        index = SearchIndex(os.path.join(output_directory, 'searching'))
                        index.init_writing()

                    pipeline.index = index

        return index

    def fit(self):
        """
        Processes the creation of the contents and serializes the contents
        """

        output_path = self.__config.output_directory
        if not DEVELOPING:
            output_path = os.path.join(home_path, 'contents', self.__config.output_directory)
        os.mkdir(output_path)

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.init_writing()

        index = self.__preprocessing_operations(output_path)
        contents_producer.indexer = index

        i = 0
        for raw_content in self.__config.source:
            logger.info("Processing item %d", i)
            content = contents_producer.create_content(raw_content)
            content.serialize(output_path)
            i += 1

        # once all the data has been processed, if the contents_producer used the search index, everything regarding
        # the index is cleared and set to normal
        if index is not None:
            index.stop_writing()
            contents_producer.delete_indexer()

        for interface in interfaces:
            interface.stop_writing()

        for field_name in self.__config.get_field_name_list():
            for pipeline in self.__config.get_pipeline_list(field_name):
                if isinstance(pipeline, CollectionBasedTechnique):
                    pipeline.delete_refactored()

    def __str__(self):
        return "ContentAnalyzer"

    def __repr__(self):
        msg = "< " + "ContentAnalyzer: " + "config = " + str(self.__config) + "; >"
        return msg


class ContentsProducer:
    """
    Singleton class which encapsulates the creation process of the items,
    The creation process is specified in the config parameter of ContentAnalyzer and
    is supposed to be the same for each item.
    """
    __instance = None

    @staticmethod
    def get_instance():
        """
        returns the singleton instance
        Returns:
            ContentsProducer: instance
        """
        # Static access method
        if ContentsProducer.__instance is None:
            ContentsProducer.__instance = ContentsProducer()
        return ContentsProducer.__instance

    def __init__(self):
        self.__config: ContentAnalyzerConfig = None
        self.__indexer = None
        # Virtually private constructor.
        if ContentsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        ContentsProducer.__instance = self

    @property
    def indexer(self):
        return self.__indexer

    @indexer.setter
    def indexer(self, indexer):
        self.__indexer = indexer

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def delete_indexer(self):
        del self.__indexer

    def __get_timestamp(self, raw_content: Dict) -> str:
        """
        Search for timestamp as dataset field. If there isn't a field called 'timestamp', than
        the timestamp will be the one returned by the system.
        """
        timestamp = None
        if self.__config.content_type != "item":
            if "timestamp" in raw_content.keys():
                timestamp = raw_content["timestamp"]
            else:
                timestamp = time.time()

        return timestamp

    def __create_field(self, raw_content: Dict, field_name: str, content_id: str, timestamp: str):
        """
        Create a new field for the specified content

        Args:
            raw_content (Dict): Raw content for the new field
            field_name (str): Name of the new field
            content_id (str): Id of the content to which add the field
            timestamp (str)

        Returns:
            field (ContentField)
        """
        print(content_id)
        if isinstance(raw_content[field_name], list):
            timestamp = raw_content[field_name][1]
            field_data = raw_content[field_name][0]
        else:
            field_data = raw_content[field_name]

        # serialize for explanation
        memory_interface = self.__config.get_memory_interface(field_name)
        if memory_interface is not None:
            memory_interface.new_field(field_name, field_data)

        # produce representations
        field = ContentField(timestamp)

        pipeline_list = self.__config.get_pipeline_list(field_name)

        for i, pipeline in enumerate(pipeline_list):
            logger.info("processing representation %d", i)
            if isinstance(pipeline, CollectionBasedTechnique):

                field.append(str(i), pipeline.produce_content(content_id))

            elif isinstance(pipeline, SingleContentTechnique):

                field.append(str(i), pipeline.produce_content(field_data))

            elif isinstance(pipeline, SearchIndexing):

                pipeline.field_name = field_name
                pipeline.pipeline_id = str(i)
                pipeline.produce_content(field_data)

            elif pipeline is None:

                field.append(str(i), self.__decode_field_data(field_data))

        return field

    def __decode_field_data(self, field_data: str):
        # Decode string into dict or list
        try:
            loaded = json.loads(field_data)
        except json.JSONDecodeError:
            try:
                # in case the dict is {'foo': 1} json expects {"foo": 1}
                reformatted_field_data = field_data.replace("\'", "\"")
                loaded = json.loads(reformatted_field_data)
            except json.JSONDecodeError:
                # if it has issues decoding we consider the data as str
                loaded = reformatted_field_data

        # if the decoded is a list, maybe it is an EmbeddingField repr
        if isinstance(loaded, list):
            arr = np.array(loaded)
            # if the array has only numbers then we consider it as a dense vector
            # else it is not and we consider the field data as a string
            if issubclass(arr.dtype.type, np.number):
                result = EmbeddingField(arr)
                return result
            else:
                result = StringField(field_data)
                return result

        # if the decoded is a dict, maybe it is a FeaturesBagField
        elif isinstance(loaded, dict):
            # if all values of the dict are numbers then we consider it as a bag of words
            # else it is not and we consider it as a string
            if len(loaded.values()) != 0 and \
                    all(isinstance(value, (float, int)) for value in loaded.values()):

                result = FeaturesBagField(loaded)
                return result
            else:
                result = StringField(field_data)
                return result

        # if the decoded is a string, then it is a StringField
        elif isinstance(loaded, str):
            result = StringField(loaded)
            return result

    def create_content(self, raw_content: Dict):
        """
        Creates a content processing every field in the specified way.
        This method is iteratively invoked by the fit method.

        Args:
            raw_content (dict): Raw data from which the content will be created

        Returns:
            content (Content): an instance of content with his fields

        Raises:
            general Exception
        """

        if self.__config is None:
            raise Exception("You must set a config with set_config()")

        timestamp = self.__get_timestamp(raw_content)

        # construct id from the list of the fields that compound id
        content_id = id_merger(raw_content, self.__config.id_field_name_list)
        content = Content(content_id)

        content_id_field_name = "content_id"

        for i, ex_retrieval in enumerate(self.__config.exogenous_properties_retrieval):
            lod_properties = ex_retrieval.get_properties(str(i), raw_content)
            content.append_exogenous_rep(str(i), lod_properties)

        if self.indexer is not None:
            self.__indexer.new_content()
            self.__indexer.new_field(content_id_field_name, content_id)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.new_content()
            interface.new_field(content_id_field_name, content_id)

        # produce
        for field_name in self.__config.get_field_name_list():
            logger.info("Processing field: %s", field_name)
            # search for timestamp override on specific field
            content.append(field_name, self.__create_field(raw_content, field_name, content_id, timestamp))

        if self.__indexer is not None:
            content.index_document_id = self.__indexer.serialize_content()

        for interface in interfaces:
            interface.serialize_content()

        return content

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        msg = "< " + "ContentsProducer:" + "" \
                                           "config = " + str(self.__config) + " >"
        return msg
