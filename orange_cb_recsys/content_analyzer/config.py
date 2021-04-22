import time
from typing import List, Dict, Set, Union

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FieldContentProductionTechnique
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import ExogenousPropertiesRetrieval
from orange_cb_recsys.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class FieldConfig:
    """
    Class that represents the configuration of a single field. A field can have a memory interface to store the original
    data regarding a field. A field can also have a list of Field Content Production Techniques which define what kind
    of operations will be done on the data of said field. If there are multiple Field Content Production Techniques,
    multiple representations for the single field will be created. A field also has a lang attribute which is used to
    set the languages of the Information Processors inside of the Field Content Production Techniques.

    If content technique is not specified or content_technique=None, the framework will
    try to decode if the field_data is in the form of a bag of word ({'hello': 5.2, 'world':2})
    or in the form of an embedding vector ([0.53653,0.784141,1.23565, ...])
    If so, it will instantiates the corresponding an object of the corresponding class.
    If not, it will instantiate a StringField representation for the field.

    Args:
        lang (str): code to specify the text language (example: "EN")
        memory_interface (InformationInterface): optional Information Interface to save the original data of the field
        pipelines_list (Union[FieldContentProductionTechnique, List[FieldContentProductionTechnique]]):
            optional value or list of the production techniques that will be used to produce different field's
            representations, each FieldContentProductionTechnique represents a pipeline
    """

    def __init__(self, lang: str = "EN",
                 memory_interface: InformationInterface = None,
                 pipelines_list: Union[FieldContentProductionTechnique,
                                       List[Union[FieldContentProductionTechnique, None]]] = None):
        if pipelines_list is None:
            pipelines_list = []

        self.__lang = lang
        self.__memory_interface: InformationInterface = memory_interface
        self.__pipelines_list: List[FieldContentProductionTechnique] = pipelines_list

        if not isinstance(self.__pipelines_list, list):
            self.__pipelines_list = [self.__pipelines_list]

        for pipeline in self.__pipelines_list:
            if pipeline is not None:
                pipeline.lang = self.__lang

    @property
    def lang(self):
        return self.__lang

    @property
    def memory_interface(self) -> InformationInterface:
        return self.__memory_interface

    @memory_interface.setter
    def memory_interface(self, memory_interface: InformationInterface):
        self.__memory_interface = memory_interface

    def append_pipeline(self, pipeline: FieldContentProductionTechnique = None):
        if pipeline is not None:
            pipeline.lang = self.__lang
        self.__pipelines_list.append(pipeline)

    def extend_pipeline_list(self, pipeline_list: List[FieldContentProductionTechnique]):
        for pipeline in pipeline_list:
            if pipeline is not None:
                pipeline.lang = self.__lang
        self.__pipelines_list.extend(pipeline_list)

    @property
    def pipeline_list(self) -> List[FieldContentProductionTechnique]:
        for pipeline in self.__pipelines_list:
            yield pipeline

    def __str__(self):
        return "FieldConfig"

    def __repr__(self):
        return "< " + "FieldConfig: " + "" \
                "pipelines_list = " + str(self.__pipelines_list) + " >"


class ContentAnalyzerConfig:
    """
    Class that represents the configuration for the content analyzer.

    Args:
        content_type (str): defines what kind of entity is being subject to the content analyzer (for example it might
            be 'user' or 'item')
        source (RawInformationSource): raw data source to iterate on for extracting the contents
        id_field_name_list (Union[str, List[str]]): value or list of the fields names containing the content's id,
            it's a list instead of single value for handling complex id composed of multiple fields
        field_config_dict (Dict<str, FieldConfig>): store the config for each field_name
        output_directory (str): path of the results serialized content instance
        field_config_dict (Dict[str, FieldConfig]): dictionary representing each field that the user wants to be
            considered and its configuration, the keys are the name of the fields and the values are Field Config
        exogenous_properties_retrieval(Union[ExogenousPropertiesRetrieval, List[ExogenousPropertiesRetrieval]]):
            optional value or list of techniques that retrieve exogenous properties that represent the contents
    """

    def __init__(self, content_type: str,
                 source: RawInformationSource,
                 id_field_name_list: Union[str, List[str]],
                 output_directory: str,
                 field_config_dict: Dict[str, FieldConfig] = None,
                 exogenous_properties_retrieval:
                 Union[ExogenousPropertiesRetrieval, List[ExogenousPropertiesRetrieval]] = None):
        if field_config_dict is None:
            field_config_dict = {}
        if exogenous_properties_retrieval is None:
            exogenous_properties_retrieval = []

        self.__output_directory: str = output_directory + str(time.time())
        self.__content_type = content_type.lower()
        self.__field_config_dict: Dict[str, FieldConfig] = field_config_dict
        self.__source: RawInformationSource = source
        self.__id_field_name_list: List[str] = id_field_name_list
        self.__exogenous_properties_retrieval: List[ExogenousPropertiesRetrieval] = exogenous_properties_retrieval

        if not isinstance(self.__exogenous_properties_retrieval, list):
            self.__exogenous_properties_retrieval = [self.__exogenous_properties_retrieval]

        if not isinstance(self.__id_field_name_list, list):
            self.__id_field_name_list = [self.__id_field_name_list]

    def append_exogenous_properties_retrieval(self, exogenous_properties_retrieval: ExogenousPropertiesRetrieval):
        self.__exogenous_properties_retrieval.append(exogenous_properties_retrieval)

    @property
    def exogenous_properties_retrieval(self) -> ExogenousPropertiesRetrieval:
        for ex_retrieval in self.__exogenous_properties_retrieval:
            yield ex_retrieval

    @property
    def output_directory(self):
        return self.__output_directory

    @property
    def content_type(self):
        return self.__content_type

    @property
    def id_field_name_list(self):
        return self.__id_field_name_list

    @property
    def source(self) -> RawInformationSource:
        return self.__source

    def get_memory_interface(self, field_name: str) -> InformationInterface:
        return self.__field_config_dict[field_name].memory_interface

    def get_field_config(self, field_name: str):
        return self.__field_config_dict[field_name]

    def get_pipeline_list(self, field_name: str) -> List[FieldContentProductionTechnique]:
        """
        Get the list of the pipelines specified for the input field

        Args:
            field_name (str): name of the field

        Returns:
            List[FieldContentProductionTechnique]: the list of pipelines specified for the input field
        """
        for pipeline in self.__field_config_dict[field_name].pipeline_list:
            yield pipeline

    def get_field_name_list(self) -> List[str]:
        """
        Get the list of the field names

        Returns:
            List<str>: list of config dictionary keys
        """
        return list(self.__field_config_dict.keys())

    def get_interfaces(self) -> Set[InformationInterface]:
        """
        get the list of field interfaces

        Returns:
            List<InformationInterface>: list of config dict values
        """
        interfaces = set()
        for key in self.__field_config_dict.keys():
            if self.__field_config_dict[key].memory_interface is not None:
                interfaces.add(self.__field_config_dict[key].memory_interface)
        return interfaces

    def append_field_config(self, field_name: str, field_config: FieldConfig):
        self.__field_config_dict[field_name] = field_config

    def __str__(self):
        return str(self.__id_field_name_list)

    def __repr__(self):
        msg = "< " + "ContentAnalyzerConfig: " + "id_field_name = " + str(self.__id_field_name_list) + "; " + \
              "source = " + str(self.__source) + "; field_config_dict = " + str(self.__field_config_dict) + "; " +\
              "content_type = " + str(self.__content_type) + ">"
        return msg
