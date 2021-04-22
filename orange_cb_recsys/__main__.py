from typing import Dict, Union
import orange_cb_recsys.utils.runnable_instances as r_i

import os
import json
import sys
import yaml
import inspect
import pandas as pd

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, FieldConfig
from orange_cb_recsys.content_analyzer.content_analyzer_main import ContentAnalyzer
from orange_cb_recsys.content_analyzer.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import \
    RatingsImporter, RatingsFieldConfig
from orange_cb_recsys.recsys import RecSysConfig, RecSys
from orange_cb_recsys.utils.const import logger

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_analyzer/config.json")

"""
All the available implementations are extracted
"""

runnable_instances = r_i.get()


def __dict_detector(technique: Union[dict, list]):
    """
    Detects a class constructor inside of a dictionary and replaces the parameters with the actual object.
    A class constructor is identified by a 'class' parameter inside of a dictionary.

    This method is also useful in case there are class constructors inside another class constructor

    {"class": 'test_class'} will be transformed into TestClass()

    where TestClass is the class associated to the alias 'test_class' in the runnable_instances file

    If a list of objects is specified such as:

    [{"class": 'class1'}, {"class": 'class2'}]

    this method will call itself recursively so that each dictionary will be transformed into the corresponding object

    If the technique doesn't match any of these cases, for example:

    {"parameter": "value", "parameter2": "value2"}

    It's nor a list nor a dictionary representing a class constructor, nothing will be done and it will be returned as
    it is

    Args:
        technique: dictionary or list to check in order to transform any class constructors into actual objects
    """
    if isinstance(technique, list):
        techniques = []
        for element in technique:
            techniques.append(__dict_detector(element))
        return techniques
    elif isinstance(technique, dict) and 'class' in technique.keys():
        parameter_class_name = technique.pop('class')
        try:
            return runnable_instances[parameter_class_name](**technique)
        except TypeError:
            passed_parameters = list(technique.keys())
            actual_parameters = list(inspect.signature(
                runnable_instances[parameter_class_name].__init__).parameters.keys())
            actual_parameters.remove("self")
            raise TypeError("The following parameters: " + str(passed_parameters) + "\n" +
                            "Don't match the class' constructor parameters: " + str(actual_parameters))
    else:
        return technique


def __object_extractor(object_dict: Dict, parameter: str):
    """
    In case an object is specified in the config file, this method extracts the class name and the class parameters
    and instantiates the object itself. The class name has to be specified in the config file by a 'class' parameter,
    the object parameters are defined with the same names as the actual parameters.

    If the 'class' parameter is not specified or the class name is not a valid implementation a KeyError exception is
    thrown. A TypeError exception is thrown if the parameters specified for the object don't match the actual parameters

    EXAMPLE:

        in the configuration file there is:

            "preprocessing_list": {"class": "nltk", "lemmatization": True}

        In order to instantiate the object for the parameter preprocessing_list the dictionary is extracted, in this
        instance in particular, this function's parameters will be

        object_dict: {"class": "nltk", "lemmatization": True}
        parameter: "preprocessing_list"

        The "class" parameter is extracted from the object_dict, and the class, corresponding to the 'nltk'
        alias, is retrieved (using the runnable_instances). The remaining items in the dictionary are used as
        the object's constructor parameters and the object is instantiated and returned. In case objects are defined
        inside of the object_dict the dict_detector method replaces them with the actual object instance

        {"class": "nltk", "lemmatization": True} will be transformed into NLTK(lemmatization=True) and returned

    Args:
        object_dict (dict): dictionary containing the class name and the object parameters
        parameter (str): parameter name in the config file for which the object is being created

    Returns:
        extracted_object: instance of the actual object created from the object_dict
    """
    try:
        class_name = object_dict.pop('class').lower()
    except KeyError:
        raise KeyError("You must specify an object using the class parameter for '%s'" % parameter)
    try:
        for key in object_dict.keys():
            object_dict[key] = __dict_detector(object_dict[key])
    except TypeError as type_error:
        raise type_error
    try:
        extracted_object = runnable_instances[class_name](**object_dict)
    except (KeyError, ValueError):
        raise ValueError("%s is not an existing implementaton" % class_name)
    except TypeError:
        passed_parameters = list(object_dict.keys())
        actual_parameters = list(inspect.signature(runnable_instances[class_name].__init__).parameters.keys())
        actual_parameters.remove("self")
        raise TypeError("The following parameters: " + str(passed_parameters) + "\n" +
                        "Don't match the class' constructor parameters: " + str(actual_parameters))
    return extracted_object


def __complete_path(directory: str) -> str:
    """
    Used to find a complete directory from the one passed as an argument with the last file's complete name. If there
    are multiple files with the same name, the last created file will be extracted. This is particularly useful in this
    project since the files created by the Content Analyzer are named after the time at which they were created
    (using time.time). Without this it would be impossible to run a Recsys and a Content Analyzer in the same config
    file, because the user wouldn't be able to define the directory name where users, items or ratings are stored.

    If no file is found, a ValueError exception is thrown.

    EXAMPLE:

        directory passed as argument: '../file_to_find'
        actual existing directory: '../file_to_find1.something'

        the output of the function will be: '../file_to_find1.something'

        if two directories exist such as '../file_to_find1.something' and '../file_to_find2.something',
        the last created file between 'file_to_find1' and 'file_to_find2' will be extracted

    Args:
        directory (str): directory where the file is stored. The last part of the directory is the partial name , or
            the complete one even, of the file to find

    Returns (str):
        file_directory: complete directory

    """
    base_dir = os.path.dirname(os.path.abspath(directory))
    file_name = os.path.basename(os.path.normpath(directory))
    files = [os.path.join(base_dir, name) for name in os.listdir(base_dir)]
    files.sort(key=os.path.getctime)
    files.reverse()
    for file_directory in files:
        name = os.path.basename(os.path.normpath(file_directory))
        if file_name in str(name):
            return file_directory
    raise FileNotFoundError("File not found for path: %s" % directory)


def __content_config_run(content_config: Dict):
    """
    Method that extracts the parameters for the creation and fitting of a Content Analyzer. It checks the dictionary
    for the various keys representing the parameters of the Content Analyzer and extracts their values

    Args:
        content_config (dict): dictionary that represents a config defined in the config file, in this case it
            represents the config of the content analyzer
    """
    try:
        # if one or more lines in the content config dictionary aren't recognised as actual parameters a warning is
        # given to the user
        for config_line in content_config.keys():
            if config_line not in ['content_type', 'source_type', 'raw_source_path', 'id_field_name',
                                   'output_directory', 'get_lod_properties', 'fields']:

                logger.warning("%s is not a parameter for content analyzer, therefore it will be skipped" % config_line)

        # if any of the parameters that must be defined are not defined, a KeyError exception is thrown showing
        # the missing parameters
        unspecified_parameters = []
        if 'content_type' not in content_config.keys():
            unspecified_parameters.append('content_type')
        if 'source_type' not in content_config.keys():
            unspecified_parameters.append('source_type')
        if 'raw_source_path' not in content_config.keys():
            unspecified_parameters.append('raw_source_path')
        if 'id_field_name' not in content_config.keys():
            unspecified_parameters.append('id_field_name')
        if 'output_directory' not in content_config.keys():
            unspecified_parameters.append('output_directory')

        if len(unspecified_parameters) != 0:
            raise KeyError("The following obligatory parameters for Content Analyzer were not specified: " +
                           str(unspecified_parameters))

        # could be 'item' or 'user' and so on
        source_type = content_config['source_type'].lower()

        content_analyzer_config = ContentAnalyzerConfig(
            content_config["content_type"],
            runnable_instances[source_type]
            (file_path=content_config["raw_source_path"]),
            content_config['id_field_name'],
            content_config['output_directory'])

        if 'get_lod_properties' in content_config.keys():
            if not isinstance(content_config['get_lod_properties'], list):
                content_config['get_lod_properties'] = [content_config['get_lod_properties']]
            for ex_retrieval in content_config['get_lod_properties']:
                content_analyzer_config.append_exogenous_properties_retrieval(__object_extractor(
                    ex_retrieval, 'get_lod_properties'))

        if 'fields' in content_config.keys():
            if not isinstance(content_config['fields'], list):
                content_config['fields'] = [content_config['fields']]

            for field_dict in content_config['fields']:
                if 'lang' in field_dict.keys():
                    field_config = FieldConfig(field_dict['lang'])
                else:
                    field_config = FieldConfig()

                # setting the content analyzer config

                if 'pipeline_list' in field_dict.keys():
                    if not isinstance(field_dict['pipeline_list'], list):
                        field_dict['pipeline_list'] = [field_dict['pipeline_list']]

                    for pipeline_dict in field_dict['pipeline_list']:
                        # content production settings
                        if isinstance(pipeline_dict, dict):
                            field_config.append_pipeline(__object_extractor(pipeline_dict, 'pipeline_list'))
                        else:
                            field_config.append_pipeline(None)
                # verify that the memory interface is set
                if 'memory_interface' in field_dict.keys():
                    field_config.memory_interface = __object_extractor(
                        field_dict['memory_interface'], 'memory_interface')

                content_analyzer_config.append_field_config(field_dict["field_name"], field_config)

        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()
    except (KeyError, ValueError, TypeError) as e:
        raise e


def __embedding_learner_run(config_dict: Dict):
    """
    Method that extracts the parameters for the creation and fitting of an Embedding Learner. It checks the dictionary
    for the various keys representing the parameters of the Embedding Learner and extracts their values

    Args:
        config_dict(dict): dictionary that represents a config defined in the config file, in this case it represents
            the config of the embedding learner
    """
    try:
        unspecified_parameters = []
        if 'embedding_class' not in config_dict.keys():
            unspecified_parameters.append('embedding_class')
        if 'source_type' not in config_dict.keys():
            unspecified_parameters.append('source_type')
        if 'raw_source_path' not in config_dict.keys():
            unspecified_parameters.append('raw_source_path')
        if 'preprocessor' not in config_dict.keys():
            unspecified_parameters.append('preprocessor')
        if 'fields' not in config_dict.keys():
            unspecified_parameters.append('fields')

        if len(unspecified_parameters) != 0:
            raise KeyError("The following obligatory parameters for Embedding Learner were not specified: " +
                           str(unspecified_parameters))

        source_type = config_dict['source_type'].lower()
        preprocessor = __object_extractor(config_dict['preprocessor'], 'preprocessor')

        if not isinstance(config_dict['fields'], list):
            config_dict['fields'] = [config_dict['fields']]

        optional_parameters = {}
        if 'additional_parameters' in config_dict.keys():
            if isinstance(config_dict['additional_parameters'], dict):
                optional_parameters = config_dict['additional_parameters']
            else:
                raise ValueError("The field 'additional_parameters' must contain a dictionary")

        embedding_learner = runnable_instances[config_dict['embedding_class']](
            source=runnable_instances[source_type](file_path=config_dict["raw_source_path"]),
            preprocessor=preprocessor,
            field_list=config_dict['fields'],
            kwargs=optional_parameters
        )

        if isinstance(embedding_learner, EmbeddingLearner):
            embedding_learner.fit()
            embedding_learner.save()

    except (KeyError, ValueError, TypeError) as e:
        raise e


def __rating_config_run(config_dict: Dict) -> pd.DataFrame():
    """
    Method that extracts the parameters for the creation of a Ratings Importer. It checks the dictionary
    for the various keys representing the parameters of the Ratings Importer and extracts their values

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it
            represents the config of the ratings importer
    """
    try:
        for config_line in config_dict.keys():
            if config_line not in ['source_type', 'fields', 'raw_source_path', 'output_directory',
                                   'from_field_name', 'to_field_name', 'timestamp_field_name']:
                logger.warning("%s is not a parameter for rating config, therefore it will be skipped" % config_line)

        unspecified_parameters = []
        if 'source_type' not in config_dict.keys():
            unspecified_parameters.append('source_type')
        if 'raw_source_path' not in config_dict.keys():
            unspecified_parameters.append('raw_source_path')
        if 'output_directory' not in config_dict.keys():
            unspecified_parameters.append('output_directory')
        if 'from_field_name' not in config_dict.keys():
            unspecified_parameters.append('from_field_name')
        if 'to_field_name' not in config_dict.keys():
            unspecified_parameters.append('to_field_name')
        if 'timestamp_field_name' not in config_dict.keys():
            unspecified_parameters.append('timestamp_field_name')

        if len(unspecified_parameters) != 0:
            raise KeyError("The following obligatory parameters for Ratings Importer were not specified: " +
                           str(unspecified_parameters))

        rating_configs = []
        source_type = config_dict['source_type'].lower()

        if 'fields' in config_dict.keys():
            if not isinstance(config_dict['fields'], list):
                config_dict['fields'] = [config_dict['fields']]
            for field in config_dict["fields"]:
                processor = __object_extractor(field['processor'], 'processor')
                rating_configs.append(
                    RatingsFieldConfig(field_name=field["field_name"],
                                       processor=processor)
                )

        return RatingsImporter(
            source=runnable_instances[source_type](file_path=config_dict["raw_source_path"]),
            output_directory=config_dict["output_directory"],
            rating_configs=rating_configs,
            from_field_name=config_dict["from_field_name"],
            to_field_name=config_dict["to_field_name"],
            timestamp_field_name=config_dict["timestamp_field_name"]
        ).import_ratings()
    except (KeyError, ValueError, TypeError) as e:
        raise e


def __recsys_config_run(config_dict: Dict) -> RecSysConfig:
    """
    Method that extracts the parameters for the creation of a Recsys Config. It checks the dictionary
    for the various keys representing the parameters of the Recsys config and extracts their values

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it represents
            the config of the recommender system
    """
    try:
        unspecified_parameters = []
        if 'users_directory' not in config_dict.keys():
            unspecified_parameters.append('users_directory')
        if 'items_directory' not in config_dict.keys():
            unspecified_parameters.append('items_directory')

        if len(unspecified_parameters) != 0:
            raise KeyError("The following obligatory parameters for Recsys Config were not specified: " +
                           str(unspecified_parameters))

        users_directory = __complete_path(config_dict['users_directory'])
        items_directory = __complete_path(config_dict['items_directory'])

        if 'score_prediction_algorithm' in config_dict.keys():
            score_prediction_algorithm = __object_extractor(
                config_dict['score_prediction_algorithm'], 'score_prediction_algorithm')
        else:
            score_prediction_algorithm = None

        if 'ranking_algorithm' in config_dict.keys():
            ranking_algorithm = __object_extractor(config_dict['ranking_algorithm'], 'ranking_algorithm')
        else:
            ranking_algorithm = None

        if 'rating_frame' in config_dict.keys():
            if isinstance(config_dict['rating_frame'], str):
                rating_frame = __complete_path(config_dict['rating_frame'])
            else:
                rating_frame = __rating_config_run(config_dict['rating_frame'])
        else:
            rating_frame = None

        recsys_config = RecSysConfig(
            users_directory=users_directory,
            items_directory=items_directory,
            score_prediction_algorithm=score_prediction_algorithm,
            ranking_algorithm=ranking_algorithm,
            rating_frame=rating_frame
        )

        return recsys_config
    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __recsys_run(config_dict: Dict):
    """
    Method that extracts the parameters for the creation of a Recsys. It checks the dictionary
    for the various keys representing the parameters of the Recsys and extracts their values. Also it allows
    to define what kind of ranking or predictions the user wants from the recommender system.

    N.B.: for now it prints the recsys run results, this will be improved

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it represents
            the config of the predictions or rankings the user wants to run
    """
    try:
        for config_line in config_dict.keys():
            if config_line not in ['users_directory', 'items_directory', 'score_prediction_algorithm',
                                   'ranking_algorithm', 'rating_frame', 'predictions', 'rankings']:
                logger.warning("%s is not a parameter for recsys run, therefore it will be skipped" % config_line)

        recsys = RecSys(config=__recsys_config_run(config_dict))

        if 'predictions' in config_dict.keys():
            if not isinstance(config_dict['predictions'], list):
                config_dict['predictions'] = [config_dict['predictions']]
            for prediction_parameters in config_dict['predictions']:
                user_id = prediction_parameters['user_id']
                if 'item_to_predict_list' in prediction_parameters.keys():
                    item_to_predict_id_list = prediction_parameters['item_to_predict_list']
                else:
                    item_to_predict_id_list = None
                print(recsys.fit_predict(user_id, item_to_predict_id_list))

        if 'rankings' in config_dict.keys():
            if not isinstance(config_dict['rankings'], list):
                config_dict['rankings'] = [config_dict['rankings']]
            for ranking_parameters in config_dict['rankings']:
                user_id = ranking_parameters['user_id']
                recs_number = ranking_parameters['recs_number']
                if 'candidate_item_id_list' in ranking_parameters.keys():
                    candidate_item_id_list = ranking_parameters['candidate_item_id_list']
                else:
                    candidate_item_id_list = None
                print(recsys.fit_ranking(user_id, recs_number, candidate_item_id_list))

    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __eval_config_run(config_dict: Dict):
    """
    Method that extracts the parameters for the creation of a Eval Model. It checks the dictionary
    for the various keys representing the parameters of the Eval Model and extracts their values

    N.B.: for now it prints the eval fitting results, this will be improved

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file
    """
    try:
        for config_line in config_dict.keys():
            if config_line not in ['content_type', 'users_directory', 'items_directory', 'score_prediction_algorithm',
                                   'ranking_algorithm', 'rating_frame', 'metric_list', 'partitioning', 'recs_number']:
                logger.warning("%s is not a parameter for eval model, therefore it will be skipped"
                               % config_line)

        unspecified_parameters = []
        if 'partitioning' not in config_dict.keys():
            unspecified_parameters.append('partitioning')

        if len(unspecified_parameters) != 0:
            raise KeyError("The following obligatory parameters for Eval Model were not specified: " +
                           str(unspecified_parameters))

        recsys_config = __recsys_config_run(config_dict)

        eval_class = config_dict["content_type"].lower()

        metric_list = []
        if "metric_list" in config_dict.keys():
            if not isinstance(config_dict['metric_list'], list):
                config_dict['metric_list'] = [config_dict['metric_list']]
            for metric in config_dict["metric_list"]:
                metric = __object_extractor(metric, 'metric')
                metric_list.append(metric)

        if eval_class.lower() == "prediction_alg_eval_model" or eval_class.lower() == "ranking_alg_eval_model":
            partitioning = __object_extractor(config_dict['partitioning'], 'partitioning')

            eval_model = runnable_instances[eval_class](recsys_config, partitioning, metric_list)
            print(eval_model.fit())

        else:
            recs_number = config_dict['recs_number']

            eval_model = runnable_instances[eval_class](recsys_config, recs_number, metric_list)
            print(eval_model.fit())
    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def script_run(config_list_dict: Union[dict, list]):
    """
    Method that controls the entire process of script running. It checks that the contents loaded from a script match
    the required prerequisites. The data must contain dictionaries and each dictionary must have a "module" key
    that is used to understand what kind of operation it is supposed to perform (if it's meant for a Recommender System,
    a Content Analyzer, ...). If any of these prerequisites aren't matched a ValueError or KeyError exception is thrown

    Args:
        config_list_dict: single dictionary or list of dictionaries extracted from the config file
    """
    try:

        if not isinstance(config_list_dict, list):
            config_list_dict = [config_list_dict]

        if not all(isinstance(config_dict, dict) for config_dict in config_list_dict):
            raise ValueError("The list in the script must contain dictionaries only")

        for config_dict in config_list_dict:

            if "module" in config_dict.keys():
                if config_dict["module"].lower() == "rating":
                    del config_dict["module"]
                    __rating_config_run(config_dict)
                elif config_dict["module"].lower() == "content_analyzer":
                    del config_dict["module"]
                    __content_config_run(config_dict)
                elif config_dict["module"].lower() == "recsys":
                    del config_dict["module"]
                    __recsys_run(config_dict)
                elif config_dict["module"].lower() == "eval_model":
                    del config_dict["module"]
                    __eval_config_run(config_dict)
                elif config_dict["module"].lower() == "embedding_learner":
                    del config_dict["module"]
                    __embedding_learner_run(config_dict)
                else:
                    raise ValueError("You must specify a valid module: "
                                     "[rating, content_analyzer, recsys, eval_model]")

            else:
                raise KeyError("A 'module' parameter must be specified and the value must be one of the following: "
                               "[rating, content_analyzer, recsys, eval_model]")

    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.endswith('.yml'):
        extracted_data = yaml.load(open(config_path), Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        extracted_data = json.load(open(config_path))
    else:
        raise ValueError("Wrong file extension")

    script_run(extracted_data)
