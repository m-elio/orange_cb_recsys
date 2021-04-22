import os
from unittest import TestCase
import lzma
import pickle
import numpy as np

from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig, FieldConfig
from orange_cb_recsys.content_analyzer.content_representation.content_field import StringField, FeaturesBagField, \
    EmbeddingField
from orange_cb_recsys.content_analyzer.field_content_production_techniques import EmbeddingTechnique, \
    Centroid, GensimDownloader, SearchIndexing
from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../datasets/movies_info_reduced.json")


class TestContentsProducer(TestCase):
    def test_create_content(self):
        file_path_content_analyzer = os.path.join(THIS_DIR, "../../test/content_analyzer/movielens_test*")
        plot_config = FieldConfig(pipelines_list=BabelPyEntityLinking())
        content_analyzer_config = ContentAnalyzerConfig('ITEM', JSONFile(file_path), ["imdbID"], "movielens_test")
        content_analyzer_config.append_field_config("Plot", plot_config)
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()
        """
        glob_path = (glob.glob(file_path_content_analyzer)[0])
        content_list = []
        for filename in os.listdir(glob_path):
            if filename.endswith(".xz"):
                with lzma.open(glob_path + "/" + filename, "rb") as content_file:
                    content = pickle.load(content_file)
                    content_list.append(content)
        self.assertIn("tt0112281", content_list[0].content_id)
        representation = content_list[0].field_dict["Plot"].get_representation("0")
        self.assertIsInstance(representation, FeaturesBagField)
        """

    def test_create_content_tfidf(self):
        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path),
            id_field_name_list=['imdbID'],
            output_directory="movielens_test",
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[SkLearnTfIdf()]
            ))

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_create_content_search_index(self):
        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path),
            id_field_name_list=['imdbID'],
            output_directory='movielens_test'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[SearchIndexing()]
            )
        )

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_create_content_embedding(self):
        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path),
            id_field_name_list=['imdbID'],
            output_directory="movielens_test",
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[EmbeddingTechnique(
                                                combining_technique=Centroid(),
                                                embedding_source=GensimDownloader(name='glove-twitter-25'),
                                                granularity='doc',
                                                preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)])
                                ]
            ))

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_decode_field_data_string(self):
        file_path_test_decode = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_string.json")
        test_dir = os.path.join(THIS_DIR, "../../datasets/test_decode/")

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path_test_decode),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_string_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[None]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_string_' in str(name):

                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), StringField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, str)
                    break

    def test_decode_field_data_tfidf(self):
        file_path_test_decode = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_tfidf.json")
        test_dir = os.path.join(THIS_DIR, "../../datasets/test_decode/")

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path_test_decode),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_tfidf_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[None]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_tfidf_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, dict)
                    break

    def test_decode_field_data_embedding(self):
        file_path_test_decode = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_embedding.json")
        test_dir = os.path.join(THIS_DIR, "../../datasets/test_decode/")

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(file_path_test_decode),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_embedding_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[None]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_embedding_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), EmbeddingField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, np.ndarray)
                    break
